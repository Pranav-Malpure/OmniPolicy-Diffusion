"""
Diffusion Transformer (DiT) denoiser for Latent Diffusion Policy.

Architecture (per paper §3.3 / §3.4):
  - Noisy latent z_s (B, latent_dim) is split into n_tokens patches and
    projected to embed_dim, giving a token sequence (B, n_tokens, embed_dim).
  - Timestep s and robot embedding e are combined into a context vector c.
  - N=6 DiT blocks apply multi-head self-attention + FFN, each conditioned
    on c via Adaptive Layer Norm (AdaLN-Zero).
  - Output head projects back to (B, latent_dim) — the predicted noise ε.

AdaLN-Zero (from Peebles & Xiao 2022):
  The modulation MLP is zero-initialised so the initial model is the
  identity transform, which gives stable loss at the start of training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Timestep embedding  (sinusoidal → MLP)
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Maps a scalar diffusion timestep s to a fixed-dim embedding."""

    def __init__(self, embed_dim: int, max_steps: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        freqs = torch.exp(-math.log(max_steps) * torch.arange(half) / (half - 1))
        self.register_buffer("freqs", freqs)   # (half,)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer or float timestep in [0, max_steps)
        t = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0)   # (B, half)
        emb = torch.cat([t.sin(), t.cos()], dim=-1)              # (B, embed_dim)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# AdaLN-Zero modulation
# ---------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    """
    Predicts (scale, shift, gate) from context c for one DiT sub-layer.
    Zero-initialised linear so initial output is (0, 0, 0) → identity residual.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # c: (B, embed_dim)
        out = self.linear(F.silu(c))          # (B, 3*embed_dim)
        scale, shift, gate = out.chunk(3, dim=-1)
        return scale, shift, gate


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN: x * (1 + scale) + shift.  Broadcasts over token dim."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Single DiT block
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    One Transformer block with AdaLN-Zero conditioning.

    Sub-layers:
      1. AdaLN → MSA → gate + residual
      2. AdaLN → FFN → gate + residual
    """

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

        self.adaLN_attn = AdaLNZero(embed_dim)
        self.adaLN_ffn  = AdaLNZero(embed_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, n_tokens, embed_dim)   c: (B, embed_dim)

        # --- attention sub-layer ---
        scale1, shift1, gate1 = self.adaLN_attn(c)
        x_mod = modulate(self.norm1(x), scale1, shift1)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod)
        x = x + gate1.unsqueeze(1) * attn_out

        # --- FFN sub-layer ---
        scale2, shift2, gate2 = self.adaLN_ffn(c)
        x_mod = modulate(self.norm2(x), scale2, shift2)
        x = x + gate2.unsqueeze(1) * self.ffn(x_mod)

        return x


# ---------------------------------------------------------------------------
# Full Diffusion Transformer
# ---------------------------------------------------------------------------

class DiffusionTransformer(nn.Module):
    """
    DiT denoiser:  (z_s, s, robot_id)  →  predicted noise ε

    Args:
        latent_dim:   dimension of the action VAE latent (64)
        embed_dim:    internal Transformer width (256)
        n_heads:      attention heads (8)
        n_layers:     number of DiT blocks (6, per paper)
        n_tokens:     how many tokens to split the latent into (8 → 8×8=64)
        num_robots:   number of distinct robot embodiments (2: panda, xarm6)
        mlp_ratio:    FFN expansion factor (4.0)
        max_steps:    total DDPM timesteps (1000)
    """

    def __init__(
        self,
        latent_dim:  int = 64,
        embed_dim:   int = 256,
        n_heads:     int = 8,
        n_layers:    int = 6,
        n_tokens:    int = 8,
        num_robots:  int = 3,
        mlp_ratio:   float = 4.0,
        max_steps:   int = 1000,
    ):
        super().__init__()
        assert latent_dim % n_tokens == 0, "latent_dim must be divisible by n_tokens"
        self.latent_dim  = latent_dim
        self.embed_dim   = embed_dim
        self.n_tokens    = n_tokens
        self.token_dim   = latent_dim // n_tokens   # e.g. 8

        # --- input projection: each latent patch → embed_dim ---
        self.patch_embed = nn.Linear(self.token_dim, embed_dim)

        # --- learnable positional encoding for the n_tokens patches ---
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # --- conditioning signals ---
        self.time_emb  = TimestepEmbedding(embed_dim, max_steps)
        self.robot_emb = nn.Embedding(num_robots, embed_dim)

        # Projects (time_emb + robot_emb) to embed_dim context vector
        self.cond_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # --- DiT blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, n_heads, mlp_ratio) for _ in range(n_layers)
        ])

        # --- output head: project tokens back to latent patches ---
        self.norm_out  = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out  = nn.Linear(embed_dim, self.token_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    # ------------------------------------------------------------------

    def forward(
        self,
        z_s:      torch.Tensor,            # (B, latent_dim)  noisy latent
        t:        torch.Tensor,            # (B,)             diffusion timestep
        robot_id: torch.Tensor,            # (B,)             robot index
        vis_emb:  Optional[torch.Tensor] = None,  # (B, embed_dim)  future visual context
    ) -> torch.Tensor:
        """Returns predicted noise ε of shape (B, latent_dim)."""
        B = z_s.size(0)

        # --- tokenise latent ---
        x = z_s.view(B, self.n_tokens, self.token_dim)  # (B, n_tokens, token_dim)
        x = self.patch_embed(x) + self.pos_emb           # (B, n_tokens, embed_dim)

        # --- build context c = f(timestep + robot_id [+ vis]) ---
        c = self.time_emb(t) + self.robot_emb(robot_id)  # (B, embed_dim)
        if vis_emb is not None:
            c = c + vis_emb
        c = self.cond_proj(c)                             # (B, embed_dim)

        # --- DiT blocks ---
        for block in self.blocks:
            x = block(x, c)

        # --- output: reconstruct latent-dim noise ---
        x = self.norm_out(x)                             # (B, n_tokens, embed_dim)
        x = self.proj_out(x)                             # (B, n_tokens, token_dim)
        eps = x.reshape(B, self.latent_dim)              # (B, latent_dim)

        return eps

    @torch.no_grad()
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# DDPM noise schedule utilities  (used by training + inference)
# ---------------------------------------------------------------------------

class DDPMSchedule:
    """
    Linear beta schedule and pre-computed alpha / alpha_bar tensors.
    Supports both DDPM (full 1000-step) and DDIM (fast subset) sampling.
    """

    def __init__(
        self,
        n_steps:    int   = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 0.02,
        device:     str   = "cpu",
    ):
        self.n_steps = n_steps
        betas              = torch.linspace(beta_start, beta_end, n_steps, device=device)
        alphas             = 1.0 - betas
        alpha_bar          = torch.cumprod(alphas, dim=0)
        alpha_bar_prev     = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.betas         = betas
        self.alphas        = alphas
        self.alpha_bar     = alpha_bar          # ᾱ_t
        self.alpha_bar_prev= alpha_bar_prev     # ᾱ_{t-1}
        self.sqrt_alpha_bar        = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

    def q_sample(
        self,
        z0:  torch.Tensor,   # (B, latent_dim)  clean latent
        t:   torch.Tensor,   # (B,)              integer timestep
        eps: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: z_t = √ᾱ_t · z0 + √(1-ᾱ_t) · ε"""
        if eps is None:
            eps = torch.randn_like(z0)
        a  = self.sqrt_alpha_bar[t].view(-1, 1)
        sa = self.sqrt_one_minus_alpha_bar[t].view(-1, 1)
        return a * z0 + sa * eps, eps

    @torch.no_grad()
    def ddpm_step(
        self,
        model:    DiffusionTransformer,
        z_t:      torch.Tensor,   # (B, latent_dim)
        t:        int,
        robot_id: torch.Tensor,   # (B,)
        vis_emb:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One reverse DDPM step: z_{t-1} from z_t."""
        t_tensor = torch.full((z_t.size(0),), t, device=z_t.device, dtype=torch.long)
        eps_pred = model(z_t, t_tensor, robot_id, vis_emb)

        alpha     = self.alphas[t]
        alpha_bar = self.alpha_bar[t]
        beta      = self.betas[t]

        coef      = beta / (1.0 - alpha_bar).sqrt()
        mean      = (z_t - coef * eps_pred) / alpha.sqrt()

        if t > 0:
            noise  = torch.randn_like(z_t)
            # posterior variance = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
            var    = beta * (1.0 - self.alpha_bar_prev[t]) / (1.0 - alpha_bar)
            z_prev = mean + var.sqrt() * noise
        else:
            z_prev = mean

        return z_prev

    @torch.no_grad()
    def ddim_sample(
        self,
        model:    DiffusionTransformer,
        z_T:      torch.Tensor,    # (B, latent_dim)  starting noise
        robot_id: torch.Tensor,    # (B,)
        ddim_steps: int = 20,
        vis_emb:  Optional[torch.Tensor] = None,
        eta:      float = 0.0,     # 0 = deterministic DDIM
    ) -> torch.Tensor:
        """
        Fast DDIM reverse process in ddim_steps instead of n_steps.
        Returns denoised z_0.
        """
        step_indices = torch.linspace(0, self.n_steps - 1, ddim_steps, dtype=torch.long)
        step_indices = step_indices.flip(0)   # go T → 0

        z = z_T
        for i, t_cur in enumerate(step_indices):
            t_cur  = t_cur.item()
            t_prev = step_indices[i + 1].item() if i + 1 < len(step_indices) else 0

            t_tensor  = torch.full((z.size(0),), t_cur,  device=z.device, dtype=torch.long)
            eps_pred  = model(z, t_tensor, robot_id, vis_emb)

            ab_cur  = self.alpha_bar[t_cur]
            ab_prev = self.alpha_bar[t_prev]

            # DDIM update
            z0_pred = (z - (1.0 - ab_cur).sqrt() * eps_pred) / ab_cur.sqrt()
            dir_xt  = (1.0 - ab_prev - eta**2 * (1.0 - ab_cur)).sqrt() * eps_pred
            noise   = eta * (1.0 - ab_cur).sqrt() * torch.randn_like(z) if eta > 0 else 0.0
            z = ab_prev.sqrt() * z0_pred + dir_xt + noise   # type: ignore[operator]

        return z


# ---------------------------------------------------------------------------
# Observation encoders  (plug into vis_emb slot of DiffusionTransformer)
# ---------------------------------------------------------------------------

class StateEncoder(nn.Module):
    """
    Encodes a zero-padded proprioceptive state vector to embed_dim.
    Used for obs_mode="state".

    Input : (B, state_dim)   e.g. (B, 64) after zero-padding
    Output: (B, embed_dim)
    """

    def __init__(self, state_dim: int = 64, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class VisualEncoder(nn.Module):
    """
    Lightweight CNN that encodes a single RGB frame to embed_dim.
    Used for obs_mode="rgb".

    Input : (B, 3, H, W)   e.g. (B, 3, 128, 128), values in [0, 1]
    Output: (B, embed_dim)

    Architecture: 4 × (Conv→BN→ReLU stride-2) → global avg pool → Linear
    Spatial resolution: 128 → 64 → 32 → 16 → 8 → pool → (512,) → embed_dim
    """

    def __init__(self, embed_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32,  kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,          64,  kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,          128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128,         256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256,         512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # (B, 512, 1, 1)
        )
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: (B, 3, H, W)
        feat = self.cnn(rgb).flatten(1)   # (B, 512)
        return self.proj(feat)            # (B, embed_dim)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, latent_dim = 4, 64
    device = "cpu"

    model    = DiffusionTransformer(latent_dim=64, embed_dim=256, n_heads=8, n_layers=6)
    schedule = DDPMSchedule(n_steps=1000, device=device)

    print(f"DiffusionTransformer — {model.count_params():,} parameters")

    z0       = torch.randn(B, latent_dim)
    t        = torch.randint(0, 1000, (B,))
    robot_id = torch.zeros(B, dtype=torch.long)

    z_t, eps = schedule.q_sample(z0, t)
    eps_pred = model(z_t, t, robot_id)

    print(f"z0      : {z0.shape}")
    print(f"z_t     : {z_t.shape}")
    print(f"eps_pred: {eps_pred.shape}")
    assert eps_pred.shape == (B, latent_dim)

    # DDIM sampling from pure noise
    z_T  = torch.randn(B, latent_dim)
    rid  = torch.zeros(B, dtype=torch.long)
    z_gen = schedule.ddim_sample(model, z_T, rid, ddim_steps=20)
    print(f"DDIM z_gen: {z_gen.shape}")
    print("All checks passed.")
