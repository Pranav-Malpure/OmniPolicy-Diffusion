import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionEncoder(nn.Module):
    """1D-CNN encoder: (B, action_dim, K) -> mu, logvar of shape (B, latent_dim)"""

    def __init__(self, action_dim: int, latent_dim: int, k_steps: int):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.k_steps = k_steps

        self.conv1 = nn.Conv1d(out_channels=32, in_channels=7, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(out_channels=64, in_channels=32, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(out_channels=128, in_channels=64, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(out_channels=256, in_channels=128, kernel_size=3, stride=2, padding=1, padding_mode='replicate')
        self.bn4 = nn.BatchNorm1d(256)

        flat_dim = 256 * (k_steps // 8)
        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        # x: (B, action_dim, K)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)


class ActionDecoder(nn.Module):
    """1D-TransposedCNN decoder: z (B, latent_dim) -> (B, action_dim, K)"""

    def __init__(self, action_dim: int, latent_dim: int, k_steps: int):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.k_steps = k_steps
        self.t_compressed = k_steps // 8   # temporal size after encoder (2 for K=16)

        flat_dim = 256 * self.t_compressed
        self.fc = nn.Linear(latent_dim, flat_dim)

        self.deconv1 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1     = nn.BatchNorm1d(128)
        self.deconv2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2     = nn.BatchNorm1d(64)
        self.deconv3 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3     = nn.BatchNorm1d(32)
        self.conv_out = nn.Conv1d(in_channels=32, out_channels=action_dim, kernel_size=3, padding=1)

    def forward(self, z):
        # z: (B, latent_dim)
        x = self.fc(z)
        x = x.view(x.size(0), 256, self.t_compressed)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        return self.conv_out(x)   # (B, action_dim, K) — no activation, raw delta poses


class ActionVAE(nn.Module):
    """
    Full Action VAE for temporal action chunk compression.
    Input : action sequence  (B, action_dim, K)
    Output: reconstructed sequence (B, action_dim, K), mu, logvar
    """
    def __init__(self, action_dim: int = 7, latent_dim: int = 64, k_steps: int = 16):
        super().__init__()
        self.encoder = ActionEncoder(action_dim, latent_dim, k_steps)
        self.decoder = ActionDecoder(action_dim, latent_dim, k_steps)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # deterministic at inference

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mean latent (no sampling). Used for diffusion training."""
        mu, _ = self.encoder(x)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector back to an action chunk."""
        return self.decoder(z)


def vae_loss(recon: torch.Tensor,
             target: torch.Tensor,
             mu: torch.Tensor,
             logvar: torch.Tensor,
             beta: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO loss = MSE reconstruction + β * KL divergence.
    Args:
        recon:   (B, action_dim, K) — reconstructed action chunk
        target:  (B, action_dim, K) — ground-truth action chunk
        mu:      (B, latent_dim)
        logvar:  (B, latent_dim)
        beta:    KL weight; ramp from 0 → target with a schedule to avoid posterior collapse
    Returns:
        total_loss, recon_loss, kl_loss  (all scalars)
    """
    recon_loss = F.mse_loss(recon, target, reduction='mean')
    # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))  averaged over batch
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
