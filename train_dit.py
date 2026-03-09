"""
Training script for the Latent Diffusion Transformer (DiT).

Requires a pre-trained ActionVAE checkpoint (from train_vae.py).
The VAE encoder is frozen; the DiT learns to denoise in the VAE latent space.

Usage:
    # Phase 1 — robot-id + timestep conditioning only (fastest, good baseline)
    python train_dit.py --panda-h5 demos/panda.h5 --xarm6-h5 demos/xarm6.h5 \\
        --vae-ckpt checkpoints/vae/vae_best.pt --obs-mode none

    # Phase 2 — add proprioceptive state conditioning
    python train_dit.py ... --obs-mode state

    # Phase 3 — add RGB visual conditioning
    python train_dit.py ... --obs-mode rgb

--obs-mode controls what gets injected into the DiT's vis_emb slot:
  none  : only robot_id + diffusion timestep  (DiT paper baseline)
  state : + proprioceptive state MLP          (cheap, no image needed)
  rgb   : + CNN-encoded RGB frame             (full visual policy)
"""

import argparse
import os
import json
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from action_vae import ActionVAE
from diffusion_transformer import (
    DiffusionTransformer, DDPMSchedule, StateEncoder, VisualEncoder,
)
from data.dataloader import make_dit_dataloaders, MAX_STATE_DIM


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Latent DiT on frozen VAE latents")

    # Data
    p.add_argument("--panda-h5",  type=str, default=None)
    p.add_argument("--xarm6-h5",  type=str, default=None)
    p.add_argument("--k-steps",   type=int, default=16)
    p.add_argument("--stride",    type=int, default=8)

    # VAE (frozen)
    p.add_argument("--vae-ckpt",   type=str, required=True,
                   help="Path to vae_best.pt from train_vae.py")
    p.add_argument("--norm-stats", type=str, default=None,
                   help="Path to norm_stats.pt; if None, inferred from --vae-ckpt directory")

    # Observation mode
    p.add_argument("--obs-mode",  type=str, default="none",
                   choices=["none", "state", "rgb"],
                   help="Conditioning: none | state (proprioception) | rgb (camera)")

    # DiT model
    p.add_argument("--latent-dim", type=int,   default=64)
    p.add_argument("--embed-dim",  type=int,   default=256)
    p.add_argument("--n-heads",    type=int,   default=8)
    p.add_argument("--n-layers",   type=int,   default=6)
    p.add_argument("--n-tokens",   type=int,   default=8)
    p.add_argument("--num-robots", type=int,   default=3)

    # Diffusion schedule
    p.add_argument("--ddpm-steps",  type=int,   default=1000)
    p.add_argument("--beta-start",  type=float, default=1e-4)
    p.add_argument("--beta-end",    type=float, default=0.02)

    # Training
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num-workers", type=int,   default=4)

    # Checkpointing
    p.add_argument("--ckpt-dir",   type=str, default="checkpoints/dit")
    p.add_argument("--save-every", type=int, default=10)

    # Misc
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--device",  type=str, default="auto")
    p.add_argument("--wandb-run", type=str, default=None)
    p.add_argument("--no-wandb",  action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():   return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def load_frozen_vae(ckpt_path: str, args, device: torch.device) -> ActionVAE:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    saved = ckpt.get("args", {})
    vae = ActionVAE(
        action_dim = saved.get("action_dim", args.latent_dim),
        latent_dim = saved.get("latent_dim", args.latent_dim),
        k_steps    = saved.get("k_steps",    args.k_steps),
    ).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"[VAE] Loaded frozen encoder from {ckpt_path}")
    return vae


def build_obs_encoder(obs_mode: str, embed_dim: int, device: torch.device) -> Optional[nn.Module]:
    if obs_mode == "state":
        enc = StateEncoder(state_dim=MAX_STATE_DIM, embed_dim=embed_dim).to(device)
        print(f"[ObsEnc] StateEncoder — {sum(p.numel() for p in enc.parameters()):,} params")
        return enc
    if obs_mode == "rgb":
        enc = VisualEncoder(embed_dim=embed_dim).to(device)
        print(f"[ObsEnc] VisualEncoder — {sum(p.numel() for p in enc.parameters()):,} params")
        return enc
    return None   # obs_mode == "none"


def run_epoch(
    dit:      DiffusionTransformer,
    obs_enc:  Optional[nn.Module],
    vae:      ActionVAE,
    loader,
    schedule: DDPMSchedule,
    optimizer: Optional[optim.Optimizer],
    device:   torch.device,
    training: bool,
) -> float:
    dit.train(training)
    if obs_enc is not None:
        obs_enc.train(training)

    total_loss = 0.0
    n_batches  = 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for chunks, robot_ids, obs in loader:
            chunks    = chunks.to(device)                    # (B, 7, K)
            robot_ids = robot_ids.to(device)                 # (B,)
            obs       = obs.to(device)

            # Encode actions → clean latent z0 (frozen VAE, no grad)
            with torch.no_grad():
                z0 = vae.encode(chunks)                      # (B, latent_dim)

            # Sample random timesteps and add noise
            t    = torch.randint(0, schedule.n_steps, (z0.size(0),), device=device)
            z_t, eps = schedule.q_sample(z0, t)              # both (B, latent_dim)

            # Build visual/state embedding if needed
            vis_emb: Optional[torch.Tensor] = None
            if obs_enc is not None:
                vis_emb = obs_enc(obs)

            # Predict noise
            eps_pred = dit(z_t, t, robot_ids, vis_emb)      # (B, latent_dim)
            loss     = nn.functional.mse_loss(eps_pred, eps)

            if training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(dit.parameters()) + (list(obs_enc.parameters()) if obs_enc else []),
                    max_norm=1.0,
                )
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.panda_h5 is None and args.xarm6_h5 is None:
        raise ValueError("Provide at least one of --panda-h5 or --xarm6-h5")

    device = get_device(args.device)
    print(f"Device  : {device}")
    print(f"Obs mode: {args.obs_mode}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # W&B
    # -----------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        robots = []
        if args.panda_h5:  robots.append("panda")
        if args.xarm6_h5:  robots.append("xarm6")
        wandb.init(
            project  = "OmniPolicyDiffusion",
            name     = args.wandb_run,
            config   = vars(args),
            tags     = ["dit", f"obs-{args.obs_mode}"] + robots,
            save_code= True,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_loader, val_loader, _ = make_dit_dataloaders(
        panda_h5    = args.panda_h5,
        xarm6_h5    = args.xarm6_h5,
        obs_mode    = args.obs_mode,
        k_steps     = args.k_steps,
        stride      = args.stride,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        normalize   = True,
        seed        = args.seed,
    )

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------
    vae     = load_frozen_vae(args.vae_ckpt, args, device)
    obs_enc = build_obs_encoder(args.obs_mode, args.embed_dim, device)

    dit = DiffusionTransformer(
        latent_dim = args.latent_dim,
        embed_dim  = args.embed_dim,
        n_heads    = args.n_heads,
        n_layers   = args.n_layers,
        n_tokens   = args.n_tokens,
        num_robots = args.num_robots,
        max_steps  = args.ddpm_steps,
    ).to(device)

    schedule = DDPMSchedule(args.ddpm_steps, args.beta_start, args.beta_end, str(device))

    total_params = dit.count_params() + (
        sum(p.numel() for p in obs_enc.parameters()) if obs_enc else 0
    )
    print(f"DiT — {dit.count_params():,} params  |  ObsEnc — {total_params - dit.count_params():,} params")

    # Joint optimiser over DiT + obs encoder
    all_params = list(dit.parameters()) + (list(obs_enc.parameters()) if obs_enc else [])
    optimizer  = optim.AdamW(all_params, lr=args.lr, weight_decay=1e-5)
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    if use_wandb:
        wandb.config.update({"total_params": total_params}, allow_val_change=True)
        wandb.watch(dit, log="gradients", log_freq=200)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_loss = float("inf")
    history = []

    print(f"\n{'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>9}  {'LR':>8}  {'Time':>6}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss = run_epoch(dit, obs_enc, vae, train_loader, schedule, optimizer,  device, training=True)
        va_loss = run_epoch(dit, obs_enc, vae, val_loader,   schedule, None,       device, training=False)

        scheduler.step()
        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{epoch:>6}  {tr_loss:>10.6f}  {va_loss:>9.6f}  {lr:>8.2e}  {elapsed:>5.1f}s")

        row = {"epoch": epoch, "train_mse": tr_loss, "val_mse": va_loss, "lr": lr}
        history.append(row)
        if use_wandb:
            wandb.log(row)

        # Best checkpoint
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_path = os.path.join(args.ckpt_dir, "dit_best.pt")
            ckpt = {
                "epoch":     epoch,
                "dit":       dit.state_dict(),
                "val_mse":   va_loss,
                "args":      vars(args),
            }
            if obs_enc is not None:
                ckpt["obs_enc"] = obs_enc.state_dict()
            torch.save(ckpt, best_path)
            if use_wandb:
                wandb.run.summary["best_val_mse"] = va_loss   # type: ignore[union-attr]
                wandb.run.summary["best_epoch"]   = epoch     # type: ignore[union-attr]
                wandb.save(best_path)

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"dit_epoch{epoch:04d}.pt")
            ckpt = {"epoch": epoch, "dit": dit.state_dict(), "args": vars(args)}
            if obs_enc is not None:
                ckpt["obs_enc"] = obs_enc.state_dict()
            torch.save(ckpt, ckpt_path)

    # -----------------------------------------------------------------------
    # Save final + history
    # -----------------------------------------------------------------------
    final_path = os.path.join(args.ckpt_dir, "dit_final.pt")
    final_ckpt = {"epoch": args.epochs, "dit": dit.state_dict(), "args": vars(args)}
    if obs_enc is not None:
        final_ckpt["obs_enc"] = obs_enc.state_dict()
    torch.save(final_ckpt, final_path)

    history_path = os.path.join(args.ckpt_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val MSE = {best_val_loss:.6f}")
    print(f"  Best  → {best_path}")
    print(f"  Final → {final_path}")

    if use_wandb:
        wandb.save(final_path)
        wandb.save(history_path)
        wandb.finish()


if __name__ == "__main__":
    main()
