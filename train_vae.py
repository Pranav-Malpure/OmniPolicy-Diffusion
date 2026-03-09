"""
Training script for the Action VAE.

Usage:
    python train_vae.py --panda-h5 demos/panda.h5 --xarm6-h5 demos/xarm6.h5
    python train_vae.py --panda-h5 demos/panda.h5          # single-robot baseline

Key design choices:
  - Beta warmup: KL weight ramps 0 -> beta_max over warmup_epochs to avoid
    posterior collapse (a known failure mode noted in the midterm report).
  - Checkpoints saved on best validation reconstruction loss (not total loss),
    so KL annealing doesn't distort early model selection.
  - Normalisation stats are saved alongside model weights so inference is
    reproducible without the original dataset.
"""

import argparse
import os
import json
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from action_vae import ActionVAE, vae_loss
from data.dataloader import make_dataloaders


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train ActionVAE on ManiSkill trajectories")

    # Data
    p.add_argument("--panda-h5",  type=str, default=None, help="Path to Panda h5 file")
    p.add_argument("--xarm6-h5",  type=str, default=None, help="Path to xArm6 h5 file")
    p.add_argument("--k-steps",   type=int, default=16,   help="Action chunk length")
    p.add_argument("--stride",    type=int, default=1,    help="Sliding window stride")

    # Model
    p.add_argument("--latent-dim",  type=int,   default=64,   help="VAE latent dimension")
    p.add_argument("--action-dim",  type=int,   default=7,    help="Action dimension")

    # Training
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch-size",  type=int,   default=256)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--num-workers", type=int,   default=4)

    # Beta (KL) schedule: ramps linearly from 0 to beta-max over warmup-epochs
    p.add_argument("--beta-max",       type=float, default=1e-4,
                   help="Maximum KL weight (keep small to avoid posterior collapse)")
    p.add_argument("--warmup-epochs",  type=int,   default=50,
                   help="Epochs over which beta ramps from 0 to beta-max")

    # Checkpointing
    p.add_argument("--ckpt-dir",  type=str, default="/pers_vol/checkpoints/vae",
                   help="Directory to save checkpoints and normalisation stats")
    p.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   help="'auto', 'cpu', 'cuda', or 'mps'")

    # W&B
    p.add_argument("--wandb-run",  type=str, default=None,
                   help="W&B run name (default: auto-generated)")
    p.add_argument("--no-wandb",   action="store_true",
                   help="Disable W&B logging entirely")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def beta_schedule(epoch: int, warmup_epochs: int, beta_max: float) -> float:
    """Linear warmup from 0 to beta_max, then constant."""
    if warmup_epochs <= 0:
        return beta_max
    return beta_max * min(1.0, epoch / warmup_epochs)


def run_epoch(model, loader, optimizer, beta, device, training: bool):
    model.train(training)
    total_loss = recon_sum = kl_sum = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for chunks, _ in loader:            # robot_id not used by VAE
            chunks = chunks.to(device)      # (B, 7, K)

            recon, mu, logvar = model(chunks)
            loss, recon_l, kl_l = vae_loss(recon, chunks, mu, logvar, beta=beta)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            recon_sum  += recon_l.item()
            kl_sum     += kl_l.item()
            n_batches  += 1

    return total_loss / n_batches, recon_sum / n_batches, kl_sum / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.panda_h5 is None and args.xarm6_h5 is None:
        raise ValueError("Provide at least one of --panda-h5 or --xarm6-h5")

    # Append timestamp so every run gets its own directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.ckpt_dir = f"{args.ckpt_dir}_{timestamp}"

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint dir: {args.ckpt_dir}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # W&B init
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
            tags     = ["vae", "action-vae"] + robots,
            save_code= True,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_loader, val_loader, dataset = make_dataloaders(
        panda_h5    = args.panda_h5,
        xarm6_h5    = args.xarm6_h5,
        k_steps     = args.k_steps,
        stride      = args.stride,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        normalize   = True,
        seed        = args.seed,
    )

    # Save normalisation stats so inference doesn't need the dataset
    mean, std = dataset.get_normalisation()
    if mean is not None and std is not None:
        norm_path = os.path.join(args.ckpt_dir, "norm_stats.pt")
        torch.save({"mean": mean, "std": std}, norm_path)
        print(f"Normalisation stats saved → {norm_path}")
        if use_wandb:
            wandb.save(norm_path)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = ActionVAE(
        action_dim = args.action_dim,
        latent_dim = args.latent_dim,
        k_steps    = args.k_steps,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ActionVAE — {n_params:,} trainable parameters")
    if use_wandb:
        wandb.config.update({"n_params": n_params}, allow_val_change=True)
        wandb.watch(model, log="gradients", log_freq=100)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_recon = float("inf")
    history = []

    print(f"\n{'Epoch':>6}  {'β':>8}  "
          f"{'Train loss':>11}  {'Train recon':>12}  {'Train KL':>9}  "
          f"{'Val loss':>9}  {'Val recon':>10}  {'Val KL':>7}  {'Time':>6}")
    print("-" * 95)

    for epoch in range(1, args.epochs + 1):
        beta = beta_schedule(epoch, args.warmup_epochs, args.beta_max)
        t0   = time.time()

        tr_loss, tr_recon, tr_kl = run_epoch(model, train_loader, optimizer, beta, device, training=True)
        va_loss, va_recon, va_kl = run_epoch(model, val_loader,   optimizer, beta, device, training=False)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"{epoch:>6}  {beta:>8.2e}  "
              f"{tr_loss:>11.6f}  {tr_recon:>12.6f}  {tr_kl:>9.6f}  "
              f"{va_loss:>9.6f}  {va_recon:>10.6f}  {va_kl:>7.6f}  {elapsed:>5.1f}s")

        row = {
            "epoch": epoch, "beta": beta,
            "train_loss": tr_loss, "train_recon": tr_recon, "train_kl": tr_kl,
            "val_loss":   va_loss, "val_recon":   va_recon, "val_kl":   va_kl,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        if use_wandb:
            wandb.log(row)

        # Save best model (by val reconstruction — robust to beta schedule)
        if va_recon < best_val_recon:
            best_val_recon = va_recon
            best_path = os.path.join(args.ckpt_dir, "vae_best.pt")
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_recon":  va_recon,
                "args":       vars(args),
            }, best_path)
            if use_wandb:
                wandb.run.summary["best_val_recon"] = va_recon   # type: ignore[union-attr]
                wandb.run.summary["best_epoch"]     = epoch      # type: ignore[union-attr]
                wandb.save(best_path)

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"vae_epoch{epoch:04d}.pt")
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args":      vars(args),
            }, ckpt_path)

    # -----------------------------------------------------------------------
    # Save final model + training history
    # -----------------------------------------------------------------------
    final_path = os.path.join(args.ckpt_dir, "vae_final.pt")
    torch.save({"epoch": args.epochs, "model": model.state_dict(), "args": vars(args)}, final_path)

    history_path = os.path.join(args.ckpt_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val recon = {best_val_recon:.6f}")
    print(f"  Best model  → {best_path}")
    print(f"  Final model → {final_path}")
    print(f"  History     → {history_path}")

    if use_wandb:
        wandb.save(final_path)
        wandb.save(history_path)
        wandb.finish()


if __name__ == "__main__":
    main()
