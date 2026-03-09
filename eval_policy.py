"""
Evaluation script for the OmniPolicy Latent Diffusion Policy.

Runs the full inference pipeline in ManiSkill:
  obs → ObsEncoder → DiT DDIM → z₀ → VAE decode → action chunk → env

Records videos via RecordEpisode and logs everything (success rate, return,
inference latency, videos) to W&B project OmniPolicyDiffusion.

Usage:
    # Evaluate obs-mode=none checkpoint on both robots
    python eval_policy.py \\
        --dit-ckpt  checkpoints/dit/dit_best.pt \\
        --vae-ckpt  checkpoints/vae/vae_best.pt \\
        --obs-mode  none

    # Evaluate rgb checkpoint, panda only
    python eval_policy.py \\
        --dit-ckpt  checkpoints/dit/dit_best.pt \\
        --vae-ckpt  checkpoints/vae/vae_best.pt \\
        --obs-mode  rgb \\
        --robots    panda

    # Cross-embodiment transfer index: pass two --dit-ckpt paths
    #   first = single-robot model, second = cross-robot model
    python eval_policy.py \\
        --dit-ckpt  checkpoints/dit_panda_only/dit_best.pt \\
        --dit-ckpt2 checkpoints/dit_cross/dit_best.pt \\
        --vae-ckpt  checkpoints/vae/vae_best.pt \\
        --obs-mode  none
"""

import argparse
import os
import time
from collections import defaultdict
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import wandb

import mani_skill.envs  # noqa: F401 — registers ManiSkill envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from action_vae import ActionVAE
from diffusion_transformer import (
    DiffusionTransformer, DDPMSchedule, StateEncoder, VisualEncoder,
)
from data.dataloader import ROBOT_IDS, MAX_STATE_DIM

# ---------------------------------------------------------------------------
# Robot configs
# ---------------------------------------------------------------------------
ROBOT_CONFIGS = {
    "panda": {
        "robot_uids": "panda",
        "robot_id":    ROBOT_IDS["panda"],
    },
    "xarm6": {
        "robot_uids": "xarm6_robotiq",
        "robot_id":    ROBOT_IDS["xarm6"],
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate OmniPolicy DiT in ManiSkill")

    # Checkpoints
    p.add_argument("--dit-ckpt",  type=str, required=True,
                   help="DiT checkpoint (dit_best.pt)")
    p.add_argument("--dit-ckpt2", type=str, default=None,
                   help="Optional second DiT ckpt for cross-embodiment transfer index")
    p.add_argument("--vae-ckpt",  type=str, required=True,
                   help="VAE checkpoint (vae_best.pt)")
    p.add_argument("--norm-stats", type=str, default=None,
                   help="Path to norm_stats.pt; inferred from --vae-ckpt dir if not given")

    # Policy
    p.add_argument("--obs-mode",   type=str, default="none",
                   choices=["none", "state", "rgb"])
    p.add_argument("--ddim-steps", type=int, default=20,
                   help="DDIM steps for inference (20 is fast; 100 for higher quality)")
    p.add_argument("--k-steps",    type=int, default=16,
                   help="Action chunk length — must match training")

    # Evaluation
    p.add_argument("--robots",   type=str, nargs="+",
                   default=["panda", "xarm6"],
                   choices=["panda", "xarm6"],
                   help="Which robots to evaluate")
    p.add_argument("--n-episodes", type=int, default=50,
                   help="Episodes per robot")
    p.add_argument("--max-steps",  type=int, default=100,
                   help="Max env steps per episode")

    # Recording
    p.add_argument("--video-dir", type=str, default="eval_videos",
                   help="Directory to save video files")
    p.add_argument("--no-video",  action="store_true",
                   help="Disable video recording")

    # Misc
    p.add_argument("--device",    type=str, default="auto")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--wandb-run", type=str, default=None)
    p.add_argument("--no-wandb",  action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():        return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def load_vae(vae_ckpt: str, device: torch.device) -> tuple[ActionVAE, dict]:
    ckpt    = torch.load(vae_ckpt, map_location=device, weights_only=True)
    saved   = ckpt.get("args", {})
    vae     = ActionVAE(
        action_dim = saved.get("action_dim", 7),
        latent_dim = saved.get("latent_dim", 64),
        k_steps    = saved.get("k_steps",    16),
    ).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, saved


def load_norm_stats(vae_ckpt: str, norm_stats_path: Optional[str], device: torch.device):
    if norm_stats_path is None:
        norm_stats_path = os.path.join(os.path.dirname(vae_ckpt), "norm_stats.pt")
    stats = torch.load(norm_stats_path, map_location=device, weights_only=True)
    mean  = stats["mean"].to(device)   # (7,)
    std   = stats["std"].to(device)    # (7,)
    return mean, std


def load_dit(dit_ckpt: str, obs_mode: str, device: torch.device):
    ckpt   = torch.load(dit_ckpt, map_location=device, weights_only=True)
    saved  = ckpt.get("args", {})

    dit = DiffusionTransformer(
        latent_dim = saved.get("latent_dim", 64),
        embed_dim  = saved.get("embed_dim",  256),
        n_heads    = saved.get("n_heads",    8),
        n_layers   = saved.get("n_layers",   6),
        n_tokens   = saved.get("n_tokens",   8),
        num_robots = saved.get("num_robots", 3),
        max_steps  = saved.get("ddpm_steps", 1000),
    ).to(device)
    dit.load_state_dict(ckpt["dit"])
    dit.eval()

    obs_enc: Optional[torch.nn.Module] = None
    if obs_mode == "state" and "obs_enc" in ckpt:
        obs_enc = StateEncoder(state_dim=MAX_STATE_DIM, embed_dim=saved.get("embed_dim", 256)).to(device)
        obs_enc.load_state_dict(ckpt["obs_enc"])
        obs_enc.eval()
    elif obs_mode == "rgb" and "obs_enc" in ckpt:
        obs_enc = VisualEncoder(embed_dim=saved.get("embed_dim", 256)).to(device)
        obs_enc.load_state_dict(ckpt["obs_enc"])
        obs_enc.eval()

    schedule = DDPMSchedule(
        n_steps    = saved.get("ddpm_steps",  1000),
        beta_start = saved.get("beta_start",  1e-4),
        beta_end   = saved.get("beta_end",    0.02),
        device     = str(device),
    )

    return dit, obs_enc, schedule, saved.get("latent_dim", 64)


def make_env(robot_name: str, obs_mode: str, output_dir: str,
             n_episodes: int, max_steps: int, record: bool, seed: int):
    """Create a single-env ManiSkill wrapped with RecordEpisode for video."""
    cfg     = ROBOT_CONFIGS[robot_name]
    gym_obs = "rgb+state" if obs_mode in ("rgb", "state") else "state"

    raw_env = gym.make(
        "PickCube-v1",
        num_envs        = 1,
        obs_mode        = gym_obs,
        control_mode    = "pd_ee_delta_pose",
        render_mode     = "rgb_array",
        robot_uids      = cfg["robot_uids"],
        sim_backend     = "gpu" if torch.cuda.is_available() else "cpu",
        reconfiguration_freq = 1,   # re-randomise cube position every reset
    )

    raw_env = FlattenRGBDObservationWrapper(
        raw_env,
        rgb   = (obs_mode == "rgb"),
        depth = False,
        state = True,
    )

    if record:
        os.makedirs(output_dir, exist_ok=True)
        raw_env = RecordEpisode(
            raw_env,
            output_dir           = output_dir,
            save_video           = True,
            save_trajectory      = False,
            trajectory_name      = f"{robot_name}_eval",
            max_steps_per_video  = max_steps,
            video_fps            = 30,
            wandb_video_freq     = 1,   # upload every episode to W&B
        )

    env = ManiSkillVectorEnv(raw_env, num_envs=1,
                             ignore_terminations=False, record_metrics=True)
    return env


@torch.no_grad()
def extract_obs_emb(obs: dict, obs_mode: str,
                    obs_enc: Optional[torch.nn.Module],
                    device: torch.device) -> Optional[torch.Tensor]:
    """Convert env obs dict → vis_emb tensor for DiT, or None."""
    if obs_mode == "none" or obs_enc is None:
        return None

    if obs_mode == "state":
        state = obs["state"].float().to(device)    # (1, state_dim)
        # Zero-pad to MAX_STATE_DIM
        B, D = state.shape
        if D < MAX_STATE_DIM:
            pad   = torch.zeros(B, MAX_STATE_DIM - D, device=device)
            state = torch.cat([state, pad], dim=1)
        else:
            state = state[:, :MAX_STATE_DIM]
        return obs_enc(state)                      # (1, embed_dim)

    if obs_mode == "rgb":
        rgb = obs["rgb"].float().to(device)        # (1, H, W, C) or (1, C, H, W)
        if rgb.shape[-1] in (3, 4):                # (B, H, W, C) → (B, C, H, W)
            rgb = rgb.permute(0, 3, 1, 2)
        rgb = rgb[..., :3, :, :] / 255.0 if rgb.max() > 1.0 else rgb[..., :3, :, :]
        return obs_enc(rgb)                        # (1, embed_dim)

    return None


@torch.no_grad()
def run_policy_episode(
    env,
    dit:       DiffusionTransformer,
    obs_enc:   Optional[torch.nn.Module],
    vae:       ActionVAE,
    schedule:  DDPMSchedule,
    norm_mean: torch.Tensor,      # (7,)
    norm_std:  torch.Tensor,      # (7,)
    robot_id:  int,
    obs_mode:  str,
    k_steps:   int,
    latent_dim: int,
    ddim_steps: int,
    max_steps:  int,
    device:    torch.device,
) -> dict:
    obs, _ = env.reset()
    done   = torch.zeros(1, dtype=torch.bool)
    total_reward = 0.0
    success      = False
    step         = 0
    plan_times   = []

    rid = torch.tensor([robot_id], device=device, dtype=torch.long)

    while step < max_steps and not done.all():
        # ---- plan a new chunk ----
        t_plan = time.perf_counter()

        vis_emb = extract_obs_emb(obs, obs_mode, obs_enc, device)
        z_T     = torch.randn(1, latent_dim, device=device)
        z0      = schedule.ddim_sample(dit, z_T, rid,
                                       ddim_steps=ddim_steps,
                                       vis_emb=vis_emb)           # (1, latent_dim)
        chunk   = vae.decode(z0)                                  # (1, 7, K)
        # Denormalise: chunk is in z-score space
        chunk   = chunk * norm_std[None, :, None] + norm_mean[None, :, None]
        chunk   = chunk.squeeze(0).T.cpu().numpy()                # (K, 7)

        plan_times.append(time.perf_counter() - t_plan)

        # ---- execute K steps ----
        for k in range(k_steps):
            if done.all() or step >= max_steps:
                break
            action = torch.from_numpy(chunk[k]).unsqueeze(0)     # (1, 7)
            obs, rew, terminated, truncated, info = env.step(action.to(device))
            total_reward += float(rew.cpu().mean())
            done          = terminated | truncated
            step         += 1

        if "final_info" in info:
            s = info["final_info"].get("success", None)
            if s is not None:
                success = bool(s.any().cpu())

    # also check last info
    if not success and "success" in info:
        success = bool(info["success"].any().cpu())

    return {
        "success":        success,
        "total_reward":   total_reward,
        "steps":          step,
        "mean_plan_ms":   float(np.mean(plan_times)) * 1000,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = get_device(args.device)
    torch.manual_seed(args.seed)

    # -----------------------------------------------------------------------
    # W&B
    # -----------------------------------------------------------------------
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project  = "OmniPolicyDiffusion",
            name     = args.wandb_run,
            config   = vars(args),
            tags     = ["eval", f"obs-{args.obs_mode}"] + args.robots,
            save_code= True,
        )

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    vae, vae_args  = load_vae(args.vae_ckpt, device)
    norm_mean, norm_std = load_norm_stats(args.vae_ckpt, args.norm_stats, device)
    dit, obs_enc, schedule, latent_dim = load_dit(args.dit_ckpt, args.obs_mode, device)

    # Optional second checkpoint for cross-embodiment transfer index
    dit2: Optional[DiffusionTransformer] = None
    obs_enc2: Optional[torch.nn.Module]  = None
    if args.dit_ckpt2:
        dit2, obs_enc2, _, _ = load_dit(args.dit_ckpt2, args.obs_mode, device)

    print(f"\nDevice : {device}")
    print(f"ObsMode: {args.obs_mode}  |  DDIM steps: {args.ddim_steps}")
    print(f"Robots : {args.robots}  |  Episodes per robot: {args.n_episodes}")

    # -----------------------------------------------------------------------
    # Evaluate per robot
    # -----------------------------------------------------------------------
    all_results: dict[str, dict] = {}   # robot → {model_tag → metrics}

    def evaluate_model(tag: str, d: DiffusionTransformer, oe: Optional[torch.nn.Module]):
        for robot_name in args.robots:
            cfg      = ROBOT_CONFIGS[robot_name]
            video_dir = os.path.join(args.video_dir, f"{tag}_{robot_name}")
            env      = make_env(
                robot_name  = robot_name,
                obs_mode    = args.obs_mode,
                output_dir  = video_dir,
                n_episodes  = args.n_episodes,
                max_steps   = args.max_steps,
                record      = not args.no_video,
                seed        = args.seed,
            )

            successes, returns, latencies = [], [], []
            print(f"\n[{tag}] {robot_name}  ({args.n_episodes} episodes)")

            for ep in range(args.n_episodes):
                result = run_policy_episode(
                    env        = env,
                    dit        = d,
                    obs_enc    = oe,
                    vae        = vae,
                    schedule   = schedule,
                    norm_mean  = norm_mean,
                    norm_std   = norm_std,
                    robot_id   = cfg["robot_id"],
                    obs_mode   = args.obs_mode,
                    k_steps    = args.k_steps,
                    latent_dim = latent_dim,
                    ddim_steps = args.ddim_steps,
                    max_steps  = args.max_steps,
                    device     = device,
                )
                successes.append(result["success"])
                returns.append(result["total_reward"])
                latencies.append(result["mean_plan_ms"])

                status = "✓" if result["success"] else "✗"
                print(f"  ep {ep+1:3d}/{args.n_episodes}  {status}  "
                      f"return={result['total_reward']:.2f}  "
                      f"plan={result['mean_plan_ms']:.1f}ms")

            sr      = float(np.mean(successes))
            ret     = float(np.mean(returns))
            lat_ms  = float(np.mean(latencies))

            print(f"  → SR={sr:.2%}  MeanReturn={ret:.3f}  LatencyPerPlan={lat_ms:.1f}ms")

            key = f"{tag}/{robot_name}"
            all_results[key] = {"sr": sr, "return": ret, "latency_ms": lat_ms}

            if use_wandb:
                wandb.log({
                    f"{key}/success_rate":    sr,
                    f"{key}/mean_return":     ret,
                    f"{key}/latency_ms":      lat_ms,
                    f"{key}/ddim_steps":      args.ddim_steps,
                })

            env.close()

    # Run main checkpoint
    evaluate_model("model1", dit, obs_enc)

    # Run second checkpoint (for transfer index)
    if dit2 is not None:
        evaluate_model("model2", dit2, obs_enc2)

    # -----------------------------------------------------------------------
    # Cross-Embodiment Transfer Index
    # -----------------------------------------------------------------------
    if dit2 is not None and len(args.robots) >= 2:
        print("\n--- Cross-Embodiment Transfer Index ---")
        # Transfer index = (SR_cross - SR_single) / SR_single  for each robot
        for robot in args.robots:
            k1 = f"model1/{robot}"
            k2 = f"model2/{robot}"
            if k1 in all_results and k2 in all_results:
                sr1 = all_results[k1]["sr"]
                sr2 = all_results[k2]["sr"]
                ti  = (sr2 - sr1) / (sr1 + 1e-8)
                print(f"  {robot}: single={sr1:.2%}  cross={sr2:.2%}  TI={ti:+.3f}")
                if use_wandb:
                    wandb.log({f"transfer_index/{robot}": ti})

    # -----------------------------------------------------------------------
    # Summary table to W&B
    # -----------------------------------------------------------------------
    if use_wandb:
        table = wandb.Table(columns=["model", "robot", "success_rate", "mean_return", "latency_ms"])
        for key, metrics in all_results.items():
            tag, robot = key.split("/", 1)
            table.add_data(tag, robot,
                           round(metrics["sr"], 4),
                           round(metrics["return"], 4),
                           round(metrics["latency_ms"], 2))
        wandb.log({"eval/summary": table})
        wandb.finish()

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
