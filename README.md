# OmniPolicy-Diffusion
Cross-Embodiment Latent Diffusion Policy for robotic manipulation — a single generative model controlling Franka Panda (7-DOF) and UFactory xArm6 (6-DOF) on PickCube-v1 in ManiSkill 3.

## File Overview

- **`action_vae.py`** — 1D-CNN Action VAE: encoder, decoder, reparameterization, and VAE loss with β-KL scheduling. Compresses K=16 action chunks into a 64-dim latent.
- **`diffusion_transformer.py`** — Diffusion Transformer (DiT) with AdaLN-Zero conditioning, DDPM/DDIM noise schedule, and observation encoders (StateEncoder, VisualEncoder) for the three conditioning modes.
- **`train_vae.py`** — Training script for the Action VAE with β warmup, cosine LR schedule, best-checkpoint saving, and W&B logging to project `OmniPolicyDiffusion`.
- **`train_dit.py`** — Training script for the DiT on frozen VAE latents; supports `--obs-mode none|state|rgb` to switch conditioning, with full W&B logging.
- **`eval_policy.py`** — Evaluation script that runs the full inference pipeline (DDIM → VAE decode → chunk execution) in ManiSkill, records videos via `RecordEpisode`, and logs success rate, mean return, and cross-embodiment transfer index to W&B.
- **`data/dataloader.py`** — `ActionChunkDataset` (for VAE training) and `DiTDataset` (for DiT training with optional state/RGB observations); sliding window chunking with temporal repeat padding and z-score normalisation.
