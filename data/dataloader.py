"""
ActionChunkDataset / DiTDataset: load 7-DOF delta end-effector actions from ManiSkill
h5 files, chunk into fixed-length windows, and optionally return observations.

ActionChunkDataset  — used by VAE training (returns chunk, robot_id)
DiTDataset          — used by DiT training (returns chunk, robot_id, obs)

obs_mode options for DiTDataset:
  "none"  — no visual context; obs is a 1-D zero placeholder
  "state" — proprioceptive state vector, zero-padded to MAX_STATE_DIM
  "rgb"   — RGB image (3, H, W) normalised to [0, 1]
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


# Robot ID registry — extend as needed
ROBOT_IDS = {"panda": 0, "xarm6": 1, "xarm7": 2}

# Largest proprioceptive state dim across all supported robots; smaller states are zero-padded
MAX_STATE_DIM = 64

OBS_MODES = ("none", "state", "rgb")


class ActionChunkDataset(Dataset):
    """
    Builds a flat index of all (file, episode, start_step) triples across
    one or more h5 files.  Each __getitem__ returns a (K, 7) action chunk
    and its integer robot_id.

    Args:
        h5_paths:   list of (path, robot_name) tuples, e.g.
                    [("demos/panda.h5", "panda"), ("demos/xarm6.h5", "xarm6")]
        k_steps:    chunk length (paper uses K=16)
        stride:     sliding window stride (stride=1 → maximum data; stride=K → non-overlapping)
        normalize:  if True, z-score each feature dim using stats computed at init
    """

    def __init__(self,
                 h5_paths: list[tuple[str, str]],
                 k_steps: int = 16,
                 stride: int = 1,
                 normalize: bool = True):
        self.k_steps   = k_steps
        self.stride    = stride
        self.normalize = normalize

        # index: list of (file_idx, episode_key, start_t, actual_len)
        # actual_len == k_steps for full windows; < k_steps for the padded tail chunk
        self._index: list[tuple[int, str, int, int]] = []
        self._files: list[tuple[str, int]] = []   # (path, robot_id) per file_idx

        for path, robot_name in h5_paths:
            assert robot_name in ROBOT_IDS, f"Unknown robot '{robot_name}'. Add it to ROBOT_IDS."
            robot_id  = ROBOT_IDS[robot_name]
            file_idx  = len(self._files)
            self._files.append((path, robot_id))

            with h5py.File(path, 'r') as f:
                for ep_key in f.keys():
                    ep_grp = f[ep_key]
                    assert isinstance(ep_grp, h5py.Group)
                    ds = ep_grp['actions']
                    assert isinstance(ds, h5py.Dataset)
                    T = ds.shape[0]   # number of action steps
                    for start in range(0, T, stride):
                        actual_len = min(k_steps, T - start)
                        self._index.append((file_idx, ep_key, start, actual_len))

        print(f"[ActionChunkDataset] {len(self._index):,} chunks from {len(self._files)} file(s)")
        for path, rid in self._files:
            print(f"  robot_id={rid}  {path}")

        # Compute normalisation stats from a random subsample (max 50k chunks)
        if normalize:
            self._mean, self._std = self._compute_stats(max_samples=50_000)
        else:
            self._mean: Optional[torch.Tensor] = None
            self._std:  Optional[torch.Tensor] = None

        # Handles are opened lazily per-worker to avoid sharing file descriptors
        # across forked DataLoader workers (unsafe with h5py).
        self._h5_handles: list[h5py.File | None] = [None] * len(self._files)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def _get_handle(self, file_idx: int) -> h5py.File:
        """Return (and lazily open) the H5 handle for file_idx.

        Opening on first access means each forked DataLoader worker gets its
        own file descriptor rather than sharing one inherited from the parent.
        """
        if self._h5_handles[file_idx] is None:
            path = self._files[file_idx][0]
            self._h5_handles[file_idx] = h5py.File(path, 'r')
        return self._h5_handles[file_idx]  # type: ignore[return-value]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_idx, ep_key, start, actual_len = self._index[idx]
        robot_id = self._files[file_idx][1]

        ep_grp = self._get_handle(file_idx)[ep_key]
        assert isinstance(ep_grp, h5py.Group)
        ds = ep_grp['actions']
        assert isinstance(ds, h5py.Dataset)
        actions: np.ndarray = ds[start : start + actual_len]   # (actual_len, 7)

        if actual_len < self.k_steps:
            pad = np.repeat(actions[-1:], self.k_steps - actual_len, axis=0)
            actions = np.concatenate([actions, pad], axis=0)  # (K, 7)

        chunk = torch.from_numpy(actions.astype(np.float32))  # (K, 7)

        if self.normalize and self._mean is not None and self._std is not None:
            chunk = (chunk - self._mean) / self._std

        # VAE expects (action_dim, K) — transpose here
        chunk = chunk.T  # (7, K)

        return chunk, robot_id

    # ------------------------------------------------------------------
    def _compute_stats(self, max_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-dim mean and std over a random subsample of chunks."""
        rng     = np.random.default_rng(42)
        n       = min(max_samples, len(self._index))
        chosen  = rng.choice(len(self._index), size=n, replace=False)

        all_actions = []
        for idx in chosen:
            file_idx, ep_key, start, actual_len = self._index[idx]
            path = self._files[file_idx][0]
            with h5py.File(path, 'r') as f:
                ep_grp = f[ep_key]
                assert isinstance(ep_grp, h5py.Group)
                ds = ep_grp['actions']
                assert isinstance(ds, h5py.Dataset)
                a: np.ndarray = ds[start : start + actual_len]
                if actual_len < self.k_steps:
                    pad = np.repeat(a[-1:], self.k_steps - actual_len, axis=0)
                    a = np.concatenate([a, pad], axis=0)
                all_actions.append(a.astype(np.float32))

        all_actions = np.stack(all_actions)  # (n, K, 7)
        flat        = all_actions.reshape(-1, 7)
        mean        = torch.from_numpy(flat.mean(0))   # (7,)
        std         = torch.from_numpy(flat.std(0).clip(min=1e-6))
        print(f"[ActionChunkDataset] normalisation mean={mean.numpy().round(4)}")
        print(f"[ActionChunkDataset] normalisation std ={std.numpy().round(4)}")
        return mean, std

    def get_normalisation(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (mean, std) tensors — save these alongside model weights."""
        return self._mean, self._std

    def __del__(self):
        for h in getattr(self, '_h5_handles', []):
            if h is not None:
                try:
                    h.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_dataloaders(
    panda_h5: Optional[str],
    xarm6_h5: Optional[str],
    k_steps:   int   = 16,
    stride:    int   = 1,
    batch_size: int  = 256,
    val_split:  float = 0.05,
    num_workers: int  = 4,
    normalize:   bool = True,
    seed:        int  = 42,
) -> tuple[DataLoader, DataLoader, ActionChunkDataset]:
    """
    Returns (train_loader, val_loader, dataset).

    The dataset object is returned so callers can save normalisation stats.
    Pass xarm6_h5=None (or panda_h5=None) to use a single-robot split.
    """
    h5_paths = []
    if panda_h5:
        h5_paths.append((panda_h5, "panda"))
    if xarm6_h5:
        h5_paths.append((xarm6_h5, "xarm6"))

    dataset = ActionChunkDataset(h5_paths, k_steps=k_steps, stride=stride, normalize=normalize)

    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[make_dataloaders] train={n_train:,}  val={n_val:,}  batch={batch_size}")
    return train_loader, val_loader, dataset


# ---------------------------------------------------------------------------
# DiTDataset — extends ActionChunkDataset with observation support
# ---------------------------------------------------------------------------

class DiTDataset(ActionChunkDataset):
    """
    Same sliding-window action chunks as ActionChunkDataset, but also returns
    an observation tensor at the start of each chunk for DiT conditioning.

    Returns:
        chunk    : (action_dim, K)  normalised action chunk
        robot_id : int
        obs      : tensor whose shape depends on obs_mode:
                     "none"  → (1,)           zero placeholder
                     "state" → (MAX_STATE_DIM,) zero-padded proprioceptive state
                     "rgb"   → (3, H, W)       float32 in [0, 1]
    """

    def __init__(
        self,
        h5_paths:  list[tuple[str, str]],
        k_steps:   int  = 16,
        stride:    int  = 8,
        normalize: bool = True,
        obs_mode:  str  = "none",
    ):
        assert obs_mode in OBS_MODES, f"obs_mode must be one of {OBS_MODES}"
        super().__init__(h5_paths, k_steps=k_steps, stride=stride, normalize=normalize)
        self.obs_mode = obs_mode

    # ------------------------------------------------------------------
    # Override return type via overload to avoid basedpyright incompatibility
    def __getitem__(self, idx: int):  # type: ignore[override]
        chunk, robot_id = super().__getitem__(idx)

        if self.obs_mode == "none":
            obs = torch.zeros(1)
            return chunk, robot_id, obs

        file_idx, ep_key, start, _ = self._index[idx]
        ep_grp = self._get_handle(file_idx)[ep_key]
        assert isinstance(ep_grp, h5py.Group)

        if self.obs_mode == "state":
            obs_grp = ep_grp['obs']
            assert isinstance(obs_grp, h5py.Group)
            ds = obs_grp['state']
            assert isinstance(ds, h5py.Dataset)
            state: np.ndarray = ds[start]          # (state_dim,)
            obs_np = state.astype(np.float32)
            # Zero-pad to MAX_STATE_DIM so batches from different robots collate
            if obs_np.shape[0] < MAX_STATE_DIM:
                obs_np = np.pad(obs_np, (0, MAX_STATE_DIM - obs_np.shape[0]))
            else:
                obs_np = obs_np[:MAX_STATE_DIM]
            obs = torch.from_numpy(obs_np)         # (MAX_STATE_DIM,)

        else:  # "rgb"
            obs_grp = ep_grp['obs']
            assert isinstance(obs_grp, h5py.Group)
            sensor_grp = obs_grp['sensor_data']
            assert isinstance(sensor_grp, h5py.Group)
            cam_grp = sensor_grp['base_camera']
            assert isinstance(cam_grp, h5py.Group)
            ds = cam_grp['rgb']
            assert isinstance(ds, h5py.Dataset)
            rgb: np.ndarray = ds[start]            # (H, W, 3) uint8
            # (H, W, 3) → (3, H, W), float [0, 1]
            obs = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        return chunk, robot_id, obs


# ---------------------------------------------------------------------------
# DiT factory helper
# ---------------------------------------------------------------------------

def make_dit_dataloaders(
    panda_h5:    Optional[str],
    xarm6_h5:    Optional[str],
    obs_mode:    str   = "none",
    k_steps:     int   = 16,
    stride:      int   = 8,
    batch_size:  int   = 256,
    val_split:   float = 0.05,
    num_workers: int   = 4,
    normalize:   bool  = True,
    seed:        int   = 42,
) -> tuple[DataLoader, DataLoader, DiTDataset]:
    """
    Returns (train_loader, val_loader, dataset) for DiT training.
    obs_mode controls what observation tensor is returned alongside each chunk.
    """
    h5_paths = []
    if panda_h5:
        h5_paths.append((panda_h5, "panda"))
    if xarm6_h5:
        h5_paths.append((xarm6_h5, "xarm6"))

    dataset = DiTDataset(
        h5_paths,
        k_steps=k_steps,
        stride=stride,
        normalize=normalize,
        obs_mode=obs_mode,
    )

    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[make_dit_dataloaders] obs_mode={obs_mode}  train={n_train:,}  val={n_val:,}  batch={batch_size}")
    return train_loader, val_loader, dataset


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    panda_h5 = sys.argv[1] if len(sys.argv) > 1 else None
    xarm6_h5 = sys.argv[2] if len(sys.argv) > 2 else None

    if panda_h5 is None:
        print("Usage: python dataloader.py <panda.h5> [xarm6.h5]")
        sys.exit(1)

    train_loader, val_loader, ds = make_dataloaders(
        panda_h5=panda_h5,
        xarm6_h5=xarm6_h5,
        k_steps=16,
        stride=8,
        batch_size=32,
        num_workers=0,
    )

    chunk, robot_id = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  chunk shape : {chunk.shape}")        # (32, 7, 16)
    print(f"  robot_ids   : {robot_id.tolist()}")
    print(f"  chunk min/max: {chunk.min():.4f} / {chunk.max():.4f}")
