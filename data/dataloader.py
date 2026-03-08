"""
ActionChunkDataset: loads 7-DOF delta end-effector actions from ManiSkill h5 files,
chunks them into fixed-length windows of size K, and labels each chunk with a robot ID.

Both Panda and xArm6 use pd_ee_delta_pose -> actions are always (T, 7), robot-agnostic.
The robot_id label (int) is what allows the downstream DiT to condition on embodiment.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


# Robot ID registry 
ROBOT_IDS = {"panda": 0, "xarm6": 1}


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
                 stride: int = 8,
                 normalize: bool = True):
        self.k_steps   = k_steps
        self.stride    = stride
        self.normalize = normalize

        # index: list of (file_idx, episode_key, start_t)
        self._index: list[tuple[int, str, int]] = []
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
                    for start in range(0, T - k_steps + 1, stride):
                        self._index.append((file_idx, ep_key, start))

        print(f"[ActionChunkDataset] {len(self._index):,} chunks from {len(self._files)} file(s)")
        for path, rid in self._files:
            print(f"  robot_id={rid}  {path}")

        # Compute normalisation stats from a random subsample (max 50k chunks)
        if normalize:
            self._mean, self._std = self._compute_stats(max_samples=50_000)
        else:
            self._mean: Optional[torch.Tensor] = None
            self._std:  Optional[torch.Tensor] = None

        # Keep files open for fast repeated access
        self._h5_handles: list[h5py.File] = [
            h5py.File(path, 'r') for path, _ in self._files
        ]

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_idx, ep_key, start = self._index[idx]
        robot_id = self._files[file_idx][1]

        ep_grp = self._h5_handles[file_idx][ep_key]
        assert isinstance(ep_grp, h5py.Group)
        ds = ep_grp['actions']
        assert isinstance(ds, h5py.Dataset)
        actions: np.ndarray = ds[start : start + self.k_steps]   # (K, 7)

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
            file_idx, ep_key, start = self._index[idx]
            path = self._files[file_idx][0]
            with h5py.File(path, 'r') as f:
                ep_grp = f[ep_key]
                assert isinstance(ep_grp, h5py.Group)
                ds = ep_grp['actions']
                assert isinstance(ds, h5py.Dataset)
                a: np.ndarray = ds[start : start + self.k_steps]
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
    stride:    int   = 8,
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
