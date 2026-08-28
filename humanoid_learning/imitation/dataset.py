"""Episode loading and episode-level train/validation split for Behavior
Cloning, built on top of ``humanoid_learning.data.demo_dataset`` (which owns
raw NPZ/index.csv I/O -- see PROJECT_CONTEXT.md, Demonstration Dataset).

Split is at the episode level and seeded, so no timestep from a validation
episode ever appears in the training set (and vice versa) -- see
PROJECT_CONTEXT.md, Dataset Loader.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from humanoid_learning.data import demo_dataset


@dataclass
class Episode:
    episode_id: int
    observations: np.ndarray  # (T, obs_dim) float32
    actions: np.ndarray  # (T, action_dim) float32


def load_episodes(datasets_dir: str | Path, index_file: str = "index.csv", only_success: bool = True) -> list[Episode]:
    datasets_dir = Path(datasets_dir)
    index_path = datasets_dir / index_file
    with open(index_path, newline="") as f:
        rows = list(csv.DictReader(f))

    episodes: list[Episode] = []
    for row in rows:
        success = row["success"] == "True"
        if only_success and not success:
            continue
        data = demo_dataset.load_episode(datasets_dir / row["file"])
        obs = np.asarray(data["observations"], dtype=np.float32)
        act = np.asarray(data["actions"], dtype=np.float32)
        assert obs.ndim == 2 and act.ndim == 2, f"episode {row['episode_id']}: expected 2D arrays"
        assert obs.shape[0] == act.shape[0], f"episode {row['episode_id']}: observation/action length mismatch"
        assert np.isfinite(obs).all(), f"episode {row['episode_id']}: non-finite observation"
        assert np.isfinite(act).all(), f"episode {row['episode_id']}: non-finite action"
        episodes.append(Episode(episode_id=int(row["episode_id"]), observations=obs, actions=act))

    if not episodes:
        raise ValueError(f"no episodes loaded from {index_path} (only_success={only_success})")
    return episodes


def split_episodes(episodes: list[Episode], val_fraction: float, seed: int) -> tuple[list[Episode], list[Episode]]:
    """Shuffles episode order deterministically (seeded) and splits by
    episode id so an episode's timesteps land entirely in train or entirely
    in validation, never both."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    n = len(episodes)
    n_val = max(1, round(n * val_fraction))
    n_val = min(n_val, n - 1)  # always keep at least one training episode

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    val_idx = set(order[:n_val].tolist())

    train_episodes = [ep for i, ep in enumerate(episodes) if i not in val_idx]
    val_episodes = [ep for i, ep in enumerate(episodes) if i in val_idx]

    train_ids = {ep.episode_id for ep in train_episodes}
    val_ids = {ep.episode_id for ep in val_episodes}
    assert train_ids.isdisjoint(val_ids), "train/validation episode leakage detected"

    return train_episodes, val_episodes


class TimestepDataset(Dataset):
    """Flattens a list of episodes into (observation, action) timestep pairs.

    Observations are normalized here (with a Normalizer already fit on the
    training split); actions are returned as-is -- see policy.py docstring
    for why no separate action normalization is applied.
    """

    def __init__(self, episodes: list[Episode], obs_normalizer):
        obs_list = [ep.observations for ep in episodes]
        act_list = [ep.actions for ep in episodes]
        obs = np.concatenate(obs_list, axis=0)
        act = np.concatenate(act_list, axis=0)
        self.observations = obs_normalizer.normalize(obs)
        self.actions = act.astype(np.float32)

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.observations[idx]),
            torch.from_numpy(self.actions[idx]),
        )


def count_samples(episodes: list[Episode]) -> int:
    return sum(ep.observations.shape[0] for ep in episodes)
