"""Dataset-level bookkeeping: index.csv rows and episode reload.

Loading for Behavior Cloning training itself (batching, train/val split) is
Phase 3 scope -- this module only provides what Phase 2 needs to prove the
dataset it wrote back is readable and correct (see PROJECT_CONTEXT.md,
Dataset Reload Test).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

INDEX_FIELDS = [
    "episode_id",
    "file",
    "success",
    "length",
    "seed",
    "object_x",
    "object_y",
    "object_z",
    "final_left_error",
    "final_right_error",
]


def append_index_row(index_path: str | Path, row: dict) -> None:
    index_path = Path(index_path)
    write_header = not index_path.exists()
    with open(index_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in INDEX_FIELDS})


def load_episode(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def next_episode_id(output_dir: str | Path) -> int:
    """Scans output_dir for existing episode_NNNNNN.npz files and returns the
    next free id, so re-running collect_demos.py appends instead of
    overwriting a dataset already on disk."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 1
    existing = sorted(output_dir.glob("episode_*.npz"))
    if not existing:
        return 1
    ids = [int(p.stem.split("_")[1]) for p in existing]
    return max(ids) + 1
