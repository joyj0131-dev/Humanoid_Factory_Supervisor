"""Phase 3: train the BC MLP policy on the Foundation reaching dataset.

Example:
    python scripts/train_bc.py
    python scripts/train_bc.py --epochs 50 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanoid_learning.imitation import dataset as bc_dataset
from humanoid_learning.imitation import policy as bc_policy
from humanoid_learning.imitation.trainer import BCConfig, train_bc

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "bc.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--dataset-dir", type=str, default="datasets")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/bc")
    parser.add_argument("--results-dir", type=str, default="results/bc")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None)
    args = parser.parse_args()

    config = BCConfig.from_yaml(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.val_fraction is not None:
        config.val_fraction = args.val_fraction
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.hidden_sizes is not None:
        config.hidden_sizes = tuple(args.hidden_sizes)

    episodes = bc_dataset.load_episodes(args.dataset_dir)
    result = train_bc(episodes, config)

    print(f"\nBest epoch: {result.best_epoch}  best_val_loss={result.best_val_loss:.6f}")

    checkpoint_dir = Path(args.checkpoint_dir)
    best_net = result.build_net(use_best=True)
    last_net = result.build_net(use_best=False)
    bc_policy.save_checkpoint(
        checkpoint_dir / "best.pt",
        best_net,
        result.obs_normalizer,
        result.hidden_sizes,
        bc_policy.OBS_DIM,
        bc_policy.ACTION_DIM,
        extra={"epoch": result.best_epoch, "val_loss": result.best_val_loss},
    )
    bc_policy.save_checkpoint(
        checkpoint_dir / "last.pt",
        last_net,
        result.obs_normalizer,
        result.hidden_sizes,
        bc_policy.OBS_DIM,
        bc_policy.ACTION_DIM,
        extra={"epoch": config.epochs - 1, "val_loss": result.history[-1]["val_loss"]},
    )
    print(f"Saved checkpoints: {checkpoint_dir}/best.pt, {checkpoint_dir}/last.pt")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(result.history)
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
