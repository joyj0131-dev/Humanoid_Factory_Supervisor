"""Phase 3 slim evaluation: run a BC checkpoint in BimanualReachEnv under
Seen / Unseen / Noise conditions (PROJECT_CONTEXT.md, Phase 3).

EE error is reported as a Foundation debugging metric only, not the
project's final metric -- see PROJECT_CONTEXT.md, Evaluation.

Example:
    python scripts/evaluate_bc.py
    python scripts/evaluate_bc.py --checkpoint checkpoints/bc/last.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from humanoid_learning.envs.task_config import EnvConfig
from humanoid_learning.evaluation import config as eval_config_module
from humanoid_learning.evaluation import evaluator
from humanoid_learning.evaluation.metrics import format_success_rate
from humanoid_learning.imitation import policy as bc_policy

ENV_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "environment.yaml"
EVAL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "evaluation.yaml"


def run_condition(name: str, base_env_config: EnvConfig, cond, seeds: list[int], policy) -> list:
    env = evaluator.make_env_for_condition(base_env_config, cond.object_x_range, cond.object_y_range)
    results = evaluator.run_evaluation(
        env, policy, seeds, obs_noise_std=cond.obs_noise_std, noise_seed=cond.noise_seed
    )

    n_success = sum(r.success for r in results)
    left_errors = [r.left_ee_error for r in results]
    right_errors = [r.right_ee_error for r in results]
    lengths = [r.length for r in results]
    failure_reasons: dict[str, int] = {}
    for r in results:
        if r.failure_reason:
            failure_reasons[r.failure_reason] = failure_reasons.get(r.failure_reason, 0) + 1

    print(f"\n[{name}]")
    print(f"  Success rate:        {format_success_rate(n_success, len(results))}")
    print(f"  Mean left EE error:  {np.mean(left_errors):.4f}")
    print(f"  Mean right EE error: {np.mean(right_errors):.4f}")
    print(f"  Mean episode length: {np.mean(lengths):.1f}")
    if failure_reasons:
        print(f"  Failure reasons:     {failure_reasons}")

    return results


def save_results_csv(path: Path, condition_results: dict[str, list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "seed",
                "success",
                "length",
                "left_ee_error",
                "right_ee_error",
                "mean_ee_error",
                "failure_reason",
            ],
        )
        writer.writeheader()
        for condition, results in condition_results.items():
            for r in results:
                writer.writerow(
                    {
                        "condition": condition,
                        "seed": r.seed,
                        "success": r.success,
                        "length": r.length,
                        "left_ee_error": r.left_ee_error,
                        "right_ee_error": r.right_ee_error,
                        "mean_ee_error": r.mean_ee_error,
                        "failure_reason": r.failure_reason or "",
                    }
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bc/best.pt")
    parser.add_argument("--env-config", type=str, default=str(ENV_CONFIG_PATH))
    parser.add_argument("--eval-config", type=str, default=str(EVAL_CONFIG_PATH))
    parser.add_argument("--results-dir", type=str, default="results/bc")
    args = parser.parse_args()

    base_env_config = EnvConfig.from_yaml(args.env_config)
    eval_cfg = eval_config_module.EvalConfig.from_yaml(args.eval_config)
    policy = bc_policy.load_checkpoint(args.checkpoint)
    seeds = eval_cfg.seeds()

    condition_results = {
        "seen": run_condition("Seen", base_env_config, eval_cfg.seen, seeds, policy),
        "unseen": run_condition("Unseen", base_env_config, eval_cfg.unseen, seeds, policy),
        "noise": run_condition("Noise", base_env_config, eval_cfg.noise, seeds, policy),
    }

    csv_path = Path(args.results_dir) / "eval_results.csv"
    save_results_csv(csv_path, condition_results)
    print(f"\nSaved evaluation results: {csv_path}")


if __name__ == "__main__":
    main()
