"""Collect Expert demonstrations for BimanualReachEnv.

Two ways to bound a run -- pick one:
    --attempts N               run exactly N episodes, keep whatever succeeds
    --successful-episodes N    keep attempting until N episodes succeed
                                (bounded by --max-attempts as a safety cap)

Examples:
    python scripts/collect_demos.py --attempts 1 --seed 0
    python scripts/collect_demos.py --attempts 10 --seed 0
    python scripts/collect_demos.py --successful-episodes 50 --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from humanoid_learning.data import demo_dataset
from humanoid_learning.data.recorder import EpisodeRecorder
from humanoid_learning.envs.humanoid_reach_env import BimanualReachEnv
from humanoid_learning.envs.task_config import EnvConfig
from humanoid_learning.expert.scripted_expert import ExpertConfig, ScriptedExpert

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "environment.yaml"


def run_one_attempt(env: BimanualReachEnv, expert: ScriptedExpert, recorder: EpisodeRecorder, seed: int):
    recorder.reset()
    obs, reset_info = env.reset(seed=seed)
    # the object can get nudged by self/table contact during a rough
    # attempt, so the spawn position (what matters for characterizing the
    # training distribution later) must be captured here, not from the
    # final step's info.
    spawn_object_position = reset_info["object_position"]
    info = reset_info
    for _ in range(env.config.max_episode_steps):
        action = expert.act()
        recorder.record(obs, action)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info, spawn_object_position


def classify_failure(info: dict, truncated_without_success: bool) -> str | None:
    if info.get("unstable"):
        return "NUMERICAL_ERROR"
    if info["success"]:
        return None
    if truncated_without_success:
        return "MAX_STEPS"
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=None, help="run exactly this many episodes")
    parser.add_argument(
        "--successful-episodes", type=int, default=None, help="keep attempting until this many succeed"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="safety cap when using --successful-episodes (default: 20x the target)",
    )
    parser.add_argument("--seed", type=int, default=0, help="base seed; attempt i uses seed + i")
    parser.add_argument("--output-dir", type=str, default="datasets")
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--gain", type=float, default=0.5)
    args = parser.parse_args()

    if (args.attempts is None) == (args.successful_episodes is None):
        parser.error("pass exactly one of --attempts or --successful-episodes")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.csv"

    config = EnvConfig.from_yaml(CONFIG_PATH)
    env = BimanualReachEnv(config)
    expert = ScriptedExpert(env, ExpertConfig(damping=args.damping, gain=args.gain))
    recorder = EpisodeRecorder()

    episode_id = demo_dataset.next_episode_id(output_dir)

    target_successes = args.successful_episodes
    max_attempts = args.attempts
    if target_successes is not None and max_attempts is None:
        max_attempts = args.max_attempts or target_successes * 20

    attempts = 0
    successes = 0
    failure_counts: dict[str, int] = {}
    lengths, left_errors, right_errors = [], [], []

    start = time.time()
    while True:
        if target_successes is not None and successes >= target_successes:
            break
        if attempts >= max_attempts:
            break

        seed = args.seed + attempts
        info, spawn_object_position = run_one_attempt(env, expert, recorder, seed)
        attempts += 1
        truncated_without_success = len(recorder) >= env.config.max_episode_steps and not info["success"]
        reason = classify_failure(info, truncated_without_success)

        if reason is None:
            file_name = f"episode_{episode_id:06d}.npz"
            recorder.save(
                output_dir / file_name,
                success=True,
                object_position=spawn_object_position,
                left_target=env.left_target,
                right_target=env.right_target,
                final_left_error=info["left_ee_error"],
                final_right_error=info["right_ee_error"],
                seed=seed,
                episode_length=len(recorder),
                env_config_id=config.config_id(),
            )
            demo_dataset.append_index_row(
                index_path,
                {
                    "episode_id": episode_id,
                    "file": file_name,
                    "success": True,
                    "length": len(recorder),
                    "seed": seed,
                    "object_x": spawn_object_position[0],
                    "object_y": spawn_object_position[1],
                    "object_z": spawn_object_position[2],
                    "final_left_error": info["left_ee_error"],
                    "final_right_error": info["right_ee_error"],
                },
            )
            successes += 1
            episode_id += 1
        else:
            failure_counts[reason] = failure_counts.get(reason, 0) + 1

        lengths.append(len(recorder))
        left_errors.append(info["left_ee_error"])
        right_errors.append(info["right_ee_error"])

    elapsed = time.time() - start

    print(f"\nTotal attempts:        {attempts}")
    print(f"Successful episodes:   {successes}")
    print(f"Failed episodes:       {attempts - successes}")
    print(f"Success rate:          {successes / attempts:.1%}" if attempts else "Success rate:          n/a")
    for reason, count in sorted(failure_counts.items()):
        print(f"  {reason}: {count}")
    print(f"Mean episode length:   {np.mean(lengths):.1f}")
    print(f"Mean left EE error:    {np.mean(left_errors):.4f}")
    print(f"Mean right EE error:   {np.mean(right_errors):.4f}")
    print(f"Elapsed:               {elapsed:.1f}s")


if __name__ == "__main__":
    main()
