"""Open the Phase 1 scene in MuJoCo's interactive viewer for a visual check.

This builds the exact same model BimanualReachEnv uses (fixed-base G1 +
table + randomized object, via model_builder.build_model), not a separate
copy, so what you see here matches what Expert/BC/PPO will actually run on.

By default the viewer just holds the reset pose -- nothing drives the arms
on its own. Pass --wiggle for a harmless sine-wave test action (proves
actuation works, not intelligent). Pass --expert to watch the actual
Phase 2 ScriptedExpert (damped least squares IK) drive both arms to the
pre-grasp targets, resetting to a new object position on every
success/timeout.

Run locally (needs a real display, not a headless/SSH session without X):
    python scripts/view_scene.py
    python scripts/view_scene.py --seed 7
    python scripts/view_scene.py --wiggle
    python scripts/view_scene.py --expert
    python scripts/view_scene.py --expert --seed 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np

from humanoid_learning.envs.humanoid_reach_env import BimanualReachEnv
from humanoid_learning.envs.task_config import EnvConfig
from humanoid_learning.expert.scripted_expert import ExpertConfig, ScriptedExpert

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "environment.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="object randomization seed")
    parser.add_argument(
        "--wiggle",
        action="store_true",
        help="drive the arms with a slow sine-wave action to visually confirm actuation",
    )
    parser.add_argument(
        "--expert",
        action="store_true",
        help="drive the arms with the Phase 2 ScriptedExpert (DLS IK) toward the pre-grasp targets",
    )
    args = parser.parse_args()

    config = EnvConfig.from_yaml(CONFIG_PATH)
    env = BimanualReachEnv(config)
    obs, info = env.reset(seed=args.seed)

    print(f"object position: {info['object_position']}")
    print(f"left target:  {env._left_target}")
    print(f"right target: {env._right_target}")
    print("Close the viewer window to exit.")

    if not args.wiggle and not args.expert:
        mujoco.viewer.launch(env.model, env.data)
        return

    expert = ScriptedExpert(env, ExpertConfig()) if args.expert else None
    episode_seed = args.seed

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        t = 0.0
        while viewer.is_running():
            step_start = time.time()
            if expert is not None:
                action = expert.act()
            else:
                action = np.full(14, np.sin(t), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if expert is not None:
                    status = "SUCCESS" if terminated else "TIMEOUT"
                    print(
                        f"[{status}] left_err={info['left_ee_error']:.3f} "
                        f"right_err={info['right_ee_error']:.3f} -- resetting to a new object position"
                    )
                    episode_seed += 1
                obs, info = env.reset(seed=episode_seed)
            viewer.sync()
            t += 0.05
            elapsed = time.time() - step_start
            time.sleep(max(0.0, 1 / 30 - elapsed))


if __name__ == "__main__":
    main()
