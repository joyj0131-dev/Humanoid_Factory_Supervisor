"""Open the Phase 1 scene in MuJoCo's interactive viewer for a visual check.

This builds the exact same model BimanualReachEnv uses (fixed-base G1 +
table + randomized object, via model_builder.build_model), not a separate
copy, so what you see here matches what Expert/BC/PPO will actually run on.

By default the viewer just holds the reset pose -- there is no Expert/BC/PPO
policy yet (that's Phase 2+), so nothing drives the arms on its own. Pass
--wiggle to feed a slow, harmless sine-wave action through env.step() every
frame, purely to visually confirm the 14-dim Joint Position Delta action
actually moves the arms (this is a verification aid, not a controller).

Run locally (needs a real display, not a headless/SSH session without X):
    python scripts/view_scene.py
    python scripts/view_scene.py --seed 7
    python scripts/view_scene.py --wiggle
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

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "environment.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="object randomization seed")
    parser.add_argument(
        "--wiggle",
        action="store_true",
        help="drive the arms with a slow sine-wave action to visually confirm actuation",
    )
    args = parser.parse_args()

    config = EnvConfig.from_yaml(CONFIG_PATH)
    env = BimanualReachEnv(config)
    obs, info = env.reset(seed=args.seed)

    print(f"object position: {info['object_position']}")
    print(f"left target:  {env._left_target}")
    print(f"right target: {env._right_target}")
    print("Close the viewer window to exit.")

    if not args.wiggle:
        mujoco.viewer.launch(env.model, env.data)
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        t = 0.0
        while viewer.is_running():
            step_start = time.time()
            action = np.full(14, np.sin(t), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset(seed=args.seed)
            viewer.sync()
            t += 0.05
            elapsed = time.time() - step_start
            time.sleep(max(0.0, 1 / 30 - elapsed))


if __name__ == "__main__":
    main()
