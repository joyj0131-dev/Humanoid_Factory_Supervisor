"""Read-only FK measurement of the Sharpa Wave hand's ulnar (pinky-side)
local axis, used to define the Horizontal-Wrap Posture Gate's ulnar-
edge-down constraint (Constraint C) from a MEASURED axis instead of a
guessed one -- same "never guess joint/body geometry" rule as
scripts/measure_sharpa_side_grasp_axes.py, extended with the thumb-vs-
pinky spread across the palm. Never modifies the env/expert/model.

Method: for each finger, take its root/base body position in WORLD
frame, express it in the palm's OWN local frame via palm_R.T @
(world_pos - palm_pos) -- a rigid-body-invariant quantity (any vector
attached to the hand transforms as world = R @ local). Measured at BOTH
the stand pose and FOREARM_FORWARD_REACH's end pose to confirm the
result is a structural property of the mesh, not an artifact of
whichever pose the arm happens to be holding.

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/measure_sharpa_ulnar_axis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    BimanualGraspState, SharpaBimanualGraspExpert, SIDES,
)

HAND_BODIES = {
    "left": {
        "thumb_base": "left_left_thumb_CMC_VL", "index_base": "left_left_index_MCP_VL",
        "middle_base": "left_left_middle_MCP_VL", "ring_base": "left_left_ring_MCP_VL",
        "pinky_base": "left_left_pinky_MC",
    },
    "right": {
        "thumb_base": "right_right_thumb_CMC_VL", "index_base": "right_right_index_MCP_VL",
        "middle_base": "right_right_middle_MCP_VL", "ring_base": "right_right_ring_MCP_VL",
        "pinky_base": "right_right_pinky_MC",
    },
}


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=2500, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def bpos(env: SharpaGraspEnv, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"body not found in compiled model: {name}"
    return env.data.xpos[bid].copy()


def measure(env: SharpaGraspEnv, label: str) -> None:
    print(f"\n===== {label} =====")
    approach = np.array([1.0, 0.0, 0.0])
    closing = np.array([-0.145, 0.989, 0.0002])
    closing = closing / np.linalg.norm(closing)
    cross_axis = np.cross(approach, closing)
    cross_axis = cross_axis / np.linalg.norm(cross_axis)
    for side in SIDES:
        palm_pos, palm_R = env.palm_pose(side)
        hb = HAND_BODIES[side]
        locals_ = {name: palm_R.T @ (bpos(env, bn) - palm_pos) for name, bn in hb.items()}
        print(f"[{side}] palm_pos={palm_pos.round(4)}")
        for f in ("thumb_base", "index_base", "middle_base", "ring_base", "pinky_base"):
            print(f"       local_{f:12s}={locals_[f].round(4)}  z={locals_[f][2]:+.4f}")
        ulnar_raw = locals_["pinky_base"] - locals_["thumb_base"]
        ulnar_n = ulnar_raw / np.linalg.norm(ulnar_raw)
        sign = 1.0 if np.dot(ulnar_raw, cross_axis) > 0 else -1.0
        print(f"       raw pinky-thumb axis (normalized) = {ulnar_n.round(4)}")
        print(f"       cross(approach,closing) candidate  = {cross_axis.round(4)}  ULNAR_SIGN={sign:+.0f}")


def main() -> None:
    env = make_env()
    env.reset(seed=0)
    measure(env, "STAND POSE (reset)")

    expert = SharpaBimanualGraspExpert(env)
    for _ in range(700):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE,
                             BimanualGraspState.WRIST_SIDE_GRASP_ALIGN):
            break
        action = expert.step()
        env.step(action)
    measure(env, f"AFTER FOREARM_FORWARD_REACH (state={expert.state.name}, far from table)")
    print("\nExpect: IDENTICAL local vectors/signs between the two poses above (rigid, "
          "pose-independent property) and per-finger z ordered thumb<index<middle<ring<pinky.")


if __name__ == "__main__":
    main()
