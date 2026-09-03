"""Read-only FK measurement of the Sharpa Wave hand's real palm/finger
axes, at the stand pose and at the (already-fixed) FOREARM_FORWARD_REACH
end pose. Never modifies the env/expert/model -- pure observation, used
to define the Side-Grasp Posture Gate's target geometry from MEASURED
axes instead of guessed quaternions (see this session's mandate, Section
5: "좌표계가 확정되기 전에 quaternion을 손으로 추측해 넣지 마라").

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/measure_sharpa_side_grasp_axes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    BimanualGraspState,
    SharpaBimanualGraspExpert,
)

SIDES = ("left", "right")
NONTHUMB = ("index", "middle", "ring", "pinky")


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=2500, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def finger_root_body(side: str, finger: str) -> str:
    # Proximal phalanx ("PP") is each finger's root segment nearest the
    # palm (thumb/pinky have an extra MC segment closer still, but PP is
    # present for all 5 and is the first segment whose axis meaningfully
    # points "into" the finger, not the palm/CMC swivel).
    return sc.sharpa_body(side, finger, "PP")


def measure(env: SharpaGraspEnv, label: str) -> dict:
    print(f"\n===== {label} =====")
    out: dict = {}
    for side in SIDES:
        palm_pos, palm_R = env.palm_pose(side)
        # palm-frame LOCAL axes expressed in WORLD frame (columns of palm_R):
        # per model_builder.py's documented convention, col0=approach(local +X),
        # col1=closing(local Y, sign mirrored per side), col2=lateral(local Z).
        approach_axis = palm_R[:, 0]
        closing_axis = palm_R[:, 1]
        lateral_axis = palm_R[:, 2]

        tips = {f: env.fingertip_pos(side, f) for f in sc.FINGERS}
        roots = {}
        for f in sc.FINGERS:
            bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, finger_root_body(side, f))
            roots[f] = env.data.xpos[bid].copy()

        finger_dirs = {f: (tips[f] - roots[f]) for f in sc.FINGERS}
        finger_dirs_n = {f: v / np.linalg.norm(v) for f, v in finger_dirs.items()}
        nonthumb_avg_dir = np.mean([finger_dirs_n[f] for f in NONTHUMB], axis=0)
        nonthumb_avg_dir /= np.linalg.norm(nonthumb_avg_dir)

        tip_centroid = np.mean([tips[f] for f in sc.FINGERS], axis=0)
        nonthumb_tip_centroid = np.mean([tips[f] for f in NONTHUMB], axis=0)

        thumb_to_nonthumb = nonthumb_tip_centroid - tips["thumb"]

        world_down = np.array([0.0, 0.0, -1.0])
        finger_down_angle_deg = float(np.degrees(np.arccos(
            np.clip(np.dot(nonthumb_avg_dir, world_down), -1.0, 1.0))))

        print(f"[{side}] palm_pos={palm_pos.round(4)}")
        print(f"        approach_axis(world)={approach_axis.round(4)}")
        print(f"        closing_axis(world) ={closing_axis.round(4)}")
        print(f"        lateral_axis(world) ={lateral_axis.round(4)}")
        print(f"        nonthumb finger-down axis(world)={nonthumb_avg_dir.round(4)}  "
              f"angle_vs_world_down={finger_down_angle_deg:.2f}deg")
        print(f"        tip_centroid(all5)={tip_centroid.round(4)}  nonthumb_tip_centroid={nonthumb_tip_centroid.round(4)}")
        print(f"        thumb_tip={tips['thumb'].round(4)}  thumb->nonthumb_centroid vec={thumb_to_nonthumb.round(4)}")
        for f in sc.FINGERS:
            print(f"        {f:6s}: root={roots[f].round(4)} tip={tips[f].round(4)} dir={finger_dirs_n[f].round(4)}")

        out[side] = {
            "palm_pos": palm_pos, "palm_R": palm_R,
            "approach_axis": approach_axis, "closing_axis": closing_axis, "lateral_axis": lateral_axis,
            "finger_down_axis": nonthumb_avg_dir, "finger_down_angle_deg": finger_down_angle_deg,
            "tip_centroid": tip_centroid, "nonthumb_tip_centroid": nonthumb_tip_centroid,
            "thumb_tip": tips["thumb"],
        }

    # mirror check
    l, r = out["left"], out["right"]
    print("\n[mirror check]")
    print(f"  approach_axis L vs R (expect same, not mirrored): {l['approach_axis'].round(4)} vs {r['approach_axis'].round(4)}")
    print(f"  closing_axis  L vs R (expect opposite sign, Y-mirror): {l['closing_axis'].round(4)} vs {r['closing_axis'].round(4)}")
    print(f"  finger_down_axis L vs R: {l['finger_down_axis'].round(4)} vs {r['finger_down_axis'].round(4)}")
    return out


def main() -> None:
    env = make_env()
    env.reset(seed=0)
    measure(env, "STAND POSE (reset)")

    expert = SharpaBimanualGraspExpert(env)
    for _ in range(600):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        action = expert.step()
        env.step(action)
        if expert.state == BimanualGraspState.FOREARM_DESCEND:
            break
    measure(env, f"AFTER FOREARM_FORWARD_REACH (state={expert.state.name})")

    obj_pos = expert._object_pos()
    half = env.config.object_half_size
    print(f"\n[object] center={obj_pos.round(4)} half_size={half} "
          f"top_z={obj_pos[2]+half:.4f} bottom_z={obj_pos[2]-half:.4f} "
          f"left_face_y={obj_pos[1]+half:.4f} right_face_y={obj_pos[1]-half:.4f}")


if __name__ == "__main__":
    main()
