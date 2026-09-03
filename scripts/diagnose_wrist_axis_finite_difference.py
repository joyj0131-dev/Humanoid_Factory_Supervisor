"""Finite-difference FK audit of each wrist joint's effect on table
clearance vs side-grasp topology (Section 8 of this session's mandate).
Never assumes "wrist_pitch" by name -- perturbs each of the 3 wrist
joints (roll/pitch/yaw, both sides) by +-0.01rad from the ACTUAL live
qpos at two representative points (the ARM_LATERAL_CLEARANCE peak-
contact tick, and a FOREARM_SIDE_DESCEND waypoint), and measures via FK
on a scratch MjData (never touching the live rollout):

  - table clearance of the lowest-touching fingertip geom (thumb_DP for
    clearance; nonthumb *_DP centroid for side_descend)
  - palm inward angle (_object_facing_angle_deg)
  - finger-down angle (_finger_down_angle_deg)
  - fingertip centroid height
  - distance to the object's assigned side face
  - min torso<->wrist body distance (approx, body origin distance)
  - joint-limit margin at the perturbed qpos

Read-only: never modifies SharpaBimanualGraspExpert/SharpaGraspEnv.

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/diagnose_wrist_axis_finite_difference.py
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
    SIDES,
    BimanualGraspState,
    SharpaBimanualGraspExpert,
    _finger_down_angle_deg,
    _object_facing_angle_deg,
)

WRIST_SUFFIXES = ("wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint")
DELTA = 0.01  # rad, ~0.57deg


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=3000, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def table_top_z(env: SharpaGraspEnv) -> float:
    return env.config.table_pos[2] + env.config.table_half_size[2]


def measure(env: SharpaGraspEnv, scratch: mujoco.MjData, side: str, fingertip_ref: str) -> dict:
    obj_pos = env.data.qpos[env._object_qpos_adr: env._object_qpos_adr + 3]
    mujoco.mj_forward(env.model, scratch)
    palm_pos, palm_R = scratch.site_xpos[env._left_palm_site if side == "left" else env._right_palm_site].copy(), \
        scratch.site_xmat[env._left_palm_site if side == "left" else env._right_palm_site].reshape(3, 3).copy()
    if fingertip_ref == "thumb":
        tip_id = env._fingertip_site[(side, "thumb")]
        tip_pos = scratch.site_xpos[tip_id].copy()
    else:
        tips = [scratch.site_xpos[env._fingertip_site[(side, f)]].copy() for f in ("index", "middle", "ring", "pinky")]
        tip_pos = np.mean(tips, axis=0)
    inward = _object_facing_angle_deg(side, palm_R, palm_pos, obj_pos)
    finger_down = _finger_down_angle_deg(palm_R)
    half = env.config.object_half_size
    side_face_y = half if side == "left" else -half
    dist_to_side_face = abs(tip_pos[1] - side_face_y)
    tb = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    wb = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
    torso_wrist_dist = float(np.linalg.norm(scratch.xpos[tb] - scratch.xpos[wb]))
    return {
        "table_clearance_m": float(tip_pos[2] - table_top_z(env)),
        "inward_angle_deg": inward, "finger_down_deg": finger_down,
        "tip_height_m": float(tip_pos[2]), "dist_to_side_face_m": dist_to_side_face,
        "torso_wrist_dist_m": torso_wrist_dist,
    }


def run_fd(env: SharpaGraspEnv, label: str, fingertip_ref: str) -> None:
    print(f"\n===== {label} (fingertip_ref={fingertip_ref}) =====")
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = env.data.qpos
    base = {s: measure(env, scratch, s, fingertip_ref) for s in SIDES}
    for s in SIDES:
        print(f"  BASE {s}: {base[s]}")

    for s in SIDES:
        for suffix in WRIST_SUFFIXES:
            jname = f"{s}_{suffix}"
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qadr = env.model.jnt_qposadr[jid]
            lo, hi = env.model.jnt_range[jid]
            for sign in (+1, -1):
                scratch.qpos[:] = env.data.qpos
                scratch.qpos[qadr] += sign * DELTA
                m = measure(env, scratch, s, fingertip_ref)
                d_clear = m["table_clearance_m"] - base[s]["table_clearance_m"]
                d_inward = m["inward_angle_deg"] - base[s]["inward_angle_deg"]
                d_down = m["finger_down_deg"] - base[s]["finger_down_deg"]
                d_side = m["dist_to_side_face_m"] - base[s]["dist_to_side_face_m"]
                margin = min(scratch.qpos[qadr] - lo, hi - scratch.qpos[qadr])
                print(f"    {jname} {'+':>1s}{sign*DELTA:+.3f}rad: "
                      f"d_clearance={d_clear*1000:+7.3f}mm  d_inward_angle={d_inward:+6.3f}deg  "
                      f"d_finger_down={d_down:+6.3f}deg  d_dist_side_face={d_side*1000:+7.3f}mm  "
                      f"joint_margin={margin:.4f}rad")


def main() -> None:
    # Point 1: ARM_LATERAL_CLEARANCE at its measured peak-contact tick (state_step=68).
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    for _ in range(700):
        if expert.state == BimanualGraspState.ARM_LATERAL_CLEARANCE and expert._state_step == 68:
            break
        if expert.state not in (BimanualGraspState.STABLE_START, BimanualGraspState.ARM_LATERAL_CLEARANCE):
            break
        action = expert.step()
        env.step(action)
    print(f"reached state={expert.state.name} state_step={expert._state_step}")
    run_fd(env, "ARM_LATERAL_CLEARANCE @ state_step=68 (peak thumb-table contact tick)", "thumb")

    # Point 2: FOREARM_SIDE_DESCEND, mid-trajectory (persistent contact throughout).
    env2 = make_env()
    expert2 = SharpaBimanualGraspExpert(env2)
    env2.reset(seed=0)
    for _ in range(1100):
        if expert2.state == BimanualGraspState.FOREARM_SIDE_DESCEND and expert2._state_step >= 60:
            break
        if expert2.state == BimanualGraspState.FAILURE:
            break
        action = expert2.step()
        env2.step(action)
    print(f"\nreached state={expert2.state.name} state_step={expert2._state_step} reason={expert2.failure_reason}")
    if expert2.state == BimanualGraspState.FOREARM_SIDE_DESCEND:
        run_fd(env2, "FOREARM_SIDE_DESCEND @ state_step>=60 (persistent contact)", "nonthumb")
    else:
        print("  FOREARM_SIDE_DESCEND not reached alive (failed earlier) -- skipping this point")


if __name__ == "__main__":
    main()
