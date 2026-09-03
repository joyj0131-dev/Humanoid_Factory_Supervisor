"""Feasibility map for Sharpa preshape curl vs table clearance vs remaining
inward closure travel, measured at the FOREARM_SIDE_DESCEND target pose
(where the current curl=0.95 workaround lives) under the CURRENT, verified
Functional Orientation (_object_facing_R weights 0.68/0.32, unchanged this
session).

State-preserving: never permanently perturbs the rollout. Read-only.

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/audit_sharpa_curl_table_feasibility.py
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
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv, ACTION_DIM, N_GROUPS_PER_HAND
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SIDES, BimanualGraspState, SharpaBimanualGraspExpert

SYN_VALUES = (0.0, 0.2, 0.35, 0.5, 0.7, 0.95)
NONTHUMB = ("index", "middle", "ring", "pinky")


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=3000, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def set_synergy(env: SharpaGraspEnv, side: str, syn: float) -> None:
    """Kinematic (quasi-static) set: writes ctrl AND directly writes qpos to
    the joint target (bypassing actuator dynamics) so a single mj_forward
    gives a valid FK/contact-geometry reading without needing to step the
    simulator to convergence. Used for the static feasibility screen only
    -- final candidates are re-verified via real env.step() physics."""
    side_idx = 0 if side == "left" else 1
    for g in (0, 1, 2, 3):
        env._group_synergy[side_idx * N_GROUPS_PER_HAND + g] = syn
        open_t, close_t = env._group_open[side][g], env._group_close[side][g]
        target = open_t + syn * (close_t - open_t)
        env.data.ctrl[env._group_act_ids[side][g]] = target
        env.data.qpos[env._group_qpos_adr[side][g]] = target
        env.data.qvel[env._group_dof_adr[side][g]] = 0.0
    mujoco.mj_forward(env.model, env.data)


def joint_target_at(env: SharpaGraspEnv, side: str, group_idx: int, syn: float) -> np.ndarray:
    open_t, close_t = env._group_open[side][group_idx], env._group_close[side][group_idx]
    return open_t + syn * (close_t - open_t)


def min_table_clearance(env: SharpaGraspEnv, side: str) -> float:
    """Signed min distance (m) from any of this side's finger geoms to the
    table geom -- approximated via fingertip Z minus table top Z (table top
    assumed at object_pos z - half - table thickness is unknown here, so we
    use actual contact force as the primary safety signal and fingertip Z
    as a secondary geometric one)."""
    tips_z = [env.fingertip_pos(side, f)[2] for f in sc.FINGERS]
    return float(min(tips_z))


def remaining_travel(env: SharpaGraspEnv, side: str, syn_from: float, obj_pos: np.ndarray) -> dict:
    saved = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env._group_synergy.copy())
    set_synergy(env, side, syn_from)
    tips0 = {f: env.fingertip_pos(side, f).copy() for f in NONTHUMB}
    joints0 = {g: joint_target_at(env, side, g, syn_from) for g in (1, 2, 3)}
    set_synergy(env, side, 1.0)
    tips1 = {f: env.fingertip_pos(side, f).copy() for f in NONTHUMB}
    joints1 = {g: joint_target_at(env, side, g, 1.0) for g in (1, 2, 3)}
    env.data.qpos[:], env.data.qvel[:], env.data.ctrl[:] = saved[0], saved[1], saved[2]
    env._group_synergy[:] = saved[3]
    mujoco.mj_forward(env.model, env.data)
    disp = {f: float(np.linalg.norm(tips1[f] - tips0[f])) for f in NONTHUMB}
    obj_comp = {}
    for f in NONTHUMB:
        d = tips1[f] - tips0[f]
        to_obj = obj_pos - tips0[f]
        to_obj = to_obj / (np.linalg.norm(to_obj) + 1e-12)
        obj_comp[f] = float(np.dot(d, to_obj))
    joint_travel = {g: float(np.linalg.norm(joints1[g] - joints0[g])) for g in (1, 2, 3)}
    return {"fingertip_mm": {f: disp[f] * 1000 for f in NONTHUMB},
            "obj_component_mm": {f: obj_comp[f] * 1000 for f in NONTHUMB},
            "joint_travel_rad": joint_travel}


def thumb_aperture(env: SharpaGraspEnv, side: str) -> float:
    thumb_tip = env.fingertip_pos(side, "thumb")
    centroid = np.mean([env.fingertip_pos(side, f) for f in NONTHUMB], axis=0)
    return float(np.linalg.norm(centroid - thumb_tip)) * 1000


def main() -> None:
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    print("Driving to FOREARM_SIDE_DESCEND entry (pre-curl-ramp target pose)...")
    reached = False
    for _ in range(2500):
        if expert.state == BimanualGraspState.FOREARM_SIDE_DESCEND and expert._side_descend_waypoint >= expert.config.side_descend_waypoints:
            reached = True
            break
        if expert.state in (BimanualGraspState.FAILURE, BimanualGraspState.FINGERTIP_PRECONTACT):
            reached = expert.state == BimanualGraspState.FINGERTIP_PRECONTACT
            break
        action = expert.step()
        env.step(action)
    print(f"reached_target_pose={reached} state={expert.state.name} step={expert._total_step}")
    obj_pos = expert._object_pos()
    print(f"object_pos={obj_pos.round(4)}\n")

    for side in SIDES:
        print(f"{'='*25} SIDE={side} {'='*25}")
        for syn in SYN_VALUES:
            saved = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env._group_synergy.copy())
            set_synergy(env, side, syn)
            tip_z_min = min_table_clearance(env, side)
            palm_pos, palm_R = env.palm_pose(side)
            side_face_dist = float(abs(palm_pos[1] - obj_pos[1]) - env.config.object_half_size)
            aperture = thumb_aperture(env, side)
            table_force = env._hand_table_contact_force()
            torso_force = env._torso_arm_collision_force()
            env.data.qpos[:], env.data.qvel[:], env.data.ctrl[:] = saved[0], saved[1], saved[2]
            env._group_synergy[:] = saved[3]
            mujoco.mj_forward(env.model, env.data)

            travel = remaining_travel(env, side, syn, obj_pos)
            mean_fingertip_mm = float(np.mean(list(travel["fingertip_mm"].values())))
            mean_obj_comp_mm = float(np.mean(list(travel["obj_component_mm"].values())))
            mean_joint_rad = float(np.mean(list(travel["joint_travel_rad"].values())))

            print(f"  syn={syn:.2f}  fingertip_min_z={tip_z_min:.4f}  side_face_dist={side_face_dist:.4f}  "
                  f"table_force={table_force:.2f}N torso_force={torso_force:.2f}N "
                  f"thumb_aperture_mm={aperture:.1f}  remaining->1.0: fingertip_mm={mean_fingertip_mm:.2f} "
                  f"obj_comp_mm={mean_obj_comp_mm:.2f} joint_rad={mean_joint_rad:.3f}")
        print()

    # Palm height candidates: static check only (current, +10/+20/+30mm), report fingertip Z at syn=0.35
    print(f"{'='*25} PALM HEIGHT CANDIDATES (static, syn=0.35) {'='*25}")
    for dh_mm in (0, 10, 20, 30):
        for side in SIDES:
            saved = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env._group_synergy.copy())
            set_synergy(env, side, 0.35)
            tip_z_min = min_table_clearance(env, side) + dh_mm / 1000.0
            env.data.qpos[:], env.data.qvel[:], env.data.ctrl[:] = saved[0], saved[1], saved[2]
            env._group_synergy[:] = saved[3]
            mujoco.mj_forward(env.model, env.data)
            print(f"  dh=+{dh_mm}mm side={side} projected_fingertip_min_z={tip_z_min:.4f} (table top ~ obj_z-half={obj_pos[2]-env.config.object_half_size:.4f})")


if __name__ == "__main__":
    main()
