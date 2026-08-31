"""Tripod Closure Planner session, Stage 2/3: plans individual thumb
fingertip targets (index/middle kept at their currently-achieved
positions) from the canonical THUMB_OPPOSE-entry pose, solves a small
3-DOF thumb-only IK for each hand, and reports static feasibility
(reachability, joint-limit margin, resulting thumb-thumb separation).
Run with:

    python scripts/plan_tripod_closure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import mujoco

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.grasp_env import FixedBaseGraspEnv
from humanoid_learning.expert.grasp_expert import BimanualSidePinchExpert, GraspExpertConfig, GraspState
from humanoid_learning.expert.tripod_closure_planner import (
    audit_hand_geometry, plan_tripod_targets, solve_finger_ik, true_fingertip_world,
)


def make_env():
    return FixedBaseGraspEnv(GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF))


env = make_env()
env.reset(seed=0)
expert = BimanualSidePinchExpert(env, GraspExpertConfig())
outcome = None
for i in range(1200):
    outcome = expert.step()
    if outcome.state in (GraspState.THUMB_OPPOSE, GraspState.FAILURE, GraspState.SUCCESS):
        break
print(f"reached state={outcome.state.name} at step {i}")
assert outcome.state == GraspState.THUMB_OPPOSE, "planner needs a real THUMB_OPPOSE-entry pose to plan from"

audit = audit_hand_geometry(env, expert)
model = env.model
scratch = mujoco.MjData(model)
scratch.qpos[:] = env.data.qpos
mujoco.mj_forward(model, scratch)

results = {}
for side in ("left", "right"):
    targets = plan_tripod_targets(env, audit, side)
    finger_qpos_adr = env._left_finger_qpos_adr if side == "left" else env._right_finger_qpos_adr
    tip_site = {"left": "left_thumb_tip", "right": "right_thumb_tip"}[side]
    base_qpos = np.array(audit["sides"][side]["current_finger_qpos"])
    thumb_target = targets["thumb"]

    solved_q, err_norm, within_limits = solve_finger_ik(
        model, scratch, side, "thumb", tip_site, thumb_target.world_pos,
        finger_qpos_adr, base_qpos, slots=[0, 1, 2],
    )
    results[side] = dict(
        target_world=thumb_target.world_pos, target_local=thumb_target.object_local, face=thumb_target.face,
        solved_q=solved_q, err_norm=err_norm, within_limits=within_limits,
        base_q=base_qpos,
    )
    print(f"\n--- {side.upper()} thumb ---")
    print(f"  current thumb qpos (thumb_0,1,2) = {np.round(base_qpos[:3], 4)}")
    print(f"  target world = {np.round(thumb_target.world_pos, 4)}  object-local = {np.round(thumb_target.object_local, 4)}  face={thumb_target.face}")
    print(f"  solved thumb qpos (thumb_0,1,2)  = {np.round(solved_q[:3], 4)}")
    print(f"  final position error = {err_norm*1000:.3f} mm   within joint limits: {within_limits}")

# resulting thumb-thumb distance if BOTH hands moved to their solved targets simultaneously
scratch.qpos[env._left_finger_qpos_adr] = results["left"]["solved_q"]
scratch.qpos[env._right_finger_qpos_adr] = results["right"]["solved_q"]
mujoco.mj_forward(model, scratch)
left_tip = true_fingertip_world(model, scratch, "left_thumb_tip")
right_tip = true_fingertip_world(model, scratch, "right_thumb_tip")
new_thumb_thumb_dist = float(np.linalg.norm(left_tip - right_tip))
print(f"\nresulting thumb-thumb distance at solved targets = {new_thumb_thumb_dist:.4f} m "
      f"(was {audit['thumb_thumb_distance_m']:.4f} m at THUMB_OPPOSE entry)")

# joint-limit margin report
for side in ("left", "right"):
    joint_names = [f"{side}_hand_thumb_{i}_joint" for i in range(3)]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    q = results[side]["solved_q"][:3]
    for jn, jid, qi in zip(joint_names, jids, q):
        lo, hi = model.jnt_range[jid]
        margin = min(qi - lo, hi - qi)
        print(f"  {jn}: q={qi:+.4f}  range=[{lo:+.4f},{hi:+.4f}]  margin={margin:.4f}")
