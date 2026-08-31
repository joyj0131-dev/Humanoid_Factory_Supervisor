"""Tests for the Tripod Closure Planner session
(humanoid_learning.expert.tripod_closure_planner). Run with:
    python scripts/test_tripod_closure_planner.py

REAL FINDING (this session): a collision-free coordinated six-fingertip
closure was investigated as a replacement for THUMB_OPPOSE's current
"one shared centroid target per hand" design. Stage 1 (fingertip
geometry audit) succeeded and produced a genuinely new fact (thumb_0 is
NOT a negligible lever at the TRUE fingertip -- a prior session's "2-5mm"
finding was measured at the uncorrected, always-static site). Stage 2
(individual per-finger contact targets) found the natural target -- the
thumb reaching its own hand's index/middle Y coordinate, so the two
hands' thumbs stop nearly colliding (measured 0.0088m apart at
THUMB_OPPOSE entry) -- is NOT kinematically reachable from the current
wrist pose: thumb_1 and thumb_2 pin at their hard joint limits with
~45-49mm of position error still remaining, and adding the wrist's own
roll/pitch/yaw as extra decision variables only reduces this to
~42-43mm (confirms a genuine reach limit, not a missing-DOF artifact).
A bisected partial target (max Y the thumb CAN reach) is achievable
individually, but Stage 3's swept-path check found that a naive
joint-space linear interpolation from the current pose to that partial
target passes through a WORSE thumb-thumb collision (0.6mm minimum,
below either endpoint) and an object-box penetration (7.2mm) along the
way. Per this session's own instruction ("정적 six-fingertip
feasibility가 없으면 dynamic controller를 수정하지 마라"), Stage 4
(live controller connection) was NOT attempted -- these tests lock in
the Stage 2/3 findings that stopped it there, not a working fix.
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
    audit_hand_geometry, plan_tripod_targets, solve_finger_ik, solve_hand_closure_ik, true_fingertip_world,
)


def make_env():
    return FixedBaseGraspEnv(GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF))


def _reach_thumb_oppose():
    env = make_env()
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    outcome = None
    for i in range(1200):
        outcome = expert.step()
        if outcome.state in (GraspState.THUMB_OPPOSE, GraspState.FAILURE, GraspState.SUCCESS):
            break
    assert outcome.state == GraspState.THUMB_OPPOSE, f"expected THUMB_OPPOSE entry, got {outcome.state}"
    return env, expert


def test_true_fingertip_differs_substantially_from_legacy_centroid():
    env, expert = _reach_thumb_oppose()
    audit = audit_hand_geometry(env, expert)
    for side in ("left", "right"):
        delta = audit["sides"][side]["legacy_vs_true_centroid_delta_m"]
        print(f"    {side}: |true - legacy centroid| = {delta:.4f} m")
        assert delta > 0.03, "the corrected true-fingertip site must differ meaningfully from the legacy body-origin centroid"


def test_thumb_0_is_not_a_negligible_lever_at_true_fingertip():
    """REAL FINDING contradicting a prior session's claim (measured at
    the uncorrected site, which sits at the joint axis of the finger's
    OWN distal link and is insensitive to upstream joints by
    construction): at the TRUE fingertip, thumb_0 has a large lever arm
    because it is upstream of thumb_1/thumb_2 in the kinematic chain."""
    env, expert = _reach_thumb_oppose()
    audit = audit_hand_geometry(env, expert)
    for side in ("left", "right"):
        travel = audit["sides"][side]["joint_tip_sensitivity"][f"{side}_hand_thumb_0_joint"]["tip_travel_m"]
        print(f"    {side} thumb_0 full-range TRUE tip travel = {travel:.4f} m")
        assert travel > 0.05, "thumb_0 must move the TRUE fingertip substantially, not ~2-5mm"


def test_thumb_thumb_distance_is_dangerously_small_at_thumb_oppose_entry():
    env, expert = _reach_thumb_oppose()
    audit = audit_hand_geometry(env, expert)
    dist = audit["thumb_thumb_distance_m"]
    print(f"    thumb-thumb distance at THUMB_OPPOSE entry = {dist:.4f} m")
    assert dist < 0.015, "REAL FINDING: both hands' thumbs converge to nearly the same point (root cause of hand-hand collisions)"


def test_full_index_middle_aligned_thumb_target_is_kinematically_infeasible():
    """REAL FINDING: matching the thumb's Y to its OWN hand's index/
    middle Y (the natural fix for the near-collision above) is NOT
    reachable with the thumb's own 3 joints from the current wrist pose
    -- thumb_1/thumb_2 pin at their hard limits with >40mm left over."""
    env, expert = _reach_thumb_oppose()
    audit = audit_hand_geometry(env, expert)
    model = env.model
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = env.data.qpos
    mujoco.mj_forward(model, scratch)

    for side in ("left", "right"):
        targets = plan_tripod_targets(env, audit, side)
        finger_qpos_adr = env._left_finger_qpos_adr if side == "left" else env._right_finger_qpos_adr
        base_q = np.array(audit["sides"][side]["current_finger_qpos"])
        solved_q, err_norm, within_limits = solve_finger_ik(
            model, scratch, side, "thumb", f"{side}_thumb_tip", targets["thumb"].world_pos,
            finger_qpos_adr, base_q, slots=[0, 1, 2],
        )
        print(f"    {side}: thumb-only IK residual = {err_norm*1000:.1f} mm (within_limits={within_limits})")
        assert err_norm > 0.03, "the full index/middle-aligned target must NOT be reachable by the thumb alone"

        thumb_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_thumb_{i}_joint") for i in range(3)]
        pinned = sum(
            1 for jid, q in zip(thumb_jids, solved_q[:3])
            if abs(q - model.jnt_range[jid][0]) < 1e-3 or abs(q - model.jnt_range[jid][1]) < 1e-3
        )
        assert pinned >= 2, "at least thumb_1 and thumb_2 must be pinned at their hard joint limits"


def test_wrist_assisted_ik_still_falls_short():
    """Adding the hand's own wrist roll/pitch/yaw as extra decision
    variables (explicitly permitted) while holding index/middle fixed
    only marginally reduces the residual -- confirms a genuine thumb
    reach limit, not a missing-DOF artifact of the thumb-only solve."""
    env, expert = _reach_thumb_oppose()
    audit = audit_hand_geometry(env, expert)
    model = env.model
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = env.data.qpos
    mujoco.mj_forward(model, scratch)

    for side in ("left", "right"):
        targets = plan_tripod_targets(env, audit, side)
        finger_qpos_adr = env._left_finger_qpos_adr if side == "left" else env._right_finger_qpos_adr
        arm_qpos_adr = env._arm_qpos_adr[:7] if side == "left" else env._arm_qpos_adr[7:]
        arm_dof_adr = env._arm_dof_adr[:7] if side == "left" else env._arm_dof_adr[7:]
        wrist_qpos_adr = arm_qpos_adr[4:7]
        wrist_dof_adr = arm_dof_adr[4:7]
        wrist_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_{n}_joint") for n in ("roll", "pitch", "yaw")]
        base_finger_q = np.array(audit["sides"][side]["current_finger_qpos"])
        base_wrist_q = scratch.qpos[wrist_qpos_adr].copy()
        index_hold = true_fingertip_world(model, scratch, f"{side}_index_tip")
        middle_hold = true_fingertip_world(model, scratch, f"{side}_middle_tip")

        res = solve_hand_closure_ik(
            model, scratch, side, targets["thumb"].world_pos, index_hold, middle_hold,
            wrist_qpos_adr, wrist_dof_adr, wrist_jids, finger_qpos_adr, base_finger_q, base_wrist_q,
        )
        print(f"    {side}: wrist+thumb IK residual = {res['err_norm']*1000:.1f} mm, wrist_delta={np.round(res['wrist_delta'],3)}")
        assert res["err_norm"] > 0.03, "wrist assistance must not fully close the gap either (genuine reach limit)"


def test_naive_swept_path_to_partial_target_collides():
    """REAL FINDING (Stage 3): even a bisected PARTIAL target the thumb
    CAN reach individually is not safely reachable via a naive
    joint-space linear interpolation from the current pose -- the path
    passes through a worse thumb-thumb near-collision and an object-box
    penetration than either endpoint. This is the reason Stage 4 (live
    controller connection) was not attempted this session."""
    env, expert = _reach_thumb_oppose()
    audit = audit_hand_geometry(env, expert)
    model = env.model
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = env.data.qpos
    mujoco.mj_forward(model, scratch)
    obj_pos = np.array(audit["object"]["pos"])
    half = audit["object"]["half_size"]

    partial_targets = {}
    for side in ("left", "right"):
        finger_qpos_adr = env._left_finger_qpos_adr if side == "left" else env._right_finger_qpos_adr
        base_q = np.array(audit["sides"][side]["current_finger_qpos"])
        thumb_now = np.array(audit["sides"][side]["current_true_tip_object_local"]["thumb"])
        idx_local = np.array(audit["sides"][side]["current_true_tip_object_local"]["index"])
        mid_local = np.array(audit["sides"][side]["current_true_tip_object_local"]["middle"])
        z = 0.5 * (idx_local[2] + mid_local[2])
        face_sign = 1.0 if np.mean([idx_local[0], mid_local[0]]) > 0 else -1.0
        x = -face_sign * (half - 0.008)
        target_y_full = 0.5 * (idx_local[1] + mid_local[1])
        lo, hi = thumb_now[1], target_y_full
        best_q = base_q.copy()
        for _ in range(16):
            mid_y = 0.5 * (lo + hi)
            solved_q, err, within = solve_finger_ik(
                model, scratch, side, "thumb", f"{side}_thumb_tip", obj_pos + np.array([x, mid_y, z]),
                finger_qpos_adr, base_q, slots=[0, 1, 2],
            )
            if within and err < 0.003:
                best_q = solved_q.copy()
                lo = mid_y
            else:
                hi = mid_y
        partial_targets[side] = best_q

    min_tt = float("inf")
    max_pen = -float("inf")
    for t in np.linspace(0.0, 1.0, 20):
        ql = (1 - t) * np.array(audit["sides"]["left"]["current_finger_qpos"]) + t * partial_targets["left"]
        qr = (1 - t) * np.array(audit["sides"]["right"]["current_finger_qpos"]) + t * partial_targets["right"]
        scratch.qpos[env._left_finger_qpos_adr] = ql
        scratch.qpos[env._right_finger_qpos_adr] = qr
        mujoco.mj_forward(model, scratch)
        lt = true_fingertip_world(model, scratch, "left_thumb_tip")
        rt = true_fingertip_world(model, scratch, "right_thumb_tip")
        min_tt = min(min_tt, float(np.linalg.norm(lt - rt)))
        for tip in (lt, rt):
            local = tip - obj_pos
            max_pen = max(max_pen, half - float(np.max(np.abs(local))))

    print(f"    swept-path min thumb-thumb distance = {min_tt*1000:.2f} mm, max object-box penetration = {max_pen*1000:.2f} mm")
    assert min_tt < 0.01, "REAL FINDING: naive linear interpolation must pass through a near-collision worse than either endpoint"
    assert max_pen > 0.0, "REAL FINDING: the naive path must also clip through the object box at some waypoint"


def test_canonical_grasp_behavior_unchanged():
    """This whole planner is read-only and never imported by the live
    control loop -- confirms the canonical streak is untouched."""
    env, expert = make_env(), None
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    outcome = None
    for _ in range(1200):
        outcome = expert.step()
        if outcome.state in (GraspState.FAILURE, GraspState.SUCCESS):
            break
    assert outcome.max_bilateral_tripod_streak == 14
    assert outcome.state == GraspState.FAILURE


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
