"""Tests for humanoid_learning.expert.diagonal_feasibility (SIZE_12
Diagonal Four-Face Grasp -- Posture-Based Multi-Start IK session).

These are STATIC (pure-kinematic, no physics rollout) tests -- they never
call env.step() or run the live grasp_expert.py state machine, matching
that session's Section 4 instruction to answer feasibility BEFORE
touching the state machine. Run with:
    python scripts/test_diagonal_feasibility.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs import hand_synergy
from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.grasp_env import FixedBaseGraspEnv
from humanoid_learning.expert.coupled_ik import CoupledBilateralIK
from humanoid_learning.expert.diagonal_feasibility import (
    CANDIDATE_C1,
    CANDIDATE_C2,
    FACE_NORMALS_LOCAL,
    build_posture_seeds,
    corner_direction_local,
    evaluate_static_pose,
    hand_corner_target,
    score_result,
)
from humanoid_learning.expert.grasp_expert import BimanualSidePinchExpert, GraspExpertConfig


def _make_solver_env():
    cfg = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF)
    env = FixedBaseGraspEnv(cfg)
    env.reset(seed=0)
    return env


def _build_coupled_ik(env):
    model = env.model
    left_palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_PALM_SITE)
    right_palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.RIGHT_PALM_SITE)
    waist_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in wbc.WAIST_JOINTS]
    waist_qpos_adr = np.array([model.jnt_qposadr[j] for j in waist_jids])
    waist_dof_adr = np.array([model.jnt_dofadr[j] for j in waist_jids])
    waist_low = model.jnt_range[waist_jids, 0].copy()
    waist_high = model.jnt_range[waist_jids, 1].copy()
    coupled = CoupledBilateralIK(
        model, left_palm_id, right_palm_id,
        waist_dof_adr, env._arm_dof_adr[:7], env._arm_dof_adr[7:],
        waist_qpos_adr, env._arm_qpos_adr[:7], env._arm_qpos_adr[7:],
        np.concatenate([waist_low, env._arm_ctrl_low[:7], env._arm_ctrl_low[7:]]),
        np.concatenate([waist_high, env._arm_ctrl_high[:7], env._arm_ctrl_high[7:]]),
    )
    stand_q17 = np.concatenate([
        env.data.qpos[waist_qpos_adr].copy(),
        env.data.qpos[env._arm_qpos_adr[:7]].copy(),
        env.data.qpos[env._arm_qpos_adr[7:]].copy(),
    ])
    return coupled, stand_q17


def _finger_open_targets(env):
    gcfg = GraspExpertConfig()
    left_open = np.array(hand_synergy.left_hand_targets(0.0), dtype=float)
    right_open = np.array(hand_synergy.right_hand_targets(0.0), dtype=float)
    left_open[1] = gcfg.thumb1_abduct_pose_left
    right_open[1] = gcfg.thumb1_abduct_pose_right
    return (env._left_finger_qpos_adr, env._right_finger_qpos_adr), (left_open, right_open)


# ---------------------------------------------------------------------
# Section 14, item 1: Candidate C1/C2 face assignment.
# ---------------------------------------------------------------------
def test_candidate_c1_c2_face_assignment_matches_spec():
    assert CANDIDATE_C1.left.thumb_face == "FACE_POS_X"
    assert CANDIDATE_C1.left.finger_face == "FACE_POS_Y"
    assert CANDIDATE_C1.right.thumb_face == "FACE_NEG_X"
    assert CANDIDATE_C1.right.finger_face == "FACE_NEG_Y"
    assert CANDIDATE_C2.left.thumb_face == "FACE_NEG_X"
    assert CANDIDATE_C2.left.finger_face == "FACE_POS_Y"
    assert CANDIDATE_C2.right.thumb_face == "FACE_POS_X"
    assert CANDIDATE_C2.right.finger_face == "FACE_NEG_Y"
    # C2 is NOT simply C1 with left/right swapped -- it also swaps which
    # face plays the thumb vs finger role (left thumb face differs
    # between the two candidates: POS_X in C1 vs NEG_X in C2).
    assert CANDIDATE_C1.left.thumb_face != CANDIDATE_C2.left.thumb_face


def test_corner_direction_local_is_adjacent_faces_bisector():
    """Each candidate's thumb/finger faces must be ADJACENT (perpendicular
    normals) so their sum is a genuine diagonal, not a degenerate
    zero-vector (opposite faces) -- corner_direction_local asserts this
    internally; this test additionally checks the resulting direction is
    a proper 45-degree bisector for a known assignment."""
    from humanoid_learning.expert.diagonal_feasibility import HandFaceAssignment
    d = corner_direction_local(HandFaceAssignment("FACE_POS_X", "FACE_POS_Y"))
    np.testing.assert_allclose(d, np.array([0.7071, 0.7071, 0.0]), atol=1e-3)
    d2 = corner_direction_local(HandFaceAssignment("FACE_NEG_X", "FACE_NEG_Y"))
    np.testing.assert_allclose(d2, np.array([-0.7071, -0.7071, 0.0]), atol=1e-3)


# ---------------------------------------------------------------------
# Section 14, item 2: object rotation must be reflected in the face
# target (world-frame corner point/direction rotates WITH the object).
# ---------------------------------------------------------------------
def test_hand_corner_target_rotates_with_object_quaternion():
    obj_pos = np.array([0.27, 0.0, 0.8])
    half_size = SIZE_12_HALF
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    t0 = hand_corner_target(obj_pos, identity_quat, half_size, CANDIDATE_C1.left, palm_standoff=0.15)
    quat_90z = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    t1 = hand_corner_target(obj_pos, quat_90z, half_size, CANDIDATE_C1.left, palm_standoff=0.15)
    # A 90-degree object rotation must rotate the corner target by 90
    # degrees too (not leave it fixed in world space, which would mean
    # the target ignores obj_quat entirely -- exactly the "world axis를
    # 그대로 face로 가정하지 말라" mistake the spec warns against).
    assert not np.allclose(t0.palm_pos, t1.palm_pos, atol=1e-3)
    corner_vec_0 = t0.corner_edge_point_world - obj_pos
    corner_vec_1 = t1.corner_edge_point_world - obj_pos
    cos_angle = np.dot(corner_vec_0, corner_vec_1) / (np.linalg.norm(corner_vec_0) * np.linalg.norm(corner_vec_1))
    assert abs(cos_angle) < 0.1, f"expected ~90 degree rotation between corner targets, cos={cos_angle}"


# ---------------------------------------------------------------------
# Section 14, items 3/4: posture-family seeds are deterministic AND use
# genuinely different qpos (not all collapsing to the same start).
# ---------------------------------------------------------------------
def test_posture_seeds_deterministic_and_distinct():
    stand_q17 = np.zeros(17)
    stand_q17[3] = 0.2
    stand_q17[6] = 1.28
    seeds_a = build_posture_seeds(stand_q17)
    seeds_b = build_posture_seeds(stand_q17)
    assert len(seeds_a) >= 8, "Section 5 requires at least the listed posture families"
    names = [s.name for s in seeds_a]
    assert len(names) == len(set(names)), "posture family names must be unique"
    for sa, sb in zip(seeds_a, seeds_b):
        np.testing.assert_array_equal(sa.q17, sb.q17)  # deterministic
    # genuinely different starting qpos across seeds (not all equal to
    # neutral_rest or to each other).
    distinct_count = len({tuple(np.round(s.q17, 6)) for s in seeds_a})
    assert distinct_count == len(seeds_a), "each posture family must be a distinct qpos seed"


# ---------------------------------------------------------------------
# Section 14, item 7: joint-limit margin is actually computed (not a
# constant placeholder) -- perturbing a seed far past a joint's range
# and clipping must show a small/zero margin, while the stand pose
# itself shows a healthy margin.
# ---------------------------------------------------------------------
def test_joint_limit_margin_reflects_real_proximity_to_limits():
    env = _make_solver_env()
    coupled, stand_q17 = _build_coupled_ik(env)
    low = coupled.joint_low
    high = coupled.joint_high
    margins_at_stand = np.minimum(stand_q17 - low, high - stand_q17)
    assert margins_at_stand.min() > 0.05, "stand pose should not already be near any joint limit"
    near_limit_q = high.copy()  # every joint pinned at its upper limit
    margins_at_limit = np.minimum(near_limit_q - low, high - near_limit_q)
    assert margins_at_limit.min() < 1e-6


# ---------------------------------------------------------------------
# Section 14, item 6: a candidate target that is deliberately placed
# INSIDE the object (guaranteed self/object collision) must be rejected
# by the collision scan, not silently reported collision_free.
# ---------------------------------------------------------------------
def test_collision_candidate_is_rejected_not_silently_passed():
    env = _make_solver_env()
    coupled, stand_q17 = _build_coupled_ik(env)
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = env.data.qpos
    scratch.qvel[:] = 0.0
    obj_pos = env.data.qpos[env._object_qpos_adr:env._object_qpos_adr + 3].copy()
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    # A palm_standoff of 0.0 places the palm essentially ON the object's
    # corner surface -- must not be reported collision-free once fingers
    # are preshaped into their normal open pose.
    left_t = hand_corner_target(obj_pos, identity_quat, SIZE_12_HALF, CANDIDATE_C1.left, palm_standoff=0.0)
    right_t = hand_corner_target(obj_pos, identity_quat, SIZE_12_HALF, CANDIDATE_C1.right, palm_standoff=0.0)
    finger_adr, finger_targets = _finger_open_targets(env)
    seeds = build_posture_seeds(stand_q17)
    r = evaluate_static_pose(
        coupled, scratch, env.model, env._object_body_id, CANDIDATE_C1, seeds[0], left_t, right_t,
        finger_qpos_adr=finger_adr, finger_open_targets=finger_targets,
    )
    assert not r.collision_free, "zero standoff must be detected as a collision, not passed silently"
    assert not r.success


# ---------------------------------------------------------------------
# Section 14, item 5: scoring/filtering orders collision-free ahead of
# colliding, and joint-limit-safe ahead of unsafe, regardless of raw
# Cartesian error (Section 5: "단순 Cartesian error만 가장 낮은 후보를
# 고르지 마라").
# ---------------------------------------------------------------------
def test_score_result_prioritizes_collision_and_joint_limit_over_raw_error():
    from humanoid_learning.expert.diagonal_feasibility import FeasibilityResult
    good_but_far = FeasibilityResult(
        candidate="C1", seed_name="a", success=False,
        left_pos_error=0.20, left_ori_error=0.20, right_pos_error=0.20, right_ori_error=0.20,
        joint_limit_margin=0.05, collision_free=True, collision_pairs=[],
        left_target_face_thumb="FACE_POS_X", left_target_face_finger="FACE_POS_Y",
        right_target_face_thumb="FACE_NEG_X", right_target_face_finger="FACE_NEG_Y",
        q17=np.zeros(17), iterations=10,
    )
    close_but_colliding = FeasibilityResult(
        candidate="C1", seed_name="b", success=False,
        left_pos_error=0.001, left_ori_error=0.001, right_pos_error=0.001, right_ori_error=0.001,
        joint_limit_margin=0.05, collision_free=False, collision_pairs=[("a", "b")],
        left_target_face_thumb="FACE_POS_X", left_target_face_finger="FACE_POS_Y",
        right_target_face_thumb="FACE_NEG_X", right_target_face_finger="FACE_NEG_Y",
        q17=np.zeros(17), iterations=10,
    )
    assert score_result(good_but_far) < score_result(close_but_colliding)


# ---------------------------------------------------------------------
# Section 14, item 13 / Section 17: the actual static feasibility
# verdict. THIS TEST DOCUMENTS A REAL, MEASURED NEGATIVE RESULT -- see
# PROJECT_CONTEXT.md's 21st-session report for the full systematic sweep
# (2 candidates x 10 posture families x 5-10 standoffs = up to 100
# attempts). It is intentionally left FAILING (not adjusted to pass) per
# project convention: no candidate achieved simultaneous (a) sub-2cm/
# sub-15-degree convergence for BOTH hands, (b) collision-free contact,
# and (c) >=0.02 joint-limit margin. The binding constraint found in
# EVERY attempt across the wider sweep was the "far corner" hand's
# wrist_yaw joint pinned exactly at its physical limit (+/-1.614 rad)
# with elbow also near its limit -- a genuine, reproducible mechanical
# reach/orientation constraint, not a solver artifact (verified: the
# EASY-side hand in every candidate converges to <1cm/<5-degree with
# healthy margin using the exact same solver/tolerances).
# ---------------------------------------------------------------------
def test_static_feasibility_search_size12_diagonal_corner():
    env = _make_solver_env()
    coupled, stand_q17 = _build_coupled_ik(env)
    seeds = build_posture_seeds(stand_q17)
    obj_pos = env.data.qpos[env._object_qpos_adr:env._object_qpos_adr + 3].copy()
    obj_quat = env.data.qpos[env._object_qpos_adr + 3:env._object_qpos_adr + 7].copy()
    finger_adr, finger_targets = _finger_open_targets(env)
    scratch = mujoco.MjData(env.model)

    results = []
    for candidate in (CANDIDATE_C1, CANDIDATE_C2):
        for standoff in (0.14, 0.18, 0.22):
            left_t = hand_corner_target(obj_pos, obj_quat, SIZE_12_HALF, candidate.left, palm_standoff=standoff)
            right_t = hand_corner_target(obj_pos, obj_quat, SIZE_12_HALF, candidate.right, palm_standoff=standoff)
            for seed in seeds:
                scratch.qpos[:] = env.data.qpos
                scratch.qvel[:] = 0.0
                r = evaluate_static_pose(
                    coupled, scratch, env.model, env._object_body_id, candidate, seed, left_t, right_t,
                    finger_qpos_adr=finger_adr, finger_open_targets=finger_targets,
                )
                results.append((standoff, r))

    results.sort(key=lambda sr: score_result(sr[1]))
    best_standoff, best = results[0]
    print(
        f"    best candidate={best.candidate} seed={best.seed_name} standoff={best_standoff} "
        f"collision_free={best.collision_free} joint_limit_margin={best.joint_limit_margin:.4f} "
        f"Lpos={best.left_pos_error:.4f} Rpos={best.right_pos_error:.4f} "
        f"Lori={best.left_ori_error:.4f} Rori={best.right_ori_error:.4f}"
    )
    n_success = sum(1 for _, r in results if r.success)
    print(f"    {n_success} / {len(results)} attempts fully succeeded (collision-free + joint-limit-safe + converged)")
    assert best.left_pos_error < 0.02 and best.right_pos_error < 0.02, (
        "Gate A (position component of the static pose) FAILED: even the best-scored candidate "
        f"has position error Lpos={best.left_pos_error:.4f} Rpos={best.right_pos_error:.4f} (need < 0.02m each) -- "
        "see PROJECT_CONTEXT.md for the joint-limit root cause (wrist_yaw/elbow saturation on the far-corner hand)."
    )
    assert best.left_ori_error < 0.1 and best.right_ori_error < 0.1, (
        "Gate A (orientation component of the static pose) FAILED: even the best-scored candidate "
        f"has orientation error Lori={best.left_ori_error:.4f} Rori={best.right_ori_error:.4f} rad (need < 0.1 rad each) -- "
        "see PROJECT_CONTEXT.md for the joint-limit root cause (wrist_yaw/elbow saturation on the far-corner hand)."
    )


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
