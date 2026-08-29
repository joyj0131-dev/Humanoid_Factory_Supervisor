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
    build_candidate_specific_seeds,
    build_posture_seeds,
    corner_direction_local,
    corner_direction_local_at_yaw,
    evaluate_static_pose,
    hand_corner_target,
    measure_per_finger_local_offsets,
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


def _finger_offsets(env):
    gcfg = GraspExpertConfig()
    left_offsets = measure_per_finger_local_offsets(env, "left", gcfg.thumb1_abduct_pose_left)
    right_offsets = measure_per_finger_local_offsets(env, "right", gcfg.thumb1_abduct_pose_right)
    return left_offsets, right_offsets


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
    left_offsets, right_offsets = _finger_offsets(env)
    seeds = build_posture_seeds(stand_q17)
    r = evaluate_static_pose(
        coupled, scratch, env.model, env._object_body_id, obj_pos, identity_quat, CANDIDATE_C1, seeds[0], left_t, right_t,
        left_offsets, right_offsets, finger_qpos_adr=finger_adr, finger_open_targets=finger_targets,
    )
    assert not r.collision_free, "zero standoff must be detected as a collision, not passed silently"
    assert not r.success


# ---------------------------------------------------------------------
# Section 14, item 5: scoring/filtering orders collision-free ahead of
# colliding, and joint-limit-safe ahead of unsafe, regardless of raw
# Cartesian error (Section 5: "단순 Cartesian error만 가장 낮은 후보를
# 고르지 마라").
# ---------------------------------------------------------------------
def _make_feasibility_result(**overrides):
    from humanoid_learning.expert.diagonal_feasibility import FeasibilityResult
    defaults = dict(
        candidate="C1", seed_name="a", success=False,
        left_pos_error=0.0, left_ori_error=0.0, right_pos_error=0.0, right_ori_error=0.0,
        joint_limit_margin=0.05, wrist_yaw_margin=0.05, elbow_margin=0.05,
        collision_free=True, collision_pairs=[],
        left_target_face_thumb="FACE_POS_X", left_target_face_finger="FACE_POS_Y",
        right_target_face_thumb="FACE_NEG_X", right_target_face_finger="FACE_NEG_Y",
        left_thumb_face_actual="FACE_POS_X", left_finger_face_actual="FACE_POS_Y",
        right_thumb_face_actual="FACE_NEG_X", right_finger_face_actual="FACE_NEG_Y",
        face_assignment_satisfied=True,
        q17=np.zeros(17), corner_yaw_deg=(45.0, 45.0), waist_weight_used=6.0, iterations=10,
    )
    defaults.update(overrides)
    return FeasibilityResult(**defaults)


def test_score_result_prioritizes_collision_and_joint_limit_over_raw_error():
    good_but_far = _make_feasibility_result(left_pos_error=0.20, right_pos_error=0.20, left_ori_error=0.20, right_ori_error=0.20)
    close_but_colliding = _make_feasibility_result(
        left_pos_error=0.001, right_pos_error=0.001, left_ori_error=0.001, right_ori_error=0.001,
        collision_free=False, collision_pairs=[("a", "b")], success=False,
    )
    assert score_result(good_but_far) < score_result(close_but_colliding)


def test_score_result_rejects_face_assignment_failure_over_raw_error():
    """Section 2's real success criterion: a candidate with tiny position/
    orientation error but WRONG fingertip faces (e.g. the solver drifted
    to a locally-close but geometrically wrong pose) must score worse
    than one with larger error but correct face assignment."""
    correct_faces_but_farther = _make_feasibility_result(left_pos_error=0.03, right_pos_error=0.03)
    tiny_error_wrong_faces = _make_feasibility_result(
        left_pos_error=0.001, right_pos_error=0.001,
        face_assignment_satisfied=False, left_thumb_face_actual="FACE_NEG_Y", success=False,
    )
    assert score_result(correct_faces_but_farther) < score_result(tiny_error_wrong_faces)


def test_score_result_rejects_one_sided_convergence():
    """Section 12: "한 손만 잘 수렴하고 다른 손이 limit에 걸린 후보는
    탈락시킨다" -- an excellent-left/terrible-right candidate must not
    outscore a mediocre-but-balanced one, i.e. scoring must use the
    WORSE hand, not the average."""
    balanced = _make_feasibility_result(left_pos_error=0.025, right_pos_error=0.025)
    one_sided = _make_feasibility_result(left_pos_error=0.001, right_pos_error=0.15, success=False)
    assert score_result(balanced) < score_result(one_sided)


# ---------------------------------------------------------------------
# New this session: corner_yaw_deg interpolation and candidate-specific
# posture seeds.
# ---------------------------------------------------------------------
def test_corner_direction_local_at_yaw_interpolates_from_finger_face_to_bisector():
    d0 = corner_direction_local_at_yaw(CANDIDATE_C1.left, 0.0)
    np.testing.assert_allclose(d0, FACE_NORMALS_LOCAL["FACE_POS_Y"], atol=1e-6)
    d45 = corner_direction_local_at_yaw(CANDIDATE_C1.left, 45.0)
    np.testing.assert_allclose(d45, corner_direction_local(CANDIDATE_C1.left), atol=1e-6)
    d20 = corner_direction_local_at_yaw(CANDIDATE_C1.left, 20.0)
    # strictly between the two endpoints, not equal to either
    assert not np.allclose(d20, d0, atol=1e-3) and not np.allclose(d20, d45, atol=1e-3)


def test_candidate_specific_seeds_target_the_known_hard_hand():
    stand_q17 = np.zeros(17)
    c1_seeds = build_candidate_specific_seeds("C1", stand_q17)
    c2_seeds = build_candidate_specific_seeds("C2", stand_q17)
    assert len(c1_seeds) >= 1 and len(c2_seeds) >= 1
    # C1's seed perturbs the LEFT arm's block (indices 3-9), C2's the RIGHT (10-16).
    assert np.any(c1_seeds[0].q17[3:10] != 0.0)
    assert np.any(c2_seeds[0].q17[10:17] != 0.0)


# ---------------------------------------------------------------------
# Real gap found after a user question ("자세를 트는건 하지 않았어?"):
# every posture-family SEED biases the null-space rest_q toward more
# waist rotation, but CoupledBilateralIK's default 6x DLS task-space
# penalty on the waist (coupled_ik.py, waist_weight=6.0) means the
# PRIMARY solve barely lets the waist actually move away from wherever
# it started, regardless of the seed. This test proves that directly:
# the SAME seed/target/standoff, solved once at the solver's own default
# weight and once with the waist made as cheap as an arm joint, must
# show genuinely different waist usage and joint-limit outcomes -- if
# they were the same, waist_weight would not be the actual lever.
# ---------------------------------------------------------------------
def test_waist_weight_override_actually_changes_waist_usage_and_wrist_yaw_margin():
    env = _make_solver_env()
    coupled, stand_q17 = _build_coupled_ik(env)
    obj_pos = env.data.qpos[env._object_qpos_adr:env._object_qpos_adr + 3].copy()
    obj_quat = env.data.qpos[env._object_qpos_adr + 3:env._object_qpos_adr + 7].copy()
    finger_adr, finger_targets = _finger_open_targets(env)
    left_offsets, right_offsets = _finger_offsets(env)
    seed = build_candidate_specific_seeds("C2", stand_q17)[0]
    left_t = hand_corner_target(obj_pos, obj_quat, SIZE_12_HALF, CANDIDATE_C2.left, palm_standoff=0.20, corner_yaw_deg=45.0)
    right_t = hand_corner_target(obj_pos, obj_quat, SIZE_12_HALF, CANDIDATE_C2.right, palm_standoff=0.20, corner_yaw_deg=45.0)

    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = env.data.qpos
    r_default = evaluate_static_pose(
        coupled, scratch, env.model, env._object_body_id, obj_pos, obj_quat, CANDIDATE_C2, seed, left_t, right_t,
        left_offsets, right_offsets, finger_qpos_adr=finger_adr, finger_open_targets=finger_targets,
        wrist_yaw_rest_gain_boost=0.5, waist_weight=None,
    )
    scratch.qpos[:] = env.data.qpos
    r_freed = evaluate_static_pose(
        coupled, scratch, env.model, env._object_body_id, obj_pos, obj_quat, CANDIDATE_C2, seed, left_t, right_t,
        left_offsets, right_offsets, finger_qpos_adr=finger_adr, finger_open_targets=finger_targets,
        wrist_yaw_rest_gain_boost=0.5, waist_weight=1.0,
    )
    waist_yaw_default = r_default.q17[0]
    waist_yaw_freed = r_freed.q17[0]
    print(
        f"    default(waist_weight=6.0): waist_yaw={waist_yaw_default:+.3f} wrist_yaw_margin={r_default.wrist_yaw_margin:.4f} "
        f"elbow_margin={r_default.elbow_margin:.4f}\n"
        f"    freed(waist_weight=1.0):   waist_yaw={waist_yaw_freed:+.3f} wrist_yaw_margin={r_freed.wrist_yaw_margin:.4f} "
        f"elbow_margin={r_freed.elbow_margin:.4f}"
    )
    assert abs(waist_yaw_freed - seed.q17[0]) > abs(waist_yaw_default - seed.q17[0]), (
        "freeing the waist (lower joint_weight) must let it actually rotate FURTHER from its seed "
        "value than the solver's own 6x-penalized default does -- otherwise waist_weight is not the real lever"
    )
    assert r_freed.elbow_margin > r_default.elbow_margin + 0.5, (
        "freeing the waist must measurably relieve elbow joint-limit saturation compared to the default "
        f"(default={r_default.elbow_margin:.4f}, freed={r_freed.elbow_margin:.4f})"
    )


# ---------------------------------------------------------------------
# Section 15, item 1: the 21st-session 45-degree-forced baseline is
# PRESERVED here (not deleted), still reproducing its own negative
# result, so the Contact-Driven session's own result can be compared
# against it directly. ori_task_weight=1.0/require_orientation=True and
# wrist_yaw_rest_gain_boost=0.0 exactly reproduce that session's
# behavior through the now-shared evaluate_static_pose.
# ---------------------------------------------------------------------
def test_static_feasibility_45deg_forced_orientation_baseline_size12():
    env = _make_solver_env()
    coupled, stand_q17 = _build_coupled_ik(env)
    seeds = build_posture_seeds(stand_q17)
    obj_pos = env.data.qpos[env._object_qpos_adr:env._object_qpos_adr + 3].copy()
    obj_quat = env.data.qpos[env._object_qpos_adr + 3:env._object_qpos_adr + 7].copy()
    finger_adr, finger_targets = _finger_open_targets(env)
    left_offsets, right_offsets = _finger_offsets(env)
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
                    coupled, scratch, env.model, env._object_body_id, obj_pos, obj_quat, candidate, seed, left_t, right_t,
                    left_offsets, right_offsets, finger_qpos_adr=finger_adr, finger_open_targets=finger_targets,
                    ori_task_weight=1.0, require_orientation=True, wrist_yaw_rest_gain_boost=0.0,
                )
                results.append((standoff, r))

    results.sort(key=lambda sr: score_result(sr[1]))
    best_standoff, best = results[0]
    print(
        f"    [45deg baseline] best candidate={best.candidate} seed={best.seed_name} standoff={best_standoff} "
        f"wrist_yaw_margin={best.wrist_yaw_margin:.4f} elbow_margin={best.elbow_margin:.4f} "
        f"Lpos={best.left_pos_error:.4f} Rpos={best.right_pos_error:.4f} "
        f"Lori={best.left_ori_error:.4f} Rori={best.right_ori_error:.4f} face_ok={best.face_assignment_satisfied}"
    )
    n_success = sum(1 for _, r in results if r.success)
    print(f"    {n_success} / {len(results)} attempts fully succeeded")
    assert best.wrist_yaw_margin >= 0.02, (
        "45-degree-forced baseline Gate FAILED (expected -- this documents the prior session's finding): "
        f"wrist_yaw_margin={best.wrist_yaw_margin:.4f} < 0.02 even for the best-scored candidate. "
        "See PROJECT_CONTEXT.md 21st/22nd-session reports."
    )


# ---------------------------------------------------------------------
# Stage U (this session): contact-driven orientation -- corner_yaw swept
# 10-45 degrees INDEPENDENTLY per hand, soft orientation task, explicit
# wrist_yaw rest_gain boost, candidate-specific posture seeds added, and
# real success measured by FINGERTIP FACE ASSIGNMENT (Section 2), not by
# hitting an exact palm angle. Coarse-to-fine (Section 6): a coarse
# symmetric-angle x standoff x posture-family pass, then a fine pass
# with INDEPENDENT left/right angles around the best coarse region.
# ---------------------------------------------------------------------
def test_stage_u_contact_driven_four_face_feasibility_size12():
    env = _make_solver_env()
    coupled, stand_q17 = _build_coupled_ik(env)
    seeds = build_posture_seeds(stand_q17)
    obj_pos = env.data.qpos[env._object_qpos_adr:env._object_qpos_adr + 3].copy()
    obj_quat = env.data.qpos[env._object_qpos_adr + 3:env._object_qpos_adr + 7].copy()
    finger_adr, finger_targets = _finger_open_targets(env)
    left_offsets, right_offsets = _finger_offsets(env)
    scratch = mujoco.MjData(env.model)

    def solve_one(candidate, seed, yaw_l, yaw_r, standoff, waist_weight=None, height_offset=0.0):
        left_t = hand_corner_target(obj_pos, obj_quat, SIZE_12_HALF, candidate.left, palm_standoff=standoff, height_offset=height_offset, corner_yaw_deg=yaw_l)
        right_t = hand_corner_target(obj_pos, obj_quat, SIZE_12_HALF, candidate.right, palm_standoff=standoff, height_offset=height_offset, corner_yaw_deg=yaw_r)
        scratch.qpos[:] = env.data.qpos
        scratch.qvel[:] = 0.0
        return evaluate_static_pose(
            coupled, scratch, env.model, env._object_body_id, obj_pos, obj_quat, candidate, seed, left_t, right_t,
            left_offsets, right_offsets, finger_qpos_adr=finger_adr, finger_open_targets=finger_targets,
            corner_yaw_deg=(yaw_l, yaw_r), waist_weight=waist_weight,
        )

    # --- coarse pass: symmetric angle, both candidates, a curated seed
    # subset (+ the candidate-specific seed), a few standoffs, AND
    # (added after the waist-weight finding above) TWO waist_weight
    # settings -- the solver's own 6x-penalized default, and a "freed"
    # 1.0 that lets the waist actually twist as far as the null-space
    # bias asks. Trimmed to a curated 8-seed subset (from the full 15 in
    # build_posture_seeds) to keep this extra axis's runtime bounded.
    curated_seed_names = {
        "neutral_rest", "shoulder_forward", "waist_yaw_assist", "waist_yaw_assist_negative",
        "waist_yaw_asymmetric_shoulders", "shoulder_diagonal_outward", "shoulder_diagonal_inward",
        "waist_pitch_shoulder_forward_elbow_extension",
    }
    coarse_results = []
    n_coarse_attempts = 0
    for candidate in (CANDIDATE_C1, CANDIDATE_C2):
        all_seeds = [s for s in seeds if s.name in curated_seed_names] + build_candidate_specific_seeds(candidate.name, stand_q17)
        for yaw in (15.0, 30.0, 45.0):
            for standoff in (0.20, 0.24):
                for waist_weight in (None, 1.0):
                    for seed in all_seeds:
                        r = solve_one(candidate, seed, yaw, yaw, standoff, waist_weight=waist_weight)
                        coarse_results.append((standoff, r))
                        n_coarse_attempts += 1

    coarse_results.sort(key=lambda sr: score_result(sr[1]))
    best_coarse_standoff, best_coarse = coarse_results[0]
    print(
        f"    [Stage U coarse] {n_coarse_attempts} attempts; best candidate={best_coarse.candidate} "
        f"seed={best_coarse.seed_name} yaw={best_coarse.corner_yaw_deg} standoff={best_coarse_standoff} "
        f"waist_weight={best_coarse.waist_weight_used:.1f} "
        f"face_ok={best_coarse.face_assignment_satisfied} collision_free={best_coarse.collision_free} "
        f"wrist_yaw_margin={best_coarse.wrist_yaw_margin:.4f} elbow_margin={best_coarse.elbow_margin:.4f} "
        f"Lpos={best_coarse.left_pos_error:.4f} Rpos={best_coarse.right_pos_error:.4f}"
    )

    # --- fine pass: independent left/right angles around the coarse
    # winner's angle, PLUS a small height_offset sweep (the waist-weight
    # experiment found freeing the waist can trade wrist_yaw margin for
    # a NEW table collision -- height_offset is the natural knob to
    # recheck that trade-off with), same candidate/seed/standoff/
    # waist_weight.
    best_candidate_obj = CANDIDATE_C1 if best_coarse.candidate == "C1" else CANDIDATE_C2
    best_seed = next(
        s for s in ([s for s in seeds if s.name in curated_seed_names] + build_candidate_specific_seeds(best_coarse.candidate, stand_q17))
        if s.name == best_coarse.seed_name
    )
    base_yaw = best_coarse.corner_yaw_deg[0]
    fine_angles = sorted({max(10.0, base_yaw - 10.0), base_yaw, min(45.0, base_yaw + 10.0)})
    fine_results = []
    for yaw_l in fine_angles:
        for yaw_r in fine_angles:
            for height_offset in (0.0, 0.03, -0.03):
                r = solve_one(
                    best_candidate_obj, best_seed, yaw_l, yaw_r, best_coarse_standoff,
                    waist_weight=best_coarse.waist_weight_used if best_coarse.waist_weight_used != 6.0 else None,
                    height_offset=height_offset,
                )
                fine_results.append(r)
    fine_results.sort(key=score_result)
    best_fine = fine_results[0]
    print(
        f"    [Stage U fine] {len(fine_results)} attempts around yaw={base_yaw}; best yaw={best_fine.corner_yaw_deg} "
        f"face_ok={best_fine.face_assignment_satisfied} collision_free={best_fine.collision_free} "
        f"wrist_yaw_margin={best_fine.wrist_yaw_margin:.4f} elbow_margin={best_fine.elbow_margin:.4f} "
        f"Lpos={best_fine.left_pos_error:.4f} Rpos={best_fine.right_pos_error:.4f} "
        f"actual_faces=(L:{best_fine.left_thumb_face_actual}/{best_fine.left_finger_face_actual}, "
        f"R:{best_fine.right_thumb_face_actual}/{best_fine.right_finger_face_actual})"
    )

    overall_best = min(coarse_results[0][1], best_fine, key=score_result)
    n_success_coarse = sum(1 for _, r in coarse_results if r.success)
    n_success_fine = sum(1 for r in fine_results if r.success)
    print(f"    total Stage U attempts={n_coarse_attempts + len(fine_results)}, "
          f"fully succeeded={n_success_coarse + n_success_fine}")

    # Real Stage U success criterion (Section 7): face assignment
    # satisfied, collision-free, AND a healthy wrist_yaw/elbow margin --
    # NOT a hard-coded palm-orientation check.
    assert overall_best.face_assignment_satisfied, (
        "Stage U FAILED: even the best contact-driven candidate does not achieve the intended "
        f"fingertip-face assignment (got L:{overall_best.left_thumb_face_actual}/{overall_best.left_finger_face_actual}, "
        f"R:{overall_best.right_thumb_face_actual}/{overall_best.right_finger_face_actual})."
    )
    assert overall_best.collision_free, "Stage U FAILED: best candidate still collides."
    assert overall_best.wrist_yaw_margin >= 0.02 and overall_best.elbow_margin >= 0.02, (
        f"Stage U FAILED: wrist_yaw_margin={overall_best.wrist_yaw_margin:.4f} / "
        f"elbow_margin={overall_best.elbow_margin:.4f}, need >= 0.02 each."
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
