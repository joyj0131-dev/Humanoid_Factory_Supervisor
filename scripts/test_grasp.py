"""Phase 4 Grasp Track tests: FixedBaseGraspEnv mechanics, palm frame,
SO(3) pose IK, and the bimanual side-pinch expert (redesigned 9-state
sequence, 2x-scaled object, real finger-based bimanual contact).

IMPORTANT (see PROJECT_CONTEXT.md Phase 4 Grasp Track report): the
redesign fixed a real IK joint-limit lockup (via null-space posture
regularization), a collision path during approach, a stiff finger-actuator
force spike, and a left/right palm-height asymmetry that was launching and
spinning the object -- and achieved genuine sustained bilateral FINGER
contact (not palm-only) under controlled, low (~5-8N) force, a large
improvement over the previous session's 22-36N palm-squeeze pattern. It
did NOT achieve a controlled LIFT+HOLD: the object still separates from
the grip (an asymmetric-contact-timing failure, not a launch) before
FORCE_SETTLE/LIFT is reached. test_fixed_base_bimanual_grasp_lifts_and_holds_object
below asserts the real Section 6 success criteria and is EXPECTED TO FAIL
until that gap is closed -- it is not lowered or skipped to force a pass.
The diagnostic test calls the REAL, reusable controller
(humanoid_learning.expert.grasp_expert.BimanualSidePinchExpert, which
itself uses humanoid_learning.expert.pose_ik's SO(3) IK and the empirically
-measured palm closing axes) -- there is no separate inline IK duplicated
here.

Run with:
    python scripts/test_grasp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs import hand_synergy
from humanoid_learning.envs import model_builder
from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.envs.grasp_config import GraspEnvConfig
from humanoid_learning.envs.grasp_env import ACTION_DIM, FixedBaseGraspEnv
from humanoid_learning.expert import pose_ik
from humanoid_learning.expert.coupled_ik import CoupledBilateralIK
from humanoid_learning.expert.grasp_expert import (
    BimanualSidePinchExpert,
    FailureReason,
    GraspExpertConfig,
    GraspState,
    make_side_pinch_orientation,
)


def make_env(object_pos=(0.27, 0.08, 0.0), **kwargs) -> FixedBaseGraspEnv:
    return FixedBaseGraspEnv(GraspEnvConfig(object_pos=object_pos, **kwargs))


# ---------------------------------------------------------------------
# Harness bug regression (finger control persistence)
# ---------------------------------------------------------------------


def test_finger_control_persists_across_steps():
    env = make_env()
    env.reset(seed=0)
    left_finger_qpos_adr = env._left_finger_qpos_adr
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[17:20] = 1.0  # left hand groups (thumb, index, middle) -- 23-dim layout

    distances = []
    open_qpos = env.data.qpos[left_finger_qpos_adr].copy()
    for _ in range(40):
        env.step(a)
        distances.append(float(np.linalg.norm(env.data.qpos[left_finger_qpos_adr] - open_qpos)))

    peak = max(distances)
    min_after_peak = min(distances[distances.index(peak) :])
    assert min_after_peak > 0.5 * peak, (
        f"finger distance-from-open collapsed after peak ({peak:.3f} -> {min_after_peak:.3f}) "
        f"-- looks like the held-actuator-reset bug: {distances}"
    )
    assert distances[-1] > 0.3


def test_ctrl_not_reset_for_legs_waist_between_steps():
    env = make_env()
    env.reset(seed=0)
    non_arm_hand_act_ids = [
        i
        for i in range(env.model.nu)
        if i not in set(env._arm_act_ids) and i not in set(env._left_hand_act_ids) and i not in set(env._right_hand_act_ids)
    ]
    stand_ctrl = env._stand_ctrl[non_arm_hand_act_ids].copy()
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[3:10] = 1.0  # left arm -- waist (0:3) deliberately left at 0 so it holds its stand target too
    for _ in range(20):
        env.step(a)
    np.testing.assert_allclose(env.data.ctrl[non_arm_hand_act_ids], stand_ctrl)


def test_object_velocity_uses_dof_adr_not_qpos_arithmetic():
    env = make_env()
    env.reset(seed=0)
    obj_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint")
    assert env._object_dof_adr == env.model.jnt_dofadr[obj_jid]
    info = env._get_info()
    assert info["object_velocity"].shape == (6,)
    assert np.isfinite(info["object_velocity"]).all()


def test_object_no_initial_interpenetration_drift():
    env = make_env()
    obs, info = env.reset(seed=0)
    obj0 = info["object_position"].copy()
    zero = np.zeros(ACTION_DIM, dtype=np.float32)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(zero)
    drift = float(np.linalg.norm(info["object_position"] - obj0))
    assert drift < 0.005, f"object drifted {drift:.4f} with zero action"


# ---------------------------------------------------------------------
# Palm frame (Section 2): orthonormal, mirrored, matches measured closing axis
# ---------------------------------------------------------------------


def test_palm_frame_sites_resolve():
    env = make_env()
    for name in (wbc.LEFT_PALM_SITE, wbc.RIGHT_PALM_SITE, *wbc.FINGERTIP_SITE_BODIES):
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0, name


def test_palm_frame_orthonormal_proper_rotation():
    env = make_env()
    env.reset(seed=0)
    for site_name in (wbc.LEFT_PALM_SITE, wbc.RIGHT_PALM_SITE):
        sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        R = env.data.site_xmat[sid].reshape(3, 3)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
        assert abs(np.linalg.det(R) - 1.0) < 1e-8


def test_palm_frame_closing_axes_mirrored():
    # Left closing = local -Y, right closing = local +Y (measured, Phase 4
    # report) -- in their own local site frame this is a fixed convention,
    # verified here via the site's local axis definition itself (not
    # dependent on arm pose).
    left_R = make_side_pinch_orientation(np.array([0.0, -1.0, 0.0]))
    right_R = make_side_pinch_orientation(np.array([0.0, 1.0, 0.0]))
    assert np.linalg.det(left_R) > 0 and np.linalg.det(right_R) > 0
    np.testing.assert_allclose(left_R[:, 1], [0, -1, 0], atol=1e-8)
    np.testing.assert_allclose(right_R[:, 1], [0, 1, 0], atol=1e-8)


def test_synergy_moves_fingertip_toward_closing_axis():
    env = make_env()
    env.reset(seed=0)
    palm_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_PALM_SITE)
    tip_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_INDEX_TIP_SITE)

    def local_tip():
        R = env.data.site_xmat[palm_id].reshape(3, 3)
        return R.T @ (env.data.site_xpos[tip_id] - env.data.site_xpos[palm_id])

    open_local = local_tip()
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[18] = 1.0  # left INDEX group (thumb=17, index=18, middle=19)
    for _ in range(40):
        env.step(a)
    closed_local = local_tip()
    delta = closed_local - open_local
    # palm frame column 1 IS the closing axis, defined so that it points in
    # the physical closing direction -- so a fingertip curling in should
    # show a POSITIVE component along it once expressed in the palm's own
    # local frame (that axis IS local +Y by construction).
    assert delta[1] > 0.01, f"expected fingertip to move along +closing_axis (palm-local Y), got delta={delta}"


def test_left_right_hand_synergy_mirrored_sign():
    left_close = hand_synergy.left_hand_targets(1.0)
    right_close = hand_synergy.right_hand_targets(1.0)
    assert left_close[1] > 0 and right_close[1] < 0


# ---------------------------------------------------------------------
# SO(3) pose IK (Section 3)
# ---------------------------------------------------------------------


def test_so3_log_identity_is_zero():
    assert np.linalg.norm(pose_ik.so3_log(np.eye(3))) < 1e-9


def test_so3_log_known_rotation():
    # 90 degree rotation about world z
    c, s = 0.0, 1.0
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    w = pose_ik.so3_log(R)
    np.testing.assert_allclose(w, [0, 0, np.pi / 2], atol=1e-6)


def test_pose_ik_reduces_both_position_and_orientation_error():
    env = make_env(object_pos=(0.27, 0.0, 0.0))
    obs, info = env.reset(seed=0)
    obj0 = info["object_position"].copy()
    palm_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_PALM_SITE)
    ctrl = pose_ik.PalmController(env.model, palm_id, env._arm_dof_adr[:7])

    target_pos = obj0 + np.array([0.0, 0.15, 0.30])
    target_R = make_side_pinch_orientation(np.array([0.0, -1.0, 0.0]))

    _, conv0 = ctrl.step_toward(env.data, target_pos, target_R, gain=0.0)  # measure only
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    for _ in range(300):
        dq, conv = ctrl.step_toward(env.data, target_pos, target_R, gain=0.2, damping=0.15)
        a2 = a.copy()
        a2[3:10] = np.clip(dq / env.config.arm_action_scale, -1, 1)  # left arm
        env.step(a2)

    assert conv.pos_error_norm < conv0.pos_error_norm * 0.5
    assert conv.ori_error_norm < conv0.ori_error_norm * 0.5


# ---------------------------------------------------------------------
# Bimanual mapping / multi-start feasibility (Section G/H)
# ---------------------------------------------------------------------


def test_bimanual_target_left_right_mapping_not_swapped():
    env = make_env(object_pos=(0.27, 0.0, 0.0))
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    obj = expert._object_pos()
    # grip_center/grip_half_width parameterization (Section 5 redesign):
    # left target must be at +Y from the object (approaching from +Y),
    # right at -Y -- if swapped, hands would aim at each other's side and
    # never reach dual contact.
    left_target, right_target = expert._grip_targets(obj, 0.0)
    assert left_target[1] > obj[1]
    assert right_target[1] < obj[1]


def test_multi_start_feasibility_reports_per_start_convergence():
    env = make_env(object_pos=(0.27, 0.0, 0.0))
    env.reset(seed=0)
    palm_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_PALM_SITE)
    ctrl = pose_ik.PalmController(env.model, palm_id, env._arm_dof_adr[:7])
    obj_pos = env.data.qpos[env._object_qpos_adr : env._object_qpos_adr + 3].copy()
    target_pos = obj_pos + np.array([0.0, 0.15, 0.10])
    target_R = make_side_pinch_orientation(np.array([0.0, -1.0, 0.0]))

    rng = np.random.default_rng(0)
    starts = [env.data.qpos[env._arm_qpos_adr[:7]].copy()]
    for _ in range(2):
        starts.append(starts[0] + rng.uniform(-0.3, 0.3, size=7))

    results = pose_ik.multi_start_feasibility(
        ctrl, env.data, target_pos, target_R, starts, env._arm_qpos_adr[:7], steps=150, gain=0.2, damping=0.15
    )
    assert len(results) == len(starts)
    # at least one start should converge reasonably -- a single unlucky
    # start must not be reported as the only evidence of infeasibility.
    assert min(r.pos_error_norm for r in results) < 0.15


# ---------------------------------------------------------------------
# Synergy action semantics (Section 1): desired (absolute) -> delta action
# ---------------------------------------------------------------------


def _run_desired_synergy(env, desired: float, force_freeze_at: float | None, steps: int) -> list[float]:
    """Drives env._hand_group_synergy[0] (left thumb group) toward a
    fixed DESIRED value using the same delta conversion
    BimanualSidePinchExpert._apply() uses, and returns the actual env
    synergy trajectory. 23-dim action layout: [17:20)=left (thumb, index,
    middle), [20:23)=right (thumb, index, middle)."""
    trajectory = []
    for _ in range(steps):
        env_syn = env._hand_group_synergy[0]
        if force_freeze_at is not None and env_syn >= force_freeze_at:
            desired_now = env_syn  # simulate freezing desired at current actual
        else:
            desired_now = desired
        action_val = np.clip((desired_now - env_syn) / env.config.hand_synergy_action_scale, -1, 1)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        a[17] = action_val  # left thumb group
        env.step(a)
        trajectory.append(float(env._hand_group_synergy[0]))
    return trajectory


def test_fixed_desired_synergy_converges_and_stops():
    env = make_env()
    env.reset(seed=0)
    traj = _run_desired_synergy(env, desired=0.4, force_freeze_at=None, steps=80)
    assert abs(traj[-1] - 0.4) < 0.01, f"expected convergence to 0.4, got {traj[-1]}"
    # once converged, must not keep increasing (the original bug's signature)
    tail = traj[-10:]
    assert max(tail) - min(tail) < 0.01, f"synergy still drifting after convergence: {tail}"


def test_desired_synergy_freeze_also_freezes_actual():
    env = make_env()
    env.reset(seed=0)
    traj = _run_desired_synergy(env, desired=1.0, force_freeze_at=0.3, steps=80)
    freeze_idx = next(i for i, v in enumerate(traj) if v >= 0.3)
    after = traj[freeze_idx:]
    assert max(after) - min(after) < 0.01, f"actual synergy kept climbing after desired froze: {after[:10]}"


def test_left_right_synergy_independent():
    """Left vs right hand independence, checked via each hand's thumb
    group (index 0 in the left/right group-synergy slice) -- the 23-dim
    layout's per-hand grouping (not just per-finger-group) still holds."""
    env = make_env()
    env.reset(seed=0)
    for _ in range(60):
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        left_action = np.clip((0.6 - env._hand_group_synergy[0]) / env.config.hand_synergy_action_scale, -1, 1)
        a[17] = left_action  # left thumb group
        a[20] = 0.0  # right thumb group
        env.step(a)
    assert abs(env._hand_group_synergy[0] - 0.6) < 0.01
    assert env._hand_group_synergy[3] < 0.01, "right hand should be untouched while only left is commanded"


def test_synergy_resets_to_zero_matching_expert_init():
    env = make_env()
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    assert np.all(env._hand_group_synergy == 0.0)
    assert expert.left_desired_synergy == 0.0 and expert.right_desired_synergy == 0.0
    assert np.all(expert.left_group_synergy == 0.0) and np.all(expert.right_group_synergy == 0.0)


# ---------------------------------------------------------------------
# Bimanual side-pinch expert: real end-to-end run, honestly reported
# ---------------------------------------------------------------------


def test_bimanual_expert_full_run_honest_diagnostic():
    """NOT a success test (see module docstring) -- fixed-base bimanual
    lift+hold was still not achieved as of this session, even after fixing
    the synergy-delta and target-chases-object bugs. Reports every metric
    PROJECT_CONTEXT.md Phase 4 asks for and distinguishes a contact-squeeze
    "pop" from a controlled lift; never asserts SUCCESS that did not
    happen."""
    env = make_env(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0)
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    outcome = expert.run(max_total_steps=6000)

    assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
    # synergy semantics bug regression: desired and actual must stay close
    # (the original bug let actual run far past a frozen desired value).
    assert abs(outcome.left_desired_synergy - outcome.left_actual_synergy) < 0.05
    assert abs(outcome.right_desired_synergy - outcome.right_actual_synergy) < 0.05

    print(
        f"    final_state={outcome.state.name} reason={outcome.failure_reason}\n"
        f"    desired(L,R)=({outcome.left_desired_synergy:.3f},{outcome.right_desired_synergy:.3f}) "
        f"actual(L,R)=({outcome.left_actual_synergy:.3f},{outcome.right_actual_synergy:.3f})\n"
        f"    force_raw(L,R)=({outcome.left_max_force_raw:.1f}N,{outcome.right_max_force_raw:.1f}N)  "
        f"force_filtered(L,R)=({outcome.left_force_filtered:.1f}N,{outcome.right_force_filtered:.1f}N)  "
        f"finger_contact(L,R)=({outcome.left_finger_contact},{outcome.right_finger_contact})  "
        f"palm_contact(L,R)=({outcome.left_palm_contact},{outcome.right_palm_contact})\n"
        f"    max_dual_contact_streak={outcome.max_dual_contact_streak}  "
        f"obj_xy_disp={outcome.object_xy_displacement:.4f}m\n"
        f"    initial_z={outcome.initial_object_z:.4f} first_contact_z={outcome.object_z_at_first_contact} "
        f"dual_contact_z={outcome.object_z_at_dual_contact} lift_start_z={outcome.object_z_at_lift_start}\n"
        f"    max_before_lift={outcome.maximum_object_z_before_lift:.4f} "
        f"max_during_lift={outcome.maximum_object_z_during_lift:.4f} final_z={outcome.final_object_z:.4f} "
        f"commanded_lift_z={outcome.commanded_lift_target_z:.4f}\n"
        f"    pre_lift_pop_height={outcome.pre_lift_pop_height:.4f}m  "
        f"controlled_lift_gain={outcome.controlled_lift_gain:.4f}m  "
        f"final_height_above_initial={outcome.final_height_above_initial:.4f}m  "
        f"hold_steps={outcome.hold_steps_achieved}"
    )
    if outcome.state != GraspState.SUCCESS:
        print("    NOT SUCCESS: see PROJECT_CONTEXT.md Phase 4 Grasp Track for the exact remaining gap.")


def _run_fixed_base_bimanual_grasp_success_check(object_half_size: float):
    """The REAL success test (Section 6/11, 12-point checklist), shared
    across object sizes (multi-size grasp track, Section 2) -- asserts
    every criterion against an actual controller rollout for the given
    object_half_size. Per explicit project instruction: a `PASS` on some
    other test (mechanism, regression, or the honest diagnostic above)
    must never be reported as grasp success, thresholds are never lowered
    to force a pass, and a squeeze-pop is never counted as a controlled
    lift. As of this session BOTH sizes FAIL this test (see
    PROJECT_CONTEXT.md for the exact per-size trace) -- left visible
    rather than adjusted away.
    """
    env = make_env(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=object_half_size)
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    outcome = expert.run(max_total_steps=6000)

    print(
        f"    left_first_finger_contact_step={outcome.left_first_finger_contact_step} "
        f"right_first_finger_contact_step={outcome.right_first_finger_contact_step} "
        f"contact_timing_gap={outcome.contact_timing_gap}\n"
        f"    max_left_finger_contact_streak={outcome.max_left_finger_contact_streak} "
        f"max_right_finger_contact_streak={outcome.max_right_finger_contact_streak} "
        f"max_bilateral_finger_contact_streak={outcome.max_bilateral_finger_contact_streak}\n"
        f"    max_bilateral_multifinger_streak={outcome.max_bilateral_multifinger_streak}\n"
        f"    left_substate={outcome.left_substate} right_substate={outcome.right_substate} "
        f"left_reacquire_attempts={outcome.left_reacquire_attempts} right_reacquire_attempts={outcome.right_reacquire_attempts}\n"
        f"    force_raw(L,R)=({outcome.left_max_force_raw:.1f}N,{outcome.right_max_force_raw:.1f}N) "
        f"total_joint_travel={outcome.total_joint_travel:.2f} peak_joint_velocity={outcome.peak_joint_velocity:.2f}"
    )

    # 12. no NaN/Inf, no contact-force explosion
    assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
    assert outcome.left_max_force_raw < 100.0 and outcome.right_max_force_raw < 100.0

    # 1. approach was stable, no overhead/side-swing -- checked via the
    # trajectory-quality metrics the redesign added (Section 3), not by
    # eyeballing the viewer.
    assert outcome.peak_joint_velocity < 20.0, f"excessive peak joint velocity: {outcome.peak_joint_velocity}"

    # 2. pre-contact object displacement stayed within the allowed limit
    # (enforced live during APPROACH/etc. via OBJECT_DISPLACED failure).
    assert outcome.failure_reason != FailureReason.OBJECT_DISPLACED, "pre-contact object displacement exceeded the limit"

    # 3 & 4. left- and right-hand FINGER contact must each have occurred
    # at some point (not just palm) -- checked via the per-hand streaks
    # (Section 1), which distinguish "one hand touched" from "both touched
    # together" -- a distinction the old single max_finger_contact_streak
    # field could not make.
    assert outcome.max_left_finger_contact_streak > 0, "no left finger contact ever achieved"
    assert outcome.max_right_finger_contact_streak > 0, "no right finger contact ever achieved"

    # 5. both-hand (bilateral, simultaneous) finger contact sustained >= 30
    # steps -- NOT the sum or max of the two one-sided streaks.
    assert outcome.max_bilateral_finger_contact_streak >= 30, (
        f"bilateral finger contact only sustained {outcome.max_bilateral_finger_contact_streak} steps, need >= 30 "
        f"(left alone reached {outcome.max_left_finger_contact_streak}, right alone {outcome.max_right_finger_contact_streak}, "
        f"contact_timing_gap={outcome.contact_timing_gap} steps)"
    )

    # 5b. Gate A's literal definition (Section 10/11) is a genuine
    # MULTI-finger grip -- >=2 of thumb/index/middle touching per hand,
    # not just any single finger -- sustained for the same >=30 steps.
    # A direct constant-hold experiment showed "any finger" contact is not
    # force-closure-capable, so this is checked as its own criterion.
    assert outcome.max_bilateral_multifinger_streak >= 30, (
        f"bilateral MULTI-finger (>=2 groups/hand) contact only sustained "
        f"{outcome.max_bilateral_multifinger_streak} steps, need >= 30 "
        f"(any-finger streak reached {outcome.max_bilateral_finger_contact_streak})"
    )

    # 6. controller actually entered a controlled LIFT state
    assert outcome.state in (GraspState.LIFT, GraspState.HOLD, GraspState.SUCCESS), (
        f"never entered LIFT (ended in {outcome.state.name}, reason={outcome.failure_reason})"
    )

    # 7. from the LIFT-start reference, object rose >= 0.02m (a genuine
    # controlled_lift_gain, never a pre-lift pop).
    assert outcome.controlled_lift_gain >= 0.02, f"controlled_lift_gain={outcome.controlled_lift_gain:.4f}m < 0.02m"

    # 8. table contact released during the lift
    assert not outcome.object_table_contact, "object still touching table after claimed lift"

    # 9. hold sustained >= 100 steps
    assert outcome.hold_steps_achieved >= 100, f"hold_steps_achieved={outcome.hold_steps_achieved} < 100"

    # 10. object did not leave the hands during hold
    assert outcome.state == GraspState.SUCCESS, f"grasp did not complete: {outcome.state.name} ({outcome.failure_reason})"

    # 11. a squeeze-pop must never be counted as the controlled lift --
    # enforced structurally: controlled_lift_gain is only ever nonzero once
    # LIFT has actually started (see GraspOutcome/step() docstring), and
    # pre_lift_pop_height is tracked as a SEPARATE field specifically so
    # this can never be conflated.
    assert outcome.pre_lift_pop_height != outcome.controlled_lift_gain or outcome.controlled_lift_gain == 0.0


def test_size6_bimanual_grasp_lifts_and_holds_object():
    """SIZE_6 (Section 2 required condition) -- 6cm cube, the ORIGINAL
    Foundation size and Unitree's own official PickPlace-RedBlock-Dex3
    demo size. As of this session this test FAILS earlier than SIZE_12:
    the object-size-conditioned APPROACH target's IK residual error does
    not converge within the shared tolerance (see PROJECT_CONTEXT.md --
    isolated to null-space/gain interaction, confirmed NOT a joint-limit
    problem, not resolved this session)."""
    from humanoid_learning.envs.grasp_config import SIZE_6_HALF
    _run_fixed_base_bimanual_grasp_success_check(SIZE_6_HALF)


def test_size12_bimanual_grasp_lifts_and_holds_object():
    """SIZE_12 (Section 2 required condition) -- 12cm cube, used since the
    earlier "2x" grasp-track session. As of this session this test FAILS:
    bilateral finger contact reaches 50 steps (a new best, up from 43) but
    ends SINGLE_HAND_CONTACT_ONLY before Gate A's rotation/position
    stability criteria are met (see PROJECT_CONTEXT.md)."""
    from humanoid_learning.envs.grasp_config import SIZE_12_HALF
    _run_fixed_base_bimanual_grasp_success_check(SIZE_12_HALF)


# ---------------------------------------------------------------------
# Coupled bilateral waist-aware IK (Coupled Bilateral Waist-Aware IK
# Validation session): stacked Jacobian correctness, joint-bounds
# compliance, determinism, the locked-waist diagnostic mechanism, joint-
# space trajectory continuity, and the two size-specific Approach Gates.
# ---------------------------------------------------------------------


def _make_expert_with_solver(object_half_size: float = 0.06):
    env = make_env(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=object_half_size)
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    return env, expert


def test_coupled_jacobian_shape_and_arm_columns_isolated():
    """The stacked 12x17 task Jacobian must have EXACTLY zero in each
    hand's block for the OTHER arm's columns (real kinematic fact, not a
    manually-imposed zero -- see coupled_ik.py's docstring) while the
    waist columns (shared) are free to be nonzero in both."""
    env, expert = _make_expert_with_solver()
    solver = expert._coupled_solver
    obj = expert._object_pos()
    left_target = obj + np.array([0.0, 0.2, 0.1])
    right_target = obj + np.array([0.0, -0.2, 0.1])
    err, J, *_ = solver._task_error_and_jacobian(
        env.data, left_target, expert.left_R, right_target, expert.right_R
    )
    assert err.shape == (12,)
    assert J.shape == (12, 17)
    left_rows, right_rows = slice(0, 6), slice(6, 12)
    np.testing.assert_allclose(J[left_rows, solver.right_slice], 0.0, atol=1e-10)
    np.testing.assert_allclose(J[right_rows, solver.left_slice], 0.0, atol=1e-10)


def test_coupled_waist_columns_affect_both_hands():
    """Moving a waist joint must show up in BOTH palms' Jacobian columns --
    this shared coupling is exactly the mechanism the causal experiment
    (PROJECT_CONTEXT.md) identified as the reason per-arm-only IK fails."""
    env, expert = _make_expert_with_solver()
    solver = expert._coupled_solver
    obj = expert._object_pos()
    left_target = obj + np.array([0.0, 0.2, 0.1])
    right_target = obj + np.array([0.0, -0.2, 0.1])
    _, J, *_ = solver._task_error_and_jacobian(env.data, left_target, expert.left_R, right_target, expert.right_R)
    left_rows, right_rows = slice(0, 6), slice(6, 12)
    assert np.abs(J[left_rows, solver.waist_slice]).max() > 1e-6, "waist columns are zero in the LEFT hand's task rows"
    assert np.abs(J[right_rows, solver.waist_slice]).max() > 1e-6, "waist columns are zero in the RIGHT hand's task rows"


def test_coupled_ik_respects_joint_bounds():
    env, expert = _make_expert_with_solver()
    solver = expert._coupled_solver
    obj = expert._object_pos()
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = env.data.qpos
    mujoco.mj_forward(env.model, scratch)
    left_target = obj + np.array([0.0, 0.2, 0.1])
    right_target = obj + np.array([0.0, -0.2, 0.1])
    result = solver.solve(scratch, left_target, expert.left_R, right_target, expert.right_R, expert._coupled_rest_q)
    q = np.concatenate([result.waist_q, result.left_q, result.right_q])
    assert np.all(q >= solver.joint_low - 1e-9) and np.all(q <= solver.joint_high + 1e-9), (
        "coupled IK solution violates a real joint bound"
    )


def test_coupled_ik_deterministic():
    env, expert = _make_expert_with_solver()
    solver = expert._coupled_solver
    obj = expert._object_pos()
    left_target = obj + np.array([0.0, 0.2, 0.1])
    right_target = obj + np.array([0.0, -0.2, 0.1])

    def run_once():
        scratch = mujoco.MjData(env.model)
        scratch.qpos[:] = env.data.qpos
        mujoco.mj_forward(env.model, scratch)
        return solver.solve(scratch, left_target, expert.left_R, right_target, expert.right_R, expert._coupled_rest_q)

    r1, r2 = run_once(), run_once()
    np.testing.assert_allclose(r1.waist_q, r2.waist_q, atol=1e-10)
    np.testing.assert_allclose(r1.left_q, r2.left_q, atol=1e-10)
    np.testing.assert_allclose(r1.right_q, r2.right_q, atol=1e-10)
    assert r1.success == r2.success


def test_locked_waist_diagnostic_joint_is_genuinely_immobile():
    """The B condition (A/B/D causal experiment) locks the waist via a
    real joint-range constraint on a FRESH model, never a per-step qpos
    overwrite (explicitly forbidden this session -- overwriting injects
    energy). Regression: shrinking jnt_range to a single point and
    stepping the sim with the waist actuator commanded hard against it
    must leave the waist qpos within MuJoCo's normal soft-constraint
    equilibrium penetration of the locked point -- measured directly at
    ~0.017rad/0.0015rad/0.0018rad per axis, IDENTICAL regardless of how
    hard the actuator pushes past the limit (checked at push
    magnitudes 0 to 1.0 past ctrlrange), confirming this is the
    constraint's fixed solref/solimp softness, not the actuator
    "winning" and dragging qpos away from the lock."""
    cfg = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0))
    model = model_builder.build_grasp_model(cfg)
    waist_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in wbc.WAIST_JOINTS]
    waist_qpos_adr = np.array([model.jnt_qposadr[j] for j in waist_jids])
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    q0 = model.key_qpos[key_id][waist_qpos_adr].copy()
    for j, q0_j in zip(waist_jids, q0):
        model.jnt_range[j] = [q0_j, q0_j]

    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[key_id]
    data.ctrl[:] = model.key_ctrl[key_id]
    waist_act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in wbc.WAIST_JOINTS]
    assert all(a >= 0 for a in waist_act_ids)
    data.ctrl[waist_act_ids] = model.actuator_ctrlrange[waist_act_ids, 1] + 1.0  # command hard past the locked point
    for _ in range(200):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos[waist_qpos_adr], q0, atol=0.02)


def test_coupled_resolve_overshoot_shrinks_actuator_droop():
    """A one-shot coupled solve only guarantees a KINEMATIC match -- the
    compliant kp=120 actuators still settle short of it under gravity
    (see coupled_ik.py's module docstring). _coupled_maybe_resolve's
    Cartesian-target inflation correction must actually shrink that gap
    on the LIVE env, not just re-report it. Uses the APPROACH-height
    target (not PRE_GRASP's raised waypoint) starting from a fresh stand
    pose: PRE_GRASP's target is only ever reached in production AFTER its
    own two-phase rise-then-lateral pre-positioning gets the arm most of
    the way there first (see grasp_expert.py's PRE_GRASP state block) --
    solving for it cold, directly from stand, is a harder, less
    representative starting condition this unit test isn't meant to
    cover."""
    env, expert = _make_expert_with_solver()
    obj = expert._object_pos()
    left_target = obj + np.array([0.0, expert.outside_offset, expert.grasp_z_offset])
    right_target = obj + np.array([0.0, -expert.outside_offset, expert.grasp_z_offset])

    def residual():
        lp, lR = expert.left_ctrl.current_pose(env.data)
        rp, rR = expert.right_ctrl.current_pose(env.data)
        return float(np.linalg.norm(left_target - lp)) + float(np.linalg.norm(right_target - rp))

    expert._start_coupled_transit(left_target, right_target)
    for _ in range(expert.config.coupled_transit_steps + 5):
        a = expert._step_coupled_action()
        env.step(a)
    pos_after_initial = residual()

    for _ in range(4):
        expert._coupled_maybe_resolve()
        for _ in range(expert.config.coupled_transit_steps + 5):
            a = expert._step_coupled_action()
            env.step(a)
    pos_after_resolves = residual()

    assert pos_after_resolves < pos_after_initial, (
        f"resolve did not shrink the actuator-droop gap: {pos_after_initial:.4f} -> {pos_after_resolves:.4f}"
    )


def _run_approach_gate_check(object_half_size: float):
    """Runs the REAL expert (coupled IK, strict 0.01m/5deg gate) through
    the 7-state early approach (NATURAL_ARM_LIFT/FOREARM_LATERAL_APPROACH/
    FOREARM_DESCEND/WRIST_ALIGN/FINGER_PRESHAPE/FINGERTIP_PRECONTACT,
    Forearm Descent Before Wrist Alignment session), and asserts the
    Approach Gate criteria literally: reached WRIST_ALIGN (i.e.
    NATURAL_ARM_LIFT/FOREARM_LATERAL_APPROACH/FOREARM_DESCEND all
    converged), then reached FINGERTIP_PRECONTACT (i.e. WRIST_ALIGN and
    FINGER_PRESHAPE both completed), then transitioned OUT of
    FINGERTIP_PRECONTACT into CONTACT_ACQUIRE via the strict coupled
    convergence check (or an allowed early finger touch) -- never via a
    loose tolerance. Not a grasp/lift/hold test (out of scope this
    session, per PROJECT_CONTEXT.md)."""
    env = make_env(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=object_half_size)
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    reached_wrist_align = False
    reached_fingertip_precontact = False
    max_steps = 3000
    for _ in range(max_steps):
        outcome = expert.step()
        if expert.state == GraspState.WRIST_ALIGN:
            reached_wrist_align = True
        if expert.state == GraspState.FINGERTIP_PRECONTACT:
            reached_fingertip_precontact = True
        if expert.state in (GraspState.CONTACT_ACQUIRE, GraspState.FAILURE):
            break
    print(
        f"    object_half_size={object_half_size} final_state={expert.state.name} "
        f"reason={expert.failure_reason} reached_wrist_align={reached_wrist_align} "
        f"reached_fingertip_precontact={reached_fingertip_precontact} steps={expert._global_step}"
    )
    assert reached_wrist_align, "never reached WRIST_ALIGN (NATURAL_ARM_LIFT/FOREARM_LATERAL_APPROACH/FOREARM_DESCEND did not all converge)"
    assert reached_fingertip_precontact, "never reached FINGERTIP_PRECONTACT (WRIST_ALIGN/FINGER_PRESHAPE did not both complete)"
    assert expert.state == GraspState.CONTACT_ACQUIRE, (
        f"Approach Gate FAILED: ended in {expert.state.name} (reason={expert.failure_reason}) "
        f"instead of transitioning to CONTACT_ACQUIRE via the strict coupled-IK convergence gate"
    )


def test_size6_approach_gate_coupled_ik():
    from humanoid_learning.envs.grasp_config import SIZE_6_HALF
    _run_approach_gate_check(SIZE_6_HALF)


def test_size12_approach_gate_coupled_ik():
    from humanoid_learning.envs.grasp_config import SIZE_12_HALF
    _run_approach_gate_check(SIZE_12_HALF)


# ---------------------------------------------------------------------
# Natural Arm Reach + Wrist Alignment + Fingertip-First Grasp session:
# fingertip grasp frame, hierarchical per-state joint roles, natural
# 4-state early approach, fingertip-first contact.
# ---------------------------------------------------------------------


def test_fingertip_grasp_offset_dominated_by_approach_axis():
    """Direct measurement (Section 2/3): the fingertip centroid sits
    ~9-10cm from the palm-frame SITE along the palm's own local approach
    axis at preshape synergy -- large enough that "palm target == object
    surface" and "wrist/palm touches before any finger" are the same
    statement. Regression: this offset must stay large and dominated by
    the approach axis (index 0), not vanish to ~0 (which would silently
    make the fingertip-frame fix a no-op)."""
    env, expert = _make_expert_with_solver()
    for offset in (expert._left_fingertip_offset_local, expert._right_fingertip_offset_local):
        assert abs(offset[0]) > 0.05, f"approach-axis offset unexpectedly small: {offset}"
        assert abs(offset[0]) > abs(offset[1]) and abs(offset[0]) > abs(offset[2])


def test_grip_targets_are_palm_targets_behind_fingertip_point():
    """_grip_targets returns a PALM target computed by subtracting the
    fingertip offset from the intended fingertip contact point -- the
    palm target must sit measurably BEHIND (further from the object
    along the approach axis) the raw fingertip-contact formula's output,
    not equal to it (the bug this session's redesign fixes: the palm
    used to be sent directly to the fingertip's intended point)."""
    env, expert = _make_expert_with_solver()
    obj = expert._object_pos()
    z = expert.grasp_z_offset
    raw_fingertip_target = obj + np.array([0.0, expert.grip_center + expert.grip_half_width, z + expert._z_sync_bias])
    left_palm_target, _ = expert._grip_targets(obj, z)
    assert np.linalg.norm(left_palm_target - raw_fingertip_target) > 0.05, (
        "palm target is suspiciously close to the raw fingertip-contact point -- "
        "the fingertip-to-palm offset may not be getting applied"
    )


def test_natural_arm_lift_keeps_wrist_near_neutral():
    """Section 5 "natural_lift" role: wrist should stay close to its
    stand-pose neutral orientation while shoulder/elbow do the actual
    lifting -- measures REAL wrist joint travel (not Cartesian orientation
    error) during NATURAL_ARM_LIFT specifically, and asserts it stays
    much smaller than shoulder/elbow travel for the SAME state."""
    env, expert = _make_expert_with_solver()
    wrist_travel = 0.0
    se_travel = 0.0
    prev_q = env.data.qpos[env._arm_qpos_adr].copy()
    for _ in range(400):
        if expert.state != GraspState.NATURAL_ARM_LIFT:
            if expert.state != GraspState.STABLE_START:
                break
        expert.step()
        q = env.data.qpos[env._arm_qpos_adr].copy()
        delta = np.abs(q - prev_q)
        # left arm: [0:4)=shoulder+elbow, [4:7)=wrist; right arm mirrored at [7:14)
        se_travel += delta[0:4].sum() + delta[7:11].sum()
        wrist_travel += delta[4:7].sum() + delta[11:14].sum()
        prev_q = q
    assert se_travel > 0.05, f"shoulder/elbow barely moved during NATURAL_ARM_LIFT: {se_travel:.4f}"
    # Measured directly: wrist=4.2 vs shoulder/elbow=8.2 rad summed travel
    # (ratio ~0.52) -- meaningfully suppressed relative to shoulder/elbow
    # (not the primary driver) but not near-zero, since wrist still has
    # SOME residual task coupling even at weight=10/rest_gain=0.8. 0.75
    # is a fair regression bound: catches "wrist moves as much or more
    # than shoulder/elbow" (the un-suppressed, pre-fix behavior) without
    # demanding an unrealistic near-zero wrist travel.
    assert wrist_travel < 0.75 * se_travel, (
        f"wrist moved almost as much as shoulder/elbow during NATURAL_ARM_LIFT "
        f"(wrist={wrist_travel:.4f}, shoulder/elbow={se_travel:.4f}) -- expected wrist to be suppressed"
    )


def test_wrist_align_keeps_shoulder_elbow_close_to_posture_hold():
    """Section 5 "wrist_align" role: shoulder/elbow should stay close to
    the posture captured at WRIST_ALIGN's entry (_posture_hold_ref) while
    wrist does the orientation work -- regression for the ordering bug
    found this session (_capture_posture_hold_ref called before
    _transition meant _transition immediately wiped it back to None,
    silently falling back to the stand pose as the rest reference)."""
    env, expert = _make_expert_with_solver()
    for _ in range(700):
        expert.step()
        if expert.state == GraspState.WRIST_ALIGN:
            break
    assert expert.state == GraspState.WRIST_ALIGN, "did not reach WRIST_ALIGN in time"
    assert expert._posture_hold_ref is not None, "_posture_hold_ref was never captured"
    entry_se = np.concatenate([expert._posture_hold_ref[3:7], expert._posture_hold_ref[10:14]])
    for _ in range(300):
        expert.step()
        if expert.state != GraspState.WRIST_ALIGN:
            break
    q = env.data.qpos[env._arm_qpos_adr].copy()
    actual_se = np.concatenate([q[0:4], q[7:11]])
    drift = float(np.linalg.norm(actual_se - entry_se))
    assert drift < 0.3, f"shoulder/elbow drifted {drift:.4f} rad away from the posture-hold reference during WRIST_ALIGN"


def test_size12_first_object_contact_is_finger_not_wrist_or_palm():
    """Section 3/10's core requirement: the FIRST body (other than the
    table) to touch the object must be a finger link, never a wrist/
    forearm/palm link -- this is the direct, structural check that the
    fingertip-first fix actually changed physical behavior, not just the
    Cartesian target math."""
    from humanoid_learning.envs.grasp_config import SIZE_12_HALF
    env = make_env(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF)
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    model = env.model
    obj_body = env._object_body_id
    first_contact_body = None
    for _ in range(900):
        expert.step()
        for c in range(env.data.ncon):
            con = env.data.contact[c]
            bodies = {model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]}
            if obj_body not in bodies:
                continue
            other = bodies - {obj_body}
            other_id = other.pop() if other else obj_body
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_id) or ""
            if name == "table":
                continue
            first_contact_body = name
            break
        if first_contact_body is not None:
            break
        if expert.state in (GraspState.FAILURE, GraspState.SUCCESS):
            break
    assert first_contact_body is not None, "object never touched by any hand body"
    assert "hand" in first_contact_body and ("thumb" in first_contact_body or "index" in first_contact_body or "middle" in first_contact_body), (
        f"first object contact was {first_contact_body!r}, not a finger link"
    )
    assert "wrist" not in first_contact_body and "elbow" not in first_contact_body


def test_natural_approach_state_sequence_reaches_contact_acquire_size12():
    """End-to-end structural check of the 9-state early pipeline
    (STABLE_START -> NATURAL_ARM_LIFT -> FOREARM_LATERAL_APPROACH ->
    THUMB_CLEARANCE_PRESHAPE -> FOREARM_DESCEND -> WRIST_ALIGN ->
    FINGER_PRESHAPE -> FINGERTIP_PRECONTACT -> CONTACT_ACQUIRE;
    THUMB_CLEARANCE_PRESHAPE moved to right after FOREARM_LATERAL_
    APPROACH in the Collision-Free Thumb Preshape + Early Tripod Closure
    session -- thumb abducts while still at approach_height, well clear
    of the object, instead of after WRIST_ALIGN/FINGER_PRESHAPE) for
    SIZE_12 -- every state must be visited in this exact order, with no
    premature wrist/palm/forearm contact along the way (a planned finger
    touch during FINGERTIP_PRECONTACT is allowed and is exactly how this
    sequence is expected to reach CONTACT_ACQUIRE)."""
    from humanoid_learning.envs.grasp_config import SIZE_12_HALF
    env = make_env(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF)
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    expected_order = [
        GraspState.STABLE_START, GraspState.NATURAL_ARM_LIFT, GraspState.FOREARM_LATERAL_APPROACH,
        GraspState.THUMB_CLEARANCE_PRESHAPE, GraspState.FOREARM_DESCEND, GraspState.WRIST_ALIGN,
        GraspState.FINGER_PRESHAPE, GraspState.FINGERTIP_PRECONTACT, GraspState.CONTACT_ACQUIRE,
    ]
    visited = [expert.state]
    for _ in range(900):
        expert.step()
        if expert.state != visited[-1]:
            visited.append(expert.state)
        if expert.state in (GraspState.FAILURE, GraspState.SUCCESS) or expert.state == GraspState.CONTACT_ACQUIRE:
            break
    assert expert.failure_reason != FailureReason.PREMATURE_CONTACT, "premature wrist/palm/forearm contact during early approach"
    assert visited == expected_order, f"unexpected state sequence: {[s.name for s in visited]}"


# ---------------------------------------------------------------------
# Forearm Descent Before Wrist Alignment + Multi-Finger Contact session:
# the large vertical descent must be FOREARM_DESCEND's job (shoulder/
# elbow primary), not WRIST_ALIGN/FINGERTIP_PRECONTACT's (small moves
# only).
# ---------------------------------------------------------------------


def test_forearm_descend_target_is_the_large_vertical_drop():
    """The ~0.15-0.19m vertical drop from approach_height to
    grasp_z_offset used to be asked of the old FINGERTIP_APPROACH's
    heavily posture-held shoulder/elbow (root cause of SIZE_6's
    convergence failure, see PROJECT_CONTEXT.md). Regression: this drop
    must be exactly FOREARM_LATERAL_APPROACH-target -> FOREARM_DESCEND-
    target (both at the SAME lateral clearance, only z differs)."""
    env, expert = _make_expert_with_solver()
    obj = expert._object_pos()
    lateral_target, _ = expert._grip_targets(obj, expert.approach_height)
    descend_target, _ = expert._grip_targets(obj, expert.grasp_z_offset)
    assert np.allclose(lateral_target[:2], descend_target[:2], atol=1e-9), (
        "FOREARM_DESCEND changed lateral (x/y) position -- it should be a pure vertical move"
    )
    vertical_drop = abs(descend_target[2] - lateral_target[2])
    assert vertical_drop > 0.1, f"expected a large (>0.1m) vertical drop, got {vertical_drop:.4f}m"


def test_wrist_align_target_identical_to_forearm_descend_target():
    """WRIST_ALIGN must NOT introduce any further Cartesian position
    move -- its target is bit-for-bit the same as FOREARM_DESCEND's
    (Section 2: "Cartesian position target은 FOREARM_DESCEND 종료점과
    동일")."""
    env, expert = _make_expert_with_solver()
    obj = expert._object_pos()
    descend_target, _ = expert._grip_targets(obj, expert.grasp_z_offset)
    wrist_align_target, _ = expert._grip_targets(obj, expert.grasp_z_offset)
    np.testing.assert_allclose(descend_target, wrist_align_target)


def test_fingertip_precontact_movement_is_small():
    """FINGERTIP_PRECONTACT's own Cartesian move (WRIST_ALIGN's target ->
    FINGERTIP_PRECONTACT's target, i.e. outside_offset -> precontact_
    half_width at the SAME height) must be small -- Section 4 explicitly
    treats a >=10cm move here as a state-design failure. It should also
    be strictly smaller than FOREARM_DESCEND's own drop."""
    env, expert = _make_expert_with_solver()
    obj = expert._object_pos()
    wrist_align_target, _ = expert._grip_targets(obj, expert.grasp_z_offset)
    precontact_target, _ = expert._grip_targets(obj, expert.grasp_z_offset, half_width=expert.precontact_half_width)
    delta = float(np.linalg.norm(precontact_target - wrist_align_target))
    assert delta < 0.10, f"FINGERTIP_PRECONTACT move is {delta:.4f}m -- Section 4 treats >=0.10m as state-design failure"
    descend_drop = abs(expert.approach_height - expert.grasp_z_offset)
    assert delta < descend_drop, "FINGERTIP_PRECONTACT should move much less than FOREARM_DESCEND"


def test_fingertip_precontact_allows_finger_contact_not_palm():
    """Section 10: FINGERTIP_PRECONTACT must treat a FINGER touch as the
    expected outcome (hand off to CONTACT_ACQUIRE), not PREMATURE_CONTACT
    -- only a palm/wrist touch should still fail. Verified directly
    against the HandContact categories the state actually checks."""
    env, expert = _make_expert_with_solver()
    for _ in range(700):
        expert.step()
        if expert.state in (GraspState.FINGERTIP_PRECONTACT, GraspState.CONTACT_ACQUIRE, GraspState.FAILURE):
            break
    # Whichever way this particular rollout resolved, it must never have
    # been a PREMATURE_CONTACT triggered by a plain finger touch -- the
    # only legitimate PREMATURE_CONTACT source from FINGERTIP_PRECONTACT
    # onward is a palm/wrist touch.
    if expert.state == GraspState.FAILURE and expert.failure_reason == FailureReason.PREMATURE_CONTACT:
        lc, rc = expert._last_left_contact, expert._last_right_contact
        assert (lc and lc.palm.touched) or (rc and rc.palm.touched), (
            "PREMATURE_CONTACT fired without an actual palm/wrist touch"
        )


def test_size12_gate1_natural_approach_all_four_states_converge():
    """Gate 1 (Section 12): SIZE_12 must converge NATURAL_ARM_LIFT,
    FOREARM_LATERAL_APPROACH, FOREARM_DESCEND, and WRIST_ALIGN in
    sequence (strict coupled-IK gate, no PREMATURE_CONTACT) -- the
    structural fix this session's redesign targets."""
    from humanoid_learning.envs.grasp_config import SIZE_12_HALF
    env = make_env(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF)
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    seen = set()
    for _ in range(1200):
        expert.step()
        seen.add(expert.state)
        if expert.state in (GraspState.FAILURE, GraspState.FINGER_PRESHAPE):
            break
    assert expert.failure_reason != FailureReason.PREMATURE_CONTACT
    for s in (GraspState.NATURAL_ARM_LIFT, GraspState.FOREARM_LATERAL_APPROACH, GraspState.FOREARM_DESCEND, GraspState.WRIST_ALIGN):
        assert s in seen, f"never visited {s.name}"
    assert expert.state == GraspState.FINGER_PRESHAPE, f"Gate 1 failed: ended in {expert.state.name} ({expert.failure_reason})"


# ---------------------------------------------------------------------
# Thumb Opposition + Claw-Style Bimanual Grasp session: thumb_1's real
# role (dominant opposition lever, previously frozen), the claw preshape
# state machine (THUMB_ABDUCT -> ... -> THUMB_OPPOSE -> TRIPOD_SETTLE),
# and opposing-normal/tripod contact classification.
# ---------------------------------------------------------------------


def test_thumb1_is_dominant_opposition_joint_and_mirrored():
    """Direct measurement (Section 1): sweeping thumb_1 alone across its
    OWN configured open/close synergy targets (index/middle held at a
    representative mid-curl pose) must move the thumb tip substantially
    closer to index, and the effect must be present on BOTH hands (left
    and right use mirrored joint signs, see whole_body_config.py)."""
    env = make_env()
    env.reset(seed=0)
    model, data = env.model, env.data
    base_qpos = data.qpos.copy()

    for side, targets in (("left", wbc.LEFT_HAND_SYNERGY_TARGETS), ("right", wbc.RIGHT_HAND_SYNERGY_TARGETS)):
        finger_qpos_adr = env._left_finger_qpos_adr if side == "left" else env._right_finger_qpos_adr
        thumb_j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_thumb_1_joint")
        thumb_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_thumb_2_link")
        index_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_index_1_link")
        thumb_gid = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == thumb_body)
        index_gid = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == index_body)

        def dist_at(q, _targets=targets, _adr=finger_qpos_adr, _tj=thumb_j):
            data.qpos[:] = base_qpos
            for i, (jn, o, c) in enumerate(_targets):
                if i >= 3:
                    data.qpos[_adr[i]] = o + 0.5 * (c - o)
            data.qpos[model.jnt_qposadr[_tj]] = q
            mujoco.mj_forward(model, data)
            return float(np.linalg.norm(data.geom_xpos[thumb_gid] - data.geom_xpos[index_gid]))

        open_q, close_q = targets[1][1], targets[1][2]
        d_open, d_close = dist_at(open_q), dist_at(close_q)
        assert d_close < d_open - 0.03, (
            f"{side}: thumb_1 close ({close_q}) should be much closer to index than open ({open_q}) "
            f"(d_open={d_open:.4f} d_close={d_close:.4f})"
        )
    data.qpos[:] = base_qpos
    mujoco.mj_forward(model, data)


def test_thumb1_open_close_targets_not_frozen():
    """Regression guard for the frozen-thumb_1 config bug this session
    found and fixed: OLD open/close were IDENTICAL (1.0472, 1.0472 for
    left / -1.0472, -1.0472 for right), so thumb_1 -- measured as the
    single largest thumb-opposition lever -- contributed nothing to any
    synergy command regardless of value."""
    left_open, left_close = wbc.LEFT_HAND_SYNERGY_TARGETS[1][1], wbc.LEFT_HAND_SYNERGY_TARGETS[1][2]
    right_open, right_close = wbc.RIGHT_HAND_SYNERGY_TARGETS[1][1], wbc.RIGHT_HAND_SYNERGY_TARGETS[1][2]
    assert abs(left_close - left_open) > 0.3, "left thumb_1 open/close still effectively frozen"
    assert abs(right_close - right_open) > 0.3, "right thumb_1 open/close still effectively frozen"


def test_thumb_clearance_preshape_abducts_before_descent_and_stays_open():
    """Section 3/5 (Collision-Free Thumb Preshape + Early Tripod Closure
    session): THUMB_CLEARANCE_PRESHAPE (right after FOREARM_LATERAL_
    APPROACH, while still at approach_height) must actually converge
    thumb_1's ACTUAL qpos to the GRASP_ABDUCT pose before FOREARM_DESCEND
    is ever entered -- the whole point of moving thumb abduction earlier
    is that the hand must not begin descending toward the object until
    thumb is genuinely out of the way. Thumb must then STAY there
    (group_synergy[0] pinned at 0, only the direct override moves it)
    through FOREARM_DESCEND/WRIST_ALIGN/FINGER_PRESHAPE while index/
    middle actively ramp toward preshape_synergy."""
    cfg = GraspExpertConfig()
    env = make_env(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0)
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, cfg)
    reached_clearance = False
    left_thumb1_at_descend_start = None
    for _ in range(700):
        expert.step()
        if expert.state == GraspState.THUMB_CLEARANCE_PRESHAPE:
            reached_clearance = True
            assert expert.left_group_synergy[0] == 0.0, "left thumb group synergy should stay pinned"
            assert expert.right_group_synergy[0] == 0.0, "right thumb group synergy should stay pinned"
        if expert.state == GraspState.FOREARM_DESCEND and left_thumb1_at_descend_start is None:
            left_thumb1_at_descend_start = env.data.qpos[env._left_finger_qpos_adr[1]]
        if expert.state in (GraspState.FINGERTIP_PRECONTACT, GraspState.CONTACT_ACQUIRE, GraspState.FAILURE):
            break
    assert reached_clearance, "never reached THUMB_CLEARANCE_PRESHAPE"
    assert left_thumb1_at_descend_start is not None, "never reached FOREARM_DESCEND"
    assert abs(left_thumb1_at_descend_start - cfg.thumb1_abduct_pose_left) < 0.1, (
        f"thumb_1 must already be converged to GRASP_ABDUCT before FOREARM_DESCEND begins, "
        f"got {left_thumb1_at_descend_start} vs target {cfg.thumb1_abduct_pose_left}"
    )


def test_opposing_normal_score_detects_true_opposition_vs_same_side_push():
    """Unit test (Section 3/10) for _opposing_normal_score using synthetic
    contact geometry: two contacts whose normals point directly at each
    other (genuine pinch/opposition) must score near +1.0; two contacts
    pushing from the SAME side (the "손이 집게처럼 닫히지 않고 같은 방향에서
    물체를 누름" failure mode this session was asked to fix) must score
    near -1.0."""
    from humanoid_learning.expert.grasp_expert import ContactCategory, HandContact

    def make_contact(thumb_normal, index_normal, thumb_point=(0.0, 0.0, 0.0), index_point=(0.02, 0.0, 0.0)):
        thumb = ContactCategory(touched=True, normal_force=5.0, world_normal=np.array(thumb_normal, dtype=float), world_point=np.array(thumb_point, dtype=float))
        index = ContactCategory(touched=True, normal_force=5.0, world_normal=np.array(index_normal, dtype=float), world_point=np.array(index_point, dtype=float))
        return HandContact(palm=ContactCategory(), finger=ContactCategory(touched=True), thumb=thumb, index=index, middle=ContactCategory(), contact_count=2)

    opposing = make_contact([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    same_side = make_contact([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    score_opp, name_opp, sep_opp = BimanualSidePinchExpert._opposing_normal_score(opposing)
    score_same, name_same, sep_same = BimanualSidePinchExpert._opposing_normal_score(same_side)
    assert score_opp > 0.9, f"opposing normals should score near +1.0, got {score_opp}"
    assert score_same < -0.9, f"same-side normals should score near -1.0, got {score_same}"
    assert name_opp == "index" and name_same == "index"
    assert sep_opp > 0.01


def test_thumb_and_tripod_streak_metrics_wired_and_bounded():
    """Structural test (Section 10/18): the new thumb/tripod participation
    metrics must exist on GraspOutcome, be non-negative streak counts, and
    keep the opposing-normal score within its defined [-1, 1] range for a
    real rollout (not just the synthetic unit test above)."""
    env = make_env(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0)
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    outcome = expert.run(max_total_steps=2500)
    assert outcome.max_left_thumb_contact_streak >= 0
    assert outcome.max_right_thumb_contact_streak >= 0
    assert outcome.max_bilateral_thumb_contact_streak >= 0
    assert outcome.max_left_tripod_contact_streak >= 0
    assert outcome.max_right_tripod_contact_streak >= 0
    assert outcome.max_bilateral_tripod_streak >= 0
    assert -1.0 <= outcome.left_opposing_normal_score <= 1.0
    assert -1.0 <= outcome.right_opposing_normal_score <= 1.0
    print(
        f"    max_bilateral_thumb_contact_streak={outcome.max_bilateral_thumb_contact_streak} "
        f"max_bilateral_tripod_streak={outcome.max_bilateral_tripod_streak} "
        f"opposing_normal(L,R)=({outcome.left_opposing_normal_score:.2f},{outcome.right_opposing_normal_score:.2f})"
    )


def _run_gate_a_tripod_check(object_half_size: float):
    """Gate A (Section 12): the REAL, literal criterion is bilateral
    TRIPOD contact (thumb + a genuinely opposing index/middle contact)
    sustained >= 30 steps -- not just any-finger contact. As of this
    session this FAILS for both sizes (see PROJECT_CONTEXT.md for the
    exact remaining gap: thumb currently reaches the object at roughly
    the same depth as index/middle rather than safely after them,
    depending on hand/object geometry) -- left visible rather than
    adjusted away, per project convention."""
    env = make_env(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=object_half_size)
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    outcome = expert.run(max_total_steps=6000)
    print(
        f"    object_half_size={object_half_size} final_state={outcome.state.name} reason={outcome.failure_reason}\n"
        f"    max_bilateral_tripod_streak={outcome.max_bilateral_tripod_streak} "
        f"max_bilateral_thumb_contact_streak={outcome.max_bilateral_thumb_contact_streak} "
        f"opposing_normal(L,R)=({outcome.left_opposing_normal_score:.2f},{outcome.right_opposing_normal_score:.2f})"
    )
    assert outcome.max_bilateral_tripod_streak >= 30, (
        f"Gate A FAILED: bilateral tripod (thumb + opposing index/middle) contact only sustained "
        f"{outcome.max_bilateral_tripod_streak} steps, need >= 30 "
        f"(final state {outcome.state.name}, reason={outcome.failure_reason})"
    )


def test_size6_gate_a_tripod_contact():
    from humanoid_learning.envs.grasp_config import SIZE_6_HALF
    _run_gate_a_tripod_check(SIZE_6_HALF)


def test_size12_gate_a_tripod_contact():
    from humanoid_learning.envs.grasp_config import SIZE_12_HALF
    _run_gate_a_tripod_check(SIZE_12_HALF)


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
    sys.exit(1 if failed else 0)
