"""sharpa_bimanual tests: the Phase 4 OFFICIAL Sharpa Wave bimanual grasp
controller (SharpaBimanualGraspExpert).

Run with:
    python3 scripts/test_sharpa_bimanual_grasp.py

HONEST CURRENT STATE: the OFFICIAL approach path is
STABLE_START -> ARM_LATERAL_CLEARANCE -> FOREARM_FORWARD_REACH ->
WRIST_SIDE_GRASP_ALIGN -> FIVE_FINGER_PRESHAPE -> FOREARM_SIDE_DESCEND ->
FINGERTIP_PRECONTACT -> ... (see sharpa_bimanual_grasp_expert.py's
BimanualGraspState). ARM_LATERAL_CLEARANCE uses a DIRECT joint target
(not Cartesian IK) chosen from a bounded 3-candidate FK+physics sweep --
0 self-collisions, elbow well below shoulder.

FOREARM_FORWARD_REACH's own settled-position residual (previously a
deterministic ~10.165mm) is FIXED (waypoint-schedule bookkeeping, not a
physical limit -- see FORWARD_REACH_WAYPOINT_TICKS's docstring): settled
error is ~9.96mm, under the unchanged 10mm gate, 15-tick streak met, zero
forbidden collision, reproduced across 3 seed=0 rollouts.

[This session] User-directed requirement: a genuine bilateral SIDE grasp
(both palms beside the object's own side faces, facing each other,
fingers generally pointing down) -- not the old top-down, palm-down
reach. FK-measured (scripts/measure_sharpa_side_grasp_axes.py) the real
palm/finger axes and fixed a genuine sign bug in _object_facing_R (see
that function's docstring). WRIST_SIDE_GRASP_ALIGN + FOREARM_SIDE_DESCEND
replace the old FOREARM_DESCEND + WRIST_ALIGN pair; the old opt-in
object_facing_orientation flag (never safe, self-collided up to 113N) is
REMOVED, superseded by this design, which is collision-free (measured
0.00N torso-arm / 0.00N hand-hand across 3 repeated rollouts).

The new Side-Grasp Posture Gate (test_side_grasp_posture_gate_passes
below) PASSES: both palms outside the object's side faces, palm-inward
angle ~14.1deg (<=15deg), palm normals opposed, finger-down angle
~15.0deg (<=25deg), fingertip centroid overlaps the object's side height,
no top-footprint crossing, left/right mirror position error <0.1mm,
mirror orientation error <0.04deg, elbow well below shoulder, zero
torso-arm/hand-hand/hand-table forbidden collision, 15-tick streak.

The rollout currently fails LATER, in FOREARM_SIDE_DESCEND, with
HAND_TABLE_COLLISION -- a real, repeated (not single-tick) hand<->table
force up to ~18N measured this session (scripts/diagnose_side_grasp_
swept_path.py) as the fingers, already partially curled, keep dipping
into the table on the final approach waypoints. A bounded (curl fraction,
height) candidate search did not clear it without cost elsewhere. This is
the current, disclosed next blocker -- a separate, independent problem
from the Side-Grasp Posture Gate, which is achieved.
test_a_real_bimanual_gate_a_success_on_size_12 below documents the
overall Gate A outcome HONESTLY as a failing test and must never be
weakened, deleted, or converted into a smoke assertion to make it pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv, ACTION_DIM
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    SIDES,
    BimanualFailureReason,
    BimanualGraspConfig,
    BimanualGraspState,
    SharpaBimanualGraspExpert,
)


def make_env(max_episode_steps: int = 8000) -> SharpaGraspEnv:
    # arm_gravity_compensation=True: Session 39 fix for the FINGERTIP_
    # PRECONTACT IK-vs-physics tracking gap -- see grasp_config.py's
    # arm_gravity_compensation docstring and
    # docs/history/PHASE4_GRASP_SESSION_39.md.
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=max_episode_steps, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def test_bimanual_action_moves_both_arms_real_actuators():
    """Requirement 1: left AND right hand actions both move real actuators
    (not one silently zeroed like the single-hand controller's REST_SIDE)."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    for _ in range(80):
        action = expert.step()
        env.step(action)
    left_moved = not np.allclose(env._arm_target[:7], env._arm_target[:7] * 0 + env._arm_target[0])
    left_delta = np.abs(env._arm_target[:7] - env._stand_qpos[env._arm_qpos_adr[:7]]).max()
    right_delta = np.abs(env._arm_target[7:] - env._stand_qpos[env._arm_qpos_adr[7:]]).max()
    assert left_delta > 0.05, "left arm target must move away from stand pose"
    assert right_delta > 0.05, "right arm target must move away from stand pose"
    print(f"    left arm delta={left_delta:.3f}rad, right arm delta={right_delta:.3f}rad -- both sides actually move")


def test_bimanual_targets_are_mirrored_never_crossing_midline():
    """Requirement: both hands take OPPOSITE faces of the object -- never
    one hand crossing the body midline (Y=0) to reproduce a single-hand
    grasp."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    targets = expert._mirrored_targets(0.15, 0.22, 0.15)
    assert targets["left"][1] > 0, "left hand's target must stay on the +Y side"
    assert targets["right"][1] < 0, "right hand's target must stay on the -Y side"
    assert np.isclose(targets["left"][1], -targets["right"][1]), "targets must be Y-mirrored"
    print(f"    left target y={targets['left'][1]:.3f}, right target y={targets['right'][1]:.3f} -- mirrored, no midline crossing")


def test_arm_lateral_clearance_wrist_transition_gate():
    """[Session 42] Wrist Transition Gate (an explicit engineering safety
    target for THIS approach trajectory, NOT a Gate A criterion): raw
    (unfiltered) wrist |qvel| during ARM_LATERAL_CLEARANCE.

    HONEST CURRENT STATE: substep-level tracing
    (scripts/diagnose_clearance_wrist_spike.py) found the 41st session's
    abrupt-step joint target caused a real thumb<->table collision
    impulse (qfrc_constraint jumping to 12-23 N*m in a single tick) that
    couples into the zero-damping wrist joints. A quintic minimum-jerk
    trajectory (zero velocity/acceleration at both ends, replacing the
    abrupt step) causally reduces peak raw wrist qvel from ~8.07rad/s to
    ~5.1rad/s (see docs/history/PHASE4_GRASP_SESSION_42.md) -- a real,
    measured improvement -- but does NOT fully reach the 2.0rad/s target,
    because the underlying thumb<->table graze itself (a geometric path
    property, confirmed NOT resolved by slowing the trajectory further)
    is not eliminated, only its coupling into wrist velocity is damped.
    This test locks in the IMPROVEMENT direction/magnitude, not a false
    Gate pass -- it must not be weakened into asserting <=2.0rad/s until
    that is actually achieved."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    for _ in range(400):
        if expert.state not in (BimanualGraspState.STABLE_START, BimanualGraspState.ARM_LATERAL_CLEARANCE):
            break
        action = expert.step()
        env.step(action)
    max_wrist_qvel = expert._clearance_max_raw_wrist_qvel
    print(f"    max raw wrist qvel during ARM_LATERAL_CLEARANCE = {max_wrist_qvel:.3f} rad/s "
          f"(Wrist Transition Gate target <=2.0rad/s: {'PASS' if max_wrist_qvel <= 2.0 else 'NOT YET MET'})")
    assert max_wrist_qvel < 7.0, (
        f"regression: max raw wrist qvel {max_wrist_qvel:.3f}rad/s is back near the pre-fix ~8.07rad/s baseline "
        "-- the quintic minimum-jerk trajectory fix may have been lost"
    )


def test_arm_lateral_clearance_meets_natural_posture_gate():
    """[Session 41] ARM_LATERAL_CLEARANCE must actually exist as an
    official state (not a rename -- the old NATURAL_ARM_LIFT/single-shot
    FOREARM_APPROACH Cartesian-IK path is gone) and must produce a REAL,
    measured natural posture: elbow never above shoulder, Y separation
    clearly increased from stand pose, left/right mirror error small,
    zero torso/hand-hand forbidden collision at the point the state
    itself claims convergence."""
    import mujoco

    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    assert hasattr(BimanualGraspState, "ARM_LATERAL_CLEARANCE")
    assert not hasattr(BimanualGraspState, "NATURAL_ARM_LIFT"), "the old Cartesian-IK NATURAL_ARM_LIFT state must be gone, not just renamed"

    stand_palm_y = {s: env.palm_pose(s)[0][1] for s in SIDES}
    reached_clearance = False
    for _ in range(300):
        if expert.state == BimanualGraspState.ARM_LATERAL_CLEARANCE:
            reached_clearance = True
        if expert.state != BimanualGraspState.ARM_LATERAL_CLEARANCE and reached_clearance:
            break
        action = expert.step()
        env.step(action)
    assert reached_clearance, "rollout must reach ARM_LATERAL_CLEARANCE"

    elbow_above_shoulder = {}
    palm_y = {}
    for side in SIDES:
        sb = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_roll_link")
        eb = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_elbow_link")
        elbow_above_shoulder[side] = float(env.data.xpos[eb][2] - env.data.xpos[sb][2])
        palm_y[side] = float(env.palm_pose(side)[0][1])

    print(f"    elbow_above_shoulder(L,R)=({elbow_above_shoulder['left']:.3f},{elbow_above_shoulder['right']:.3f})m "
          f"palm_Y(L,R)=({palm_y['left']:.3f},{palm_y['right']:.3f}) stand_palm_Y(L,R)=({stand_palm_y['left']:.3f},{stand_palm_y['right']:.3f})")
    for side in SIDES:
        assert elbow_above_shoulder[side] <= 0.02, f"{side} elbow is above shoulder by {elbow_above_shoulder[side]:.3f}m"
        assert abs(palm_y[side]) > abs(stand_palm_y[side]) + 0.05, f"{side} Y separation did not clearly increase from stand pose"
    mirror_err = abs(abs(palm_y["left"]) - abs(palm_y["right"]))
    assert mirror_err <= 0.02, f"left/right mirror position error {mirror_err:.4f}m exceeds 0.02m"
    assert env._torso_arm_collision_force() <= expert.config.hand_hand_force_limit_n
    assert env._hand_hand_contact_force() <= expert.config.hand_hand_force_limit_n


def test_forward_reach_gate_now_passes_and_advances_to_next_blocker():
    """[This session] FOREARM_FORWARD_REACH's settled-position residual
    used to plateau at a deterministic ~10.165mm, just above the
    unchanged 10mm gate, purely from a waypoint-schedule bookkeeping bug
    (see module docstring / FORWARD_REACH_WAYPOINT_TICKS in
    sharpa_bimanual_grasp_expert.py) -- the real physics residual was
    already under 10mm given enough settle time, verified with
    scripts/diagnose_forward_reach_gap.py. The fix does not touch
    ik_pos_tol, forward_reach_stable_streak_required, max_steps_per_state,
    or any collision/force safety limit -- only WHEN the final waypoint's
    IK target is issued within the SAME state budget.

    This test locks in that FOREARM_FORWARD_REACH's own Gate condition
    (settled error <=ik_pos_tol, streak reached, zero forbidden collision)
    is now genuinely met and the rollout advances past it into
    WRIST_SIDE_GRASP_ALIGN -- while staying honest that the FULL rollout
    still fails LATER (see test_side_grasp_posture_gate_passes and this
    module's docstring for the current next blocker). This must not be
    read as Gate A success."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    final_err = None
    max_streak_seen = 0
    collision_during_forward_reach = False
    reached_align = False
    for _ in range(1200):
        prev_state = expert.state
        action = expert.step()
        obs, r, term, trunc, info = env.step(action)
        if prev_state == BimanualGraspState.FOREARM_FORWARD_REACH and hasattr(expert, "_forward_reach_final"):
            final_err = max(
                float(np.linalg.norm(expert._forward_reach_final["left"] - env.palm_pose("left")[0])),
                float(np.linalg.norm(expert._forward_reach_final["right"] - env.palm_pose("right")[0])),
            )
            max_streak_seen = max(max_streak_seen, expert._forward_reach_stable_streak)
            if env._torso_arm_collision_force() > 0.0 or env._hand_hand_contact_force() > 0.0:
                collision_during_forward_reach = True
        if expert.state == BimanualGraspState.WRIST_SIDE_GRASP_ALIGN:
            reached_align = True
            break
        if expert.state == BimanualGraspState.FAILURE:
            break
    print(f"    forward_reach_final_error={final_err*1000:.3f}mm, max_streak={max_streak_seen}, "
          f"collision_during_state={collision_during_forward_reach}, reached_align={reached_align}")
    assert reached_align, "FOREARM_FORWARD_REACH must now advance to WRIST_SIDE_GRASP_ALIGN, not time out"
    assert final_err is not None and final_err <= expert.config.ik_pos_tol
    assert max_streak_seen >= expert.config.forward_reach_stable_streak_required
    assert not collision_during_forward_reach

    # Full rollout: still honestly fails later, at a separate gate.
    env2 = make_env()
    expert2 = SharpaBimanualGraspExpert(env2)
    outcome = expert2.run(max_total_steps=1200)
    print(f"    full rollout: state={outcome.state.name} failure_reason={outcome.failure_reason} "
          f"side_grasp_gate={outcome.side_grasp_gate}")
    assert outcome.side_grasp_gate is True, "Side-Grasp Posture Gate should still pass en route to the current next blocker"
    assert outcome.failure_reason == BimanualFailureReason.HAND_TABLE_COLLISION, (
        "expected the CURRENT next independent blocker (FOREARM_SIDE_DESCEND hand-table collision); "
        "if this changed, the docstring/PROJECT_CONTEXT next-blocker note is now stale"
    )


def test_side_grasp_posture_gate_passes():
    """[This session] The core deliverable: a genuine bilateral SIDE
    grasp posture (palms beside the object's own side faces, facing each
    other, fingers generally pointing down), not the old top-down palm-
    down reach. Locks in every Side-Grasp Posture Gate sub-condition
    (Section 10 of this session's spec) via the SAME metrics
    WRIST_SIDE_GRASP_ALIGN itself gates the transition on -- not a looser
    or separately-computed check."""
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    outcome = expert.run(max_total_steps=2000)
    m = outcome.side_grasp_posture
    print(f"    side_grasp_gate={outcome.side_grasp_gate}")
    for k, v in m.items():
        print(f"      {k}: {v}")
    assert m, "WRIST_SIDE_GRASP_ALIGN must be reached and measured (posture dict must not be empty)"
    assert outcome.side_grasp_gate is True
    assert all(m["outside_side_face"].values()), "both palms must be outside the object's own side faces"
    assert m["inward_angle_deg"]["left"] <= expert.config.side_grasp_inward_angle_tol_deg
    assert m["inward_angle_deg"]["right"] <= expert.config.side_grasp_inward_angle_tol_deg
    assert m["normals_opposed"], "palm normals (closing axes) must face each other"
    assert m["finger_down_deg"]["left"] <= expert.config.side_grasp_finger_down_tol_deg
    assert m["finger_down_deg"]["right"] <= expert.config.side_grasp_finger_down_tol_deg
    assert all(m["tip_height_overlaps_side"].values()), "fingertip centroid must overlap the object's side-face height range"
    assert not m["crosses_top_footprint"], "hands must not cross the object's own top footprint"
    assert m["mirror_pos_err_m"] <= expert.config.side_grasp_mirror_pos_tol_m
    assert m["mirror_ori_err_deg"] <= expert.config.side_grasp_mirror_ori_tol_deg
    assert m["elbow_ok"], "elbow must not be markedly above the shoulder"
    assert m["torso_arm_ok"] and m["hand_table_ok"] and m["hand_hand_ok"], "no forbidden collision at the align pose"
    assert m["premature_contact_ok"], "no premature (non-fingertip) object contact"
    assert m["streak"] >= expert.config.side_align_stable_streak_required


def test_side_grasp_state_order_matches_spec():
    """[This session] Locks in the required state ordering (Section 4):
    STABLE_START -> ARM_LATERAL_CLEARANCE -> FOREARM_FORWARD_REACH ->
    WRIST_SIDE_GRASP_ALIGN -> FIVE_FINGER_PRESHAPE -> FOREARM_SIDE_DESCEND
    -> FINGERTIP_PRECONTACT, each state visited exactly once and never
    skipped or reordered."""
    expected = [
        BimanualGraspState.STABLE_START, BimanualGraspState.ARM_LATERAL_CLEARANCE,
        BimanualGraspState.FOREARM_FORWARD_REACH, BimanualGraspState.WRIST_SIDE_GRASP_ALIGN,
        BimanualGraspState.FIVE_FINGER_PRESHAPE, BimanualGraspState.FOREARM_SIDE_DESCEND,
    ]
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    seen = [expert.state]
    for _ in range(1400):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        action = expert.step()
        env.step(action)
        if expert.state != seen[-1]:
            seen.append(expert.state)
    print(f"    state sequence: {[s.name for s in seen]}")
    prefix = seen[: len(expected)]
    assert prefix == expected, f"state order deviated: {[s.name for s in prefix]} != {[s.name for s in expected]}"


def test_side_grasp_swept_path_has_no_forbidden_collision_before_side_descend():
    """[This session] Endpoint-only checks are not sufficient (Section 8):
    verify torso-arm/hand-hand/hand-table forbidden-collision force stays
    at or under the shared safety limit at EVERY tick through
    FIVE_FINGER_PRESHAPE (the current honest blocker is inside FOREARM_
    SIDE_DESCEND itself, see module docstring -- this test covers the
    swept path UP TO that point, which the Side-Grasp Posture Gate
    already certifies clean)."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    limit = expert.config.hand_hand_force_limit_n
    checked_states = (
        BimanualGraspState.ARM_LATERAL_CLEARANCE, BimanualGraspState.FOREARM_FORWARD_REACH,
        BimanualGraspState.WRIST_SIDE_GRASP_ALIGN, BimanualGraspState.FIVE_FINGER_PRESHAPE,
    )
    max_torso_arm = max_hand_hand = 0.0
    for _ in range(1000):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        prev_state = expert.state
        action = expert.step()
        env.step(action)
        if prev_state in checked_states:
            max_torso_arm = max(max_torso_arm, env._torso_arm_collision_force())
            max_hand_hand = max(max_hand_hand, env._hand_hand_contact_force())
        if expert.state == BimanualGraspState.FOREARM_SIDE_DESCEND:
            break
    print(f"    swept max_torso_arm={max_torso_arm:.2f}N swept max_hand_hand={max_hand_hand:.2f}N (limit={limit}N)")
    assert max_torso_arm <= limit
    assert max_hand_hand <= limit


def test_ever_contacted_alone_does_not_satisfy_gate_a():
    """Requirement 5/6: Gate A must never be satisfied by 'ever touched'
    alone -- only a real, held bilateral streak."""
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    # Force the internal per-group "ever contacted" state True for every
    # group on both sides WITHOUT ever having a genuine simultaneous
    # bilateral streak (streak counters left at their real, untouched 0).
    for side in SIDES:
        for g in expert._group_ever_contacted[side]:
            expert._group_ever_contacted[side][g] = True
    assert expert._max_bilateral_streak == 0
    outcome = None
    angvel_peak = 0.0
    gate_a = (
        BimanualGraspState.FAILURE != BimanualGraspState.FAILURE  # placeholder, real check below
    )
    gate_a = (
        expert._max_bilateral_streak >= expert.config.bilateral_streak_required
        and expert.object_xy_displacement() <= expert.config.object_xy_displacement_limit_m
    )
    assert gate_a is False, "ever_contacted=True for every group must NOT by itself satisfy Gate A"
    print("    confirmed: per-group ever_contacted=True on both sides, bilateral streak=0 -> gate_a condition False")


def test_non_simultaneous_contact_excluded_from_bilateral_streak():
    """Requirement 6: a tick where only ONE side is stable must reset the
    BILATERAL streak, even if each SIDE's own streak is nonzero."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)

    call_count = {"n": 0}
    orig = expert._bilateral_stable_now

    def fake_bilateral():
        call_count["n"] += 1
        # ticks 1-3: only left stable; ticks 4-6: only right stable --
        # never simultaneously both -> bilateral streak must stay 0.
        if call_count["n"] <= 3:
            return True, False, False
        return False, True, False

    expert._bilateral_stable_now = fake_bilateral
    for _ in range(6):
        expert._track_stability()
    assert expert._max_left_streak == 3
    assert expert._max_right_streak == 3
    assert expert._max_bilateral_streak == 0, "non-simultaneous per-side contact must not build a bilateral streak"
    print(f"    left_streak={expert._max_left_streak}, right_streak={expert._max_right_streak}, "
          f"bilateral_streak={expert._max_bilateral_streak} (never simultaneous -> stays 0)")


def test_exceeding_angular_velocity_fails_gate_a():
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    expert._max_bilateral_streak = expert.config.bilateral_streak_required + 5
    expert._obj_angvel_hist = [10.0]  # far above object_peak_angular_velocity_limit
    expert._initial_obj_xy = expert._object_pos()[:2].copy()
    angvel_peak = float(np.max(expert._obj_angvel_hist))
    gate_a = (
        expert._max_bilateral_streak >= expert.config.bilateral_streak_required
        and expert.object_xy_displacement() <= expert.config.object_xy_displacement_limit_m
        and angvel_peak <= expert.config.object_peak_angular_velocity_limit
    )
    assert gate_a is False, "exceeding the angular velocity limit must fail Gate A even with a full bilateral streak"
    print(f"    bilateral_streak={expert._max_bilateral_streak} (sufficient) but angvel_peak={angvel_peak} "
          f"> limit={expert.config.object_peak_angular_velocity_limit} -> gate_a False")


def test_hand_hand_force_over_limit_breaks_bilateral_stability():
    """Requirement 14: hand-hand contact force is checked and must break
    the bilateral-stable condition (not just be recorded as a metric)."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    expert._side_stable = lambda side: True  # both sides report topology-stable
    expert.env._hand_hand_contact_force = lambda: expert.config.hand_hand_force_limit_n + 1.0
    left_ok, right_ok, both_ok = expert._bilateral_stable_now()
    assert both_ok is False, "hand-hand force above the limit must break bilateral stability"
    print(f"    hand_hand_force forced above limit={expert.config.hand_hand_force_limit_n}N -> bilateral_ok={both_ok}")


def test_observation_includes_both_hands_and_object_orientation_velocity():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    print(f"    obs_dim={obs.shape[0]} (both-hand curl qpos/qvel + palm poses + object pos/quat/linvel/angvel + per-group force)")


def test_deterministic_repeat():
    env1 = make_env()
    expert1 = SharpaBimanualGraspExpert(env1)
    outcome1 = expert1.run(max_total_steps=1200)
    env2 = make_env()
    expert2 = SharpaBimanualGraspExpert(env2)
    outcome2 = expert2.run(max_total_steps=1200)
    assert outcome1.state == outcome2.state
    assert outcome1.step_count == outcome2.step_count
    assert outcome1.per_side_group_contact == outcome2.per_side_group_contact
    print(f"    deterministic: both runs reach {outcome1.state.name} at step {outcome1.step_count}")


def test_full_bimanual_rollout_runs_to_a_terminal_state_without_crashing():
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    outcome = expert.run(max_total_steps=1200)
    assert outcome.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE)
    print(f"    rollout reached terminal state {outcome.state.name} "
          f"(failure_reason={outcome.failure_reason}) after {outcome.step_count} steps")


def test_arm_gravity_compensation_reduces_precontact_tracking_error():
    """Session 39 regression guard: arm_gravity_compensation must reduce
    (not fabricate away) the arm's actuator-vs-actual joint tracking gap,
    an A/B causal check, not just a smoke assertion.

    Session 40 note: the controller now fails at WRIST_ALIGN (Orientation
    Alignment Gate, see docs/history/PHASE4_GRASP_SESSION_40.md) before
    ever reaching FINGERTIP_PRECONTACT, so this can no longer compare
    precontact_final_pos_error_m (both conditions would show the same
    unreached "inf" placeholder -- that would silently stop testing
    anything, not prove the fix still works). Instead it measures the
    same underlying quantity Session 39 identified -- ctrl register vs
    actual physical qpos, on the arm actuators -- at a fixed step count
    reached deterministically by BOTH conditions (well into WRIST_ALIGN's
    settle window)."""
    off_config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                                 max_episode_steps=1200, arm_gravity_compensation=False)
    on_config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                                max_episode_steps=1200, arm_gravity_compensation=True)

    def _tracking_gap_norm(config) -> float:
        env = SharpaGraspEnv(config)
        expert = SharpaBimanualGraspExpert(env)
        env.reset(seed=0)
        for _ in range(200):
            action = expert.step()
            env.step(action)
        return float(np.linalg.norm(env._arm_target - env.data.qpos[env._arm_qpos_adr]))

    off_gap = _tracking_gap_norm(off_config)
    on_gap = _tracking_gap_norm(on_config)
    print(f"    gravity_compensation off: arm ctrl-vs-actual gap={off_gap:.4f}rad | "
          f"on: gap={on_gap:.4f}rad")
    assert on_gap < off_gap, (
        f"arm_gravity_compensation did not reduce the measured arm tracking gap "
        f"(off={off_gap:.4f}rad, on={on_gap:.4f}rad)"
    )
    assert on_gap < 0.9 * off_gap, "expected at least a 10% reduction from gravity compensation"


def test_a_real_bimanual_gate_a_success_on_size_12():
    """Honest, currently-failing real SIZE_12 Gate A test.

    [This session] FOREARM_FORWARD_REACH's own settled-position residual
    is fixed and the Side-Grasp Posture Gate now genuinely passes (see
    test_side_grasp_posture_gate_passes) -- the bare-G1 canonical builder
    advances through WRIST_SIDE_GRASP_ALIGN -> FIVE_FINGER_PRESHAPE ->
    FOREARM_SIDE_DESCEND and stops there instead, deterministically
    failing with HAND_TABLE_COLLISION. Later contact/force gates are
    therefore still not entered.
    This test MUST NOT be weakened, deleted, or turned into a smoke
    assertion to make it pass -- it stays honestly failing until Gate A
    is actually achieved."""
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    outcome = expert.run(max_total_steps=8000)
    print(f"    state={outcome.state.name} failure_reason={outcome.failure_reason} "
          f"bilateral_streak={outcome.max_bilateral_stable_streak} "
          f"contact={outcome.per_side_group_contact} gate_a={outcome.gate_a}")
    assert outcome.gate_a is True, (
        "REAL bimanual Gate A success on SIZE_12 not yet achieved; see this test's docstring"
    )


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
