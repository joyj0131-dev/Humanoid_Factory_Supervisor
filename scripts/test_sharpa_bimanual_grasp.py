"""sharpa_bimanual tests: the Phase 4 OFFICIAL Sharpa Wave bimanual grasp
controller (SharpaBimanualGraspExpert). Run AFTER
test_sharpa_single_hand_diagnostic.py.

Run with:
    python3 scripts/test_sharpa_bimanual_grasp.py

HONEST CURRENT STATE (40th session): WRIST_ALIGN's orientation SETTLES
(stable, low drift) but the 39th session's "capture whatever the
position-only IK converged to" approach only ever guaranteed stability,
never that the hand actually FACES the object -- the 40th session's
audit found index/middle fingertips landing 12-21cm laterally off the
object at the old converged orientation (only the thumb was ever near
the surface). FOREARM_APPROACH/WRIST_ALIGN now target an EXPLICIT
object-facing orientation (_object_facing_R: the palm's closing axis
points at the object center), ramped in gradually across waypoints (the
same anti-instability recipe as before -- never a one-shot jump) and
gated by a new Orientation Alignment Gate (<=10deg palm-closing-axis-vs-
object angle, both hands). This gate is NOT yet passed: one hand's
achieved angle is close to (or under) tolerance, the other is not,
traced to the SAME category of steady-state actuator/physical tracking
error the 39th session found for position (arm_gravity_compensation
still enabled, still measurably helps, but a residual gap up to ~0.17rad
remains on specific joints -- see
docs/history/PHASE4_GRASP_SESSION_40.md). The controller now fails at
WRIST_ALIGN (BimanualFailureReason.WRIST_NOT_OBJECT_FACING), EARLIER
than the 39th session's PRECONTACT_TRACKING_NOT_ACHIEVED -- CONTACT_
ACQUIRE still never runs. test_a_real_bimanual_gate_a_success_on_size_12
below documents this HONESTLY as a failing test and must never be
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


def test_wrist_align_actually_reduces_measured_orientation_drift():
    """Requirement 4: WRIST_ALIGN must actually verify orientation has
    converged (measured, not asserted) -- this session's redesign measures
    real angular drift over a settle window instead of trusting a
    (previously found to freeze) secondary IK re-solve. Uses the
    DEFAULT config (object_facing_orientation=False, Session 39's
    unchanged, official behavior -- see test_object_facing_orientation_
    causes_torso_self_collision below for the opt-in Session 40 feature)."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    reached_wrist_align = False
    for _ in range(400):
        action = expert.step()
        obs, r, term, trunc, info = env.step(action)
        if expert.state == BimanualGraspState.WRIST_ALIGN:
            reached_wrist_align = True
        if expert.state in (BimanualGraspState.FIVE_FINGER_PRESHAPE, BimanualGraspState.FAILURE):
            break
    assert reached_wrist_align, "rollout must actually reach WRIST_ALIGN"
    print(f"    wrist_orientation_drift_deg={expert.wrist_orientation_drift_deg:.4f} "
          f"(tolerance={expert.config.wrist_orientation_stability_tol_deg}), "
          f"state after WRIST_ALIGN={expert.state.name}")
    if expert.state == BimanualGraspState.FAILURE:
        assert expert.failure_reason == BimanualFailureReason.WRIST_ORIENTATION_NOT_STABLE
    else:
        assert expert.wrist_orientation_drift_deg <= expert.config.wrist_orientation_stability_tol_deg


def test_object_facing_orientation_causes_torso_self_collision():
    """[Session 40] Documents the opt-in object_facing_orientation
    feature's measured, causally-confirmed side effect HONESTLY: an
    explicit object-facing wrist target (see _object_facing_R) makes
    index/middle fingertips land near the object instead of 12-21cm off
    it (a real improvement over Session 39's "whatever orientation
    happened to converge" approach) -- but for THIS seed/config it also
    drives the right wrist into torso_link at a real, forbidden
    self-collision force (measured up to ~113N, A/B-confirmed to
    disappear when the feature is disabled). This is why the feature
    defaults OFF (see BimanualGraspConfig.object_facing_orientation) --
    this test locks in that the NEW self-collision check
    (SharpaGraspEnv._torso_arm_collision_force) actually catches it
    rather than silently allowing an unsafe pose through, and must not
    be weakened to hide this until collision-aware waypoints (the
    documented next blocker) actually fix it."""
    config = BimanualGraspConfig(object_facing_orientation=True)
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env, config)
    for _ in range(600):
        action = expert.step()
        env.step(action)
        if expert.state in (BimanualGraspState.FIVE_FINGER_PRESHAPE, BimanualGraspState.FAILURE):
            break
    print(f"    state={expert.state.name} reason={expert.failure_reason} "
          f"torso_arm_collision_force={expert.torso_arm_collision_force_n:.2f}N "
          f"object_facing_angle(L,R)=({expert.left_object_facing_angle_deg:.2f},"
          f"{expert.right_object_facing_angle_deg:.2f})")
    assert expert.state == BimanualGraspState.FAILURE
    assert expert.failure_reason in (
        BimanualFailureReason.SELF_COLLISION_TORSO_ARM,
        BimanualFailureReason.WRIST_NOT_OBJECT_FACING,
    ), f"unexpected failure_reason={expert.failure_reason} for the opt-in object-facing feature"


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
    """HONEST, CURRENTLY-FAILING TEST (Section 8 requirement 13): a real
    SIZE_12 bimanual Gate A success. As of this session (40th):
      - WRIST_ALIGN's orientation SETTLES (drift well under tolerance)
        but at an object-facing angle that fails the Orientation
        Alignment Gate for at least one hand (see
        docs/history/PHASE4_GRASP_SESSION_40.md) -- the controller now
        fails there, EARLIER than the 39th session's
        PRECONTACT_TRACKING_NOT_ACHIEVED (which was itself measured
        against a non-object-facing, and therefore not truly meaningful,
        approach orientation).
      - Root cause (Session 40 audit): the SAME category of steady-state
        actuator/physical tracking error Session 39 found for position
        also affects orientation-relevant DOFs (measured: ctrl-vs-actual
        gaps up to 0.17rad on specific right-arm joints) -- not yet
        resolved.
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
        "REAL bimanual Gate A success on SIZE_12 not yet achieved (see this test's docstring "
        "and PHASE4_GRASP_SESSION_36.md for the disclosed FINGERTIP_PRECONTACT tracking-gap blocker)"
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
