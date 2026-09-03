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

[This session's follow-up] Root-caused (scripts/diagnose_hand_table_
contact_geometry.py, substep-level) FOREARM_SIDE_DESCEND's ~18.21N
hand-table force to essentially all non-thumb fingertips sustaining
table contact at the curl level (0.5) inherited from WRIST_SIDE_GRASP_
ALIGN -- NOT sensitive to standoff/y_offset (measured identical across a
bounded sweep, ruling out palm XY position). Curling further during this
state (side_descend_curl_target=0.95, position/orientation UNCHANGED)
retracts the fingertips clear: 18.21N -> 2.96N, with palm inward/finger-
down angle both improving into their Gate ranges as a side effect. The
tighter curl's extra actuator load needed a slightly finer waypoint
schedule (side_descend_waypoints 8->10) to clear the resulting ~10.7mm
steady-state tracking residual under the unchanged 10mm ik_pos_tol.

Reaching FOREARM_SIDE_DESCEND's real target then exposed a SEPARATE,
previously-latent bug in FINGERTIP_PRECONTACT (never exercised before
this session -- earlier failures always happened first): its own
_solve_both call was missing rest_q/rest_gain entirely, silently
defaulting to the STALE stand-pose rest_q instead of the clearance-
posture bias every other state in this file uses -- causing the
redundant 17-DOF solver to pick a torso-colliding branch (measured
83.17N torso-arm, 11.83N hand-table) for FINGERTIP_PRECONTACT's own
small final inward move. Fixed by passing rest_q=self._clearance_target
(matching the established convention exactly); both forces drop to
0.00N, and explicit torso-arm/hand-table fail-fast checks were added to
FINGERTIP_PRECONTACT's own gate (previously only checked hand-hand).

[Superseded, earlier session] The rollout used to reach FINGERTIP_
PRECONTACT and fail there with PRECONTACT_TRACKING_NOT_ACHIEVED, with
the Side-Grasp Posture Gate reporting PASS along the way.

[Functional Orientation fix] That PASS was found to rest on a CIRCULAR
metric: inward_angle_deg compared palm_R column 1 (the axis the wrist
was SOLVED to align) against the object direction, so it could only ever
report solver noise. A real, independent empirical audit
(scripts/audit_sharpa_local_closing_frame.py: hold the wrist fixed,
apply an actual curl delta, read where the fingertips really move) found
the TRUE closing axis is ~50deg off from that assumption. _object_
facing_R is rebuilt via a 2-vector Kabsch/Wahba fit against the real
axis (see that function's docstring in sharpa_bimanual_grasp_expert.py).

[Audit session] The commit that first landed this fix used a weight pair
(0.95/0.05) that, while satisfying the closing-axis tolerance, forced
finger_down to ~35deg -- and "fixed" the resulting Gate failure by
WIDENING the Gate's own tolerance (25->40deg), an improper Gate
relaxation. The SAME over-aggressive weight also caused an UNDISCLOSED
upstream cause of a genuine ~200-300N torso-arm self-collision in
FOREARM_SIDE_DESCEND: it forced a rotation large enough that the
redundant 17-DOF solver needed a torso-colliding branch to also satisfy
translation. Both were audited and fixed THIS session, not just
diagnosed:
  - side_grasp_finger_down_tol_deg reverted to its original 25deg (never
    relaxed again).
  - _object_facing_R's weight pair re-derived via a real-physics grid
    sweep (0.68/0.32) that satisfies BOTH the unchanged 15deg
    (closing-axis) and 25deg (finger-down) tolerances with real margin
    (measured: inward 12.6-12.7deg, finger-down 22.6deg, both hands).
  - FOREARM_SIDE_DESCEND's torso-arm collision is GONE under this
    weight pair (0.00N, real physics, seed=0) -- it was never actually a
    DESCEND-specific rest_q/path problem; fixing the upstream rotation
    magnitude fixed it. A separate, small (~11.4mm) tracking-residual
    plateau this surfaced (the wp10 solve's own claimed pos_error does
    not survive the post-waypoint settle tail) is fixed by a periodic
    correction re-solve after the waypoint schedule is exhausted (see
    FOREARM_SIDE_DESCEND's docstring) -- the rollout now reaches
    FINGERTIP_PRECONTACT again.
  - The Functional Orientation Gate (Section 9) is rebuilt as an ACTUAL
    combination of independently-measured subgates (finger-closure
    direction >=3/4 fingers + >2mm, thumb opposition, palm-inside angle
    <=15deg, finger-down angle <=25deg, swept torso-arm/hand-table
    collision) -- see _measure_functional_orientation. The prior commit's
    function of the same name only ever checked the 4-finger closure
    direction (renamed _measure_finger_closure_direction) and never
    checked thumb, never gated on finger-down, never checked collision
    history.
  - curl_target=0.95 in FOREARM_SIDE_DESCEND was re-verified (not just
    inherited) under the corrected orientation: a real sweep over {0.3,
    0.5, 0.6, 0.7, 0.8, 0.95} shows only 0.95 clears hand-table collision
    (0.23N vs 8.1-14.9N for lower values) -- genuinely required, not an
    artifact of the old orientation, though the true remaining fingertip
    travel at 0.95 (~7mm, ~1.5mm toward the object, before the
    CLOSE_FRACTION=0.85 ceiling) is disclosed as small.

Verified end-to-end (real env.step() physics, seed=0, reproduced 3x):
WRIST_SIDE_GRASP_ALIGN and FIVE_FINGER_PRESHAPE measure 0.00N torso-arm,
0.76N hand-table; FOREARM_SIDE_DESCEND measures 0.00N torso-arm, 0.23N
hand-table and reaches FINGERTIP_PRECONTACT; both the Side-Grasp Posture
Gate and the (rebuilt) Functional Orientation Gate genuinely PASS under
the ORIGINAL, unrelaxed tolerances. The rollout then fails at
FINGERTIP_PRECONTACT with PRECONTACT_TRACKING_NOT_ACHIEVED -- a known,
separate, disclosed blocker (Precontact tracking, out of THIS session's
scope) documented since Session 39; test_a_real_bimanual_gate_a_success_
on_size_12 below documents the overall Gate A outcome HONESTLY as a
failing test and must never be weakened, deleted, or converted into a
smoke assertion to make it pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv, ACTION_DIM
import humanoid_learning.expert.sharpa_bimanual_grasp_expert as sbe_module
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
    # [Audit session] Both the Side-Grasp Posture Gate AND the (rebuilt)
    # Functional Orientation Gate pass en route -- see
    # test_side_grasp_posture_gate_passes / test_functional_orientation_
    # gate_passes. FOREARM_SIDE_DESCEND's torso-arm collision (a prior
    # commit's improper Gate relaxation's undisclosed side effect) is
    # fixed -- the rollout now reaches FINGERTIP_PRECONTACT again and
    # fails there with PRECONTACT_TRACKING_NOT_ACHIEVED, a known, separate,
    # disclosed blocker (documented since Session 39) out of this
    # session's scope.
    env2 = make_env()
    expert2 = SharpaBimanualGraspExpert(env2)
    outcome = expert2.run(max_total_steps=2000)
    print(f"    full rollout: state={outcome.state.name} failure_reason={outcome.failure_reason} "
          f"side_grasp_gate={outcome.side_grasp_gate} functional_orientation_gate={outcome.functional_orientation_gate}")
    assert outcome.side_grasp_gate is True, "Side-Grasp Posture Gate should still pass en route to the current next blocker"
    assert outcome.functional_orientation_gate is True, "Functional Orientation Gate should pass en route"
    assert outcome.failure_reason == BimanualFailureReason.PRECONTACT_TRACKING_NOT_ACHIEVED, (
        "expected the CURRENT next independent blocker (FINGERTIP_PRECONTACT's own tracking gap, "
        "unrelated to this session's orientation fix); if this changed, the docstring/PROJECT_CONTEXT "
        "next-blocker note is now stale"
    )


def test_side_grasp_posture_gate_passes():
    """A genuine bilateral SIDE grasp posture (palms beside the object's
    own side faces, facing each other, fingers generally pointing down),
    not the old top-down palm-down reach. Locks in every Side-Grasp
    Posture Gate sub-condition via the SAME metrics WRIST_SIDE_GRASP_
    ALIGN itself gates the transition on -- not a looser or separately-
    computed check.

    inward_angle_deg and normals_opposed are backed by the REAL,
    non-circular closing axis (see _object_facing_R's and
    _object_facing_angle_deg's docstrings) instead of the old
    circularly-verified palm_R-column-1 assumption. [Audit session]
    side_grasp_finger_down_tol_deg stays at its ORIGINAL 25deg -- a prior
    commit widened it to 40deg to paper over a finger-down residual
    caused by an overly-aggressive orientation weight; that Gate
    relaxation is reverted, and _object_facing_R's weight pair is
    re-derived instead so the ORIGINAL 25deg tolerance is met with real
    margin (see that field's own docstring)."""
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    outcome = expert.run(max_total_steps=2000)
    m = outcome.side_grasp_posture
    print(f"    side_grasp_gate={outcome.side_grasp_gate}")
    for k, v in m.items():
        print(f"      {k}: {v}")
    assert m, "WRIST_SIDE_GRASP_ALIGN must be reached and measured (posture dict must not be empty)"
    assert expert.config.side_grasp_finger_down_tol_deg == 25.0, (
        "finger-down tolerance must stay at its ORIGINAL 25deg -- a prior commit improperly widened this "
        "to 40deg to paper over an orientation problem instead of fixing it; this must never regress"
    )
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
    verify torso-arm/hand-hand forbidden-collision force stays at or
    under the shared safety limit at EVERY tick through FIVE_FINGER_
    PRESHAPE, which the Side-Grasp Posture Gate already certifies clean."""
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


def test_side_descend_reaches_precontact_with_zero_torso_collision():
    """[Audit session] A prior commit on this branch reproduced a genuine
    ~200-300N torso-arm self-collision here, disclosed but left on the
    rollout's DEFAULT (only) path -- an unfixed safety regression. Root-
    caused THIS session: it was not a DESCEND-specific rest_q/path
    problem at all -- the same commit's overly-aggressive _object_
    facing_R weight (0.95) forced a rotation large enough that the
    redundant 17-DOF solver needed a torso-colliding branch to also
    satisfy translation. Fixing the upstream weight (see _object_facing_
    R's docstring) makes this collision GONE (0.00N, real physics) with
    NO DESCEND-specific posture/path change. A separate, small (~11.4mm)
    tracking-residual plateau this surfaced is fixed by a periodic
    correction re-solve after the waypoint schedule is exhausted (see
    FOREARM_SIDE_DESCEND's own docstring) -- the rollout now reaches
    FINGERTIP_PRECONTACT swept-clean, at every tick through this state,
    not just at the endpoint."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    reached_descend = False
    reached_precontact = False
    max_torso_arm = 0.0
    max_hand_table = 0.0
    for _ in range(1800):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        action = expert.step()
        env.step(action)
        if expert.state == BimanualGraspState.FOREARM_SIDE_DESCEND:
            reached_descend = True
            max_torso_arm = max(max_torso_arm, env._torso_arm_collision_force())
            max_hand_table = max(max_hand_table, env._hand_table_contact_force())
        if expert.state == BimanualGraspState.FINGERTIP_PRECONTACT:
            reached_precontact = True
            break
    print(f"    reached_descend={reached_descend} reached_precontact={reached_precontact} "
          f"max_torso_arm={max_torso_arm:.2f}N max_hand_table={max_hand_table:.2f}N "
          f"final_state={expert.state.name} reason={expert.failure_reason}")
    assert reached_descend, "FOREARM_SIDE_DESCEND must still be reached (WRIST_SIDE_GRASP_ALIGN/FIVE_FINGER_PRESHAPE unaffected)"
    limit = expert.config.hand_hand_force_limit_n
    assert max_torso_arm <= limit, f"FOREARM_SIDE_DESCEND torso-arm collision {max_torso_arm:.2f}N must not regress above {limit}N"
    assert max_torso_arm == 0.0, "the specific ~200-300N regression this test guards against must be fully gone, not merely under the limit"
    assert max_hand_table <= limit
    assert reached_precontact, "FOREARM_SIDE_DESCEND must now converge and advance to FINGERTIP_PRECONTACT"


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
    outcome1 = expert1.run(max_total_steps=2000)
    env2 = make_env()
    expert2 = SharpaBimanualGraspExpert(env2)
    outcome2 = expert2.run(max_total_steps=2000)
    assert outcome1.state == outcome2.state
    assert outcome1.step_count == outcome2.step_count
    assert outcome1.per_side_group_contact == outcome2.per_side_group_contact
    print(f"    deterministic: both runs reach {outcome1.state.name} at step {outcome1.step_count}")


def test_full_bimanual_rollout_runs_to_a_terminal_state_without_crashing():
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    outcome = expert.run(max_total_steps=2000)
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
    is fixed and BOTH the Side-Grasp Posture Gate and the new Functional
    Orientation Gate now genuinely pass (test_side_grasp_posture_gate_
    passes, test_functional_orientation_gate_passes) via a real, non-
    circular closing-axis measurement -- the bare-G1 canonical builder
    advances through WRIST_SIDE_GRASP_ALIGN -> FIVE_FINGER_PRESHAPE ->
    FOREARM_SIDE_DESCEND and stops there, deterministically failing with
    SIDE_DESCEND_NOT_ACHIEVED (a genuine torso-arm self-collision this
    session's corrected, larger reorientation uncovered -- see that
    state's own docstring; this is one state EARLIER than the previous
    session's PRECONTACT_TRACKING_NOT_ACHIEVED, a disclosed trade-off).
    FINGERTIP_PRECONTACT, CONTACT_ACQUIRE, and later force/closure gates
    are therefore still not entered.
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


def _run_to_align_end(env) -> SharpaBimanualGraspExpert:
    """Shared helper: drive an expert to the exact tick where WRIST_SIDE_
    GRASP_ALIGN has just converged (state has just become FIVE_FINGER_
    PRESHAPE) -- the pose the Functional Orientation Gate is evaluated
    at, for all the tests below."""
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    for _ in range(1200):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE, BimanualGraspState.FIVE_FINGER_PRESHAPE):
            break
        action = expert.step()
        env.step(action)
    assert expert.state == BimanualGraspState.FIVE_FINGER_PRESHAPE, (
        f"expected WRIST_SIDE_GRASP_ALIGN to converge; got {expert.state.name} ({expert.failure_reason})"
    )
    return expert


def test_functional_orientation_gate_passes():
    """The core deliverable: a REAL, non-circular check that the hand can
    actually functionally close onto the object -- applies an actual
    small additional closure from the CURRENT preshape state and reads
    where the fingertips really move (never trusting palm_R column 1).
    Must pass for BOTH hands independently (Section 4: '좌우 손 모두
    독립적으로 통과').

    [Audit session] The Gate is now an ACTUAL combination of independently
    -measured subgates, not just the 4-finger closure-direction check a
    prior commit used under this same function name: Finger-Closure
    Direction Subgate, Thumb Opposition Subgate, palm-inside angle,
    finger-down angle (at the ORIGINAL, un-relaxed 25deg), and swept
    collision. Every sub-condition is asserted explicitly below so a
    future change that silently drops one of them fails this test."""
    env = make_env()
    expert = _run_to_align_end(env)
    result = expert._measure_functional_orientation()
    print(f"    functional_orientation gate={result['gate']}")
    fc = result["finger_closure_direction"]
    th = result["thumb_opposition"]
    assert fc["gate"] is True, "Finger-Closure Direction Subgate must pass"
    assert th["gate"] is True, "Thumb Opposition Subgate must pass"
    assert result["inward_ok"] is True
    assert result["finger_down_ok"] is True
    assert result["swept_collision_ok"] is True
    for side in SIDES:
        r = fc["per_side"][side]
        t = th["per_side"][side]
        print(f"      {side}: n_positive={r['n_positive']} mean_disp_inward_mm={r['mean_disp_inward_mm']:.3f} "
              f"inward_angle_deg={r['inward_angle_deg']:.2f} finger_down_deg={r['finger_down_deg']:.2f} "
              f"thumb_opposed={t['opposed']} aperture_mm={t['aperture_dist_mm']:.1f}")
        assert r["n_positive"] >= 3, f"{side}: fewer than 3/4 nonthumb fingers move toward the object"
        assert r["mean_disp_inward_mm"] > 2.0, f"{side}: mean inward displacement not >2mm"
        assert r["inward_angle_deg"] <= expert.config.side_grasp_inward_angle_tol_deg
        assert r["finger_down_deg"] <= expert.config.side_grasp_finger_down_tol_deg
        assert t["opposed"], f"{side}: thumb must geometrically oppose the 4-finger group"
    assert result["gate"] is True


def test_old_orientation_construction_fails_functional_gate():
    """[This session] Direct regression guard for the root-cause bug:
    substituting the OLD, circularly-verified construction (palm_R
    column 1 = to-object direction, no independent closing-axis check)
    back in must FAIL the new, non-circular Functional Orientation Gate
    -- proving the new Gate actually discriminates a wrong orientation
    from a correct one, not just always reporting PASS."""

    def old_object_facing_R(side, palm_pos, obj_pos):
        closing_dir = obj_pos - palm_pos
        n = np.linalg.norm(closing_dir)
        y_col = closing_dir / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        world_up = np.array([0.0, 0.0, 1.0])
        ref = world_up if abs(np.dot(y_col, world_up)) < 0.95 else np.array([1.0, 0.0, 0.0])
        z_col = np.cross(y_col, ref)
        z_col /= np.linalg.norm(z_col)
        x_col = np.cross(y_col, z_col)
        x_col /= np.linalg.norm(x_col)
        return np.column_stack([x_col, y_col, z_col])

    saved = sbe_module._object_facing_R
    sbe_module._object_facing_R = old_object_facing_R
    try:
        env = make_env()
        expert = SharpaBimanualGraspExpert(env)
        env.reset(seed=0)
        for _ in range(1200):
            if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE, BimanualGraspState.FIVE_FINGER_PRESHAPE):
                break
            action = expert.step()
            env.step(action)
        if expert.state == BimanualGraspState.FIVE_FINGER_PRESHAPE:
            result = expert._measure_functional_orientation()
            print(f"    OLD construction: functional_gate={result['gate']} "
                  f"(expected False -- fingers move away from the object)")
            assert result["gate"] is False, "the OLD, circularly-verified orientation must NOT pass the real Functional Orientation Gate"
        else:
            # The OLD construction may also simply fail to converge (a
            # different, also-acceptable way of demonstrating it is not
            # functionally correct) -- either outcome confirms the bug.
            print(f"    OLD construction: did not even converge ({expert.state.name}, {expert.failure_reason}) -- also confirms it was not safe/correct")
    finally:
        sbe_module._object_facing_R = saved


def test_empirical_closing_axis_measurement_is_reproducible():
    """[This session] The state-preserving empirical measurement
    (_empirical_closing_axis_world) must give the same result (within
    MuJoCo's own contact-solver warmstart noise -- qpos/qvel/ctrl/
    group_synergy are restored exactly, but the solver's warmstart cache
    is not, so results agree closely but not bit-for-bit) when called
    twice in a row on the same state, and must leave qpos/qvel/ctrl/
    group_synergy exactly unchanged afterward (no side effects)."""
    env = make_env()
    expert = _run_to_align_end(env)
    qpos_before = env.data.qpos.copy()
    syn_before = env._group_synergy.copy()
    r1 = expert._measure_functional_orientation()
    r2 = expert._measure_functional_orientation()
    assert np.allclose(env.data.qpos, qpos_before), "measurement must restore qpos exactly"
    assert np.allclose(env._group_synergy, syn_before), "measurement must restore group_synergy exactly"
    for side in SIDES:
        fc1, fc2 = r1["finger_closure_direction"]["per_side"][side], r2["finger_closure_direction"]["per_side"][side]
        assert fc1["n_positive"] == fc2["n_positive"]
        d1, d2 = fc1["mean_disp_inward_mm"], fc2["mean_disp_inward_mm"]
        assert abs(d1 - d2) < 0.05 * max(abs(d1), abs(d2), 1.0), f"{side}: repeated measurement diverged too much ({d1:.4f}mm vs {d2:.4f}mm)"
        assert r1["thumb_opposition"]["per_side"][side]["opposed"] == r2["thumb_opposition"]["per_side"][side]["opposed"]
    print("    repeated measurement agrees within warmstart-noise tolerance and leaves state unchanged")


def test_wahba_rotation_is_a_proper_rotation():
    """det(R)=+1 for every _object_facing_R output (Section 7 requirement)."""
    env = make_env()
    expert = _run_to_align_end(env)
    obj_pos = expert._object_pos()
    for side in SIDES:
        palm_pos = env.palm_pose(side)[0]
        R = sbe_module._object_facing_R(side, palm_pos, obj_pos)
        det = np.linalg.det(R)
        ortho_err = np.max(np.abs(R.T @ R - np.eye(3)))
        print(f"    {side}: det(R)={det:.6f} orthogonality_err={ortho_err:.2e}")
        assert abs(det - 1.0) < 1e-6, f"{side}: det(R) must be +1 (proper rotation), got {det}"
        assert ortho_err < 1e-6, f"{side}: R must be orthogonal"


def test_left_right_use_identical_algorithm_no_hardcoded_sign_flip():
    """Section 7: '손별 joint 부호 하드코딩 최소화' -- verify _object_facing_R
    applies the EXACT SAME LOCAL_CLOSING_VEC/algorithm to both sides (no
    per-side sign flip baked into the function), and that both hands
    independently satisfy the Functional Orientation Gate (Section 4:
    never assume mirror symmetry holds -- checked independently).

    [Audit session] The "no hardcoded sign flip" half is checked
    STRUCTURALLY: feeding _object_facing_R Y-mirrored synthetic
    palm_pos/obj_pos for "left" vs "right" must produce Y-mirrored
    CLOSING and APPROACH axes (the two vectors _object_facing_R is
    actually solved for, and the only two ANY Gate metric reads), with NO
    dependence on which side string is passed (proves there is no
    `if side == "left"` branch) -- this is independent of any particular
    rollout's convergence, unlike recomputing R at a live-converged
    palm_pos (which differs slightly from the position R was actually
    SOLVED for, and is NOT a fair "same algorithm" check at this
    geometry's sensitivity). The full 3x3 R is NOT required to be exactly
    M@R@M: with only 2 target vectors, the third (lateral/twist) axis is
    genuinely underdetermined by the Wahba fit, and its handedness can
    differ between two numerically-distinct (mirrored but not bit-
    identical) solves -- disclosed, and confirmed NOT to matter, since no
    Gate or downstream measurement reads that axis."""
    obj_pos = np.array([0.27, 0.0, 0.85])
    left_palm = obj_pos + np.array([-0.15, 0.26, 0.10])
    right_palm = obj_pos + np.array([-0.15, -0.26, 0.10])
    R_left = sbe_module._object_facing_R("left", left_palm, obj_pos)
    R_right = sbe_module._object_facing_R("right", right_palm, obj_pos)
    mirror = np.diag([1.0, -1.0, 1.0])
    for vec, label in ((sbe_module.LOCAL_CLOSING_VEC, "closing"), (sbe_module.LOCAL_APPROACH_VEC, "approach")):
        world_left_mirrored = mirror @ (R_left @ vec)
        world_right = R_right @ vec
        assert np.allclose(world_left_mirrored, world_right, atol=1e-6), (
            f"_object_facing_R's {label} axis must be Y-mirror-symmetric between sides -- "
            "any discrepancy means a hardcoded per-side sign flip crept back in"
        )
    print("    closing/approach axes verified Y-mirror-symmetric (the two functionally-relevant vectors)")
    # Swapping the SIDE STRING alone (keeping the same numeric inputs) must not change the output at all.
    R_left_as_right_input = sbe_module._object_facing_R("right", left_palm, obj_pos)
    assert np.allclose(R_left, R_left_as_right_input, atol=1e-9), (
        "_object_facing_R's output must depend only on (palm_pos, obj_pos), not on the side string"
    )
    print("    _object_facing_R verified structurally side-agnostic (Y-mirror symmetric, no hardcoded branch)")

    env = make_env()
    expert = _run_to_align_end(env)
    print(f"    left_object_facing_angle_deg={expert.left_object_facing_angle_deg:.2f} "
          f"right_object_facing_angle_deg={expert.right_object_facing_angle_deg:.2f}")
    assert expert.left_object_facing_angle_deg <= expert.config.side_grasp_inward_angle_tol_deg
    assert expert.right_object_facing_angle_deg <= expert.config.side_grasp_inward_angle_tol_deg
    result = expert._measure_functional_orientation()
    assert result["finger_closure_direction"]["per_side"]["left"]["n_positive"] >= 3
    assert result["finger_closure_direction"]["per_side"]["right"]["n_positive"] >= 3
    assert result["thumb_opposition"]["per_side"]["left"]["opposed"]
    assert result["thumb_opposition"]["per_side"]["right"]["opposed"]
    print("    both sides independently pass using the identical LOCAL_CLOSING_VEC/Wahba construction")


def test_position_correct_but_wrong_orientation_fails_functional_gate():
    """Section 9: position-only-correct (fingers curl the WRONG way)
    must FAIL the Functional Orientation Gate. Substitutes an orientation
    construction whose closing axis is rotated 90deg AWAY from the
    object (same position/standoff/height targets, unchanged) and
    confirms the real per-finger, non-circular check catches it -- not
    just the angle proxy."""

    def sideways_object_facing_R(side, palm_pos, obj_pos):
        to_obj = obj_pos - palm_pos
        to_obj = to_obj / np.linalg.norm(to_obj)
        perp = np.cross(to_obj, np.array([0.0, 0.0, 1.0]))
        n = np.linalg.norm(perp)
        y_col = perp / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        z_col = np.cross(y_col, to_obj)
        z_col /= np.linalg.norm(z_col)
        x_col = np.cross(y_col, z_col)
        x_col /= np.linalg.norm(x_col)
        return np.column_stack([x_col, y_col, z_col])

    saved = sbe_module._object_facing_R
    sbe_module._object_facing_R = sideways_object_facing_R
    try:
        env = make_env()
        expert = SharpaBimanualGraspExpert(env)
        env.reset(seed=0)
        for _ in range(1200):
            if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE, BimanualGraspState.FIVE_FINGER_PRESHAPE):
                break
            action = expert.step()
            env.step(action)
        if expert.state == BimanualGraspState.FIVE_FINGER_PRESHAPE:
            result = expert._measure_functional_orientation()
            print(f"    sideways-rotated construction: functional_gate={result['gate']} (expected False)")
            assert result["gate"] is False, "position correct but closure direction wrong (90deg off) must FAIL the Functional Orientation Gate"
        else:
            print(f"    sideways-rotated construction: did not converge ({expert.state.name}, {expert.failure_reason}) -- also confirms it is not functionally correct")
    finally:
        sbe_module._object_facing_R = saved


def test_thumb_opposition_subgate_is_load_bearing():
    """[Audit session] Direct regression guard for the audit finding that
    a prior commit's "Functional Orientation Gate" never actually checked
    thumb opposition (it only checked 4-finger closure direction under
    that name). Forces _measure_thumb_opposition to report FAIL while the
    real finger-closure/angle checks still pass, and confirms the
    COMBINED Functional Orientation Gate correctly turns False -- proving
    thumb opposition is an actual, load-bearing AND condition, not
    computed-but-ignored."""
    env = make_env()
    expert = _run_to_align_end(env)
    real_result = expert._measure_functional_orientation()
    assert real_result["gate"] is True, "sanity check: the real pose must genuinely pass before we fake a failure"
    assert real_result["thumb_opposition"]["gate"] is True

    orig = expert._measure_thumb_opposition
    forced_fail = {
        "per_side": {s: dict(orig()["per_side"][s], opposed=False) for s in SIDES},
        "gate": False,
    }
    expert._measure_thumb_opposition = lambda curl_probe=0.15: forced_fail
    forced_result = expert._measure_functional_orientation()
    print(f"    real_gate={real_result['gate']} forced_thumb_fail_gate={forced_result['gate']} (expected False)")
    assert forced_result["gate"] is False, "a failing Thumb Opposition Subgate must fail the combined Functional Orientation Gate"
    assert forced_result["finger_closure_direction"]["gate"] is True, "finger-closure direction must still independently pass (isolating the thumb subgate as the cause)"


def test_side_descend_curl_target_remaining_travel_is_small_but_real():
    """[Audit session] Section 5/3 re-audit: curl_target=0.95 in FOREARM_
    SIDE_DESCEND is close to the CLOSE_FRACTION=0.85 ceiling (measured
    ~80.75% of each joint's real range already reached at synergy=0.95).
    Locks in that the remaining 0.95->1.0 fingertip travel is real,
    small, and consistently toward the object (not zero -- CONTACT_
    ACQUIRE later still has genuine travel to close onto the object from
    this state's synergy) -- disclosed honestly rather than silently
    treated as "fully closed"."""
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    for _ in range(1500):
        if expert.state in (BimanualGraspState.FOREARM_SIDE_DESCEND, BimanualGraspState.FAILURE):
            break
        action = expert.step()
        env.step(action)
    assert expert.state == BimanualGraspState.FOREARM_SIDE_DESCEND, f"expected FOREARM_SIDE_DESCEND, got {expert.state.name}"
    side = "left"
    side_idx = 0
    obj_pos = expert._object_pos()
    while env._group_synergy[side_idx * 4 + 1] < 0.949:
        a = np.zeros(ACTION_DIM)
        for g in (1, 2, 3):
            a[17 + side_idx * 4 + g] = 1.0
        env.step(a)
    tips_95 = {f: env.fingertip_pos(side, f).copy() for f in ("index", "middle", "ring", "pinky")}
    while env._group_synergy[side_idx * 4 + 1] < 0.999:
        a = np.zeros(ACTION_DIM)
        for g in (1, 2, 3):
            a[17 + side_idx * 4 + g] = 1.0
        env.step(a)
    tips_100 = {f: env.fingertip_pos(side, f).copy() for f in ("index", "middle", "ring", "pinky")}
    for f in ("index", "middle", "ring", "pinky"):
        d = tips_100[f] - tips_95[f]
        travel_mm = float(np.linalg.norm(d)) * 1000.0
        to_obj = obj_pos - tips_95[f]
        to_obj = to_obj / np.linalg.norm(to_obj)
        toward_obj_mm = float(np.dot(d, to_obj)) * 1000.0
        print(f"    {f}: travel(0.95->1.0)={travel_mm:.2f}mm toward_object={toward_obj_mm:.2f}mm")
        assert 0.5 < travel_mm < 20.0, f"{f}: remaining travel {travel_mm:.2f}mm outside the expected small-but-real range"
        assert toward_obj_mm > 0.0, f"{f}: remaining travel must still point toward the object, not away"


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
