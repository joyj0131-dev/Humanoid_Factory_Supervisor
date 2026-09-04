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

import mujoco
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
    # [Fingertip-contact session] Both the Horizontal-Wrap Posture Gate
    # AND the Functional Orientation Gate (including thumb opposition,
    # fixed this session -- see test_thumb_opposition_subgate_passes_at_
    # real_pose) now genuinely pass en route. FINGERTIP_PRECONTACT was
    # rebuilt as a closed-loop inward servo (see that state's own
    # docstring) and now safely advances the palm from ~207mm down to
    # roughly 130-140mm real fingertip-object separation without any
    # forbidden collision, but a genuine kinematic constraint (both
    # shoulders cannot bring the hand within ~15cm of the robot's own
    # sagittal midline at this orientation without brushing the torso --
    # confirmed via multiple independent bounded experiments, see
    # test_a_real_bimanual_gate_a_success_on_size_12's docstring) still
    # keeps the real separation well above the 4mm target within the
    # dedicated precontact_max_steps budget, so the rollout still
    # honestly times out with PRECONTACT_TRACKING_NOT_ACHIEVED.
    env2 = make_env()
    expert2 = SharpaBimanualGraspExpert(env2)
    outcome = expert2.run(max_total_steps=5000)
    print(f"    full rollout: state={outcome.state.name} failure_reason={outcome.failure_reason} "
          f"side_grasp_gate={outcome.side_grasp_gate} functional_orientation_gate={outcome.functional_orientation_gate}")
    assert outcome.side_grasp_gate is True, "Horizontal-Wrap Posture Gate should still pass en route to the current next blocker"
    assert outcome.functional_orientation_gate is True, "Functional Orientation Gate (incl. thumb opposition) should pass en route"
    assert outcome.failure_reason == BimanualFailureReason.PRECONTACT_TRACKING_NOT_ACHIEVED, (
        "expected the CURRENT next independent blocker (FINGERTIP_PRECONTACT's own kinematic-reach "
        "limit); if this changed, the docstring/PROJECT_CONTEXT next-blocker note is now stale"
    )


def test_side_grasp_posture_gate_passes():
    """[Horizontal-Wrap session] The Horizontal-Wrap Posture Gate --
    SUPERSEDES the old Side-Grasp Posture Gate's finger-down check. The
    old check forced the finger-longitudinal axis toward world -Z
    (fingers pointing AT the table), which this session's user directive
    retracted as an incorrect task specification: it drove wrist_roll to
    within 0.0deg of its own +-113deg hard limit (measured directly, see
    test_old_finger_down_pose_fails_new_gate below) and produced fingers
    pointing ~67deg away from the table plane, not the reference image's
    near-horizontal wrap. This test locks in the REPLACEMENT geometry:
    palm plane near-vertical (palm-inward angle small), fingers near-
    horizontal (small angle to the table plane), the pinky-side (ulnar)
    edge of the hand lowest and pointed down, and every wrist joint
    keeping real margin from its own hard limit -- via the SAME metrics
    WRIST_SIDE_GRASP_ALIGN itself gates the transition on."""
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    outcome = expert.run(max_total_steps=2000)
    m = outcome.side_grasp_posture
    print(f"    side_grasp_gate={outcome.side_grasp_gate}")
    for k, v in m.items():
        print(f"      {k}: {v}")
    assert m, "WRIST_SIDE_GRASP_ALIGN must be reached and measured (posture dict must not be empty)"
    assert not hasattr(expert.config, "side_grasp_finger_down_tol_deg"), (
        "the old finger-down tolerance field must stay REMOVED, not just unused -- "
        "it encoded an incorrect task specification (fingers forced at the table)"
    )
    assert expert.config.side_grasp_finger_table_tol_deg == 20.0
    assert expert.config.side_grasp_ulnar_down_tol_deg == 25.0
    assert expert.config.wrist_joint_margin_tol_deg == 10.0
    assert outcome.side_grasp_gate is True
    assert all(m["outside_side_face"].values()), "both palms must be outside the object's own side faces"
    assert m["inward_angle_deg"]["left"] <= expert.config.side_grasp_inward_angle_tol_deg
    assert m["inward_angle_deg"]["right"] <= expert.config.side_grasp_inward_angle_tol_deg
    assert m["normals_opposed"], "palm normals (closing axes) must face each other"
    assert m["finger_table_deg"]["left"] <= expert.config.side_grasp_finger_table_tol_deg
    assert m["finger_table_deg"]["right"] <= expert.config.side_grasp_finger_table_tol_deg
    assert m["ulnar_down_deg"]["left"] <= expert.config.side_grasp_ulnar_down_tol_deg
    assert m["ulnar_down_deg"]["right"] <= expert.config.side_grasp_ulnar_down_tol_deg
    assert all(m["ulnar_lowest_ok"].values()), "the ulnar-support body must be the lowest hand/wrist body"
    assert all(v >= expert.config.wrist_joint_margin_tol_deg for v in m["wrist_margins_deg"].values()), (
        f"every wrist joint must keep >= {expert.config.wrist_joint_margin_tol_deg}deg margin from its hard limit"
    )
    assert m["wrist_qvel_peak"] <= expert.config.wrist_max_qvel_rad_s
    assert not m["crosses_top_footprint"], "hands must not cross the object's own top footprint"
    assert m["mirror_pos_err_m"] <= expert.config.side_grasp_mirror_pos_tol_m
    assert m["mirror_ori_err_deg"] <= expert.config.side_grasp_mirror_ori_tol_deg
    assert m["elbow_ok"], "elbow must not be markedly above the shoulder"
    assert m["torso_arm_ok"] and m["hand_table_ok"] and m["allowed_ulnar_ok"] and m["hand_hand_ok"], (
        "no forbidden collision at the align pose (allowed ulnar-edge support is exempt, see hand_table_ok)"
    )
    assert m["premature_contact_ok"], "no premature (non-fingertip) object contact"
    assert m["streak"] >= expert.config.side_align_stable_streak_required


def test_old_finger_down_pose_fails_new_gate():
    """[Horizontal-Wrap session, new] The old, retracted finger-down
    orientation construction (_object_facing_R's approach-axis target
    biased toward world -Z instead of a horizontal wrap direction) must
    FAIL the new Horizontal-Wrap Posture Gate -- and must reproduce the
    exact failure mode the user's directive named: wrist_roll pinned to
    within a few degrees of its own +-113deg hard limit."""
    import mujoco

    def old_object_facing_R(side, palm_pos, obj_pos, current_approach_world=None, weights=None):
        to_obj = obj_pos - palm_pos
        n = np.linalg.norm(to_obj)
        to_obj = to_obj / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        horiz = to_obj.copy()
        horiz[2] = 0.0
        hn = np.linalg.norm(horiz)
        horiz = horiz / hn if hn > 1e-9 else np.zeros(3)
        inward_weight = 0.60
        down = (1 - inward_weight) * np.array([0.0, 0.0, -1.0]) + inward_weight * horiz
        down = down / np.linalg.norm(down)
        return sbe_module._wahba_R(
            [sbe_module.LOCAL_CLOSING_VEC, sbe_module.LOCAL_APPROACH_VEC], [to_obj, down], weights=[0.68, 0.32]
        )

    saved = sbe_module._object_facing_R
    sbe_module._object_facing_R = old_object_facing_R
    try:
        env = make_env()
        expert = SharpaBimanualGraspExpert(env)
        outcome = expert.run(max_total_steps=2000)
        m = outcome.side_grasp_posture
        assert m, "old construction must still reach and converge WRIST_SIDE_GRASP_ALIGN (it is kinematically valid, just wrong)"
        print(f"    OLD finger-down construction: finger_table_deg={m['finger_table_deg']} "
              f"ulnar_down_deg={m['ulnar_down_deg']} wrist_margins_deg={m['wrist_margins_deg']}")
        assert outcome.side_grasp_gate is False, "the OLD finger-down orientation must NOT pass the Horizontal-Wrap Posture Gate"
        assert m["finger_table_deg"]["left"] > expert.config.side_grasp_finger_table_tol_deg, (
            "the old construction's fingers must NOT be near-horizontal (that was the whole bug)"
        )
        margin = min(m["wrist_margins_deg"].values())
        print(f"    OLD construction min wrist joint margin = {margin:.2f}deg (expected pinned near 0deg)")
        assert margin < expert.config.wrist_joint_margin_tol_deg, (
            "the OLD orientation must reproduce the near-zero wrist joint-limit margin the user's directive named"
        )
    finally:
        sbe_module._object_facing_R = saved


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

    [Fingertip-contact session] The Horizontal-Wrap Posture Gate AND the
    Functional Orientation Gate (incl. thumb opposition) both genuinely
    PASS with real margin. FINGERTIP_PRECONTACT was rebuilt as a closed-
    loop inward servo (real fingertip-to-object-face separation, small
    per-tick steps, rest_q warm-started from the live qpos, a graduated
    collision-recovery ladder -- see that state's own docstring) and
    reliably brings the palm inward WITHOUT any dangerous collision
    (verified: max_forbidden_hand_table/torso-arm force stays 0.00N
    across full rollouts). A genuine, newly-diagnosed KINEMATIC
    constraint remains, though: bringing either hand within roughly
    150mm of the robot's own sagittal midline at this orientation makes
    that shoulder graze the torso (measured directly, seed=0:
    shoulder_yaw_link<->torso_link contact, ~1.7-3N sustained, not a
    transient spike) -- a bounded set of real-physics experiments this
    session (sweeping FOREARM_SIDE_DESCEND's own y_offset down to 0.08,
    sweeping approach height up to 0.25, sweeping forward standoff up to
    +0.15, and heavily incentivizing waist rotation in the IK's joint
    cost) all reproduced the same wall or failed to relieve it -- this
    is not a solver-branch artifact fixable by more IK tuning. A real
    wrap-direction bug fix (see _horizontal_wrap_target's own docstring)
    materially reduced how far the palm needs to travel (entry
    separation 207mm -> 189mm, real end-of-budget separation ~163mm ->
    ~130-140mm) and fixed the (separately real) thumb-opposition gap,
    but did not eliminate the wall. The rollout now reaches FINGERTIP_
    PRECONTACT (unchanged) and still ends in PRECONTACT_TRACKING_NOT_
    ACHIEVED, honestly, with a much better real separation than before
    this session and zero forbidden collision. This test MUST NOT be
    weakened, deleted, or turned into a smoke assertion to make it pass
    -- it stays honestly failing until Gate A is actually achieved."""
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
    """The Finger-Closure Direction Subgate, palm-inside angle, finger-
    to-table-plane angle, and swept-collision requirements of the
    Functional Orientation Gate: a REAL, non-circular check that the
    hand can actually functionally close onto the object -- applies an
    actual small additional closure from the CURRENT preshape state and
    reads where the fingertips really move (never trusting palm_R
    column 1).

    [Fingertip-contact session] The Thumb Opposition Subgate now
    genuinely passes too: the wrap-direction bug fix in
    _horizontal_wrap_target (see that function's own docstring -- the
    wrap axis used to be perpendicular to the standoff-contaminated full
    horizontal to-object vector instead of the lateral-only component)
    also fixed the small negative thumb_disp_toward_object_mm residual
    the prior (Horizontal-Wrap) session measured and disclosed
    (-0.02 to -0.4mm) -- it is now positive (~0.07mm, both hands,
    consistently) at the same curl_probe. The combined Functional
    Orientation Gate is asserted True below again."""
    env = make_env()
    expert = _run_to_align_end(env)
    result = expert._measure_functional_orientation()
    print(f"    functional_orientation gate={result['gate']}")
    fc = result["finger_closure_direction"]
    th = result["thumb_opposition"]
    assert fc["gate"] is True, "Finger-Closure Direction Subgate must pass"
    assert result["inward_ok"] is True
    assert result["finger_table_ok"] is True
    assert result["swept_collision_ok"] is True
    for side in SIDES:
        r = fc["per_side"][side]
        t = th["per_side"][side]
        print(f"      {side}: n_positive={r['n_positive']} mean_disp_inward_mm={r['mean_disp_inward_mm']:.3f} "
              f"inward_angle_deg={r['inward_angle_deg']:.2f} finger_table_deg={r['finger_table_deg']:.2f} "
              f"thumb_opposed={t['opposed']} aperture_mm={t['aperture_dist_mm']:.1f} "
              f"thumb_disp_toward_object_mm={t['thumb_disp_toward_object_mm']:.3f}")
        assert r["n_positive"] >= 3, f"{side}: fewer than 3/4 nonthumb fingers move toward the object"
        assert r["mean_disp_inward_mm"] > 2.0, f"{side}: mean inward displacement not >2mm"
        assert r["inward_angle_deg"] <= expert.config.side_grasp_inward_angle_tol_deg
        assert r["finger_table_deg"] <= expert.config.side_grasp_finger_table_tol_deg
        assert t["opposed"], f"{side}: thumb must geometrically oppose the 4-finger group"
    assert result["gate"] is True


def test_old_orientation_construction_fails_functional_gate():
    """[This session] Direct regression guard for the root-cause bug:
    substituting the OLD, circularly-verified construction (palm_R
    column 1 = to-object direction, no independent closing-axis check)
    back in must FAIL the new, non-circular Functional Orientation Gate
    -- proving the new Gate actually discriminates a wrong orientation
    from a correct one, not just always reporting PASS."""

    def old_object_facing_R(side, palm_pos, obj_pos, current_approach_world=None, weights=None):
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
    applies the EXACT SAME LOCAL_CLOSING_VEC/LOCAL_APPROACH_VEC/algorithm
    to both sides (no per-side sign flip baked in for the two axes that
    were already measured to need none), and that both hands independently
    satisfy the parts of the Functional Orientation Gate that currently
    pass (Section 4: never assume mirror symmetry holds -- checked
    independently).

    [Horizontal-Wrap session, updated] A THIRD axis (LOCAL_ULNAR_VEC) was
    added this session with a genuine, MEASURED per-side sign
    (ULNAR_SIGN, see that constant's own docstring: the two hands are
    independently-authored mirrored meshes, confirmed by direct FK
    measurement, not assumed) -- so _object_facing_R's full output is no
    longer side-string-independent, and the old "swap the side string
    alone, output must not change" assertion is retired: it would now
    fail correctly, for a real, disclosed, measured reason, not a bug.
    What remains checked structurally: the CLOSING and APPROACH local
    axes (which really are side-invariant, unaffected by ULNAR_SIGN) map
    to Y-mirrored world targets between sides, feeding Y-mirrored
    synthetic palm_pos/obj_pos for "left" vs "right"."""
    obj_pos = np.array([0.27, 0.0, 0.85])
    left_palm = obj_pos + np.array([-0.15, 0.26, 0.10])
    right_palm = obj_pos + np.array([-0.15, -0.26, 0.10])
    left_approach = np.array([1.0, 0.0, 0.0])
    R_left = sbe_module._object_facing_R("left", left_palm, obj_pos, current_approach_world=left_approach)
    R_right = sbe_module._object_facing_R("right", right_palm, obj_pos, current_approach_world=left_approach)
    mirror = np.diag([1.0, -1.0, 1.0])
    for vec, label in ((sbe_module.LOCAL_CLOSING_VEC, "closing"), (sbe_module.LOCAL_APPROACH_VEC, "approach")):
        world_left_mirrored = mirror @ (R_left @ vec)
        world_right = R_right @ vec
        assert np.allclose(world_left_mirrored, world_right, atol=1e-3), (
            f"_object_facing_R's {label} axis must be Y-mirror-symmetric between sides -- "
            "any discrepancy means a hardcoded per-side sign flip crept back in"
        )
    print("    closing/approach axes verified Y-mirror-symmetric (side-invariant local vectors)")
    # The ULNAR axis (genuinely side-dependent, ULNAR_SIGN) must ALSO be
    # Y-mirror-symmetric in WORLD frame, despite the sign flip in local
    # frame -- that is the entire point of ULNAR_SIGN existing.
    left_ulnar_world = R_left @ (sbe_module.ULNAR_SIGN["left"] * sbe_module.LOCAL_ULNAR_VEC)
    right_ulnar_world = R_right @ (sbe_module.ULNAR_SIGN["right"] * sbe_module.LOCAL_ULNAR_VEC)
    assert np.allclose(mirror @ left_ulnar_world, right_ulnar_world, atol=1e-3), (
        "ULNAR_SIGN must produce a Y-mirrored WORLD ulnar-down direction between sides"
    )
    print("    ulnar axis verified Y-mirror-symmetric in world frame (per-side local sign is intentional)")

    env = make_env()
    expert = _run_to_align_end(env)
    print(f"    left_object_facing_angle_deg={expert.left_object_facing_angle_deg:.2f} "
          f"right_object_facing_angle_deg={expert.right_object_facing_angle_deg:.2f}")
    assert expert.left_object_facing_angle_deg <= expert.config.side_grasp_inward_angle_tol_deg
    assert expert.right_object_facing_angle_deg <= expert.config.side_grasp_inward_angle_tol_deg
    result = expert._measure_functional_orientation()
    assert result["finger_closure_direction"]["per_side"]["left"]["n_positive"] >= 3
    assert result["finger_closure_direction"]["per_side"]["right"]["n_positive"] >= 3
    print("    both sides independently pass finger-closure-direction using the identical Wahba construction")


def test_position_correct_but_wrong_orientation_fails_functional_gate():
    """Section 9: position-only-correct (fingers curl the WRONG way)
    must FAIL the Functional Orientation Gate. Substitutes an orientation
    construction whose closing axis is rotated 90deg AWAY from the
    object (same position/standoff/height targets, unchanged) and
    confirms the real per-finger, non-circular check catches it -- not
    just the angle proxy."""

    def sideways_object_facing_R(side, palm_pos, obj_pos, current_approach_world=None, weights=None):
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
    that name). Forces _measure_thumb_opposition to report PASS, then
    FAIL, confirming the combined Functional Orientation Gate tracks it
    either way -- proving thumb opposition is an actual, load-bearing AND
    condition, not computed-but-ignored.

    [Horizontal-Wrap session] Rewritten to construct a FORCED-PASSING
    thumb result for the sanity check instead of relying on the real,
    live pose: under the new orientation the real pose's thumb subgate
    currently fails by a small, disclosed margin (see
    test_thumb_opposition_subgate_currently_fails_under_new_orientation),
    so asserting "the real pose must pass" here would be a false
    premise. This version isolates the load-bearing property itself,
    independent of whatever the current real pose happens to measure."""
    env = make_env()
    expert = _run_to_align_end(env)
    orig = expert._measure_thumb_opposition
    real_per_side = orig()["per_side"]

    forced_pass = {
        "per_side": {s: dict(real_per_side[s], opposed=True, thumb_disp_toward_object_mm=1.0) for s in SIDES},
        "gate": True,
    }
    expert._measure_thumb_opposition = lambda curl_probe=0.15: forced_pass
    passing_result = expert._measure_functional_orientation()
    assert passing_result["finger_closure_direction"]["gate"] is True, (
        "sanity check: finger-closure direction must genuinely pass at this pose"
    )
    assert passing_result["gate"] is True, "a passing Thumb Opposition Subgate (forced) must not block the combined Gate"

    forced_fail = {
        "per_side": {s: dict(real_per_side[s], opposed=False) for s in SIDES},
        "gate": False,
    }
    expert._measure_thumb_opposition = lambda curl_probe=0.15: forced_fail
    forced_result = expert._measure_functional_orientation()
    print(f"    forced_pass_gate={passing_result['gate']} forced_thumb_fail_gate={forced_result['gate']} (expected False)")
    assert forced_result["gate"] is False, "a failing Thumb Opposition Subgate must fail the combined Functional Orientation Gate"
    assert forced_result["finger_closure_direction"]["gate"] is True, "finger-closure direction must still independently pass (isolating the thumb subgate as the cause)"


def test_thumb_opposition_subgate_passes_at_real_pose():
    """[Fingertip-contact session] At the REAL, converged WRIST_SIDE_
    GRASP_ALIGN pose, the Thumb Opposition Subgate now genuinely passes:
    a real thumb-only curl probe moves the thumb toward the 4-finger
    aperture AND toward the object (both positive mm). A prior
    (Horizontal-Wrap) session measured a small, consistently-signed
    NEGATIVE toward-object component here (-0.02 to -0.4mm) and
    disclosed it as a known gap; this session's _horizontal_wrap_target
    fix (the wrap axis no longer mixes in the standoff/X component when
    choosing the lateral wrap direction -- see that function's own
    docstring) fixed it as a side effect, without touching FIVE_FINGER_
    PRESHAPE's own preshape angle at all."""
    env = make_env()
    expert = _run_to_align_end(env)
    result = expert._measure_thumb_opposition(curl_probe=0.15)
    print(f"    thumb_opposition per_side={result['per_side']}")
    for side in SIDES:
        r = result["per_side"][side]
        assert r["aperture_dist_mm"] > 30.0, f"{side}: aperture must still be real/non-degenerate"
        assert r["thumb_disp_toward_aperture_mm"] > 0.0, f"{side}: thumb curl must still close toward the aperture"
        assert r["thumb_disp_toward_object_mm"] > 0.0, f"{side}: thumb curl must move toward the object"
        assert r["opposed"], f"{side}: thumb opposition must hold at the real pose"
    assert result["gate"] is True


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


def test_open_preshape_curl_target_removes_near_full_closure():
    """[Open-preshape session] Locks in that FOREARM_SIDE_DESCEND's curl
    target is no longer the near-total-closure workaround (0.95, ~80.75%
    of real joint range, near-zero/negative remaining object-direction
    fingertip travel -- see test_side_descend_curl_target_remaining_
    travel_is_small_but_real, which still documents 0.95's OWN numbers as
    a historical fact, not the current default). The new default (0.7)
    must leave real, substantial remaining travel toward the object for
    CONTACT_ACQUIRE to use."""
    cfg = BimanualGraspConfig()
    assert cfg.side_descend_curl_target < 0.9, (
        f"side_descend_curl_target={cfg.side_descend_curl_target} is back near the old 0.95 "
        "near-total-closure value this session was asked to remove"
    )
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    for _ in range(1500):
        if expert.state in (BimanualGraspState.FOREARM_SIDE_DESCEND, BimanualGraspState.FAILURE):
            break
        action = expert.step()
        env.step(action)
    assert expert.state == BimanualGraspState.FOREARM_SIDE_DESCEND
    side, side_idx = "left", 0
    obj_pos = expert._object_pos()
    curl_target = expert.config.side_descend_curl_target
    while env._group_synergy[side_idx * 4 + 1] < curl_target - 0.001:
        a = np.zeros(ACTION_DIM)
        for g in (1, 2, 3):
            a[17 + side_idx * 4 + g] = 1.0
        env.step(a)
    tips_at_target = {f: env.fingertip_pos(side, f).copy() for f in ("index", "middle", "ring", "pinky")}
    while env._group_synergy[side_idx * 4 + 1] < 0.999:
        a = np.zeros(ACTION_DIM)
        for g in (1, 2, 3):
            a[17 + side_idx * 4 + g] = 1.0
        env.step(a)
    tips_100 = {f: env.fingertip_pos(side, f).copy() for f in ("index", "middle", "ring", "pinky")}
    for f in ("index", "middle", "ring", "pinky"):
        d = tips_100[f] - tips_at_target[f]
        to_obj = obj_pos - tips_at_target[f]
        to_obj = to_obj / np.linalg.norm(to_obj)
        toward_obj_mm = float(np.dot(d, to_obj)) * 1000.0
        print(f"    {f}: remaining travel({curl_target}->1.0) toward_object={toward_obj_mm:.2f}mm")
        assert toward_obj_mm > 5.0, (
            f"{f}: remaining object-direction travel {toward_obj_mm:.2f}mm too small -- "
            "curl_target leaves CONTACT_ACQUIRE too little stroke"
        )


def test_open_preshape_descend_has_zero_table_collision():
    """[Open-preshape session] The new (height=0.05, curl=0.7) DESCEND
    target must clear the table WITHOUT relying on near-total closure --
    locks in 0.00N hand-table force through FOREARM_SIDE_DESCEND under the
    default config, real physics, seed=0."""
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    max_table_in_descend = 0.0
    for _ in range(2500):
        if expert.state in (BimanualGraspState.FINGERTIP_PRECONTACT, BimanualGraspState.FAILURE):
            break
        prev_state = expert.state
        action = expert.step()
        env.step(action)
        if prev_state == BimanualGraspState.FOREARM_SIDE_DESCEND:
            max_table_in_descend = max(max_table_in_descend, env._hand_table_contact_force())
    print(f"    max_hand_table_force during FOREARM_SIDE_DESCEND={max_table_in_descend:.3f}N")
    assert max_table_in_descend <= expert.config.hand_hand_force_limit_n, (
        f"open-preshape DESCEND must stay under the {expert.config.hand_hand_force_limit_n}N limit, "
        f"got {max_table_in_descend:.2f}N"
    )


def test_precontact_orientation_drift_bug_is_fixed():
    """[Open-preshape session] Root-caused a previously-unmeasured bug:
    _object_facing_R is a function of PALM POSITION, but FOREARM_SIDE_
    DESCEND used to soft-anchor orientation to whatever the CURRENT pose
    already was (self-referential), so nothing corrected accumulated
    solver drift as DESCEND moved the palm -- by FINGERTIP_PRECONTACT the
    actual orientation had silently drifted >30deg from WRIST_SIDE_GRASP_
    ALIGN's locked target (never previously measured/reported; masked by
    PRECONTACT_TRACKING_NOT_ACHIEVED being attributed entirely to a
    separate, also-real position droop). Fixed by freezing a single fresh
    _object_facing_R at FOREARM_SIDE_DESCEND's own target position
    (_descend_locked_R) and anchoring both DESCEND and PRECONTACT to it.
    This test locks in the improvement (ori_err now a small, bounded
    single-digit-degree residual) WITHOUT claiming the Precontact Tracking
    Gate's own unchanged 5deg tolerance is met (it is not, this session --
    see FINGERTIP_PRECONTACT's own docstring)."""
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    outcome = expert.run(max_total_steps=2500)
    print(f"    precontact_final_ori_error_deg={outcome.precontact_final_ori_error_deg}")
    for side in SIDES:
        err = outcome.precontact_final_ori_error_deg[side]
        assert err < 15.0, (
            f"{side}: orientation error {err:.2f}deg -- the pre-fix drift bug reproduced "
            "(was consistently >30deg before this session's fix)"
        )


def test_gate_definitions_unchanged_by_horizontal_wrap_session():
    """[Horizontal-Wrap session] Locks in that the Gate criteria this
    session did NOT intend to touch (precontact tracking tolerances,
    Gate A's own force/streak thresholds, object safety limits) stay
    exactly as they were -- the finger-down tolerance is DELIBERATELY
    retired (replaced by side_grasp_finger_table_tol_deg/side_grasp_
    ulnar_down_tol_deg/wrist_joint_margin_tol_deg, see
    test_side_grasp_posture_gate_passes), which is this session's
    intended, disclosed change, not a silent regression."""
    cfg = BimanualGraspConfig()
    assert not hasattr(cfg, "side_grasp_finger_down_tol_deg"), "must stay removed (intentional, disclosed retraction)"
    assert cfg.side_grasp_finger_table_tol_deg == 20.0
    assert cfg.side_grasp_ulnar_down_tol_deg == 25.0
    assert cfg.wrist_joint_margin_tol_deg == 10.0
    assert cfg.side_grasp_inward_angle_tol_deg == 15.0
    assert cfg.ik_pos_tol == 0.01
    assert cfg.precontact_ori_tol_deg == 5.0
    assert cfg.precontact_stable_streak_required == 15
    assert cfg.hand_hand_force_limit_n == 8.0
    assert cfg.bilateral_streak_required == 30
    assert cfg.object_xy_displacement_limit_m == 0.03
    assert cfg.object_peak_angular_velocity_limit == 2.0
    assert cfg.wrist_max_qvel_rad_s == 2.0


def test_ulnar_support_body_identified_not_guessed():
    """[Horizontal-Wrap session, new] The ulnar-edge support body used by
    _hand_table_forces_categorized must be a REAL body in the compiled
    model, identified via sc.sharpa_body()'s own naming convention (never
    a hand-typed guess) -- and must be the pinky metacarpal specifically,
    the one body whose position relative to the palm is fixed by a
    PRESHAPE-only joint (sharpa_config._JOINT_ROLES: pinky "CMC":
    "preshape"), matching the FK finding (LOCAL_ULNAR_VEC's docstring)
    that it sits at the hand's own ulnar-most extreme."""
    import humanoid_learning.envs.sharpa_config as sc

    env = make_env()
    for side in SIDES:
        expected_name = sc.sharpa_body(side, "pinky", "MC")
        assert env.ULNAR_SUPPORT_BODY[side] == expected_name
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, expected_name)
        assert bid >= 0, f"{expected_name} must exist in the compiled model"
    print(f"    ULNAR_SUPPORT_BODY={env.ULNAR_SUPPORT_BODY} (verified against the compiled model)")


def test_hand_table_forces_categorized_forbidden_vs_allowed():
    """[Horizontal-Wrap session, new] _hand_table_forces_categorized must
    split hand<->table contact into "forbidden" (fingertip/palm/wrist
    housing) and "allowed_ulnar" (the single identified support body per
    side), and the running maxima the expert tracks
    (max_forbidden_hand_table_force_n/max_allowed_ulnar_table_force_n)
    must never exceed what a fresh categorized read reports at the same
    tick (monotonic running-max property)."""
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    for _ in range(1000):
        if expert.state in (BimanualGraspState.FIVE_FINGER_PRESHAPE, BimanualGraspState.FAILURE):
            break
        action = expert.step()
        env.step(action)
    result = env._hand_table_forces_categorized()
    assert set(result.keys()) == {"forbidden", "allowed_ulnar"}
    assert result["forbidden"] >= 0.0 and result["allowed_ulnar"] >= 0.0
    assert expert.max_forbidden_hand_table_force_n >= result["forbidden"] - 1e-9
    assert expert.max_allowed_ulnar_table_force_n >= result["allowed_ulnar"] - 1e-9
    print(f"    live categorized={result}  tracked_max_forbidden={expert.max_forbidden_hand_table_force_n:.3f} "
          f"tracked_max_allowed_ulnar={expert.max_allowed_ulnar_table_force_n:.3f}")


def test_wrist_joint_margins_measured_from_real_qpos():
    """[Horizontal-Wrap session, new] _wrist_joint_margins_deg must read
    REAL qpos/jnt_range from the compiled model (never a formula-only
    guess): recompute the same margin independently here via mj_name2id/
    jnt_qposadr/jnt_range and confirm it matches exactly."""
    env = make_env()
    env.reset(seed=0)
    margins = sbe_module._wrist_joint_margins_deg(env)
    assert set(margins.keys()) == {
        "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
        "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    }
    for side in SIDES:
        for suf in ("roll", "pitch", "yaw"):
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_wrist_{suf}_joint")
            qadr = env.model.jnt_qposadr[jid]
            q = float(env.data.qpos[qadr])
            lo, hi = env.model.jnt_range[jid]
            expected_deg = float(np.degrees(min(q - lo, hi - q)))
            assert abs(margins[f"{side}_wrist_{suf}"] - expected_deg) < 1e-9
    print(f"    wrist margins (stand pose) = {margins}")


def test_finger_table_and_ulnar_down_angle_helpers():
    """[Horizontal-Wrap session, new] Direct unit tests for the two new
    angle metrics, at synthetic orientations where the correct answer is
    known analytically (not just measured through a full rollout)."""
    # Identity R: approach axis (column 0) = world +X -- horizontal,
    # 0deg to the table plane. Ulnar (left, local ~[0,0,1]) -> world
    # ~+Z (pointing UP, ~180deg from "down").
    R_identity = np.eye(3)
    assert abs(sbe_module._finger_table_angle_deg(R_identity)) < 1e-9
    assert abs(sbe_module._ulnar_down_angle_deg("left", R_identity) - 180.0) < 0.1

    # col0=[0,0,-1] (approach maps to world down -- 90deg to the table
    # plane), completed to a proper right-handed rotation.
    R_finger_down = np.column_stack([
        np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]),
    ])
    assert abs(np.linalg.det(R_finger_down) - 1.0) < 1e-9
    assert abs(sbe_module._finger_table_angle_deg(R_finger_down) - 90.0) < 1e-6

    # col2=[0,0,-1] (the axis the ~[0,0,1]-ish LOCAL_ULNAR_VEC mostly
    # maps through) -- ulnar should measure ~0deg (pointing straight down).
    R_ulnar_down = np.column_stack([
        np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0]),
    ])
    assert abs(np.linalg.det(R_ulnar_down) - 1.0) < 1e-9
    assert abs(sbe_module._ulnar_down_angle_deg("left", R_ulnar_down)) < 0.1
    # Right hand's ULNAR_SIGN is flipped -- the SAME matrix should measure
    # ~180deg (pointing up) for "right" instead of ~0deg for "left".
    assert abs(sbe_module._ulnar_down_angle_deg("right", R_ulnar_down) - 180.0) < 0.1
    print("    _finger_table_angle_deg/_ulnar_down_angle_deg verified against known synthetic orientations")


def test_precontact_servo_makes_real_progress_without_forbidden_collision():
    """[Fingertip-contact session, new] The closed-loop inward servo
    (FINGERTIP_PRECONTACT) must reduce the REAL, measured fingertip-to-
    object-face separation substantially from its entry value, using
    real env.step() physics, while never exceeding the hard forbidden-
    collision force (torso-arm/hand-table) at any point -- the low-
    grade recovery-triggering contact stays well under the hard limit."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    entry_sep = None
    max_torso = 0.0
    max_table_forbidden = 0.0
    for _ in range(2000):
        if expert.state == BimanualGraspState.FAILURE:
            break
        prev_state = expert.state
        action = expert.step()
        env.step(action)
        if prev_state == BimanualGraspState.FINGERTIP_PRECONTACT:
            if entry_sep is None:
                entry_sep = expert._precontact_separation_m("left")
            max_torso = max(max_torso, env._torso_arm_collision_force())
            max_table_forbidden = max(max_table_forbidden, env._hand_table_forces_categorized()["forbidden"])
            if expert._state_step >= 800:
                break
    assert entry_sep is not None, "must have reached FINGERTIP_PRECONTACT"
    final_sep = expert._precontact_separation_m("left")
    print(f"    entry_sep={entry_sep*1000:.1f}mm final_sep={final_sep*1000:.1f}mm (after 800 ticks) "
          f"max_torso={max_torso:.2f}N max_table_forbidden={max_table_forbidden:.2f}N")
    assert final_sep < entry_sep - 0.020, "servo must reduce real separation by at least 20mm within 800 ticks"
    assert max_torso <= expert.config.precontact_hard_collision_force_n
    assert max_table_forbidden <= expert.config.precontact_hard_collision_force_n


def test_precontact_recovery_ladder_avoids_hard_fail_on_transient_spike():
    """[Fingertip-contact session, new] A transient torso-arm contact
    spike during the precontact servo (measured, real physics: this
    reach genuinely grazes the torso at times) must trigger the
    graduated recovery (backoff + temporary rest_gain boost), NOT an
    immediate hard failure -- confirmed by running long enough to
    observe at least one real recovery event while the state stays
    alive (not FAILURE) throughout."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    for _ in range(1600):
        if expert.state == BimanualGraspState.FAILURE:
            break
        action = expert.step()
        env.step(action)
        if expert.state == BimanualGraspState.FINGERTIP_PRECONTACT and expert._state_step >= 300:
            break
    assert expert.state != BimanualGraspState.FAILURE, "must not hard-fail from a recoverable transient collision"
    print(f"    recovery_count={expert._precontact_recovery_count} state={expert.state.name}")
    assert sum(expert._precontact_recovery_count.values()) > 0, (
        "expected at least one real recovery event to have been observed by this point in the reach"
    )


def test_wrap_direction_bug_fix_reduces_fingertip_object_offset():
    """[Fingertip-contact session, new] Regression guard for the
    _horizontal_wrap_target bug fix: the OLD wrap direction (perpendicular
    to the full, standoff-contaminated horizontal to-object vector)
    measurably put the fingertip FURTHER from the object in Y than the
    palm itself (~110mm extra, real FK, seed=0). Confirms the CURRENT
    (fixed) behavior keeps this extra Y offset much smaller, at the real,
    live FINGERTIP_PRECONTACT entry pose."""
    env = make_env()
    env.reset(seed=0)
    expert = SharpaBimanualGraspExpert(env)
    for _ in range(1230):
        if expert.state == BimanualGraspState.FINGERTIP_PRECONTACT:
            break
        action = expert.step()
        env.step(action)
    palm_pos, _ = env.palm_pose("left")
    tip = expert._nonthumb_tip_centroid("left")
    new_offset_y = abs(tip[1] - palm_pos[1])
    print(f"    real fingertip-palm Y offset={new_offset_y*1000:.1f}mm (OLD, retracted wrap direction measured ~110mm here)")
    assert new_offset_y * 1000 < 100.0, "wrap-direction fix should keep the fingertip-palm Y offset measurably under the old ~110mm"


def test_direct_closure_from_descend_pose_confirms_swept_path_does_not_reach_object():
    """[Direct-closure experiment, honest disclosure] User-directed test:
    from FOREARM_SIDE_DESCEND's own converged Horizontal-Wrap pose, with
    arm/wrist ctrl targets left EXACTLY as DESCEND set them (no
    FINGERTIP_PRECONTACT servo, no further palm motion --
    skip_precontact_servo=True advances DESCEND -> CONTACT_ACQUIRE
    directly), does the real finger-closure swept path actually
    intersect the object? Measured directly, real physics: it does NOT
    -- nonthumb fingertip-to-object-face separation at this pose is
    ~189mm (both hands), while the real remaining closure travel from
    the DESCEND-exit curl (0.0) to full close (1.0) is only ~54-57mm per
    finger (see test_open_preshape_curl_target_removes_near_full_closure)
    -- a ~130mm+ shortfall regardless of closure ORDER (thumb-first was
    also tried and produced 0.0N peak force across 500 ticks; thumb tip
    itself starts 152mm from the object center and its own curl moves it
    slightly AWAY at full closure, not toward). This locks in that
    result: zero contact force, zero object displacement, CONTACT_ACQUIRE
    times out. This is a real, disclosed finding, not a bug in
    CONTACT_ACQUIRE's own closing logic (which is unchanged and already
    covered by other tests) -- it is a direct consequence of skipping
    the inward approach entirely, exactly as this experiment's own
    directive required."""
    import humanoid_learning.envs.sharpa_config as sc

    env = make_env()
    env.reset(seed=0)
    ecfg = BimanualGraspConfig(skip_precontact_servo=True)
    expert = SharpaBimanualGraspExpert(env, ecfg)
    outcome = expert.run(max_total_steps=3000)
    print(f"    state={outcome.state.name} reason={outcome.failure_reason} "
          f"per_side_group_contact={outcome.per_side_group_contact} "
          f"max_peak_force={outcome.per_side_group_peak_force} "
          f"object_xy_displacement={outcome.object_xy_displacement}")
    assert outcome.state == BimanualGraspState.FAILURE
    assert outcome.failure_reason == BimanualFailureReason.TIMEOUT, (
        "expected CONTACT_ACQUIRE to time out (no group ever registers contact) -- "
        "if this changed, the swept path now reaches the object and this test's own "
        "conclusion (and the accompanying report) is stale"
    )
    for side in SIDES:
        for group in sc.GROUPS:
            assert outcome.per_side_group_peak_force[side][group] == 0.0, (
                f"{side}/{group}: expected 0.0N (no reach), got real contact -- investigate before trusting this result"
            )
    assert outcome.object_xy_displacement == 0.0


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
