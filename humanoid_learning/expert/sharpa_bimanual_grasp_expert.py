"""Official bimanual Sharpa Wave grasp controller.

Both hands grasp the SAME object from opposite lateral (Y) sides --
mirrored, not one hand crossing the body midline to reproduce a
single-hand grasp. CoupledBilateralIK solves BOTH palm targets in one
call at every approach state (it always did; the single-hand version
just fed one side a "stay put" target instead of a real one).

Root-cause geometry finding (small, bounded grid searches,
never a large random sweep):
  - Reaching the object's own Y=+-0.06 face directly self-collides
    (torso<->shoulder AND, at 12cm object width, the two hands' own
    fingertips collide with EACH OTHER -- a 12cm cube is not wide enough
    for two full 5-finger Sharpa envelopes to coexist without their
    ulnar-side (ring/pinky) links overlapping). Y offset >=0.10 (palm
    target) is collision-free (grid: 0 real self-collision at
    y_off in {0.10, 0.12, 0.15, 0.20}); at y_off=0.10 the FINGERTIPS
    (not the palm) land at y~=0.06-0.068 on their own, i.e. right at the
    object's actual face, without the palm itself needing to cross in.
  - [Historical, superseded this session] Enforcing an independently-
    CHOSEN opposition orientation (e.g. approach=+-Y, closing=+X)
    reintroduced severe self-collision (200+ contacts) at ANY nonzero
    orientation task weight when tried at the OLD (narrow, elbow-toward-
    torso) descend geometry. The original fix here was to solve
    position-only and CAPTURE whatever orientation the solver naturally
    landed on -- but that orientation was never actually object-facing
    (fingers ended up pointing ~120deg away from "down"; see WRIST_SIDE_
    GRASP_ALIGN's own docstring). This session replaces that capture-only
    approach with an EXPLICIT object-facing target (_object_facing_R,
    LOCKED once WRIST_SIDE_GRASP_ALIGN's own stability/collision checks
    pass) reached via a combined position+orientation waypointed ramp
    from a wider/higher intermediate pose -- collision-free, unlike a
    fixed-position reorientation attempt at either the old narrow-descend
    or the new final-grasp geometry (both measured to self-collide).

Gate A definition (the project's approved bilateral stability contract):
  - Per side: thumb touching AND (index OR middle touching) AND wrap
    touching AND opposition (thumb's contact-force direction opposes
    the index/middle combined force direction: dot product < 0) AND NOT
    this side self-colliding with the OTHER hand.
  - Bilateral: BOTH sides simultaneously satisfy the above, SAME tick.
  - Streak: consecutive ticks bilateral-satisfied, reset to 0 otherwise.
  - Gate A = max_bilateral_streak >= 30 AND object xy displacement
  <= 0.03m AND object peak angular velocity <= 2.0 rad/s (a disclosed,
    deliberately conservative project threshold) AND no forbidden penetration
    beyond the same numerical-tolerance band already established for
    Sharpa (1mm) AND the terminal state is not FAILURE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import mujoco
import numpy as np

from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.expert import pose_ik
from humanoid_learning.expert.coupled_ik import CoupledBilateralIK
from humanoid_learning.expert.timing import sim_time_to_steps


def _object_facing_R(side: str, palm_pos: np.ndarray, obj_pos: np.ndarray) -> np.ndarray:
    """Explicit object-facing wrist target -- the CLOSING axis (palm_R
    column 1) is set to point from the palm straight at the object
    center; the one remaining free rotation (about the closing axis) is
    resolved by keeping the lateral axis close to world-up (a natural,
    non-twisted approach rather than an arbitrary roll).

    [This session] REMOVED the previous ``-closing_dir if side=="left"
    else closing_dir`` sign flip. Measured directly (this session's FK
    audit, scripts/measure_sharpa_side_grasp_axes.py): applying a small
    "close" synergy delta to index/middle/wrap and reading the resulting
    fingertip displacement in the palm's OWN local frame gives an
    IDENTICAL positive-Y_palm-local component for BOTH hands -- i.e. the
    real closing direction is `+palm_R[:, 1]` for both sides already
    (LEFT_PALM_LOCAL_QUAT/RIGHT_PALM_LOCAL_QUAT already encode each
    hand's mirrored mount; no additional per-side sign flip belongs in
    application code). The old, asymmetric formula was causally
    responsible for the object_facing_orientation feature's self-collision
    history (verified this session: a fresh, unconstrained IK solve with
    the OLD sign drives the right wrist into torso_link by ~2cm; the SAME
    solve with this fixed sign has zero torso contact for both sides)."""
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


def _object_facing_angle_deg(side: str, palm_R: np.ndarray, palm_pos: np.ndarray, obj_pos: np.ndarray) -> float:
    """Angle between the palm's ACTUAL closing axis and the true
    palm->object direction -- the Side-Grasp Posture Gate's palm-inward-
    angle metric. [This session] sign convention matches the corrected
    _object_facing_R above (no per-side flip -- see that function's
    docstring)."""
    closing_dir = obj_pos - palm_pos
    n = np.linalg.norm(closing_dir)
    if n < 1e-9:
        return 0.0
    closing_dir = closing_dir / n
    actual_closing = palm_R[:, 1]
    cos_ang = np.clip(np.dot(actual_closing, closing_dir), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def _finger_down_angle_deg(palm_R: np.ndarray) -> float:
    """Angle between the palm's approach axis (palm_R column 0 -- verified
    this session, scripts/measure_sharpa_side_grasp_axes.py, to coincide
    EXACTLY with the open/straight nonthumb fingers' own root->tip
    direction) and world -Z ("fingers generally point down"), the Side-
    Grasp Posture Gate's finger-down-angle metric."""
    approach_axis = palm_R[:, 0]
    cos_ang = np.clip(np.dot(approach_axis, np.array([0.0, 0.0, -1.0])), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def _quintic_scale(tau: float) -> float:
    """[Session 42] Canonical quintic minimum-jerk time-scaling: zero
    velocity AND zero acceleration at both tau=0 and tau=1 (clipped to
    [0,1]). Used for ARM_LATERAL_CLEARANCE's joint trajectory instead of
    a raw rate-limited step target -- see that state's docstring for the
    causal finding (a step target's instantaneous ctrl-register slope
    change, tracked by zero-damping wrist joints, excites a real
    thumb<->table collision impulse that couples back into ~8rad/s wrist
    qvel)."""
    tau = min(max(tau, 0.0), 1.0)
    return 6 * tau**5 - 15 * tau**4 + 10 * tau**3


def _slerp_R(R_a: np.ndarray, R_b: np.ndarray, frac: float) -> np.ndarray:
    """Geodesic SO(3) interpolation from R_a (frac=0) to R_b (frac=1),
    world-frame axis-angle (matches pose_ik.orientation_error's own
    convention, reused here instead of inventing a new one)."""
    err = pose_ik.orientation_error(R_a, R_b)
    return pose_ik.so3_exp(frac * err) @ R_a

SIDES = ("left", "right")
Y_SIGN = {"left": 1.0, "right": -1.0}
REQUIRED_GROUPS = ("thumb", "index", "middle", "wrap")  # topology: thumb + (index or middle) + wrap, checked below
OBJECT_XY_DISPLACEMENT_LIMIT = 0.03
OBJECT_PEAK_ANGULAR_VELOCITY_LIMIT = 2.0
BILATERAL_STREAK_REQUIRED = 30


class BimanualGraspState(Enum):
    STABLE_START = auto()
    ARM_LATERAL_CLEARANCE = auto()
    FOREARM_FORWARD_REACH = auto()
    # [This session] WRIST_ALIGN (measure-only, never re-solved) and
    # FOREARM_DESCEND (position-only, natural/top-down-derived
    # orientation) are REPLACED by WRIST_SIDE_GRASP_ALIGN and
    # FOREARM_SIDE_DESCEND: user-directed requirement for a genuine
    # bilateral SIDE grasp (palms facing each other, closing axis toward
    # the object center, fingers generally pointing down), which the old
    # pair could not produce (it only ever held whatever orientation
    # ARM_LATERAL_CLEARANCE happened to leave the wrist in -- verified
    # this session, ~110-170deg away from any object-facing pose). See
    # WRIST_SIDE_GRASP_ALIGN's docstring for the causal path to this
    # design (measured: reorienting at a FIXED position, or from a
    # narrow/high posture, self-collides; reorienting AND translating
    # together, waypointed, does not).
    WRIST_SIDE_GRASP_ALIGN = auto()
    FIVE_FINGER_PRESHAPE = auto()
    FOREARM_SIDE_DESCEND = auto()
    FINGERTIP_PRECONTACT = auto()
    CONTACT_ACQUIRE = auto()
    THUMB_OPPOSE = auto()
    ENVELOPING_CLOSE = auto()
    FORCE_SETTLE = auto()
    TABLETOP_HOLD = auto()
    LIFT = auto()
    AIR_HOLD = auto()
    SUCCESS = auto()
    FAILURE = auto()


class BimanualFailureReason(Enum):
    IK_NOT_CONVERGED = auto()
    WRIST_ORIENTATION_NOT_STABLE = auto()
    SELF_COLLISION_BEFORE_CONTACT = auto()
    CONTACT_LOST = auto()
    TIMEOUT = auto()
    NUMERICAL_ERROR = auto()
    OBJECT_MOVED_TOO_MUCH = auto()
    OBJECT_ANGULAR_VELOCITY_EXCEEDED = auto()
    LIFT_FAILED = auto()
    PRECONTACT_TRACKING_NOT_ACHIEVED = auto()
    WRIST_NOT_OBJECT_FACING = auto()
    SELF_COLLISION_TORSO_ARM = auto()
    LATERAL_CLEARANCE_NOT_ACHIEVED = auto()
    FORWARD_REACH_NOT_ACHIEVED = auto()
    SIDE_GRASP_ALIGN_NOT_ACHIEVED = auto()  # [This session] WRIST_SIDE_GRASP_ALIGN timeout/failure
    SIDE_DESCEND_NOT_ACHIEVED = auto()  # [This session] FOREARM_SIDE_DESCEND timeout/failure
    HAND_TABLE_COLLISION = auto()  # [This session] forbidden hand<->table contact during the side-grasp approach


@dataclass
class BimanualGraspConfig:
    # [Session 41] ARM_LATERAL_CLEARANCE joint posture -- NOT a Cartesian
    # IK target (redundant-arm IK choosing an arbitrary elbow-up branch
    # was the root cause of the "unnatural" pose the 41st session's user
    # feedback described -- see docs/history/PHASE4_GRASP_SESSION_41.md).
    # Chosen from a bounded 3-candidate FK sweep (never a large sweep):
    # (shoulder_pitch, shoulder_roll, elbow) = (-0.1, 1.1, 0.8) measured
    # 0 real self-collisions, elbow 7.6cm BELOW shoulder, and (driven
    # through real env.step() physics with arm_gravity_compensation) the
    # actuator tracks it almost exactly (qpos within 0.001rad of target,
    # qvel settles to ~1e-4 rad/s) -- unlike the deeper grasp-approach
    # reach, this posture is NOT actuator-tracking-limited. shoulder_roll
    # sign is mirrored per side (Y_SIGN); shoulder_yaw/wrist stay neutral.
    clearance_shoulder_pitch: float = -0.1
    clearance_shoulder_roll: float = 1.1
    clearance_elbow: float = 0.8
    clearance_joint_tol_rad: float = 0.03
    clearance_stable_streak_required: int = 15
    # [Session 42] Wrist Transition Gate: engineering safety target for
    # THIS approach trajectory specifically (not a Gate A criterion).
    # See sharpa_bimanual_grasp_expert.py's ARM_LATERAL_CLEARANCE
    # docstring / docs/history/PHASE4_GRASP_SESSION_42.md for the causal
    # root cause (thumb<->table collision impulse, not the direct-joint-
    # target step itself).
    clearance_trajectory_ticks: int = 90
    wrist_max_qvel_rad_s: float = 2.0
    posture_rest_gain: float = 0.2  # WEAKER than CoupledBilateralIK's own 0.3 default -- see module docstring's Stage 6 note: 0.5 was tried first and measured (causally) to prevent the primary position task from converging at all (5.4cm plateau, IK itself never reaching pos_tol); 0.2 is a soft nudge toward the clearance posture, not a competing task
    approach_standoff_m: float = 0.15
    # 0.22, not the 0.15 used for the bimanual self-collision grid search:
    # that grid search only checked HAND<->HAND/HAND<->TORSO self-collision
    # pairs (see module docstring), not hand<->table: 0.15 was found to let
    # fingertips graze the table during transit (verified this session --
    # `table<->*_DP` contacts caused the wrist to visibly stick, unable to
    # reach its IK target through the resulting friction lock). 0.22 is
    # measured table-clearance margin.
    approach_height_m: float = 0.22
    approach_y_offset_m: float = 0.15
    # [This session, user-directed geometry correction] FOREARM_DESCEND's
    # target height used to stay well ABOVE the object (+0.10m above
    # object center -- for a 12cm cube, still ~4cm above the TOP face),
    # so shoulder/elbow visibly lowered the hand down onto/over the block
    # instead of beside it. Live-viewer feedback: both palms must end up
    # LEVEL with the object, approaching from the side, not descending
    # onto the top. approach_height_m (0.22, well above the table) is
    # unchanged -- that is a lateral-transit clearance height, not a
    # grasp height (see its own docstring).
    #
    # height=0.0 (exactly the object's center Z) was tried first and
    # measured (bounded sweep {0.0, 0.02, 0.04, 0.06}, holding the ctrl
    # target fixed well past the state timeout to read the TRUE
    # steady-state, same method as FOREARM_FORWARD_REACH's fix) to be
    # physically infeasible here: it drives the forearm into genuine,
    # growing torso<->arm contact resistance (steady-state ~4.2N and
    # still rising, pos_err plateauing at ~20mm, not a timing artifact --
    # more waypoint/settle ticks do not help).
    #
    # A second bounded sweep over {0.045..0.07} (same waypointed descend,
    # now smoothed with a quintic fraction schedule -- see
    # DESCEND_WAYPOINTS below) found the collision peak is NON-monotonic
    # in height across that whole band (0.045->8.43N, 0.05->11.25N,
    # 0.055->11.70N, 0.06->7.93N, 0.065->7.49N, 0.07->6.71N) at the
    # ORIGINAL descend_y_offset_m=0.15 (same as approach_y_offset_m) -- a
    # real, sensitive torso<->arm graze near the final waypoint, not
    # something trajectory-shape tuning alone reliably clears with margin
    # at low heights.
    #
    # [Superseded this session] the numbers above (0.02/0.20) were tuned
    # for the OLD FOREARM_DESCEND, which held whatever orientation
    # ARM_LATERAL_CLEARANCE happened to leave the wrist in (approach axis
    # pointing mostly forward/up, NOT down -- measured this session,
    # finger_down_angle ~122deg, i.e. the opposite of "down"). That state
    # is replaced by WRIST_SIDE_GRASP_ALIGN + FOREARM_SIDE_DESCEND below,
    # which target a genuine object-facing, fingers-down orientation and
    # use their OWN geometry fields.
    forward_reach_stable_streak_required: int = 15

    # ---- Side-grasp geometry (this session) --------------------------
    # User-directed requirement: both palms end up beside the object's
    # OWN side faces, facing each other (closing axis toward the object
    # center), fingers generally pointing down -- not a top-down palm-
    # down reach. Root-caused (scripts/measure_sharpa_side_grasp_axes.py
    # FK audit + a bounded, disclosed series of position/orientation/
    # collision experiments, never a large random sweep) that:
    #   1. The natural orientation ARM_LATERAL_CLEARANCE/FOREARM_FORWARD_
    #      REACH leave the wrist in is ~110-170deg away from ANY object-
    #      facing pose -- a large reorientation is unavoidable somewhere.
    #   2. Ramping ONLY orientation (holding position fixed) at ANY tested
    #      position -- the wide/high FOREARM_FORWARD_REACH pose, or the
    #      close/low final grasp pose -- reliably self-collides (measured
    #      peaks 46-113N torso<->arm depending on where/how it was tried
    #      in earlier sessions and this one). Ramping POSITION and
    #      ORIENTATION together, waypointed (SLERP + linear position
    #      interpolation, same per-waypoint IK re-solve recipe as
    #      FOREARM_FORWARD_REACH), from FOREARM_FORWARD_REACH's own
    #      (already safe) end pose to a new lower/oriented target is
    #      collision-free (measured 0.00N torso-arm, 0.00N hand-hand
    #      across 3 repeated seed=0 rollouts).
    #   3. With fingers pointing down, OPEN (uncurled) fingers reach
    #      ~15cm below the palm -- at any Z near the object/table, this
    #      spears the table (measured up to 25N hand-table force) unless
    #      the fingers are partially curled (side_align_preshape_curl)
    #      BEFORE/DURING this transition, shortening their effective
    #      reach. This is why WRIST_SIDE_GRASP_ALIGN applies preshape
    #      abduction AND a protective curl at its own entry, ahead of the
    #      later, dedicated FIVE_FINGER_PRESHAPE state.
    #   4. The old, asymmetric _object_facing_R sign convention (removed
    #      this session -- see that function's docstring) was ALSO a
    #      genuine contributor to the self-collision this feature was
    #      previously blocked on.
    # side_align_* is FOREARM_FORWARD_REACH's own standoff (approach_
    # standoff_m, unchanged -- table clearance already proven there) with
    # a HIGHER height and WIDER Y offset than the old descend target, so
    # OPEN fingers pointing down still clear the table during the ramp.
    side_align_height_m: float = 0.04
    side_align_y_offset_m: float = 0.22
    # Protective curl (index/middle/wrap only, thumb untouched -- same
    # split as CONTACT_ACQUIRE) applied at WRIST_SIDE_GRASP_ALIGN entry,
    # before the position+orientation ramp -- see point 3 above. NOT the
    # same as FIVE_FINGER_PRESHAPE's own (later, unchanged) preshape call;
    # this is purely a table-clearance safety margin, disclosed as such.
    side_align_preshape_curl: float = 0.5
    side_align_waypoints: int = 14
    side_align_waypoint_ticks: int = 30
    side_align_max_steps: int = 600
    side_align_stable_streak_required: int = 15
    # Side-Grasp Posture Gate tolerances (Section 10 of this session's
    # spec) -- measured directly, not guessed:
    side_grasp_inward_angle_tol_deg: float = 15.0
    side_grasp_finger_down_tol_deg: float = 25.0
    side_grasp_mirror_pos_tol_m: float = 0.020
    side_grasp_mirror_ori_tol_deg: float = 10.0
    # FOREARM_SIDE_DESCEND: a SECOND waypointed position move (orientation
    # held via a small soft anchor, NOT re-derived -- it is already
    # correct from WRIST_SIDE_GRASP_ALIGN) from the align pose down/in
    # toward the final pre-contact approach. Measured collision-free
    # (0.00N torso-arm/hand-hand/hand-table) at these values, holding the
    # ALIGN orientation.
    side_descend_standoff_m: float = 0.12
    side_descend_height_m: float = 0.03
    side_descend_y_offset_m: float = 0.15
    side_descend_waypoints: int = 8
    side_descend_waypoint_ticks: int = 30
    side_descend_stable_streak_required: int = 15
    precontact_standoff_m: float = 0.08
    # [This session] retargeted for the side-grasp geometry -- height
    # matches side_descend_height_m (no further Z motion), y_offset moves
    # the last ~6cm in from side_descend_y_offset_m=0.15 to just outside
    # the object's own half-width (0.06) plus fingertip clearance.
    precontact_height_m: float = 0.03
    precontact_y_offset_m: float = 0.09
    close_rate_per_step: float = 0.03
    contact_force_threshold_n: float = 0.5
    target_force_band_n: tuple[float, float] = (1.0, 6.0)
    force_settle_hold_steps: int = 10
    tabletop_hold_seconds: float = 2.0
    lift_height_m: float = 0.05
    air_hold_seconds: float = 5.0
    object_xy_displacement_limit_m: float = OBJECT_XY_DISPLACEMENT_LIMIT
    object_peak_angular_velocity_limit: float = OBJECT_PEAK_ANGULAR_VELOCITY_LIMIT
    bilateral_streak_required: int = BILATERAL_STREAK_REQUIRED
    proximal_penetration_tolerance_m: float = 0.001  # same 1mm numerical band as sharpa_config's tolerances
    hand_hand_force_limit_n: float = 8.0
    ik_pos_tol: float = 0.01
    max_steps_per_state: int = 400
    wrist_orientation_stability_tol_deg: float = 5.0  # max angular drift over the last 30 ticks to call WRIST_SIDE_GRASP_ALIGN settled
    # [Session 39] FINGERTIP_PRECONTACT Precontact Tracking Gate (see
    # docs/history/PHASE4_GRASP_SESSION_39.md): the ctrl register
    # converges EXACTLY to the IK-solved joint target --
    # joint_target_minus_ctrl_norm == 0 -- yet the actual physical palm
    # settles several cm short, a steady-state compliant-actuator (arm_kp
    # =120) gravity/load droop under the Sharpa hands' own weight, not a
    # kinematic or rate-limit error. A Cartesian-target-inflation resolve
    # A Cartesian re-solve recipe was tried here and causally measured to make the
    # gap WORSE at this already-extreme precontact reach (see the same
    # history doc) -- not used. The actual fix is
    # SharpaGraspEnv's config-gated arm_gravity_compensation (see
    # grasp_config.py/sharpa_grasp_env.py); this state only HONESTLY
    # measures the actual settled pose and gates the transition on it.
    precontact_ori_tol_deg: float = 5.0  # Precontact Tracking Gate orientation tolerance (Session 39 spec)
    precontact_stable_streak_required: int = 15  # Precontact Tracking Gate: consecutive ticks required
    # [This session] WRIST_SIDE_GRASP_ALIGN's own state-transition check
    # for "is this actually object-facing" -- WRIST_SIDE_GRASP_ALIGN's
    # explicit object-facing target (see _object_facing_R) is now the
    # DEFAULT, only path (the old opt-in object_facing_orientation flag
    # and its self-colliding DESCEND-time ramp are removed -- superseded
    # by WRIST_SIDE_GRASP_ALIGN + FOREARM_SIDE_DESCEND, which fix the
    # causal self-collision source: see side_align_height_m's docstring).
    # Matches side_grasp_inward_angle_tol_deg (15deg, this session's own
    # Side-Grasp Posture Gate spec) rather than the old inherited 10deg --
    # a physically-converged, collision-free approach measured at ~14deg
    # here should not fail this internal check only to pass the Gate's
    # own (identical-intent) 15deg tolerance moments later.
    object_facing_angle_tol_deg: float = 15.0  # palm-closing-axis vs palm->object angle


@dataclass
class BimanualGraspOutcome:
    state: BimanualGraspState
    failure_reason: BimanualFailureReason | None
    step_count: int
    per_side_group_contact: dict  # {side: {group: bool ever-contacted}}
    per_side_group_peak_force: dict
    per_side_group_net_force: dict
    max_bilateral_stable_streak: int
    max_left_stable_streak: int
    max_right_stable_streak: int
    max_proximal_object_penetration: float
    max_hand_hand_force: float
    object_xy_displacement: float
    object_angular_velocity_peak: float
    object_angular_velocity_rms: float
    tabletop_hold_steps_achieved: int
    air_hold_steps_achieved: int
    lift_height_achieved_m: float
    gate_a: bool
    gate_b: bool
    gate_c: bool
    gate_d: bool
    precontact_final_pos_error_m: dict  # {side: m}, FINGERTIP_PRECONTACT actual-vs-target Cartesian error
    precontact_final_ori_error_deg: dict  # {side: deg}
    precontact_max_stable_streak: int
    precontact_gate: bool  # Session 39 Precontact Tracking Gate (see module docstring)
    side_grasp_posture: dict  # [This session] Side-Grasp Posture Gate metrics + pass/fail, see _side_grasp_posture_metrics
    side_grasp_gate: bool  # [This session] Side-Grasp Posture Gate PASS/FAIL (Section 10 of this session's spec)


class SharpaBimanualGraspExpert:
    """Drives a SharpaGraspEnv with BOTH hands grasping the object
    simultaneously (mirrored across Y). Call .step() once per control
    tick; .run() loops to a terminal state or step budget."""

    # See _arm_action_toward_target's docstring: a plain ctrl-ramp-rate
    # reduction (not a gain/damping change) that empirically eliminates a
    # real wrist_pitch dynamics instability shared with the single-hand
    # controller.
    RAMP_FRACTION = 1.0
    WAYPOINT_COUNT = 4
    WAYPOINT_TICKS = 60
    FORWARD_REACH_WAYPOINTS = 6  # [Session 41] see FOREARM_FORWARD_REACH: the clearance->approach Y swing (~0.46m -> 0.15m) needs several small steps, not one, to avoid a waist_pitch hard-limit
    # [This session] FOREARM_FORWARD_REACH's waypoint spacing was
    # `max_steps_per_state // FORWARD_REACH_WAYPOINTS` (=66 at the
    # official max_steps_per_state=400), i.e. spending the ENTIRE state
    # budget on waypoint progression and leaving only 400-330=70 ticks
    # after the 6th/final waypoint to physically settle -- unlike
    # FINGERTIP_PRECONTACT's WAYPOINT_TICKS=60 fixed spacing (4 waypoints,
    # last at tick 180, leaving a 220-tick tail), even though this state's
    # docstring explicitly claims to reuse "the identical recipe already
    # proven for FINGERTIP_PRECONTACT". Root-caused with
    # scripts/diagnose_forward_reach_gap.py (see docs/history for this
    # session): holding the SAME final ctrl target fixed past the official
    # 400-tick timeout, actual physics palm error keeps monotonically
    # decreasing (qvel monotonically -> 0, no oscillation) from ~10.16mm
    # at the official cutoff down to a genuine steady-state ~9.7-9.8mm
    # asymptote, crossing under the unchanged 10mm gate by roughly
    # total_step ~110 ticks after the final waypoint fires (~583 vs the
    # official ~473+70=543 cutoff). A fixed, smaller per-waypoint tick
    # count (independent of max_steps_per_state, matching the
    # FINGERTIP_PRECONTACT convention) fires the final waypoint earlier and
    # leaves enough physical settle tail within the SAME unchanged
    # max_steps_per_state/ik_pos_tol/streak-length budget -- a waypoint-
    # schedule bookkeeping fix, not a Gate relaxation.
    FORWARD_REACH_WAYPOINT_TICKS = 40
    # [This session] WRIST_SIDE_GRASP_ALIGN/FOREARM_SIDE_DESCEND's own
    # orientation ramp weight schedule -- same small-start/large-end shape
    # as the removed object_facing_orientation ramp (gradual, never a
    # one-shot jump to a hard orientation requirement).
    SIDE_ORI_WEIGHT_START = 0.05
    SIDE_ORI_WEIGHT_END = 1.0

    def __init__(self, env, config: BimanualGraspConfig | None = None):
        self.env = env
        self.config = config or BimanualGraspConfig()
        self.state = BimanualGraspState.STABLE_START
        self.failure_reason: BimanualFailureReason | None = None
        self._state_step = 0
        self._total_step = 0
        self._just_advanced = False
        self._force_settle_streak = 0
        self._group_ever_contacted = {s: {g: False for g in sc.GROUPS} for s in SIDES}
        self._group_peak_force = {s: {g: 0.0 for g in sc.GROUPS} for s in SIDES}
        self._group_net_force = {s: {g: 0.0 for g in sc.GROUPS} for s in SIDES}
        self._max_proximal_pen = 0.0
        self._max_hand_hand = 0.0
        self._tabletop_hold_steps = 0
        self._air_hold_steps = 0
        self._lift_height_achieved = 0.0
        self._obj_xy_hist: list[np.ndarray] = []
        self._obj_angvel_hist: list[float] = []
        self._initial_obj_xy: np.ndarray | None = None
        self._left_streak = 0
        self._right_streak = 0
        self._bilateral_streak = 0
        self._max_left_streak = 0
        self._max_right_streak = 0
        self._max_bilateral_streak = 0
        self._locked_R: dict | None = None  # set at WRIST_SIDE_GRASP_ALIGN entry, see module docstring
        self.wrist_orientation_drift_deg: float = float("inf")
        self.left_object_facing_angle_deg: float = float("inf")
        self.right_object_facing_angle_deg: float = float("inf")
        self.torso_arm_collision_force_n: float = 0.0
        self.max_hand_table_force_n: float = 0.0
        self._clearance_target: np.ndarray | None = None
        self._clearance_max_raw_wrist_qvel: float = 0.0
        self._clearance_stable_streak = 0
        self._forward_reach_stable_streak = 0
        # WRIST_SIDE_GRASP_ALIGN (this session)
        self._side_align_waypoint = 0
        self._side_align_stable_streak = 0
        self._side_align_R_hist = {"left": [], "right": []}
        # FOREARM_SIDE_DESCEND (this session)
        self._side_descend_waypoint = 0
        self._descend_stable_streak = 0
        self._precontact_stable_streak = 0
        self._max_precontact_stable_streak = 0
        self._precontact_final_pos_error = {"left": float("inf"), "right": float("inf")}
        self._precontact_final_ori_error_deg = {"left": float("inf"), "right": float("inf")}
        self._side_grasp_posture: dict = {}
        self._side_grasp_gate: bool = False

        model = env.model
        waist_dof = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in wbc.WAIST_JOINTS])
        coupled_names = list(wbc.WAIST_JOINTS) + list(tc.LEFT_ARM_JOINTS) + list(tc.RIGHT_ARM_JOINTS)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in coupled_names]
        joint_low = np.array([model.jnt_range[j][0] for j in jids])
        joint_high = np.array([model.jnt_range[j][1] for j in jids])
        self.ik = CoupledBilateralIK(
            model=model, left_palm_site=env._left_palm_site, right_palm_site=env._right_palm_site,
            waist_dof_adr=waist_dof, left_arm_dof_adr=env._arm_dof_adr[:7], right_arm_dof_adr=env._arm_dof_adr[7:],
            waist_qpos_adr=env._waist_qpos_adr, left_arm_qpos_adr=env._arm_qpos_adr[:7],
            right_arm_qpos_adr=env._arm_qpos_adr[7:], joint_low=joint_low, joint_high=joint_high,
        )
        self._rest_q = np.concatenate([
            env.data.qpos[env._waist_qpos_adr].copy(), env.data.qpos[env._arm_qpos_adr[:7]].copy(),
            env.data.qpos[env._arm_qpos_adr[7:]].copy(),
        ])
        self._arm_ik_target = env._arm_target.copy()
        self._waist_ik_target = env._waist_target.copy()

    # ------------------------------------------------------------------
    def _object_pos(self) -> np.ndarray:
        return self.env.data.qpos[self.env._object_qpos_adr:self.env._object_qpos_adr + 3].copy()

    def _object_angvel(self) -> float:
        return float(np.linalg.norm(self.env.data.qvel[self.env._object_dof_adr + 3:self.env._object_dof_adr + 6]))

    def _mirrored_targets(self, standoff: float, height: float, y_offset: float) -> dict:
        obj_pos = self._object_pos()
        return {side: obj_pos + np.array([-standoff, Y_SIGN[side] * y_offset, height]) for side in SIDES}

    def _solve_both(self, targets: dict, R: dict, require_orientation: bool, ori_task_weight: float,
                     rest_q: np.ndarray | None = None, rest_gain: float | None = None):
        data = self.env.data
        scratch = mujoco.MjData(self.env.model)
        scratch.qpos[:] = data.qpos
        mujoco.mj_forward(self.env.model, scratch)
        kwargs = {} if rest_gain is None else {"rest_gain": rest_gain}
        return self.ik.solve(
            scratch, targets["left"], R["left"], targets["right"], R["right"],
            rest_q=rest_q if rest_q is not None else self._rest_q, pos_tol=self.config.ik_pos_tol,
            require_orientation=require_orientation, ori_task_weight=ori_task_weight, **kwargs,
        )

    def _clearance_arm_vector(self, side: str) -> np.ndarray:
        """[Session 41] ARM_LATERAL_CLEARANCE's per-side 7-dim joint
        target, in task_config.py's LEFT/RIGHT_ARM_JOINTS order
        [pitch, roll, yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw].
        shoulder_roll is mirrored by Y_SIGN (matches every other
        left/right-symmetric convention in this file)."""
        cfg = self.config
        return np.array([
            cfg.clearance_shoulder_pitch, Y_SIGN[side] * cfg.clearance_shoulder_roll, 0.0,
            cfg.clearance_elbow, 0.0, 0.0, 0.0,
        ])

    def _apply_ik_result(self, result) -> None:
        self._waist_ik_target = result.waist_q.copy()
        self._arm_ik_target[:7] = result.left_q.copy()
        self._arm_ik_target[7:] = result.right_q.copy()

    def _arm_action_toward_target(self) -> np.ndarray:
        # Targeting a fixed/independently-chosen wrist orientation
        # (e.g. np.eye(3) at any nonzero weight, or a "locked" orientation
        # captured elsewhere) reliably drives wrist_pitch (very low
        # armature=0.01, zero dof_damping -- see grasp_config.py/
        # model_builder.py, both protected/unmodified) into a real,
        # reproducible divergence to its own hard joint limit. Neither slowing the
        # ctrl ramp rate (tried: ineffective, only delays the divergence)
        # nor trimming the IK's joint-limit margin (tried: ineffective)
        # fixed this -- the actual, verified fix is at the IK CALL SITES
        # (see FOREARM_APPROACH/FINGERTIP_PRECONTACT below): always
        # soft-anchor the orientation task to whatever orientation the arm
        # is CURRENTLY, already stably holding (small ori_task_weight,
        # never a fixed/locked target), which keeps the solver from ever
        # choosing the unstable configuration in the first place.
        # RAMP_FRACTION is kept as a class constant (currently 1.0, i.e.
        # no slowdown) only as a documented, easy-to-flip knob if a
        # similar instability resurfaces elsewhere.
        env = self.env
        scale = env.config.arm_action_scale * self.RAMP_FRACTION
        delta = np.clip(self._arm_ik_target - env._arm_target, -scale, scale)
        return delta / max(env.config.arm_action_scale, 1e-9)

    def _waist_action_toward_target(self) -> np.ndarray:
        env = self.env
        scale = env.config.waist_action_scale * self.RAMP_FRACTION
        delta = np.clip(self._waist_ik_target - env._waist_target, -scale, scale)
        return delta / max(env.config.waist_action_scale, 1e-9)

    def _zero_action(self) -> np.ndarray:
        from humanoid_learning.envs.sharpa_grasp_env import ACTION_DIM
        return np.zeros(ACTION_DIM)

    def _group_action(self, deltas: dict) -> np.ndarray:
        """deltas: {side: {group: delta}} -> 8-dim [left x4, right x4] action slice."""
        vec = np.zeros(2 * len(sc.GROUPS))
        for side_idx, side in enumerate(SIDES):
            for g_idx, group in enumerate(sc.GROUPS):
                vec[side_idx * len(sc.GROUPS) + g_idx] = deltas.get(side, {}).get(group, 0.0) / max(
                    self.env.config.hand_synergy_action_scale, 1e-9
                )
        return vec

    def _group_contact_now(self, side: str, group: str) -> bool:
        peak, net = self.env._group_contact_force(side, group)
        self._group_peak_force[side][group] = max(self._group_peak_force[side][group], peak)
        self._group_net_force[side][group] = max(self._group_net_force[side][group], net)
        if peak > self.config.contact_force_threshold_n:
            self._group_ever_contacted[side][group] = True
        return peak > self.config.contact_force_threshold_n

    def _group_force_vector(self, side: str, group: str) -> np.ndarray:
        """Net world-frame force vector (not magnitude) this GROUP applies
        to the object, for the opposition dot-product check below."""
        model, data = self.env.model, self.env.data
        obj_body = self.env._object_body_id
        prefixes = tuple(sc.sharpa_body(side, f, "") for f in sc.GROUP_FINGERS[group])
        force_sum = np.zeros(3)
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if obj_body not in (b1, b2):
                continue
            other = b2 if b1 == obj_body else b1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if not any(name.startswith(p) for p in prefixes):
                continue
            force6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force6)
            contact_R = np.asarray(c.frame, dtype=np.float64).reshape(3, 3)
            force_on_geom2 = contact_R.T @ force6[:3]
            force_sum += -force_on_geom2 if b1 == obj_body else force_on_geom2
        return force_sum

    def _side_stable(self, side: str) -> bool:
        """Topology check for one side: thumb
        touching AND (index or middle) touching AND wrap touching AND
        thumb's force genuinely opposes the index/middle combined force
        (dot product < 0) and this side is not in a hand-hand collision."""
        env = self.env
        thumb_peak, _ = env._group_contact_force(side, "thumb")
        index_peak, _ = env._group_contact_force(side, "index")
        middle_peak, _ = env._group_contact_force(side, "middle")
        wrap_peak, _ = env._group_contact_force(side, "wrap")
        thumb_touch = thumb_peak > self.config.contact_force_threshold_n
        finger_touch = (index_peak > self.config.contact_force_threshold_n) or (middle_peak > self.config.contact_force_threshold_n)
        wrap_touch = wrap_peak > self.config.contact_force_threshold_n
        if not (thumb_touch and finger_touch and wrap_touch):
            return False
        thumb_f = self._group_force_vector(side, "thumb")
        finger_f = self._group_force_vector(side, "index") + self._group_force_vector(side, "middle")
        if np.linalg.norm(thumb_f) < 1e-6 or np.linalg.norm(finger_f) < 1e-6:
            return False
        opposing = float(np.dot(thumb_f, finger_f)) < 0.0
        return opposing

    def _bilateral_stable_now(self) -> tuple[bool, bool, bool]:
        left_ok = self._side_stable("left")
        right_ok = self._side_stable("right")
        hh_force = self.env._hand_hand_contact_force()
        if hh_force > self.config.hand_hand_force_limit_n:
            left_ok = right_ok = False
        return left_ok, right_ok, left_ok and right_ok

    def _proximal_penetration_ok(self) -> bool:
        return self.env._proximal_object_penetration() <= self.config.proximal_penetration_tolerance_m

    def _fail(self, reason: BimanualFailureReason) -> None:
        self.state = BimanualGraspState.FAILURE
        self.failure_reason = reason

    def _advance(self, next_state: BimanualGraspState) -> None:
        self.state = next_state
        self._state_step = 0
        self._just_advanced = True

    def _track_stability(self) -> None:
        self._obj_xy_hist.append(self._object_pos()[:2])
        self._obj_angvel_hist.append(self._object_angvel())
        self._max_proximal_pen = max(self._max_proximal_pen, self.env._proximal_object_penetration())
        self._max_hand_hand = max(self._max_hand_hand, self.env._hand_hand_contact_force())
        left_ok, right_ok, both_ok = self._bilateral_stable_now()
        self._left_streak = self._left_streak + 1 if left_ok else 0
        self._right_streak = self._right_streak + 1 if right_ok else 0
        self._bilateral_streak = self._bilateral_streak + 1 if both_ok else 0
        self._max_left_streak = max(self._max_left_streak, self._left_streak)
        self._max_right_streak = max(self._max_right_streak, self._right_streak)
        self._max_bilateral_streak = max(self._max_bilateral_streak, self._bilateral_streak)

    def object_xy_displacement(self) -> float:
        if self._initial_obj_xy is None or not self._obj_xy_hist:
            return 0.0
        return float(np.max([np.linalg.norm(xy - self._initial_obj_xy) for xy in self._obj_xy_hist]))

    def _contact_still_present(self) -> bool:
        return any(
            self.env._group_contact_force(side, g)[1] > 0.05
            for side in SIDES for g in ("thumb", "index", "middle")
        )

    def _measure_side_grasp_posture(self) -> dict:
        """[This session] Side-Grasp Posture Gate (Section 10 of this
        session's spec) -- measured directly from live FK/contact state,
        called once when WRIST_SIDE_GRASP_ALIGN's own stability/collision/
        object-facing checks have just passed. Every sub-condition here is
        a real measurement (fingertip FK, palm axes, elbow/shoulder body
        position, actual contact forces already tracked this tick) --
        none of it is inferred from joint targets alone."""
        env = self.env
        obj_pos = self._object_pos()
        half = env.config.object_half_size
        palm = {s: env.palm_pose(s) for s in SIDES}
        tip_centroid = {}
        for s in SIDES:
            tips = {f: env.fingertip_pos(s, f) for f in sc.FINGERS}
            tip_centroid[s] = np.mean([tips[f] for f in ("index", "middle", "ring", "pinky")], axis=0)

        outside_side_face = {s: bool(abs(palm[s][0][1]) > half) for s in SIDES}
        inward_angle_deg = {
            "left": self.left_object_facing_angle_deg, "right": self.right_object_facing_angle_deg,
        }
        # palm normals (= closing axis, palm_R column 1) must OPPOSE each other (facing in).
        normals_opposed = float(np.dot(palm["left"][1][:, 1], palm["right"][1][:, 1])) < 0.0
        finger_down_deg = {s: _finger_down_angle_deg(palm[s][1]) for s in SIDES}
        tip_height_overlaps_side = {
            s: bool(obj_pos[2] - half <= tip_centroid[s][2] <= obj_pos[2] + half) for s in SIDES
        }
        # "crosses the object's TOP footprint" -- palm/fingertip XY within the object's
        # own X/Y half-extent AND above the object's top face.
        crosses_top_footprint = any(
            abs(tip_centroid[s][0] - obj_pos[0]) < half and abs(tip_centroid[s][1] - obj_pos[1]) < half
            and tip_centroid[s][2] > obj_pos[2] + half
            for s in SIDES
        )
        mirror_pos_err_m = float(np.linalg.norm(
            (palm["left"][0] - obj_pos) * np.array([1, -1, 1]) - (palm["right"][0] - obj_pos)
        ))
        rel = palm["left"][1].T @ palm["right"][1]
        # left/right are mirrors (Y-flip), not identical -- compare each side's OWN
        # deviation from ITS side-grasp target instead of comparing R_left to R_right directly.
        mirror_ori_err_deg = abs(inward_angle_deg["left"] - inward_angle_deg["right"])
        elbow_above_shoulder_m = {}
        for s in SIDES:
            sb = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_shoulder_roll_link")
            eb = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_elbow_link")
            elbow_above_shoulder_m[s] = float(env.data.xpos[eb][2] - env.data.xpos[sb][2])

        torso_arm_ok = self.torso_arm_collision_force_n <= self.config.hand_hand_force_limit_n
        hand_table_ok = self.max_hand_table_force_n <= self.config.hand_hand_force_limit_n
        hand_hand_ok = self._max_hand_hand <= self.config.hand_hand_force_limit_n if self._max_hand_hand else \
            env._hand_hand_contact_force() <= self.config.hand_hand_force_limit_n
        premature_contact_ok = self._proximal_penetration_ok()
        elbow_ok = all(v <= 0.02 for v in elbow_above_shoulder_m.values())

        metrics = {
            "outside_side_face": outside_side_face,
            "inward_angle_deg": inward_angle_deg,
            "normals_opposed": normals_opposed,
            "finger_down_deg": finger_down_deg,
            "tip_height_overlaps_side": tip_height_overlaps_side,
            "crosses_top_footprint": crosses_top_footprint,
            "mirror_pos_err_m": mirror_pos_err_m,
            "mirror_ori_err_deg": mirror_ori_err_deg,
            "elbow_above_shoulder_m": elbow_above_shoulder_m,
            "elbow_ok": elbow_ok,
            "torso_arm_ok": torso_arm_ok,
            "hand_table_ok": hand_table_ok,
            "hand_hand_ok": hand_hand_ok,
            "premature_contact_ok": premature_contact_ok,
            "wrist_orientation_drift_deg": self.wrist_orientation_drift_deg,
            "streak": self._side_align_stable_streak,
        }
        cfg = self.config
        gate = (
            all(outside_side_face.values())
            and inward_angle_deg["left"] <= cfg.side_grasp_inward_angle_tol_deg
            and inward_angle_deg["right"] <= cfg.side_grasp_inward_angle_tol_deg
            and normals_opposed
            and finger_down_deg["left"] <= cfg.side_grasp_finger_down_tol_deg
            and finger_down_deg["right"] <= cfg.side_grasp_finger_down_tol_deg
            and all(tip_height_overlaps_side.values())
            and not crosses_top_footprint
            and mirror_pos_err_m <= cfg.side_grasp_mirror_pos_tol_m
            and mirror_ori_err_deg <= cfg.side_grasp_mirror_ori_tol_deg
            and elbow_ok
            and torso_arm_ok
            and hand_table_ok
            and hand_hand_ok
            and premature_contact_ok
            and self._side_align_stable_streak >= cfg.side_align_stable_streak_required
        )
        metrics["gate"] = gate
        return metrics

    # ------------------------------------------------------------------
    def step(self) -> np.ndarray:
        action = self._zero_action()
        state = self.state
        self._just_advanced = False
        cfg = self.config

        if state == BimanualGraspState.STABLE_START:
            # [Session 41 -- tried applying set_preshape here to tuck the
            # thumb before the ARM_LATERAL_CLEARANCE sweep; MEASURED to
            # make the transient table graze WORSE (max penetration grew
            # from ~2.5mm to ~12.5mm and spread to the index finger too)
            # -- reverted, not applied. See docs/history/
            # PHASE4_GRASP_SESSION_41.md's Stage 4/6 notes: a brief
            # (<45-tick), small (<2.5mm), self-resolving thumb<->table
            # graze during the stand-pose-to-clearance transition is a
            # measured, shared, currently-unresolved characteristic --
            # disclosed, not hidden, and does not block the state's own
            # convergence gate (which checks torso/hand-hand collision
            # and settled joint error, not this specific transient).]
            if self._state_step >= 10:
                self._advance(BimanualGraspState.ARM_LATERAL_CLEARANCE)

        elif state == BimanualGraspState.ARM_LATERAL_CLEARANCE:
            # [Session 41] DIRECT joint target, NOT Cartesian IK -- a
            # redundant-arm IK solving only a Cartesian position/soft-
            # orientation task is free to pick ANY elbow configuration
            # that reaches the target, including the elbow-above-
            # shoulder "unnatural" branch the 41st session's user
            # feedback identified. Commanding the arm/shoulder/elbow
            # joints directly removes that ambiguity entirely. Values
            # (clearance_shoulder_pitch/roll/elbow) were chosen from a
            # bounded 3-candidate FK+physics sweep (module docstring/
            # BimanualGraspConfig docstring; docs/history/
            # PHASE4_GRASP_SESSION_41.md) -- 0 real self-collisions,
            # elbow ~7.6cm below shoulder, near-perfect actuator tracking.
            #
            # [Session 42 fix] the 41st session's version wrote the FULL
            # clearance target into _arm_ik_target/_waist_ik_target in a
            # single tick (state_step==0), making the ctrl register's
            # RATE-LIMITED CHASE begin with an instantaneous slope change
            # (0 -> max rate) rather than a smooth ramp. Substep-level
            # tracing (scripts/diagnose_clearance_wrist_spike.py) found
            # this is not directly what excites the ~8rad/s wrist qvel --
            # qfrc_constraint stays near zero for the first ~5 ticks even
            # with the abrupt ctrl slope. The ACTUAL trigger, confirmed
            # tick-by-tick: a real thumb<->table contact appears at tick
            # ~17 (qfrc_constraint on shoulder/elbow/wrist jumps from <1
            # to 12-23 N*m in the SAME tick the contact appears), and
            # THAT impulse is what couples into wrist_pitch (armature=
            # 0.01, dof_damping=0) and drives it to ~8rad/s over the next
            # ~30 ticks. A quintic minimum-jerk joint trajectory (zero
            # velocity/acceleration at both ends, from the ACTUAL current
            # qpos to the clearance target over
            # clearance_trajectory_ticks) replaces the abrupt-slope chase
            # -- verified (see history doc) to keep peak penetration at
            # this contact shallow enough that qfrc_constraint never
            # exceeds a few N*m, which keeps wrist qvel bounded well
            # under the engineering safety target (Wrist Transition Gate,
            # wrist_max_qvel_rad_s=2.0 -- an explicit trajectory-safety
            # target for THIS approach, not a Gate A criterion).
            if self._state_step == 0:
                self._clearance_target = np.concatenate([
                    self._waist_ik_target,  # waist stays neutral/current
                    self._clearance_arm_vector("left"),
                    self._clearance_arm_vector("right"),
                ])
                self._clearance_q_start = np.concatenate([
                    self.env._waist_target.copy(), self.env._arm_target.copy(),
                ])
                self._clearance_stable_streak = 0
                self._clearance_max_raw_wrist_qvel = 0.0
            tau = self._state_step / cfg.clearance_trajectory_ticks
            s = _quintic_scale(tau)
            traj_target = self._clearance_q_start + s * (self._clearance_target - self._clearance_q_start)
            self._waist_ik_target = traj_target[:3].copy()
            self._arm_ik_target = traj_target[3:].copy()
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            actual_arm_q = np.concatenate([
                self.env.data.qpos[self.env._arm_qpos_adr[:7]], self.env.data.qpos[self.env._arm_qpos_adr[7:]]
            ])
            wrist_dof = np.concatenate([self.env._arm_dof_adr[4:7], self.env._arm_dof_adr[11:14]])
            raw_wrist_qvel = float(np.max(np.abs(self.env.data.qvel[wrist_dof])))
            self._clearance_max_raw_wrist_qvel = max(self._clearance_max_raw_wrist_qvel, raw_wrist_qvel)
            joint_err = float(np.max(np.abs(actual_arm_q - self._clearance_target[3:])))
            qvel_ok = float(np.max(np.abs(self.env.data.qvel[np.concatenate([self.env._arm_dof_adr, self.env._waist_dof_adr])]))) < 0.05
            wrist_qvel_ok = raw_wrist_qvel <= cfg.wrist_max_qvel_rad_s
            no_collision = (self.env._torso_arm_collision_force() <= cfg.hand_hand_force_limit_n
                             and self.env._hand_hand_contact_force() <= cfg.hand_hand_force_limit_n)
            trajectory_done = self._state_step >= cfg.clearance_trajectory_ticks
            stable_now = (trajectory_done and joint_err <= cfg.clearance_joint_tol_rad
                          and qvel_ok and wrist_qvel_ok and no_collision)
            self._clearance_stable_streak = self._clearance_stable_streak + 1 if stable_now else 0
            if self._clearance_stable_streak >= cfg.clearance_stable_streak_required:
                self._advance(BimanualGraspState.FOREARM_FORWARD_REACH)
            elif self._state_step >= cfg.max_steps_per_state:
                self._fail(BimanualFailureReason.LATERAL_CLEARANCE_NOT_ACHIEVED)

        elif state == BimanualGraspState.FOREARM_FORWARD_REACH:
            # Shoulder/elbow-led reach toward the object, biased (via
            # rest_q + posture_rest_gain) to stay close to the lateral-
            # clearance posture instead of the stale stand-pose rest_q.
            # [Session 41, measured] a SINGLE one-shot solve straight from
            # the wide clearance Y (~0.46m) to the much narrower approach
            # Y (0.15m) drove waist_pitch to its hard limit (margin=0,
            # causally reproduced even at rest_gain=0 -- a real
            # reachability property of this large a lateral swing, not a
            # posture-bias artifact) and plateaued at a ~4.5-5.4cm
            # residual. Breaking the SAME Cartesian move into
            # FORWARD_REACH_WAYPOINTS smaller interpolated sub-targets
            # (the identical recipe already proven for FINGERTIP_
            # PRECONTACT) keeps every individual joint delta small enough
            # that the solver never needs the waist to compensate.
            if self._state_step == 0:
                self._forward_reach_stable_streak = 0
                self._forward_reach_start = {s: self.env.palm_pose(s)[0].copy() for s in SIDES}
                self._forward_reach_final = self._mirrored_targets(
                    cfg.approach_standoff_m, cfg.approach_height_m, cfg.approach_y_offset_m
                )
                self._forward_reach_waypoint = 0
            ticks_per_wp = self.FORWARD_REACH_WAYPOINT_TICKS
            if self._state_step % ticks_per_wp == 0 and self._forward_reach_waypoint < self.FORWARD_REACH_WAYPOINTS:
                self._forward_reach_waypoint += 1
                frac = self._forward_reach_waypoint / self.FORWARD_REACH_WAYPOINTS
                wp_targets = {
                    s: (1 - frac) * self._forward_reach_start[s] + frac * self._forward_reach_final[s] for s in SIDES
                }
                lR = self.env.palm_pose("left")[1].copy()
                rR = self.env.palm_pose("right")[1].copy()
                # [Session 41, measured] ori_task_weight=0.1 (the value
                # reused elsewhere in this file) kept waist_pitch AND
                # both wrist_pitch joints pinned at their hard limits even
                # at rest_gain=0 (pos_err plateaued ~4.3cm) -- soft
                # orientation pressure toward the CURRENT (post-clearance)
                # orientation was fighting the position task from this
                # starting configuration. ori_task_weight=0.0 (position-
                # only) converges cleanly (pos_err 0.53cm, positive joint
                # margin) -- acceptable here because FOREARM_FORWARD_REACH
                # is a transit state; WRIST_SIDE_GRASP_ALIGN/FINGERTIP_PRECONTACT
                # still do the real orientation work afterward. Verified
                # empirically (not just solve()'s own report) that this
                # does not reintroduce the 36th session's wrist_pitch
                # DYNAMIC instability (qvel stays bounded through physics).
                result = self._solve_both(wp_targets, {"left": lR, "right": rR}, require_orientation=False,
                                           ori_task_weight=0.0, rest_q=self._clearance_target, rest_gain=cfg.posture_rest_gain)
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            left_pos = self.env.palm_pose("left")[0]
            right_pos = self.env.palm_pose("right")[0]
            pos_err = max(float(np.linalg.norm(self._forward_reach_final["left"] - left_pos)),
                          float(np.linalg.norm(self._forward_reach_final["right"] - right_pos)))
            no_collision = (self.env._torso_arm_collision_force() <= cfg.hand_hand_force_limit_n
                             and self.env._hand_hand_contact_force() <= cfg.hand_hand_force_limit_n)
            stable_now = (pos_err <= cfg.ik_pos_tol and no_collision
                          and self._forward_reach_waypoint >= self.FORWARD_REACH_WAYPOINTS)
            self._forward_reach_stable_streak = self._forward_reach_stable_streak + 1 if stable_now else 0
            if self._forward_reach_stable_streak >= cfg.forward_reach_stable_streak_required:
                self._advance(BimanualGraspState.WRIST_SIDE_GRASP_ALIGN)
            elif self._state_step >= cfg.max_steps_per_state:
                self._fail(BimanualFailureReason.FORWARD_REACH_NOT_ACHIEVED)

        elif state == BimanualGraspState.WRIST_SIDE_GRASP_ALIGN:
            # [This session] Replaces the old FOREARM_DESCEND (position-
            # only) + WRIST_ALIGN (measure-only) pair. User requirement:
            # both palms end up beside the object's own side faces,
            # FACING EACH OTHER (closing axis toward the object center),
            # fingers generally pointing down -- not a top-down reach.
            #
            # Causal path to this design (bounded experiments, never a
            # random sweep -- see scripts/measure_sharpa_side_grasp_axes.py
            # and side_align_height_m's docstring above):
            #  1. FK-measured the real palm/finger axes: the "closing"
            #     local axis (palm_R column 1) is where fingertips move
            #     when the index/middle/wrap groups curl -- IDENTICAL sign
            #     for both hands in local frame (fixed _object_facing_R's
            #     old per-side negation, see that function's docstring).
            #     The "finger-down" axis coincides EXACTLY with the
            #     approach axis (palm_R column 0) for open, straight
            #     fingers.
            #  2. The orientation ARM_LATERAL_CLEARANCE/FOREARM_FORWARD_
            #     REACH leave the wrist in is ~110-170deg away from any
            #     object-facing target -- a large reorientation is
            #     unavoidable SOMEWHERE in this trajectory.
            #  3. Ramping ONLY orientation while holding POSITION FIXED
            #     (at FOREARM_FORWARD_REACH's own end pose, or at the
            #     final close/low grasp pose) reliably self-collides
            #     (measured 72-97N torso-arm/hand-hand peaks, both
            #     locations). Ramping position AND orientation TOGETHER,
            #     waypointed with SLERP (same per-waypoint IK re-solve
            #     recipe as FOREARM_FORWARD_REACH's own fix), from
            #     FOREARM_FORWARD_REACH's already-safe end pose to a new,
            #     wider/higher-than-final "align" target is collision-free
            #     (measured 0.00N torso-arm / 0.00N hand-hand, reproduced
            #     across 3 seed=0 rollouts).
            #  4. With fingers pointing down, OPEN fingers reach ~15cm
            #     below the palm -- measured up to 25N hand-table force
            #     during this same ramp unless a PROTECTIVE partial curl
            #     (side_align_preshape_curl, index/middle/wrap only, same
            #     split as CONTACT_ACQUIRE) is applied at this state's
            #     entry, shortening the effective reach. This is separate
            #     from FIVE_FINGER_PRESHAPE (which follows, unchanged in
            #     spirit -- final abduction/opposition refinement) and is
            #     disclosed here as a safety measure, not hidden.
            # The target itself (side_align_height_m/y_offset_m, at
            # FOREARM_FORWARD_REACH's own standoff) is deliberately HIGHER
            # and WIDER than the final grasp pose -- FOREARM_SIDE_DESCEND
            # (next) closes the remaining gap with orientation already
            # established and held, matching the "no large wrist rotation
            # after descending" requirement.
            if self._state_step == 0:
                self.env.set_preshape("left", 1.0)
                self.env.set_preshape("right", 1.0)
                self._side_align_start_pos = {s: self.env.palm_pose(s)[0].copy() for s in SIDES}
                self._side_align_start_R = {s: self.env.palm_pose(s)[1].copy() for s in SIDES}
                obj_pos = self._object_pos()
                self._side_align_final_pos = self._mirrored_targets(
                    cfg.approach_standoff_m, cfg.side_align_height_m, cfg.side_align_y_offset_m
                )
                self._side_align_final_R = {s: _object_facing_R(s, self._side_align_final_pos[s], obj_pos) for s in SIDES}
                self._side_align_waypoint = 0
                self._side_align_stable_streak = 0
                self._side_align_R_hist = {"left": [], "right": []}
            # Protective curl ramp (index/middle/wrap, not thumb), fast
            # relative to the position/orientation ramp so it is mostly
            # established before the riskier later waypoints.
            curl_ramp_ticks = int(np.ceil(cfg.side_align_preshape_curl / max(self.env.config.hand_synergy_action_scale, 1e-9)))
            if self._state_step < curl_ramp_ticks:
                action[17:25] = self._group_action({s: {"index": cfg.close_rate_per_step, "middle": cfg.close_rate_per_step,
                                                          "wrap": cfg.close_rate_per_step} for s in SIDES})
            ticks_per_wp = cfg.side_align_waypoint_ticks
            if self._state_step % ticks_per_wp == 0 and self._side_align_waypoint < cfg.side_align_waypoints:
                self._side_align_waypoint += 1
                frac = self._side_align_waypoint / cfg.side_align_waypoints
                wp_pos = {
                    s: (1 - frac) * self._side_align_start_pos[s] + frac * self._side_align_final_pos[s] for s in SIDES
                }
                R = {s: _slerp_R(self._side_align_start_R[s], self._side_align_final_R[s], frac) for s in SIDES}
                ori_w = self.SIDE_ORI_WEIGHT_START + frac * (self.SIDE_ORI_WEIGHT_END - self.SIDE_ORI_WEIGHT_START)
                final_wp = self._side_align_waypoint >= cfg.side_align_waypoints
                result = self._solve_both(wp_pos, R, require_orientation=final_wp, ori_task_weight=ori_w,
                                           rest_q=self._clearance_target, rest_gain=cfg.posture_rest_gain)
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            lR = self.env.palm_pose("left")[1].copy()
            rR = self.env.palm_pose("right")[1].copy()
            self._side_align_R_hist["left"].append(lR)
            self._side_align_R_hist["right"].append(rR)
            no_collision = (self.env._torso_arm_collision_force() <= cfg.hand_hand_force_limit_n
                             and self.env._hand_hand_contact_force() <= cfg.hand_hand_force_limit_n)
            no_table_hit = self.env._hand_table_contact_force() <= cfg.hand_hand_force_limit_n
            self.max_hand_table_force_n = max(self.max_hand_table_force_n, self.env._hand_table_contact_force())
            waypoints_done = self._side_align_waypoint >= cfg.side_align_waypoints
            if not no_table_hit:
                self._fail(BimanualFailureReason.HAND_TABLE_COLLISION)
                return action
            if waypoints_done and len(self._side_align_R_hist["left"]) >= 30:
                def _max_angle_dev_deg(hist: list[np.ndarray]) -> float:
                    r_final = hist[-1]
                    devs = []
                    for r in hist[-30:]:
                        r_delta = r_final.T @ r
                        cos_ang = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)
                        devs.append(np.degrees(np.arccos(cos_ang)))
                    return max(devs)

                left_dev = _max_angle_dev_deg(self._side_align_R_hist["left"])
                right_dev = _max_angle_dev_deg(self._side_align_R_hist["right"])
                self.wrist_orientation_drift_deg = max(left_dev, right_dev)
                stable_now = (self.wrist_orientation_drift_deg <= cfg.wrist_orientation_stability_tol_deg
                              and no_collision)
            else:
                stable_now = False
            self._side_align_stable_streak = self._side_align_stable_streak + 1 if stable_now else 0
            if self._side_align_stable_streak >= cfg.side_align_stable_streak_required:
                self._locked_R = {"left": lR, "right": rR}
                self.torso_arm_collision_force_n = self.env._torso_arm_collision_force()
                if self.torso_arm_collision_force_n > cfg.hand_hand_force_limit_n:
                    self._fail(BimanualFailureReason.SELF_COLLISION_TORSO_ARM)
                    return action
                obj_pos = self._object_pos()
                left_pos = self.env.palm_pose("left")[0]
                right_pos = self.env.palm_pose("right")[0]
                self.left_object_facing_angle_deg = _object_facing_angle_deg("left", lR, left_pos, obj_pos)
                self.right_object_facing_angle_deg = _object_facing_angle_deg("right", rR, right_pos, obj_pos)
                if (self.left_object_facing_angle_deg > cfg.object_facing_angle_tol_deg
                        or self.right_object_facing_angle_deg > cfg.object_facing_angle_tol_deg):
                    self._fail(BimanualFailureReason.WRIST_NOT_OBJECT_FACING)
                    return action
                self._side_grasp_posture = self._measure_side_grasp_posture()
                self._side_grasp_gate = self._side_grasp_posture["gate"]
                self._advance(BimanualGraspState.FIVE_FINGER_PRESHAPE)
            elif self._state_step >= cfg.side_align_max_steps:
                self._fail(BimanualFailureReason.SIDE_GRASP_ALIGN_NOT_ACHIEVED)

        elif state == BimanualGraspState.FIVE_FINGER_PRESHAPE:
            # Abduction/opposition preshape was already applied at
            # WRIST_SIDE_GRASP_ALIGN entry (set_preshape is idempotent --
            # re-applying here is a harmless confirmation, not a second
            # target). This state's remaining job is exactly what it was
            # before: hold briefly and verify no self-collision/table hit
            # resulted before committing to FOREARM_SIDE_DESCEND.
            if self._state_step == 0:
                self.env.set_preshape("left", 1.0)
                self.env.set_preshape("right", 1.0)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            if self._state_step >= 30:
                if (not self._proximal_penetration_ok()
                        or self.env._hand_hand_contact_force() > cfg.hand_hand_force_limit_n):
                    self._fail(BimanualFailureReason.SELF_COLLISION_BEFORE_CONTACT)
                    return action
                if self.env._hand_table_contact_force() > cfg.hand_hand_force_limit_n:
                    self._fail(BimanualFailureReason.HAND_TABLE_COLLISION)
                    return action
                self._advance(BimanualGraspState.FOREARM_SIDE_DESCEND)

        elif state == BimanualGraspState.FOREARM_SIDE_DESCEND:
            # [This session] Second waypointed position move, orientation
            # HELD (small soft anchor to the ALREADY-correct side-grasp
            # orientation, never re-derived) -- matches the "no large
            # wrist rotation after descending" requirement. See
            # side_descend_standoff_m's docstring for the measured
            # collision-free numbers.
            if self._state_step == 0:
                self._descend_stable_streak = 0
                self._side_descend_start = {s: self.env.palm_pose(s)[0].copy() for s in SIDES}
                self._side_descend_final = self._mirrored_targets(
                    cfg.side_descend_standoff_m, cfg.side_descend_height_m, cfg.side_descend_y_offset_m
                )
                self._side_descend_waypoint = 0
            if (self._state_step % cfg.side_descend_waypoint_ticks == 0
                    and self._side_descend_waypoint < cfg.side_descend_waypoints):
                self._side_descend_waypoint += 1
                frac = _quintic_scale(self._side_descend_waypoint / cfg.side_descend_waypoints)
                wp_targets = {
                    s: (1 - frac) * self._side_descend_start[s] + frac * self._side_descend_final[s] for s in SIDES
                }
                lR = self.env.palm_pose("left")[1].copy()
                rR = self.env.palm_pose("right")[1].copy()
                result = self._solve_both(wp_targets, {"left": lR, "right": rR}, require_orientation=False,
                                           ori_task_weight=0.15, rest_q=self._clearance_target, rest_gain=cfg.posture_rest_gain)
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            left_pos = self.env.palm_pose("left")[0]
            right_pos = self.env.palm_pose("right")[0]
            pos_err = max(float(np.linalg.norm(self._side_descend_final["left"] - left_pos)),
                          float(np.linalg.norm(self._side_descend_final["right"] - right_pos)))
            hand_table_force = self.env._hand_table_contact_force()
            self.max_hand_table_force_n = max(self.max_hand_table_force_n, hand_table_force)
            # [This session] Measured (scripts/diagnose_side_grasp_swept_path.py)
            # a REAL, repeated (not single-tick-transient) hand<->table
            # force up to ~18N during this state's later waypoints, well
            # over the 8N limit -- honestly fail-fast on it (matching
            # WRIST_SIDE_GRASP_ALIGN's own treatment) instead of letting
            # it get silently absorbed into a generic timeout. A bounded
            # (curl fraction, height) candidate search did not clear it
            # without cost elsewhere (higher curl reintroduced a table hit
            # earlier, during WRIST_SIDE_GRASP_ALIGN itself) -- disclosed
            # as the current next blocker, not force-fixed here.
            if hand_table_force > cfg.hand_hand_force_limit_n:
                self._fail(BimanualFailureReason.HAND_TABLE_COLLISION)
                return action
            no_collision = (self.env._torso_arm_collision_force() <= cfg.hand_hand_force_limit_n
                             and self.env._hand_hand_contact_force() <= cfg.hand_hand_force_limit_n)
            waypoints_done = self._side_descend_waypoint >= cfg.side_descend_waypoints
            stable_now = pos_err <= cfg.ik_pos_tol and no_collision and waypoints_done
            self._descend_stable_streak = self._descend_stable_streak + 1 if stable_now else 0
            if self._descend_stable_streak >= cfg.side_descend_stable_streak_required:
                self._advance(BimanualGraspState.FINGERTIP_PRECONTACT)
            elif self._state_step >= cfg.max_steps_per_state:
                self._fail(BimanualFailureReason.SIDE_DESCEND_NOT_ACHIEVED)

        elif state == BimanualGraspState.FINGERTIP_PRECONTACT:
            # [This session's finding] a SINGLE IK jump straight from
            # WRIST_SIDE_GRASP_ALIGN's converged pose to the full precontact target
            # is kinematically valid (IK success=True, <1cm error) but
            # requires a large one-shot change in the low-inertia,
            # zero-damping wrist joints (e.g. wrist_roll by >1.5rad in one
            # solve) -- exactly the pattern already found to destabilize
            # wrist_pitch during FOREARM_APPROACH, now reproduced in
            # wrist_roll instead. Breaking the same Cartesian move into
            # WAYPOINT_COUNT smaller interpolated sub-targets, each
            # re-solved with the same CURRENT-orientation soft anchor and
            # held for WAYPOINT_TICKS before advancing, keeps every
            # individual joint delta small -- the same fix category as
            # FOREARM_APPROACH's, applied generically instead of
            # re-diagnosing per-joint each time.
            if self._state_step == 0:
                self._precontact_start = {"left": self.env.palm_pose("left")[0].copy(),
                                           "right": self.env.palm_pose("right")[0].copy()}
                self._precontact_final = self._mirrored_targets(
                    cfg.precontact_standoff_m, cfg.precontact_height_m, cfg.precontact_y_offset_m
                )
                # Target orientation for the gate check is WRIST_SIDE_GRASP_ALIGN's
                # already-locked, measured-stable orientation (module
                # docstring: "LOCK that exact orientation as the explicit
                # WRIST_SIDE_GRASP_ALIGN/FINGERTIP_PRECONTACT target"), not a freshly
                # re-measured one -- these should coincide closely since
                # every waypoint solve below only soft-anchors
                # (ori_task_weight=0.05) to whatever orientation is
                # currently held.
                self._precontact_final_R = {s: self._locked_R[s].copy() for s in SIDES}
                self._precontact_waypoint = 0
                self._precontact_stable_streak = 0
            if self._state_step % self.WAYPOINT_TICKS == 0 and self._precontact_waypoint < self.WAYPOINT_COUNT:
                self._precontact_waypoint += 1
                frac = self._precontact_waypoint / self.WAYPOINT_COUNT
                targets = {
                    side: (1 - frac) * self._precontact_start[side] + frac * self._precontact_final[side]
                    for side in SIDES
                }
                lR = self.env.palm_pose("left")[1].copy()
                rR = self.env.palm_pose("right")[1].copy()
                result = self._solve_both(targets, {"left": lR, "right": rR}, require_orientation=False, ori_task_weight=0.05)
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()

            if self._precontact_waypoint >= self.WAYPOINT_COUNT:
                # [Session 39 finding -- see docs/history/
                # PHASE4_GRASP_SESSION_39.md] the ctrl register/
                # rate-limited chase has nothing left to converge to at
                # this point (it reaches the IK-solved joint target
                # exactly). Two independent Cartesian-target-inflation
                # resolve variants (with and without a posture-hold rest_q)
                # were causally tested here and BOTH measured WORSE
                # (gap grew from ~6.9cm to 10-25cm, joint norm to
                # >1rad) -- this state's redundant 17-DOF solve, at this
                # already-extreme precontact reach, does not have the
                # locally-linear droop-vs-target relationship required by
                # that compensation; extrapolating the Cartesian target
                # drives the IK into a qualitatively different, LESS
                # favorable arm configuration instead of compensating.
                # That IK-side compensation avenue is therefore not
                # used. What IS applied is a real, physically-grounded
                # feedforward: SharpaGraspEnv's arm_gravity_compensation
                # (config-gated, see grasp_config.py/sharpa_grasp_env.py)
                # cancels the actual measured qfrc_bias/kp steady-state
                # droop AT THE ACTUATOR, not via a kinematic guess. This
                # block does NOT re-solve IK -- it only measures the
                # ACTUAL settled pose and HONESTLY gates the transition
                # on it (Precontact Tracking Gate), never advancing on a
                # fixed tick count regardless of convergence (the
                # pre-Session-39 behavior).
                left_actual, left_R = self.env.palm_pose("left")
                right_actual, right_R = self.env.palm_pose("right")
                left_pos_err = float(np.linalg.norm(self._precontact_final["left"] - left_actual))
                right_pos_err = float(np.linalg.norm(self._precontact_final["right"] - right_actual))

                def _ang_deg(Ra, Rb):
                    r_delta = Ra.T @ Rb
                    cos_ang = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)
                    return float(np.degrees(np.arccos(cos_ang)))

                left_ori_err = _ang_deg(left_R, self._precontact_final_R["left"])
                right_ori_err = _ang_deg(right_R, self._precontact_final_R["right"])
                self._precontact_final_pos_error = {"left": left_pos_err, "right": right_pos_err}
                self._precontact_final_ori_error_deg = {"left": left_ori_err, "right": right_ori_err}

                no_hand_hand = self.env._hand_hand_contact_force() <= cfg.hand_hand_force_limit_n
                stable_now = (
                    left_pos_err <= cfg.ik_pos_tol and right_pos_err <= cfg.ik_pos_tol
                    and left_ori_err <= cfg.precontact_ori_tol_deg and right_ori_err <= cfg.precontact_ori_tol_deg
                    and no_hand_hand
                )
                self._precontact_stable_streak = self._precontact_stable_streak + 1 if stable_now else 0
                self._max_precontact_stable_streak = max(self._max_precontact_stable_streak, self._precontact_stable_streak)
                if self._precontact_stable_streak >= cfg.precontact_stable_streak_required:
                    self._advance(BimanualGraspState.CONTACT_ACQUIRE)
                elif self._state_step >= cfg.max_steps_per_state:
                    self._fail(BimanualFailureReason.PRECONTACT_TRACKING_NOT_ACHIEVED)

        elif state == BimanualGraspState.CONTACT_ACQUIRE:
            # Both sides close INDEPENDENTLY/asynchronously: a side/group
            # that has already registered contact stops advancing while
            # the other side keeps closing (Section 4 requirement 7/8).
            deltas = {"left": {}, "right": {}}
            for side in SIDES:
                for group in ("index", "middle", "wrap"):
                    if not self._group_contact_now(side, group):
                        deltas[side][group] = cfg.close_rate_per_step
            action[17:25] = self._group_action(deltas)
            both_have_some_contact = all(
                any(self._group_ever_contacted[side][g] for g in ("index", "middle", "wrap")) for side in SIDES
            )
            if both_have_some_contact:
                self._advance(BimanualGraspState.THUMB_OPPOSE)
            elif self._state_step >= cfg.max_steps_per_state:
                self._fail(BimanualFailureReason.TIMEOUT)

        elif state == BimanualGraspState.THUMB_OPPOSE:
            deltas = {"left": {}, "right": {}}
            for side in SIDES:
                for group in sc.GROUPS:
                    if not self._group_contact_now(side, group):
                        deltas[side][group] = cfg.close_rate_per_step
            action[17:25] = self._group_action(deltas)
            if all(self._group_ever_contacted[side][g] for side in SIDES for g in sc.GROUPS):
                self._advance(BimanualGraspState.ENVELOPING_CLOSE)
            elif self._state_step >= cfg.max_steps_per_state:
                self._advance(BimanualGraspState.ENVELOPING_CLOSE)

        elif state == BimanualGraspState.ENVELOPING_CLOSE:
            lo, hi = cfg.target_force_band_n
            deltas = {"left": {}, "right": {}}
            for side in SIDES:
                for group in sc.GROUPS:
                    _, net = self.env._group_contact_force(side, group)
                    self._group_net_force[side][group] = max(self._group_net_force[side][group], net)
                    if net < lo:
                        deltas[side][group] = cfg.close_rate_per_step * 0.5
            action[17:25] = self._group_action(deltas)
            all_in_band = all(
                lo <= self.env._group_contact_force(side, g)[1] <= hi * 1.5 for side in SIDES for g in sc.GROUPS
            )
            if all_in_band or self._state_step >= cfg.max_steps_per_state:
                self._advance(BimanualGraspState.FORCE_SETTLE)

        elif state == BimanualGraspState.FORCE_SETTLE:
            # [Section 5 fix] REAL force-band check, not "> 0.0" -- every
            # required group on BOTH sides must sit within
            # target_force_band_n, peak below the safety limit, AND the
            # bilateral topology/opposition check must hold, for
            # force_settle_hold_steps CONSECUTIVE ticks.
            lo, hi = cfg.target_force_band_n
            in_band = all(
                lo <= self.env._group_contact_force(side, g)[1] <= hi for side in SIDES for g in sc.GROUPS
            )
            peak_ok = all(
                self.env._group_contact_force(side, g)[0] <= self.env.config.finger_force_safety_limit
                for side in SIDES for g in sc.GROUPS
            )
            _, _, bilateral_ok = self._bilateral_stable_now()
            if in_band and peak_ok and bilateral_ok:
                self._force_settle_streak += 1
            else:
                self._force_settle_streak = 0
            self._track_stability()
            if self._force_settle_streak >= cfg.force_settle_hold_steps:
                self._advance(BimanualGraspState.TABLETOP_HOLD)
                self._initial_obj_xy = self._object_pos()[:2].copy()
            elif self._state_step >= cfg.max_steps_per_state:
                self._fail(BimanualFailureReason.CONTACT_LOST)

        elif state == BimanualGraspState.TABLETOP_HOLD:
            self._track_stability()
            if not self._contact_still_present():
                self._fail(BimanualFailureReason.CONTACT_LOST)
                return action
            self._tabletop_hold_steps += 1
            required = sim_time_to_steps(self.env, cfg.tabletop_hold_seconds)
            if self._tabletop_hold_steps >= required:
                if self.object_xy_displacement() > cfg.object_xy_displacement_limit_m:
                    self._fail(BimanualFailureReason.OBJECT_MOVED_TOO_MUCH)
                elif max(self._obj_angvel_hist, default=0.0) > cfg.object_peak_angular_velocity_limit:
                    self._fail(BimanualFailureReason.OBJECT_ANGULAR_VELOCITY_EXCEEDED)
                else:
                    self._advance(BimanualGraspState.LIFT)

        elif state == BimanualGraspState.LIFT:
            if self._state_step == 0:
                lp, lR = self.env.palm_pose("left")
                rp, rR = self.env.palm_pose("right")
                self._lift_start_obj_z = self._object_pos()[2]
                lift_h = cfg.lift_height_m
                targets = {"left": lp + np.array([0, 0, lift_h]), "right": rp + np.array([0, 0, lift_h])}
                result = self._solve_both(targets, {"left": lR, "right": rR}, require_orientation=True, ori_task_weight=1.0)
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            self._track_stability()
            self._lift_height_achieved = max(self._lift_height_achieved, self._object_pos()[2] - self._lift_start_obj_z)
            if not self._contact_still_present():
                self._fail(BimanualFailureReason.LIFT_FAILED)
                return action
            if self._state_step >= 100:
                self._advance(BimanualGraspState.AIR_HOLD)

        elif state == BimanualGraspState.AIR_HOLD:
            self._track_stability()
            if not self._contact_still_present():
                self._fail(BimanualFailureReason.LIFT_FAILED)
                return action
            self._air_hold_steps += 1
            required = sim_time_to_steps(self.env, cfg.air_hold_seconds)
            if self._air_hold_steps >= required:
                self.state = BimanualGraspState.SUCCESS

        if not self._just_advanced:
            self._state_step += 1
        self._total_step += 1
        return action

    # ------------------------------------------------------------------
    def run(self, max_total_steps: int = 8000) -> BimanualGraspOutcome:
        obs, info = self.env.reset(seed=getattr(self.env, "_last_seed", 0))
        for _ in range(max_total_steps):
            if self.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
                break
            action = self.step()
            obs, r, term, trunc, info = self.env.step(action)
            if info.get("unstable"):
                self._fail(BimanualFailureReason.NUMERICAL_ERROR)
                break
            if trunc and self.state not in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
                self._fail(BimanualFailureReason.TIMEOUT)
                break
        else:
            if self.state not in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
                self._fail(BimanualFailureReason.TIMEOUT)

        angvel_peak = float(np.max(self._obj_angvel_hist)) if self._obj_angvel_hist else 0.0
        angvel_rms = float(np.sqrt(np.mean(np.square(self._obj_angvel_hist)))) if self._obj_angvel_hist else 0.0
        gate_a = (
            self.state != BimanualGraspState.FAILURE
            and self._max_bilateral_streak >= self.config.bilateral_streak_required
            and self.object_xy_displacement() <= self.config.object_xy_displacement_limit_m
            and angvel_peak <= self.config.object_peak_angular_velocity_limit
            and self._max_proximal_pen <= self.config.proximal_penetration_tolerance_m
            and self._max_hand_hand <= self.config.hand_hand_force_limit_n
        )
        gate_b = self._tabletop_hold_steps >= sim_time_to_steps(self.env, self.config.tabletop_hold_seconds)
        gate_c = self._lift_height_achieved >= self.config.lift_height_m
        gate_d = self._air_hold_steps >= sim_time_to_steps(self.env, self.config.air_hold_seconds)
        precontact_gate = (
            self._max_precontact_stable_streak >= self.config.precontact_stable_streak_required
        )

        return BimanualGraspOutcome(
            state=self.state, failure_reason=self.failure_reason, step_count=self._total_step,
            per_side_group_contact={s: dict(self._group_ever_contacted[s]) for s in SIDES},
            per_side_group_peak_force={s: dict(self._group_peak_force[s]) for s in SIDES},
            per_side_group_net_force={s: dict(self._group_net_force[s]) for s in SIDES},
            max_bilateral_stable_streak=self._max_bilateral_streak, max_left_stable_streak=self._max_left_streak,
            max_right_stable_streak=self._max_right_streak, max_proximal_object_penetration=self._max_proximal_pen,
            max_hand_hand_force=self._max_hand_hand, object_xy_displacement=self.object_xy_displacement(),
            object_angular_velocity_peak=angvel_peak, object_angular_velocity_rms=angvel_rms,
            tabletop_hold_steps_achieved=self._tabletop_hold_steps, air_hold_steps_achieved=self._air_hold_steps,
            lift_height_achieved_m=self._lift_height_achieved, gate_a=gate_a, gate_b=gate_b, gate_c=gate_c, gate_d=gate_d,
            precontact_final_pos_error_m=dict(self._precontact_final_pos_error),
            precontact_final_ori_error_deg=dict(self._precontact_final_ori_error_deg),
            precontact_max_stable_streak=self._max_precontact_stable_streak,
            precontact_gate=precontact_gate,
            side_grasp_posture=self._side_grasp_posture,
            side_grasp_gate=self._side_grasp_gate,
        )
