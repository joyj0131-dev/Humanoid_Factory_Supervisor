"""Bimanual finger-based side-pinch grasp expert (Phase 4 Grasp Track).

State sequence (see PROJECT_CONTEXT.md Phase 4 Grasp Track, Forearm
Descent Before Wrist Alignment + Multi-Finger Contact session):
STABLE_START -> NATURAL_ARM_LIFT -> FOREARM_LATERAL_APPROACH ->
FOREARM_DESCEND -> WRIST_ALIGN -> FINGER_PRESHAPE -> FINGERTIP_PRECONTACT
-> CONTACT_ACQUIRE -> FINGER_CLOSE -> FORCE_SETTLE -> LIFT -> HOLD ->
SUCCESS / FAILURE.

FOREARM_LATERAL_APPROACH/FOREARM_DESCEND/WRIST_ALIGN/FINGERTIP_PRECONTACT
replace the previous session's FOREARM_APPROACH/WRIST_ALIGN/
FINGERTIP_APPROACH 3-state design. Root cause of that design's SIZE_6
convergence failure, found by DIRECT MEASUREMENT this session: the old
WRIST_ALIGN target sat at ``approach_height`` and the old FINGERTIP_
APPROACH target sat at ``grasp_z_offset`` -- a ~14.75cm (SIZE_6) / 18.5cm
(SIZE_12) VERTICAL DROP that FINGERTIP_APPROACH's heavily posture-held
shoulder/elbow (weight 3.0-12.0 tried) had to cover using mostly wrist,
which cannot: wrist rotation only translates the palm by the short
palm-offset lever arm (a few cm at most). SIZE_12 happened to still
converge; SIZE_6 plateaued at ~0.022m residual and timed out. The fix is
structural, not a tolerance/weight retune: the big vertical descent is
now its OWN state (FOREARM_DESCEND, shoulder/elbow PRIMARY, same as
FOREARM_LATERAL_APPROACH) that runs BEFORE wrist alignment, so by the
time WRIST_ALIGN/FINGERTIP_PRECONTACT run, the Cartesian position target
is ALREADY at the final grasp height and changes only by a few cm
(FINGERTIP_PRECONTACT's own small surface-normal approach) or not at all
(WRIST_ALIGN/FINGER_PRESHAPE hold position, work orientation/fingers only).

Every one of this controller's Cartesian targets is still computed as an
intended FINGERTIP contact point and converted to a palm/wrist IK target
via the fixed fingertip-centroid offset (``_measure_fingertip_grasp_
offset``/``_fingertip_target_to_palm_target``, Natural Arm Reach session)
-- every state from FOREARM_LATERAL_APPROACH onward aims the FINGERS at
the object, not the wrist. Palm/wrist is never placed AT the object
surface.

Each of the 7 early-approach states gives shoulder/elbow and wrist
DIFFERENT roles (Section 5, hierarchical IK weights via
``CoupledBilateralIK.group_vector``/``_coupled_role_profile``) instead of
solving all 17 coupled DoF with one flat cost the whole time, which
previously let shoulder and wrist "fight" and oscillate together:
NATURAL_ARM_LIFT/FOREARM_LATERAL_APPROACH/FOREARM_DESCEND make shoulder/
elbow cheap (primary reach DoF) and wrist expensive+pulled toward a
transport-safe orientation; WRIST_ALIGN flips that, making shoulder/elbow
expensive and pulled to a POSTURE-HOLD reference frozen at FOREARM_
DESCEND's exit (not the stand pose) while wrist becomes the cheap,
primary alignment DoF; FINGERTIP_PRECONTACT keeps wrist/fingers primary
but deliberately does NOT fully lock shoulder/elbow (a moderate, not
extreme, weight) so the small remaining position correction has
somewhere to come from besides wrist alone.

Root cause of the overhead/wide-swing motion found in the previous session
(measured baseline: 59 rad summed joint travel, 25 rad/s peak finite-
difference velocity, left shoulder pitch swinging 166 degrees): each state
set an absolute Cartesian target and let the IK converge to it over many
steps with NO rate limiting on the COMMANDED target itself -- only the
per-step action was clipped, so a state transition could jump the target
by tens of centimeters instantly, and an unnecessarily high/wide waypoint
geometry (approach_height=0.30m, outside_offset=0.15m) made that jump large
even when converged. Fixed here via:
  - ``_TargetTracker``: the target actually fed to the IK moves toward each
    state's goal at a bounded rate (``max_target_step`` m/step) -- state
    transitions change the GOAL, never the commanded target, so motion is
    continuous by construction.
  - Much smaller approach_height/outside_offset (just enough clearance for
    the object, not an overhead reach).
  - ``grip_center``/``grip_half_width`` replace independent left/right
    inward offsets for the contact-acquire/force-settle phase, driven by
    force feedback (Section 5).

Two semantics bugs fixed in the prior session are preserved unchanged:
absolute-desired-synergy -> delta-action conversion, and frozen (not
object-chasing) targets for pre-contact and lift/hold phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import mujoco
import numpy as np

from humanoid_learning.envs import hand_synergy
from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.expert import pose_ik
from humanoid_learning.expert.coupled_ik import CoupledBilateralIK, CoupledIKResult


class GraspState(Enum):
    STABLE_START = auto()
    NATURAL_ARM_LIFT = auto()
    FOREARM_LATERAL_APPROACH = auto()
    # THUMB_CLEARANCE_PRESHAPE moved HERE (Phase 4 -- Collision-Free Thumb
    # Preshape + Early Tripod Closure session), right after FOREARM_
    # LATERAL_APPROACH and BEFORE FOREARM_DESCEND -- direct measurement
    # found thumb's abduct transit clips SIZE_12's object if attempted
    # after the hand has already descended near the object (the previous
    # ordering, when this state was named THUMB_ABDUCT and sat right
    # before FINGERTIP_PRECONTACT); abducting while still at
    # approach_height, well clear of the object vertically, needs no
    # collision workaround at all for the cases tested so far.
    THUMB_CLEARANCE_PRESHAPE = auto()
    FOREARM_DESCEND = auto()
    WRIST_ALIGN = auto()
    FINGER_PRESHAPE = auto()
    FINGERTIP_PRECONTACT = auto()
    CONTACT_ACQUIRE = auto()
    THUMB_OPPOSE = auto()
    TRIPOD_SETTLE = auto()
    FINGER_CLOSE = auto()
    FORCE_SETTLE = auto()
    LIFT = auto()
    HOLD = auto()
    SUCCESS = auto()
    FAILURE = auto()


class HandSubstate(Enum):
    """Per-hand CONTACT_ACQUIRE substate (Section 4 redesign). The two
    hands do NOT touch on the same simulation step -- measured via
    left-only/right-only calibration: left first-contact at step 182,
    right at step 255, a ~70-step gap, at different y-offsets (0.147 vs
    0.127) and heights (0.833 vs 0.816). A single shared grip_half_width
    driving both targets meant the already-touching hand kept getting
    pushed further in while waiting for the other -- exactly the
    mechanism that built up an asymmetric squeeze before FINGER_CLOSE ever
    started. Each hand now tracks its own substate independently."""
    APPROACHING = auto()
    FIRST_CONTACT = auto()
    FORCE_HOLD = auto()
    CONTACT_LOST = auto()
    REACQUIRE = auto()
    READY = auto()


class FailureReason(Enum):
    PREMATURE_CONTACT = auto()
    OBJECT_DISPLACED = auto()
    IK_NOT_CONVERGED = auto()
    SINGLE_HAND_CONTACT_ONLY = auto()
    PALM_ONLY_CONTACT = auto()
    CONTACT_LOST = auto()
    LIFT_FAILED = auto()
    DROP = auto()
    TIMEOUT = auto()
    NUMERICAL_ERROR = auto()
    THUMB_CONTACT_FAILED = auto()


def make_side_pinch_orientation(closing_world: np.ndarray, approach_hint: np.ndarray = np.array([1.0, 0.0, 0.0])) -> np.ndarray:
    closing = closing_world / np.linalg.norm(closing_world)
    approach = approach_hint - np.dot(approach_hint, closing) * closing
    approach /= np.linalg.norm(approach)
    lateral = np.cross(approach, closing)
    return np.column_stack([approach, closing, lateral])


class _TargetTracker:
    """Moves a commanded 3D point toward a goal at a bounded rate -- the
    fix for the "target jumps on state transition" failure mode. Call
    ``set_goal`` when a state changes the destination (cheap, just changes
    where we're heading); call ``step`` every control tick to advance the
    commanded point and get the value to actually feed the IK."""

    def __init__(self, initial: np.ndarray, max_step: float):
        self.commanded = initial.copy()
        self.goal = initial.copy()
        self.max_step = max_step

    def set_goal(self, goal: np.ndarray) -> None:
        self.goal = goal.copy()

    def snap(self, point: np.ndarray) -> None:
        self.commanded = point.copy()
        self.goal = point.copy()

    def step(self) -> np.ndarray:
        delta = self.goal - self.commanded
        dist = float(np.linalg.norm(delta))
        if dist > self.max_step:
            self.commanded = self.commanded + delta * (self.max_step / dist)
        else:
            self.commanded = self.goal.copy()
        return self.commanded.copy()


class _MinJerkJointTrajectory:
    """Smooth joint-space setpoint interpolation for the coupled-IK
    solve's output (Coupled Bilateral Waist-Aware IK Validation session).
    Uses the classic 10t^3-15t^4+6t^5 minimum-jerk profile, which has zero
    velocity AND zero acceleration at both endpoints -- unlike
    ``_TargetTracker``'s constant-rate ramp, this cannot itself introduce
    a velocity discontinuity at the moment a coupled solve starts or
    finishes. Operates in the SAME space as ``_TargetTracker``'s target
    (accumulated actuator setpoint, not live qpos), so its output plugs
    into a joint-space action delta the same way the old Cartesian
    tracker's output plugged into ``_apply``."""

    def __init__(self, q_start: np.ndarray, q_target: np.ndarray, duration_steps: int):
        self.q_start = q_start.copy()
        self.q_target = q_target.copy()
        self.duration = max(1, duration_steps)
        self.elapsed = 0

    def step(self) -> np.ndarray:
        t = min(1.0, self.elapsed / self.duration)
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        self.elapsed += 1
        return self.q_start + s * (self.q_target - self.q_start)


@dataclass
class GraspExpertConfig:
    # approach_height / outside_offset / grasp_z_offset /
    # grip_half_width_contact / grip_half_width_min are NOT listed here
    # any more -- Section 3 of the multi-size grasp track explicitly
    # forbids hardcoding per-size geometry or branching on size equality.
    # BimanualSidePinchExpert.__init__ now DERIVES all five from
    # env.config.object_half_size (see _derive_size_conditioned_geometry),
    # using linear formulas reverse-engineered to reproduce this file's
    # OLD hardcoded 12cm-object values (half_size=0.06) EXACTLY at that
    # size, so the already-tuned 12cm behavior is preserved bit-for-bit
    # while 6cm/9cm get principled, non-hardcoded values instead of a
    # separate controller or an if/else branch on size.
    preshape_synergy: float = 0.15  # partial pre-curl before approach; kept
    # constant across sizes -- a per-joint curl AMOUNT, not a Cartesian
    # distance, and no calibration evidence (Section 5 sweep) yet shows it
    # needs to vary with object size within the 6-12cm range tested.

    max_target_step: float = 0.004  # meters per control step (rate limit, see _TargetTracker)
    min_state_steps: int = 8  # minimum ticks in a state before checking its exit condition

    lift_height: float = 0.06
    lift_step: float = 0.002
    finger_close_steps: int = 60
    settle_stable_steps: int = 20
    finger_contact_hold_steps: int = 30  # Section 6: finger contact must persist this long
    hold_steps: int = 100
    max_steps_per_state: int = 600
    object_displacement_limit: float = 0.03

    # Tuned empirically after the null-space fix (see pose_ik.step_toward
    # docstring): the original ik_gain=0.2/ik_damping=0.15 never converged
    # closer than ~0.13-0.23m position error even after 300 steps because
    # left_shoulder_roll_joint pinned at its hard limit (2.25 rad) and
    # other joints thrashed trying to compensate -- confirmed via a direct
    # diagnostic. Higher gains (>=0.35) reconverge fast but oscillate under
    # the compliant kp=120 arm dynamics (position feedback lag causes
    # overshoot) -- measured up to 0.1m peak-to-peak palm swing at a FROZEN
    # target. gain=0.3/damping=0.05/maxdq=0.12/null=0.4 was the best point
    # found: converges to ~0.039m position error with near-zero residual
    # oscillation (z_std < 0.001 over 100 static-target steps).
    ik_gain: float = 0.3
    ik_damping: float = 0.05
    ik_max_dq: float = 0.12  # rad/step joint-space rate limit
    # Lowered from 0.4 during the multi-size grasp track: measured directly
    # that null_gain=0.4 was fighting task convergence specifically in the
    # x-direction for the 6cm object's (smaller-offset) APPROACH target --
    # residual error plateaued at 0.15-0.24m (never reaching the object)
    # vs 0.08m at null_gain=0.0 (which grazes a joint limit by <1cm) and
    # 0.16m at null_gain=0.2 (safe joint margin, still not fully
    # converged but far closer than 0.4). 0.2 keeps the original
    # joint-limit-avoidance purpose (this fixed a real lockup during an
    # earlier session) while being noticeably less obstructive to the
    # primary task across both required object sizes.
    null_gain: float = 0.2  # elbow-down/rest-posture null-space pull, see pose_ik.py
    # ik_gain=0.3 was tuned against a DIFFERENT target during the earlier
    # joint-limit-lockup fix and turned out to be only marginally stable:
    # direct measurement against FINGER_PRESHAPE's actual target showed a
    # genuinely DIVERGING oscillation (0.10 -> 0.67 m/s and still rising
    # after 80 steps at a completely static target) -- the real cause of
    # the 9-25N first-contact impact forces (not target movement rate, see
    # _apply's ik_gain comment). A lower gain here converged smoothly and
    # monotonically for the same target with an equal-or-better residual
    # position error. Used from FINGER_PRESHAPE onward, where the arm is
    # already near the target and covering distance fast no longer matters.
    precision_ik_gain: float = 0.15
    # Position-error threshold (m) below which a holding-phase hand's IK
    # is not re-solved at all (dq forced to 0) rather than chasing sub-mm
    # residual error -- see _apply's deadband comment. Orientation
    # deadband uses 4x this value (radians vs meters, different scale).
    ik_holding_deadband: float = 0.012

    # Coupled bilateral waist-aware IK: NATURAL_ARM_LIFT/FOREARM_APPROACH/
    # WRIST_ALIGN/FINGERTIP_APPROACH's precision convergence gate ONLY --
    # these are the strict criteria the Coupled Bilateral Waist-Aware IK
    # session explicitly forbade relaxing (no more 0.3m/0.09m loose
    # tolerances counting as success), preserved unchanged this session.
    coupled_pos_tol: float = 0.01  # meters
    coupled_ori_tol_deg: float = 5.0
    # Consecutive steps the ACTUAL palm pose (not the trajectory tracker's
    # internal reference) must stay within tolerance, with the waist also
    # not actively moving, before the gate opens -- a single lucky frame
    # crossing the threshold is not "converged".
    coupled_stable_steps: int = 15
    coupled_waist_qvel_tol: float = 0.05  # rad/s, waist considered "settled" below this
    # Minimum-jerk joint-space trajectory duration for each coupled solve
    # (initial AND each resolve) -- chosen to keep peak joint velocity
    # bounded even for the largest single-solve joint deltas measured in
    # the A/B/D experiment (~0.3-0.5 rad across several joints at once).
    # A shorter duration for resolves specifically, and a post-trajectory
    # settle delay before measuring the next resolve's correction, were
    # BOTH tried and reverted: each measurably made an already-converging
    # size WORSE (SIZE_12 regressed from full 0.007m/0.9deg convergence to
    # a stuck 0.024m/4.4deg plateau) while not reliably fixing SIZE_6 --
    # this project's resolve dynamics are sensitive enough to timing that
    # a uniform 80-step duration with no extra settle delay is the only
    # setting confirmed to fully converge SIZE_12, so it is kept even
    # though SIZE_6 remains an open, honestly-reported gap (see
    # PROJECT_CONTEXT.md).
    coupled_transit_steps: int = 80
    # A one-shot kinematic solve only guarantees FK(q_target) matches the
    # Cartesian target -- it says nothing about whether the physical
    # (compliant kp=120) actuator actually REACHES q_target. Direct
    # measurement found it does not: a real steady-state gravity droop
    # (position AND orientation) persisted after the initial trajectory
    # finished. The resolve step corrects this via a CARTESIAN-target
    # inflation (re-solve for true_target + damping*measured_residual,
    # both position via vector addition and orientation via pose_ik.
    # so3_exp -- see _coupled_maybe_resolve and coupled_ik.py's module
    # docstring for why a JOINT-SPACE overshoot was tried first and
    # measured to diverge) -- bounded here so a genuinely infeasible
    # target still times out and fails honestly.
    coupled_max_resolves: int = 8

    # Hierarchical per-state joint roles (Natural Arm Reach + Wrist
    # Alignment + Fingertip-First Grasp session, Section 5): instead of
    # solving all 17 coupled DoF with one flat cost the whole time (which
    # let shoulder and wrist correct the SAME error simultaneously and
    # oscillate against each other), each early-approach state gives
    # shoulder/elbow and wrist different "how expensive is it to move
    # this DoF" (joint_weight, higher = more suppressed in the primary
    # task) and "how strongly is it pulled toward its rest reference"
    # (rest_gain) values, built via CoupledBilateralIK.group_vector.
    # waist keeps this project's already-validated 6.0/0.3 default in
    # every role (Coupled Bilateral Waist-Aware IK session) -- only the
    # shoulder/elbow vs wrist balance changes per state.
    coupled_waist_weight: float = 6.0
    coupled_waist_rest_gain: float = 0.3
    # NATURAL_ARM_LIFT: shoulder/elbow are the primary reach DoF (cheap,
    # weight=1, light regularization so they can move freely); wrist is
    # suppressed (expensive) and pulled hard toward the stand pose's
    # neutral orientation -- the arm should rise without the wrist
    # snapping toward the object early.
    natural_lift_shoulder_elbow_weight: float = 1.0
    natural_lift_shoulder_elbow_rest_gain: float = 0.05
    natural_lift_wrist_weight: float = 10.0
    natural_lift_wrist_rest_gain: float = 0.8
    # FOREARM_LATERAL_APPROACH: shoulder/elbow-primary, but wrist is NOT
    # given an extra per-DOF cost penalty here (weight=1, same as
    # shoulder/elbow) -- direct measurement found that any wrist penalty
    # above 1.0 (tried 2.0 and 4.0) left the lateral pre-grasp target's
    # position error permanently stuck at 0.028-0.033m (never reaching
    # the 0.01m tolerance) because the fingertip-corrected target
    # genuinely needs SOME wrist contribution to reach precisely;
    # weight=1.0 converges cleanly. The actual "wrist shouldn't do the
    # real work yet" intent for this state is enforced by ori_task_weight
    # /require_orientation instead (orientation, not position, is what
    # wrist would otherwise aggressively chase) -- rest_gain still pulls
    # wrist gently toward neutral so it does not drift further than
    # position alone requires.
    forearm_lateral_shoulder_elbow_weight: float = 1.0
    forearm_lateral_shoulder_elbow_rest_gain: float = 0.05
    forearm_lateral_wrist_weight: float = 1.0
    forearm_lateral_wrist_rest_gain: float = 0.4
    # FOREARM_DESCEND (Forearm Descent Before Wrist Alignment session):
    # the big ~0.15-0.19m vertical drop from approach_height to
    # grasp_z_offset used to be asked of FINGERTIP_APPROACH's heavily
    # posture-held shoulder/elbow -- direct measurement found wrist
    # rotation alone (bounded by the short palm-offset lever arm) cannot
    # cover that distance, which is why SIZE_6 plateaued at ~0.022m and
    # timed out while SIZE_12 (marginally) still converged. This state
    # exists so the descent happens with the SAME shoulder/elbow-primary
    # treatment as FOREARM_LATERAL_APPROACH (which is already validated
    # to converge cleanly for a large Cartesian move) -- same weights,
    # reused rather than re-tuned from scratch.
    forearm_descend_shoulder_elbow_weight: float = 1.0
    forearm_descend_shoulder_elbow_rest_gain: float = 0.05
    forearm_descend_wrist_weight: float = 1.0
    forearm_descend_wrist_rest_gain: float = 0.4
    # WRIST_ALIGN: roles flip -- shoulder/elbow become expensive and are
    # pulled toward a POSTURE-HOLD reference frozen at THIS state's entry
    # (not the stand pose, see _posture_hold_ref), so they resist further
    # gross motion; wrist becomes the cheap, primary DoF for the
    # remaining position+orientation convergence. Cartesian target is now
    # the SAME as FOREARM_DESCEND's (no big move left to make), so this
    # heavier lock is safe -- it is no longer being asked to also cover a
    # large descent the way the old (pre-this-session) FINGERTIP_APPROACH
    # was.
    wrist_align_shoulder_elbow_weight: float = 8.0
    wrist_align_shoulder_elbow_rest_gain: float = 1.0
    wrist_align_wrist_weight: float = 1.0
    wrist_align_wrist_rest_gain: float = 0.05
    # FINGERTIP_PRECONTACT: explicitly NOT a wrist-only solve (Section 6)
    # -- this state's own Cartesian move is small (outside_offset down to
    # precontact_half_width, a few cm of lateral closing at the SAME
    # height WRIST_ALIGN already converged), so shoulder/elbow can be
    # given a moderate (not near-total, unlike the old FINGERTIP_APPROACH
    # which also had to cover a large vertical drop) weight/rest_gain --
    # enough participation to help close the last few cm without
    # reverting to "wrist does everything".
    fingertip_precontact_shoulder_elbow_weight: float = 4.0
    fingertip_precontact_shoulder_elbow_rest_gain: float = 0.5
    fingertip_precontact_wrist_weight: float = 1.5
    fingertip_precontact_wrist_rest_gain: float = 0.2

    finger_force_limit: float = 8.0
    target_grip_force: float = 3.0
    max_safe_grip_force: float = 8.0
    lift_force_limit: float = 12.0
    settle_pressure_step: float = 0.0004
    force_ema_alpha: float = 0.3  # EMA smoothing factor for logged/regulated force

    # Section 4 (async per-hand CONTACT_ACQUIRE): a hand approaches at this
    # rate per step until it first touches, then HOLDS at a low setpoint
    # (not the eventual grasp force) purely to avoid pushing further into
    # the object while the other hand is still catching up -- the
    # calibration run showed a ~70-step gap between hands, and letting the
    # first hand keep advancing for that whole window is what built up an
    # asymmetric squeeze before FINGER_CLOSE ever started.
    contact_acquire_approach_step: float = 0.0003
    contact_acquire_relief_step: float = 0.0002
    contact_acquire_hold_force: float = 1.5  # N, low "just wait" setpoint
    contact_acquire_max_reacquire_attempts: int = 3
    # Matches Section 5 Gate A's literal bilateral-finger-contact duration
    # requirement (30 steps) -- was 10, an arbitrary earlier placeholder.
    # NOTE: no longer used to gate the CONTACT_ACQUIRE -> THUMB_OPPOSE
    # transition (see thumb_oppose_entry_debounce_steps below) -- kept for
    # any other caller/diagnostic that still reads it.
    both_ready_stable_steps: int = 30
    # Phase 4 -- Collision-Free Thumb Preshape + Early Tripod Closure
    # session, Section 8: a short DEBOUNCE (reject one-frame grazes), not
    # a sustained-hold requirement -- direct trace found bilateral index/
    # middle contact cannot survive anywhere near both_ready_stable_steps
    # (30) worth of consecutive steps WITHOUT thumb's opposing support in
    # the first place (contact flickers on a 6-9 step cycle), so demanding
    # 30 stable steps before ever letting thumb engage was requiring the
    # very stability thumb exists to provide.
    thumb_oppose_entry_debounce_steps: int = 4
    # Thumb Opposition + Claw-Style Bimanual Grasp session: matches Gate
    # A's literal "bilateral tripod contact >= 30 step" requirement for
    # TRIPOD_SETTLE, same duration as both_ready_stable_steps above but
    # kept as its own field since the two states check different
    # conditions (any-finger bilateral vs genuine thumb+opposing-finger
    # tripod) and may need to diverge later.
    tripod_settle_stable_steps: int = 30
    # Phase-specific thumb_1 poses (Thumb Opposition + Claw-Style Bimanual
    # Grasp session, interactive-viewer follow-up): thumb_1 has THREE
    # distinct required poses that a single 2-point (open,close) synergy
    # interpolation cannot represent -- TRANSPORT/REST (whole_body_
    # config.py's own open target, self-collision-safe, used everywhere
    # by default including STABLE_START/NATURAL_ARM_LIFT), GRASP_ABDUCT
    # (this field -- full kinematic clearance, applied ONLY from
    # THUMB_ABDUCT through just before THUMB_OPPOSE, once the arm has
    # already left the body), and opposed/CLOSE (whole_body_config.py's
    # own close target, used during THUMB_OPPOSE). Applying GRASP_ABDUCT
    # globally (as a same-day config default) was tried and reverted: it
    # fixed the real clearance problem but made the self-collision-prone
    # pose the default REST pose everywhere, which is a strictly worse
    # trade -- see whole_body_config.py's LEFT/RIGHT_HAND_SYNERGY_TARGETS
    # comment. Values are thumb_1's own measured kinematic range extremes
    # (left range=[-0.724312, 1.0472], right range=[-1.0472, 0.724312]).
    thumb1_abduct_pose_left: float = -0.724312
    thumb1_abduct_pose_right: float = 0.724312
    # Rate limit (rad/step) for the direct thumb_1 override ramp below --
    # matches this controller's general "finger target rate limit"
    # principle (Section: Index/middle acquisition), not an instant snap.
    # (A thumb_0 "detour" pose/rate used to live here too, needed when
    # thumb abducted AFTER the hand had already descended next to the
    # object -- removed once abduction moved to THUMB_CLEARANCE_PRESHAPE,
    # see _update_thumb1_override's docstring for the re-verification.)
    thumb1_override_rate: float = 0.05
    # Section 7: object must not be actively perturbed for a state to
    # accept a transition into finger closing/lifting.
    max_object_speed_for_transition: float = 0.05  # m/s
    max_object_angular_speed_for_transition: float = 0.3  # rad/s
    # Section 5: grip_center/grip_half_width control coordinate limits.
    grip_center_max_correction: float = 0.02  # m, anti-windup cap on cumulative center bias
    force_deadband: float = 0.5  # N, ignore force error smaller than this


@dataclass
class ContactCategory:
    touched: bool = False
    normal_force: float = 0.0
    tangential_force: float = 0.0
    # world-frame contact normal (pointing FROM the object surface TOWARD
    # the touching body) and contact point, taken from whichever single
    # contact in this category currently has the largest resultant force
    # (Thumb Opposition + Claw-Style Bimanual Grasp session, Section 3/10:
    # needed to score whether two fingers' contacts are genuinely opposing
    # each other, not both pushing from the same side).
    world_normal: np.ndarray | None = None
    world_point: np.ndarray | None = None

    @property
    def resultant(self) -> float:
        return float(np.hypot(self.normal_force, self.tangential_force))


@dataclass
class HandContact:
    palm: ContactCategory
    finger: ContactCategory
    thumb: ContactCategory
    index: ContactCategory
    middle: ContactCategory
    contact_count: int = 0

    @property
    def any_touch(self) -> bool:
        return self.palm.touched or self.finger.touched

    @property
    def summed_normal_force(self) -> float:
        return self.palm.normal_force + self.finger.normal_force

    @property
    def max_force(self) -> float:
        return max(self.palm.resultant, self.finger.resultant)


@dataclass
class GraspOutcome:
    state: GraspState
    failure_reason: FailureReason | None
    max_dual_contact_streak: int
    max_finger_contact_streak: int
    max_palm_only_streak: int
    max_left_finger_contact_streak: int
    max_right_finger_contact_streak: int
    max_bilateral_finger_contact_streak: int
    max_bilateral_multifinger_streak: int
    max_left_thumb_contact_streak: int
    max_right_thumb_contact_streak: int
    max_bilateral_thumb_contact_streak: int
    max_left_tripod_contact_streak: int
    max_right_tripod_contact_streak: int
    max_bilateral_tripod_streak: int
    left_opposing_normal_score: float
    right_opposing_normal_score: float
    left_thumb_contact_separation: float
    right_thumb_contact_separation: float
    left_first_finger_contact_step: int | None
    right_first_finger_contact_step: int | None
    bilateral_first_finger_contact_step: int | None
    contact_timing_gap: int | None
    left_substate: str
    right_substate: str
    left_reacquire_attempts: int
    right_reacquire_attempts: int
    left_max_force_raw: float
    right_max_force_raw: float
    left_force_filtered: float
    right_force_filtered: float
    left_desired_synergy: float  # mean of left_group_synergy -- kept for backward-compat reporting
    right_desired_synergy: float
    left_actual_synergy: float
    right_actual_synergy: float
    # Per-finger-group state (Dynamic-Aware Multi-Start IK + Multi-Finger
    # Grasp/Lift/Hold session), each a (thumb, index, middle) 3-tuple.
    left_group_synergy: tuple[float, float, float]
    right_group_synergy: tuple[float, float, float]
    left_group_contact: tuple[bool, bool, bool]
    right_group_contact: tuple[bool, bool, bool]
    left_group_force_raw: tuple[float, float, float]
    right_group_force_raw: tuple[float, float, float]
    left_finger_contact: bool
    right_finger_contact: bool
    left_palm_contact: bool
    right_palm_contact: bool
    object_xy_displacement: float
    object_table_contact: bool
    initial_object_z: float
    object_z_at_first_contact: float | None
    object_z_at_dual_contact: float | None
    object_z_at_lift_start: float | None
    maximum_object_z_before_lift: float
    maximum_object_z_during_lift: float
    final_object_z: float
    commanded_lift_target_z: float
    pre_lift_pop_height: float
    controlled_lift_gain: float
    final_height_above_initial: float
    hold_steps_achieved: int
    total_joint_travel: float
    peak_joint_velocity: float
    peak_target_jump: float


class BimanualSidePinchExpert:
    def __init__(self, env, config: GraspExpertConfig | None = None):
        self.env = env
        self.config = config or GraspExpertConfig()
        cfg = self.config

        model = env.model
        left_palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_PALM_SITE)
        right_palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.RIGHT_PALM_SITE)
        assert left_palm_id >= 0 and right_palm_id >= 0
        self.left_ctrl = pose_ik.PalmController(model, left_palm_id, env._arm_dof_adr[:7])
        self.right_ctrl = pose_ik.PalmController(model, right_palm_id, env._arm_dof_adr[7:])

        self.left_R = make_side_pinch_orientation(np.array([0.0, -1.0, 0.0]))
        self.right_R = make_side_pinch_orientation(np.array([0.0, 1.0, 0.0]))

        self._derive_size_conditioned_geometry(env.config.object_half_size)

        # Coupled bilateral waist-aware IK (Coupled Bilateral Waist-Aware
        # IK Validation session): waist(3)+left_arm(7)+right_arm(7)=17 DoF
        # solved as ONE task, used for PRE_GRASP/APPROACH only (scope
        # limited to those two states this session -- CONTACT_ACQUIRE
        # onward is untouched, still the legacy per-arm-only IK). Root
        # cause validated via a direct A/B/D causal experiment: an
        # isolated single-arm reproduction of a 6cm-object APPROACH target
        # converges to <0.1 degree orientation error, but the same target
        # in the full bimanual rollout plateaus at 30-150 degrees; locking
        # the waist (B) alone recovers most of that gap, and this coupled
        # solver (D) does BETTER than either the uncoupled baseline (A) or
        # the locked-waist diagnostic (B) on both required object sizes.
        waist_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in wbc.WAIST_JOINTS]
        self._waist_qpos_adr = np.array([model.jnt_qposadr[j] for j in waist_jids])
        self._waist_dof_adr = np.array([model.jnt_dofadr[j] for j in waist_jids])
        waist_joint_low = model.jnt_range[waist_jids, 0].copy()
        waist_joint_high = model.jnt_range[waist_jids, 1].copy()
        coupled_low = np.concatenate([waist_joint_low, env._arm_ctrl_low[:7], env._arm_ctrl_low[7:]])
        coupled_high = np.concatenate([waist_joint_high, env._arm_ctrl_high[:7], env._arm_ctrl_high[7:]])
        self._coupled_solver = CoupledBilateralIK(
            model, left_palm_id, right_palm_id,
            self._waist_dof_adr, env._arm_dof_adr[:7], env._arm_dof_adr[7:],
            self._waist_qpos_adr, env._arm_qpos_adr[:7], env._arm_qpos_adr[7:],
            coupled_low, coupled_high,
        )
        # Scratch MjData the coupled solver iterates on -- Section 6
        # explicitly forbids mutating live env/expert state during a
        # diagnostic/planning solve. Allocated once and overwritten from
        # the live state each time a solve is triggered.
        self._scratch_data = mujoco.MjData(model)

        # Fingertip grasp frame (Natural Arm Reach + Wrist Alignment +
        # Fingertip-First Grasp session) -- see _measure_fingertip_grasp_
        # offset's docstring. Every Cartesian target computed from here on
        # (_grip_targets) is expressed as an INTENDED FINGERTIP contact
        # point and converted to a palm/wrist IK target via this fixed
        # offset, instead of commanding the palm site directly to the
        # contact point.
        self._left_fingertip_offset_local = self._measure_fingertip_grasp_offset("left")
        self._right_fingertip_offset_local = self._measure_fingertip_grasp_offset("right")

        # Phase-specific thumb_0/thumb_1 direct override (see
        # GraspExpertConfig.thumb1_abduct_pose_* and env.thumb1_ctrl_
        # override_*).
        self._left_thumb1_transport_pose = wbc.LEFT_HAND_SYNERGY_TARGETS[1][1]
        self._right_thumb1_transport_pose = wbc.RIGHT_HAND_SYNERGY_TARGETS[1][1]
        self._left_thumb1_ramp = self._left_thumb1_transport_pose
        self._right_thumb1_ramp = self._right_thumb1_transport_pose
        self._thumb_clearance_streak = 0

        self._coupled_traj: _MinJerkJointTrajectory | None = None  # None = not tracking a coupled trajectory
        self._coupled_last_result: CoupledIKResult | None = None
        self._coupled_target_pos: tuple[np.ndarray, np.ndarray] | None = None  # TRUE (left, right) Cartesian goal
        self._coupled_stable_streak = 0
        self._in_coupled_phase = False  # True only while an early-approach state is tracking a coupled trajectory
        self._coupled_resolve_count = 0
        self._coupled_ik_profile: tuple = (None, None, None, 1.0, True, 0.7)  # (joint_weight, rest_gain, rest_q, ori_task_weight, require_orientation, resolve_damping)
        self._posture_hold_ref: np.ndarray | None = None  # frozen 17-dim q, captured at some states' entry

        # Fingertip sites, for distance-based approach deceleration
        # (Section 3) -- measured directly on the real controller: the
        # right hand's palm was moving at 0.46 m/s (essentially
        # max_target_step's implied ceiling, ~0.4 m/s) at the instant of
        # first contact, with no deceleration at all as it neared the
        # surface. That impact velocity, not any state-machine logic, is
        # what produced the 9-25N force spikes and 1-2cm pop.
        self._left_tip_sites = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
            for s in (wbc.LEFT_THUMB_TIP_SITE, wbc.LEFT_INDEX_TIP_SITE, wbc.LEFT_MIDDLE_TIP_SITE)
        ]
        self._right_tip_sites = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
            for s in (wbc.RIGHT_THUMB_TIP_SITE, wbc.RIGHT_INDEX_TIP_SITE, wbc.RIGHT_MIDDLE_TIP_SITE)
        ]

        # Elbow-down/rest-posture null-space target -- the standing pose's
        # own arm configuration (naturally elbow-down, joint-limit-safe).
        self._left_rest_q = env._stand_qpos[env._arm_qpos_adr[:7]].copy()
        self._right_rest_q = env._stand_qpos[env._arm_qpos_adr[7:]].copy()
        self._waist_rest_q = env._stand_qpos[self._waist_qpos_adr].copy()
        # Combined [waist, left, right] rest posture for the coupled
        # solver's null-space regularization -- same standing pose used by
        # the per-arm rest targets above, just concatenated in the
        # solver's fixed 17-DoF ordering.
        self._coupled_rest_q = np.concatenate([self._waist_rest_q, self._left_rest_q, self._right_rest_q])

        left_pos0, _ = self.left_ctrl.current_pose(env.data)
        right_pos0, _ = self.right_ctrl.current_pose(env.data)
        self.left_tracker = _TargetTracker(left_pos0, cfg.max_target_step)
        self.right_tracker = _TargetTracker(right_pos0, cfg.max_target_step)

        self.state = GraspState.STABLE_START
        self.state_step = 0
        # Per-finger-group (thumb, index, middle) desired synergy --
        # Dynamic-Aware Multi-Start IK + Multi-Finger Grasp/Lift/Hold
        # session. left/right_desired_synergy (scalars) are kept as a
        # DERIVED mean of these 3 groups for every existing threshold
        # check/report that predates per-group control -- see
        # _sync_scalar_synergy. STABLE_START..CONTACT_ACQUIRE drive all 3
        # groups identically (no behavior change from the shared-scalar
        # design); FINGER_CLOSE/FORCE_SETTLE are what actually diverge
        # them per group now, based on each group's OWN measured contact
        # force -- direct measurement found a single shared scalar could
        # not guarantee the thumb stayed engaged while index/middle were
        # independently regulated (see PROJECT_CONTEXT.md).
        self.left_group_synergy = np.zeros(3, dtype=np.float64)  # (thumb, index, middle)
        self.right_group_synergy = np.zeros(3, dtype=np.float64)
        self.left_desired_synergy = 0.0
        self.right_desired_synergy = 0.0
        self.max_dual_contact_streak = 0
        self.max_finger_contact_streak = 0
        self.max_palm_only_streak = 0
        self.max_left_finger_contact_streak = 0
        self.max_right_finger_contact_streak = 0
        self.max_bilateral_finger_contact_streak = 0
        self.max_bilateral_multifinger_streak = 0
        self._dual_streak = 0
        self._finger_streak = 0
        self._palm_only_streak = 0
        self._left_finger_streak = 0
        self._right_finger_streak = 0
        self._bilateral_finger_streak = 0
        self._bilateral_multifinger_streak = 0
        # Thumb Opposition + Claw-Style Bimanual Grasp session (Section 10)
        self.max_left_thumb_contact_streak = 0
        self.max_right_thumb_contact_streak = 0
        self.max_bilateral_thumb_contact_streak = 0
        self._left_thumb_streak = 0
        self._right_thumb_streak = 0
        self._bilateral_thumb_streak = 0
        self.max_left_tripod_contact_streak = 0
        self.max_right_tripod_contact_streak = 0
        self.max_bilateral_tripod_streak = 0
        self._left_tripod_streak = 0
        self._right_tripod_streak = 0
        self._bilateral_tripod_streak = 0
        self.left_opposing_normal_score = 0.0
        self.right_opposing_normal_score = 0.0
        self.left_thumb_contact_separation = 0.0
        self.right_thumb_contact_separation = 0.0
        self.left_first_finger_contact_step: int | None = None
        self.right_first_finger_contact_step: int | None = None
        self.bilateral_first_finger_contact_step: int | None = None
        self._global_step = 0
        self.state_transition_log: list[tuple[int, str, str, str]] = []
        self.settle_stable_streak = 0
        self._contact_loss_grace = 0
        self.left_force_ema = 0.0
        self.right_force_ema = 0.0
        self.left_max_force_raw = 0.0
        self.right_max_force_raw = 0.0
        # Per-group (thumb, index, middle) raw force this tick -- NOT
        # EMA-filtered, matching CONTACT_ACQUIRE's existing rationale for
        # using raw force there (a fast single-step contact/loss event
        # must be reacted to immediately, not smoothed away).
        self.left_group_force_raw = np.zeros(3, dtype=np.float64)
        self.right_group_force_raw = np.zeros(3, dtype=np.float64)
        self.failure_reason: FailureReason | None = None
        self._fingertip_precontact_best_err = np.inf
        self._fingertip_precontact_plateau_streak = 0

        self.grip_center = 0.0  # world y offset from object center
        self.grip_half_width = self.outside_offset
        self._z_sync_bias = 0.0  # see _update_z_sync
        self._left_ik_max_dq = cfg.ik_max_dq
        self._right_ik_max_dq = cfg.ik_max_dq

        # Section 4: per-hand CONTACT_ACQUIRE substates and independent
        # approach offsets (replacing the single shared half_width during
        # this phase only -- FINGER_CLOSE/FORCE_SETTLE still use the
        # symmetric grip_center/grip_half_width coordinates).
        self.left_substate = HandSubstate.APPROACHING
        self.right_substate = HandSubstate.APPROACHING
        self.left_approach_offset = self.outside_offset
        self.right_approach_offset = self.outside_offset
        self.left_reacquire_attempts = 0
        self.right_reacquire_attempts = 0
        self._left_ca_loss_grace = 0
        self._right_ca_loss_grace = 0
        self._both_ready_streak = 0
        self.left_first_contact_pose: np.ndarray | None = None
        self.right_first_contact_pose: np.ndarray | None = None
        self.left_first_contact_synergy: float | None = None
        self.right_first_contact_synergy: float | None = None
        self.contact_timing_gap: int | None = None  # |left_step - right_step| once both known
        self.lift_target_z = 0.0
        self._obj_ref: np.ndarray | None = None
        self._contact_ref: np.ndarray | None = None

        self.lift_left_start_pos: np.ndarray | None = None
        self.lift_right_start_pos: np.ndarray | None = None
        self.hold_left_target: np.ndarray | None = None
        self.hold_right_target: np.ndarray | None = None

        self.object_z_at_first_contact: float | None = None
        self.object_z_at_dual_contact: float | None = None
        self.object_z_at_lift_start: float | None = None
        self.maximum_object_z_before_lift = -np.inf
        self.maximum_object_z_during_lift = -np.inf
        self.final_object_z = 0.0

        self._left_raise_start = left_pos0.copy()
        self._right_raise_start = right_pos0.copy()

        self._qpos_history: list[np.ndarray] = []
        self.total_joint_travel = 0.0
        self.peak_joint_velocity = 0.0
        self.peak_target_jump = 0.0
        self._last_left_contact: HandContact | None = None
        self._last_right_contact: HandContact | None = None

    def _derive_size_conditioned_geometry(self, object_half_size: float) -> None:
        """Computes approach/grip geometry from object half-extent instead
        of hardcoding it per size or branching on a size check (Section 3,
        multi-size grasp track). Each formula was reverse-engineered to
        reproduce this module's OLD hardcoded values EXACTLY at
        half_size=0.06 (the 12cm object these were tuned against over
        several sessions), so that already-validated behavior is preserved
        bit-for-bit while 6cm/9cm objects get principled values instead of
        a copy-pasted second controller:
          grip_half_width_contact = half_size + 0.008        -> 0.068 @ 0.06 (was 0.068)
          grip_half_width_min     = half_size - 0.01          -> 0.05  @ 0.06 (was 0.05)
          outside_offset          = max(contact+0.082, 0.15)  -> 0.15  @ 0.06 (was 0.15)
          grasp_z_offset          = -0.25 * half_size          -> -0.015 @ 0.06 (was -0.015)
          approach_height         = half_size + 0.11           -> 0.17  @ 0.06 (was 0.17)
        The 0.008/0.082/0.11 constants are fingertip-reach/clearance
        margins, not calibrated per object size.

        outside_offset's floor (0.15) is a genuinely NEW, evidence-based
        correction found this session: at half_size=0.03 (6cm), the plain
        "contact + 0.082" formula gives 0.12, and a direct IK sweep at
        that exact lateral offset got permanently STUCK at a settled
        (non-oscillating) pos_err=0.22m / ori_err=65 degrees -- not
        marginal instability, a genuine equilibrium the solver cannot
        escape. Sweeping ONLY the lateral offset (same target height, same
        fixed side-pinch orientation, same gains) at 0.15m instead
        immediately converged to pos_err=0.046m / ori_err=4.9 degrees --
        i.e. the object being small does NOT mean the hand needs to
        tuck in closer to reach it; bringing the palm too close to the
        body's own centerline puts the fixed side-pinch orientation out
        of comfortable reach for this arm, regardless of the object's
        size. The floor leaves 12cm (already 0.15) byte-for-byte
        unchanged and only widens the smaller sizes' derived offset back
        up to the same practical minimum."""
        self._object_half = object_half_size
        self.grip_half_width_contact = object_half_size + 0.008
        self.grip_half_width_min = max(0.015, object_half_size - 0.01)
        self.outside_offset = max(self.grip_half_width_contact + 0.082, 0.15)
        self.grasp_z_offset = -0.25 * object_half_size
        self.approach_height = object_half_size + 0.11
        # FINGERTIP_PRECONTACT's lateral target (Forearm Descent Before
        # Wrist Alignment session, Section 2/4): a small, fixed 1cm
        # clearance beyond grip_half_width_contact -- close enough that
        # "fingertip to object surface" is on the order of a centimeter
        # (not outside_offset's much larger transport clearance), while
        # still leaving CONTACT_ACQUIRE's own incremental, force-
        # monitored approach (contact_acquire_approach_step) to cover the
        # final few mm rather than jumping straight to contact.
        self.precontact_half_width = self.grip_half_width_contact + 0.01

    def _measure_fingertip_grasp_offset(self, side: str) -> np.ndarray:
        """Measures, ONCE via forward kinematics (Natural Arm Reach +
        Wrist Alignment + Fingertip-First Grasp session, Section 2/3), the
        fixed vector from the palm-frame SITE (attached to the wrist link,
        ~5cm forward of the wrist joint -- see model_builder._add_grasp_
        sites / whole_body_config.py) to the fingertip grasp center
        (thumb/index/middle tip centroid) at preshape synergy, expressed
        in the PALM FRAME's own local coordinates.

        Root cause this exists to fix: every Cartesian target this
        controller has ever computed (_grip_targets et al.) was fed
        DIRECTLY to the palm-frame site's IK, i.e. it commanded the
        WRIST/PALM to the intended contact point -- not the fingertips.
        Direct measurement found the fingertip centroid sits
        ~9.4cm/2.0cm/0.1cm (approach/closing/lateral palm-local axes) away
        from the palm site at this controller's preshape_synergy=0.15, a
        large enough offset that "palm target == object surface" and
        "wrist/palm touches before any finger" are the same statement.
        Fingers are anatomically DISTAL to the wrist -- their position in
        palm-LOCAL coordinates depends only on finger joint angles, not on
        the arm's shoulder/elbow/wrist qpos -- so this offset is measured
        once at the CURRENT (arbitrary) arm pose and is valid everywhere
        it is used afterward, exactly like the fixed palm-frame closing
        axis convention already is.

        thumb_1 pinned to its ORIGINAL frozen value HERE ONLY (Thumb
        Opposition + Claw-Style Bimanual Grasp session): whole_body_
        config.py's thumb_1 open/close targets were corrected (previously
        identical -> frozen at all synergy values) so the new THUMB_ABDUCT/
        THUMB_OPPOSE states can actually command thumb_1 across its full
        opposition range. But this offset is the reference the ENTIRE
        untouched natural-approach pipeline targets against, and it was
        tuned over many sessions assuming thumb_1 sits at its old constant
        value (1.0472/-1.0472) at every synergy level -- letting thumb_1
        actually move here (to a much more abducted value at preshape
        synergy=0.15) shifted the centroid by several cm and broke SIZE_12
        approach convergence (measured directly: reached FINGERTIP_
        PRECONTACT but not CONTACT_ACQUIRE with 3-point average moving,
        broke even earlier with thumb dropped from the average entirely).
        Pinning thumb_1 to its old constant for THIS measurement only
        reproduces the exact pre-fix offset vector byte-for-byte (verified
        directly), leaving the natural approach untouched, while
        thumb_1's actual config range is still fully available to every
        state that sets group synergy directly (FINGER_PRESHAPE onward).

        BODY ORIGIN, not the corrected fingertip SITE (Phase 4 --
        Collision-Free Thumb Preshape + Early Tripod Closure session):
        whole_body_config.py's FINGERTIP_SITE_LOCAL_POS was corrected to
        sit at each finger's true distal tip (previously (0,0,0), which
        measured zero displacement across the whole distal-curl range on
        all 6 sites -- a real inaccuracy, needed for the NEW swept-path/
        clearance work this session adds). But swapping that corrected,
        ~5.9cm-longer site into THIS measurement shifts the centroid
        enough (+5.4cm approach axis) to break the untouched natural
        approach the same way the thumb_1 pin above was needed for
        (measured directly: FOREARM_LATERAL_APPROACH itself stops
        converging). Using each fingertip body's own ORIGIN (data.xpos)
        instead of its site reproduces the OLD (0,0,0)-site value exactly
        -- this is the SAME preshape-synergy fingertip-centroid
        convention the approach pipeline was tuned against, just
        expressed without depending on the site definition at all, so
        correcting that site for other callers can never perturb this
        measurement again."""
        model = self.env.model
        scratch = self._scratch_data
        scratch.qpos[:] = self.env.data.qpos
        scratch.qvel[:] = 0.0
        finger_qpos_adr = self.env._left_finger_qpos_adr if side == "left" else self.env._right_finger_qpos_adr
        targets = hand_synergy.left_hand_targets(self.config.preshape_synergy) if side == "left" else hand_synergy.right_hand_targets(self.config.preshape_synergy)
        targets = np.array(targets, dtype=np.float64)
        targets[1] = 1.0472 if side == "left" else -1.0472  # thumb_1, pinned -- see docstring
        scratch.qpos[finger_qpos_adr] = targets
        mujoco.mj_forward(model, scratch)

        palm_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_PALM_SITE if side == "left" else wbc.RIGHT_PALM_SITE)
        tip_bodies = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in (
                ("left_hand_thumb_2_link", "left_hand_index_1_link", "left_hand_middle_1_link")
                if side == "left" else
                ("right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link")
            )
        ]
        palm_pos = scratch.site_xpos[palm_site].copy()
        palm_R = scratch.site_xmat[palm_site].reshape(3, 3).copy()
        centroid = np.mean([scratch.xpos[b] for b in tip_bodies], axis=0)
        return palm_R.T @ (centroid - palm_pos)

    def _fingertip_target_to_palm_target(self, fingertip_target: np.ndarray, R: np.ndarray, side: str) -> np.ndarray:
        """Inverts the fixed fingertip-centroid offset (see
        _measure_fingertip_grasp_offset) to get the palm/wrist IK target
        that places the FINGERTIPS -- not the palm -- at
        ``fingertip_target`` when the hand is at orientation ``R``."""
        offset_local = self._left_fingertip_offset_local if side == "left" else self._right_fingertip_offset_local
        return fingertip_target - R @ offset_local

    # ------------------------------------------------------------------
    def _object_pos(self) -> np.ndarray:
        return self.env.data.qpos[self.env._object_qpos_adr : self.env._object_qpos_adr + 3].copy()

    def _object_table_contact(self) -> bool:
        model, data = self.env.model, self.env.data
        obj_body = self.env._object_body_id
        table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
        for i in range(data.ncon):
            c = data.contact[i]
            bodies = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
            if obj_body in bodies and table_body in bodies:
                return True
        return False

    def _sync_scalar_synergy(self) -> None:
        """Keeps left/right_desired_synergy (scalars) as the mean of
        left/right_group_synergy -- every threshold check and outcome
        field that predates per-finger-group control (Dynamic-Aware
        Multi-Start IK + Multi-Finger Grasp/Lift/Hold session) keeps
        working against a single representative number without having to
        be individually rewritten for 3 groups."""
        self.left_desired_synergy = float(self.left_group_synergy.mean())
        self.right_desired_synergy = float(self.right_group_synergy.mean())

    # States where thumb_1 stays on the normal TRANSPORT/REST<->CLOSE
    # group-synergy pipeline (whole_body_config.py's own OPEN/CLOSE table)
    # -- everything from THUMB_CLEARANCE_PRESHAPE onward uses the
    # phase-specific direct override below instead (moved earlier, Phase
    # 4 -- Collision-Free Thumb Preshape + Early Tripod Closure session --
    # thumb now finishes abducting BEFORE the vertical descent, and stays
    # abducted through FOREARM_DESCEND/WRIST_ALIGN/FINGER_PRESHAPE too,
    # so those three are no longer in this "no override" list).
    _THUMB1_NO_OVERRIDE_STATES = (
        GraspState.STABLE_START, GraspState.NATURAL_ARM_LIFT, GraspState.FOREARM_LATERAL_APPROACH,
        GraspState.FAILURE, GraspState.SUCCESS,
    )

    def _update_thumb1_override(self) -> None:
        """Sets env.thumb1_ctrl_override_left/right BEFORE env.step()
        (Thumb Opposition + Claw-Style Bimanual Grasp session, Section:
        phase-specific thumb pose separation). Writing to env.data.ctrl
        AFTER env.step() does NOT work -- env.step() unconditionally
        recomputes the whole thumb group's ctrl from the group synergy
        scalar every call, so a post-hoc write gets wiped before the next
        physics step ever uses it (measured directly: had zero effect,
        thumb kept touching early exactly as before this override
        existed). env.thumb1_ctrl_override_* is the correct hook -- see
        grasp_env.py's reset()/step() -- applied by env.step() itself
        AFTER the group loop, so it wins for this one joint every step.

        Drives thumb_1's actual open<->close motion: GRASP_ABDUCT pose
        (config.thumb1_abduct_pose_*) from THUMB_CLEARANCE_PRESHAPE
        onward, interpolating toward the CLOSE pose as THUMB_OPPOSE's own
        per-group force regulation (already applied to left/right_
        group_synergy[0] in that state's block) advances, so thumb_1
        continues smoothly from wherever it was rather than snapping back
        through the TRANSPORT pose. Before THUMB_CLEARANCE_PRESHAPE, the
        override is cleared (None) so the thumb group's normal TRANSPORT/
        REST<->CLOSE synergy interpolation (whole_body_config.py's own
        table) drives it -- correct for STABLE_START/NATURAL_ARM_LIFT/
        FOREARM_LATERAL_APPROACH, where thumb should just sit at the safe
        REST pose. Rate-limited (Section: Index/middle acquisition's
        "finger target rate limit" principle) -- never an instant jump.

        No thumb_0 detour (Phase 4 -- Collision-Free Thumb Preshape +
        Early Tripod Closure session): an earlier session found thumb_1's
        direct path clipped SIZE_12's object and routed around it via a
        thumb_0 detour, but that was specific to abducting AFTER the hand
        had already descended next to the object. Moving abduction to
        THUMB_CLEARANCE_PRESHAPE (right after FOREARM_LATERAL_APPROACH,
        still at approach_height) was re-verified with a dense 40-point
        sweep of thumb_1's full range at the live configuration: zero
        collisions anywhere, with thumb_0 left untouched at its own
        normal TRANSPORT value the whole time. The detour is no longer
        needed and was removed rather than kept as unused complexity."""
        cfg = self.config
        override_active = self.state not in self._THUMB1_NO_OVERRIDE_STATES
        if override_active:
            left_close = wbc.LEFT_HAND_SYNERGY_TARGETS[1][2]
            right_close = wbc.RIGHT_HAND_SYNERGY_TARGETS[1][2]
            t_l = float(np.clip(self.left_group_synergy[0], 0.0, 1.0))
            t_r = float(np.clip(self.right_group_synergy[0], 0.0, 1.0))
            left_goal = cfg.thumb1_abduct_pose_left + t_l * (left_close - cfg.thumb1_abduct_pose_left)
            right_goal = cfg.thumb1_abduct_pose_right + t_r * (right_close - cfg.thumb1_abduct_pose_right)
            rate1 = cfg.thumb1_override_rate
            self._left_thumb1_ramp += np.clip(left_goal - self._left_thumb1_ramp, -rate1, rate1)
            self._right_thumb1_ramp += np.clip(right_goal - self._right_thumb1_ramp, -rate1, rate1)
            self.env.thumb1_ctrl_override_left = self._left_thumb1_ramp
            self.env.thumb1_ctrl_override_right = self._right_thumb1_ramp
        else:
            # Not yet in the grasp-specific abduct phase -- track the
            # TRANSPORT pose so the ramp starts from the right place the
            # instant THUMB_CLEARANCE_PRESHAPE begins, but clear the
            # override so env.step()'s own group-synergy pipeline
            # actually drives the actuator (it already targets this same
            # safe pose at synergy 0).
            self._left_thumb1_ramp = self._left_thumb1_transport_pose
            self._right_thumb1_ramp = self._right_thumb1_transport_pose
            self.env.thumb1_ctrl_override_left = None
            self.env.thumb1_ctrl_override_right = None

    def _thumb_clearance_ready(self) -> bool:
        """Gate for THUMB_CLEARANCE_PRESHAPE -> FOREARM_DESCEND (Phase 4
        -- Collision-Free Thumb Preshape + Early Tripod Closure session,
        Section 5): uses ACTUAL qpos/qvel/contact, never the tracker's
        commanded target or the override ramp's own setpoint -- a ramp
        that has merely been COMMANDED to its endpoint says nothing about
        whether the physical joint actually got there or is still
        settling. thumb_0 is not checked here (no detour needed at this
        abduction point, see _update_thumb1_override's docstring) -- it
        stays on the normal group-synergy pipeline throughout."""
        cfg = self.config
        model, data = self.env.model, self.env.data
        left_thumb1_q = data.qpos[self.env._left_finger_qpos_adr[1]]
        right_thumb1_q = data.qpos[self.env._right_finger_qpos_adr[1]]
        left_thumb1_v = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hand_thumb_1_joint")]]
        right_thumb1_v = data.qvel[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hand_thumb_1_joint")]]

        pos_tol = 0.05  # rad
        vel_tol = 0.15  # rad/s
        converged = (
            abs(left_thumb1_q - cfg.thumb1_abduct_pose_left) < pos_tol
            and abs(right_thumb1_q - cfg.thumb1_abduct_pose_right) < pos_tol
        )
        stable = abs(left_thumb1_v) < vel_tol and abs(right_thumb1_v) < vel_tol
        left_c = self._contact("left")
        right_c = self._contact("right")
        no_contact = not (left_c.any_touch or right_c.any_touch)
        return converged and stable and no_contact

    @staticmethod
    def _group_force_raw(contact: HandContact) -> np.ndarray:
        """(thumb, index, middle) raw resultant contact force for one
        hand's HandContact snapshot."""
        return np.array([contact.thumb.resultant, contact.index.resultant, contact.middle.resultant])

    @staticmethod
    def _opposing_normal_score(contact: HandContact) -> tuple[float, str | None, float]:
        """(score, opposing_finger_name, separation) for one hand's
        HandContact snapshot (Thumb Opposition + Claw-Style Bimanual
        Grasp session, Section 3/10). Picks whichever of index/middle is
        touching AND scores best against thumb's contact normal:
        score = -dot(thumb_normal, other_normal), so +1.0 means the two
        normals point directly at each other (genuine opposition, pinch-
        like), 0.0 means perpendicular, -1.0 means both push from the
        SAME side (exactly the failure mode this session reported: "손이
        집게처럼 닫히지 않고 같은 방향에서 물체를 누름"). ``separation`` is
        the world-space distance between the two contact points -- a
        near-zero separation means the contacts are collapsing onto the
        same spot rather than spanning the object. Returns
        (0.0, None, 0.0) if thumb isn't touching or neither index nor
        middle is."""
        if not contact.thumb.touched or contact.thumb.world_normal is None:
            return 0.0, None, 0.0
        best_score, best_name, best_sep = 0.0, None, 0.0
        for name, cat in (("index", contact.index), ("middle", contact.middle)):
            if not cat.touched or cat.world_normal is None:
                continue
            score = -float(np.dot(contact.thumb.world_normal, cat.world_normal))
            sep = float(np.linalg.norm(contact.thumb.world_point - cat.world_point))
            if best_name is None or score > best_score:
                best_score, best_name, best_sep = score, name, sep
        return best_score, best_name, best_sep

    def _contact(self, side: str) -> HandContact:
        model, data = self.env.model, self.env.data
        obj_body = self.env._object_body_id
        wrist_names = (f"{side}_wrist_roll_link", f"{side}_wrist_pitch_link", f"{side}_wrist_yaw_link")
        finger_prefix = f"{side}_hand"
        thumb_prefix = f"{side}_hand_thumb"
        index_prefix = f"{side}_hand_index"
        middle_prefix = f"{side}_hand_middle"

        cats = {k: ContactCategory() for k in ("palm", "finger", "thumb", "index", "middle")}
        cat_max_resultant = {k: -1.0 for k in cats}
        contact_count = 0
        for i in range(data.ncon):
            c = data.contact[i]
            bodies = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
            if obj_body not in bodies:
                continue
            other = bodies - {obj_body}
            other_id = other.pop() if other else obj_body
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_id) or ""
            if name not in wrist_names and not name.startswith(finger_prefix):
                continue
            contact_count += 1

            # mj_contactForce returns [normal, tangent1, tangent2, torque...]
            # in the CONTACT's own frame -- decompose properly instead of a
            # single norm mixing normal and friction together.
            force6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force6)
            normal = float(force6[0])
            tangential = float(np.hypot(force6[1], force6[2]))
            resultant = float(np.hypot(normal, tangential))

            # contact.frame's first row is the world-frame normal, pointing
            # geom1 -> geom2 by MuJoCo convention -- flip so it always
            # points OBJECT -> other body, regardless of which geom slot
            # the object landed in (Section 3/10: needed for the opposing-
            # normal check, see ContactCategory.world_normal).
            frame_normal = np.array(c.frame[0:3], dtype=np.float64)
            world_normal = frame_normal if model.geom_bodyid[c.geom1] == obj_body else -frame_normal
            world_point = np.array(c.pos, dtype=np.float64)

            def _accum(key: str):
                cat = cats[key]
                cat.touched = True
                cat.normal_force = max(cat.normal_force, abs(normal))
                cat.tangential_force = max(cat.tangential_force, tangential)
                if resultant >= cat_max_resultant[key]:
                    cat_max_resultant[key] = resultant
                    cat.world_normal = world_normal
                    cat.world_point = world_point

            if name in wrist_names:
                _accum("palm")
            elif name.startswith(finger_prefix):
                _accum("finger")
                if name.startswith(thumb_prefix):
                    _accum("thumb")
                elif name.startswith(index_prefix):
                    _accum("index")
                elif name.startswith(middle_prefix):
                    _accum("middle")

        return HandContact(contact_count=contact_count, **cats)

    def _apply(
        self, left_target_pos, right_target_pos, left_group_synergy=None, right_group_synergy=None,
        null_gain: float | None = None, left_max_dq: float | None = None, right_max_dq: float | None = None,
        ik_gain: float | None = None, left_is_holding: bool = False, right_is_holding: bool = False,
    ):
        cfg = self.config
        # ik_gain override: direct experiment found cfg.ik_gain=0.3 is only
        # MARGINALLY stable -- for some targets (e.g. FINGER_PRESHAPE's)
        # it produces a genuinely DIVERGING oscillation (palm velocity
        # 0.44->0.57 m/s and still rising after 80 steps at a completely
        # STATIC target), not the settled small-residual behavior found
        # when this gain was originally tuned against a different target.
        # A lower gain (0.15) with null_gain=0 converges smoothly and
        # monotonically for the same target (0.03->0.04 m/s, actually a
        # slightly BETTER final position error too) -- this is what
        # actually determines first-contact impact velocity, not the
        # target's own movement rate (see left_max_dq comment below, which
        # was an earlier, ineffective attempt at the same problem).
        ikg = ik_gain if ik_gain is not None else cfg.ik_gain
        # The null-space projector is only EXACT when the damped
        # pseudo-inverse has zero damping; with damping>0 (needed for
        # numerical stability) it leaks a small component into task space.
        # Direct trace during CONTACT_ACQUIRE showed the held palm z
        # drifting from 0.802m to 0.824m over ~15 steps with NO target
        # change and a hand that was supposedly just "holding position" --
        # traced to this leak being pulled toward the (higher) rest pose.
        # Precision-holding phases (contact acquisition onward) disable
        # the null-space pull entirely; phases that need it for reachability
        # (PRE_GRASP/APPROACH, where the earlier joint-limit lockup
        # happened) keep it.
        ng = null_gain if null_gain is not None else cfg.null_gain
        # left_max_dq/right_max_dq: per-hand joint-space rate override.
        # Direct trace found that slowing the TARGET's own movement rate
        # (max_target_step) does nothing about first-contact impact speed
        # when the actual palm is lagging BEHIND an already-static target
        # from an earlier state (measured: 17mm backlog carried into
        # CONTACT_ACQUIRE from APPROACH/FINGER_PRESHAPE's 6cm convergence
        # tolerance) -- the IK's own gain/damping "catches up" to that
        # backlog at up to 0.46 m/s regardless of how slowly the target
        # itself creeps. Capping the JOINT velocity itself (not the
        # target) is what actually limits impact speed independent of
        # where the backlog came from.
        lmdq = left_max_dq if left_max_dq is not None else cfg.ik_max_dq
        rmdq = right_max_dq if right_max_dq is not None else cfg.ik_max_dq
        left_q = self.env.data.qpos[self.env._arm_qpos_adr[:7]]
        right_q = self.env.data.qpos[self.env._arm_qpos_adr[7:]]
        left_dq, left_conv = self.left_ctrl.step_toward(
            self.env.data, left_target_pos, self.left_R, gain=ikg, damping=cfg.ik_damping,
            max_dq=lmdq, arm_qpos=left_q, rest_qpos=self._left_rest_q, null_gain=ng,
        )
        right_dq, right_conv = self.right_ctrl.step_toward(
            self.env.data, right_target_pos, self.right_R, gain=ikg, damping=cfg.ik_damping,
            max_dq=rmdq, arm_qpos=right_q, rest_qpos=self._right_rest_q, null_gain=ng,
        )
        # dq low-pass filter while holding -- TRIED AND REVERTED (Dynamic-
        # Aware Multi-Start IK + Multi-Finger Grasp/Lift/Hold session). The
        # A/B/C separation experiment (see PROJECT_CONTEXT.md Known Issues)
        # isolated the IK/grip-center re-solve loop as the dominant single
        # contributor to object rotation during holding, and a hard on/off
        # "IK deadband" tried for the same problem already regressed Gate A
        # (43-57 -> 26 step streak) and was reverted. An EMA low-pass on dq
        # while left_is_holding/right_is_holding (continuous, no on/off
        # boundary, hypothesized to avoid the deadband's chattering) was
        # tried as a third approach and directly measured WORSE, not
        # better: max_bilateral_multifinger_streak 57->43,
        # max_dual_contact_streak 148->103, obj_xy_disp 0.089->0.143m on
        # the same SIZE_12 seed. Slowing the correction makes it lag behind
        # the object's actual slip instead of catching it -- same
        # "more tolerance/damping is not automatically better" pattern as
        # the deadband and the earlier contact-loss grace-period reverts.
        # left_is_holding/right_is_holding are kept as parameters (unused
        # in this function body) since a future, different use may still
        # want them.
        # IK deadband (Section 11 "target continuity" / "don't chase small
        # noise"): an A/B/C separation experiment isolated the IK/grip-
        # center re-targeting loop as the DOMINANT source of sustained
        # object rotation during a holding phase -- constant-hold gave
        # mean |omega|=0.27 rad/s, adding the finger-force loop alone gave
        # 0.51, and the full loop (this IK re-solve included) gave 1.68.
        # The goal barely moves once the grip is settled, so re-solving IK
        # every single step even for a sub-mm residual error was
        # continuously perturbing the arm; once the palm is already within
        # this band of its target, hold the current ctrl exactly (dq=0)
        # instead of re-chasing sub-mm noise.
        # An IK deadband (force dq=0 once within a small position/
        # orientation band, instead of continuously re-solving toward a
        # near-static target) was tried here and REVERTED: it empirically
        # made Gate A worse, not better -- max_bilateral_finger_contact_
        # streak dropped from 43-57 to 26 on the same seed, and the
        # isolated A/B/C replay showed contact lost within 6 steps under
        # the full controller (vs. never within a 60-step window before).
        # The on/off discontinuity at the deadband boundary appears to
        # create its own chattering, trading one source of disturbance for
        # another rather than removing it. Left as a documented negative
        # result (see PROJECT_CONTEXT.md) -- the A/B/C experiment's
        # diagnosis (IK/grip-center loop as the largest single
        # contributor) stands, but deadbanding was not the fix.
        # Action is 23-dim (waist3 + arm14 + left_hand_groups3 +
        # right_hand_groups3, see grasp_env.py's module docstring). This
        # method is only used for states OTHER than the early-approach
        # states' coupled-IK precision phase (see _step_coupled_action)
        # -- those states never articulate the waist, so a[0:3] is left
        # at zero (hold current waist target).
        a = np.zeros(23, dtype=np.float64)
        a[3:10] = np.clip(left_dq / self.env.config.arm_action_scale, -1, 1)
        a[10:17] = np.clip(right_dq / self.env.config.arm_action_scale, -1, 1)

        left_g = left_group_synergy if left_group_synergy is not None else self.left_group_synergy
        right_g = right_group_synergy if right_group_synergy is not None else self.right_group_synergy
        env_group_syn = self.env._hand_group_synergy
        scale = self.env.config.hand_synergy_action_scale
        a[17:20] = np.clip((left_g - env_group_syn[0:3]) / scale, -1, 1)
        a[20:23] = np.clip((right_g - env_group_syn[3:6]) / scale, -1, 1)

        return a.astype(np.float32), left_conv, right_conv

    # ---- Coupled bilateral waist-aware IK (early-approach states only) --
    def _coupled_role_profile(self, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, float]:
        """Builds (joint_weight, rest_gain, rest_q, ori_task_weight,
        require_orientation, resolve_damping) for one of the 5
        early-approach states' hierarchical joint roles (Section 5,
        Forearm Descent Before Wrist Alignment session). See
        GraspExpertConfig's coupled_waist_weight/natural_lift_*/
        forearm_lateral_*/forearm_descend_*/wrist_align_*/
        fingertip_precontact_* fields for the actual numbers.
        WRIST_ALIGN/FINGERTIP_PRECONTACT use a POSTURE-HOLD rest_q (frozen
        at WRIST_ALIGN's entry -- i.e. FOREARM_DESCEND's exit, captured in
        self._posture_hold_ref) instead of the stand pose, so their high
        shoulder/elbow rest_gain resists drifting from wherever the arm
        already is, not pulls it back to standing.

        NATURAL_ARM_LIFT/FOREARM_LATERAL_APPROACH/FOREARM_DESCEND
        additionally de-prioritize orientation entirely (low
        ori_task_weight, require_orientation=False) -- per-DOF
        joint_weight alone cannot stop the solver from fighting for the
        exact final grasp orientation this early (the primary task still
        targets the full 6D pose no matter how "expensive" wrist is), so
        these three states solve position as the real objective and let
        orientation drift passively; WRIST_ALIGN re-solves the SAME
        target with orientation reinstated as a full, required part of
        the task.

        ``resolve_damping`` (0.0 otherwise, 0.5 for WRIST_ALIGN/
        FINGERTIP_PRECONTACT): _coupled_maybe_resolve's Cartesian-target
        inflation was found to DIVERGE for restricted-DOF roles at full
        (0.7) damping -- with shoulder/elbow heavily posture-held, wrist
        alone has much less redundancy than the full 7-DOF arm, and
        direct measurement showed inflating the target too aggressively
        pushed the solve into a worse configuration every resolve (0.01m/
        5deg residual growing to 0.19m/14deg over 7 resolves at damping
        0.7; 0.5 converges cleanly). NATURAL_ARM_LIFT/FOREARM_LATERAL_
        APPROACH/FOREARM_DESCEND have the full 7-DOF arm free (shoulder/
        elbow are the PRIMARY DoF, not posture-held), so the original
        0.7 damping is safe and effective there."""
        cfg = self.config
        gv = CoupledBilateralIK.group_vector
        if role == "natural_lift":
            weight = gv(cfg.coupled_waist_weight, cfg.natural_lift_shoulder_elbow_weight, cfg.natural_lift_wrist_weight)
            rest_gain = gv(cfg.coupled_waist_rest_gain, cfg.natural_lift_shoulder_elbow_rest_gain, cfg.natural_lift_wrist_rest_gain)
            rest_q = self._coupled_rest_q
            ori_task_weight, require_orientation, resolve_damping = 0.1, False, 0.7
        elif role == "forearm_lateral":
            weight = gv(cfg.coupled_waist_weight, cfg.forearm_lateral_shoulder_elbow_weight, cfg.forearm_lateral_wrist_weight)
            rest_gain = gv(cfg.coupled_waist_rest_gain, cfg.forearm_lateral_shoulder_elbow_rest_gain, cfg.forearm_lateral_wrist_rest_gain)
            rest_q = self._coupled_rest_q
            ori_task_weight, require_orientation, resolve_damping = 0.3, False, 0.7
        elif role == "forearm_descend":
            weight = gv(cfg.coupled_waist_weight, cfg.forearm_descend_shoulder_elbow_weight, cfg.forearm_descend_wrist_weight)
            rest_gain = gv(cfg.coupled_waist_rest_gain, cfg.forearm_descend_shoulder_elbow_rest_gain, cfg.forearm_descend_wrist_rest_gain)
            rest_q = self._coupled_rest_q
            ori_task_weight, require_orientation, resolve_damping = 0.3, False, 0.7
        elif role == "wrist_align":
            weight = gv(cfg.coupled_waist_weight, cfg.wrist_align_shoulder_elbow_weight, cfg.wrist_align_wrist_weight)
            rest_gain = gv(cfg.coupled_waist_rest_gain, cfg.wrist_align_shoulder_elbow_rest_gain, cfg.wrist_align_wrist_rest_gain)
            rest_q = self._posture_hold_ref if self._posture_hold_ref is not None else self._coupled_rest_q
            ori_task_weight, require_orientation, resolve_damping = 1.0, True, 0.5
        elif role == "fingertip_precontact":
            weight = gv(cfg.coupled_waist_weight, cfg.fingertip_precontact_shoulder_elbow_weight, cfg.fingertip_precontact_wrist_weight)
            rest_gain = gv(cfg.coupled_waist_rest_gain, cfg.fingertip_precontact_shoulder_elbow_rest_gain, cfg.fingertip_precontact_wrist_rest_gain)
            rest_q = self._posture_hold_ref if self._posture_hold_ref is not None else self._coupled_rest_q
            ori_task_weight, require_orientation, resolve_damping = 1.0, True, 0.5
        else:
            raise ValueError(f"unknown coupled IK role: {role}")
        return weight, rest_gain, rest_q, ori_task_weight, require_orientation, resolve_damping

    def _capture_posture_hold_ref(self) -> None:
        """Freezes the CURRENTLY COMMANDED 17-dim coupled setpoint
        (env._waist_target/_arm_target, matching every other "commanded
        not actual" convention in this controller) as the posture-hold
        reference for WRIST_ALIGN/FINGERTIP_APPROACH's high shoulder/
        elbow rest_gain -- called once at entry to WRIST_ALIGN, not
        re-captured on later resolves within the same state."""
        self._posture_hold_ref = np.concatenate([self.env._waist_target, self.env._arm_target])

    def _start_coupled_transit(
        self, left_target_pos: np.ndarray, right_target_pos: np.ndarray,
        left_true_target: np.ndarray | None = None, right_true_target: np.ndarray | None = None,
        left_target_R: np.ndarray | None = None, right_target_R: np.ndarray | None = None,
        joint_weight: np.ndarray | None = None, rest_gain: np.ndarray | float | None = None,
        rest_q: np.ndarray | None = None,
        ori_task_weight: float = 1.0, require_orientation: bool = True,
        resolve_damping: float = 0.7,
    ) -> None:
        """Solves the 17-DoF coupled IK ONCE on the scratch MjData (a copy
        of the live state, never the live state itself -- Section 6) and
        starts a minimum-jerk joint-space trajectory from the CURRENTLY
        COMMANDED setpoint (env._waist_target/_arm_target, not the live
        physical qpos, matching how _TargetTracker always tracked the
        commanded Cartesian point rather than the actual palm) toward the
        solve's result.

        ``left_target_pos``/``right_target_pos`` is what gets SOLVED for
        (a resolve may pass an INFLATED position here to compensate for
        measured actuator droop -- see _coupled_maybe_resolve).
        ``left_true_target``/``right_true_target`` (defaulting to the
        solve targets themselves, for the initial call) is the ACTUAL
        desired Cartesian goal used by _coupled_converged -- these must
        stay separate so an inflated planning target is never mistaken
        for the real Approach Gate criterion.

        ``joint_weight``/``rest_gain``/``rest_q`` (Natural Arm Reach +
        Wrist Alignment session, Section 5 -- hierarchical per-state joint
        roles) let each state give shoulder/elbow/wrist different task-
        priority and posture-hold treatment (e.g. NATURAL_ARM_LIFT makes
        wrist "expensive" and pulls it to a neutral rest pose; WRIST_ALIGN
        does the opposite). Defaults to the coupled solver's own
        constructor default (waist-only weighting) and the stand-pose
        rest reference, matching the original PRE_GRASP/APPROACH
        behavior exactly when a caller does not override them. The
        resolved profile is cached (_coupled_ik_profile) so a later
        _coupled_maybe_resolve reuses the SAME per-state role weighting."""
        scratch = self._scratch_data
        scratch.qpos[:] = self.env.data.qpos
        scratch.qvel[:] = self.env.data.qvel
        scratch.ctrl[:] = self.env.data.ctrl
        mujoco.mj_forward(self.env.model, scratch)

        lR = left_target_R if left_target_R is not None else self.left_R
        rR = right_target_R if right_target_R is not None else self.right_R
        rest_q_used = rest_q if rest_q is not None else self._coupled_rest_q
        rest_gain_used = 0.3 if rest_gain is None else rest_gain
        q_start = np.concatenate([self.env._waist_target, self.env._arm_target])
        result = self._coupled_solver.solve(
            scratch, left_target_pos, lR, right_target_pos, rR, rest_q_used,
            rest_gain=rest_gain_used, joint_weight=joint_weight,
            ori_task_weight=ori_task_weight, require_orientation=require_orientation,
        )
        self._coupled_last_result = result
        self._coupled_target_pos = (
            (left_true_target if left_true_target is not None else left_target_pos).copy(),
            (right_true_target if right_true_target is not None else right_target_pos).copy(),
        )
        self._coupled_ik_profile = (joint_weight, rest_gain, rest_q, ori_task_weight, require_orientation, resolve_damping)
        q_target = np.concatenate([result.waist_q, result.left_q, result.right_q])
        self._coupled_traj = _MinJerkJointTrajectory(q_start, q_target, self.config.coupled_transit_steps)
        self._coupled_stable_streak = 0

    def _coupled_maybe_resolve(self) -> None:
        """Once the current trajectory has finished but the ACTUAL palm
        pose still isn't within tolerance (compliant-actuator steady-state
        droop under gravity, not a kinematic infeasibility -- see
        coupled_ik.py's module docstring), re-solves for an INFLATED
        Cartesian target (true target + a fraction of the measured
        residual) via the SAME bounded, line-search-protected solve()
        used for the initial solve -- NOT a naive re-solve of the
        unchanged original target (which just reproduces ~the same joint
        solution and never closes the gap), and NOT a direct joint-space
        overshoot (an earlier version of this tried adding the measured
        joint-space shortfall directly to the joint target every resolve
        and measured it DIVERGE -- droop is not constant enough across a
        changing pose for that to be a stable correction; going through
        solve() again keeps every resolve bounded and kinematically
        sane).

        BOTH position and orientation are inflated -- an earlier version
        of this only inflated position (leaving the orientation target
        fixed at the true desired R) and measured the residual ORIENTATION
        error plateau at ~7-9 degrees indefinitely across 6+ resolves
        (position alone got close, ~0.014-0.03m, but never below the
        5-degree orientation tolerance): the droop this is compensating
        for has an orientation component too, and only correcting the
        position half of it left orientation permanently uncorrected.
        Orientation is inflated via the SO(3) exp map applied to the
        measured axis-angle error (pose_ik.so3_exp), the rotational
        analogue of adding a vector to a position target.

        Damped fraction (not the full measured residual) avoids
        overshooting past the target on the next round. Bounded by
        coupled_max_resolves so a genuinely infeasible target still times
        out and fails honestly rather than looping forever."""
        if self._coupled_traj.elapsed < self._coupled_traj.duration:
            return
        left_target_pos, right_target_pos = self._coupled_target_pos
        joint_weight, rest_gain, rest_q, ori_task_weight, require_orientation, damping = self._coupled_ik_profile
        if self._coupled_converged(left_target_pos, right_target_pos, require_orientation=require_orientation):
            return
        if self._coupled_resolve_count >= self.config.coupled_max_resolves:
            return
        left_actual, left_actual_R = self.left_ctrl.current_pose(self.env.data)
        right_actual, right_actual_R = self.right_ctrl.current_pose(self.env.data)
        left_inflated = left_target_pos + damping * (left_target_pos - left_actual)
        right_inflated = right_target_pos + damping * (right_target_pos - right_actual)
        # Orientation is only inflated when this state's profile actually
        # REQUIRES it -- a state that deliberately deprioritizes
        # orientation (NATURAL_ARM_LIFT/FOREARM_APPROACH, see
        # _coupled_role_profile) never converges its orientation, so
        # inflating that ALREADY-large, untracked residual by 0.7x on
        # every resolve would ratchet the orientation target further and
        # further from reality for no benefit -- measured directly:
        # solve()'s own reported orientation residual climbed from 5.5 to
        # 20+ degrees over 7 resolves before this fix, without helping
        # (or even while hurting) the position convergence these two
        # states actually care about.
        if require_orientation:
            left_ori_err = pose_ik.orientation_error(left_actual_R, self.left_R)
            right_ori_err = pose_ik.orientation_error(right_actual_R, self.right_R)
            left_R_inflated = pose_ik.so3_exp(damping * left_ori_err) @ self.left_R
            right_R_inflated = pose_ik.so3_exp(damping * right_ori_err) @ self.right_R
        else:
            left_R_inflated, right_R_inflated = self.left_R, self.right_R
        self._start_coupled_transit(
            left_inflated, right_inflated, left_target_pos, right_target_pos,
            left_target_R=left_R_inflated, right_target_R=right_R_inflated,
            ori_task_weight=ori_task_weight, require_orientation=require_orientation,
            joint_weight=joint_weight, rest_gain=rest_gain, rest_q=rest_q,
            resolve_damping=damping,
        )
        self._coupled_resolve_count += 1

    def _step_coupled_action(self, left_group_synergy: np.ndarray | None = None, right_group_synergy: np.ndarray | None = None) -> np.ndarray:
        """Converts the current point on the min-jerk joint trajectory into
        a 23-dim action DIRECTLY (bypassing ``_apply``'s per-arm Cartesian
        IK entirely for the early-approach states) -- the action is just
        the delta between this step's reference joint position and the
        currently-accumulated setpoint, exactly mirroring how the old
        Cartesian ``_TargetTracker`` fed ``_apply``."""
        q_ref = self._coupled_traj.step()
        waist_ref, left_ref, right_ref = q_ref[0:3], q_ref[3:10], q_ref[10:17]
        waist_dq = waist_ref - self.env._waist_target
        left_dq = left_ref - self.env._arm_target[:7]
        right_dq = right_ref - self.env._arm_target[7:]

        a = np.zeros(23, dtype=np.float64)
        a[0:3] = np.clip(waist_dq / self.env.config.waist_action_scale, -1, 1)
        a[3:10] = np.clip(left_dq / self.env.config.arm_action_scale, -1, 1)
        a[10:17] = np.clip(right_dq / self.env.config.arm_action_scale, -1, 1)
        left_g = left_group_synergy if left_group_synergy is not None else self.left_group_synergy
        right_g = right_group_synergy if right_group_synergy is not None else self.right_group_synergy
        env_group_syn = self.env._hand_group_synergy
        scale = self.env.config.hand_synergy_action_scale
        a[17:20] = np.clip((left_g - env_group_syn[0:3]) / scale, -1, 1)
        a[20:23] = np.clip((right_g - env_group_syn[3:6]) / scale, -1, 1)
        return a.astype(np.float32)

    def _coupled_converged(self, left_target_pos: np.ndarray, right_target_pos: np.ndarray, require_orientation: bool = True) -> bool:
        """Strict Approach-Gate convergence check (Section 10): the ACTUAL
        palm pose against the TRUE Cartesian target, position<=1cm (AND
        orientation<=5deg for BOTH hands, when ``require_orientation`` --
        NATURAL_ARM_LIFT/FOREARM_APPROACH pass False since they
        deliberately do not solve for orientation, see
        _coupled_role_profile), AND the waist not actively moving --
        sustained for coupled_stable_steps consecutive calls (tracked by
        the caller via self._coupled_stable_streak). A single frame
        crossing the threshold is not treated as convergence."""
        left_pos, left_R = self.left_ctrl.current_pose(self.env.data)
        right_pos, right_R = self.right_ctrl.current_pose(self.env.data)
        lp = float(np.linalg.norm(left_target_pos - left_pos))
        rp = float(np.linalg.norm(right_target_pos - right_pos))
        waist_qvel = float(np.linalg.norm(self.env.data.qvel[self._waist_dof_adr]))
        pos_ok = lp < self.config.coupled_pos_tol and rp < self.config.coupled_pos_tol and waist_qvel < self.config.coupled_waist_qvel_tol
        if not require_orientation:
            return pos_ok
        lo = float(np.linalg.norm(pose_ik.orientation_error(left_R, self.left_R)))
        ro = float(np.linalg.norm(pose_ik.orientation_error(right_R, self.right_R)))
        ori_tol = np.radians(self.config.coupled_ori_tol_deg)
        return pos_ok and lo < ori_tol and ro < ori_tol

    def _grip_targets(self, base_pos: np.ndarray, z: float, half_width: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (left_palm_target, right_palm_target) -- the palm/wrist
        IK targets. The formula below computes the INTENDED FINGERTIP
        grasp-contact point (unchanged from every prior session's
        geometry, still size-conditioned via half_width/z) and then
        converts it to a palm target via the fixed fingertip-centroid
        offset (Section 3, Natural Arm Reach session) -- every caller of
        this function (approach waypoints, contact-acquire, force-settle,
        lift/hold) now aims the FINGERS at the geometry that was always
        intended for them, instead of the wrist/palm site.

        ``half_width`` (Forearm Descent Before Wrist Alignment session,
        Section 3): defaults to ``self.grip_half_width`` (the shared,
        mutable transport/contact-acquire lateral coordinate) when not
        given, but callers that need a DIFFERENT lateral target without
        mutating that shared state (e.g. FINGERTIP_PRECONTACT's small
        surface-normal approach to ``precontact_half_width``, distinct
        from the wide transport clearance every earlier state uses) can
        pass one explicitly."""
        hw = half_width if half_width is not None else self.grip_half_width
        left_fingertip_target = base_pos + np.array([0.0, self.grip_center + hw, z + self._z_sync_bias])
        right_fingertip_target = base_pos + np.array([0.0, self.grip_center - hw, z - self._z_sync_bias])
        left = self._fingertip_target_to_palm_target(left_fingertip_target, self.left_R, "left")
        right = self._fingertip_target_to_palm_target(right_fingertip_target, self.right_R, "right")
        return left, right

    def _min_fingertip_distance(self, side: str, obj_pos: np.ndarray) -> float:
        """Signed distance from the nearest of a hand's 3 fingertip sites
        to the (axis-aligned, unrotated pre-contact) object surface --
        used to schedule approach deceleration (Section 3). Negative means
        already penetrating."""
        half = self.env.config.object_half_size
        sites = self._left_tip_sites if side == "left" else self._right_tip_sites
        dists = []
        for sid in sites:
            local = self.env.data.site_xpos[sid] - obj_pos
            q = np.abs(local) - half
            outside = float(np.linalg.norm(np.maximum(q, 0.0)))
            inside = float(min(0.0, np.max(q)))
            dists.append(outside + inside)
        return min(dists)

    def _approach_speed_scale(self, distance: float) -> float:
        """Distance-based deceleration profile (Section 3): full speed
        while far from the surface, ramping down to a small fraction right
        at contact. Direct measurement on the un-decelerated controller
        showed a palm approach velocity of 0.46 m/s at the instant of
        first contact (essentially max_target_step's ~0.4 m/s ceiling,
        applied with zero slowdown) -- the direct cause of 9-25N impact
        force spikes and a 1-2cm object pop. Tuned so the last few
        millimeters are covered at roughly 1/20 of the transit speed."""
        near = 0.02  # start decelerating within 2cm of the surface
        min_fraction = 0.05
        if distance <= 0.0:
            return min_fraction
        if distance >= near:
            return 1.0
        t = distance / near
        return min_fraction + (1.0 - min_fraction) * t

    def _update_z_sync(self) -> None:
        """Both palms are commanded to the SAME z, but their IK converges
        to slightly different residuals (measured: left 5.3mm higher than
        right right at FINGER_CLOSE entry) since the two arms' kinematics
        are mirrored, not identical. That small mismatch was enough to
        impart a roll torque on the object during squeezing -- direct
        trace showed the object spinning up to 3+ rad/s about the pinch
        axis and being flicked away before any deliberate lift began. Bias
        each side's target z by half the currently-measured gap, closing
        it directly rather than hoping the IK residuals happen to match."""
        left_actual, _ = self.left_ctrl.current_pose(self.env.data)
        right_actual, _ = self.right_ctrl.current_pose(self.env.data)
        gap = left_actual[2] - right_actual[2]
        self._z_sync_bias = np.clip(self._z_sync_bias - 0.3 * gap, -0.02, 0.02)

    def _contact_acquire_hand_step(
        self, touch: bool, force: float, offset: float, substate: HandSubstate, reacquire_attempts: int, loss_grace: int
    ) -> tuple[float, HandSubstate, int, int]:
        """One hand's async CONTACT_ACQUIRE step (Section 4). APPROACHING
        hands close the gap at a fixed slow rate until they first touch;
        FIRST_CONTACT immediately becomes FORCE_HOLD, which holds position
        (only relieving outward if force exceeds a small "just wait"
        setpoint -- NOT the eventual grasp force) instead of continuing to
        press inward while the other hand is still approaching. A held
        hand that loses contact gets a limited number of REACQUIRE
        attempts before being left as CONTACT_LOST for the caller to
        decide whether the whole grasp attempt has failed."""
        cfg = self.config
        if substate == HandSubstate.APPROACHING:
            if touch:
                substate = HandSubstate.FIRST_CONTACT
                loss_grace = 0
            else:
                offset = max(self.grip_half_width_min, offset - cfg.contact_acquire_approach_step)
        elif substate == HandSubstate.FIRST_CONTACT:
            substate = HandSubstate.FORCE_HOLD
        elif substate == HandSubstate.FORCE_HOLD:
            if touch:
                loss_grace = 0
                if force > cfg.contact_acquire_hold_force:
                    offset = min(self.outside_offset, offset + cfg.contact_acquire_relief_step)
                elif force < 1.0:
                    # A single-point graze registers "touched" for one step
                    # then the force decays to ~0 within 2-3 steps even
                    # though the target never moved (measured: left hand
                    # 6.5N -> 0.0N over 4 steps while holding position).
                    # React early (below 1.0N, full approach rate) to firm
                    # up a weak/marginal touch before it fully decays.
                    offset = max(self.grip_half_width_min, offset - cfg.contact_acquire_approach_step)
                # else: hold position exactly -- fix for the measured
                # asymmetric-squeeze pattern (see HandSubstate).
            else:
                # A 2-step no-touch grace period was tried here (tolerate
                # brief discretization noise before giving up the hold) but
                # made outcomes WORSE, not better: it let the object be
                # repeatedly re-grazed across many REACQUIRE cycles instead
                # of failing cleanly, and direct trace showed this
                # compounding into a violent tumble (angular velocity up to
                # 8.2 rad/s, well past any "settling" magnitude) before
                # finally failing OBJECT_DISPLACED at step 378 -- a worse,
                # later, and less diagnosable failure than the immediate
                # SINGLE_HAND_CONTACT_ONLY this reverts to. A momentary
                # no-touch reading here is treated as a real separation.
                substate = HandSubstate.CONTACT_LOST
                loss_grace = 0
        elif substate == HandSubstate.CONTACT_LOST:
            if reacquire_attempts < cfg.contact_acquire_max_reacquire_attempts:
                substate = HandSubstate.REACQUIRE
                reacquire_attempts += 1
            # else: stays CONTACT_LOST, terminal for this hand -- caller
            # decides whether that fails the whole attempt.
        elif substate == HandSubstate.REACQUIRE:
            if touch:
                substate = HandSubstate.FORCE_HOLD
                loss_grace = 0
            else:
                offset = max(self.grip_half_width_min, offset - cfg.contact_acquire_approach_step)
        # READY: terminal, nothing to do (state machine has moved on).
        return offset, substate, reacquire_attempts, loss_grace

    # ------------------------------------------------------------------
    def step(self) -> GraspOutcome:
        cfg = self.config
        obj = self._object_pos()
        env_obj_vel = self.env.data.qvel[self.env._object_dof_adr : self.env._object_dof_adr + 6].copy()
        if self._obj_ref is None:
            self._obj_ref = obj.copy()
        self.final_object_z = obj[2]

        left_c = self._contact("left")
        right_c = self._contact("right")
        self._last_left_contact, self._last_right_contact = left_c, right_c
        left_touch, left_force_raw = left_c.any_touch, left_c.max_force
        right_touch, right_force_raw = right_c.any_touch, right_c.max_force
        self.left_max_force_raw = max(self.left_max_force_raw, left_force_raw)
        self.right_max_force_raw = max(self.right_max_force_raw, right_force_raw)

        a_ema = cfg.force_ema_alpha
        self.left_force_ema = a_ema * left_force_raw + (1 - a_ema) * self.left_force_ema
        self.right_force_ema = a_ema * right_force_raw + (1 - a_ema) * self.right_force_ema
        left_force = self.left_force_ema
        right_force = self.right_force_ema

        # Per-group (thumb, index, middle) raw force -- RAW, not EMA
        # filtered, matching CONTACT_ACQUIRE's existing rationale (fast
        # single-step contact/loss events need an immediate reaction, not
        # a smoothed one). Used by FINGER_CLOSE/FORCE_SETTLE's per-group
        # regulation (Dynamic-Aware Multi-Start IK + Multi-Finger Grasp/
        # Lift/Hold session).
        self.left_group_force_raw = self._group_force_raw(left_c)
        self.right_group_force_raw = self._group_force_raw(right_c)

        both_finger = left_c.finger.touched and right_c.finger.touched
        palm_only = (left_touch or right_touch) and not (left_c.finger.touched or right_c.finger.touched)
        # Section 10/11 (Dynamic-Aware Multi-Start IK + Multi-Finger
        # Grasp/Lift/Hold session): "any finger" (both_finger above) does
        # not distinguish a single grazing contact from a genuine
        # multi-point grip -- tracked separately so Gate A's literal
        # "thumb 외 추가 finger contact" requirement can be measured and
        # reported, not just assumed from the looser both_finger streak.
        left_group_count = int(left_c.thumb.touched) + int(left_c.index.touched) + int(left_c.middle.touched)
        right_group_count = int(right_c.thumb.touched) + int(right_c.index.touched) + int(right_c.middle.touched)
        both_multifinger = left_group_count >= 2 and right_group_count >= 2
        self._bilateral_multifinger_streak = self._bilateral_multifinger_streak + 1 if both_multifinger else 0
        self.max_bilateral_multifinger_streak = max(self.max_bilateral_multifinger_streak, self._bilateral_multifinger_streak)
        # Section 10 (Thumb Opposition + Claw-Style Bimanual Grasp
        # session): thumb participation tracked globally (every state),
        # same pattern as the finger streaks above -- "the thumb touched
        # at some point" is a materially different (much stronger) claim
        # than "some finger touched", and this project's own history
        # shows the two get silently conflated unless tracked separately
        # (see Section 15's grasp_expert.py comment on the right hand's
        # thumb never touching for 60 steps while index/middle flickered).
        both_thumb = left_c.thumb.touched and right_c.thumb.touched
        self._left_thumb_streak = self._left_thumb_streak + 1 if left_c.thumb.touched else 0
        self._right_thumb_streak = self._right_thumb_streak + 1 if right_c.thumb.touched else 0
        self._bilateral_thumb_streak = self._bilateral_thumb_streak + 1 if both_thumb else 0
        self.max_left_thumb_contact_streak = max(self.max_left_thumb_contact_streak, self._left_thumb_streak)
        self.max_right_thumb_contact_streak = max(self.max_right_thumb_contact_streak, self._right_thumb_streak)
        self.max_bilateral_thumb_contact_streak = max(self.max_bilateral_thumb_contact_streak, self._bilateral_thumb_streak)
        self._dual_streak = self._dual_streak + 1 if (left_touch and right_touch) else 0
        self._finger_streak = self._finger_streak + 1 if both_finger else 0
        self._palm_only_streak = self._palm_only_streak + 1 if palm_only else 0
        self.max_dual_contact_streak = max(self.max_dual_contact_streak, self._dual_streak)
        self.max_finger_contact_streak = max(self.max_finger_contact_streak, self._finger_streak)
        self.max_palm_only_streak = max(self.max_palm_only_streak, self._palm_only_streak)

        # Section 1: left-only / right-only / bilateral finger streaks kept
        # SEPARATE -- a "12-step finger contact streak" could previously
        # mean "one hand touched for 12 steps" or "both touched together
        # for 12 steps" and the diagnostic could not tell which.
        self._left_finger_streak = self._left_finger_streak + 1 if left_c.finger.touched else 0
        self._right_finger_streak = self._right_finger_streak + 1 if right_c.finger.touched else 0
        self._bilateral_finger_streak = self._bilateral_finger_streak + 1 if both_finger else 0
        self.max_left_finger_contact_streak = max(self.max_left_finger_contact_streak, self._left_finger_streak)
        self.max_right_finger_contact_streak = max(self.max_right_finger_contact_streak, self._right_finger_streak)
        self.max_bilateral_finger_contact_streak = max(self.max_bilateral_finger_contact_streak, self._bilateral_finger_streak)
        if self.left_first_finger_contact_step is None and left_c.finger.touched:
            self.left_first_finger_contact_step = self._global_step
        if self.right_first_finger_contact_step is None and right_c.finger.touched:
            self.right_first_finger_contact_step = self._global_step
        if self.bilateral_first_finger_contact_step is None and both_finger:
            self.bilateral_first_finger_contact_step = self._global_step

        if self.object_z_at_first_contact is None and (left_touch or right_touch):
            self.object_z_at_first_contact = obj[2]
        if self.state not in (GraspState.LIFT, GraspState.HOLD):
            self.maximum_object_z_before_lift = max(self.maximum_object_z_before_lift, obj[2])

        pre_contact_obj = self._obj_ref

        if self.state in (
            GraspState.STABLE_START,
            GraspState.NATURAL_ARM_LIFT,
            GraspState.FOREARM_LATERAL_APPROACH,
            GraspState.FOREARM_DESCEND,
            GraspState.WRIST_ALIGN,
            GraspState.FINGER_PRESHAPE,
            GraspState.FINGERTIP_PRECONTACT,
            GraspState.CONTACT_ACQUIRE,
        ):
            disp = float(np.linalg.norm(obj[:2] - self._obj_ref[:2]))
            if disp > cfg.object_displacement_limit:
                self.failure_reason = FailureReason.OBJECT_DISPLACED
                self.state = GraspState.FAILURE

        # Reset both trackers/joint-rate caps to their normal values by
        # default -- CONTACT_ACQUIRE below overrides them per-hand based on
        # measured fingertip-object distance (Section 3 deceleration).
        # Without this reset, a reduction from an earlier CONTACT_ACQUIRE
        # visit would silently persist into unrelated states.
        self.left_tracker.max_step = cfg.max_target_step
        self.right_tracker.max_step = cfg.max_target_step
        self._left_ik_max_dq = cfg.ik_max_dq
        self._right_ik_max_dq = cfg.ik_max_dq

        # ---- state goal-setting (only changes the TRACKER's goal, never
        # snaps the commanded target -- see _TargetTracker) ----
        if self.state == GraspState.STABLE_START:
            left_goal, right_goal = self.left_tracker.commanded, self.right_tracker.commanded  # hold still
            if self.state_step >= cfg.min_state_steps:
                self._left_raise_start = self.left_tracker.commanded.copy()
                self._right_raise_start = self.right_tracker.commanded.copy()
                self._transition(GraspState.NATURAL_ARM_LIFT)

        elif self.state == GraspState.NATURAL_ARM_LIFT:
            # Step 1 (Natural Arm Reach session): shoulder+elbow lift the
            # arm straight up (x/y frozen at wherever the arm started) to
            # a safe height above the object -- wrist is suppressed
            # (Section 5 "natural_lift" role: expensive + pulled hard
            # toward the stand pose's neutral orientation) and orientation
            # is NOT required to converge here (_coupled_role_profile),
            # so the arm rises without the wrist snapping toward the
            # object early.
            safe_z = pre_contact_obj[2] + self.approach_height
            left_target = np.array([self._left_raise_start[0], self._left_raise_start[1], safe_z])
            right_target = np.array([self._right_raise_start[0], self._right_raise_start[1], safe_z])
            if not self._in_coupled_phase:
                weight, rest_gain, rest_q, ori_w, req_ori, resolve_damp = self._coupled_role_profile("natural_lift")
                self._start_coupled_transit(
                    left_target, right_target, joint_weight=weight, rest_gain=rest_gain, rest_q=rest_q,
                    ori_task_weight=ori_w, require_orientation=req_ori, resolve_damping=resolve_damp,
                )
                self._in_coupled_phase = True
            else:
                self._coupled_maybe_resolve()
            if left_touch or right_touch:
                self.failure_reason = FailureReason.PREMATURE_CONTACT
                self.state = GraspState.FAILURE
            else:
                if self.state_step >= cfg.min_state_steps and self._coupled_converged(left_target, right_target, require_orientation=False):
                    self._coupled_stable_streak += 1
                else:
                    self._coupled_stable_streak = 0
                if self._coupled_stable_streak >= cfg.coupled_stable_steps:
                    left_actual, _ = self.left_ctrl.current_pose(self.env.data)
                    right_actual, _ = self.right_ctrl.current_pose(self.env.data)
                    self.left_tracker.snap(left_actual)
                    self.right_tracker.snap(right_actual)
                    self._transition(GraspState.FOREARM_LATERAL_APPROACH)

        elif self.state == GraspState.FOREARM_LATERAL_APPROACH:
            # Step 2: shoulder+elbow move the (still wrist-neutral-ish)
            # hands laterally to the pre-grasp waypoint beside the object,
            # STILL at the safe approach_height -- no vertical descent
            # yet (Section 3 fingertip grasp frame -- this target is the
            # INTENDED FINGERTIP contact point at approach_height,
            # converted to a palm target by _grip_targets). Orientation
            # still not required (Section 5 "forearm_lateral" role) --
            # WRIST_ALIGN does that work later, once FOREARM_DESCEND has
            # already brought the arm down to the final grasp height.
            left_lat_goal, right_lat_goal = self._grip_targets(pre_contact_obj, self.approach_height)
            if not self._in_coupled_phase:
                weight, rest_gain, rest_q, ori_w, req_ori, resolve_damp = self._coupled_role_profile("forearm_lateral")
                self._start_coupled_transit(
                    left_lat_goal, right_lat_goal, joint_weight=weight, rest_gain=rest_gain, rest_q=rest_q,
                    ori_task_weight=ori_w, require_orientation=req_ori, resolve_damping=resolve_damp,
                )
                self._in_coupled_phase = True
            else:
                self._coupled_maybe_resolve()
            if left_touch or right_touch:
                self.failure_reason = FailureReason.PREMATURE_CONTACT
                self.state = GraspState.FAILURE
            else:
                if self.state_step >= cfg.min_state_steps and self._coupled_converged(left_lat_goal, right_lat_goal, require_orientation=False):
                    self._coupled_stable_streak += 1
                else:
                    self._coupled_stable_streak = 0
                if self._coupled_stable_streak >= cfg.coupled_stable_steps:
                    left_actual, _ = self.left_ctrl.current_pose(self.env.data)
                    right_actual, _ = self.right_ctrl.current_pose(self.env.data)
                    self.left_tracker.snap(left_actual)
                    self.right_tracker.snap(right_actual)
                    self._transition(GraspState.THUMB_CLEARANCE_PRESHAPE)

        elif self.state == GraspState.THUMB_CLEARANCE_PRESHAPE:
            # New state, moved EARLY (Phase 4 -- Collision-Free Thumb
            # Preshape + Early Tripod Closure session, Section 3): thumb
            # abducts here, while the hand is STILL at approach_height --
            # well clear of the object vertically -- instead of after
            # WRIST_ALIGN/FINGER_PRESHAPE when the hand has already
            # descended next to it (the previous, now-removed THUMB_ABDUCT
            # placement, which needed a thumb_0 detour to avoid clipping
            # SIZE_12's object). Arm/wrist target is UNCHANGED from
            # FOREARM_LATERAL_APPROACH (same "forearm_lateral" role, same
            # approach_height) -- this state only drives the thumb.
            left_lat_goal, right_lat_goal = self._grip_targets(pre_contact_obj, self.approach_height)
            if not self._in_coupled_phase:
                weight, rest_gain, rest_q, ori_w, req_ori, resolve_damp = self._coupled_role_profile("forearm_lateral")
                self._start_coupled_transit(
                    left_lat_goal, right_lat_goal, joint_weight=weight, rest_gain=rest_gain, rest_q=rest_q,
                    ori_task_weight=ori_w, require_orientation=req_ori, resolve_damping=resolve_damp,
                )
                self._in_coupled_phase = True
            else:
                self._coupled_maybe_resolve()
            if left_touch or right_touch:
                self.failure_reason = FailureReason.PREMATURE_CONTACT
                self.state = GraspState.FAILURE
            else:
                ready = self._thumb_clearance_ready()
                self._thumb_clearance_streak = self._thumb_clearance_streak + 1 if ready else 0
                if self._thumb_clearance_streak >= cfg.coupled_stable_steps:
                    left_actual, _ = self.left_ctrl.current_pose(self.env.data)
                    right_actual, _ = self.right_ctrl.current_pose(self.env.data)
                    self.left_tracker.snap(left_actual)
                    self.right_tracker.snap(right_actual)
                    self._transition(GraspState.FOREARM_DESCEND)

        elif self.state == GraspState.FOREARM_DESCEND:
            # Step 3 (Forearm Descent Before Wrist Alignment session):
            # the big ~0.15-0.19m vertical drop from approach_height to
            # grasp_z_offset -- STILL at the SAME lateral (outside_offset)
            # clearance as FOREARM_LATERAL_APPROACH, only z changes.
            # Shoulder+elbow remain the PRIMARY DoF here (same
            # "forearm_lateral"-style weighting, reused as "forearm_
            # descend"), NOT the heavily posture-held role the old
            # (pre-this-session) FINGERTIP_APPROACH used for this same
            # descent -- direct measurement found wrist rotation alone
            # cannot cover a drop this large (bounded by the short
            # palm-offset lever arm), which is exactly why SIZE_6 used to
            # plateau at ~0.022m residual here. Orientation still not
            # required -- WRIST_ALIGN handles that next, at this SAME
            # position.
            left_descend_goal, right_descend_goal = self._grip_targets(pre_contact_obj, self.grasp_z_offset)
            if not self._in_coupled_phase:
                weight, rest_gain, rest_q, ori_w, req_ori, resolve_damp = self._coupled_role_profile("forearm_descend")
                self._start_coupled_transit(
                    left_descend_goal, right_descend_goal, joint_weight=weight, rest_gain=rest_gain, rest_q=rest_q,
                    ori_task_weight=ori_w, require_orientation=req_ori, resolve_damping=resolve_damp,
                )
                self._in_coupled_phase = True
            else:
                self._coupled_maybe_resolve()
            if left_touch or right_touch:
                self.failure_reason = FailureReason.PREMATURE_CONTACT
                self.state = GraspState.FAILURE
            else:
                if self.state_step >= cfg.min_state_steps and self._coupled_converged(left_descend_goal, right_descend_goal, require_orientation=False):
                    self._coupled_stable_streak += 1
                else:
                    self._coupled_stable_streak = 0
                if self._coupled_stable_streak >= cfg.coupled_stable_steps:
                    left_actual, _ = self.left_ctrl.current_pose(self.env.data)
                    right_actual, _ = self.right_ctrl.current_pose(self.env.data)
                    self.left_tracker.snap(left_actual)
                    self.right_tracker.snap(right_actual)
                    self._transition(GraspState.WRIST_ALIGN)
                    self._capture_posture_hold_ref()

        elif self.state == GraspState.WRIST_ALIGN:
            # Step 4: shoulder/elbow now posture-hold (Section 5
            # "wrist_align" role: expensive + pulled toward the pose
            # FOREARM_DESCEND ended at, captured in _posture_hold_ref --
            # NOT the stand pose) while wrist roll/pitch/yaw becomes the
            # cheap, primary DoF for the position+orientation convergence
            # this state actually requires (full side-pinch orientation,
            # fingertips aimed at the object surface). SAME Cartesian
            # target as FOREARM_DESCEND -- no more big moves left to make,
            # only the joint roles change.
            left_descend_goal, right_descend_goal = self._grip_targets(pre_contact_obj, self.grasp_z_offset)
            if not self._in_coupled_phase:
                weight, rest_gain, rest_q, ori_w, req_ori, resolve_damp = self._coupled_role_profile("wrist_align")
                self._start_coupled_transit(
                    left_descend_goal, right_descend_goal, joint_weight=weight, rest_gain=rest_gain, rest_q=rest_q,
                    ori_task_weight=ori_w, require_orientation=req_ori, resolve_damping=resolve_damp,
                )
                self._in_coupled_phase = True
            else:
                self._coupled_maybe_resolve()
            if left_touch or right_touch:
                self.failure_reason = FailureReason.PREMATURE_CONTACT
                self.state = GraspState.FAILURE
            else:
                if self.state_step >= cfg.min_state_steps and self._coupled_converged(left_descend_goal, right_descend_goal):
                    self._coupled_stable_streak += 1
                else:
                    self._coupled_stable_streak = 0
                if self._coupled_stable_streak >= cfg.coupled_stable_steps:
                    left_actual, _ = self.left_ctrl.current_pose(self.env.data)
                    right_actual, _ = self.right_ctrl.current_pose(self.env.data)
                    self.left_tracker.snap(left_actual)
                    self.right_tracker.snap(right_actual)
                    self._transition(GraspState.FINGER_PRESHAPE)

        elif self.state == GraspState.FINGER_PRESHAPE:
            # Step 5: arm/wrist pose held exactly where WRIST_ALIGN left
            # it (same target, same "wrist_align" role/posture-hold
            # reference) -- only the finger joints move, curling toward
            # preshape_synergy before anything approaches the object.
            #
            # CLAW PRESHAPE (Thumb Opposition + Claw-Style Bimanual Grasp
            # session): index/middle curl to preshape_synergy as before,
            # but THUMB (group index 0) stays PINNED at 0 (fully abducted)
            # -- it no longer ramps here at all. Two reasons: (1) this is
            # the correct claw behavior per spec (thumb stays open/out of
            # the way while index/middle preshape; it only closes later,
            # in THUMB_OPPOSE, once index/middle already have contact);
            # (2) a direct regression was measured letting thumb ramp
            # here too -- even though whole_body_config.py's thumb_1 fix
            # (see that file) only changes the ALREADY-CORRECT approach
            # target computation, physically moving thumb through an
            # intermediate pose during this state introduced a genuine
            # arm-convergence plateau in FINGERTIP_PRECONTACT for SIZE_12
            # that did not exist when thumb stayed at its already-tested
            # abducted (synergy=0) pose throughout. Direct trace found the
            # OLD (frozen-thumb) baseline was never actually relying on
            # full convergence there either -- it was exiting via an
            # incidental early thumb-object touch caused by the very
            # thumb_1 freeze bug this session fixes. Pinning thumb here
            # keeps its physical trajectory identical to what WRIST_ALIGN
            # (measured stable) already tolerates.
            self.left_group_synergy[1:] = np.minimum(cfg.preshape_synergy, self.left_group_synergy[1:] + 1.0 / cfg.finger_close_steps)
            self.right_group_synergy[1:] = np.minimum(cfg.preshape_synergy, self.right_group_synergy[1:] + 1.0 / cfg.finger_close_steps)
            self._sync_scalar_synergy()
            if left_touch or right_touch:
                self.failure_reason = FailureReason.PREMATURE_CONTACT
                self.state = GraspState.FAILURE
            elif (
                self.state_step >= cfg.min_state_steps
                and self.left_group_synergy[1:].min() >= cfg.preshape_synergy
                and self.right_group_synergy[1:].min() >= cfg.preshape_synergy
            ):
                left_actual, _ = self.left_ctrl.current_pose(self.env.data)
                right_actual, _ = self.right_ctrl.current_pose(self.env.data)
                self.left_tracker.snap(left_actual)
                self.right_tracker.snap(right_actual)
                self._transition(GraspState.FINGERTIP_PRECONTACT)

        elif self.state == GraspState.FINGERTIP_PRECONTACT:
            if self.state_step == 0:
                self._fingertip_precontact_best_err = np.inf
                self._fingertip_precontact_plateau_streak = 0
            # Step 6: NOT a repeat of the old FINGERTIP_APPROACH's big
            # vertical drop -- z stays at grasp_z_offset (unchanged from
            # WRIST_ALIGN/FINGER_PRESHAPE); only the LATERAL clearance
            # shrinks, from outside_offset down to precontact_half_width
            # (Section 2/4: a small, few-cm surface-normal approach).
            # Shoulder/elbow get a MODERATE (not near-total) weight
            # (Section 5 "fingertip_precontact" role) -- explicitly not a
            # wrist-only solve, since this move is small enough that
            # shoulder/elbow can safely help without re-introducing the
            # "ask restricted DoF for a large motion" bug this session
            # fixes. Contact category is distinguished here (Section 10):
            # a FINGER touching is the expected, allowed outcome of this
            # state (transitions straight to CONTACT_ACQUIRE) -- only a
            # palm/wrist touch is still PREMATURE_CONTACT.
            left_pre_goal, right_pre_goal = self._grip_targets(pre_contact_obj, self.grasp_z_offset, half_width=self.precontact_half_width)
            if not self._in_coupled_phase:
                weight, rest_gain, rest_q, ori_w, req_ori, resolve_damp = self._coupled_role_profile("fingertip_precontact")
                self._start_coupled_transit(
                    left_pre_goal, right_pre_goal, joint_weight=weight, rest_gain=rest_gain, rest_q=rest_q,
                    ori_task_weight=ori_w, require_orientation=req_ori, resolve_damping=resolve_damp,
                )
                self._in_coupled_phase = True
            else:
                self._coupled_maybe_resolve()
            if left_c.palm.touched or right_c.palm.touched:
                self.failure_reason = FailureReason.PREMATURE_CONTACT
                self.state = GraspState.FAILURE
            elif (left_c.index.touched or left_c.middle.touched) or (right_c.index.touched or right_c.middle.touched):
                # An index/middle finger reaching the object during this
                # small approach is the intended outcome, not a failure --
                # hand off to CONTACT_ACQUIRE's own per-hand substate
                # machine, which is what actually manages first-contact
                # force/holding. THUMB touching is deliberately EXCLUDED
                # from this check (Thumb Opposition + Claw-Style Bimanual
                # Grasp session): thumb is meant to stay abducted through
                # this whole phase (see THUMB_ABDUCT) and only closes in
                # THUMB_OPPOSE, after index/middle already hold contact --
                # a direct trace found thumb reaching the object FIRST
                # (before index/middle) and this branch then treating that
                # single point as "the hand made contact", locking
                # CONTACT_ACQUIRE's per-hand substate into FORCE_HOLD
                # against thumb alone while continuing to press inward,
                # spiking to 22.9N and displacing the object 3cm+ within
                # 15 steps -- a single point of contact is never a stable
                # hold regardless of which finger group it is.
                left_actual, _ = self.left_ctrl.current_pose(self.env.data)
                right_actual, _ = self.right_ctrl.current_pose(self.env.data)
                self.left_tracker.snap(left_actual)
                self.right_tracker.snap(right_actual)
                self.left_approach_offset = self.precontact_half_width
                self.right_approach_offset = self.precontact_half_width
                self._transition(GraspState.CONTACT_ACQUIRE)
            else:
                if self.state_step >= cfg.min_state_steps and self._coupled_converged(left_pre_goal, right_pre_goal):
                    self._coupled_stable_streak += 1
                else:
                    self._coupled_stable_streak = 0
                # Plateau hand-off (Thumb Opposition + Claw-Style Bimanual
                # Grasp session): direct measurement found this state has
                # a genuine, pre-existing ~2.4cm physical convergence
                # residual for SIZE_12 -- NOT introduced by this session's
                # thumb_1 fix (the coupled solve never touches thumb's
                # DOFs; confirmed directly that the arm's own position-
                # error trajectory is identical regardless of thumb's
                # config). It was previously always masked by the
                # thumb_1 freeze bug: thumb sat permanently at its
                # "close" value and incidentally touched the object
                # early, exiting via the branch above before this plateau
                # was ever reached. With thumb correctly abducted (no
                # longer touching early), the plateau is exposed. Since
                # CONTACT_ACQUIRE is explicitly designed to close the
                # REMAINING gap via real contact feedback (not open-loop
                # Cartesian targeting), a stalled-but-bounded residual is
                # hand-off, not a failure -- this does not touch
                # coupled_pos_tol/coupled_ori_tol_deg/coupled_stable_steps
                # (the strict convergence path above is unchanged and
                # still preferred whenever it's actually reachable).
                cur_err = max(
                    float(np.linalg.norm(left_pre_goal - self.left_ctrl.current_pose(self.env.data)[0])),
                    float(np.linalg.norm(right_pre_goal - self.right_ctrl.current_pose(self.env.data)[0])),
                )
                if self.state_step < cfg.min_state_steps or cur_err >= self._fingertip_precontact_best_err - 0.001:
                    self._fingertip_precontact_plateau_streak += 1
                else:
                    self._fingertip_precontact_plateau_streak = 0
                self._fingertip_precontact_best_err = min(self._fingertip_precontact_best_err, cur_err)
                plateaued = (
                    self._fingertip_precontact_plateau_streak >= 3 * cfg.coupled_stable_steps
                    and cur_err < 0.05
                )
                if self._coupled_stable_streak >= cfg.coupled_stable_steps or plateaued:
                    left_actual, _ = self.left_ctrl.current_pose(self.env.data)
                    right_actual, _ = self.right_ctrl.current_pose(self.env.data)
                    self.left_tracker.snap(left_actual)
                    self.right_tracker.snap(right_actual)
                    self.left_approach_offset = self.precontact_half_width
                    self.right_approach_offset = self.precontact_half_width
                    self._transition(GraspState.CONTACT_ACQUIRE)

        elif self.state == GraspState.CONTACT_ACQUIRE:
            self._update_z_sync()
            # INDEX/MIDDLE only (Thumb Opposition + Claw-Style Bimanual
            # Grasp session) -- this state's whole per-hand substate
            # machine (_contact_acquire_hand_step) is about acquiring the
            # index/middle contact surface; thumb stays abducted and
            # participates later, in THUMB_OPPOSE. See the matching
            # comment in FINGERTIP_PRECONTACT for the direct measurement
            # (thumb-only contact triggering a false FORCE_HOLD, force
            # spiking to 22.9N, 3cm+ object displacement) that made this
            # necessary.
            left_touch_finger = left_c.index.touched or left_c.middle.touched
            right_touch_finger = right_c.index.touched or right_c.middle.touched

            # Uses RAW force, not the EMA-filtered left_force/right_force --
            # a direct trace showed a single-point graze spike to 6.5N then
            # decay to 0.0N raw over 4 steps while the SLOW-lagging filtered
            # value was still reading >1.5N, so FORCE_HOLD kept issuing
            # "relieve" (retreat) commands for several steps AFTER the real
            # contact was already gone -- actively pushing the hand away
            # right when it needed to hold or firm up. This phase needs a
            # fast reaction to a fast single-step event; filtering that
            # away here was the bug.
            #
            # index/middle force ONLY here (not left_c.max_force, which
            # would include thumb) -- matches left_touch_finger/
            # right_touch_finger above. A premature thumb-only touch is
            # handled by the separate safety clamp right below instead of
            # by this substate machine (which is specifically about
            # index/middle acquisition).
            left_im_force = max(left_c.index.resultant, left_c.middle.resultant)
            right_im_force = max(right_c.index.resultant, right_c.middle.resultant)
            self.left_approach_offset, self.left_substate, self.left_reacquire_attempts, self._left_ca_loss_grace = self._contact_acquire_hand_step(
                left_touch_finger, left_im_force, self.left_approach_offset, self.left_substate, self.left_reacquire_attempts, self._left_ca_loss_grace
            )
            self.right_approach_offset, self.right_substate, self.right_reacquire_attempts, self._right_ca_loss_grace = self._contact_acquire_hand_step(
                right_touch_finger, right_im_force, self.right_approach_offset, self.right_substate, self.right_reacquire_attempts, self._right_ca_loss_grace
            )
            # Safety clamp (Thumb Opposition + Claw-Style Bimanual Grasp
            # session): thumb reaching the object BEFORE index/middle is
            # allowed to happen (its exact timing depends on geometry, not
            # something this phase forces), but the substate machine above
            # only reacts to index/middle force -- without this, a hand
            # whose thumb already touched would keep approaching blind to
            # that, pressing thumb alone from 0 to 22.9N and displacing
            # the object 3cm+ within 15 steps (measured directly). Stop
            # closing that hand's offset any further once thumb force
            # exceeds the same safe limit FINGER_CLOSE/FORCE_SETTLE use.
            if left_c.thumb.resultant > cfg.max_safe_grip_force:
                self.left_approach_offset = min(self.outside_offset, self.left_approach_offset + cfg.contact_acquire_relief_step)
            if right_c.thumb.resultant > cfg.max_safe_grip_force:
                self.right_approach_offset = min(self.outside_offset, self.right_approach_offset + cfg.contact_acquire_relief_step)

            if self.left_first_contact_pose is None and self.left_substate in (HandSubstate.FIRST_CONTACT, HandSubstate.FORCE_HOLD):
                self.left_first_contact_pose, _ = self.left_ctrl.current_pose(self.env.data)
                self.left_first_contact_synergy = self.left_desired_synergy
            if self.right_first_contact_pose is None and self.right_substate in (HandSubstate.FIRST_CONTACT, HandSubstate.FORCE_HOLD):
                self.right_first_contact_pose, _ = self.right_ctrl.current_pose(self.env.data)
                self.right_first_contact_synergy = self.right_desired_synergy
            if (
                self.contact_timing_gap is None
                and self.left_first_finger_contact_step is not None
                and self.right_first_finger_contact_step is not None
            ):
                self.contact_timing_gap = abs(self.left_first_finger_contact_step - self.right_first_finger_contact_step)

            # First contact was found (direct trace) to nudge the object
            # upward by ~1-2cm as a side effect regardless of which hand
            # makes it first -- a FIXED z target (tied to the episode-start
            # object height) left both hands' targets behind that rise, so
            # by the time the second hand also caught on, the object had
            # already climbed past where either fingertip could hold it,
            # and both lost contact together within ~5 steps. Track the
            # z-height once EITHER hand has live finger contact (still not
            # a full live-chase: frozen again the instant _both_ready_streak
            # completes into _contact_ref, and never used for the pop-vs-
            # lift metrics, which depend only on _obj_ref/state bookkeeping).
            if left_touch_finger or right_touch_finger:
                self._contact_ref = obj.copy()
            z_ref = self._contact_ref if self._contact_ref is not None else pre_contact_obj
            # Independent per-hand targets during this phase -- NOT the
            # shared grip_half_width used elsewhere (see HandSubstate
            # docstring for why sharing it here built up an asymmetric
            # squeeze while one hand waited for the other).
            left_fingertip_goal = z_ref + np.array([0.0, self.grip_center + self.left_approach_offset, self.grasp_z_offset + self._z_sync_bias])
            right_fingertip_goal = z_ref + np.array([0.0, self.grip_center - self.right_approach_offset, self.grasp_z_offset - self._z_sync_bias])
            left_goal = self._fingertip_target_to_palm_target(left_fingertip_goal, self.left_R, "left")
            right_goal = self._fingertip_target_to_palm_target(right_fingertip_goal, self.right_R, "right")
            self.left_tracker.set_goal(left_goal)
            self.right_tracker.set_goal(right_goal)

            # Distance-based approach deceleration (Section 3). First
            # attempt slowed the TARGET's own movement rate
            # (max_target_step) -- direct trace showed this had NO effect,
            # because the actual palm was lagging 17mm BEHIND an already-
            # static target carried over from APPROACH/FINGER_PRESHAPE's
            # loose 6cm convergence tolerance, and the IK's own gain was
            # "catching up" to that backlog at up to 0.46 m/s regardless of
            # how slowly the target itself crept. The real lever is the
            # JOINT velocity cap (ik_max_dq): at its default (0.12 rad),
            # arm_action_scale (0.05 rad/step) is the actual binding
            # constraint and this has no effect; dropping it BELOW 0.05
            # near the surface makes it bind and genuinely slows the
            # physical joint motion. Applied to whichever hand is actively
            # closing the gap (APPROACHING/REACQUIRE); a hand already
            # holding contact uses the normal rate.
            if self.left_substate in (HandSubstate.APPROACHING, HandSubstate.REACQUIRE):
                left_dist = self._min_fingertip_distance("left", obj)
                self._left_ik_max_dq = cfg.ik_max_dq * self._approach_speed_scale(left_dist)
            else:
                self._left_ik_max_dq = cfg.ik_max_dq
            if self.right_substate in (HandSubstate.APPROACHING, HandSubstate.REACQUIRE):
                right_dist = self._min_fingertip_distance("right", obj)
                self._right_ik_max_dq = cfg.ik_max_dq * self._approach_speed_scale(right_dist)
            else:
                self._right_ik_max_dq = cfg.ik_max_dq

            # Gate A's actual definition (Section 5) is bilateral FINGER
            # CONTACT sustained -- not "both hands simultaneously in the
            # FORCE_HOLD substate". Requiring the substate match was
            # stricter than that: a hand cycling FORCE_HOLD -> (momentary
            # miss) -> CONTACT_LOST -> REACQUIRE -> FORCE_HOLD keeps
            # touching almost the whole time (measured: 43-57 consecutive
            # steps of real bilateral finger contact) but resets this
            # streak on every substate blip, so CONTACT_ACQUIRE looped for
            # 400+ steps and eventually failed OBJECT_DISPLACED despite
            # Gate A's actual contact-duration criterion already being met.
            # Section 10/11 (Dynamic-Aware Multi-Start IK + Multi-Finger
            # Grasp/Lift/Hold session): a direct constant-hold experiment
            # (zero control action, no re-solve, from a live bilateral-
            # contact snapshot) measured the object displacing 5.2cm and
            # reaching 2.0 rad/s angular velocity PASSIVELY -- the "any
            # touch per hand" contact topology is not force-closure-
            # capable. Requiring >=2 finger groups HERE (before leaving
            # CONTACT_ACQUIRE) was tried and measured WORSE: this state's
            # only lever is the whole-hand approach offset at a FIXED
            # preshape curl, so once one group touches and FORCE_HOLD
            # freezes that offset, nothing in this state can recruit a
            # second group -- it just dwells until OBJECT_DISPLACED times
            # out (measured directly). Recruiting additional groups is
            # FINGER_CLOSE's job (it actively curls under-touching groups
            # via per-group synergy) -- so this gate stays "any finger
            # touching" to let the state machine reach FINGER_CLOSE, and
            # the >=2-groups requirement is enforced later, at FINGER_CLOSE
            # and FORCE_SETTLE's own readiness/LIFT gates instead.
            both_ready = left_touch_finger and right_touch_finger
            obj_speed = float(np.linalg.norm(env_obj_vel[:3]))
            obj_ang_speed = float(np.linalg.norm(env_obj_vel[3:]))
            stable_enough = obj_speed < cfg.max_object_speed_for_transition and obj_ang_speed < cfg.max_object_angular_speed_for_transition
            # The streak itself only requires both hands actually holding
            # contact -- gating the COUNTER on instantaneous object
            # stability (rather than just the final commit) meant the
            # streak could never accumulate through the object's own
            # first-contact settling bump (measured: object still moving
            # ~0.1m/s, 2x the stability threshold, for the entire ~5-step
            # window both hands were in FORCE_HOLD together), so bilateral
            # contact was lost before the gate ever had a chance to open.
            # Stability is still REQUIRED at the moment of the actual
            # transition below, just not accumulated into the streak.
            self._both_ready_streak = self._both_ready_streak + 1 if both_ready else 0

            both_terminally_lost = (
                self.left_substate == HandSubstate.CONTACT_LOST
                and self.left_reacquire_attempts >= cfg.contact_acquire_max_reacquire_attempts
            ) or (
                self.right_substate == HandSubstate.CONTACT_LOST
                and self.right_reacquire_attempts >= cfg.contact_acquire_max_reacquire_attempts
            )

            # THUMB_OPPOSE entry (Phase 4 -- Collision-Free Thumb Preshape
            # + Early Tripod Closure session, Section 8): a full
            # both_ready_stable_steps=30 hold BEFORE thumb ever engages is
            # the wrong requirement -- direct trace found the object
            # cannot hold bilateral index/middle contact for anywhere
            # near 30 consecutive steps WITHOUT thumb's opposing support
            # (contact flickers on a 6-9 step cycle). Requiring 30 stable
            # steps here demands the exact stability thumb is supposed to
            # PROVIDE before ever letting thumb start providing it. Switched
            # to a short debounce (thumb_oppose_entry_debounce_steps, 3-5
            # steps -- just enough to reject a single-frame graze, not a
            # sustained-hold requirement) plus two explicit safety checks
            # at the moment of transition: impact force still in the safe
            # band, and the object still within its displacement budget.
            # both_ready_stable_steps/stable_enough (the object velocity
            # check) are NOT used for this gate anymore -- they were
            # tuned for the old, much stricter 30-step requirement.
            obj_disp_from_ref = float(np.linalg.norm(obj[:2] - self._obj_ref[:2])) if self._obj_ref is not None else 0.0
            impact_force_safe = left_c.max_force < cfg.max_safe_grip_force and right_c.max_force < cfg.max_safe_grip_force
            within_displacement_budget = obj_disp_from_ref < cfg.object_displacement_limit
            if (
                self._both_ready_streak >= cfg.thumb_oppose_entry_debounce_steps
                and impact_force_safe and within_displacement_budget
            ):
                self.object_z_at_dual_contact = obj[2]
                self.left_substate = HandSubstate.READY
                self.right_substate = HandSubstate.READY
                # Freeze x/y/z as of THIS moment (not episode-start) for
                # FINGER_CLOSE/FORCE_SETTLE targeting -- using the
                # episode-start reference caused the finger height target
                # to drift out of alignment once the object was disturbed
                # during approach/contact-acquire.
                self._contact_ref = obj.copy()
                # Switch to the symmetric grip_center/grip_half_width
                # coordinate (Section 5) for FINGER_CLOSE/FORCE_SETTLE,
                # continuous with wherever the two independent offsets
                # ended up.
                self.grip_half_width = 0.5 * (self.left_approach_offset + self.right_approach_offset)
                self._transition(GraspState.THUMB_OPPOSE)
            elif both_terminally_lost or (
                self.left_approach_offset <= self.grip_half_width_min
                and self.right_approach_offset <= self.grip_half_width_min
                and not (left_touch_finger or right_touch_finger)
                and self.state_step > cfg.max_steps_per_state // 2
            ):
                self.failure_reason = (
                    FailureReason.SINGLE_HAND_CONTACT_ONLY
                    if (self.left_substate != HandSubstate.APPROACHING or self.right_substate != HandSubstate.APPROACHING)
                    else FailureReason.CONTACT_LOST
                )
                self.state = GraspState.FAILURE

        elif self.state == GraspState.THUMB_OPPOSE:
            # New state (Thumb Opposition + Claw-Style Bimanual Grasp
            # session, Section 5): index/middle already have contact from
            # CONTACT_ACQUIRE -- ONLY the thumb group actively closes
            # here, using its own per-group force regulation, while
            # index/middle get gentle force-holding regulation (same
            # np.where pattern FINGER_CLOSE/FORCE_SETTLE already use) so
            # they don't drift/over-squeeze while thumb catches up.
            self._update_z_sync()
            if left_c.finger.touched and right_c.finger.touched:
                self._contact_ref = obj.copy()
            left_goal, right_goal = self._grip_targets(self._contact_ref, self.grasp_z_offset)
            self.left_tracker.set_goal(left_goal)
            self.right_tracker.set_goal(right_goal)

            step_size = 1.0 / cfg.finger_close_steps
            over, under = cfg.max_safe_grip_force, cfg.finger_force_limit
            # thumb (group 0): actively ramp closed unless already over force
            for side_syn, side_force in (
                (self.left_group_synergy, self.left_group_force_raw),
                (self.right_group_synergy, self.right_group_force_raw),
            ):
                if side_force[0] > over:
                    side_syn[0] = max(0.0, side_syn[0] - step_size)
                elif side_force[0] < under:
                    side_syn[0] = min(1.0, side_syn[0] + step_size)
            # index/middle (groups 1,2): gentle hold, same regulation FINGER_CLOSE uses
            self.left_group_synergy[1:] = np.where(
                self.left_group_force_raw[1:] > over, np.maximum(0.0, self.left_group_synergy[1:] - step_size),
                np.where(self.left_group_force_raw[1:] < under, np.minimum(1.0, self.left_group_synergy[1:] + step_size), self.left_group_synergy[1:]),
            )
            self.right_group_synergy[1:] = np.where(
                self.right_group_force_raw[1:] > over, np.maximum(0.0, self.right_group_synergy[1:] - step_size),
                np.where(self.right_group_force_raw[1:] < under, np.minimum(1.0, self.right_group_synergy[1:] + step_size), self.right_group_synergy[1:]),
            )
            self._sync_scalar_synergy()

            left_thumb_ready = left_c.thumb.touched and self.left_group_force_raw[0] < over
            right_thumb_ready = right_c.thumb.touched and self.right_group_force_raw[0] < over

            if not (left_touch and right_touch):
                self.failure_reason = FailureReason.CONTACT_LOST
                self.state = GraspState.FAILURE
            elif left_thumb_ready and right_thumb_ready:
                self._transition(GraspState.TRIPOD_SETTLE)
            elif (
                self.left_group_synergy[0] >= 1.0 and self.right_group_synergy[0] >= 1.0
                and self.state_step > cfg.max_steps_per_state // 2
            ):
                # Thumb fully closed (per-group synergy saturated) on
                # BOTH hands but still not registering contact on at
                # least one -- a genuine thumb-can't-reach failure, kept
                # distinct from the generic TIMEOUT/CONTACT_LOST reasons
                # so it is diagnosable (Section 2/18) rather than lumped
                # in with unrelated failure modes.
                self.failure_reason = FailureReason.THUMB_CONTACT_FAILED
                self.state = GraspState.FAILURE

        elif self.state == GraspState.TRIPOD_SETTLE:
            # New state (Thumb Opposition + Claw-Style Bimanual Grasp
            # session, Section 5/12): verifies a genuine tripod grip
            # (thumb touched AND at least one of index/middle touched,
            # sustained) before ever entering FORCE_SETTLE/LIFT -- Gate
            # A's literal requirement, not just "some finger touching".
            self._update_z_sync()
            if left_c.finger.touched and right_c.finger.touched:
                self._contact_ref = obj.copy()
            left_goal, right_goal = self._grip_targets(self._contact_ref, self.grasp_z_offset)
            self.left_tracker.set_goal(left_goal)
            self.right_tracker.set_goal(right_goal)

            fine_step = 0.5 / cfg.finger_close_steps
            over, under = cfg.max_safe_grip_force, cfg.target_grip_force
            self.left_group_synergy = np.where(
                self.left_group_force_raw > over, np.maximum(0.0, self.left_group_synergy - fine_step),
                np.where(self.left_group_force_raw < under, np.minimum(1.0, self.left_group_synergy + fine_step), self.left_group_synergy),
            )
            self.right_group_synergy = np.where(
                self.right_group_force_raw > over, np.maximum(0.0, self.right_group_synergy - fine_step),
                np.where(self.right_group_force_raw < under, np.minimum(1.0, self.right_group_synergy + fine_step), self.right_group_synergy),
            )
            self._sync_scalar_synergy()

            left_score, left_opp_finger, left_sep = self._opposing_normal_score(left_c)
            right_score, right_opp_finger, right_sep = self._opposing_normal_score(right_c)
            self.left_opposing_normal_score = left_score
            self.right_opposing_normal_score = right_score
            self.left_thumb_contact_separation = left_sep
            self.right_thumb_contact_separation = right_sep

            # Tripod = thumb touched + (index or middle) touched + those
            # two contacts genuinely opposing (score > 0: normals point at
            # least partly toward each other, not both from the same
            # side -- see _opposing_normal_score docstring for the "같은
            # 방향에서 물체를 누름" failure mode this rules out).
            left_tripod = left_c.thumb.touched and left_opp_finger is not None and left_score > 0.0
            right_tripod = right_c.thumb.touched and right_opp_finger is not None and right_score > 0.0
            both_tripod = left_tripod and right_tripod

            self._left_tripod_streak = self._left_tripod_streak + 1 if left_tripod else 0
            self._right_tripod_streak = self._right_tripod_streak + 1 if right_tripod else 0
            self._bilateral_tripod_streak = self._bilateral_tripod_streak + 1 if both_tripod else 0
            self.max_left_tripod_contact_streak = max(self.max_left_tripod_contact_streak, self._left_tripod_streak)
            self.max_right_tripod_contact_streak = max(self.max_right_tripod_contact_streak, self._right_tripod_streak)
            self.max_bilateral_tripod_streak = max(self.max_bilateral_tripod_streak, self._bilateral_tripod_streak)

            self.settle_stable_streak = self.settle_stable_streak + 1 if both_tripod else 0
            if not (left_touch and right_touch):
                self.failure_reason = FailureReason.CONTACT_LOST
                self.state = GraspState.FAILURE
            elif self.settle_stable_streak >= cfg.tripod_settle_stable_steps:
                self.settle_stable_streak = 0
                self._transition(GraspState.FORCE_SETTLE)

        elif self.state == GraspState.FINGER_CLOSE:
            self._update_z_sync()
            # The closing motion was found (via direct trace) to launch the
            # object upward as a side effect -- up to +0.13 m/s, peaking
            # ~0.013m higher -- before it falls back through the
            # fingers' STATIONARY target height and contact is lost. A
            # real hand grasping something that's rising while still
            # verifiably in contact would move with it, not hold a fixed
            # point in space; re-anchoring the (still-frozen-at-the-instant,
            # not continuously live) contact reference while BOTH hands
            # remain in finger contact lets the grip track the object it is
            # actually holding, rather than watching it fly past a target
            # that never moved. This does not weaken the pop-vs-lift
            # distinction: that still solely depends on _obj_ref/state
            # transition bookkeeping recorded elsewhere, not on where this
            # tracker's target currently sits.
            if left_c.finger.touched and right_c.finger.touched:
                self._contact_ref = obj.copy()

            # Per-GROUP force regulation (Dynamic-Aware Multi-Start IK +
            # Multi-Finger Grasp/Lift/Hold session) -- direct measurement
            # found the OLD single-scalar-per-hand version could not tell
            # "thumb is over-forced" from "index is over-forced", so a
            # spike from ONE group (e.g. index/middle squeezing while
            # thumb barely touches) relieved the WHOLE hand's curl,
            # including the group that still needed to close -- measured
            # directly leading to the right hand's thumb never making
            # contact at all through a 60-step window before the object
            # rotated/slipped out. Each group now reacts only to its OWN
            # raw contact force.
            step_size = 1.0 / cfg.finger_close_steps
            over = cfg.max_safe_grip_force
            under = cfg.finger_force_limit
            self.left_group_synergy = np.where(
                self.left_group_force_raw > over, np.maximum(0.0, self.left_group_synergy - step_size),
                np.where(self.left_group_force_raw < under, np.minimum(1.0, self.left_group_synergy + step_size), self.left_group_synergy),
            )
            self.right_group_synergy = np.where(
                self.right_group_force_raw > over, np.maximum(0.0, self.right_group_synergy - step_size),
                np.where(self.right_group_force_raw < under, np.minimum(1.0, self.right_group_synergy + step_size), self.right_group_synergy),
            )
            self._sync_scalar_synergy()
            # grip_half_width widening (backing the WHOLE hand away, a
            # position-level response) still keys off the hand's
            # aggregate EMA force -- unchanged from the shared-scalar
            # design, since retreating position is not something a single
            # finger group can do independently.
            if left_force > cfg.max_safe_grip_force:
                self.grip_half_width = min(self.outside_offset, self.grip_half_width + min(0.02, 0.002 * (left_force - cfg.max_safe_grip_force)))
            if right_force > cfg.max_safe_grip_force:
                self.grip_half_width = min(self.outside_offset, self.grip_half_width + min(0.02, 0.002 * (right_force - cfg.max_safe_grip_force)))

            # Left/right force balancing (Section 5): a hand under
            # noticeably more force than the other is squeezing the object
            # off-center, which is what was tipping/rotating it out of the
            # other hand's grip. A small grip_center nudge toward the
            # LOWER-force side gives that side a touch more room. Deadband
            # avoids reacting to force-sensor noise; anti-windup caps the
            # accumulated correction so a persistent asymmetry cannot walk
            # grip_center arbitrarily far off-center.
            force_diff = left_force - right_force
            if abs(force_diff) > cfg.force_deadband:
                self.grip_center = float(np.clip(
                    self.grip_center + 0.3 * cfg.settle_pressure_step * np.sign(right_force - left_force),
                    -cfg.grip_center_max_correction, cfg.grip_center_max_correction,
                ))

            # The tracker's own max_target_step rate limit (see
            # _TargetTracker) already bounds how fast this can physically
            # move -- an earlier attempt to additionally *snap* the target
            # straight to the widened goal on a high-force reading
            # overshot and released contact entirely (widened from 0.106 to
            # 0.223 in ~5 steps, well past the object's 0.06 half-size).
            # Ordinary rate-limited tracking is both smooth and, per
            # max_target_step, already about as fast a retreat as the arm
            # can safely make.
            left_goal, right_goal = self._grip_targets(self._contact_ref, self.grasp_z_offset)
            self.left_tracker.set_goal(left_goal)
            self.right_tracker.set_goal(right_goal)

            # Section 10: "ready" requires a genuine multi-point grip
            # (>=2 of thumb/index/middle touching), not just enough mean
            # curl -- a hand can reach desired_synergy>0.45 through one
            # dominant group while the others still float free. FINGER_CLOSE
            # keeps curling non-touching groups every step (see the
            # per-group np.where above), so this condition can still be
            # met here -- it isn't a deadlock the way gating CONTACT_ACQUIRE
            # on the same requirement was (that state has no closing lever).
            left_ready = left_group_count >= 2 and self.left_desired_synergy > 0.45 and left_force < cfg.max_safe_grip_force
            right_ready = right_group_count >= 2 and self.right_desired_synergy > 0.45 and right_force < cfg.max_safe_grip_force
            if left_touch and right_touch and left_ready and right_ready:
                self.settle_stable_streak += 1
            else:
                self.settle_stable_streak = 0
            if left_touch and right_touch:
                self._contact_loss_grace = 0
            else:
                # A single-frame contact-detection dropout during active
                # squeezing (e.g. a momentary micro-slip as force
                # redistributes) does not by itself mean the grasp has
                # actually failed -- only a SUSTAINED loss does. Grace
                # period matches the object's own settle timescale, not an
                # arbitrarily long tolerance.
                self._contact_loss_grace += 1
            if self._contact_loss_grace > 5:
                self.failure_reason = FailureReason.CONTACT_LOST
                self.state = GraspState.FAILURE
            elif self.settle_stable_streak >= 10:
                self.settle_stable_streak = 0
                self._transition(GraspState.FORCE_SETTLE)

        elif self.state == GraspState.FORCE_SETTLE:
            self._update_z_sync()
            if left_c.finger.touched and right_c.finger.touched:
                self._contact_ref = obj.copy()

            step = cfg.settle_pressure_step
            # FINGER_CLOSE actively increases synergy every step while
            # force stays low -- freezing it the instant this state took
            # over (the original design) removed the very thing that had
            # been counteracting gravity/settling, and contact slipped
            # away within ~2 steps every single trial regardless of any
            # grip_half_width tuning. Keep a small continued synergy trim
            # here too, gentler than FINGER_CLOSE's, instead of a hard
            # freeze.
            # Per-GROUP fine trim -- same rationale as FINGER_CLOSE's
            # per-group regulation, gentler magnitude.
            fine_step = 0.5 / cfg.finger_close_steps
            self.left_group_synergy = np.where(
                self.left_group_force_raw > cfg.max_safe_grip_force, np.maximum(0.0, self.left_group_synergy - fine_step),
                np.where(self.left_group_force_raw < cfg.target_grip_force, np.minimum(1.0, self.left_group_synergy + fine_step), self.left_group_synergy),
            )
            self.right_group_synergy = np.where(
                self.right_group_force_raw > cfg.max_safe_grip_force, np.maximum(0.0, self.right_group_synergy - fine_step),
                np.where(self.right_group_force_raw < cfg.target_grip_force, np.minimum(1.0, self.right_group_synergy + fine_step), self.right_group_synergy),
            )
            self._sync_scalar_synergy()

            if left_force > cfg.max_safe_grip_force or right_force > cfg.max_safe_grip_force:
                widen = max(step, min(0.02, 0.002 * (max(left_force, right_force) - cfg.max_safe_grip_force)))
                self.grip_half_width = min(self.outside_offset, self.grip_half_width + widen)
            elif left_force < cfg.target_grip_force and right_force < cfg.target_grip_force:
                self.grip_half_width = max(self.grip_half_width_min, self.grip_half_width - step)
            # left/right force asymmetry -> fine grip_center correction
            # (deadband + anti-windup, see FINGER_CLOSE's identical comment)
            if abs(left_force - right_force) > cfg.force_deadband:
                self.grip_center = float(np.clip(
                    self.grip_center + 0.5 * step * np.sign(right_force - left_force),
                    -cfg.grip_center_max_correction, cfg.grip_center_max_correction,
                ))

            left_goal, right_goal = self._grip_targets(self._contact_ref, self.grasp_z_offset)
            self.left_tracker.set_goal(left_goal)
            self.right_tracker.set_goal(right_goal)

            in_band = left_force < cfg.max_safe_grip_force and right_force < cfg.max_safe_grip_force and left_touch and right_touch
            # Section 10: settling must hold a genuine multi-point grip
            # (>=2 groups per hand), matching Gate B's "no collapse to
            # thumb-only" requirement -- any-finger alone already proved
            # unstable under passive holding (constant-hold experiment).
            finger_ok = left_group_count >= 2 and right_group_count >= 2
            self.settle_stable_streak = self.settle_stable_streak + 1 if (in_band and finger_ok) else 0
            if left_touch and right_touch:
                self._contact_loss_grace = 0
            else:
                self._contact_loss_grace += 1
            if self._contact_loss_grace > 5:
                self.failure_reason = FailureReason.CONTACT_LOST
                self.state = GraspState.FAILURE
            elif self.settle_stable_streak >= cfg.settle_stable_steps:
                # Section 7 LIFT-entry gate: bilateral finger contact must
                # have been sustained at least finger_contact_hold_steps
                # (30) and the object must not currently be moving/
                # rotating -- entering LIFT while the object is still
                # bouncing from the squeeze is exactly how a pop got
                # miscounted as a lift in earlier sessions.
                obj_speed = float(np.linalg.norm(env_obj_vel[:3]))
                obj_ang_speed = float(np.linalg.norm(env_obj_vel[3:]))
                # Gate A's literal criterion is a bilateral MULTI-finger
                # streak (>=2 groups per hand), not just any-finger contact
                # (see Section 10/11 constant-hold finding above) -- using
                # the looser _bilateral_finger_streak here let LIFT trigger
                # off a topology already shown to be passively unstable.
                bilateral_ok = self._bilateral_multifinger_streak >= cfg.finger_contact_hold_steps
                object_stable = obj_speed < cfg.max_object_speed_for_transition and obj_ang_speed < cfg.max_object_angular_speed_for_transition
                if bilateral_ok and object_stable:
                    self.lift_left_start_pos = self.left_tracker.commanded.copy()
                    self.lift_right_start_pos = self.right_tracker.commanded.copy()
                    self.object_z_at_lift_start = obj[2]
                    self.settle_stable_streak = 0
                    self._transition(GraspState.LIFT)
                # else: keep waiting in FORCE_SETTLE -- not yet a failure,
                # just not ready (settle_stable_streak is NOT reset here so
                # a brief object jiggle doesn't force a full new 20-step wait).

        elif self.state == GraspState.LIFT:
            if max(left_force, right_force) < cfg.lift_force_limit:
                self.lift_target_z = min(cfg.lift_height, self.lift_target_z + cfg.lift_step)
            left_goal = self.lift_left_start_pos + np.array([0.0, 0.0, self.lift_target_z])
            right_goal = self.lift_right_start_pos + np.array([0.0, 0.0, self.lift_target_z])
            self.left_tracker.set_goal(left_goal)
            self.right_tracker.set_goal(right_goal)
            self.maximum_object_z_during_lift = max(self.maximum_object_z_during_lift, obj[2])
            if not (left_c.finger.touched and right_c.finger.touched):
                self.failure_reason = FailureReason.DROP
                self.state = GraspState.FAILURE
            elif self.lift_target_z >= cfg.lift_height:
                self.hold_left_target = left_goal.copy()
                self.hold_right_target = right_goal.copy()
                self._transition(GraspState.HOLD)

        elif self.state == GraspState.HOLD:
            self.left_tracker.set_goal(self.hold_left_target)
            self.right_tracker.set_goal(self.hold_right_target)
            self.maximum_object_z_during_lift = max(self.maximum_object_z_during_lift, obj[2])
            height_above_initial = obj[2] - self._obj_ref[2]
            if not (left_c.finger.touched and right_c.finger.touched):
                self.failure_reason = FailureReason.DROP
                self.state = GraspState.FAILURE
            elif height_above_initial < 0.02:
                self.failure_reason = FailureReason.LIFT_FAILED
                self.state = GraspState.FAILURE
            elif self.state_step >= cfg.hold_steps:
                self.state = GraspState.SUCCESS

        else:
            self.left_tracker.set_goal(self.left_tracker.commanded)
            self.right_tracker.set_goal(self.right_tracker.commanded)

        if self._in_coupled_phase:
            # PRE_GRASP/APPROACH's coupled-IK precision phase: the action
            # comes directly from the min-jerk joint trajectory tracker,
            # bypassing _apply's per-arm Cartesian IK entirely (see
            # _start_coupled_transit/_step_coupled_action). The Cartesian
            # trackers are intentionally left un-stepped here -- they get
            # snapped to the real converged palm pose at the moment this
            # phase exits, so downstream states see a continuous target.
            a = self._step_coupled_action(self.left_group_synergy, self.right_group_synergy)
        else:
            left_target = self.left_tracker.step()
            right_target = self.right_tracker.step()
            left_hand = self.left_group_synergy
            right_hand = self.right_group_synergy
            # FINGER_PRESHAPE added to this list (was CONTACT_ACQUIRE onward
            # only) -- direct measurement found the palm velocity oscillating
            # and DIVERGING (0.10 -> 0.67 m/s, still rising) during
            # FINGER_PRESHAPE despite its target being completely static; a
            # lower ik_gain fixes this (see _apply's ik_gain comment), and the
            # arm is already near the target by FINGER_PRESHAPE so null-space
            # reachability is no longer at stake the way it is during the
            # long PRE_GRASP/APPROACH transit.
            precision_states = (
                GraspState.FINGER_PRESHAPE, GraspState.CONTACT_ACQUIRE, GraspState.FINGER_CLOSE,
                GraspState.FORCE_SETTLE, GraspState.LIFT, GraspState.HOLD,
            )
            is_precision = self.state in precision_states
            null_gain = 0.0 if is_precision else None
            # A lower ik_gain during APPROACH too (matching precision_ik_gain)
            # was tried after root-causing genuine bimanual coupling (see
            # module-level PROJECT_CONTEXT.md notes and _apply's docstring):
            # despite the fixed-base robot having no shared arm DOF, the waist
            # (kp=500) still drifts up to ~2.5 degrees, oscillating, from BOTH
            # arms' reaction torque while straining toward a barely-reachable
            # target -- freezing one arm let the other converge to <0.1 degree
            # on a target that plateaus at 60+ degrees with both active. Lower
            # APPROACH gain was meant to reduce that reaction torque, but
            # measured WORSE on both sizes (12cm regressed 50->18-step
            # bilateral streak; 6cm still timed out) -- reverted. The coupling
            # diagnosis stands; this particular mitigation did not work.
            # This whole ik_gain issue is now moot for PRE_GRASP/APPROACH
            # themselves (they use the coupled solver above), but is kept
            # unchanged for every state from FINGER_PRESHAPE onward, which
            # this session does not touch.
            ik_gain = cfg.precision_ik_gain if is_precision else None
            # A hand is "holding" (IK deadband applies, see _apply) once it has
            # stopped actively closing the approach gap -- either the whole
            # state machine is past CONTACT_ACQUIRE, or (found necessary after
            # the A/B/C diagnosis snapshot turned out to still be inside
            # CONTACT_ACQUIRE: Gate A's 30+-step bilateral holds happen THERE,
            # not only in FINGER_CLOSE/FORCE_SETTLE) this hand's own substate
            # is FORCE_HOLD/READY rather than APPROACHING/REACQUIRE.
            holding_states = (GraspState.FINGER_CLOSE, GraspState.FORCE_SETTLE, GraspState.LIFT, GraspState.HOLD)
            if self.state in holding_states:
                left_is_holding, right_is_holding = True, True
            elif self.state == GraspState.CONTACT_ACQUIRE:
                left_is_holding = self.left_substate in (HandSubstate.FORCE_HOLD, HandSubstate.READY)
                right_is_holding = self.right_substate in (HandSubstate.FORCE_HOLD, HandSubstate.READY)
            else:
                left_is_holding, right_is_holding = False, False
            a, lc, rc = self._apply(
                left_target, right_target, left_hand, right_hand, null_gain=null_gain, ik_gain=ik_gain,
                left_max_dq=self._left_ik_max_dq, right_max_dq=self._right_ik_max_dq,
                left_is_holding=left_is_holding, right_is_holding=right_is_holding,
            )

        if self.state not in (GraspState.SUCCESS, GraspState.FAILURE):
            self._update_thumb1_override()
            qpos_before = self.env.data.qpos[self.env._arm_qpos_adr].copy()
            self.env.step(a)
            qpos_after = self.env.data.qpos[self.env._arm_qpos_adr].copy()
            delta = np.abs(qpos_after - qpos_before)
            self.total_joint_travel += float(delta.sum())
            step_vel = delta.max() / (self.env.config.frame_skip * 0.002)
            self.peak_joint_velocity = max(self.peak_joint_velocity, float(step_vel))

            if not np.isfinite(self.env.data.qpos).all() or not np.isfinite(self.env.data.qvel).all():
                self.failure_reason = FailureReason.NUMERICAL_ERROR
                self.state = GraspState.FAILURE

            self.state_step += 1
            if self.state_step > self.config.max_steps_per_state and self.state not in (GraspState.SUCCESS, GraspState.FAILURE):
                self.failure_reason = FailureReason.TIMEOUT
                self.state = GraspState.FAILURE

        if self.state == GraspState.FAILURE and (
            not self.state_transition_log or self.state_transition_log[-1][2] != "FAILURE"
        ):
            self.state_transition_log.append(
                (self._global_step, self.state_transition_log[-1][2] if self.state_transition_log else "?", "FAILURE", str(self.failure_reason))
            )

        self._global_step += 1
        return self._outcome()

    def _transition(self, new_state: GraspState, reason: str = "") -> None:
        self.state_transition_log.append((self._global_step, self.state.name, new_state.name, reason))
        self.state = new_state
        self.state_step = 0
        # Every state transition ends any in-progress coupled-IK
        # trajectory -- a stale target/trajectory from a PREVIOUS state
        # must never leak into the next state's action routing.
        # _posture_hold_ref is intentionally NOT cleared here: WRIST_ALIGN
        # captures it once (_capture_posture_hold_ref, called AFTER this
        # method so the freeze survives) and FINGER_PRESHAPE/
        # FINGERTIP_APPROACH deliberately reuse the SAME frozen reference
        # across their own transitions (Section 5 "wrist_align"/
        # "fingertip_approach" roles) -- clearing it here was an ordering
        # bug: capturing it BEFORE calling _transition() meant this line
        # immediately wiped it back to None, silently falling back to the
        # stand-pose rest reference (a very different, "wrong" posture)
        # with a strong rest_gain pull -- direct measurement traced this
        # to a monotonic divergence (0.005m/4.6deg residual growing to
        # 0.12m/31deg over 7 resolves) since the null-space term was
        # fighting the primary task the whole time, not holding a
        # meaningful reference.
        self._in_coupled_phase = False
        self._coupled_traj = None
        self._coupled_target_pos = None
        self._coupled_stable_streak = 0
        self._coupled_resolve_count = 0
        self._coupled_ik_profile = (None, None, None, 1.0, True, 0.7)

    def _outcome(self) -> GraspOutcome:
        obj = self._object_pos()
        env_group_syn = self.env._hand_group_synergy
        env_left_syn, env_right_syn = float(env_group_syn[0:3].mean()), float(env_group_syn[3:6].mean())
        lc, rc = self._last_left_contact, self._last_right_contact
        pre_lift_pop = max(0.0, self.maximum_object_z_before_lift - self._obj_ref[2]) if self._obj_ref is not None else 0.0
        controlled_gain = (
            self.maximum_object_z_during_lift - self.object_z_at_lift_start
            if self.object_z_at_lift_start is not None and np.isfinite(self.maximum_object_z_during_lift)
            else 0.0
        )
        return GraspOutcome(
            state=self.state,
            failure_reason=self.failure_reason,
            max_dual_contact_streak=self.max_dual_contact_streak,
            max_finger_contact_streak=self.max_finger_contact_streak,
            max_palm_only_streak=self.max_palm_only_streak,
            max_left_finger_contact_streak=self.max_left_finger_contact_streak,
            max_right_finger_contact_streak=self.max_right_finger_contact_streak,
            max_bilateral_finger_contact_streak=self.max_bilateral_finger_contact_streak,
            max_bilateral_multifinger_streak=self.max_bilateral_multifinger_streak,
            max_left_thumb_contact_streak=self.max_left_thumb_contact_streak,
            max_right_thumb_contact_streak=self.max_right_thumb_contact_streak,
            max_bilateral_thumb_contact_streak=self.max_bilateral_thumb_contact_streak,
            max_left_tripod_contact_streak=self.max_left_tripod_contact_streak,
            max_right_tripod_contact_streak=self.max_right_tripod_contact_streak,
            max_bilateral_tripod_streak=self.max_bilateral_tripod_streak,
            left_opposing_normal_score=self.left_opposing_normal_score,
            right_opposing_normal_score=self.right_opposing_normal_score,
            left_thumb_contact_separation=self.left_thumb_contact_separation,
            right_thumb_contact_separation=self.right_thumb_contact_separation,
            left_first_finger_contact_step=self.left_first_finger_contact_step,
            right_first_finger_contact_step=self.right_first_finger_contact_step,
            bilateral_first_finger_contact_step=self.bilateral_first_finger_contact_step,
            contact_timing_gap=self.contact_timing_gap,
            left_substate=self.left_substate.name,
            right_substate=self.right_substate.name,
            left_reacquire_attempts=self.left_reacquire_attempts,
            right_reacquire_attempts=self.right_reacquire_attempts,
            left_max_force_raw=self.left_max_force_raw,
            right_max_force_raw=self.right_max_force_raw,
            left_force_filtered=self.left_force_ema,
            right_force_filtered=self.right_force_ema,
            left_desired_synergy=self.left_desired_synergy,
            right_desired_synergy=self.right_desired_synergy,
            left_actual_synergy=env_left_syn,
            right_actual_synergy=env_right_syn,
            left_group_synergy=tuple(float(x) for x in self.left_group_synergy),
            right_group_synergy=tuple(float(x) for x in self.right_group_synergy),
            left_group_contact=(bool(lc.thumb.touched), bool(lc.index.touched), bool(lc.middle.touched)) if lc else (False, False, False),
            right_group_contact=(bool(rc.thumb.touched), bool(rc.index.touched), bool(rc.middle.touched)) if rc else (False, False, False),
            left_group_force_raw=tuple(float(x) for x in self.left_group_force_raw),
            right_group_force_raw=tuple(float(x) for x in self.right_group_force_raw),
            left_finger_contact=bool(lc.finger.touched) if lc else False,
            right_finger_contact=bool(rc.finger.touched) if rc else False,
            left_palm_contact=bool(lc.palm.touched) if lc else False,
            right_palm_contact=bool(rc.palm.touched) if rc else False,
            object_xy_displacement=float(np.linalg.norm(obj[:2] - self._obj_ref[:2])) if self._obj_ref is not None else 0.0,
            object_table_contact=self._object_table_contact(),
            initial_object_z=float(self._obj_ref[2]) if self._obj_ref is not None else 0.0,
            object_z_at_first_contact=self.object_z_at_first_contact,
            object_z_at_dual_contact=self.object_z_at_dual_contact,
            object_z_at_lift_start=self.object_z_at_lift_start,
            maximum_object_z_before_lift=(
                self.maximum_object_z_before_lift if np.isfinite(self.maximum_object_z_before_lift) else float(obj[2])
            ),
            maximum_object_z_during_lift=(
                self.maximum_object_z_during_lift if np.isfinite(self.maximum_object_z_during_lift) else float(obj[2])
            ),
            final_object_z=self.final_object_z,
            commanded_lift_target_z=self.lift_target_z,
            pre_lift_pop_height=pre_lift_pop,
            controlled_lift_gain=controlled_gain,
            final_height_above_initial=(self.final_object_z - self._obj_ref[2] if self._obj_ref is not None else 0.0),
            hold_steps_achieved=self.state_step if self.state == GraspState.HOLD else 0,
            total_joint_travel=self.total_joint_travel,
            peak_joint_velocity=self.peak_joint_velocity,
            peak_target_jump=self.peak_target_jump,
        )

    def run(self, max_total_steps: int = 6000) -> GraspOutcome:
        outcome = self._outcome()
        for _ in range(max_total_steps):
            outcome = self.step()
            if self.state in (GraspState.SUCCESS, GraspState.FAILURE):
                break
        return outcome
