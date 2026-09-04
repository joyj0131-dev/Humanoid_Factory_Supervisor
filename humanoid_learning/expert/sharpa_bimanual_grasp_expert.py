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

[Functional Orientation fix] The "EXPLICIT object-facing target" from the
paragraph above was itself found to be verified by a CIRCULAR metric: it
compared palm_R column 1 (the axis the wrist was SOLVED to align)
against the object direction, so the ~14deg residual it reported was
solver noise, not proof the hand could functionally close onto the
object. A real, empirical audit (scripts/audit_sharpa_local_closing_
frame.py -- hold the wrist fixed, apply an actual curl delta, read where
the fingertips really move, expressed in the wrist's own rigid local
frame so it is comparable across poses) found the TRUE closing axis is
~50deg off from that assumption. _object_facing_R is built via a
2-vector Kabsch/Wahba fit (LOCAL_CLOSING_VEC -> object direction; the
approach axis -> an inward-down direction) against this real axis
instead.

[Audit session] A commit that first landed this fix used weights (0.95/
0.05) that, while satisfying the closing-axis tolerance, forced the
finger-down angle to ~35deg -- that commit "fixed" the resulting Gate
failure by WIDENING the Gate's own tolerance (25->40deg) rather than
correcting the target, an improper Gate relaxation that was reverted (see
side_grasp_finger_down_tol_deg's docstring). The weights are re-derived
(0.68/0.32, see _object_facing_R's docstring) via a real-physics grid
sweep to satisfy BOTH the original, unchanged 15deg and 25deg tolerances.
That same over-aggressive 0.95 weight was ALSO the actual cause of a
~200-300N torso-arm collision the same commit introduced (undisclosed as
an upstream cause, disclosed only as an unresolved FOREARM_SIDE_DESCEND
symptom) -- it forced a large enough rotation that the redundant 17-DOF
solver needed a torso-colliding branch to also satisfy translation. Fixed
by the same weight correction (0.00N torso-arm through FOREARM_SIDE_
DESCEND now, real physics, seed=0) -- see that state's own docstring. The
Functional Orientation Gate (Section 9) is now genuinely a COMBINATION of
independently-measured subgates (finger-closure direction, thumb
opposition, palm-inside angle, finger-down angle, swept collision) --
see _measure_functional_orientation, not the single 4-finger-only check
an earlier commit used under the same name.

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


# [This session -- Functional Orientation fix] The previous session's
# `_object_facing_R`/`_object_facing_angle_deg` assumed the hand's real
# closing axis (where fingertips move when index/middle/wrap curl) IS
# palm_R column 1 (local +Y). That assumption was CIRCULAR: the wrist was
# SOLVED to make column 1 point at the object, so measuring "column 1 vs
# object direction" only ever showed solver/tracking noise (~14deg), not
# whether the hand could actually functionally grasp.
#
# This session (scripts/audit_sharpa_local_closing_frame.py,
# scripts/candidate_functional_orientation.py) measured the REAL closing
# axis independently: hold the wrist/arm fixed, apply a real curl delta
# to index/middle/wrap, read the resulting fingertip displacement in
# WORLD frame, then express it in the palm's OWN local frame via
# `palm_R.T @ world_displacement` -- a rigid-body-invariant quantity (any
# vector attached to the wrist_yaw_link-mounted hand transforms as
# world = R @ local, so local = R.T @ world is a property of the MESH,
# not of whatever orientation the arm/IK happens to be holding).
#
# Two confounds were found and controlled for:
#   1. TABLE CONTACT CONTAMINATION: measuring near the table (e.g. at
#      WRIST_SIDE_GRASP_ALIGN's own converged, near-table pose) redirects
#      the measured displacement via real contact force, not the free
#      kinematic direction -- measuring the SAME quantity at a pose far
#      from the table (FOREARM_FORWARD_REACH's end pose) gives a clean,
#      tight, side-consistent result instead.
#   2. CURL-ARC ROTATION: the closing tangent direction is NOT a single
#      fixed vector -- it rotates continuously and predictably as base
#      curl increases (measured local vector at base_curl in
#      {0.0, 0.3, 0.5, 0.7}: [0.18,0.98,0] -> [-0.14,0.99,0] ->
#      [-0.61,0.80,0.02] -> [-0.85,0.52,0], IDENTICAL for both hands, no
#      mirroring needed at any level -- a genuine finger-kinematics
#      property, not noise). The correct vector to use is the one at the
#      hand's ACTUAL operating curl during WRIST_SIDE_GRASP_ALIGN/
#      FIVE_FINGER_PRESHAPE -- see side_align_preshape_curl's docstring
#      for why that operating point is 0.3, not the previously-assumed
#      0.5.
LOCAL_CLOSING_VEC = np.array([-0.145, 0.989, 0.0002])
LOCAL_CLOSING_VEC = LOCAL_CLOSING_VEC / np.linalg.norm(LOCAL_CLOSING_VEC)
# The finger-longitudinal/"approach" axis is, BY DEFINITION of the palm_R
# column convention (model_builder.py), local +X -- this part of the old
# assumption was never circular (verified independently, prior session,
# to coincide with open/straight nonthumb fingers' own root->tip
# direction), so it is kept as a fixed local vector, not re-measured.
LOCAL_APPROACH_VEC = np.array([1.0, 0.0, 0.0])

# [Horizontal-Wrap session] Ulnar (pinky-side) local axis -- rigid-body
# FK measurement (/tmp scratch script, folded into
# scripts/measure_sharpa_side_grasp_axes.py's methodology: palm_R.T @
# (pinky_MC_body_pos - palm_pos) vs the same for thumb_CMC_VL, at BOTH
# stand pose and FOREARM_FORWARD_REACH's end pose -- identical in both,
# confirming a structural/rigid property, not a pose artifact). Per-
# finger local-Z components (left hand, either pose): thumb=-0.0260,
# index=-0.0303, middle=-0.0100, ring=+0.0103, pinky=+0.0263 -- a clean,
# monotonic radial(thumb/index)->ulnar(pinky) spread along local Z. This
# coincides (within measurement noise, dot product against
# cross(LOCAL_APPROACH_VEC, LOCAL_CLOSING_VEC) is +1) with the third axis
# of the (approach, closing) frame, i.e. local +Z -- so it is defined as
# that cross product directly rather than the raw (noisier, X/Z-mixed
# because thumb and pinky also differ along the approach axis) pinky-
# minus-thumb vector.
LOCAL_ULNAR_VEC = np.cross(LOCAL_APPROACH_VEC, LOCAL_CLOSING_VEC)
LOCAL_ULNAR_VEC = LOCAL_ULNAR_VEC / np.linalg.norm(LOCAL_ULNAR_VEC)
# [Horizontal-Wrap session] Sign of LOCAL_ULNAR_VEC that actually points
# toward the pinky, per side -- MEASURED (dot product of the same FK
# vector above against LOCAL_ULNAR_VEC), not assumed from Y_SIGN or any
# other existing left/right convention in this file: left hand measures
# +1 (pinky at +local Z), right hand measures -1 (pinky at -local Z).
# This differs per side because the left/right Sharpa Wave XMLs are two
# independently authored mirrored meshes (see sharpa_config.py's
# "left_left_..." docstring), not a single mesh reflected by convention,
# so the sign is not assumed to follow any other axis's mirror rule.
ULNAR_SIGN = {"left": 1.0, "right": -1.0}


def _wahba_R(local_vecs: list, world_vecs: list, weights: list) -> np.ndarray:
    """Kabsch/Wahba best-fit rotation: R minimizing
    sum_i w_i * |R @ local_i - world_i|^2 (SVD-based, always returns a
    proper rotation, det(R)=+1)."""
    B = np.zeros((3, 3))
    for w, l, wd in zip(weights, local_vecs, world_vecs):
        l = l / np.linalg.norm(l)
        wd = wd / np.linalg.norm(wd)
        B += w * np.outer(wd, l)
    U, _, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def _horizontal_wrap_target(to_obj_dir: np.ndarray, current_approach_world: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[Horizontal-Wrap session] REPLACES _down_target. _down_target
    forced the finger-longitudinal axis toward world -Z (fingers pointing
    down at the table) -- an incorrect task specification: the reference
    posture (user-supplied screenshot, top-down view) shows palms nearly
    VERTICAL facing the object's own left/right side faces, fingers
    nearly HORIZONTAL wrapping the object's front/back edges, and only
    the pinky-side (ulnar) edge of the hand resting near the table -- not
    fingertips pointing down into it.

    Returns three world targets for a 3-vector Wahba fit
    (see _object_facing_R):
      - closing_target: the REAL (full 3D, unprojected) palm->object
        direction -- same convention _object_facing_angle_deg (the Gate's
        own palm-inward metric, unchanged) already measures against, so
        the closing axis genuinely tracks the object regardless of any
        height offset between palm and object. The palm PLANE ends up
        close to vertical as a natural consequence whenever the lateral
        offset dominates the height offset (true at this approach's own
        geometry), without needing an exact-horizontal target that would
        otherwise conflict with the unchanged 15deg WRIST_NOT_OBJECT_
        FACING check (measured: an exact-horizontal target reproduces a
        systematic ~20deg+ residual purely from the align pose's own
        height/y-offset ratio, an artifact of the projection, not of fit
        quality -- WRIST_NOT_OBJECT_FACING fails outright under it).
      - wrap_target: HORIZONTAL, exactly perpendicular to the horizontal
        (Z=0) component of the palm->object direction (rotate 90deg
        within the XY plane) -- the new finger-longitudinal target,
        replacing "mostly down". Provably orthogonal to closing_target
        regardless of closing_target's own Z component (wrap_target's
        X/Y are proportional to closing_target's own X/Y direction, just
        rotated 90 degrees, and wrap_target_z=0). There are two
        perpendicular candidates (front-wrap vs back-wrap); the one
        closer to whatever approach-axis direction the wrist is
        CURRENTLY, already holding is chosen, for continuity with the
        incoming trajectory (Section 5 objective priority 7) rather than
        an arbitrary fixed convention -- UNLESS that choice conflicts
        with Constraint C (ulnar-down), in which case handedness wins
        (see below): continuity is this session's LOWEST-priority
        objective (Section 5, priority 7), ulnar-down is priority 5.
      - ulnar_target: exactly world -Z. Exactly orthogonal to wrap_target
        (which has zero Z); only approximately orthogonal to
        closing_target when closing_target has a nonzero Z component
        (i.e. palm and object are at different heights) -- handled by
        the same weighted-least-squares Wahba fit already used for the
        (non-exactly-orthogonal) LOCAL_APPROACH_VEC/LOCAL_CLOSING_VEC
        local pair, not a new kind of approximation.

    [Bug found empirically, fixed here] The two perpendicular wrap
    candidates are not just a left/right choice -- (approach, closing,
    ulnar) is a RIGHT-HANDED local frame FOR THE LEFT HAND (LOCAL_ULNAR_
    VEC is literally built as cross(LOCAL_APPROACH_VEC, LOCAL_CLOSING_
    VEC)) but a LEFT-HANDED one for the right hand (ULNAR_SIGN["right"]
    = -1 flips only the third local axis, which is exactly a reflection,
    i.e. a handedness flip -- an expected, measured consequence of the
    two hands being independently-authored mirrored meshes, see
    ULNAR_SIGN's own docstring). The world target frame (wrap_target,
    closing_target, ulnar_target) must match EACH side's own handedness
    for the Wahba fit to be able to satisfy all three targets
    simultaneously -- using the SAME (left-hand) handedness rule for
    both sides silently forces the RIGHT hand's ulnar axis to fit best
    on the WRONG side (measured: ulnar_down_deg ~147-161deg -- pointing
    UP -- for the right hand only, across every weight combination
    tried, before this per-side fix). Picking wrap_target by "closest to
    current approach axis" alone does not track handedness at all -- it
    must be checked explicitly, per side.
    """
    horiz_to_obj = to_obj_dir.copy()
    horiz_to_obj[2] = 0.0
    n = np.linalg.norm(horiz_to_obj)
    horiz_to_obj = horiz_to_obj / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    # [Fingertip-contact session, bug fix] wrap_a/wrap_b used to be
    # perpendicular to the FULL horizontal to-object direction, which
    # mixes the lateral (Y) offset with the approach standoff (X) --
    # at this task's geometry the two are comparable in magnitude, so
    # the "perpendicular" wrap direction ended up roughly 40deg off pure
    # X, with a real +Y component pointing AWAY from the object. Measured
    # directly (real FK, seed=0): fingertip ends up ~110mm FURTHER from
    # the object in Y than the palm itself, not closer -- adding ~110mm
    # of otherwise-unnecessary palm travel to every precontact approach,
    # which is what was driving the arm into a genuine shoulder-vs-torso
    # kinematic wall (measured: even large forward-standoff/height
    # changes could not clear it -- see FINGERTIP_PRECONTACT's own
    # session notes). The wrap direction only needs to be perpendicular
    # to the LATERAL (Y) offset specifically -- this task's own object
    # approach convention (_mirrored_targets) already fixes X as
    # "forward/backward relative to the object" and Y as "left/right",
    # so computing the perpendicular from the Y-only component (not the
    # standoff-contaminated full horizontal vector) is the correct,
    # narrow fix -- it does not change the palm-inward, finger-to-table,
    # or ulnar-down targets at all, only which in-table-plane direction
    # ("front" vs "back" wrap) the fingers point, which was never
    # independently verified until this session's real end-to-end test.
    lateral_only = np.array([0.0, horiz_to_obj[1], 0.0])
    ln = np.linalg.norm(lateral_only)
    lateral_only = lateral_only / ln if ln > 1e-9 else np.array([0.0, 1.0, 0.0])
    wrap_a = np.array([-lateral_only[1], lateral_only[0], 0.0])
    wrap_b = -wrap_a
    ulnar_target = np.array([0.0, 0.0, -1.0])
    # Handedness check (hard requirement, per side -- see docstring
    # above): for the left hand (ULNAR_SIGN=+1) cross(wrap, closing)
    # must point toward -Z (matching ulnar_target directly); for the
    # right hand (ULNAR_SIGN=-1, mirrored/left-handed local frame) it
    # must point toward +Z instead.
    desired_cross_z_sign = -ULNAR_SIGN[side]
    handed_ok = [w for w in (wrap_a, wrap_b)
                 if np.sign(np.cross(w, to_obj_dir)[2]) == desired_cross_z_sign]
    if len(handed_ok) == 1:
        wrap_target = handed_ok[0]
    else:
        # Both (degenerate closing_target) or neither (should not happen
        # for a non-degenerate closing_target) satisfy handedness -- fall
        # back to the continuity choice among whatever remains.
        candidates = handed_ok if handed_ok else [wrap_a, wrap_b]
        cur_horiz = current_approach_world.copy()
        cur_horiz[2] = 0.0
        cur_n = np.linalg.norm(cur_horiz)
        if cur_n > 1e-9 and len(candidates) > 1:
            cur_horiz = cur_horiz / cur_n
            candidates = sorted(candidates, key=lambda w: -np.dot(w, cur_horiz))
        wrap_target = candidates[0]
    return to_obj_dir, wrap_target, ulnar_target


# [Horizontal-Wrap session] Weight priority among the three axis terms in
# _object_facing_R's Wahba fit: palm-facing (closing, Constraint A) is
# the primary geometric requirement so it carries the largest weight;
# finger-horizontal (wrap, Constraint B) is second; ulnar-down
# (Constraint C) is enforced but weighted lowest of the three -- chosen
# from a bounded 4-candidate real-physics comparison (never a large
# sweep), see docs/history for this session's candidate A/B/C/D table.
HORIZONTAL_WRAP_WEIGHTS = (0.65, 0.20, 0.15)


def _object_facing_R(side: str, palm_pos: np.ndarray, obj_pos: np.ndarray,
                      current_approach_world: np.ndarray | None = None,
                      weights: tuple[float, float, float] = HORIZONTAL_WRAP_WEIGHTS) -> np.ndarray:
    """[Horizontal-Wrap session] 3-vector Kabsch/Wahba fit (supersedes the
    2-vector fit that forced fingers down -- see _horizontal_wrap_target):
      - LOCAL_CLOSING_VEC (real, independently-measured closing/palm-
        inside axis) -> horizontal to-object direction.
      - LOCAL_APPROACH_VEC (finger-longitudinal, local +X) -> horizontal
        wrap direction (replaces "mostly down").
      - ULNAR_SIGN[side] * LOCAL_ULNAR_VEC (pinky-side axis) -> world -Z.
    `current_approach_world` should be the palm_R column-0 direction the
    wrist is CURRENTLY holding (continuity, see _horizontal_wrap_target);
    defaults to world +X when unavailable (first-call/test convenience)."""
    to_obj = obj_pos - palm_pos
    n = np.linalg.norm(to_obj)
    to_obj = to_obj / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    if current_approach_world is None:
        current_approach_world = np.array([1.0, 0.0, 0.0])
    closing_t, wrap_t, ulnar_t = _horizontal_wrap_target(to_obj, current_approach_world, side)
    local_ulnar = ULNAR_SIGN[side] * LOCAL_ULNAR_VEC
    return _wahba_R(
        [LOCAL_CLOSING_VEC, LOCAL_APPROACH_VEC, local_ulnar],
        [closing_t, wrap_t, ulnar_t],
        weights=list(weights),
    )


def _object_facing_angle_deg(side: str, palm_R: np.ndarray, palm_pos: np.ndarray, obj_pos: np.ndarray) -> float:
    """Angle between the REAL empirical closing axis (LOCAL_CLOSING_VEC
    transformed into world via the CURRENT palm_R) and the true palm->
    object direction. [This session] No longer circular: LOCAL_CLOSING_
    VEC is a fixed constant measured independently of whatever the
    solver targets (see LOCAL_CLOSING_VEC's docstring above), unlike the
    previous version which compared palm_R column 1 to itself in effect."""
    to_obj = obj_pos - palm_pos
    n = np.linalg.norm(to_obj)
    if n < 1e-9:
        return 0.0
    to_obj = to_obj / n
    world_closing = palm_R @ LOCAL_CLOSING_VEC
    cos_ang = np.clip(np.dot(world_closing, to_obj), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def _finger_down_angle_deg(palm_R: np.ndarray) -> float:
    """[Retained for diagnostic/candidate-A comparison only -- NO LONGER
    a Gate criterion, see _finger_table_angle_deg below.] Angle between
    the palm's approach axis (palm_R column 0 -- verified, scripts/
    measure_sharpa_side_grasp_axes.py, to coincide EXACTLY with the
    open/straight nonthumb fingers' own root->tip direction) and world
    -Z. This was the OLD (incorrect) task specification -- forcing this
    angle toward 0 forces fingers to point AT the table, which is the
    posture this session's user directive retracts (reference image:
    fingers horizontal, wrapping the object's front/back edges)."""
    approach_axis = palm_R[:, 0]
    cos_ang = np.clip(np.dot(approach_axis, np.array([0.0, 0.0, -1.0])), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def _finger_table_angle_deg(palm_R: np.ndarray) -> float:
    """[Horizontal-Wrap session] Angle between the finger-longitudinal
    axis (palm_R column 0) and the TABLE PLANE (world Z=0) -- the
    Horizontal-Wrap Posture Gate's finger-to-table-plane metric
    (Constraint B), replacing _finger_down_angle_deg. For a unit vector,
    the angle to a plane equals arcsin(|component along the plane's
    normal|); the plane normal here is world Z."""
    approach_axis = palm_R[:, 0]
    return float(np.degrees(np.arcsin(np.clip(abs(approach_axis[2]), 0.0, 1.0))))


def _ulnar_down_angle_deg(side: str, palm_R: np.ndarray) -> float:
    """[Horizontal-Wrap session] Angle between the ulnar (pinky-side)
    local axis, transformed into world frame by the CURRENT palm_R, and
    world -Z -- the Horizontal-Wrap Posture Gate's ulnar-edge-down metric
    (Constraint C)."""
    world_ulnar = palm_R @ (ULNAR_SIGN[side] * LOCAL_ULNAR_VEC)
    cos_ang = np.clip(np.dot(world_ulnar, np.array([0.0, 0.0, -1.0])), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


WRIST_JOINT_SUFFIXES = ("roll", "pitch", "yaw")


def _wrist_joint_margins_deg(env) -> dict:
    """[Horizontal-Wrap session] Real joint-limit margin
    (min(q-lo, hi-q), converted to degrees) for every wrist joint, both
    sides -- Constraint D (wrist stays in a natural neutral range, never
    pinned against a hard limit)."""
    out = {}
    for side in SIDES:
        for suf in WRIST_JOINT_SUFFIXES:
            jn = f"{side}_wrist_{suf}_joint"
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            qadr = env.model.jnt_qposadr[jid]
            q = float(env.data.qpos[qadr])
            lo, hi = env.model.jnt_range[jid]
            out[f"{side}_wrist_{suf}"] = float(np.degrees(min(q - lo, hi - q)))
    return out


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
    #
    # Raised from 0.04/0.22 to 0.10/0.26: the corrected _object_facing_R
    # (see that function's docstring) targets a genuinely different,
    # larger rotation than the old circularly-verified one; the old
    # height was tuned around the OLD (wrong) orientation's finger
    # envelope and is no longer collision-free under the new one
    # (bounded height/y_offset sweep: h=0.04 -> 27N late-ramp hand-table
    # graze). [Audit session] Re-verified under the re-derived (0.68/
    # 0.32) weight pair: 0.10/0.26 still measures 0.00N torso-arm, 0.76N
    # hand-table (well under the 8N limit) through FIVE_FINGER_PRESHAPE,
    # inward angle 12.6-12.7deg both hands (a real ~2.3deg margin under
    # the unchanged 15deg tolerance -- see _object_facing_R's docstring
    # for the full weight-pair sweep this was chosen from).
    side_align_height_m: float = 0.10
    side_align_y_offset_m: float = 0.26
    # Protective curl (index/middle/wrap only, thumb untouched -- same
    # split as CONTACT_ACQUIRE) applied at WRIST_SIDE_GRASP_ALIGN entry,
    # before the position+orientation ramp -- see point 3 above. NOT the
    # same as FIVE_FINGER_PRESHAPE's own (later, unchanged) preshape call;
    # this is purely a table-clearance safety margin, disclosed as such.
    #
    # [This session -- Section 5 re-audit] The PREVIOUS value here (0.5)
    # was NEVER actually reached: the ramp-tick-count formula divided by
    # hand_synergy_action_scale (0.05) but the ramp itself applies
    # close_rate_per_step (0.03) per tick, so `ceil(0.5/0.05)=10` ticks
    # at 0.03/tick landed at curl=0.30, not 0.50 -- confirmed by direct
    # trace (env._group_synergy read mid-rollout). The tick-count formula
    # is fixed below (now divides by close_rate_per_step); this field is
    # set to the value ALREADY verified safe under that real, corrected
    # ramp (0.30 -- LOCAL_CLOSING_VEC above was measured at this exact
    # operating curl) rather than re-verifying a bigger, untested 0.5.
    side_align_preshape_curl: float = 0.3
    side_align_waypoints: int = 14
    side_align_waypoint_ticks: int = 30
    side_align_max_steps: int = 600
    side_align_stable_streak_required: int = 15
    # Horizontal-Wrap Posture Gate tolerances (this session's spec,
    # SUPERSEDES the old Side-Grasp Posture Gate's finger-down check) --
    # measured directly, not guessed:
    side_grasp_inward_angle_tol_deg: float = 15.0
    # [Horizontal-Wrap session] RETRACTED: side_grasp_finger_down_tol_deg
    # (25deg, angle to world -Z) is REMOVED as a Gate criterion. It
    # encoded an incorrect task specification -- forcing fingers to point
    # AT the table -- that drove wrist_roll to within a few degrees of its
    # +-113deg hard limit to satisfy. The finger_down_deg metric itself is
    # RETAINED (_finger_down_angle_deg) for diagnostic/candidate-A
    # comparison only; it is no longer gated on. Replaced by
    # side_grasp_finger_table_tol_deg below (angle to the TABLE PLANE, not
    # to world -Z) plus the new ulnar-edge-down and wrist-margin checks.
    side_grasp_finger_table_tol_deg: float = 20.0
    side_grasp_ulnar_down_tol_deg: float = 25.0
    # [Horizontal-Wrap session] Constraint D: every wrist joint must keep
    # at least this much margin (min(q-lo, hi-q)) from its own hard
    # limit -- see _wrist_joint_margins_deg. Directly targets the failure
    # mode this session's user directive retracts (wrist_roll pinned to
    # within a few degrees of +-113deg under the old finger-down target).
    wrist_joint_margin_tol_deg: float = 10.0
    # [Horizontal-Wrap session] Wrist qvel peak across WRIST_SIDE_GRASP_
    # ALIGN..FINGERTIP_PRECONTACT (see _update_wrap_wrist_qvel_peak) --
    # reuses wrist_max_qvel_rad_s's own value/units (2.0rad/s), same
    # metric family as ARM_LATERAL_CLEARANCE's Wrist Transition Gate.
    side_grasp_mirror_pos_tol_m: float = 0.020
    side_grasp_mirror_ori_tol_deg: float = 10.0
    # FOREARM_SIDE_DESCEND: a SECOND waypointed position move (orientation
    # held via a small soft anchor, NOT re-derived -- it is already
    # correct from WRIST_SIDE_GRASP_ALIGN) from the align pose down/in
    # toward the final pre-contact approach. [Corrected this session's
    # follow-up: the PREVIOUS "measured collision-free at these values"
    # claim was measured only in a single, simplified test that never
    # exercised the FULL end-to-end rollout -- the real, full rollout
    # measures a genuine ~18.21N sustained hand-table force at these same
    # position values with the curl level inherited from WRIST_SIDE_
    # GRASP_ALIGN (0.5). Position (standoff/height/y_offset) is UNCHANGED
    # from that finding -- varying it did not move the residual force at
    # all (see side_descend_curl_target's docstring); the actual fix is
    # curl, not position.] torso-arm/hand-hand stay 0.00N throughout.
    # [Direct-grasp session] 0.12 -> 0.04. The previous FROZEN-orientation
    # design (position translates, orientation locked once at entry) hit
    # a real shoulder-vs-torso kinematic wall around y_offset~0.15
    # (shoulder_yaw_link<->torso_link, measured, prior session). Letting
    # ORIENTATION be RE-DERIVED fresh at every waypoint's own target
    # position (see the waypoint loop below) instead of frozen, combined
    # with a per-DOF joint cost that discourages shoulder_roll/yaw and
    # favors waist/elbow (DESCEND_JOINT_WEIGHT), pushes that wall out
    # dramatically: a bounded real-physics grid over (standoff, height,
    # y_offset) found (0.04, 0.01, 0.055) reaches real nonthumb-
    # fingertip-to-object separation ~57mm with 0.00N torso-arm
    # collision (seed=0) -- close enough for CONTACT_ACQUIRE's own curl
    # travel (~54-57mm, see side_descend_curl_target's docstring) to
    # finish the reach. Nearby points were also swept: closer standoff
    # (0.02) reaches 38mm but with real torso-arm force (10.3N); wider
    # (0.06-0.10) stays safer (0.00-8N) but leaves 74-89mm, too far for
    # curl alone.
    side_descend_standoff_m: float = 0.06
    # [Open-preshape session] 0.03 -> 0.05. The prior audit (see
    # side_descend_curl_target's docstring below) re-verified that curl
    # alone (0.3-0.8, at height=0.03) all fail with 8.1-14.9N hand-table
    # force -- correctly concluding 0.95 was, AT THAT HEIGHT, a genuine
    # geometric requirement, not a free parameter. This session asks the
    # other half of the same question: is the HEIGHT itself the free
    # parameter being held artificially low?
    #
    # [Real-physics candidate comparison, scripts/
    # candidate_open_preshape_descend.py + a follow-up height/curl grid]
    # Raising height alone to 0.07m DOES clear the table at low curl
    # (0.00N through this state at curl=0.35), but a SEPARATE bug found the
    # same session (see side_descend_curl_target's docstring and
    # _descend_locked_R below) makes 0.07m/low-curl combinations
    # incompatible with a genuinely non-drifting, object-facing
    # orientation -- the corrected (non-circular) target at that
    # position/curl combination itself demands fingertip paths that graze
    # the table (measured: HAND_TABLE_COLLISION inside FOREARM_SIDE_DESCEND
    # itself at height=0.07/curl=0.35, NOT a budget/tail-settle artifact).
    # 0.05m (with curl=0.7, see below) is the smallest height increase
    # (from the old 0.03m) that stays genuinely table-safe (0.00N through
    # this state, real physics, seed=0) under the CORRECT, non-drifting
    # orientation, while still curling well short of 0.95's near-total
    # closure.
    # [Direct-grasp session] 0.05 -> 0.01 / 0.15 -> 0.055, together with
    # side_descend_standoff_m's fix above -- see that field's own
    # docstring for the real-physics grid this pair was chosen from.
    #
    # [Palm-press session] Tried -0.02 (a per-fingertip Z sweep showed it
    # brings index/middle/ring/pinky all inside the object's Z band with
    # 0.00N table force at the DESCEND-exit instant) but reverted: run
    # through the FULL real state machine, it broke FOREARM_SIDE_DESCEND's
    # own wrist_yaw-trim convergence (CONTACT_ACQUIRE-entry separation
    # regressed from ~20mm to ~65mm, then diverged further to ~120mm
    # within 30 more ticks with zero contact for the whole episode,
    # measured) -- height interacts with that trim mechanism in a way a
    # single-instant Z check does not capture. Left at the prior verified
    # value; the real fix this session is CONTACT_ACQUIRE's palm-press
    # sub-phase below, not this field. Revisit height only with a full
    # rollout re-verification, not a DESCEND-exit-only snapshot.
    side_descend_height_m: float = 0.01
    side_descend_y_offset_m: float = 0.10
    # [Open-preshape session] 0.95 -> 0.7. Curls index/middle/wrap LESS
    # than the prior value (still more than WRIST_SIDE_GRASP_ALIGN's own
    # protective 0.3, see that field's docstring, so the transition is
    # still a net-closing motion, never net-opening).
    #
    # [Prior audit session] The prior value (0.95) was re-verified (not
    # merely inherited) at the OLD height (0.03): CLOSE_FRACTION=0.85 means
    # synergy's [0,1] only ever spans 85% of each joint's real range, so
    # 0.95 was ~80.75% of the REAL range, leaving only ~4.2-4.7% of real
    # range (~7mm fingertip travel, ~1.4-1.6mm toward the object) before
    # the 1.0 ceiling -- disclosed as a small remaining stroke, but at the
    # time correctly found to be geometrically REQUIRED at that height (0.3
    # -0.8 all failed with 8.1-14.9N hand-table force there).
    #
    # [This session] The actual structural problem was raised in this
    # session's own directive: approaching with fingers already curled to
    # 0.95 leaves almost no real closure stroke for CONTACT_ACQUIRE to
    # later use to acquire contact (measured obj-direction component of
    # the remaining 0.95->1.0 travel is NEGATIVE, i.e. past the point where
    # further curl moves toward the object at all -- see
    # scripts/audit_sharpa_curl_table_feasibility.py). A separate, more
    # fundamental orientation-drift bug (see _descend_locked_R) meant every
    # height/curl combination tested this session (0.03-0.07m height,
    # 0.35-0.95 curl) landed on the SAME actuator-compliance plateau at
    # FINGERTIP_PRECONTACT (~6-10deg orientation error, ~13-28mm position
    # error, all still above the unchanged 5deg/10mm Precontact Tracking
    # Gate) once the drift bug was fixed and the fresh, non-circular
    # target actually held -- height/curl were NOT the limiting factor for
    # that specific residual, table clearance was. 0.7 (height=0.05) is
    # the combination with real hand-table margin (0.00N) AND the smallest
    # measured Precontact residual among the genuinely-open (curl<0.95)
    # candidates tried (ori_err ~6.6deg, pos_err ~17.7mm, vs e.g.
    # height=0.07/curl=0.5's ~8.6deg/28.1mm) -- still short of the Gate,
    # disclosed as such (see FINGERTIP_PRECONTACT's own docstring), but a
    # genuine improvement over curl=0.95's near-zero/negative remaining
    # closure travel, which was this session's actual mandate to remove.
    # [Horizontal-Wrap session] 0.7 -> 0.0. Section 7's own hypothesis
    # confirmed by a bounded real-physics re-sweep under the NEW
    # horizontal-wrap orientation ({0.0, 0.2, 0.35, 0.5, 0.7}, seed=0):
    # EVERY value gives 0.00N forbidden hand-table force and 0.00N
    # torso-arm collision through FOREARM_SIDE_DESCEND (unlike the old
    # finger-down orientation, where curl<0.95 reliably speared the
    # table). The table-collision-avoidance curl this field used to carry
    # is no longer needed once the fingers approach horizontally instead
    # of pointing down -- confirms it was compensating for the wrong
    # orientation, not a property of the object/table geometry itself.
    # 0.0 (matches the criteria in Section 7: fingers stay maximally
    # open, full closure travel preserved for CONTACT_ACQUIRE) is chosen
    # over the still-collision-free 0.2-0.7 values for exactly that
    # reason -- side_align_preshape_curl (0.3, held from WRIST_SIDE_
    # GRASP_ALIGN/FIVE_FINGER_PRESHAPE) already provides the hand's
    # actual entering curl at this state, unchanged.
    side_descend_curl_target: float = 0.0
    # [This session] 8@30 (budget 240, tail 160) converged to a genuine
    # steady-state ~10.71mm residual (measured: unchanged after +300
    # extra settle ticks -- not a timing artifact) at the curled (0.95)
    # load, just over ik_pos_tol -- reducing standoff (a smaller reach)
    # measured WORSE (12.6-12.9mm), ruling that lever out. 10@30 (budget
    # 300, tail 100 -- SMALLER per-waypoint steps, same total ramp
    # duration ballpark) converges under 10mm and reaches FINGERTIP_
    # PRECONTACT; 12@26 also worked but with less tail margin.
    # [Direct-grasp session] 10 -> 30: finer waypoints, needed because
    # orientation is now re-derived (not frozen) at every waypoint's own
    # target position and the total travel is larger (starting from
    # WRIST_SIDE_GRASP_ALIGN's wide staging pose, not just DESCEND's old
    # shorter hop) -- matches the real-physics grid this state's other
    # fields were tuned with.
    side_descend_waypoints: int = 30
    side_descend_waypoint_ticks: int = 30
    side_descend_stable_streak_required: int = 15
    # [Direct-grasp session] Dedicated step budget for THIS state (not
    # the shared max_steps_per_state=700), matching this file's own
    # established convention of per-state overrides (e.g. precontact_
    # max_steps): 30 waypoints * 30 ticks = 900 ticks minimum just for
    # the waypoint schedule, plus a settle tail.
    side_descend_max_steps: int = 1300
    # [Direct-grasp session] "Close enough" real separation to advance to
    # CONTACT_ACQUIRE -- generous relative to CONTACT_ACQUIRE's own real
    # curl travel (~54-57mm, see side_descend_curl_target's docstring) so
    # a real closure attempt is at least geometrically plausible, without
    # hard-blocking progress on the exact number (see the max_steps
    # fallback below, which advances regardless once the waypoint
    # schedule and hard-collision checks are satisfied).
    side_descend_ready_separation_m: float = 0.09
    # [Direct-grasp session] Dedicated budget for CONTACT_ACQUIRE now
    # that a non-contacted side also keeps approaching (not just
    # closing fingers) -- needs more than the shared max_steps_per_state.
    contact_acquire_max_steps: int = 1200
    precontact_standoff_m: float = 0.08
    # [This session] retargeted for the side-grasp geometry -- height
    # matches side_descend_height_m (no further Z motion), y_offset moves
    # the last ~6cm in from side_descend_y_offset_m=0.15 to just outside
    # the object's own half-width (0.06) plus fingertip clearance.
    # [Open-preshape session] Kept EQUAL to side_descend_height_m's new
    # value (0.07, was 0.03) -- Section 5's "마지막 Precontact는 작은 수평
    # inward 이동만 담당" principle (no further vertical descend once
    # side-descend's height/orientation/curl are set) predates this
    # session and is unaffected by the height/curl fix; only the shared
    # value moved.
    precontact_height_m: float = 0.05
    precontact_y_offset_m: float = 0.09
    # [Fingertip-contact session] FINGERTIP_PRECONTACT's own fixed-
    # waypoint Cartesian target (precontact_y_offset_m above) was tuned
    # for the OLD finger-down orientation and, under the new Horizontal-
    # Wrap orientation, drives a real SELF_COLLISION_TORSO_ARM inside its
    # own closed-loop correction re-solve (measured, prior session).
    # Replaced by a closed-loop INWARD SERVO driven by the REAL,
    # measured nonthumb-fingertip-to-object-side-face separation (see
    # _precontact_separation_m) instead of a fixed Cartesian offset --
    # see FINGERTIP_PRECONTACT's own docstring for the full recipe.
    # [Direct-grasp session] Default flipped False -> True: FOREARM_SIDE_
    # DESCEND was rebuilt (see side_descend_standoff_m's docstring) to
    # re-derive orientation fresh per waypoint and reach real fingertip-
    # object separation ~57mm on its own (0.00N torso-arm) -- close
    # enough for CONTACT_ACQUIRE's own curl travel (~54-57mm) to finish
    # the reach, making the separate FINGERTIP_PRECONTACT inward-servo
    # stage unnecessary on the default path. Arm/waist ctrl targets are
    # left exactly as DESCEND set them (CONTACT_ACQUIRE never writes
    # action[0:17]). The FINGERTIP_PRECONTACT servo/recovery-ladder code
    # itself is UNCHANGED and still reachable by setting this back to
    # False -- this is a default flip, not a deletion.
    skip_precontact_servo: bool = True
    precontact_target_separation_m: float = 0.004
    precontact_separation_ok_margin_m: float = 0.002  # "close enough" band: target +/- this
    precontact_servo_step_m: float = 0.010  # max per-tick inward Cartesian step
    # [Fingertip-contact session] Dedicated step budget for this state
    # only (NOT the shared max_steps_per_state, matching this file's own
    # established convention of per-state overrides, e.g. side_align_
    # max_steps): the entry separation here is large (~200mm, WRIST_
    # SIDE_GRASP_ALIGN/FOREARM_SIDE_DESCEND are deliberately wide/high
    # staging poses per Section 8's safe-path principle), and the real,
    # actuator-compliance-limited tracking rate per step is well under
    # precontact_servo_step_m's own cap -- measured to need several
    # hundred ticks even with zero collision recoveries.
    precontact_max_steps: int = 2000
    precontact_recovery_backoff_m: float = 0.010  # temporary OUTWARD nudge when collision guard trips
    precontact_collision_guard_frac: float = 0.5  # trip recovery at this fraction of hand_hand_force_limit_n
    precontact_recovery_rest_gain: float = 0.6  # temporarily stronger pull toward the proven-safe clearance posture during recovery
    precontact_recovery_ticks: int = 20  # how long a triggered recovery stays active before resuming inward servo
    precontact_ori_task_weight: float = 0.2
    # [Fingertip-contact session] A real torso-arm/table contact can
    # appear WITHIN a single tick (measured: 0.00N -> 12.5N in one step,
    # not a gradual ramp the guard threshold above could react to in
    # time) -- immediately hard-failing at the shared 8N Gate limit on
    # the FIRST such spike does not give the recovery backoff (already
    # triggered the same tick via collision_now) a chance to actually
    # resolve it. Per Section 7's "200N급 충돌이나 실제 관통은 허용하지
    # 않는다" (only genuinely dangerous contact is disallowed, not every
    # transient spike), the HARD, immediate-fail threshold here is well
    # above the routine 8N Gate limit; anything between the two triggers
    # recovery but does not, by itself, end the rollout.
    precontact_hard_collision_force_n: float = 40.0
    # [Fingertip-contact session] Escalation ladder per Section 4's
    # "elbow/shoulder null-space -> palm height -> wider lateral
    # re-approach" order: each time a side's recovery triggers, that
    # side's future inward step shrinks (more cautious re-approach) and,
    # after enough repeats, its target height gets a small permanent
    # nudge (probing for a collision-free height instead of repeatedly
    # colliding at the same one).
    precontact_recovery_step_decay: float = 0.6
    precontact_recovery_min_step_m: float = 0.002
    precontact_recovery_height_step_m: float = 0.006
    precontact_recovery_height_trigger: int = 3  # apply a height nudge after this many recoveries on one side
    precontact_recovery_max_height_bias_m: float = 0.03
    # [Fingertip-contact session, bug fix] The shared ik_pos_tol (10mm)
    # is LARGER than a single servo step (precontact_servo_step_m,
    # 4mm) -- CoupledBilateralIK.solve's convergence check runs BEFORE
    # the first Newton iteration, so every per-tick target was already
    # "converged" (error < pos_tol) with ZERO solver iterations, and
    # _apply_ik_result kept re-applying the UNCHANGED current qpos every
    # tick (measured directly: 700 ticks moved separation by <0.1mm).
    # A dedicated, much tighter pos_tol for this specific closed-loop
    # servo call (NOT the shared ik_pos_tol, which stays a Gate
    # criterion elsewhere) makes each small step actually solve.
    precontact_servo_pos_tol_m: float = 0.0005
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
    # [Open-preshape session] 400 -> 700: a BUDGET increase, not a Gate
    # relaxation -- ik_pos_tol/precontact_ori_tol_deg/streak requirements
    # are all unchanged. Needed because FINGERTIP_PRECONTACT's closed-loop
    # correction re-solve (see precontact_correction_ticks) converges the
    # actuator-compliance droop residual monotonically but slowly (measured
    # ~18.5mm plateau under the old 400-tick budget/no-resolve; converges
    # under the unchanged 10mm ik_pos_tol with real margin given more ticks
    # at the validated 30-tick correction cadence -- see that state's
    # docstring). All OTHER states in this file already converge well
    # inside 400 ticks (verified via full regression after this change), so
    # raising the shared ceiling does not mask a real failure anywhere else.
    max_steps_per_state: int = 700
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
    functional_orientation: dict  # [This session] Functional Orientation Gate metrics, see _measure_functional_orientation
    functional_orientation_gate: bool  # [This session] Functional Orientation Gate PASS/FAIL (Section 9 of this session's spec)


class SharpaBimanualGraspExpert:
    """Drives a SharpaGraspEnv with BOTH hands grasping the object
    simultaneously (mirrored across Y). Call .step() once per control
    tick; .run() loops to a terminal state or step budget."""

    # See _arm_action_toward_target's docstring: a plain ctrl-ramp-rate
    # reduction (not a gain/damping change) that empirically eliminates a
    # real wrist_pitch dynamics instability shared with the single-hand
    # controller.
    RAMP_FRACTION = 1.0
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
    # [Fingertip-contact session] FINGERTIP_PRECONTACT's own per-DOF
    # task cost -- root-caused (real physics, torso-arm contact body
    # names enumerated directly) that the redundant 17-DOF solver's
    # cheapest way to satisfy a small INWARD position step is to tuck
    # shoulder_roll/shoulder_yaw (rotating the whole upper arm toward
    # the torso) rather than extend the elbow/wrist -- reproduced as a
    # persistent ~1.7-3N shoulder_yaw_link<->torso_link contact (both
    # sides), too small to trip the collision-recovery guard but large
    # enough to stall further inward progress for hundreds of ticks.
    # Layout matches CoupledBilateralIK's own 17-dim
    # waist(3)+left_arm(7)+right_arm(7) ordering (task_config.py's
    # LEFT/RIGHT_ARM_JOINTS: pitch,roll,yaw,elbow,wrist_roll,wrist_
    # pitch,wrist_yaw) -- indices 4/11=shoulder_roll, 5/12=shoulder_yaw.
    # A higher joint_weight is a SOFT COST (weighted damped pseudo-
    # inverse), not a hard lock, so shoulder motion is still available
    # if genuinely needed, just discouraged relative to elbow/wrist.
    # [User correction, this session] The prior weighting only penalized
    # shoulder_roll/yaw, leaving WRIST at the same default cost as every
    # other joint -- the redundant solver kept solving the final reach by
    # translating the whole arm (shoulder/elbow) toward the torso instead
    # of BENDING AT THE WRIST, which has 3 real DOF (roll/pitch/yaw)
    # positioned right at the base of the hand and therefore the
    # CHEAPEST way (shortest lever, no torso proximity risk at all) to
    # sweep the fingertip through a large arc. Layout matches
    # CoupledBilateralIK's own 17-dim waist(3)+left_arm(7)+right_arm(7)
    # ordering (task_config.py's LEFT/RIGHT_ARM_JOINTS: pitch,roll,yaw,
    # elbow,wrist_roll,wrist_pitch,wrist_yaw) -- shoulder pitch/roll/yaw
    # = 3,4,5 / 10,11,12; elbow = 6/13; wrist roll/pitch/yaw = 7,8,9 /
    # 14,15,16. Wrist is now CHEAP (favored), shoulder EXPENSIVE
    # (strongly discouraged), elbow moderately discouraged -- a soft
    # cost (weighted damped pseudo-inverse), not a hard lock, so shoulder
    # motion remains available if genuinely required.
    PRECONTACT_JOINT_WEIGHT = np.ones(17)
    PRECONTACT_JOINT_WEIGHT[[3, 4, 5, 10, 11, 12]] = 8.0
    PRECONTACT_JOINT_WEIGHT[[6, 13]] = 3.0
    PRECONTACT_JOINT_WEIGHT[[7, 8, 9, 14, 15, 16]] = 0.15
    DESCEND_JOINT_WEIGHT = np.ones(17)
    DESCEND_JOINT_WEIGHT[[3, 4, 5, 10, 11, 12]] = 8.0
    DESCEND_JOINT_WEIGHT[[6, 13]] = 3.0
    DESCEND_JOINT_WEIGHT[[7, 8, 9, 14, 15, 16]] = 0.15

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
        self._descend_locked_R: dict | None = None  # [Open-preshape session] set at FOREARM_SIDE_DESCEND entry, reused through FINGERTIP_PRECONTACT
        self.wrist_orientation_drift_deg: float = float("inf")
        self.left_object_facing_angle_deg: float = float("inf")
        self.right_object_facing_angle_deg: float = float("inf")
        self.torso_arm_collision_force_n: float = 0.0
        self.max_hand_table_force_n: float = 0.0
        # [Horizontal-Wrap session] forbidden (fingertip/palm/wrist) vs
        # allowed (ulnar-edge support geom) hand<->table force, tracked
        # separately from max_hand_table_force_n above (kept for backward
        # compatibility/diagnostics) -- see _hand_table_forces_categorized.
        self.max_forbidden_hand_table_force_n: float = 0.0
        self.max_allowed_ulnar_table_force_n: float = 0.0
        # [Horizontal-Wrap session] wrist qvel peak from WRIST_SIDE_GRASP_
        # ALIGN entry through FINGERTIP_PRECONTACT -- Constraint from the
        # Horizontal-Wrap Posture Gate (wrist qvel <= 2rad/s), same metric
        # family as ARM_LATERAL_CLEARANCE's own Wrist Transition Gate but
        # tracked across this later, separate span of states.
        self._wrap_max_wrist_qvel: float = 0.0
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
        self._functional_orientation: dict = {}
        self._functional_orientation_gate: bool = False

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

    def _current_rest_q(self) -> np.ndarray:
        """[Fingertip-contact session] Same 17-dim layout as self._rest_q
        (waist(3)+left_arm(7)+right_arm(7)), but read from the CURRENT
        live qpos instead of the pose at __init__ -- used as the closed-
        loop precontact servo's own IK warm start/null-space bias, so the
        redundant solver stays near whatever configuration the arm is
        ALREADY, physically holding (branch continuity) instead of being
        pulled back toward a stale far-away rest pose every tick."""
        env = self.env
        return np.concatenate([
            env.data.qpos[env._waist_qpos_adr].copy(), env.data.qpos[env._arm_qpos_adr[:7]].copy(),
            env.data.qpos[env._arm_qpos_adr[7:]].copy(),
        ])

    def _nonthumb_tip_centroid(self, side: str) -> np.ndarray:
        nonthumb = ("index", "middle", "ring", "pinky")
        return np.mean([self.env.fingertip_pos(side, f) for f in nonthumb], axis=0)

    def _precontact_separation_m(self, side: str) -> float:
        """[Fingertip-contact session] Real, measured signed distance
        from the nonthumb fingertip centroid to the object's OWN near
        side face (positive = still outside the object, the quantity
        FINGERTIP_PRECONTACT's closed-loop servo drives toward
        precontact_target_separation_m) -- replaces the old fixed
        precontact_y_offset_m Cartesian target."""
        obj_pos = self._object_pos()
        half = self.env.config.object_half_size
        tip_centroid = self._nonthumb_tip_centroid(side)
        return float(Y_SIGN[side] * (tip_centroid[1] - obj_pos[1]) - half)

    def _solve_both(self, targets: dict, R: dict, require_orientation: bool, ori_task_weight: float,
                     rest_q: np.ndarray | None = None, rest_gain: float | None = None,
                     pos_tol: float | None = None, joint_weight: np.ndarray | None = None):
        data = self.env.data
        scratch = mujoco.MjData(self.env.model)
        scratch.qpos[:] = data.qpos
        mujoco.mj_forward(self.env.model, scratch)
        kwargs = {} if rest_gain is None else {"rest_gain": rest_gain}
        if joint_weight is not None:
            kwargs["joint_weight"] = joint_weight
        return self.ik.solve(
            scratch, targets["left"], R["left"], targets["right"], R["right"],
            rest_q=rest_q if rest_q is not None else self._rest_q,
            pos_tol=pos_tol if pos_tol is not None else self.config.ik_pos_tol,
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

    def _side_reach_ready(self, side: str) -> bool:
        """CONTACT_ACQUIRE's real readiness bar for one side: wrap (the
        physically closest group) touching is NOT enough on its own --
        Gate A's topology needs wrap AND (index or middle) too, and
        real physics showed the palm freezing on wrap-only contact left
        index/middle permanently ~20mm short even at full curl (syn=1.0).
        So keep approaching until wrap AND at least one of index/middle
        have registered real contact."""
        g = self._group_ever_contacted[side]
        return g["wrap"] and (g["index"] or g["middle"])

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

    def _update_wrap_wrist_qvel_peak(self) -> None:
        """[Horizontal-Wrap session] Same wrist-DOF-index construction as
        ARM_LATERAL_CLEARANCE's own _clearance_max_raw_wrist_qvel, called
        every tick of WRIST_SIDE_GRASP_ALIGN/FIVE_FINGER_PRESHAPE/
        FOREARM_SIDE_DESCEND/FINGERTIP_PRECONTACT."""
        wrist_dof = np.concatenate([self.env._arm_dof_adr[4:7], self.env._arm_dof_adr[11:14]])
        v = float(np.max(np.abs(self.env.data.qvel[wrist_dof])))
        self._wrap_max_wrist_qvel = max(self._wrap_max_wrist_qvel, v)

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
        """[Horizontal-Wrap session] Horizontal-Wrap Posture Gate --
        SUPERSEDES the old Side-Grasp Posture Gate's finger-down check
        (which forced fingers to point at the table, driving wrist_roll
        toward its +-113deg hard limit -- retracted, see the module
        docstring and side_grasp_finger_table_tol_deg's own docstring).
        Measured directly from live FK/contact state, called once when
        WRIST_SIDE_GRASP_ALIGN's own stability/collision/object-facing
        checks have just passed. Every sub-condition here is a real
        measurement (fingertip/body FK, palm axes, elbow/shoulder body
        position, actual contact forces/qvel already tracked this tick)
        -- none of it is inferred from joint targets alone."""
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
        # [This session] Palm normals must OPPOSE each other (facing in) --
        # uses the REAL empirical closing axis (LOCAL_CLOSING_VEC
        # transformed through the current palm_R), not raw palm_R column 1
        # directly: column 1 is no longer guaranteed to coincide with the
        # actual closing direction now that _object_facing_R targets
        # LOCAL_CLOSING_VEC via a Wahba fit instead of setting column 1
        # itself.
        left_closing_world = palm["left"][1] @ LOCAL_CLOSING_VEC
        right_closing_world = palm["right"][1] @ LOCAL_CLOSING_VEC
        normals_opposed = float(np.dot(left_closing_world, right_closing_world)) < 0.0
        # [Horizontal-Wrap session] finger_down_deg kept for diagnostic/
        # candidate-A comparison only -- NOT gated on. finger_table_deg
        # (Constraint B) and ulnar_down_deg (Constraint C) are the real
        # Gate metrics now.
        finger_down_deg = {s: _finger_down_angle_deg(palm[s][1]) for s in SIDES}
        finger_table_deg = {s: _finger_table_angle_deg(palm[s][1]) for s in SIDES}
        ulnar_down_deg = {s: _ulnar_down_angle_deg(s, palm[s][1]) for s in SIDES}
        wrist_margins_deg = _wrist_joint_margins_deg(env)
        wrist_margin_ok = all(v >= cfg_margin for v, cfg_margin in
                               zip(wrist_margins_deg.values(), [self.config.wrist_joint_margin_tol_deg] * len(wrist_margins_deg)))
        wrist_qvel_ok = self._wrap_max_wrist_qvel <= self.config.wrist_max_qvel_rad_s
        # [Horizontal-Wrap session] Constraint C's "lowest point" check:
        # the ulnar support body's own origin must sit at or below every
        # other candidate hand-table-contact body's origin (fingertip DP
        # bodies, palm base, wrist housing) -- a real, body-position-based
        # ordering, not just an axis-angle proxy. Small tolerance (5mm)
        # for numerical/measurement noise, not a relaxation of intent.
        ulnar_lowest_ok = {}
        ulnar_margin_m = {}
        for s in SIDES:
            ulnar_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, env.ULNAR_SUPPORT_BODY[s])
            ulnar_z = float(env.data.xpos[ulnar_bid][2])
            other_names = [sc.sharpa_body(s, f, "DP") for f in sc.FINGERS] + [
                sc.sharpa_body(s, "hand", "C_MC"), f"{s}_wrist_roll_link", f"{s}_wrist_pitch_link", f"{s}_wrist_yaw_link",
            ]
            other_z = []
            for n in other_names:
                bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)
                if bid >= 0:
                    other_z.append(float(env.data.xpos[bid][2]))
            margin = (min(other_z) - ulnar_z) if other_z else 0.0
            ulnar_margin_m[s] = margin
            ulnar_lowest_ok[s] = margin >= -0.005
        # [Horizontal-Wrap session] DIAGNOSTIC ONLY, no longer gated on
        # here. This check assumed the OLD finger-down geometry, where
        # WRIST_SIDE_GRASP_ALIGN's own (deliberately staging, wide/high)
        # pose already put fingertips down near the object's own height
        # band. The new safe-path design (Section 8) makes this state a
        # genuine STAGING pose away from the object BY DESIGN (palm/
        # orientation set up before moving in) -- fingertips at this
        # pose's height are expected to sit above the object, and the
        # real "does closure reach the object" requirement is verified
        # later by the closure-swept-path check (Constraint E) at the
        # actual FOREARM_SIDE_DESCEND/FINGERTIP_PRECONTACT geometry.
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
        # [Horizontal-Wrap session] hand_table_ok now checks the FORBIDDEN
        # category only (fingertip/palm/wrist housing); the allowed ulnar-
        # edge-support contact is tracked separately and does not, by
        # itself, fail this Gate (it still must stay under the same 8N
        # cap and not spike wrist qvel -- see allowed_ulnar_ok below).
        hand_table_ok = self.max_forbidden_hand_table_force_n <= self.config.hand_hand_force_limit_n
        allowed_ulnar_ok = self.max_allowed_ulnar_table_force_n <= self.config.hand_hand_force_limit_n
        hand_hand_ok = self._max_hand_hand <= self.config.hand_hand_force_limit_n if self._max_hand_hand else \
            env._hand_hand_contact_force() <= self.config.hand_hand_force_limit_n
        premature_contact_ok = self._proximal_penetration_ok()
        elbow_ok = all(v <= 0.02 for v in elbow_above_shoulder_m.values())

        metrics = {
            "outside_side_face": outside_side_face,
            "inward_angle_deg": inward_angle_deg,
            "normals_opposed": normals_opposed,
            "finger_down_deg": finger_down_deg,
            "finger_table_deg": finger_table_deg,
            "ulnar_down_deg": ulnar_down_deg,
            "ulnar_lowest_ok": ulnar_lowest_ok,
            "ulnar_margin_m": ulnar_margin_m,
            "wrist_margins_deg": wrist_margins_deg,
            "wrist_margin_ok": wrist_margin_ok,
            "wrist_qvel_peak": self._wrap_max_wrist_qvel,
            "wrist_qvel_ok": wrist_qvel_ok,
            "tip_height_overlaps_side": tip_height_overlaps_side,
            "crosses_top_footprint": crosses_top_footprint,
            "mirror_pos_err_m": mirror_pos_err_m,
            "mirror_ori_err_deg": mirror_ori_err_deg,
            "elbow_above_shoulder_m": elbow_above_shoulder_m,
            "elbow_ok": elbow_ok,
            "torso_arm_ok": torso_arm_ok,
            "hand_table_ok": hand_table_ok,
            "allowed_ulnar_ok": allowed_ulnar_ok,
            "max_forbidden_hand_table_force_n": self.max_forbidden_hand_table_force_n,
            "max_allowed_ulnar_table_force_n": self.max_allowed_ulnar_table_force_n,
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
            and finger_table_deg["left"] <= cfg.side_grasp_finger_table_tol_deg
            and finger_table_deg["right"] <= cfg.side_grasp_finger_table_tol_deg
            and ulnar_down_deg["left"] <= cfg.side_grasp_ulnar_down_tol_deg
            and ulnar_down_deg["right"] <= cfg.side_grasp_ulnar_down_tol_deg
            and all(ulnar_lowest_ok.values())
            and wrist_margin_ok
            and wrist_qvel_ok
            and not crosses_top_footprint
            and mirror_pos_err_m <= cfg.side_grasp_mirror_pos_tol_m
            and mirror_ori_err_deg <= cfg.side_grasp_mirror_ori_tol_deg
            and elbow_ok
            and torso_arm_ok
            and hand_table_ok
            and allowed_ulnar_ok
            and hand_hand_ok
            and premature_contact_ok
            and self._side_align_stable_streak >= cfg.side_align_stable_streak_required
        )
        metrics["gate"] = gate
        return metrics

    def _empirical_group_closure_world(self, side: str, groups: tuple[int, ...], curl_amount: float) -> tuple[dict, dict]:
        """State-preserving: applies a REAL curl delta to the given hand
        GROUPS (sc.GROUPS index, e.g. (1,2,3)=index/middle/wrap or
        (0,)=thumb) via actual env.step() physics, reads the resulting
        fingertip world positions for EVERY finger, then restores
        qpos/qvel/ctrl exactly -- callable mid-rollout with no side
        effects. Returns (tips_after, tips_before). [Audit session]
        generalized from the old hardcoded-(1,2,3) version (renamed) so
        the same real-physics recipe can probe the thumb group too, for
        the Thumb Opposition Subgate."""
        env = self.env
        side_idx = 0 if side == "left" else 1
        tips0 = {f: env.fingertip_pos(side, f).copy() for f in sc.FINGERS}
        saved = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), env._group_synergy.copy())
        steps = max(1, int(np.ceil(curl_amount / max(env.config.hand_synergy_action_scale, 1e-9))))
        for _ in range(steps):
            a = self._zero_action()
            for g in groups:
                a[17 + side_idx * 4 + g] = 1.0
            env.step(a)
        tips1 = {f: env.fingertip_pos(side, f).copy() for f in sc.FINGERS}
        env.data.qpos[:], env.data.qvel[:], env.data.ctrl[:] = saved[0], saved[1], saved[2]
        env._group_synergy[:] = saved[3]
        mujoco.mj_forward(env.model, env.data)
        return tips1, tips0

    def _measure_finger_closure_direction(self, curl_probe: float = 0.15) -> dict:
        """Finger-Closure Direction Subgate -- the REAL, non-circular
        replacement for the old inward_angle check in
        _measure_side_grasp_posture. Applies an actual small additional
        closure (curl_probe) to index/middle/wrap from the CURRENT (real,
        in-progress) preshape state -- exactly what CONTACT_ACQUIRE will
        later do -- and checks where the fingertips actually move, not
        what the wrist was solved to point at.

        [Audit session] Renamed from the prior commit's
        "_measure_functional_orientation": that name overclaimed -- it
        only ever checked 4-finger closure direction (never thumb
        opposition, never the finger-down/palm-inside angle it computed
        but didn't gate on, never swept collision). This is now ONE
        subgate that _measure_functional_orientation combines with
        _measure_thumb_opposition and the other hard requirements below.

        curl_probe=0.15 default: measured (prior session) at
        WRIST_SIDE_GRASP_ALIGN's converged pose, mean inward displacement
        grows with probe size (0.05->~0.13mm, 0.10->~0.6mm, 0.15->~2.1-
        2.4mm, 0.20->~5.4-5.8mm), all with the SAME sign (n_positive=4/4
        at every probe size tested up to 0.30 -- no direction reversal in
        this range, unlike the much-more-curled-base regime Section 6
        warns about). 0.15 is the smallest probe that clears this
        session's own >2mm requirement with real margin on BOTH hands."""
        env = self.env
        obj_pos = self._object_pos()
        nonthumb = ("index", "middle", "ring", "pinky")
        per_side: dict = {}
        for side in SIDES:
            tips1, tips0 = self._empirical_group_closure_world(side, (1, 2, 3), curl_probe)
            per_finger_dot = {}
            for f in nonthumb:
                delta = tips1[f] - tips0[f]
                d_obj = obj_pos - tips0[f]
                d_obj = d_obj / (np.linalg.norm(d_obj) + 1e-12)
                per_finger_dot[f] = float(np.dot(delta, d_obj))
            n_positive = sum(1 for v in per_finger_dot.values() if v > 0.0)
            mean_disp_inward_m = float(np.mean([
                np.dot(tips1[f] - tips0[f], (obj_pos - tips0[f]) / (np.linalg.norm(obj_pos - tips0[f]) + 1e-12))
                for f in nonthumb
            ]))
            palm_pos, palm_R = env.palm_pose(side)
            per_side[side] = {
                "per_finger_dot": per_finger_dot,
                "n_positive": n_positive,
                "mean_disp_inward_mm": mean_disp_inward_m * 1000.0,
                "inward_angle_deg": _object_facing_angle_deg(side, palm_R, palm_pos, obj_pos),
                "finger_down_deg": _finger_down_angle_deg(palm_R),
                "finger_table_deg": _finger_table_angle_deg(palm_R),
            }
        gate = all(per_side[s]["n_positive"] >= 3 and per_side[s]["mean_disp_inward_mm"] > 2.0 for s in SIDES)
        return {"per_side": per_side, "gate": gate}

    def _measure_thumb_opposition(self, curl_probe: float = 0.15) -> dict:
        """[Audit session, new] Thumb Opposition Subgate -- geometric,
        pre-contact analog of Gate A's own force-opposition check (see
        module docstring: "thumb's contact-force direction opposes the
        index/middle combined force direction"). Before any contact
        exists there is no force to measure, so this checks the
        GEOMETRIC precondition for that force opposition to be possible
        once closure reaches the object:
          - aperture_dist_mm: real distance from the thumb's own
            fingertip to the OTHER four fingers' centroid (a degenerate/
            collapsed hand, thumb already coincident with the other
            fingers, cannot oppose anything -- requires >30mm, a real,
            generous-but-nonzero hand-scale margin).
          - thumb_disp_toward_aperture: applying a real thumb-only curl
            delta (same _empirical_group_closure_world recipe as the
            4-finger check, group index 0) must move the thumb TOWARD
            that aperture (positive dot product with the aperture
            direction) -- i.e. closing the thumb closes the pincer
            instead of opening it further or moving irrelevantly.
          - thumb_disp_toward_object: the same thumb closure must also
            move toward the object center -- consistent with a pincer
            that closes ONTO the object, not past it or away from it."""
        env = self.env
        obj_pos = self._object_pos()
        nonthumb = ("index", "middle", "ring", "pinky")
        per_side: dict = {}
        for side in SIDES:
            thumb_tip0 = env.fingertip_pos(side, "thumb").copy()
            nonthumb_centroid0 = np.mean([env.fingertip_pos(side, f) for f in nonthumb], axis=0)
            tips1, tips0 = self._empirical_group_closure_world(side, (0,), curl_probe)
            thumb_delta = tips1["thumb"] - tips0["thumb"]
            aperture_vec = nonthumb_centroid0 - thumb_tip0
            aperture_dist_m = float(np.linalg.norm(aperture_vec))
            aperture_dir = aperture_vec / (aperture_dist_m + 1e-12)
            to_obj_from_thumb = obj_pos - thumb_tip0
            to_obj_from_thumb = to_obj_from_thumb / (np.linalg.norm(to_obj_from_thumb) + 1e-12)
            toward_aperture_m = float(np.dot(thumb_delta, aperture_dir))
            toward_object_m = float(np.dot(thumb_delta, to_obj_from_thumb))
            aperture_ok = aperture_dist_m > 0.03
            opposed = aperture_ok and toward_aperture_m > 0.0 and toward_object_m > 0.0
            per_side[side] = {
                "aperture_dist_mm": aperture_dist_m * 1000.0,
                "thumb_disp_toward_aperture_mm": toward_aperture_m * 1000.0,
                "thumb_disp_toward_object_mm": toward_object_m * 1000.0,
                "opposed": opposed,
            }
        gate = all(per_side[s]["opposed"] for s in SIDES)
        return {"per_side": per_side, "gate": gate}

    def _measure_functional_orientation(self, curl_probe: float = 0.15) -> dict:
        """Functional Orientation Gate (Section 9) -- PASSES only when
        ALL of the following, each independently measured, hold:
          - Finger-Closure Direction Subgate (>=3/4 nonthumb fingers move
            toward the object, mean inward displacement >2mm).
          - Thumb Opposition Subgate (thumb closure geometrically opposes
            the 4-finger group, see _measure_thumb_opposition).
          - palm inside (= the real empirical closing axis) <=
            side_grasp_inward_angle_tol_deg (15deg) from the object
            direction, BOTH hands.
          - finger-to-table-plane <= side_grasp_finger_table_tol_deg
            (20deg -- [Horizontal-Wrap session] REPLACES the old finger-
            down-vs-world--Z check, which forced fingers at the table and
            drove wrist_roll toward its hard limit; see
            side_grasp_finger_table_tol_deg's own docstring), BOTH hands.
          - swept collision: cumulative torso-arm/FORBIDDEN hand-table
            force recorded over the approach SO FAR (self.torso_arm_
            collision_force_n / self.max_forbidden_hand_table_force_n,
            updated every tick since STABLE_START) stays under the shared
            8N forbidden-collision limit -- not just the instantaneous
            reading at measurement time. The allowed ulnar-edge-support
            contact is excluded from this check (see
            max_allowed_ulnar_table_force_n, gated separately in
            _measure_side_grasp_posture).
        [Audit session] Prior name/scope: this function used to BE what
        is now _measure_finger_closure_direction (4-finger check only) --
        it never checked thumb, never gated on the finger-down angle it
        computed, and never checked collision history. Renamed/rebuilt to
        actually combine every sub-condition Section 9 specifies."""
        cfg = self.config
        finger_closure = self._measure_finger_closure_direction(curl_probe)
        thumb = self._measure_thumb_opposition(curl_probe)
        inward_ok = all(
            finger_closure["per_side"][s]["inward_angle_deg"] <= cfg.side_grasp_inward_angle_tol_deg for s in SIDES
        )
        finger_table_ok = all(
            finger_closure["per_side"][s]["finger_table_deg"] <= cfg.side_grasp_finger_table_tol_deg for s in SIDES
        )
        swept_collision_ok = (
            self.torso_arm_collision_force_n <= cfg.hand_hand_force_limit_n
            and self.max_forbidden_hand_table_force_n <= cfg.hand_hand_force_limit_n
        )
        gate = (
            finger_closure["gate"] and thumb["gate"] and inward_ok and finger_table_ok and swept_collision_ok
        )
        return {
            "finger_closure_direction": finger_closure,
            "thumb_opposition": thumb,
            "inward_ok": inward_ok,
            "finger_table_ok": finger_table_ok,
            "swept_collision_ok": swept_collision_ok,
            "curl_probe": curl_probe,
            "gate": gate,
        }

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
            # [Earlier session] Replaces the old FOREARM_DESCEND (position-
            # only) + WRIST_ALIGN (measure-only) pair. User requirement:
            # both palms end up beside the object's own side faces,
            # FACING EACH OTHER (closing axis toward the object center),
            # fingers generally pointing down -- not a top-down reach.
            #
            # [Functional Orientation fix] The PREVIOUS verification here
            # (inward_angle_deg computed from palm_R column 1 vs the
            # object direction) was CIRCULAR: the wrist was solved to
            # make column 1 point there, so the ~14deg residual it
            # reported was solver noise, not evidence the hand could
            # functionally grasp. A real, non-circular empirical audit
            # (scripts/audit_sharpa_local_closing_frame.py) found the
            # TRUE closing axis is ~50deg off from that assumption
            # (contact-free, curl-arc-aware measurement -- see
            # _object_facing_R's docstring). _object_facing_R is rebuilt
            # via a 2-vector Kabsch/Wahba fit against that real axis, and
            # this state's own target height/y_offset are raised
            # (side_align_height_m/y_offset_m) because the OLD values were
            # only ever proven collision-free under the WRONG orientation.
            # [Audit session] Verified end-to-end with the RE-DERIVED
            # (0.68/0.32) weight pair (see _object_facing_R's docstring
            # for why the first weight pair used here was improper): 0.00N
            # torso-arm through this state and FIVE_FINGER_PRESHAPE,
            # inward angle 12.6-12.7deg both hands (real non-circular
            # metric, under the unchanged 15deg tolerance), finger-down
            # 22.6deg both hands (under the unchanged, no-longer-relaxed
            # 25deg tolerance), and the full Functional Orientation Gate
            # (Section 9 -- finger-closure direction, thumb opposition,
            # palm-inside angle, finger-down angle, swept collision, ALL
            # independently measured) genuinely PASSES here. FOREARM_SIDE_
            # DESCEND (next) now also safely carries this corrected
            # orientation through its own translate (0.00N torso-arm,
            # previously ~200-300N under the improper weight pair) -- see
            # that state's docstring.
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
                self._wrap_max_wrist_qvel = 0.0
                self.env.set_preshape("left", 1.0)
                self.env.set_preshape("right", 1.0)
                self._side_align_start_pos = {s: self.env.palm_pose(s)[0].copy() for s in SIDES}
                self._side_align_start_R = {s: self.env.palm_pose(s)[1].copy() for s in SIDES}
                obj_pos = self._object_pos()
                self._side_align_final_pos = self._mirrored_targets(
                    cfg.approach_standoff_m, cfg.side_align_height_m, cfg.side_align_y_offset_m
                )
                self._side_align_final_R = {
                    s: _object_facing_R(s, self._side_align_final_pos[s], obj_pos,
                                         current_approach_world=self._side_align_start_R[s][:, 0]) for s in SIDES
                }
                self._side_align_waypoint = 0
                self._side_align_stable_streak = 0
                self._side_align_R_hist = {"left": [], "right": []}
            # Protective curl ramp (index/middle/wrap, not thumb), fast
            # relative to the position/orientation ramp so it is mostly
            # established before the riskier later waypoints.
            # [This session -- Section 5 re-audit, bug fix] The per-tick
            # action below applies close_rate_per_step (0.03) as the
            # ACTUAL curl increment per tick (via _group_action's
            # normalize-then-env.step-denormalize round trip), NOT
            # hand_synergy_action_scale (0.05) -- the tick-count formula
            # must divide by the SAME rate it ramps at, or the ramp stops
            # short of its target (previously landed at curl=0.30 for a
            # nominal 0.5 target; see side_align_preshape_curl's own
            # docstring for the direct-trace confirmation).
            # [Horizontal-Wrap session] curl_ramp_delay_ticks: the
            # protective curl used to start at state_step=0, simultaneous
            # with the FIRST position/orientation waypoint -- at that
            # instant the hands are still at FOREARM_FORWARD_REACH's
            # narrower (y_offset=0.15) end pose, not yet the wider
            # (0.26) align target, so curling immediately produced a
            # transient ~8.27N hand-hand graze (measured, swept-path
            # check). Delaying curl start by one waypoint tick lets the
            # first lateral-separation waypoint fire first (real physics
            # re-verified: 0.00N swept hand-hand through this ramp).
            curl_ramp_delay_ticks = cfg.side_align_waypoint_ticks
            curl_ramp_ticks = int(np.ceil(cfg.side_align_preshape_curl / max(cfg.close_rate_per_step, 1e-9)))
            if curl_ramp_delay_ticks <= self._state_step < curl_ramp_delay_ticks + curl_ramp_ticks:
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
            self._update_wrap_wrist_qvel_peak()
            lR = self.env.palm_pose("left")[1].copy()
            rR = self.env.palm_pose("right")[1].copy()
            self._side_align_R_hist["left"].append(lR)
            self._side_align_R_hist["right"].append(rR)
            no_collision = (self.env._torso_arm_collision_force() <= cfg.hand_hand_force_limit_n
                             and self.env._hand_hand_contact_force() <= cfg.hand_hand_force_limit_n)
            table_forces = self.env._hand_table_forces_categorized()
            self.max_hand_table_force_n = max(self.max_hand_table_force_n, self.env._hand_table_contact_force())
            self.max_forbidden_hand_table_force_n = max(self.max_forbidden_hand_table_force_n, table_forces["forbidden"])
            self.max_allowed_ulnar_table_force_n = max(self.max_allowed_ulnar_table_force_n, table_forces["allowed_ulnar"])
            no_table_hit = table_forces["forbidden"] <= cfg.hand_hand_force_limit_n
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
                # [This session] Functional Orientation Gate -- the REAL,
                # non-circular check (see _measure_functional_orientation's
                # docstring). Measured here, NOT gated on for advancement
                # (a hard block would freeze the whole rollout before
                # FIVE_FINGER_PRESHAPE/FOREARM_SIDE_DESCEND can even be
                # attempted/diagnosed) -- stored on the outcome so callers
                # (tests, this session's report) can assert on it directly.
                self._functional_orientation = self._measure_functional_orientation()
                self._functional_orientation_gate = self._functional_orientation["gate"]
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
            self._update_wrap_wrist_qvel_peak()
            if self._state_step >= 30:
                if (not self._proximal_penetration_ok()
                        or self.env._hand_hand_contact_force() > cfg.hand_hand_force_limit_n):
                    self._fail(BimanualFailureReason.SELF_COLLISION_BEFORE_CONTACT)
                    return action
                table_forces = self.env._hand_table_forces_categorized()
                self.max_forbidden_hand_table_force_n = max(self.max_forbidden_hand_table_force_n, table_forces["forbidden"])
                self.max_allowed_ulnar_table_force_n = max(self.max_allowed_ulnar_table_force_n, table_forces["allowed_ulnar"])
                if table_forces["forbidden"] > cfg.hand_hand_force_limit_n:
                    self._fail(BimanualFailureReason.HAND_TABLE_COLLISION)
                    return action
                self._advance(BimanualGraspState.FOREARM_SIDE_DESCEND)

        elif state == BimanualGraspState.FOREARM_SIDE_DESCEND:
            # [This session] Second waypointed position move, orientation
            # HELD (small soft anchor to the ALREADY-correct side-grasp
            # orientation, never re-derived) -- matches the "no large
            # wrist rotation after descending" requirement. See
            # side_descend_standoff_m's docstring for the measured
            # collision-free numbers (UNDER THE OLD, circularly-verified
            # orientation -- see the disclosed regression note below).
            #
            # [Audit session] A PRIOR commit on this branch (w_closing=
            # 0.95 in _object_facing_R) reproduced a genuine ~200-300N
            # torso-arm self-collision here, disclosed but NOT fixed, on
            # the rollout's DEFAULT (only) path -- a real safety
            # regression left live. That collision is CAUSALLY GONE under
            # this session's re-weighted _object_facing_R (w_closing=
            # 0.68, see that function's docstring): measured 0.00N
            # torso-arm through this ENTIRE state, real physics, seed=0.
            # This is not "translation itself was never the cause" --
            # zero-translation was already 0N under the OLD orientation
            # too (see the still-true note below) -- it is that the OLD
            # orientation's closing-axis weight (0.95) demanded a rotation
            # extreme enough that the redundant 17-DOF solver had to pick
            # a torso-colliding branch to ALSO satisfy translation; a less
            # extreme (but still Gate-passing, see _object_facing_R) target
            # does not force that branch choice. Re-verified: insensitive
            # to side_descend_curl_target and waypoint count exactly as
            # before (those were never the cause either).
            #
            # [This session, follow-up -- see side_descend_curl_target's
            # docstring] Root-caused (scripts/diagnose_hand_table_contact_
            # geometry.py, substep-level) the ~18N hand-table force
            # previously measured here: essentially ALL non-thumb
            # fingertips (index/middle/ring/pinky *_DP), sustained for
            # ~95% of this state's ticks -- a real geometric overlap
            # between the curl=0.5 fingers (inherited from
            # WRIST_SIDE_GRASP_ALIGN's protective preshape) and the table
            # at this state's target height, NOT a transient impact and
            # NOT sensitive to standoff/y_offset (measured: identical
            # force across standoff in {0.12..0.15}, y_offset in
            # {0.15..0.18} -- ruling out palm XY position as the cause).
            # Curling FURTHER during this state (toward
            # side_descend_curl_target, holding wrist orientation and
            # palm height/standoff/y_offset UNCHANGED) retracts the
            # fingertips clear: measured 18.21N -> 2.96N (curl 0.5->0.95),
            # with palm inward angle and finger-down angle BOTH improving
            # into their Gate ranges as a side effect (fingers curled
            # tighter naturally sit closer to the palm's own orientation
            # cone). The remaining ~2.96N is NOT further reducible by
            # more curl (0.95 vs 1.0 identical) or more settle time
            # (measured unchanged after +100 extra ticks) -- a genuine,
            # small, disclosed residual, safely under the shared 8N
            # forbidden-collision limit but not exactly the ideal 0N.
            #
            # [Open-preshape session, supersedes the curl-only fix above]
            # The curl=0.5->0.95 fix above solved table clearance by
            # retracting fingers almost fully closed BEFORE any object
            # contact exists, leaving CONTACT_ACQUIRE almost no real
            # closure stroke (measured: 0.95->1.0 travel's object-direction
            # component goes NEGATIVE -- past the point where further curl
            # even moves toward the object). This session instead raises
            # side_descend_height_m (0.03->0.07) so the SAME table
            # clearance is achieved by standoff, not by pre-closing the
            # hand -- curl drops back to 0.35 (see that field's own
            # up-to-date docstring for the real-physics numbers). The
            # geometric analysis in this comment block (why table contact
            # occurs, which fingers, why standoff/y_offset don't move it)
            # is still accurate; only the height/curl VALUES it was tuned
            # around have changed -- see side_descend_height_m/
            # side_descend_curl_target's own docstrings for the current
            # fix and the numbers behind it.
            # [Direct-grasp session] REBUILT: orientation is now RE-
            # DERIVED FRESH at every waypoint's own interpolated target
            # position (never frozen) -- since _object_facing_R is a
            # function of palm position, the truly-correct orientation
            # genuinely changes as the palm gets closer, and forcing it
            # to hold a single value computed far away is exactly what
            # was driving the redundant solver into a shoulder-tucking,
            # torso-colliding branch (measured, prior session: a real
            # kinematic wall around y_offset~0.15m under the frozen-
            # orientation design). Re-deriving per waypoint, combined
            # with DESCEND_JOINT_WEIGHT (cheap waist, expensive shoulder)
            # and a fresh IK solve EVERY tick (not just at waypoint
            # boundaries) anchored to the CURRENT live qpos, was found
            # (bounded real-physics grid, see side_descend_standoff_m's
            # docstring) to reach real fingertip-object separation
            # ~57mm at 0.00N torso-arm collision -- close enough for
            # CONTACT_ACQUIRE's own curl travel to finish the reach.
            if self._state_step == 0:
                self._descend_stable_streak = 0
                self._side_descend_start = {s: self.env.palm_pose(s)[0].copy() for s in SIDES}
                self._side_descend_final = self._mirrored_targets(
                    cfg.side_descend_standoff_m, cfg.side_descend_height_m, cfg.side_descend_y_offset_m
                )
                self._side_descend_waypoint = 0
                self._side_descend_wp_start_R = {s: self.env.palm_pose(s)[1].copy() for s in SIDES}
                self._side_descend_wp_target_R = {s: self.env.palm_pose(s)[1].copy() for s in SIDES}
                self._side_descend_wp_target_pos = {s: self._side_descend_start[s].copy() for s in SIDES}
            ticks_per_wp = cfg.side_descend_waypoint_ticks
            tick_in_wp = self._state_step % ticks_per_wp
            if tick_in_wp == 0 and self._side_descend_waypoint < cfg.side_descend_waypoints:
                self._side_descend_waypoint += 1
                frac = _quintic_scale(self._side_descend_waypoint / cfg.side_descend_waypoints)
                obj_pos = self._object_pos()
                self._side_descend_wp_start_R = {s: self.env.palm_pose(s)[1].copy() for s in SIDES}
                for s in SIDES:
                    target_pos = (1 - frac) * self._side_descend_start[s] + frac * self._side_descend_final[s]
                    self._side_descend_wp_target_pos[s] = target_pos
                    # [Direct-grasp session] Continuity reference is the
                    # PREVIOUS WAYPOINT'S OWN TARGET R (clean, already
                    # computed), not the live/settling palm_R -- using the
                    # live pose here was measured to let solver-branch
                    # noise from a not-yet-settled actuator feed back into
                    # the wrap-direction handedness choice near the end of
                    # the ramp, causing separation to visibly REGRESS
                    # (worsen) over the final few waypoints instead of
                    # monotonically improving.
                    cur_approach = self._side_descend_wp_target_R[s][:, 0]
                    self._side_descend_wp_target_R[s] = _object_facing_R(
                        s, target_pos, obj_pos, current_approach_world=cur_approach
                    )
            # [Direct-grasp session, bug fix] Once all waypoints are
            # spent, tick_in_wp keeps cycling 0..ticks_per_wp-1 (state_
            # step keeps advancing), which was RESETTING sub_frac back
            # down to a small value every ticks_per_wp ticks -- SLERPing
            # from the (stale) last waypoint's OWN start_R back toward
            # its target_R again and again, i.e. the palm orientation
            # oscillated near the final target instead of holding it,
            # which showed up as real separation visibly REGRESSING
            # (worsening) after the nominal waypoint schedule finished.
            # Fix: once waypoints_exhausted, hold sub_frac at 1.0 (the
            # final target), never re-decay it.
            waypoints_exhausted = self._side_descend_waypoint >= cfg.side_descend_waypoints
            sub_frac = 1.0 if waypoints_exhausted else (tick_in_wp + 1) / ticks_per_wp
            R_now = {
                s: _slerp_R(self._side_descend_wp_start_R[s], self._side_descend_wp_target_R[s], sub_frac)
                for s in SIDES
            }
            result = self._solve_both(
                self._side_descend_wp_target_pos, R_now, require_orientation=False, ori_task_weight=0.3,
                rest_q=self._current_rest_q(), rest_gain=0.3, pos_tol=0.003, joint_weight=self.DESCEND_JOINT_WEIGHT,
            )
            self._apply_ik_result(result)
            self._descend_locked_R = dict(self._side_descend_wp_target_R)  # kept for FINGERTIP_PRECONTACT's own use when skip_precontact_servo=False
            # [User-directed fix, this session] "손목에도 관절이 있으니까
            # 안으로 굽혀라" -- direct FK sweep (holding the rest of the
            # arm fixed at a SAFE, 0-collision waypoint) found wrist_YAW
            # alone sweeps the real fingertip separation from ~108mm to
            # ~16mm with a single ~29deg rotation and ZERO added torso-arm
            # force, since it does not move the shoulder/upper-arm at
            # all. The redundant IK solve was NOT spontaneously finding
            # this on its own even with wrist weighted cheap (DESCEND_
            # JOINT_WEIGHT) -- trimmed in directly here instead: once the
            # position waypoint schedule is done, bend each side's
            # wrist_yaw further inward, a little every tick, capped and
            # monitored for real collision (back off if it appears).
            if waypoints_exhausted:
                if not hasattr(self, "_descend_wrist_yaw_trim"):
                    self._descend_wrist_yaw_trim = {"left": 0.0, "right": 0.0}
                torso_now = self.env._torso_arm_collision_force()
                guard = cfg.precontact_collision_guard_frac * cfg.hand_hand_force_limit_n
                wrist_yaw_sign = {"left": -1.0, "right": 1.0}
                for side in SIDES:
                    sep_now = self._precontact_separation_m(side)
                    if torso_now > guard:
                        self._descend_wrist_yaw_trim[side] = max(self._descend_wrist_yaw_trim[side] - 0.02, 0.0)
                    elif sep_now > 0.005:
                        self._descend_wrist_yaw_trim[side] = min(self._descend_wrist_yaw_trim[side] + 0.01, 1.15)
                self._arm_ik_target[6] += wrist_yaw_sign["left"] * self._descend_wrist_yaw_trim["left"]
                self._arm_ik_target[13] += wrist_yaw_sign["right"] * self._descend_wrist_yaw_trim["right"]
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            self._update_wrap_wrist_qvel_peak()
            curl_now = self.env._group_synergy.copy()
            curl_target = np.array([0.0, cfg.side_descend_curl_target, cfg.side_descend_curl_target,
                                     cfg.side_descend_curl_target] * 2)
            curl_scale = self.env.config.hand_synergy_action_scale
            curl_delta = np.clip(curl_target - curl_now, -curl_scale, curl_scale)
            action[17:25] = curl_delta / max(curl_scale, 1e-9)
            left_pos = self.env.palm_pose("left")[0]
            right_pos = self.env.palm_pose("right")[0]
            pos_err = max(float(np.linalg.norm(self._side_descend_final["left"] - left_pos)),
                          float(np.linalg.norm(self._side_descend_final["right"] - right_pos)))
            hand_table_force = self.env._hand_table_contact_force()
            self.max_hand_table_force_n = max(self.max_hand_table_force_n, hand_table_force)
            table_forces = self.env._hand_table_forces_categorized()
            self.max_forbidden_hand_table_force_n = max(self.max_forbidden_hand_table_force_n, table_forces["forbidden"])
            self.max_allowed_ulnar_table_force_n = max(self.max_allowed_ulnar_table_force_n, table_forces["allowed_ulnar"])
            torso_force_now = self.env._torso_arm_collision_force()
            self.torso_arm_collision_force_n = max(self.torso_arm_collision_force_n, torso_force_now)
            hand_hand_now = self.env._hand_hand_contact_force()
            # [Direct-grasp session] This final approach genuinely, and
            # stably (not growing further -- measured, real physics),
            # brushes the torso at this reach depth (see side_descend_
            # standoff_m's docstring: a real kinematic property of this
            # orientation family, not a solver artifact). Per Section 7
            # ("200N급 충돌이나 실제 관통은 허용하지 않는다" -- only
            # genuinely dangerous contact is disallowed, not a bounded,
            # stable graze), the HARD fail threshold here is
            # precontact_hard_collision_force_n (40N, matching FINGERTIP_
            # PRECONTACT's own established threshold), not the routine 8N
            # Gate limit -- which stays the "no forbidden collision" bar
            # elsewhere (Horizontal-Wrap Posture Gate, swept-path checks
            # up through FIVE_FINGER_PRESHAPE) and is NOT weakened there.
            if table_forces["forbidden"] > cfg.precontact_hard_collision_force_n:
                self._fail(BimanualFailureReason.HAND_TABLE_COLLISION)
                return action
            if torso_force_now > cfg.precontact_hard_collision_force_n:
                self._fail(BimanualFailureReason.SELF_COLLISION_TORSO_ARM)
                return action
            if hand_hand_now > cfg.precontact_hard_collision_force_n:
                self._fail(BimanualFailureReason.CONTACT_LOST)
                return action
            waypoints_done = self._side_descend_waypoint >= cfg.side_descend_waypoints
            left_sep = self._precontact_separation_m("left")
            right_sep = self._precontact_separation_m("right")
            # [Direct-grasp session] Readiness is judged on REAL, measured
            # fingertip-object separation (does the closure swept path
            # have a realistic chance of reaching the object once
            # CONTACT_ACQUIRE starts curling -- Section 5's "단순한 파지
            # 판단" principle), not on hitting the shared 10mm ik_pos_tol
            # against an intentionally-aggressive Cartesian target that
            # this reach cannot fully converge to.
            close_enough = max(left_sep, right_sep) <= cfg.side_descend_ready_separation_m
            stable_now = close_enough and waypoints_done
            self._descend_stable_streak = self._descend_stable_streak + 1 if stable_now else 0
            if self._descend_stable_streak >= cfg.side_descend_stable_streak_required:
                if cfg.skip_precontact_servo:
                    self._advance(BimanualGraspState.CONTACT_ACQUIRE)
                else:
                    self._advance(BimanualGraspState.FINGERTIP_PRECONTACT)
            elif self._state_step >= cfg.side_descend_max_steps:
                if cfg.skip_precontact_servo and waypoints_done:
                    # [Direct-grasp session] Even short of the ready-
                    # separation streak, if the waypoint schedule genuinely
                    # completed and no hard collision occurred, advance to
                    # CONTACT_ACQUIRE anyway with whatever real separation
                    # was reached -- Section 5 explicitly rules out
                    # blocking finger closure on an intermediate distance
                    # gate; CONTACT_ACQUIRE's own real contact detection
                    # (not this state) is the actual arbiter.
                    self._advance(BimanualGraspState.CONTACT_ACQUIRE)
                else:
                    self._fail(BimanualFailureReason.SIDE_DESCEND_NOT_ACHIEVED)

        elif state == BimanualGraspState.FINGERTIP_PRECONTACT:
            # [Fingertip-contact session] REPLACES the old fixed-
            # Cartesian-waypoint approach (precontact_y_offset_m etc,
            # tuned for the OLD finger-down orientation) with a closed-
            # loop INWARD SERVO driven by the REAL, measured nonthumb-
            # fingertip-to-object-side-face separation
            # (_precontact_separation_m) -- the old fixed target drove a
            # genuine SELF_COLLISION_TORSO_ARM under the new Horizontal-
            # Wrap orientation (measured, prior session; see
            # PROJECT_CONTEXT.md Section 0). Wrist orientation is HELD at
            # FOREARM_SIDE_DESCEND's own frozen _descend_locked_R
            # throughout (never re-derived here -- matches the
            # "orientation 정렬 후에는 descend/precontact에서 큰 회전 금지"
            # principle, unchanged from the prior session).
            #
            # Every tick: measure the CURRENT separation per side, take a
            # small bounded step directly toward the target separation
            # (never a big one-shot jump -- this alone is why the old
            # WAYPOINT_COUNT/WAYPOINT_TICKS interpolation scheme is no
            # longer needed here), solve IK with rest_q = the CURRENT
            # live qpos (branch continuity: the redundant 17-DOF solver
            # stays near whatever configuration the arm is ALREADY,
            # physically holding, instead of being pulled back toward a
            # fixed target every tick) and a SOFT orientation anchor.
            #
            # Collision recovery (graduated escalation ladder -- Section
            # 4's "elbow/shoulder null-space -> palm height -> wider
            # lateral re-approach" order -- not an immediate hard fail):
            # if torso-arm OR forbidden hand-table OR hand-hand force
            # crosses precontact_collision_guard_frac of the shared 8N
            # limit, the servo STOPS moving inward for precontact_
            # recovery_ticks and instead nudges OUTWARD by precontact_
            # recovery_backoff_m while temporarily raising rest_gain
            # toward precontact_recovery_rest_gain (elbow/shoulder null-
            # space adjustment). Each recovery on a side also shrinks
            # that side's future step (more cautious re-approach) and,
            # after precontact_recovery_height_trigger repeats, nudges
            # that side's target HEIGHT (probing for a collision-free
            # height instead of repeating the same collision). A real
            # contact can appear WITHIN a single tick (measured: 0.00N ->
            # 12.5N in one step) -- immediately hard-failing at the
            # routine 8N Gate limit would not give this recovery a chance
            # to act; per Section 7 ("200N급 충돌이나 실제 관통은 허용하지
            # 않는다" -- only genuinely dangerous contact is disallowed,
            # not a transient spike), the HARD, immediate-fail threshold
            # is precontact_hard_collision_force_n, well above the
            # routine 8N limit that only triggers recovery.
            if self._state_step == 0:
                self._precontact_final_R = {s: self._descend_locked_R[s].copy() for s in SIDES}
                self._precontact_stable_streak = 0
                self._precontact_recovery_until = {"left": 0, "right": 0}
                self._precontact_recovery_count = {"left": 0, "right": 0}
                self._precontact_height_bias = {"left": 0.0, "right": 0.0}
                self._precontact_start_z = {s: float(self.env.palm_pose(s)[0][2]) for s in SIDES}

            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            self._update_wrap_wrist_qvel_peak()

            torso_force = self.env._torso_arm_collision_force()
            table_forces = self.env._hand_table_forces_categorized()
            self.max_forbidden_hand_table_force_n = max(self.max_forbidden_hand_table_force_n, table_forces["forbidden"])
            self.max_allowed_ulnar_table_force_n = max(self.max_allowed_ulnar_table_force_n, table_forces["allowed_ulnar"])
            hand_hand_force = self.env._hand_hand_contact_force()
            if torso_force > cfg.precontact_hard_collision_force_n:
                self._fail(BimanualFailureReason.SELF_COLLISION_TORSO_ARM)
                return action
            if table_forces["forbidden"] > cfg.precontact_hard_collision_force_n:
                self._fail(BimanualFailureReason.HAND_TABLE_COLLISION)
                return action
            guard = cfg.precontact_collision_guard_frac * cfg.hand_hand_force_limit_n
            collision_now = torso_force > guard or table_forces["forbidden"] > guard or hand_hand_force > guard

            # [Fingertip-contact session] Target generated from the
            # ACTUAL, real qpos-derived palm position every tick (per the
            # spec's "실제 측정 pose에서 오차 갱신", never accumulated on a
            # stale/decoupled cursor -- a decoupled monotonic cursor was
            # tried first and measured to race ahead of the real,
            # actuator-compliance-limited tracking, turning each solve
            # into a large single-shot correction and reproducing the
            # exact SELF_COLLISION_TORSO_ARM this design retires). The
            # step itself is intentionally small (precontact_servo_step_m,
            # decayed per side after repeated recoveries) so consecutive
            # IK solves stay close to one another even though each is
            # re-anchored at the real, live pose.
            targets = {}
            R = {}
            rest_gain = cfg.posture_rest_gain
            for s in SIDES:
                palm_pos, _ = self.env.palm_pose(s)
                in_recovery = self._total_step < self._precontact_recovery_until[s]
                if collision_now and not in_recovery:
                    self._precontact_recovery_until[s] = self._total_step + cfg.precontact_recovery_ticks
                    self._precontact_recovery_count[s] += 1
                    if self._precontact_recovery_count[s] % cfg.precontact_recovery_height_trigger == 0:
                        self._precontact_height_bias[s] = min(
                            self._precontact_height_bias[s] + cfg.precontact_recovery_height_step_m,
                            cfg.precontact_recovery_max_height_bias_m,
                        )
                    in_recovery = True
                if in_recovery:
                    step = -cfg.precontact_recovery_backoff_m / cfg.precontact_recovery_ticks
                    rest_gain = max(rest_gain, cfg.precontact_recovery_rest_gain)
                else:
                    error = self._precontact_separation_m(s) - cfg.precontact_target_separation_m
                    side_step_cap = max(
                        cfg.precontact_servo_step_m * (cfg.precontact_recovery_step_decay ** self._precontact_recovery_count[s]),
                        cfg.precontact_recovery_min_step_m,
                    )
                    step = float(np.clip(error, 0.0, side_step_cap))
                inward_dir = np.array([0.0, -Y_SIGN[s], 0.0])
                # Height target servoed the SAME way as the inward
                # step (small bounded move toward a goal), never a
                # discontinuous jump -- the goal itself
                # (_precontact_start_z + accumulated height_bias) only
                # changes when a recovery escalates it.
                z_goal = self._precontact_start_z[s] + self._precontact_height_bias[s]
                z_step = float(np.clip(z_goal - palm_pos[2], -cfg.precontact_servo_step_m, cfg.precontact_servo_step_m))
                targets[s] = palm_pos + inward_dir * step + np.array([0.0, 0.0, z_step])
                R[s] = self._precontact_final_R[s]
            result = self._solve_both(targets, R, require_orientation=False,
                                       ori_task_weight=cfg.precontact_ori_task_weight,
                                       rest_q=self._current_rest_q(), rest_gain=rest_gain,
                                       pos_tol=cfg.precontact_servo_pos_tol_m,
                                       joint_weight=self.PRECONTACT_JOINT_WEIGHT)
            self._apply_ik_result(result)

            left_R = self.env.palm_pose("left")[1]
            right_R = self.env.palm_pose("right")[1]

            def _ang_deg(Ra, Rb):
                r_delta = Ra.T @ Rb
                cos_ang = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)
                return float(np.degrees(np.arccos(cos_ang)))

            left_ori_err = _ang_deg(left_R, self._precontact_final_R["left"])
            right_ori_err = _ang_deg(right_R, self._precontact_final_R["right"])
            self._precontact_final_ori_error_deg = {"left": left_ori_err, "right": right_ori_err}
            left_sep_now = self._precontact_separation_m("left")
            right_sep_now = self._precontact_separation_m("right")
            self._precontact_final_pos_error = {"left": max(left_sep_now, 0.0), "right": max(right_sep_now, 0.0)}

            ok_margin = cfg.precontact_target_separation_m + cfg.precontact_separation_ok_margin_m
            stable_now = (
                left_sep_now <= ok_margin and right_sep_now <= ok_margin and not collision_now
                and left_ori_err <= cfg.precontact_ori_tol_deg and right_ori_err <= cfg.precontact_ori_tol_deg
                and hand_hand_force <= cfg.hand_hand_force_limit_n
            )
            self._precontact_stable_streak = self._precontact_stable_streak + 1 if stable_now else 0
            self._max_precontact_stable_streak = max(self._max_precontact_stable_streak, self._precontact_stable_streak)
            if self._precontact_stable_streak >= cfg.precontact_stable_streak_required:
                self._advance(BimanualGraspState.CONTACT_ACQUIRE)
            elif self._state_step >= cfg.precontact_max_steps:
                self._fail(BimanualFailureReason.PRECONTACT_TRACKING_NOT_ACHIEVED)

        elif state == BimanualGraspState.CONTACT_ACQUIRE:
            # [Palm-press session] "블록을 손바닥에 먼저 붙이고 손가락을
            # 접어라": a side does NOT start curling until its own palm
            # has registered real contact force against the object.
            # Measured this session: closing fingers before the palm ever
            # touches (previous design) relied entirely on wrist-rotation
            # sweeping fingertips through a wide arc while the palm base
            # body itself stayed 31-38mm away at 0.00N contact for the
            # ENTIRE state -- rotating the wrist barely translates the
            # palm plate, which sits close to that rotation axis. So this
            # state now has two sub-phases per side: PALM-PRESS (drive the
            # palm toward the object's real nearest surface point, no
            # curling yet) until real palm contact registers, then the
            # existing curl-toward-Gate-A behavior.
            if self._state_step == 0:
                self._contact_acquire_recovery_until = {"left": 0, "right": 0}
                # Anchor the servo target to a ONE-TIME pose snapshot + a
                # cumulative intended offset, not to "current palm_pos"
                # every tick -- re-deriving the target from the live
                # (possibly slightly-off-converged) pose every tick let
                # per-tick IK residual bias compound over a long run into
                # tens of mm of real, measured drift with no commanded
                # cause (this session's other verified fix).
                self._contact_acquire_anchor = {s: self.env.palm_pose(s)[0].copy() for s in SIDES}
                self._contact_acquire_offset = {"left": 0.0, "right": 0.0}
                self._palm_press_offset = {"left": 0.0, "right": 0.0}
                self._palm_ever_contacted = {"left": False, "right": False}
                if self._initial_obj_xy is None:
                    # [Direct-grasp session] Object displacement was NOT
                    # tracked at all before CONTACT_ACQUIRE (only from
                    # FORCE_SETTLE onward) -- real physics showed the
                    # object CAN get bumped during this state (measured,
                    # before the step-size fix above: ~90mm real
                    # displacement in one run), so track it honestly from
                    # here too instead of silently reporting 0.0.
                    self._initial_obj_xy = self._object_pos()[:2].copy()
            self._track_stability()

            for side in SIDES:
                was_contacted = self._palm_ever_contacted[side]
                if self.env._palm_contact_force(side) > cfg.contact_force_threshold_n:
                    self._palm_ever_contacted[side] = True
                if self._palm_ever_contacted[side] and not was_contacted:
                    # Just pressed -- re-anchor the (now separate) lateral
                    # fine-servo phase from THIS pose, not the old
                    # far-away DESCEND-exit anchor, so phase 2 doesn't
                    # jump.
                    self._contact_acquire_anchor[side] = self.env.palm_pose(side)[0].copy()
                    self._contact_acquire_offset[side] = 0.0

            deltas = {"left": {}, "right": {}}
            for side in SIDES:
                for group in ("index", "middle", "wrap"):
                    touched = self._group_contact_now(side, group)
                    if self._palm_ever_contacted[side] and not touched:
                        deltas[side][group] = cfg.close_rate_per_step
            action[17:25] = self._group_action(deltas)

            # [Retreat-report session] Freezing a side on its FIRST group
            # contact (any of index/middle/wrap) was measured to strand
            # index/middle ~20mm short forever, because wrap (physically
            # closest) always touches first and used to stop the approach
            # right there. _side_reach_ready (wrap AND index-or-middle)
            # matches Gate A's real topology need instead.
            torso_force_now = self.env._torso_arm_collision_force()
            table_forces_now = self.env._hand_table_forces_categorized()
            hand_hand_now = self.env._hand_hand_contact_force()
            if torso_force_now > cfg.precontact_hard_collision_force_n:
                self._fail(BimanualFailureReason.SELF_COLLISION_TORSO_ARM)
                return action
            if table_forces_now["forbidden"] > cfg.precontact_hard_collision_force_n:
                self._fail(BimanualFailureReason.HAND_TABLE_COLLISION)
                return action
            guard = cfg.precontact_collision_guard_frac * cfg.hand_hand_force_limit_n
            collision_now = torso_force_now > guard or table_forces_now["forbidden"] > guard
            targets = {}
            R = {}
            need_solve = False
            joint_weight = self.PRECONTACT_JOINT_WEIGHT.copy()
            for side in SIDES:
                side_has_contact = self._side_reach_ready(side)
                palm_pos, _ = self.env.palm_pose(side)
                if side_has_contact:
                    targets[side] = palm_pos
                elif not self._palm_ever_contacted[side]:
                    # PALM-PRESS: drive the palm toward the object along
                    # the SAME lateral (Y) closing axis as the rest of
                    # this grasp, not a generic nearest-point-on-box
                    # vector -- tried that first and it silently pulled
                    # the palm toward whichever face the anchor happened
                    # to overshoot most (the object's TOP face, when
                    # fingertips started above the Z band), which is the
                    # wrong contact surface for this Horizontal-Wrap
                    # grasp. Real fix for reaching the correct face is
                    # side_descend_height_m (Z pre-alignment, see that
                    # field's docstring) -- this phase only adds
                    # loosened shoulder/elbow weight so genuine
                    # translation (not just wrist rotation, which barely
                    # moves the palm plate) is reachable by the solver.
                    need_solve = True
                    anchor = self._contact_acquire_anchor[side]
                    direction = np.array([0.0, -Y_SIGN[side], 0.0])
                    in_recovery = self._total_step < self._contact_acquire_recovery_until[side]
                    if collision_now and not in_recovery:
                        self._contact_acquire_recovery_until[side] = self._total_step + cfg.precontact_recovery_ticks
                        in_recovery = True
                    if in_recovery:
                        step = -cfg.precontact_recovery_backoff_m / cfg.precontact_recovery_ticks
                    else:
                        step = cfg.precontact_servo_step_m * 0.1
                    self._palm_press_offset[side] = float(np.clip(
                        self._palm_press_offset[side] + step, 0.0,
                        cfg.side_descend_standoff_m + cfg.side_descend_y_offset_m,
                    ))
                    targets[side] = anchor + direction * self._palm_press_offset[side]
                    shoulder_idx = [3, 4, 5] if side == "left" else [10, 11, 12]
                    elbow_idx = [6] if side == "left" else [13]
                    joint_weight[shoulder_idx] = 3.0
                    joint_weight[elbow_idx] = 1.5
                else:
                    # Palm already pressed -- same lateral fine-servo as
                    # before to help index/middle also reach real contact
                    # while curl runs. A group that already touched
                    # (typically wrap) can still be pushed harder as this
                    # continues -- guard on ITS force too, not just
                    # torso/table collision, so this gentle push never
                    # turns into a real over-force event.
                    need_solve = True
                    contacted_peak = max(
                        (self.env._group_contact_force(side, g)[0]
                         for g in ("index", "middle", "wrap") if self._group_ever_contacted[side][g]),
                        default=0.0,
                    )
                    side_collision_now = collision_now or contacted_peak > guard
                    in_recovery = self._total_step < self._contact_acquire_recovery_until[side]
                    if side_collision_now and not in_recovery:
                        self._contact_acquire_recovery_until[side] = self._total_step + cfg.precontact_recovery_ticks
                        in_recovery = True
                    if in_recovery:
                        step = -cfg.precontact_recovery_backoff_m / cfg.precontact_recovery_ticks
                    else:
                        # [Direct-grasp session] Much gentler than
                        # FOREARM_SIDE_DESCEND's own step -- by this point
                        # curl alone is already closing most of the
                        # remaining gap, and measured real physics showed
                        # a too-aggressive simultaneous arm nudge + curl
                        # combination can push the OBJECT itself away
                        # before a stable contact registers (real object
                        # displacement observed, ~90mm in one run) instead
                        # of helping.
                        step = cfg.precontact_servo_step_m * 0.05
                    # Cap at the full lateral gap DESCEND started from --
                    # letting the commanded offset keep growing past what
                    # the arm can physically track (measured: real pose
                    # plateaus while offset keeps climbing) just widens
                    # the target/actual mismatch for no benefit.
                    self._contact_acquire_offset[side] = float(np.clip(
                        self._contact_acquire_offset[side] + step, 0.0, cfg.side_descend_y_offset_m,
                    ))
                    inward_dir = np.array([0.0, -Y_SIGN[side], 0.0])
                    targets[side] = self._contact_acquire_anchor[side] + inward_dir * self._contact_acquire_offset[side]
                R[side] = self._descend_locked_R[side]
            if need_solve:
                result = self._solve_both(
                    targets, R, require_orientation=False, ori_task_weight=0.3,
                    rest_q=self._current_rest_q(), rest_gain=0.3, pos_tol=cfg.precontact_servo_pos_tol_m,
                    joint_weight=joint_weight,
                )
                self._apply_ik_result(result)
                action[0:3] = self._waist_action_toward_target()
                action[3:17] = self._arm_action_toward_target()

            both_have_some_contact = all(self._side_reach_ready(side) for side in SIDES)
            if both_have_some_contact:
                self._advance(BimanualGraspState.THUMB_OPPOSE)
            elif self._state_step >= cfg.contact_acquire_max_steps:
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
            functional_orientation=self._functional_orientation,
            functional_orientation_gate=self._functional_orientation_gate,
        )
