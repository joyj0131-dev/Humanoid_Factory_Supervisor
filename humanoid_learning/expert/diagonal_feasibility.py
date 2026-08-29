"""Static (pure-kinematic, no physics rollout) feasibility search for the
SIZE_12 Diagonal Four-Face Grasp -- Posture-Based Multi-Start IK session.

Section 4 of that session's spec explicitly requires finding whether a
collision-free, joint-limit-safe FINAL pose for the diagonal corner grasp
exists at all BEFORE touching the live state machine ("state machine을
먼저 대규모 수정하지 마라"). This module answers exactly that question,
reusing coupled_ik.CoupledBilateralIK (already built for the live
controller) directly on a caller-provided SCRATCH MjData -- it never
touches the live env or grasp_expert.py's state machine.

Two independent things are varied per attempt (Section 5: "posture
family" concept):
  - ``rest_q`` (null-space bias, same 17-DoF ordering the live coupled
    solver uses) -- pulls the solution toward a posture family without
    forcing it.
  - the STARTING ``q`` fed to solve() (i.e. the scratch data's qpos
    before calling solve) -- since CoupledBilateralIK.solve() begins its
    damped-least-squares iteration from whatever qpos the scratch data
    currently holds, seeding a genuinely different starting configuration
    (not just a different null-space target) is what actually lets the
    search escape a bad local minimum, which a rest_q-only bias cannot
    guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from humanoid_learning.envs import hand_synergy
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.expert.coupled_ik import CoupledBilateralIK, CoupledIKResult
from humanoid_learning.expert.grasp_expert import make_side_pinch_orientation

# Object-local outward face normals (Section 11 of the prior session's
# face classifier uses the SAME convention: dominant local axis + sign).
FACE_NORMALS_LOCAL: dict[str, np.ndarray] = {
    "FACE_POS_X": np.array([1.0, 0.0, 0.0]),
    "FACE_NEG_X": np.array([-1.0, 0.0, 0.0]),
    "FACE_POS_Y": np.array([0.0, 1.0, 0.0]),
    "FACE_NEG_Y": np.array([0.0, -1.0, 0.0]),
}


@dataclass(frozen=True)
class HandFaceAssignment:
    thumb_face: str
    finger_face: str  # shared index+middle target face


@dataclass(frozen=True)
class CandidateAssignment:
    name: str
    left: HandFaceAssignment
    right: HandFaceAssignment


# Section 1 of the current session: the two chiralities to test. Roles
# (which face is "thumb" vs "finger") differ between hands in BOTH
# candidates -- neither is a simple left/right mirror of the other,
# matching the literal spec text (C2 is NOT C1 with L/R swapped, it swaps
# the finger/thumb role pairing on top of that).
CANDIDATE_C1 = CandidateAssignment(
    "C1",
    left=HandFaceAssignment(thumb_face="FACE_POS_X", finger_face="FACE_POS_Y"),
    right=HandFaceAssignment(thumb_face="FACE_NEG_X", finger_face="FACE_NEG_Y"),
)
CANDIDATE_C2 = CandidateAssignment(
    "C2",
    left=HandFaceAssignment(thumb_face="FACE_NEG_X", finger_face="FACE_POS_Y"),
    right=HandFaceAssignment(thumb_face="FACE_POS_X", finger_face="FACE_NEG_Y"),
)


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    return R.reshape(3, 3)


def corner_direction_local(assignment: HandFaceAssignment) -> np.ndarray:
    """Unit vector, in object-local XY, from the object center toward the
    corner (edge) shared by the two assigned faces -- e.g. FACE_POS_X +
    FACE_POS_Y gives [1,1,0]/sqrt2, the corner between them."""
    v = FACE_NORMALS_LOCAL[assignment.thumb_face] + FACE_NORMALS_LOCAL[assignment.finger_face]
    n = np.linalg.norm(v)
    assert n > 1e-9, f"thumb_face/finger_face must be adjacent (perpendicular), got {assignment}"
    return v / n


@dataclass
class HandCornerTarget:
    palm_pos: np.ndarray
    palm_R: np.ndarray
    corner_edge_point_world: np.ndarray  # the actual object corner/edge point (no standoff)
    approach_world: np.ndarray
    closing_world: np.ndarray


def corner_direction_local_at_yaw(assignment: HandFaceAssignment, corner_yaw_deg: float) -> np.ndarray:
    """Contact-Driven Four-Face Whole-Body Feasibility session, Section 2/
    6: instead of hard-coding the 45-degree bisector (the prior session's
    ``corner_direction_local``, still used as the corner_yaw_deg=45 case),
    blends between the FINGER face's own normal (yaw=0 -- a plain frontal
    approach to the easier-to-reach face, close to this hand's existing
    default geometry) and the full diagonal bisector (yaw=45, the two
    faces' normals equally weighted). Smaller yaw asks less of wrist_yaw/
    elbow while still biasing the palm toward the thumb face somewhat;
    this lets the coarse-to-fine sweep find the SMALLEST yaw that still
    lets fingertips reach both intended faces, rather than assuming 45
    degrees is required."""
    thumb_n = FACE_NORMALS_LOCAL[assignment.thumb_face]
    finger_n = FACE_NORMALS_LOCAL[assignment.finger_face]
    t = np.clip(corner_yaw_deg / 45.0, 0.0, 1.0)
    v = (1.0 - t) * finger_n + t * (finger_n + thumb_n) / np.linalg.norm(finger_n + thumb_n)
    return v / np.linalg.norm(v)


def hand_corner_target(
    obj_pos: np.ndarray, obj_quat: np.ndarray, half_size: float, assignment: HandFaceAssignment,
    palm_standoff: float, height_offset: float = 0.0, corner_yaw_deg: float = 45.0,
    wrist_pitch_adjust: float = 0.0, wrist_roll_adjust: float = 0.0,
) -> HandCornerTarget:
    """Builds the PALM (not fingertip) position/orientation for one hand's
    corner pregrasp/final-contact pose (Section 6): palm sits OUTSIDE the
    corner edge along the diagonal, approaching straight at the edge, with
    the closing (thumb-vs-finger) axis tangent to the edge in the XY
    plane -- so thumb and fingers spread along the corner's own edge
    direction instead of along either face's normal individually.

    ``corner_yaw_deg`` (new this session, Section 2/6): 45 reproduces the
    prior session's fixed diagonal bisector exactly; smaller values bias
    the approach/palm-position direction toward the finger face's own
    normal (see corner_direction_local_at_yaw), asking less of wrist_yaw.
    ``wrist_pitch_adjust``/``wrist_roll_adjust`` (radians) apply a small
    extra rotation on top of the base orientation, independently per
    hand, matching Section 2's "wrist pitch/roll도 작은 범위에서 독립적으로
    탐색" -- these do NOT change the approach/standoff geometry, only the
    palm's final orientation, so the solver's soft orientation task has
    a slightly different target to reach without moving the hand."""
    R_obj = _quat_to_mat(obj_quat)
    corner_local = corner_direction_local_at_yaw(assignment, corner_yaw_deg)
    corner_edge_local = np.array([corner_local[0] * half_size, corner_local[1] * half_size, height_offset])
    corner_edge_world = obj_pos + R_obj @ corner_edge_local
    corner_dir_world = R_obj @ corner_local
    corner_dir_world = corner_dir_world / np.linalg.norm(corner_dir_world)

    approach_world = -corner_dir_world  # points INWARD, from outside the corner toward it
    # Tangent to the edge in the XY plane (perpendicular to corner_dir,
    # staying horizontal) -- the thumb/finger separation axis.
    closing_world = np.array([-corner_dir_world[1], corner_dir_world[0], 0.0])
    closing_norm = np.linalg.norm(closing_world)
    if closing_norm < 1e-6:
        closing_world = np.array([0.0, 1.0, 0.0])
    else:
        closing_world = closing_world / closing_norm

    palm_pos = corner_edge_world + corner_dir_world * palm_standoff
    palm_R = make_side_pinch_orientation(closing_world, approach_hint=approach_world)
    if wrist_pitch_adjust != 0.0 or wrist_roll_adjust != 0.0:
        # Small extra rotation about the palm's OWN local lateral (pitch)
        # and approach (roll) axes -- applied on the right so it composes
        # in the palm's local frame, not world frame.
        cp, sp = np.cos(wrist_pitch_adjust), np.sin(wrist_pitch_adjust)
        R_pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        cr, sr = np.cos(wrist_roll_adjust), np.sin(wrist_roll_adjust)
        R_roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        palm_R = palm_R @ R_pitch @ R_roll
    return HandCornerTarget(
        palm_pos=palm_pos, palm_R=palm_R, corner_edge_point_world=corner_edge_world,
        approach_world=approach_world, closing_world=closing_world,
    )


@dataclass
class PostureSeed:
    name: str
    q17: np.ndarray  # full [waist(3), left_arm(7), right_arm(7)] starting/rest config


def build_posture_seeds(stand_q17: np.ndarray) -> list[PostureSeed]:
    """Section 5: meaningful posture FAMILIES (not random seeds), each a
    deterministic, named perturbation of the standing rest pose in the
    solver's fixed 17-DoF [waist(3), L_shoulder_pitch/roll/yaw, L_elbow,
    L_wrist_roll/pitch/yaw, R_shoulder_pitch/roll/yaw, R_elbow, R_wrist_
    roll/pitch/yaw] ordering (task_config.LEFT_ARM_JOINTS/RIGHT_ARM_
    JOINTS order, confirmed directly against the MJCF). Magnitudes are
    modest (<=0.5 rad) deliberate deltas from the stand pose, not
    arbitrary -- exact values are refined by what the solver's own
    joint-limit-margin/collision results say, not assumed correct."""
    seeds = []

    def q(base=None):
        return (stand_q17.copy() if base is None else base.copy())

    seeds.append(PostureSeed("neutral_rest", q()))

    # shoulder-forward: increase shoulder_pitch (swings the upper arm
    # forward/up from the side) on both arms.
    v = q(); v[3] += 0.6; v[10] += 0.6
    seeds.append(PostureSeed("shoulder_forward", v))

    # shoulder-forward + elbow-extension: same shoulder change, PLUS
    # reduce elbow flexion toward straighter (elbow index 6/13 within the
    # 17-vector: waist(3)+shoulder_pitch,roll,yaw,elbow -> elbow is the
    # 4th arm DOF, global index 3+3=6 for left, 3+7+3=13 for right).
    v = q(); v[3] += 0.6; v[10] += 0.6; v[6] -= 0.5; v[13] -= 0.5
    seeds.append(PostureSeed("shoulder_forward_elbow_extension", v))

    # shoulder-diagonal-outward: shoulder_roll increases AWAY from the
    # body (left roll more positive, right roll more negative -- mirrored
    # per the stand pose's own +0.2/-0.2 sign convention) plus some yaw.
    v = q(); v[4] += 0.4; v[5] += 0.3; v[11] -= 0.4; v[12] -= 0.3
    seeds.append(PostureSeed("shoulder_diagonal_outward", v))

    # shoulder-diagonal-inward: opposite roll/yaw direction (arms angle
    # across toward the body's midline / the other hand's side).
    v = q(); v[4] -= 0.3; v[5] -= 0.3; v[11] += 0.3; v[12] += 0.3
    seeds.append(PostureSeed("shoulder_diagonal_inward", v))

    # small waist-yaw assisted: rotate the torso slightly toward the
    # diagonal (waist_yaw is coupled-vector index 0).
    v = q(); v[0] += 0.25
    seeds.append(PostureSeed("waist_yaw_assist", v))

    # small waist-pitch + shoulder-forward (waist_pitch is index 2).
    v = q(); v[2] += 0.15; v[3] += 0.5; v[10] += 0.5
    seeds.append(PostureSeed("waist_pitch_shoulder_forward", v))

    # wrist pre-rotated toward the corner (wrist_yaw is the 7th arm DOF,
    # global index 9 for left, 16 for right).
    v = q(); v[9] += 0.5; v[16] -= 0.5
    seeds.append(PostureSeed("wrist_prerotated", v))

    # left/right asymmetric: only the LEFT arm gets a shoulder-forward
    # +elbow-extension change, right stays neutral -- tests whether one
    # side needs a different family than the other (the current live
    # controller only ever moves both symmetrically).
    v = q(); v[3] += 0.6; v[6] -= 0.5
    seeds.append(PostureSeed("left_only_shoulder_forward", v))
    v = q(); v[10] += 0.6; v[13] -= 0.5
    seeds.append(PostureSeed("right_only_shoulder_forward", v))

    # Contact-Driven Four-Face Whole-Body Feasibility session, Section 4:
    # additional named families requested but missing from the prior
    # session's set.
    v = q(); v[0] -= 0.25  # small waist-yaw the OTHER direction
    seeds.append(PostureSeed("waist_yaw_assist_negative", v))

    v = q(); v[2] += 0.15  # small waist-pitch forward ALONE (no shoulder change)
    seeds.append(PostureSeed("waist_pitch_forward_only", v))

    # waist yaw + asymmetric shoulders (only right arm forward, waist
    # yaws to help bring the LEFT hand's reach around instead).
    v = q(); v[0] += 0.25; v[10] += 0.5
    seeds.append(PostureSeed("waist_yaw_asymmetric_shoulders", v))

    # waist pitch + shoulder-forward + elbow-extension, all three
    # combined (the prior session only combined pitch+shoulder, not
    # elbow too).
    v = q(); v[2] += 0.15; v[3] += 0.5; v[10] += 0.5; v[6] -= 0.4; v[13] -= 0.4
    seeds.append(PostureSeed("waist_pitch_shoulder_forward_elbow_extension", v))

    # wrist pre-aligned with a SMALL yaw (vs. wrist_prerotated's larger
    # 0.5 rad) -- Section 6's "wrist pitch/roll도 작은 범위에서 독립적으로
    # 탐색" companion at the seed level.
    v = q(); v[9] += 0.2; v[16] -= 0.2
    seeds.append(PostureSeed("wrist_prealigned_small_yaw", v))

    return seeds


def build_candidate_specific_seeds(candidate_name: str, stand_q17: np.ndarray) -> list[PostureSeed]:
    """Section 4: "C1-specific asymmetric posture"/"C2-specific
    asymmetric posture" -- unlike build_posture_seeds' generic families,
    these target the KNOWN-hard hand for each chirality specifically
    (C1's hard hand is LEFT -- corner_direction_local(C1.left) points
    +X+Y, away from this pipeline's natural +X-frontal approach; C2's
    hard hand is RIGHT, symmetric reasoning). Combines shoulder-forward +
    elbow-extension + a small waist-yaw assist toward that hand's own
    corner, all at once, specifically for the hard side only."""
    seeds = []
    v = stand_q17.copy()
    if candidate_name == "C1":
        v[0] += 0.2  # waist yaw assist toward the left hand's +X+Y corner
        v[3] += 0.6; v[6] -= 0.5  # left shoulder-forward + elbow-extension
        v[4] += 0.3  # left shoulder-roll outward
        seeds.append(PostureSeed("C1_specific_left_hard_hand", v))
    elif candidate_name == "C2":
        v[0] -= 0.2
        v[10] += 0.6; v[13] -= 0.5
        v[11] -= 0.3
        seeds.append(PostureSeed("C2_specific_right_hard_hand", v))
    return seeds


@dataclass
class FeasibilityResult:
    candidate: str
    seed_name: str
    success: bool
    left_pos_error: float
    left_ori_error: float
    right_pos_error: float
    right_ori_error: float
    joint_limit_margin: float
    wrist_yaw_margin: float  # min(left, right) wrist_yaw margin specifically (Section 5)
    elbow_margin: float  # min(left, right) elbow margin specifically
    collision_free: bool
    collision_pairs: list[tuple[str, str]]
    left_target_face_thumb: str
    left_target_face_finger: str
    right_target_face_thumb: str
    right_target_face_finger: str
    left_thumb_face_actual: str
    left_finger_face_actual: str
    right_thumb_face_actual: str
    right_finger_face_actual: str
    face_assignment_satisfied: bool  # Section 2: real success criterion, not palm orientation
    q17: np.ndarray
    corner_yaw_deg: tuple[float, float]  # (left, right) -- independent per Section 2
    waist_weight_used: float  # None (solver default, 6.0) reported as 6.0 for readability
    iterations: int


def measure_per_finger_local_offsets(env, side: str, thumb_abduct_pose: float) -> dict[str, np.ndarray]:
    """Contact-Driven Four-Face Whole-Body Feasibility session, Section 2:
    measures each of thumb/index/middle's own local (palm-frame) offset
    from the palm site, at the GRASP_ABDUCT preshape (thumb abducted
    toward its config extreme, index/middle at hand_synergy's open
    pose) -- unlike grasp_expert.BimanualSidePinchExpert._measure_
    fingertip_grasp_offset (which averages all three into one centroid
    for the UNCHANGED historical approach pipeline), this keeps each
    digit's own offset separate, using the CORRECTED fingertip TIP sites
    (whole_body_config.FINGERTIP_SITE_LOCAL_POS), so a per-digit face
    check reflects the real distal fingertip, not the frozen-axis bug
    the earlier centroid measurement was pinned to avoid disturbing."""
    model = env.model
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = env.data.qpos
    scratch.qvel[:] = 0.0
    finger_qpos_adr = env._left_finger_qpos_adr if side == "left" else env._right_finger_qpos_adr
    targets = hand_synergy.left_hand_targets(0.0) if side == "left" else hand_synergy.right_hand_targets(0.0)
    targets = np.array(targets, dtype=np.float64)
    targets[1] = thumb_abduct_pose
    scratch.qpos[finger_qpos_adr] = targets
    mujoco.mj_forward(model, scratch)
    palm_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_PALM_SITE if side == "left" else wbc.RIGHT_PALM_SITE)
    palm_pos = scratch.site_xpos[palm_site].copy()
    palm_R = scratch.site_xmat[palm_site].reshape(3, 3).copy()
    site_names = (
        (wbc.LEFT_THUMB_TIP_SITE, wbc.LEFT_INDEX_TIP_SITE, wbc.LEFT_MIDDLE_TIP_SITE) if side == "left"
        else (wbc.RIGHT_THUMB_TIP_SITE, wbc.RIGHT_INDEX_TIP_SITE, wbc.RIGHT_MIDDLE_TIP_SITE)
    )
    offsets = {}
    for key, site_name in zip(("thumb", "index", "middle"), site_names):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        world_pos = scratch.site_xpos[site_id].copy()
        offsets[key] = palm_R.T @ (world_pos - palm_pos)
    return offsets


def _classify_point_face(world_point: np.ndarray, obj_pos: np.ndarray, obj_quat: np.ndarray, ambiguity_margin: float = 0.2) -> str:
    """Same convention as grasp_expert.BimanualSidePinchExpert._classify_
    contact_face, but classifies a WORLD POINT's direction from the
    object center (not a contact normal) -- a valid proxy since a point
    genuinely near/against one face of a cube has its center-relative
    direction dominated by that face's own axis."""
    R_obj = _quat_to_mat(obj_quat)
    local_dir = R_obj.T @ (world_point - obj_pos)
    n = local_dir / (np.linalg.norm(local_dir) + 1e-12)
    abs_n = np.abs(n)
    order = np.argsort(abs_n)[::-1]
    if abs_n[order[0]] - abs_n[order[1]] < ambiguity_margin:
        return "EDGE_OR_CORNER_AMBIGUOUS"
    axis = order[0]
    sign = n[axis] > 0
    if axis == 0:
        return "FACE_POS_X" if sign else "FACE_NEG_X"
    elif axis == 1:
        return "FACE_POS_Y" if sign else "FACE_NEG_Y"
    else:
        return "FACE_TOP" if sign else "FACE_BOTTOM"


_IGNORE_COLLISION_PREFIXES_SAME_BODY = None  # placeholder, unused


def _scan_unwanted_collisions(model, data, object_body_id: int) -> list[tuple[str, str]]:
    """Broad self/robot-environment collision scan at the solved pose
    (Section 4: hand-hand, thumb-thumb, arm-torso, robot-table, premature
    robot-object contact). Table/torso/leg body names are matched by
    prefix since this project's MJCF consistently names them that way
    (confirmed via _hand_hand_contact's identical pattern in
    grasp_expert.py)."""
    pairs = []
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
        if b1 == b2:
            continue
        is_left = n1.startswith("left_hand") or n1.startswith("left_wrist")
        is_right = n2.startswith("right_hand") or n2.startswith("right_wrist")
        is_left2 = n2.startswith("left_hand") or n2.startswith("left_wrist")
        is_right2 = n1.startswith("right_hand") or n1.startswith("right_wrist")
        hand_hand = (is_left and is_right) or (is_left2 and is_right2)
        involves_object = b1 == object_body_id or b2 == object_body_id
        involves_table = "table" in n1 or "table" in n2
        involves_torso = "torso" in n1 or "torso" in n2 or "pelvis" in n1 or "pelvis" in n2
        arm_arm_same_side = False
        if not (hand_hand or involves_object or involves_table or involves_torso):
            continue
        pairs.append((n1, n2))
    return pairs


_WRIST_YAW_IDX = (9, 16)  # global 17-vector index of L/R wrist_yaw
_ELBOW_IDX = (6, 13)


def evaluate_static_pose(
    coupled_ik: CoupledBilateralIK, scratch_data: mujoco.MjData, model: mujoco.MjModel,
    object_body_id: int, obj_pos: np.ndarray, obj_quat: np.ndarray,
    candidate: CandidateAssignment, seed: PostureSeed,
    left_target: HandCornerTarget, right_target: HandCornerTarget,
    left_finger_offsets: dict[str, np.ndarray], right_finger_offsets: dict[str, np.ndarray],
    limit_margin_min: float = 0.02,
    finger_qpos_adr: tuple[np.ndarray, np.ndarray] | None = None,
    finger_open_targets: tuple[np.ndarray, np.ndarray] | None = None,
    ori_task_weight: float = 0.5,
    require_orientation: bool = False,
    wrist_yaw_rest_gain_boost: float = 0.3,
    corner_yaw_deg: tuple[float, float] = (45.0, 45.0),
    waist_weight: float | None = None,
) -> FeasibilityResult:
    """Runs ONE static solve (Section 2/4/5): seeds the scratch qpos AND
    the null-space rest_q from ``seed``, solves for the given corner
    targets, then re-runs mj_forward at the solution and scans for
    unwanted collisions (Section 4's collision checklist) AND classifies
    each digit's actual fingertip face (Section 2's real success
    criterion, not palm orientation).

    ``ori_task_weight``/``require_orientation`` default to a SOFT
    orientation task (Section 2: "palm orientation은 soft task로 설정") --
    the solver is free to deviate from the exact corner_yaw_deg
    orientation if that gets fingertip position closer; the ACTUAL
    resulting orientation and fingertip-face alignment are measured and
    reported regardless.

    ``wrist_yaw_rest_gain_boost`` (Section 5's explicit wrist_yaw-
    saturation cost): added ONLY to the wrist_yaw DOF's null-space
    rest_gain (index 9/16), biasing the solver to leave wrist_yaw nearer
    ``seed.q17``'s own value and lean on shoulder/waist instead --
    implemented via CoupledBilateralIK.solve's existing per-DOF
    ``rest_gain`` array support (no solver code changes needed).

    ``finger_qpos_adr``/``finger_open_targets`` (left, right): when given,
    fingers are set to a SAFE OPEN pose (matching the live controller's
    own THUMB_CLEARANCE_PRESHAPE convention -- thumb abducted, index/
    middle open) before the collision scan, instead of whatever arbitrary
    finger qpos the scratch data happened to hold. Without this, a small
    palm_standoff can report an "object-finger collision" that is really
    just an artifact of unshaped fingers, not a genuine standoff/reach
    infeasibility.

    ``waist_weight`` (added after a user question caught a real gap):
    ``CoupledBilateralIK`` defaults to a 6x DLS task-space penalty on the
    waist columns (see coupled_ik.py's own docstring -- a deliberate
    anti-overuse choice for the LIVE grasp controller). Every posture-
    family SEED in this module biases the null-space *rest_q* toward
    more waist rotation, but rest_q only sets where the solver is pulled
    in the LEFTOVER null space -- it does not lower the primary task's
    6x cost of actually moving the waist there. Passing a lower
    ``waist_weight`` here overrides ``coupled_ik.solve``'s per-call
    ``joint_weight`` (the solver's constructor-time default is left
    untouched for any other caller), letting the search actually verify
    whether genuine waist twist relieves wrist_yaw/elbow saturation, not
    just whether a seed NEAR more waist twist does under the default 6x
    penalty. None reproduces the solver's own default exactly."""
    coupled_ik._set_q(scratch_data, seed.q17)
    rest_gain = np.full(17, 0.15)
    rest_gain[list(_WRIST_YAW_IDX)] += wrist_yaw_rest_gain_boost
    joint_weight = None
    if waist_weight is not None:
        joint_weight = coupled_ik.joint_weight.copy()
        joint_weight[0:3] = waist_weight
    result = coupled_ik.solve(
        scratch_data,
        left_target.palm_pos, left_target.palm_R,
        right_target.palm_pos, right_target.palm_R,
        rest_q=seed.q17,
        rest_gain=rest_gain,
        ori_task_weight=ori_task_weight,
        require_orientation=require_orientation,
        limit_margin_min=limit_margin_min,
        joint_weight=joint_weight,
    )
    q_full = np.concatenate([result.waist_q, result.left_q, result.right_q])
    coupled_ik._set_q(scratch_data, q_full)
    if finger_qpos_adr is not None and finger_open_targets is not None:
        scratch_data.qpos[finger_qpos_adr[0]] = finger_open_targets[0]
        scratch_data.qpos[finger_qpos_adr[1]] = finger_open_targets[1]
    mujoco.mj_forward(model, scratch_data)
    collisions = _scan_unwanted_collisions(model, scratch_data, object_body_id)

    # Fingertip-face verification (Section 2): use the ACTUAL solved palm
    # pose (not the target) plus the pre-measured local per-digit offsets
    # to get world fingertip positions, then classify.
    left_palm_site = coupled_ik.left_site
    right_palm_site = coupled_ik.right_site
    left_palm_pos = scratch_data.site_xpos[left_palm_site].copy()
    left_palm_R = scratch_data.site_xmat[left_palm_site].reshape(3, 3).copy()
    right_palm_pos = scratch_data.site_xpos[right_palm_site].copy()
    right_palm_R = scratch_data.site_xmat[right_palm_site].reshape(3, 3).copy()
    left_thumb_world = left_palm_pos + left_palm_R @ left_finger_offsets["thumb"]
    left_finger_world = left_palm_pos + left_palm_R @ (0.5 * (left_finger_offsets["index"] + left_finger_offsets["middle"]))
    right_thumb_world = right_palm_pos + right_palm_R @ right_finger_offsets["thumb"]
    right_finger_world = right_palm_pos + right_palm_R @ (0.5 * (right_finger_offsets["index"] + right_finger_offsets["middle"]))
    left_thumb_face = _classify_point_face(left_thumb_world, obj_pos, obj_quat)
    left_finger_face = _classify_point_face(left_finger_world, obj_pos, obj_quat)
    right_thumb_face = _classify_point_face(right_thumb_world, obj_pos, obj_quat)
    right_finger_face = _classify_point_face(right_finger_world, obj_pos, obj_quat)
    face_ok = (
        left_thumb_face == candidate.left.thumb_face and left_finger_face == candidate.left.finger_face
        and right_thumb_face == candidate.right.thumb_face and right_finger_face == candidate.right.finger_face
    )

    margins = np.minimum(q_full - coupled_ik.joint_low, coupled_ik.joint_high - q_full)
    wrist_yaw_margin = float(min(margins[_WRIST_YAW_IDX[0]], margins[_WRIST_YAW_IDX[1]]))
    elbow_margin = float(min(margins[_ELBOW_IDX[0]], margins[_ELBOW_IDX[1]]))

    return FeasibilityResult(
        candidate=candidate.name, seed_name=seed.name,
        success=result.success and not collisions and face_ok,
        left_pos_error=result.left_pos_error, left_ori_error=result.left_ori_error,
        right_pos_error=result.right_pos_error, right_ori_error=result.right_ori_error,
        joint_limit_margin=result.joint_limit_margin, wrist_yaw_margin=wrist_yaw_margin, elbow_margin=elbow_margin,
        collision_free=not collisions, collision_pairs=collisions,
        left_target_face_thumb=candidate.left.thumb_face, left_target_face_finger=candidate.left.finger_face,
        right_target_face_thumb=candidate.right.thumb_face, right_target_face_finger=candidate.right.finger_face,
        left_thumb_face_actual=left_thumb_face, left_finger_face_actual=left_finger_face,
        right_thumb_face_actual=right_thumb_face, right_finger_face_actual=right_finger_face,
        face_assignment_satisfied=face_ok,
        q17=q_full, corner_yaw_deg=corner_yaw_deg,
        waist_weight_used=waist_weight if waist_weight is not None else float(coupled_ik.joint_weight[0]),
        iterations=result.iterations,
    )


def score_result(r: FeasibilityResult) -> tuple:
    """Contact-Driven Four-Face Whole-Body Feasibility session, Section
    12's explicit filter ORDER, as a sortable tuple (lower is better):
    (1) collision, (2) four-face assignment satisfied (Section 2's real
    success criterion -- NOT palm orientation), (3) fingertip position
    accuracy, (4) wrist_yaw/elbow margin (penalize a hand pinned at its
    limit even if position happens to look small), (5+) remaining
    reported-but-not-yet-scored metrics (waist/wrist travel, standoff
    naturalness) are left as diagnostics only, per the same reasoning as
    the prior session's score_result. A candidate where only ONE hand
    converges (Section 12: "한 손만 잘 수렴하고 다른 손이 limit에 걸린
    후보는 탈락시킨다") is penalized via the MAX (not mean/sum) of the
    two hands' errors, so an excellent left + terrible right cannot hide
    behind a good average."""
    collision_penalty = 0 if r.collision_free else 1
    face_penalty = 0 if r.face_assignment_satisfied else 1
    worst_pos_error = max(r.left_pos_error, r.right_pos_error)
    worst_ori_error = max(r.left_ori_error, r.right_ori_error)
    limit_penalty = -min(r.wrist_yaw_margin, r.elbow_margin)  # lower (more negative margin headroom) is worse
    return (collision_penalty, face_penalty, worst_pos_error + 0.3 * worst_ori_error, limit_penalty)
