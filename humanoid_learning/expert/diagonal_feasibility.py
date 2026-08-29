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


def hand_corner_target(
    obj_pos: np.ndarray, obj_quat: np.ndarray, half_size: float, assignment: HandFaceAssignment,
    palm_standoff: float, height_offset: float = 0.0,
) -> HandCornerTarget:
    """Builds the PALM (not fingertip) position/orientation for one hand's
    corner pregrasp/final-contact pose (Section 6): palm sits OUTSIDE the
    corner edge along the diagonal, approaching straight at the edge, with
    the closing (thumb-vs-finger) axis tangent to the edge in the XY
    plane -- so thumb and fingers spread along the corner's own edge
    direction instead of along either face's normal individually."""
    R_obj = _quat_to_mat(obj_quat)
    corner_local = corner_direction_local(assignment)  # e.g. [.707,.707,0]
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
    collision_free: bool
    collision_pairs: list[tuple[str, str]]
    left_target_face_thumb: str
    left_target_face_finger: str
    right_target_face_thumb: str
    right_target_face_finger: str
    q17: np.ndarray
    iterations: int


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


def evaluate_static_pose(
    coupled_ik: CoupledBilateralIK, scratch_data: mujoco.MjData, model: mujoco.MjModel,
    object_body_id: int, candidate: CandidateAssignment, seed: PostureSeed,
    left_target: HandCornerTarget, right_target: HandCornerTarget,
    limit_margin_min: float = 0.02,
    finger_qpos_adr: tuple[np.ndarray, np.ndarray] | None = None,
    finger_open_targets: tuple[np.ndarray, np.ndarray] | None = None,
) -> FeasibilityResult:
    """Runs ONE static solve (Section 4/5): seeds the scratch qpos AND the
    null-space rest_q from ``seed``, solves for the given corner targets,
    then re-runs mj_forward at the solution and scans for unwanted
    collisions (Section 4's collision checklist).

    ``finger_qpos_adr``/``finger_open_targets`` (left, right): when given,
    fingers are set to a SAFE OPEN pose (matching the live controller's
    own THUMB_CLEARANCE_PRESHAPE convention -- thumb abducted, index/
    middle open) before the collision scan, instead of whatever arbitrary
    finger qpos the scratch data happened to hold. Without this, a small
    palm_standoff can report an "object-finger collision" that is really
    just an artifact of unshaped fingers, not a genuine standoff/reach
    infeasibility."""
    coupled_ik._set_q(scratch_data, seed.q17)
    result = coupled_ik.solve(
        scratch_data,
        left_target.palm_pos, left_target.palm_R,
        right_target.palm_pos, right_target.palm_R,
        rest_q=seed.q17,
        rest_gain=0.15,
        ori_task_weight=1.0,
        require_orientation=True,
        limit_margin_min=limit_margin_min,
    )
    q_full = np.concatenate([result.waist_q, result.left_q, result.right_q])
    coupled_ik._set_q(scratch_data, q_full)
    if finger_qpos_adr is not None and finger_open_targets is not None:
        scratch_data.qpos[finger_qpos_adr[0]] = finger_open_targets[0]
        scratch_data.qpos[finger_qpos_adr[1]] = finger_open_targets[1]
    mujoco.mj_forward(model, scratch_data)
    collisions = _scan_unwanted_collisions(model, scratch_data, object_body_id)
    return FeasibilityResult(
        candidate=candidate.name, seed_name=seed.name, success=result.success and not collisions,
        left_pos_error=result.left_pos_error, left_ori_error=result.left_ori_error,
        right_pos_error=result.right_pos_error, right_ori_error=result.right_ori_error,
        joint_limit_margin=result.joint_limit_margin, collision_free=not collisions,
        collision_pairs=collisions,
        left_target_face_thumb=candidate.left.thumb_face, left_target_face_finger=candidate.left.finger_face,
        right_target_face_thumb=candidate.right.thumb_face, right_target_face_finger=candidate.right.finger_face,
        q17=q_full, iterations=result.iterations,
    )


def score_result(r: FeasibilityResult) -> tuple:
    """Section 5's explicit filter ORDER, as a sortable tuple (lower is
    better): (1) collision, (2) joint-limit safety, (3-ish) combined
    pos/ori error as the finest-grained tiebreaker among otherwise-safe
    candidates. Face-assignment feasibility (item 3 in the spec's
    priority list) is enforced by construction (each seed is solved
    AGAINST that candidate's own targets), and palm-standoff (item 4) is
    a fixed input, not a scored output -- so this tuple covers items
    1/2/5 directly, with 6/7 (posture naturalness/wrist travel) left as
    reported-but-not-yet-scored metrics (Section 5 note below)."""
    collision_penalty = 0 if r.collision_free else 1
    limit_penalty = 0 if r.joint_limit_margin >= 0.02 else 1
    combined_error = r.left_pos_error + r.right_pos_error + 0.3 * (r.left_ori_error + r.right_ori_error)
    return (collision_penalty, limit_penalty, combined_error)
