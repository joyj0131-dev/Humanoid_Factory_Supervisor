"""Reachable Contact Manifold Grasp Planner session (31st Phase 4 Grasp
Track session).

30th-session finding, corrected: fixing index/middle at their currently-
achieved position and sending the thumb to a single hand-coded (x, y, z)
target is NOT the same claim as "the thumb cannot physically reach a
good opposing contact." This module removes that over-constraint: EVERY
fingertip's contact position is a byproduct of a joint-space optimization
over BOTH arms' shoulder/elbow/wrist and BOTH hands' thumb/index/middle
joints, subject to soft "land on this object face, inside its margin"
costs -- not a fixed target point. Which two-cm patch of the face each
finger actually lands on is chosen by the solver, not hand-specified.

This module is READ-ONLY / static-only: it builds its own scratch
MjData, is never imported by BimanualSidePinchExpert's control loop, and
does not move env.data. tripod_closure_planner.py's Stage-1 audit
functions (true_fingertip_world, audit_hand_geometry) are reused
unchanged; its Stage-2 fixed-target solvers (solve_finger_ik,
solve_hand_closure_ik, plan_tripod_targets) are LEFT UNTOUCHED and kept
covered by their own test file as a preserved negative result -- this
module does not replace them, it supersedes their approach for the
"is a better six-fingertip grasp reachable at all" question.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.optimize import nnls

from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.expert.tripod_closure_planner import (
    FINGER_JOINT_SUFFIXES, true_fingertip_world,
)

FINGERS = ("thumb", "index", "middle")
_FINGER_SLOT = {"thumb": (0, 1, 2), "middle": (3, 4), "index": (5, 6)}
ARM_JOINT_SUFFIXES = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
N_ARM = 7
N_FINGER = 7
N_PER_HAND = N_ARM + N_FINGER  # 14
N_TOTAL = 2 * N_PER_HAND  # 28, left then right


# ---------------------------------------------------------------------
# Decision-vector <-> qpos plumbing
# ---------------------------------------------------------------------

@dataclass
class HandDofs:
    side: str
    arm_qpos_adr: np.ndarray   # 7: shoulder x3, elbow, wrist x3
    finger_qpos_adr: np.ndarray  # 7: thumb x3, middle x2, index x2
    arm_jids: list
    finger_jids: list

    @property
    def qpos_adr(self) -> np.ndarray:
        return np.concatenate([self.arm_qpos_adr, self.finger_qpos_adr])

    @property
    def jnt_range(self) -> np.ndarray:
        jids = self.arm_jids + self.finger_jids
        return np.array([self.model.jnt_range[j] for j in jids])


def build_hand_dofs(env, side: str) -> HandDofs:
    model = env.model
    arm_qpos_adr = env._arm_qpos_adr[:7] if side == "left" else env._arm_qpos_adr[7:]
    finger_qpos_adr = env._left_finger_qpos_adr if side == "left" else env._right_finger_qpos_adr
    arm_names = [f"{side}_{n}_joint" for n in ARM_JOINT_SUFFIXES]
    finger_names = [f"{side}_hand_{n}_joint" for n in FINGER_JOINT_SUFFIXES]
    arm_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in arm_names]
    finger_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in finger_names]
    hd = HandDofs(side=side, arm_qpos_adr=arm_qpos_adr, finger_qpos_adr=finger_qpos_adr,
                  arm_jids=arm_jids, finger_jids=finger_jids)
    hd.model = model
    return hd


def _collision_geom_of_body(model: mujoco.MjModel, body_name: str) -> int:
    """[33rd session] Finds the ACTUAL collision geom (contype==1) for a
    body -- every hand link in this model has two geoms at identical
    pose (an even-indexed contype=0/conaffinity=0 VISUAL duplicate and
    an odd-indexed contype=1/conaffinity=1 COLLISION geom); using the
    visual geom's id would silently query a geom real physics never
    touches. Queried by scanning contype, never guessed by index
    parity or name."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == bid and model.geom_contype[g] == 1:
            return g
    raise ValueError(f"no collision geom found for body {body_name!r}")


@dataclass
class PlannerContext:
    env: object
    scratch: mujoco.MjData
    obj_pos: np.ndarray
    obj_R: np.ndarray
    half: float
    hands: dict  # side -> HandDofs
    x0: np.ndarray  # 28-dim reference pose (CONTACT_ACQUIRE-end), for regularization
    jnt_lo: np.ndarray
    jnt_hi: np.ndarray
    obj_geom_id: int
    table_geom_id: int
    tip_collision_geom: dict  # (side, finger) -> collision geom id
    proximal_collision_geom: dict = field(default_factory=dict)  # [session 34] (side, finger) -> list of non-tip collision geom ids
    use_waist: bool = False
    waist_qpos_adr: np.ndarray | None = None
    waist_x0: np.ndarray | None = None
    waist_lo: np.ndarray | None = None
    waist_hi: np.ndarray | None = None


WAIST_SOFT_RANGE = 0.15  # rad, small soft deviation band around the canonical stand-pose waist value (Section 10)


def enable_waist(ctx: PlannerContext) -> PlannerContext:
    """[33rd session, Section 10] Adds the 3 waist joints (yaw/roll/
    pitch) as GLOBAL (not per-hand) decision variables, appended after
    the existing 28. Only invoked when 0/14 candidates are statically
    feasible with the mesh-aware target alone (Section 10's explicit
    gate) -- never combined with widening topology/posture search or
    collision tolerance in the same comparison. Pelvis/legs are not
    touched (this is the fixed-base grasp env; there is no pelvis/leg
    DOF to begin with) and this does not become a whole-body reach:
    the waist range is soft-bounded to +-WAIST_SOFT_RANGE rad around
    the CURRENT (canonical stand-pose) waist qpos, not its full
    mechanical range."""
    model = ctx.env.model
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in wbc.WAIST_JOINTS]
    qpos_adr = np.array([model.jnt_qposadr[j] for j in jids])
    waist_x0 = ctx.scratch.qpos[qpos_adr].copy()
    mech_lo = np.array([model.jnt_range[j][0] for j in jids])
    mech_hi = np.array([model.jnt_range[j][1] for j in jids])
    waist_lo = np.maximum(mech_lo, waist_x0 - WAIST_SOFT_RANGE)
    waist_hi = np.minimum(mech_hi, waist_x0 + WAIST_SOFT_RANGE)
    ctx.use_waist = True
    ctx.waist_qpos_adr = qpos_adr
    ctx.waist_x0 = waist_x0
    ctx.waist_lo = waist_lo
    ctx.waist_hi = waist_hi
    return ctx


def build_context(env, expert) -> PlannerContext:
    model = env.model
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = env.data.qpos
    mujoco.mj_forward(model, scratch)

    obj_pos = env.data.qpos[env._object_qpos_adr:env._object_qpos_adr + 3].copy()
    obj_quat = env.data.qpos[env._object_qpos_adr + 3:env._object_qpos_adr + 7].copy()
    obj_R = np.zeros(9)
    mujoco.mju_quat2Mat(obj_R, obj_quat)
    obj_R = obj_R.reshape(3, 3)

    hands = {side: build_hand_dofs(env, side) for side in ("left", "right")}
    qpos_adr_all = np.concatenate([hands["left"].qpos_adr, hands["right"].qpos_adr])
    x0 = scratch.qpos[qpos_adr_all].copy()
    ranges = np.vstack([hands["left"].jnt_range, hands["right"].jnt_range])

    obj_body = env._object_body_id
    obj_geom_id = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == obj_body)
    table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    table_geom_id = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == table_body)
    tip_collision_geom = {}
    for side in ("left", "right"):
        for finger in FINGERS:
            site = getattr(wbc, f"{'LEFT' if side == 'left' else 'RIGHT'}_{finger.upper()}_TIP_SITE")
            body_name = wbc.FINGERTIP_SITE_BODIES[site]
            tip_collision_geom[(side, finger)] = _collision_geom_of_body(model, body_name)

    # [session 34] non-tip ("proximal") link geoms per finger -- the root
    # cause of this session's largest forbidden penetration (thumb_1_link,
    # the middle-knuckle link, NOT the true fingertip thumb_2_link)
    proximal_collision_geom = {}
    for side in ("left", "right"):
        proximal_collision_geom[(side, "thumb")] = [
            _collision_geom_of_body(model, f"{side}_hand_thumb_0_link"),
            _collision_geom_of_body(model, f"{side}_hand_thumb_1_link"),
        ]
        proximal_collision_geom[(side, "index")] = [_collision_geom_of_body(model, f"{side}_hand_index_0_link")]
        proximal_collision_geom[(side, "middle")] = [_collision_geom_of_body(model, f"{side}_hand_middle_0_link")]

    return PlannerContext(
        env=env, scratch=scratch, obj_pos=obj_pos, obj_R=obj_R, half=env.config.object_half_size,
        hands=hands, x0=x0, jnt_lo=ranges[:, 0], jnt_hi=ranges[:, 1],
        obj_geom_id=obj_geom_id, table_geom_id=table_geom_id, tip_collision_geom=tip_collision_geom,
        proximal_collision_geom=proximal_collision_geom,
    )


def mesh_signed_distance(ctx: PlannerContext, side: str, finger: str) -> tuple[float, np.ndarray]:
    """[33rd session] The ACTUAL fingertip-mesh-to-object signed distance
    via mujoco.mj_geomDistance -- positive when separated, negative when
    the real collision geometry penetrates. This replaces the 31st/32nd
    session's site-position-vs-face-plane proxy, which this session
    measured to disagree with the real mesh by a CONFIGURATION-DEPENDENT
    amount (-19.5mm to +4.7mm across two tested poses, not a fixed
    calibration constant) -- see measure_fingertip_support_offset's
    docstring for the full measurement."""
    fromto = np.zeros(6)
    tip_geom = ctx.tip_collision_geom[(side, finger)]
    dist = mujoco.mj_geomDistance(ctx.env.model, ctx.scratch, tip_geom, ctx.obj_geom_id, 1.0, fromto)
    return float(dist), fromto


def _write_x(ctx: PlannerContext, x: np.ndarray) -> None:
    left_adr = ctx.hands["left"].qpos_adr
    right_adr = ctx.hands["right"].qpos_adr
    ctx.scratch.qpos[left_adr] = x[:N_PER_HAND]
    ctx.scratch.qpos[right_adr] = x[N_PER_HAND:N_TOTAL]
    if ctx.use_waist and len(x) > N_TOTAL:
        ctx.scratch.qpos[ctx.waist_qpos_adr] = x[N_TOTAL:N_TOTAL + 3]
    mujoco.mj_forward(ctx.env.model, ctx.scratch)


def x0_with_waist(ctx: PlannerContext) -> np.ndarray:
    """x0 extended with the waist's current (canonical) value -- callers
    that enable_waist() should use this instead of ctx.x0 directly for
    the initial/regularization vector."""
    assert ctx.use_waist
    return np.concatenate([ctx.x0, ctx.waist_x0])


def _tip_world(ctx: PlannerContext, side: str, finger: str) -> np.ndarray:
    site = getattr(wbc, f"{'LEFT' if side == 'left' else 'RIGHT'}_{finger.upper()}_TIP_SITE")
    return true_fingertip_world(ctx.env.model, ctx.scratch, site)


# ---------------------------------------------------------------------
# Topology: which object face each finger targets, derived from the
# CURRENT natural-approach geometry (measured, not guessed by name) --
# index/middle approach one face, thumb the opposing one, generalized to
# whichever axis the real approach data shows (Stage 1 audit already
# established this is the object's local X axis for the canonical
# yaw=0 side-pinch geometry).
# ---------------------------------------------------------------------

@dataclass
class Topology:
    name: str
    face_axis: dict  # finger -> axis index (0=x,1=y,2=z)
    face_sign: dict   # finger -> +1/-1


def derive_primary_topology(ctx: PlannerContext) -> Topology:
    """Measures the CURRENT (natural-approach) fingertip object-local
    position for one hand and reads off which axis/sign each finger is
    already closest to -- the topology is a MEASURED fact, not a naming
    guess."""
    side = "left"
    axes = {}
    signs = {}
    for finger in FINGERS:
        local = ctx.obj_R.T @ (_tip_world(ctx, side, finger) - ctx.obj_pos)
        axis = int(np.argmax(np.abs(local)))
        axes[finger] = axis
        signs[finger] = 1.0 if local[axis] > 0 else -1.0
    return Topology(name="measured_primary", face_axis=axes, face_sign=signs)


def mirrored_topology(base: Topology) -> Topology:
    """Alternative candidate: swap which face the thumb vs index/middle
    target (multi-start topology diversity, Section 7's 'mirrored
    family') -- kept purely as an additional local-search start, not
    assumed correct."""
    axes = dict(base.face_axis)
    signs = dict(base.face_sign)
    signs["thumb"] = -signs["thumb"]
    signs["index"] = -signs["index"]
    signs["middle"] = -signs["middle"]
    return Topology(name="mirrored", face_axis=axes, face_sign=signs)


# ---------------------------------------------------------------------
# Session 34: face-aware contact assignment + actual normal alignment.
#
# ROOT CAUSE FOUND THIS SESSION: mesh_signed_distance() (33rd session)
# returns the WHOLE-BOX nearest distance from a fingertip geom to the
# object -- it has NO dependency on which face was "assigned" by
# Topology. Driving that whole-box distance to 0 (the 33rd session's
# entire face-plane residual, W_MESH_FACE * dist) can be satisfied by
# touching ANY face, edge, or corner that happens to be geometrically
# nearest -- not necessarily the one the topology intended. This is
# exactly why measured_primary and mirrored (which only differ in
# assigned face_sign) produced BYTE-IDENTICAL 33rd-session results: the
# objective never looked at face_sign at all once mesh_signed_distance
# replaced the old site-vs-face-plane term. Fixed below by reintroducing
# an explicit face-PLANE-coordinate residual computed on the REAL
# mj_geomDistance near-contact point (not the old analytic site), plus
# an actual-geometry normal-alignment term and post-hoc wrong-face/
# edge-corner classification.
# ---------------------------------------------------------------------

FACE_DEFS = (
    ("FACE_POS_X", 0, 1.0), ("FACE_NEG_X", 0, -1.0),
    ("FACE_POS_Y", 1, 1.0), ("FACE_NEG_Y", 1, -1.0),
    ("FACE_TOP", 2, 1.0), ("FACE_BOTTOM", 2, -1.0),
)
FACE_PLANE_TOL = 0.005    # m, "on this face's plane" numerical band (same order as face_tol elsewhere)
EDGE_TOL = 0.004          # m, how close to a SECOND face plane counts as edge/corner (ambiguous face)
NORMAL_ANGLE_TOL_DEG = 5.0  # deg, numerical-convergence band for "actually the assigned face", not a friction cone


def _face_name(axis: int, sign: float) -> str:
    for name, a, s in FACE_DEFS:
        if a == axis and s == sign:
            return name
    raise ValueError(f"no face for axis={axis} sign={sign}")


def classify_nearest_face(local: np.ndarray, half: float) -> dict:
    """[Session 34] Given ANY point in object-local coordinates (from a
    real mj_geomDistance near-contact point, never a guess), finds which
    of the box's 6 faces it is geometrically CLOSEST to -- by minimum
    gap |half - sign*local[axis]| over all 6 (axis, sign) combinations --
    and reports whether the point is actually within FACE_PLANE_TOL of
    that face's plane, whether it is far enough from a SECOND face plane
    to not be an edge/corner (EDGE_TOL), and that face's outward normal
    in OBJECT-LOCAL frame. Face identity is determined ENTIRELY from the
    numeric coordinate -- never assumed from a finger name or which face
    a Topology intended."""
    best = None
    for name, axis, sign in FACE_DEFS:
        gap = abs(half - sign * local[axis])
        if best is None or gap < best[0]:
            best = (gap, name, axis, sign)
    gap, name, axis, sign = best
    other_axes = [a for a in range(3) if a != axis]
    is_edge_or_corner = any(abs(half - abs(local[a])) < EDGE_TOL for a in other_axes)
    normal_local = np.zeros(3)
    normal_local[axis] = sign
    return dict(face=name, axis=axis, sign=sign, plane_gap=float(gap),
                on_plane=bool(gap < FACE_PLANE_TOL), is_edge_or_corner=bool(is_edge_or_corner),
                normal_local=normal_local)


def face_assignment_metrics(ctx: "PlannerContext", side: str, finger: str, topology: Topology) -> dict:
    """[Session 34] Full face-aware contact validity report for one
    fingertip at the CURRENT scratch pose: the real mj_geomDistance
    near-contact point on the object (never the analytic site), which
    face that point is geometrically closest to (classify_nearest_face),
    whether that matches the Topology-ASSIGNED face (wrong_face), whether
    the point sits on an edge/corner (ambiguous between two faces), the
    face-plane coordinate error against the assigned face specifically,
    the tangential (face-interior) margin violation, and the angular
    error between the assigned face's outward normal and the REAL
    geometric normal at the nearest-face classification (0 deg iff the
    contact is genuinely on the assigned face's plane and interior)."""
    dist, fromto = mesh_signed_distance(ctx, side, finger)
    near_on_object = fromto[3:6]
    local = ctx.obj_R.T @ (near_on_object - ctx.obj_pos)
    axis = topology.face_axis[finger]
    sign = topology.face_sign[finger]
    nearest = classify_nearest_face(local, ctx.half)
    face_plane_error = float(local[axis] - sign * ctx.half)
    other_axes = [a for a in range(3) if a != axis]
    tangential_violation = float(max(0.0, max(abs(local[a]) for a in other_axes) - (ctx.half - MARGIN)))
    wrong_face = bool(nearest["axis"] != axis or nearest["sign"] != sign)
    assigned_normal_local = np.zeros(3)
    assigned_normal_local[axis] = sign
    normal_dot = float(np.clip(np.dot(assigned_normal_local, nearest["normal_local"]), -1.0, 1.0))
    normal_angle_deg = float(np.degrees(np.arccos(normal_dot)))
    return dict(
        side=side, finger=finger, dist=float(dist),
        object_local_point=[float(v) for v in local],
        assigned_axis=axis, assigned_sign=sign, assigned_face=_face_name(axis, sign),
        nearest_face=nearest["face"], nearest_axis=nearest["axis"], nearest_sign=nearest["sign"],
        nearest_plane_gap=nearest["plane_gap"],
        face_plane_error=face_plane_error, tangential_violation=tangential_violation,
        is_edge_or_corner=nearest["is_edge_or_corner"], wrong_face=wrong_face,
        normal_dot=normal_dot, normal_angle_deg=normal_angle_deg,
        assigned_normal_world=(ctx.obj_R @ assigned_normal_local).tolist(),
        actual_normal_world=(ctx.obj_R @ nearest["normal_local"]).tolist(),
    )


# ---------------------------------------------------------------------
# Section 5 (33rd session): TRUE fingertip support-offset measurement.
# Read-only diagnostic -- moves ONLY the scratch object's qpos along one
# world axis via a local Newton/fixed-point iteration on the REAL
# mj_geomDistance, never touches finger/arm qpos or the live env.
# ---------------------------------------------------------------------

def measure_fingertip_support_offset(ctx: PlannerContext, side: str, finger: str, approach_axis: int,
                                      approach_sign: float, max_iters: int = 30) -> dict:
    """Finds, via a LOCAL fixed-point search (NOT a wide-range binary
    search -- a first attempt at this using bisection over a wide
    [-0.5, 0.5] object-position range converged to spurious, distant
    roots since mj_geomDistance is not globally monotonic; this instead
    starts from the object placed as if the TRUE FINGERTIP SITE point
    were already touching and repeatedly shifts the object by exactly
    the measured real mesh distance -- valid because the function is
    monotonic in a small neighborhood of the actual contact), the object
    position along ``approach_axis`` (in ``approach_sign`` direction)
    where the REAL collision geom (not the site proxy) has
    mj_geomDistance == 0 against the object. Returns the support offset
    (site position minus true-touch position, along the approach axis,
    signed positive when the site sits BEYOFND the real mesh surface --
    i.e. the site would already be inside the object by that amount when
    the real mesh is exactly touching) and the final measured distance
    (should be ~0, confirming convergence). This function only moves the
    SCRATCH object qpos -- finger/arm qpos and env.data are untouched."""
    model = ctx.env.model
    tip_geom = ctx.tip_collision_geom[(side, finger)]
    site_pos = _tip_world(ctx, side, finger)
    obj_qpos_adr = ctx.env._object_qpos_adr
    saved_obj_qpos = ctx.scratch.qpos[obj_qpos_adr:obj_qpos_adr + 7].copy()

    other_axes = [a for a in range(3) if a != approach_axis]
    obj_pos_probe = site_pos.copy()
    obj_pos_probe[approach_axis] = site_pos[approach_axis] - approach_sign * ctx.half
    dist = None
    for it in range(max_iters):
        ctx.scratch.qpos[obj_qpos_adr:obj_qpos_adr + 3] = obj_pos_probe
        ctx.scratch.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, ctx.scratch)
        dist, _ = mesh_signed_distance(ctx, side, finger)
        if abs(dist) < 1e-6:
            break
        obj_pos_probe[approach_axis] -= approach_sign * dist

    face_at_touch = obj_pos_probe[approach_axis] - approach_sign * ctx.half
    offset = (site_pos[approach_axis] - face_at_touch) * approach_sign

    ctx.scratch.qpos[obj_qpos_adr:obj_qpos_adr + 7] = saved_obj_qpos
    mujoco.mj_forward(model, ctx.scratch)
    return dict(support_offset_m=offset, final_mesh_dist_m=dist, n_iters=it + 1, converged=abs(dist) < 1e-4)


# ---------------------------------------------------------------------
# Residual construction
# ---------------------------------------------------------------------

MARGIN = 0.010          # face-interior margin, m
JOINT_LIMIT_SAFE = 0.05  # rad, soft joint-limit proximity band
THUMB_THUMB_SAFE = 0.04  # m, target minimum separation (> ~2x finger half-width ~0.015m each)
W_MARGIN = 5.0
W_TT = 6.0
W_JOINT = 2.0
W_REG_ARM = 0.3
W_REG_FINGER_INDEXMID = 0.15   # soft regularization only, NOT a hard hold
W_REG_FINGER_THUMB = 0.02      # thumb is the one we WANT to move -- tiny reg only
W_WRIST_EXTREME = 0.4
W_WAIST_REG = 0.5  # small soft pull toward canonical waist value (Section 10)
W_REAL_COLLISION = 40.0  # real mesh penetration depth is meters-scale and must dominate the cost when active


_ALLOWED_TIP_BODY_NAMES = (
    "left_hand_thumb_2_link", "left_hand_index_1_link", "left_hand_middle_1_link",
    "right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link",
)


def _allowed_contact_body_ids(model: mujoco.MjModel, obj_body: int) -> set[tuple[int, int]]:
    """[32nd-session] The ONLY contacts this planner treats as intended:
    one of the 6 designated true-fingertip link bodies touching the
    object. Every other real contact -- including any OTHER hand link
    (proximal segments) touching the object, hand-vs-hand, hand-vs-table,
    or any G1 self-collision -- is forbidden. This replaces the 31st
    session's narrower, name-substring-based category list (thumb-thumb/
    palm-wrist-object/table only), which completely missed proximal
    index/middle/thumb links plowing through the object by up to 3.3cm
    and G1 self-collision (torso vs shoulder, 7.6mm) -- both found in
    this session's own audit of the 31st session's best candidate
    (measured_primary/palm_back)."""
    ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in _ALLOWED_TIP_BODY_NAMES}
    return {(min(b, obj_body), max(b, obj_body)) for b in ids}


_ARM_PREFIXES = ("hand", "wrist", "shoulder", "elbow")


def _arm_side(name: str) -> str | None:
    """[32nd-session fix] The ORIGINAL classifier only recognized
    'left_hand'/'right_hand'-prefixed bodies as "the hand", which
    completely missed wrist links (named 'left_wrist_yaw_link' etc, NOT
    'left_hand_wrist_...') and shoulder/elbow links -- so real
    object<->wrist contacts and torso<->shoulder self-collision were
    silently falling into the catch-all "other" bucket and never
    counted as forbidden. Discovered via this session's own Section 5
    audit (exact geom/body pair enumeration) of the 31st session's best
    candidate. Now recognizes the whole arm+hand chain (shoulder, elbow,
    wrist, and every hand/finger link) for each side."""
    for side in ("left", "right"):
        if any(name.startswith(f"{side}_{p}") for p in _ARM_PREFIXES):
            return side
    return None


def _classify_contact(model: mujoco.MjModel, obj_body: int, table_body: int, c) -> tuple[str, str]:
    """Returns (bucket, detail) for one real MuJoCo contact. bucket is
    one of: allowed_tip_contact, thumb_thumb, hand_hand, proximal_object,
    wrist_palm_object, self_collision, table_hand, table_object, other."""
    b1 = model.geom_bodyid[c.geom1]
    b2 = model.geom_bodyid[c.geom2]
    n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
    n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
    is_obj = obj_body in (b1, b2)
    is_table = table_body in (b1, b2)
    side1, side2 = _arm_side(n1), _arm_side(n2)
    is_left_arm = "left" in (side1, side2)
    is_right_arm = "right" in (side1, side2)

    if is_obj and (min(b1, b2), max(b1, b2)) in _allowed_contact_body_ids(model, obj_body):
        return "allowed_tip_contact", f"{n1}<->{n2}"
    if not is_obj and not is_table and is_left_arm and is_right_arm:
        return "thumb_thumb" if ("thumb" in n1 and "thumb" in n2) else "hand_hand", f"{n1}<->{n2}"
    if is_obj and (is_left_arm or is_right_arm):
        if "wrist" in n1 or "wrist" in n2 or "palm" in n1 or "palm" in n2:
            return "wrist_palm_object", f"{n1}<->{n2}"
        return "proximal_object", f"{n1}<->{n2}"
    if is_table and (is_left_arm or is_right_arm):
        return "table_hand", f"{n1}<->{n2}"
    if is_table and is_obj:
        return "table_object", f"{n1}<->{n2}"
    if not is_obj and not is_table and (is_left_arm or is_right_arm):
        return "self_collision", f"{n1}<->{n2}"
    return "other", f"{n1}<->{n2}"


def enumerate_contact_pairs(env, scratch: mujoco.MjData) -> list[dict]:
    """Section 5's exact requirement: every real penetrating contact,
    with geom names, body names, contact point, penetration depth, and
    its classification bucket -- not just aggregate counts.

    [34th session, Section 4] Extended (backward-compatible -- all
    original keys unchanged) with: body1/body2 names, signed_distance
    (c.dist, not just penetration=-dist), contype/conaffinity of both
    geoms (confirms every reported pair is a REAL collision-enabled pair,
    never the contype=0 visual duplicate -- MuJoCo's own broadphase
    cannot generate a contact for a contype=0 geom at all, so this is a
    verification field, not a fix), and -- for any pair touching the
    object -- the object-local contact point and which box face it is
    geometrically closest to (classify_nearest_face)."""
    model = env.model
    obj_body = env._object_body_id
    table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    obj_qpos_adr = env._object_qpos_adr
    obj_pos = scratch.qpos[obj_qpos_adr:obj_qpos_adr + 3]
    obj_quat = scratch.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7]
    obj_R_flat = np.zeros(9)
    mujoco.mju_quat2Mat(obj_R_flat, obj_quat)
    obj_R = obj_R_flat.reshape(3, 3)
    half = env.config.object_half_size
    rows = []
    for i in range(scratch.ncon):
        c = scratch.contact[i]
        if c.dist >= 0:
            continue
        bucket, detail = _classify_contact(model, obj_body, table_body, c)
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        row = dict(
            bucket=bucket, bodies=detail,
            geom1=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}",
            geom2=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}",
            geom1_id=int(c.geom1), geom2_id=int(c.geom2),
            body1=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body{b1}",
            body2=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body{b2}",
            pos=[float(v) for v in c.pos], penetration=float(-c.dist), signed_distance=float(c.dist),
            contype1=int(model.geom_contype[c.geom1]), conaffinity1=int(model.geom_conaffinity[c.geom1]),
            contype2=int(model.geom_contype[c.geom2]), conaffinity2=int(model.geom_conaffinity[c.geom2]),
            object_local_point=None, nearest_face=None,
        )
        if obj_body in (b1, b2):
            local = obj_R.T @ (np.array(c.pos) - obj_pos)
            nearest = classify_nearest_face(local, half)
            row["object_local_point"] = [float(v) for v in local]
            row["nearest_face"] = nearest["face"]
        rows.append(row)
    return rows


ALL_FORBIDDEN_BUCKETS = ("thumb_thumb", "hand_hand", "proximal_object", "wrist_palm_object",
                         "self_collision", "table_hand")
# The 31st-session's own (narrower) forbidden set, kept ONLY for the
# causal A/B comparison this session runs (Section 7): "wrist_palm_object"
# stands in for that session's combined palm/wrist-vs-object check, and
# "table_hand" for its table check. It never saw proximal_object,
# hand_hand, or self_collision at all.
LEGACY_FORBIDDEN_BUCKETS = ("thumb_thumb", "wrist_palm_object", "table_hand")


def _real_collision_penalty_rows(ctx: PlannerContext, buckets: tuple[str, ...] = ALL_FORBIDDEN_BUCKETS) -> list[float]:
    """Reads the ACTUAL MuJoCo contacts just computed by mj_forward
    (real mesh geometry, not the face/margin proxy above) and turns
    total penetration depth per FORBIDDEN category (``buckets``) into a
    FIXED-length set of residual rows -- summed within each category
    rather than one row per contact so the residual vector's length
    stays constant as contacts appear/disappear between finite-
    difference evaluations (a Gauss-Newton Jacobian requires a
    fixed-shape residual)."""
    model = ctx.env.model
    obj_body = ctx.env._object_body_id
    table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    scratch = ctx.scratch
    pen_by_bucket = {b: 0.0 for b in buckets}
    for i in range(scratch.ncon):
        c = scratch.contact[i]
        if c.dist >= 0:
            continue
        bucket, _ = _classify_contact(model, obj_body, table_body, c)
        if bucket in pen_by_bucket:
            pen_by_bucket[bucket] += float(-c.dist)
    return [W_REAL_COLLISION * v for v in pen_by_bucket.values()]


ALLOWED_PENETRATION_TOL = 0.001  # m (1mm) -- geom margin/gap are both 0.0 (measured), so any
# real negative mj_geomDistance IS true geometric penetration, not a margin artifact;
# 1mm is a generous solver-convergence band, not a physically-motivated "safe" depth.
ALLOWED_SEPARATION_TOL = 0.001   # m (1mm) -- symmetric band for "not yet touching"
W_MESH_FACE = 200.0  # [34th session: now ONLY used for the allowed-penetration/separation BOUND below,
# not as the primary steering term -- see face_assignment_metrics' docstring for why driving the
# whole-box nearest distance to 0 does not, by itself, guarantee contact on the ASSIGNED face]
W_ALLOWED_PENETRATION_BOUND = 500.0
W_FACE_PLANE = 200.0  # [34th session] steers the REAL near-contact point's ASSIGNED-axis coordinate
# to the assigned face's plane specifically -- THIS is what makes measured_primary vs mirrored
# actually differ again (33rd session's W_MESH_FACE*dist term had no dependency on face_sign at all).
PROXIMAL_CLEARANCE_TOL = 0.001  # m (1mm) -- same numerical-convergence-band rationale as the tip tolerances.
W_PROXIMAL_CLEARANCE = 300.0  # [34th session] smooth mj_geomDistance-based clearance term for the NON-tip
# finger links (thumb_0/1, index_0, middle_0) -- root-cause fix for this session's largest forbidden
# penetration (left/right thumb_1_link vs object, up to 1.2mm): the 33rd session's _real_collision_penalty_rows
# only sums penetration from CURRENT MuJoCo contacts (zero gradient until already touching); this term is
# smooth (mj_geomDistance is defined at any separation) so the solver feels a pull to stay clear BEFORE contact.


def residuals(x: np.ndarray, ctx: PlannerContext, topology: Topology, use_real_collision: bool = True,
              forbidden_buckets: tuple[str, ...] = ALL_FORBIDDEN_BUCKETS) -> np.ndarray:
    """[33rd session] mesh_signed_distance (real mj_geomDistance) replaced
    the 31st/32nd session's true-fingertip-SITE-vs-face-plane proxy, which
    was measured to diverge from the real mesh by a configuration-
    dependent -19.5mm to +4.7mm. [34th session correction] Driving that
    WHOLE-BOX nearest distance to 0 alone does not guarantee contact on
    the topology-ASSIGNED face (it can be satisfied by any face/edge/
    corner) -- this is why measured_primary and mirrored produced
    byte-identical 33rd-session results. Reinstated an explicit face-
    PLANE-coordinate residual (face_plane_err) computed on the REAL
    near-contact point (not the site), added an actual-geometry normal-
    alignment term (classify_nearest_face), and added a smooth proximal-
    link clearance term (root cause of the largest forbidden penetration:
    thumb_1_link, not the tip, grazing the object)."""
    _write_x(ctx, x)
    rows = []

    tip_world = {}
    for side in ("left", "right"):
        for finger in FINGERS:
            axis = topology.face_axis[finger]
            # NOTE: both hands were measured (Stage 1 audit) to target the
            # SAME object faces (index/middle both near +X, both thumbs
            # near -X) -- this is a bimanual side-pinch on a shared
            # opposing face pair, not a left/right mirrored face
            # assignment, so the topology's sign is NOT flipped per side.
            sign = topology.face_sign[finger]
            dist, fromto = mesh_signed_distance(ctx, side, finger)
            # [34th session] face-PLANE steering: drive the REAL near-contact
            # point's ASSIGNED-axis coordinate to that face's plane
            # specifically. mesh_signed_distance's whole-box nearest
            # distance (below) has NO dependency on which face was
            # assigned -- see this section's module-level docstring for
            # why that made measured_primary/mirrored indistinguishable.
            near_on_object = fromto[3:6]
            local = ctx.obj_R.T @ (near_on_object - ctx.obj_pos)
            tip_world[(side, finger)] = local
            face_plane_err = local[axis] - sign * ctx.half
            rows.append(W_FACE_PLANE * face_plane_err)
            # explicit asymmetric bound (Section 7, 33rd session): a PHYSICAL
            # safety bound on the real fingertip mesh regardless of which
            # face it ends up touching, so it still uses the whole-box
            # nearest mesh_signed_distance.
            over_pen = max(0.0, -dist - ALLOWED_PENETRATION_TOL)
            over_sep = max(0.0, dist - ALLOWED_SEPARATION_TOL)
            rows.append(W_ALLOWED_PENETRATION_BOUND * over_pen)
            rows.append(W_ALLOWED_PENETRATION_BOUND * over_sep)
            # face-interior (tangential) margin: use the REAL near-point on
            # the object (fromto[3:6], object-local frame), not the site proxy
            other_axes = [a for a in range(3) if a != axis]
            for a in other_axes:
                over = abs(local[a]) - (ctx.half - MARGIN)
                rows.append(W_MARGIN * max(0.0, over))
            # NOTE [34th session]: an explicit normal-alignment term
            # (dot(assigned_normal, classify_nearest_face(...).normal))
            # was tried here and REMOVED -- classify_nearest_face's
            # nearest-face selection is a DISCRETE branch, so its output
            # can flip between the +fd_eps and -fd_eps finite-difference
            # samples right at a face/edge boundary, injecting a spurious
            # near-infinite Jacobian entry (observed: solver cost jumped
            # from ~0.13 to 2500-34000 with it present). face_plane_err
            # (above) plus the tangential margin (above) are smooth and,
            # when BOTH are driven to ~0, geometrically already guarantee
            # the near point is on the assigned face's plane and interior
            # -- which by construction makes classify_nearest_face agree
            # and the normal angle 0. Normal/face validity is therefore
            # checked POST-HOC only (face_assignment_metrics, used by
            # run_multistart/is_statically_feasible), never inside this
            # differentiable objective.

    # [34th session] proximal (non-tip) finger-link clearance: root-cause
    # fix for this session's largest forbidden penetration (left/right
    # thumb_1_link vs object, up to 1.2mm). Smooth mj_geomDistance-based
    # term (unlike _real_collision_penalty_rows below, which only sums
    # penetration from CURRENT MuJoCo contacts and has zero gradient
    # before the links are already touching) that pulls each non-tip
    # finger link toward >= PROXIMAL_CLEARANCE_TOL separation from the
    # object at ALL times during the solve, not just after it collides.
    for side in ("left", "right"):
        for finger in FINGERS:
            for geom in ctx.proximal_collision_geom[(side, finger)]:
                fromto_p = np.zeros(6)
                d = mujoco.mj_geomDistance(ctx.env.model, ctx.scratch, geom, ctx.obj_geom_id, 1.0, fromto_p)
                rows.append(W_PROXIMAL_CLEARANCE * max(0.0, PROXIMAL_CLEARANCE_TOL - d))

    lt = tip_world[("left", "thumb")]
    rt = tip_world[("right", "thumb")]
    # both in object-local frame; convert delta back through obj_R for a true metric distance (obj_R is orthonormal so norm is frame-invariant)
    tt_dist = float(np.linalg.norm(lt - rt))
    rows.append(W_TT * max(0.0, THUMB_THUMB_SAFE - tt_dist))

    # joint-limit soft proximity (hinge, both directions) -- lo/hi extended
    # with the waist's own soft-range bounds when enable_waist() was called
    lo, hi = _full_bounds(ctx)
    below = np.maximum(0.0, (lo + JOINT_LIMIT_SAFE) - x)
    above = np.maximum(0.0, x - (hi - JOINT_LIMIT_SAFE))
    rows.extend((W_JOINT * below).tolist())
    rows.extend((W_JOINT * above).tolist())

    # regularization toward the CONTACT_ACQUIRE-end reference pose
    x0_full = _full_x0(ctx)
    for i in range(N_TOTAL):
        local_i = i % N_PER_HAND
        if local_i < N_ARM:
            w = W_REG_ARM
        else:
            finger_slot = local_i - N_ARM
            is_thumb = finger_slot < 3
            w = W_REG_FINGER_THUMB if is_thumb else W_REG_FINGER_INDEXMID
        rows.append(w * (x[i] - x0_full[i]))

    # wrist-extremity penalty (deviation from neutral 0, on top of the reg-to-x0 term above)
    for side_idx in (0, 1):
        base = side_idx * N_PER_HAND + 4  # wrist roll/pitch/yaw are arm slots 4,5,6
        for k in range(3):
            rows.append(W_WRIST_EXTREME * x[base + k])

    # waist soft-deviation regularization (Section 10): a SMALL pull back
    # toward the canonical stand-pose waist value, on top of the hard
    # soft-range bound already enforced via lo/hi above.
    if ctx.use_waist and len(x) > N_TOTAL:
        waist_x = x[N_TOTAL:N_TOTAL + 3]
        rows.extend((W_WAIST_REG * (waist_x - x0_full[N_TOTAL:N_TOTAL + 3])).tolist())

    if use_real_collision:
        rows.extend(_real_collision_penalty_rows(ctx, buckets=forbidden_buckets))

    return np.array(rows)


def _full_bounds(ctx: PlannerContext) -> tuple[np.ndarray, np.ndarray]:
    if ctx.use_waist:
        return np.concatenate([ctx.jnt_lo, ctx.waist_lo]), np.concatenate([ctx.jnt_hi, ctx.waist_hi])
    return ctx.jnt_lo, ctx.jnt_hi


def _full_x0(ctx: PlannerContext) -> np.ndarray:
    if ctx.use_waist:
        return np.concatenate([ctx.x0, ctx.waist_x0])
    return ctx.x0


def solve_manifold(ctx: PlannerContext, x_init: np.ndarray, topology: Topology,
                    max_iters: int = 150, damping: float = 0.05, gain: float = 0.7, fd_eps: float = 1e-4,
                    forbidden_buckets: tuple[str, ...] = ALL_FORBIDDEN_BUCKETS) -> dict:
    """Gauss-Newton / Levenberg-Marquardt style solve with a NUMERICAL
    (central-difference) Jacobian of the full residual stack -- chosen
    over hand-derived analytic Jacobians because the residual stack mixes
    position, hinge/margin, and regularization terms of very different
    character; a single generic numerical Jacobian is far less
    bug-prone than deriving each term's gradient by hand, and at this
    dimensionality (28 decision vars) remains cheap (pure forward
    kinematics, no contact dynamics)."""
    n_dims = len(x_init)
    lo, hi = _full_bounds(ctx)
    x = x_init.copy()
    x = np.clip(x, lo, hi)
    r = residuals(x, ctx, topology, forbidden_buckets=forbidden_buckets)
    cost = float(np.dot(r, r))
    history = [cost]

    for it in range(max_iters):
        n_r = len(r)
        J = np.zeros((n_r, n_dims))
        for j in range(n_dims):
            dx = np.zeros(n_dims)
            dx[j] = fd_eps
            r_plus = residuals(np.clip(x + dx, lo, hi), ctx, topology, forbidden_buckets=forbidden_buckets)
            r_minus = residuals(np.clip(x - dx, lo, hi), ctx, topology, forbidden_buckets=forbidden_buckets)
            J[:len(r_plus), j] = (r_plus - r_minus) / (2 * fd_eps)

        JTJ = J.T @ J + damping * np.eye(n_dims)
        step = np.linalg.solve(JTJ, -J.T @ r) * gain
        step = np.clip(step, -0.15, 0.15)
        x_new = np.clip(x + step, lo, hi)
        r_new = residuals(x_new, ctx, topology, forbidden_buckets=forbidden_buckets)
        cost_new = float(np.dot(r_new, r_new))
        if cost_new < cost:
            x, r, cost = x_new, r_new, cost_new
            damping = max(damping / 3.0, 1e-4)  # standard LM: relax damping after a successful step
        else:
            damping *= 2.5
        history.append(cost)
        if cost < 1e-6 or (len(history) > 15 and abs(history[-1] - history[-15]) < 1e-8):
            break

    _write_x(ctx, x)
    return dict(x=x, cost=cost, n_iters=it + 1, history=history, converged=cost < 1e-4)


# ---------------------------------------------------------------------
# Post-hoc verification: real collision check via MuJoCo's own contact
# detection (mj_forward already runs mj_collision), and joint-limit /
# geometry diagnostics reported in their NATIVE units.
# ---------------------------------------------------------------------

_ALL_BUCKETS = ALL_FORBIDDEN_BUCKETS


def check_pose_collisions(env, scratch: mujoco.MjData) -> dict:
    """[32nd-session rewrite] Scans the ACTUAL MuJoCo contacts at the
    current scratch pose using the SAME general classifier
    (_classify_contact) the collision-penalty residual uses -- the
    31st-session version only recognized 3 forbidden name-substring
    categories and completely missed proximal index/middle/thumb links
    plowing through the object and G1 self-collision (see
    _allowed_contact_body_ids' docstring for the exact numbers found).
    Returns per-bucket contact counts, total penetration per bucket, and
    the overall max penetration."""
    model = env.model
    obj_body = env._object_body_id
    table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    result = {b: 0 for b in _ALL_BUCKETS}
    result.update({f"{b}_penetration": 0.0 for b in _ALL_BUCKETS})
    result["allowed_tip_contacts"] = 0
    result["max_penetration"] = 0.0
    result["n_contacts"] = int(scratch.ncon)
    result["n_forbidden_contacts"] = 0
    for i in range(scratch.ncon):
        c = scratch.contact[i]
        if c.dist >= 0:
            continue
        pen = float(-c.dist)
        result["max_penetration"] = max(result["max_penetration"], pen)
        bucket, _ = _classify_contact(model, obj_body, table_body, c)
        if bucket == "allowed_tip_contact":
            result["allowed_tip_contacts"] += 1
        elif bucket in _ALL_BUCKETS:
            result[bucket] += 1
            result[f"{bucket}_penetration"] += pen
            result["n_forbidden_contacts"] += 1
    return result


# ---------------------------------------------------------------------
# Grasp wrench diagnostics: friction-pyramid approximation, rank/
# condition, and a non-negative least-squares check of whether contact
# edge forces can balance gravity (a STATIC, diagnostic indicator --
# never treated as a dynamic Gate A result).
# ---------------------------------------------------------------------

def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    return t1, t2


def build_wrench_matrix(contacts: list[dict], obj_com: np.ndarray, mu: float = 0.8, n_edges: int = 4) -> np.ndarray:
    """contacts: list of {"pos": world point, "normal": UNIT outward-from-
    object normal}. Returns G (6 x (n_contacts*n_edges)) where each
    column is the wrench (force;torque about obj_com) of one friction-
    pyramid edge, pointing INTO the object (i.e. -normal component, since
    a finger pressing on the surface pushes opposite to the outward
    normal)."""
    cols = []
    for c in contacts:
        n = -np.asarray(c["normal"], dtype=np.float64)  # into the object
        n = n / np.linalg.norm(n)
        t1, t2 = _tangent_basis(n)
        r = np.asarray(c["pos"]) - obj_com
        for k in range(n_edges):
            ang = 2 * np.pi * k / n_edges
            f = n + mu * (np.cos(ang) * t1 + np.sin(ang) * t2)
            f = f / np.linalg.norm(f)
            tau = np.cross(r, f)
            cols.append(np.concatenate([f, tau]))
    return np.array(cols).T  # (6, n_contacts*n_edges)


def _perturb(ctx: PlannerContext, side: str, joint_suffix: str, delta: float) -> np.ndarray:
    """Returns a copy of ctx.x0 with one named joint (arm or finger)
    offset by ``delta`` (rad), clipped to that joint's own range."""
    x = ctx.x0.copy()
    side_idx = 0 if side == "left" else 1
    if joint_suffix in ARM_JOINT_SUFFIXES:
        local_i = side_idx * N_PER_HAND + ARM_JOINT_SUFFIXES.index(joint_suffix)
    else:
        local_i = side_idx * N_PER_HAND + N_ARM + FINGER_JOINT_SUFFIXES.index(joint_suffix)
    x[local_i] = float(np.clip(x[local_i] + delta, ctx.jnt_lo[local_i], ctx.jnt_hi[local_i]))
    return x


def build_posture_families(ctx: PlannerContext) -> dict[str, np.ndarray]:
    """Deterministic, meaningfully-named multi-start family (Section 7):
    NOT a random parameter sweep -- each start nudges ONE physically
    interpretable joint group, applied to BOTH hands (mirrored by each
    hand's own sign convention, since jnt_range itself already encodes
    the left/right mirroring)."""
    starts = {"neutral": ctx.x0.copy()}

    def both(suffix, delta):
        x = ctx.x0.copy()
        for side in ("left", "right"):
            side_idx = 0 if side == "left" else 1
            if suffix in ARM_JOINT_SUFFIXES:
                local_i = side_idx * N_PER_HAND + ARM_JOINT_SUFFIXES.index(suffix)
            else:
                local_i = side_idx * N_PER_HAND + N_ARM + FINGER_JOINT_SUFFIXES.index(suffix)
            sign = 1.0 if (ctx.jnt_hi[local_i] - ctx.x0[local_i]) > (ctx.x0[local_i] - ctx.jnt_lo[local_i]) else -1.0
            x[local_i] = float(np.clip(x[local_i] + sign * delta, ctx.jnt_lo[local_i], ctx.jnt_hi[local_i]))
        return x

    starts["mild_wrist_roll"] = both("wrist_roll", 0.2)
    starts["mild_wrist_yaw"] = both("wrist_yaw", 0.2)
    starts["palm_forward"] = both("elbow", 0.15)
    starts["palm_back"] = both("elbow", -0.15)
    starts["thumb_abducted"] = both("thumb_0", 0.4)
    starts["index_middle_redistributed"] = both("middle_1", -0.3)
    return starts


@dataclass
class MultiStartResult:
    name: str
    topology_name: str
    x: np.ndarray
    cost: float
    n_iters: int
    converged: bool
    collisions: dict
    per_finger_face_residual: dict
    thumb_thumb_distance: float
    joint_margins_rad: dict
    joint_travel_from_canonical: float
    contact_pairs: list = field(default_factory=list)
    waist_delta_rad: np.ndarray | None = None  # [33rd session] None unless use_waist=True
    face_metrics: dict = field(default_factory=dict)  # [34th session] "{side}_{finger}" -> face_assignment_metrics dict


def run_multistart(env, expert, topologies: list[Topology] | None = None,
                    forbidden_buckets: tuple[str, ...] = ALL_FORBIDDEN_BUCKETS,
                    use_waist: bool = False) -> tuple[list[MultiStartResult], PlannerContext]:
    ctx = build_context(env, expert)
    if use_waist:
        ctx = enable_waist(ctx)
    if topologies is None:
        primary = derive_primary_topology(ctx)
        topologies = [primary, mirrored_topology(primary)]
    families = build_posture_families(ctx)
    if use_waist:
        families = {name: np.concatenate([x, ctx.waist_x0]) for name, x in families.items()}

    results = []
    for topo in topologies:
        for name, x_init in families.items():
            sol = solve_manifold(ctx, x_init, topo, max_iters=250, forbidden_buckets=forbidden_buckets)
            coll = check_pose_collisions(env, ctx.scratch)
            pairs = enumerate_contact_pairs(env, ctx.scratch)
            # [34th session] face_res now comes from the REAL mesh near-
            # contact point (face_assignment_metrics), not the old
            # analytic site (_tip_world) -- consistent with what the
            # solver itself was steering (residuals' face_plane_err).
            face_metrics = {}
            local_by_finger = {}
            for side in ("left", "right"):
                for finger in FINGERS:
                    fm = face_assignment_metrics(ctx, side, finger, topo)
                    face_metrics[f"{side}_{finger}"] = fm
                    local_by_finger[(side, finger)] = np.array(fm["object_local_point"])
            face_res = {k: v["face_plane_error"] for k, v in face_metrics.items()}
            lt = local_by_finger[("left", "thumb")]
            rt = local_by_finger[("right", "thumb")]
            tt_dist = float(np.linalg.norm(lt - rt))
            lo, hi = ctx.jnt_lo, ctx.jnt_hi
            margins = {}
            for side in ("left", "right"):
                hd = ctx.hands[side]
                jids = hd.arm_jids + hd.finger_jids
                names = [f"{side}_{n}_joint" for n in ARM_JOINT_SUFFIXES] + [f"{side}_hand_{n}_joint" for n in FINGER_JOINT_SUFFIXES]
                base = (0 if side == "left" else 1) * N_PER_HAND
                for k, jn in enumerate(names):
                    margins[jn] = float(min(sol["x"][base + k] - lo[base + k], hi[base + k] - sol["x"][base + k]))
            travel = float(np.linalg.norm(sol["x"] - _full_x0(ctx)))
            waist_delta = (sol["x"][N_TOTAL:N_TOTAL + 3] - ctx.waist_x0) if use_waist else None

            results.append(MultiStartResult(
                name=name, topology_name=topo.name, x=sol["x"], cost=sol["cost"], n_iters=sol["n_iters"],
                converged=sol["converged"], collisions=coll, per_finger_face_residual=face_res,
                waist_delta_rad=waist_delta, face_metrics=face_metrics,
                thumb_thumb_distance=tt_dist, joint_margins_rad=margins, joint_travel_from_canonical=travel,
                contact_pairs=pairs,
            ))
    return results, ctx


def is_statically_feasible(r: MultiStartResult, face_tol: float = 0.005, min_joint_margin: float = 0.02,
                            normal_angle_tol_deg: float = NORMAL_ANGLE_TOL_DEG) -> bool:
    """Section 8/9's strict static-success bar, checked mechanically (no
    post-hoc threshold relaxation): every fingertip within face_tol of
    its assigned face plane, ZERO real forbidden contacts of ANY
    category (thumb_thumb, hand_hand, proximal_object, wrist_palm_object,
    self_collision, table_hand), every joint at least min_joint_margin
    (rad) from its hard limit. [34th session] ALSO requires every
    fingertip's real near-contact point to be on the topology-ASSIGNED
    face (not merely somewhere on the object), not on an edge/corner
    (ambiguous between two faces), and within normal_angle_tol_deg of
    that face's true outward normal -- a candidate that only satisfies
    the OLD (33rd session) face_tol check by touching the wrong face or
    an edge is NOT a valid static candidate."""
    if max(abs(v) for v in r.per_finger_face_residual.values()) > face_tol:
        return False
    if r.face_metrics:
        if any(fm["wrong_face"] for fm in r.face_metrics.values()):
            return False
        if any(fm["is_edge_or_corner"] for fm in r.face_metrics.values()):
            return False
        if max(fm["normal_angle_deg"] for fm in r.face_metrics.values()) > normal_angle_tol_deg:
            return False
    if r.collisions["n_forbidden_contacts"] > 0:
        return False
    if r.collisions["max_penetration"] > 1e-4:
        return False
    if min(r.joint_margins_rad.values()) < min_joint_margin:
        return False
    return True


def wrench_diagnostics(contacts: list[dict], obj_com: np.ndarray, mass: float, g: float = 9.81) -> dict:
    G = build_wrench_matrix(contacts, obj_com)
    rank = int(np.linalg.matrix_rank(G, tol=1e-6))
    s = np.linalg.svd(G, compute_uv=False)
    condition = float(s[0] / s[-1]) if s[-1] > 1e-9 else float("inf")

    gravity_wrench = np.array([0.0, 0.0, -mass * g, 0.0, 0.0, 0.0])
    target = -gravity_wrench  # contacts must supply +mass*g upward
    lam, residual = nnls(G, target)
    achieved = G @ lam
    balance_err = float(np.linalg.norm(achieved - target))
    return dict(
        rank=rank, condition=condition, n_columns=G.shape[1],
        force_balance_residual=balance_err, edge_weights_sum=float(np.sum(lam)),
        predicted_net_wrench_error=balance_err,
    )


def contacts_from_face_metrics(r: "MultiStartResult") -> list[dict]:
    """[34th session, Section 8] Builds the wrench module's {"pos",
    "normal"} contact list ONLY from fingertips that pass face-aware
    validity (not wrong-face, not edge/corner, within
    NORMAL_ANGLE_TOL_DEG of the assigned face's true normal) -- i.e. the
    wrench matrix is never built from an analytic-normal assumption that
    face-aware validation would reject. Contacts using the real
    near-contact point and REAL actual_normal_world (not the assigned
    one) so a marginal-but-valid contact's slightly-off normal is
    reflected in the wrench, not idealized away."""
    contacts = []
    for fm in r.face_metrics.values():
        if fm["wrong_face"] or fm["is_edge_or_corner"] or fm["normal_angle_deg"] > NORMAL_ANGLE_TOL_DEG:
            continue
        contacts.append({"pos": fm["object_local_point"], "normal": fm["actual_normal_world"]})
    return contacts


# ---------------------------------------------------------------------
# Session 34: full collision census + pre-existing-vs-induced baseline
# comparison (Section 4/5) and a collision-penalty gradient sanity check
# (Section 9) -- all read-only diagnostics, never imported by the live
# control loop.
# ---------------------------------------------------------------------

def baseline_contact_signatures(make_env_fn) -> dict:
    """[34th session, Section 5] Real penetrating-contact (geom1, geom2)
    id-pair signatures at the canonical reference poses reachable by the
    UNMODIFIED expert -- used to separate contacts that ALREADY exist on
    the canonical controller's own path (pre-existing model self-contact,
    not something this planner caused) from ones a solved candidate
    newly introduces. NOTE: in this env's state machine, "CONTACT_
    ACQUIRE-end" and "THUMB_OPPOSE-entry" are the SAME transition instant
    (the loop below breaks the moment state becomes THUMB_OPPOSE), so
    there are only two DISTINCT baseline poses here, not three -- this is
    an honest fact about the state machine, not a shortcut."""
    from humanoid_learning.expert.grasp_expert import BimanualSidePinchExpert, GraspExpertConfig, GraspState

    def sig(data) -> set:
        return {(min(int(data.contact[i].geom1), int(data.contact[i].geom2)),
                 max(int(data.contact[i].geom1), int(data.contact[i].geom2)))
                for i in range(data.ncon) if data.contact[i].dist < 0}

    env_a = make_env_fn()
    env_a.reset(seed=0)
    reset_sig = sig(env_a.data)

    env_b = make_env_fn()
    env_b.reset(seed=0)
    expert_b = BimanualSidePinchExpert(env_b, GraspExpertConfig())
    for _ in range(1200):
        outcome = expert_b.step()
        if outcome.state in (GraspState.THUMB_OPPOSE, GraspState.FAILURE, GraspState.SUCCESS):
            break
    contact_acquire_end_sig = sig(env_b.data)
    return {
        "reset": reset_sig,
        "contact_acquire_end_thumb_oppose_entry": contact_acquire_end_sig,
    }


def full_collision_census(env, scratch: mujoco.MjData, meta: dict, baselines: dict | None = None) -> list[dict]:
    """[34th session, Section 4] enumerate_contact_pairs plus candidate
    metadata (topology/posture family/waist qpos, passed in via ``meta``)
    and pre-existing-at-baseline tags (Section 5) for every real
    penetrating contact at the given scratch pose."""
    rows = enumerate_contact_pairs(env, scratch)
    for row in rows:
        row.update(meta)
        key = (min(row["geom1_id"], row["geom2_id"]), max(row["geom1_id"], row["geom2_id"]))
        if baselines is not None:
            for base_name, base_sig in baselines.items():
                row[f"pre_existing_at_{base_name}"] = key in base_sig
    return rows


def collision_penalty_gradient_check(ctx: PlannerContext, topology: Topology, x: np.ndarray, joint_index: int,
                                      forbidden_buckets: tuple[str, ...] = ALL_FORBIDDEN_BUCKETS,
                                      delta: float = 1e-3) -> dict:
    """[34th session, Section 9] Finite-difference check that the FULL
    residual stack (proximal-clearance term included) has a NON-ZERO,
    correctly-signed response to nudging ONE decision variable -- i.e. a
    solver failure to fully clear a forbidden pair is a genuine local
    optimum / weight trade-off, not a flat or invisible penalty
    landscape the FD Jacobian cannot see."""
    lo, hi = _full_bounds(ctx)
    r0 = residuals(x, ctx, topology, forbidden_buckets=forbidden_buckets)
    cost0 = float(np.dot(r0, r0))
    x_plus = x.copy()
    x_plus[joint_index] = np.clip(x[joint_index] + delta, lo[joint_index], hi[joint_index])
    x_minus = x.copy()
    x_minus[joint_index] = np.clip(x[joint_index] - delta, lo[joint_index], hi[joint_index])
    r_plus = residuals(x_plus, ctx, topology, forbidden_buckets=forbidden_buckets)
    r_minus = residuals(x_minus, ctx, topology, forbidden_buckets=forbidden_buckets)
    cost_plus = float(np.dot(r_plus, r_plus))
    cost_minus = float(np.dot(r_minus, r_minus))
    return dict(cost0=cost0, cost_plus=cost_plus, cost_minus=cost_minus,
                gradient_nonzero=not (cost_plus == cost0 and cost_minus == cost0))
