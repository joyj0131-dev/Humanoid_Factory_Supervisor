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

    return PlannerContext(
        env=env, scratch=scratch, obj_pos=obj_pos, obj_R=obj_R, half=env.config.object_half_size,
        hands=hands, x0=x0, jnt_lo=ranges[:, 0], jnt_hi=ranges[:, 1],
    )


def _write_x(ctx: PlannerContext, x: np.ndarray) -> None:
    left_adr = ctx.hands["left"].qpos_adr
    right_adr = ctx.hands["right"].qpos_adr
    ctx.scratch.qpos[left_adr] = x[:N_PER_HAND]
    ctx.scratch.qpos[right_adr] = x[N_PER_HAND:]
    mujoco.mj_forward(ctx.env.model, ctx.scratch)


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
# Residual construction
# ---------------------------------------------------------------------

MARGIN = 0.010          # face-interior margin, m
JOINT_LIMIT_SAFE = 0.05  # rad, soft joint-limit proximity band
THUMB_THUMB_SAFE = 0.04  # m, target minimum separation (> ~2x finger half-width ~0.015m each)
W_FACE = 8.0
W_MARGIN = 5.0
W_TT = 6.0
W_JOINT = 2.0
W_REG_ARM = 0.3
W_REG_FINGER_INDEXMID = 0.15   # soft regularization only, NOT a hard hold
W_REG_FINGER_THUMB = 0.02      # thumb is the one we WANT to move -- tiny reg only
W_WRIST_EXTREME = 0.4
W_REAL_COLLISION = 40.0  # real mesh penetration depth is meters-scale and must dominate the cost when active


def _real_collision_penalty_rows(ctx: PlannerContext) -> list[float]:
    """Reads the ACTUAL MuJoCo contacts just computed by mj_forward
    (real mesh geometry, not the face/margin proxy above) and turns
    total penetration depth per UNDESIRED category into a FIXED-length
    (3) set of residual rows: thumb-vs-thumb, palm/wrist-vs-object,
    any hand body vs table -- summed rather than one row per contact so
    the residual vector's length stays constant as contacts appear/
    disappear between finite-difference evaluations (a Gauss-Newton
    Jacobian requires a fixed-shape residual). This is what actually
    pulls the solver out of poses the tip-position-only proxy above
    cannot see (whole-link overlap)."""
    model = ctx.env.model
    obj_body = ctx.env._object_body_id
    table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    scratch = ctx.scratch
    thumb_thumb_pen = 0.0
    palm_wrist_object_pen = 0.0
    table_pen = 0.0
    for i in range(scratch.ncon):
        c = scratch.contact[i]
        if c.dist >= 0:
            continue
        pen = float(-c.dist)
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
        is_obj = any(model.geom_bodyid[g] == obj_body for g in (c.geom1, c.geom2))
        if ("left_hand_thumb" in b1 and "right_hand_thumb" in b2) or ("right_hand_thumb" in b1 and "left_hand_thumb" in b2):
            thumb_thumb_pen += pen
        elif is_obj and ("palm" in b1 or "palm" in b2 or "wrist" in b1 or "wrist" in b2):
            palm_wrist_object_pen += pen
        elif "table" in b1 or "table" in b2:
            table_pen += pen
    return [W_REAL_COLLISION * thumb_thumb_pen, W_REAL_COLLISION * palm_wrist_object_pen, W_REAL_COLLISION * table_pen]


def residuals(x: np.ndarray, ctx: PlannerContext, topology: Topology, use_real_collision: bool = True) -> np.ndarray:
    _write_x(ctx, x)
    rows = []

    tt_dist = None
    tip_world = {}
    for side in ("left", "right"):
        for finger in FINGERS:
            local = ctx.obj_R.T @ (_tip_world(ctx, side, finger) - ctx.obj_pos)
            tip_world[(side, finger)] = local
            axis = topology.face_axis[finger]
            # NOTE: both hands were measured (Stage 1 audit) to target the
            # SAME object faces (index/middle both near +X, both thumbs
            # near -X) -- this is a bimanual side-pinch on a shared
            # opposing face pair, not a left/right mirrored face
            # assignment, so the topology's sign is NOT flipped per side.
            sign = topology.face_sign[finger]
            # face-plane residual: pull the assigned axis onto the face
            rows.append(W_FACE * (local[axis] - sign * ctx.half))
            other_axes = [a for a in range(3) if a != axis]
            for a in other_axes:
                over = abs(local[a]) - (ctx.half - MARGIN)
                rows.append(W_MARGIN * max(0.0, over))

    lt = tip_world[("left", "thumb")]
    rt = tip_world[("right", "thumb")]
    # both in object-local frame; convert delta back through obj_R for a true metric distance (obj_R is orthonormal so norm is frame-invariant)
    tt_dist = float(np.linalg.norm(lt - rt))
    rows.append(W_TT * max(0.0, THUMB_THUMB_SAFE - tt_dist))

    # joint-limit soft proximity (hinge, both directions)
    lo, hi = ctx.jnt_lo, ctx.jnt_hi
    below = np.maximum(0.0, (lo + JOINT_LIMIT_SAFE) - x)
    above = np.maximum(0.0, x - (hi - JOINT_LIMIT_SAFE))
    rows.extend((W_JOINT * below).tolist())
    rows.extend((W_JOINT * above).tolist())

    # regularization toward the CONTACT_ACQUIRE-end reference pose
    for i in range(N_TOTAL):
        local_i = i % N_PER_HAND
        if local_i < N_ARM:
            w = W_REG_ARM
        else:
            finger_slot = local_i - N_ARM
            is_thumb = finger_slot < 3
            w = W_REG_FINGER_THUMB if is_thumb else W_REG_FINGER_INDEXMID
        rows.append(w * (x[i] - ctx.x0[i]))

    # wrist-extremity penalty (deviation from neutral 0, on top of the reg-to-x0 term above)
    for side_idx in (0, 1):
        base = side_idx * N_PER_HAND + 4  # wrist roll/pitch/yaw are arm slots 4,5,6
        for k in range(3):
            rows.append(W_WRIST_EXTREME * x[base + k])

    if use_real_collision:
        rows.extend(_real_collision_penalty_rows(ctx))

    return np.array(rows)


def solve_manifold(ctx: PlannerContext, x_init: np.ndarray, topology: Topology,
                    max_iters: int = 150, damping: float = 0.05, gain: float = 0.7, fd_eps: float = 1e-4) -> dict:
    """Gauss-Newton / Levenberg-Marquardt style solve with a NUMERICAL
    (central-difference) Jacobian of the full residual stack -- chosen
    over hand-derived analytic Jacobians because the residual stack mixes
    position, hinge/margin, and regularization terms of very different
    character; a single generic numerical Jacobian is far less
    bug-prone than deriving each term's gradient by hand, and at this
    dimensionality (28 decision vars) remains cheap (pure forward
    kinematics, no contact dynamics)."""
    x = x_init.copy()
    x = np.clip(x, ctx.jnt_lo, ctx.jnt_hi)
    r = residuals(x, ctx, topology)
    cost = float(np.dot(r, r))
    history = [cost]

    for it in range(max_iters):
        n_r = len(r)
        J = np.zeros((n_r, N_TOTAL))
        for j in range(N_TOTAL):
            dx = np.zeros(N_TOTAL)
            dx[j] = fd_eps
            r_plus = residuals(np.clip(x + dx, ctx.jnt_lo, ctx.jnt_hi), ctx, topology)
            r_minus = residuals(np.clip(x - dx, ctx.jnt_lo, ctx.jnt_hi), ctx, topology)
            J[:len(r_plus), j] = (r_plus - r_minus) / (2 * fd_eps)

        JTJ = J.T @ J + damping * np.eye(N_TOTAL)
        step = np.linalg.solve(JTJ, -J.T @ r) * gain
        step = np.clip(step, -0.15, 0.15)
        x_new = np.clip(x + step, ctx.jnt_lo, ctx.jnt_hi)
        r_new = residuals(x_new, ctx, topology)
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

def check_pose_collisions(env, scratch: mujoco.MjData) -> dict:
    """Scans the ACTUAL MuJoCo contacts at the current scratch pose (real
    mesh geometry, not a proxy) and classifies each by body-name prefix,
    the same convention grasp_expert.HandContact already uses. Returns
    counts/lists so callers can distinguish an INTENDED fingertip-object
    contact from an unwanted one (thumb-thumb, palm-object, wrist-object,
    non-target finger link, table)."""
    model = env.model
    obj_body = env._object_body_id
    result = dict(thumb_thumb=0, palm_object=0, wrist_object=0, other_finger_object=0,
                   table_finger=0, max_penetration=0.0, n_contacts=int(scratch.ncon))
    for i in range(scratch.ncon):
        c = scratch.contact[i]
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
        pen = float(-c.dist) if c.dist < 0 else 0.0
        result["max_penetration"] = max(result["max_penetration"], pen)
        names = {b1, b2}
        is_obj = any(model.geom_bodyid[g] == obj_body for g in (c.geom1, c.geom2))
        if ("left_hand_thumb" in b1 and "right_hand_thumb" in b2) or ("right_hand_thumb" in b1 and "left_hand_thumb" in b2):
            result["thumb_thumb"] += 1
        elif is_obj and ("palm" in b1 or "palm" in b2 or "wrist_yaw_link" in b1 or "wrist_yaw_link" in b2):
            result["palm_object"] += 1
        elif is_obj and any("wrist" in n for n in names):
            result["wrist_object"] += 1
        elif is_obj and not any(("thumb" in n or "index" in n or "middle" in n) for n in names):
            pass  # object-vs-non-hand (e.g. table) handled by table_finger branch below
        if "table" in b1 or "table" in b2:
            if any(("thumb" in n or "index" in n or "middle" in n) for n in (b1, b2)):
                result["table_finger"] += 1
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


def run_multistart(env, expert, topologies: list[Topology] | None = None) -> tuple[list[MultiStartResult], PlannerContext]:
    ctx = build_context(env, expert)
    if topologies is None:
        primary = derive_primary_topology(ctx)
        topologies = [primary, mirrored_topology(primary)]
    families = build_posture_families(ctx)

    results = []
    for topo in topologies:
        for name, x_init in families.items():
            sol = solve_manifold(ctx, x_init, topo, max_iters=250)
            coll = check_pose_collisions(env, ctx.scratch)
            face_res = {}
            for side in ("left", "right"):
                for finger in FINGERS:
                    local = ctx.obj_R.T @ (_tip_world(ctx, side, finger) - ctx.obj_pos)
                    axis = topo.face_axis[finger]
                    sign = topo.face_sign[finger]
                    face_res[f"{side}_{finger}"] = float(local[axis] - sign * ctx.half)
            lt = ctx.obj_R.T @ (_tip_world(ctx, "left", "thumb") - ctx.obj_pos)
            rt = ctx.obj_R.T @ (_tip_world(ctx, "right", "thumb") - ctx.obj_pos)
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
            travel = float(np.linalg.norm(sol["x"] - ctx.x0))

            results.append(MultiStartResult(
                name=name, topology_name=topo.name, x=sol["x"], cost=sol["cost"], n_iters=sol["n_iters"],
                converged=sol["converged"], collisions=coll, per_finger_face_residual=face_res,
                thumb_thumb_distance=tt_dist, joint_margins_rad=margins, joint_travel_from_canonical=travel,
            ))
    return results, ctx


def is_statically_feasible(r: MultiStartResult, face_tol: float = 0.005, min_joint_margin: float = 0.02) -> bool:
    """Section 8's strict static-success bar, checked mechanically (no
    post-hoc threshold relaxation): every fingertip within face_tol of
    its assigned face plane, zero real undesired collisions/penetration,
    every joint at least min_joint_margin (rad) from its hard limit."""
    if max(abs(v) for v in r.per_finger_face_residual.values()) > face_tol:
        return False
    if r.collisions["thumb_thumb"] > 0 or r.collisions["palm_object"] > 0 or r.collisions["wrist_object"] > 0 or r.collisions["table_finger"] > 0:
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
