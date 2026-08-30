"""Thumb Contact Geometry and 10/11/12cm Controlled Size Diagnosis
session: per-substep thumb contact-event tracking (face interior/edge/
corner classification, slip/bounce/reach-loss/rotation-induced-loss/
bad-geometry classification), independent of grasp_expert.py's control
loop (read-only diagnostic, same isolation posture as grasp_wrench_
diagnostics.py -- never imported by the live controller).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import mujoco
import numpy as np


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    return R.reshape(3, 3)


class FaceRegion(Enum):
    FACE_INTERIOR = auto()
    EDGE_NEAR = auto()
    CORNER_NEAR = auto()
    AMBIGUOUS = auto()
    WRONG_FACE = auto()
    NO_CONTACT = auto()


def classify_face_region(
    world_point: np.ndarray, obj_pos: np.ndarray, obj_quat: np.ndarray, half_size: float,
    inset_margin: float, intended_face: str | None = None, normal_ambiguity_margin: float = 0.3,
) -> tuple[FaceRegion, str, np.ndarray]:
    """Classifies a contact point's position WITHIN its face (not just
    which face -- that part reuses the same dominant-local-axis logic
    grasp_expert._classify_contact_face already uses, applied here to
    the contact POINT's own direction from center, which for a point
    genuinely ON a box's surface has the same dominant axis as its
    outward normal would). Returns (region, actual_face_label,
    local_point).

    ``inset_margin`` (meters): the two non-dominant local coordinates
    must be within ``half_size - inset_margin`` of zero for
    FACE_INTERIOR; between that and ``half_size`` on exactly one axis is
    EDGE_NEAR, on both axes is CORNER_NEAR. This is a genuine, size-
    conditioned safety margin (not chosen to force a passing result --
    see the session's own PROJECT_CONTEXT.md report for the value used
    and why)."""
    R = _quat_to_mat(obj_quat)
    local = R.T @ (world_point - obj_pos)
    abs_local = np.abs(local)
    # Face membership is decided by which local coordinate is CLOSEST TO
    # half_size (a genuine surface contact should have exactly one
    # coordinate near +-half_size) -- NOT by which is simply largest in
    # magnitude. This is deliberately a DIFFERENT test than grasp_expert.
    # _classify_contact_face's normal-based "largest component" rule:
    # that rule classifies a CONTACT NORMAL (ambiguous when two axes are
    # comparably dominant, e.g. a true 45-degree edge normal), whereas
    # this classifies a CONTACT POSITION already known to be near a
    # face's plane -- a point sitting a few mm from that face's own edge
    # still clearly belongs to that face (that is exactly EDGE_NEAR, not
    # face-ambiguous), so gating on the two-largest-components gap here
    # would wrongly relabel every genuine EDGE_NEAR/CORNER_NEAR point as
    # AMBIGUOUS before the edge/corner logic below ever runs.
    dominant_axis = int(np.argmin(np.abs(abs_local - half_size)))
    if abs(abs_local[dominant_axis] - half_size) > normal_ambiguity_margin * half_size:
        return FaceRegion.AMBIGUOUS, "EDGE_OR_CORNER_AMBIGUOUS", local
    sign = local[dominant_axis] > 0
    if dominant_axis == 0:
        face = "FACE_POS_X" if sign else "FACE_NEG_X"
    elif dominant_axis == 1:
        face = "FACE_POS_Y" if sign else "FACE_NEG_Y"
    else:
        face = "FACE_TOP" if sign else "FACE_BOTTOM"

    if intended_face is not None and face != intended_face:
        return FaceRegion.WRONG_FACE, face, local

    other_axes = [a for a in range(3) if a != dominant_axis]
    near_edge = [abs_local[a] > (half_size - inset_margin) for a in other_axes]
    n_near = sum(near_edge)
    if n_near == 0:
        region = FaceRegion.FACE_INTERIOR
    elif n_near == 1:
        region = FaceRegion.EDGE_NEAR
    else:
        region = FaceRegion.CORNER_NEAR
    return region, face, local


class LossReason(Enum):
    FRICTION_SLIP = auto()
    IMPACT_BOUNCE = auto()
    KINEMATIC_REACH_LOSS = auto()
    OBJECT_ROTATION_INDUCED_LOSS = auto()
    BAD_CONTACT_GEOMETRY = auto()
    UNKNOWN = auto()


@dataclass
class ThumbContactSample:
    step: int
    touched: bool
    world_point: np.ndarray | None
    region: FaceRegion
    actual_face: str
    normal_force: float
    tangential_force: float
    tangential_vel: float
    normal_vel: float
    joint_limit_margin: float
    object_ang_speed: float


@dataclass
class ContactEvent:
    start_step: int
    end_step: int | None = None
    samples: list[ThumbContactSample] = field(default_factory=list)
    primary_reason: LossReason = LossReason.UNKNOWN
    secondary_reason: LossReason | None = None

    @property
    def duration(self) -> int:
        if self.end_step is None:
            return len(self.samples)
        return self.end_step - self.start_step

    @property
    def peak_force(self) -> float:
        return max((s.normal_force + s.tangential_force for s in self.samples), default=0.0)

    @property
    def force_impulse(self) -> float:
        return float(sum((s.normal_force + s.tangential_force) for s in self.samples))

    @property
    def peak_tangential_vel(self) -> float:
        return max((s.tangential_vel for s in self.samples), default=0.0)

    @property
    def accumulated_slip_distance(self) -> float:
        # Rough integral of tangential velocity over the samples (dt not
        # tracked per-sample here -- caller supplies samples already at a
        # fixed step cadence, so this is "distance in units of steps",
        # a comparable RELATIVE metric across conditions of the SAME cadence.
        return float(sum(s.tangential_vel for s in self.samples))

    def face_interior_fraction(self) -> float:
        if not self.samples:
            return 0.0
        return sum(1 for s in self.samples if s.region == FaceRegion.FACE_INTERIOR) / len(self.samples)


def find_thumb_contact(
    model: mujoco.MjModel, data: mujoco.MjData, side: str, obj_body_id: int,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, float, float, int]:
    """Scans data.contact for a thumb-vs-object contact (side's thumb
    link prefix), returning (touched, world_point, world_normal_obj_to_
    thumb, normal_force, tangential_force, geom_pair_contact_index).
    Picks the largest-resultant contact if more than one thumb geom
    touches simultaneously (rare, but the same 'largest wins' convention
    grasp_expert.HandContact already uses elsewhere)."""
    thumb_prefix = f"{side}_hand_thumb"
    best = None
    best_resultant = -1.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if obj_body_id not in (b1, b2):
            continue
        other = b2 if b1 == obj_body_id else b1
        other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        if not other_name.startswith(thumb_prefix):
            continue
        force6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force6)
        normal = float(force6[0])
        tangential = float(np.hypot(force6[1], force6[2]))
        resultant = float(np.hypot(normal, tangential))
        if resultant > best_resultant:
            best_resultant = resultant
            frame_normal = np.array(c.frame[0:3], dtype=np.float64)
            world_normal = frame_normal if b1 == obj_body_id else -frame_normal
            best = (np.array(c.pos, dtype=np.float64), world_normal, abs(normal), tangential, i)
    if best is None:
        return False, None, None, 0.0, 0.0, -1
    point, normal, nf, tf, idx = best
    return True, point, normal, nf, tf, idx


def relative_velocity_at_point(
    model: mujoco.MjModel, data: mujoco.MjData, obj_body_id: int, thumb_body_id: int,
    world_point: np.ndarray, world_normal: np.ndarray, obj_dof_adr: int,
) -> tuple[float, float]:
    """Relative velocity between the thumb surface and the object surface
    AT the contact point (both bodies' point-velocity, object's own via
    qvel's free-joint linear+angular, thumb's via mj_objectVelocity),
    decomposed into (normal_component, tangential_magnitude)."""
    obj_lin = data.qvel[obj_dof_adr:obj_dof_adr + 3]
    obj_ang = data.qvel[obj_dof_adr + 3:obj_dof_adr + 6]
    obj_com = data.xipos[obj_body_id]
    obj_point_vel = obj_lin + np.cross(obj_ang, world_point - obj_com)

    thumb_vel6 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, thumb_body_id, thumb_vel6, 0)
    # mj_objectVelocity with flg_local=0 returns [angular(3), linear(3)] in WORLD frame, about the body frame origin
    thumb_ang = thumb_vel6[0:3]
    thumb_lin_at_origin = thumb_vel6[3:6]
    thumb_body_pos = data.xpos[thumb_body_id]
    thumb_point_vel = thumb_lin_at_origin + np.cross(thumb_ang, world_point - thumb_body_pos)

    rel_vel = thumb_point_vel - obj_point_vel
    n = world_normal / (np.linalg.norm(world_normal) + 1e-12)
    normal_component = float(np.dot(rel_vel, n))
    tangential_vec = rel_vel - normal_component * n
    tangential_component = float(np.linalg.norm(tangential_vec))
    return normal_component, tangential_component


def classify_loss_reason(event: ContactEvent, joint_limit_threshold: float = 0.03) -> tuple[LossReason, LossReason | None]:
    """Section 6's five-way classification, applied to one finished
    ContactEvent's sample history. Order matters (checked most-specific
    first); returns (primary, secondary)."""
    if not event.samples:
        return LossReason.UNKNOWN, None
    n = len(event.samples)
    early_regions = [s.region for s in event.samples[: max(1, n // 3)]]
    bad_geometry_from_start = all(
        r in (FaceRegion.EDGE_NEAR, FaceRegion.CORNER_NEAR, FaceRegion.WRONG_FACE) for r in early_regions
    )
    if bad_geometry_from_start:
        return LossReason.BAD_CONTACT_GEOMETRY, None

    min_joint_margin = min(s.joint_limit_margin for s in event.samples)
    force_never_established = max((s.normal_force for s in event.samples), default=0.0) < 1.0
    if min_joint_margin < joint_limit_threshold or force_never_established:
        return LossReason.KINEMATIC_REACH_LOSS, None

    is_short = n <= 5
    peak_impulse_early = event.samples[0].normal_force if event.samples else 0.0
    ang_speed_at_start = event.samples[0].object_ang_speed
    ang_speed_at_end = event.samples[-1].object_ang_speed
    if is_short and peak_impulse_early > 3.0:
        return LossReason.IMPACT_BOUNCE, (
            LossReason.OBJECT_ROTATION_INDUCED_LOSS if ang_speed_at_end > ang_speed_at_start * 1.5 else None
        )

    ang_speed_rising_first = False
    half = max(1, n // 2)
    first_half_ang = [s.object_ang_speed for s in event.samples[:half]]
    second_half_ang = [s.object_ang_speed for s in event.samples[half:]]
    if second_half_ang and first_half_ang and np.mean(second_half_ang) > np.mean(first_half_ang) * 1.3:
        # rotation rose BEFORE the contact point drifted toward an edge?
        first_half_edge = any(s.region in (FaceRegion.EDGE_NEAR, FaceRegion.CORNER_NEAR) for s in event.samples[:half])
        if not first_half_edge:
            later_edge = any(s.region in (FaceRegion.EDGE_NEAR, FaceRegion.CORNER_NEAR) for s in event.samples[half:])
            if later_edge:
                return LossReason.OBJECT_ROTATION_INDUCED_LOSS, None

    avg_tangential_vel = float(np.mean([s.tangential_vel for s in event.samples]))
    if n >= 5 and avg_tangential_vel > 0.02:
        return LossReason.FRICTION_SLIP, None

    return LossReason.UNKNOWN, None
