"""Net-Torque Root Cause Isolation session: full world-frame contact
WRENCH (force + torque about the object COM) measurement, per finger
group, for SIZE_12 fixed-base grasp diagnosis.

This is a STANDALONE diagnostic module -- it reads env.data.contact
directly (same pattern as grasp_expert.HandContact/_contact()) but never
writes anything back and is never imported by grasp_expert.py's control
loop, so it cannot affect the live controller (Section 3's isolation
requirement). It exists to answer Section 5's requirement precisely:
world-frame force per contact, moment arm from the object's OWN COM (not
its geom-center approximation -- read from model.body_ipos/the object
body's actual inertial frame), and per-group summed wrench, validated
against the object's OWN measured linear/angular acceleration (Section 5:
"측정한 net wrench의 방향이 object linear/angular acceleration 방향과
일관되는지도 확인").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class GroupWrench:
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    contact_count: int = 0


def _group_key(side: str, other_body_name: str, wrist_names: tuple[str, ...]) -> str | None:
    if other_body_name in wrist_names:
        return f"{side}_palm"
    prefix = f"{side}_hand"
    if not other_body_name.startswith(prefix):
        return None
    if other_body_name.startswith(f"{side}_hand_thumb"):
        return f"{side}_thumb"
    if other_body_name.startswith(f"{side}_hand_index"):
        return f"{side}_index"
    if other_body_name.startswith(f"{side}_hand_middle"):
        return f"{side}_middle"
    return f"{side}_other_finger"


def object_com_world(model: mujoco.MjModel, data: mujoco.MjData, object_body_id: int) -> np.ndarray:
    """The object's ACTUAL center of mass in world coordinates (body
    origin + the body's own inertial-frame offset, both already resolved
    by mj_forward/mj_kinematics into data.xipos) -- not assumed to equal
    the geom/body origin, even though for this project's uniform-density
    cube they coincide (verified directly, see test)."""
    return data.xipos[object_body_id].copy()


def compute_grasp_wrench(
    model: mujoco.MjModel, data: mujoco.MjData, object_body_id: int,
    left_wrist_names: tuple[str, ...], right_wrist_names: tuple[str, ...],
) -> dict[str, GroupWrench]:
    """Scans ALL current data.contact entries touching the object body and
    sums (not just tracks the single largest, unlike grasp_expert.
    HandContact) each one's world-frame FORCE and its TORQUE about the
    object's actual COM into the appropriate group bucket -- left/right
    x {thumb, index, middle, palm}, plus 'table' and 'hand_hand' (thumb-
    thumb is a subset, reported separately).

    Sign convention (force ON THE OBJECT from the other body), derived
    from MuJoCo's own documented mj_contactForce convention (the returned
    force/torque is what geom1 exerts on geom2, in the contact frame,
    with frame row 0 the world-frame normal pointing geom1->geom2): the
    force on the object is the raw computed force when the object is
    geom2, and its negation when the object is geom1. This is the exact
    mirror of ContactCategory.world_normal's existing geom1/geom2 sign
    handling elsewhere in this codebase (see grasp_expert._contact),
    applied to the FULL vector instead of just the normal component.
    Validated in scripts/test_grasp_wrench_diagnostics.py against the
    object's actual qacc-implied net force/torque, not assumed correct.
    """
    com = object_com_world(model, data, object_body_id)
    groups: dict[str, GroupWrench] = {}

    def bucket(key: str) -> GroupWrench:
        if key not in groups:
            groups[key] = GroupWrench()
        return groups[key]

    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if object_body_id not in (b1, b2):
            continue
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
        other_name = n2 if b1 == object_body_id else n1

        force6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force6)
        R = np.array(c.frame, dtype=np.float64).reshape(3, 3)  # rows = local axes in world coords
        force_world_on_geom2 = R.T @ force6[0:3]
        torque_world_on_geom2 = R.T @ force6[3:6]
        # Force/torque as computed is what geom1 exerts on geom2 (MuJoCo
        # convention) -- flip if the OBJECT is geom1, so we always get
        # the force/torque ON THE OBJECT regardless of which geom slot it
        # landed in (mirrors world_normal's existing sign-correction).
        if b1 == object_body_id:
            force_on_object = -force_world_on_geom2
            torque_on_object = -torque_world_on_geom2
        else:
            force_on_object = force_world_on_geom2
            torque_on_object = torque_world_on_geom2

        contact_point = np.array(c.pos, dtype=np.float64)
        r = contact_point - com
        total_torque = np.cross(r, force_on_object) + torque_on_object

        is_left_hand = other_name.startswith("left_hand") or other_name in left_wrist_names
        is_right_hand = other_name.startswith("right_hand") or other_name in right_wrist_names
        if "table" in other_name:
            key = "table"
        elif is_left_hand:
            key = _group_key("left", other_name, left_wrist_names) or "left_other"
        elif is_right_hand:
            key = _group_key("right", other_name, right_wrist_names) or "right_other"
        else:
            key = "other"

        g = bucket(key)
        g.force += force_on_object
        g.torque += total_torque
        g.contact_count += 1

    return groups


def summarize_wrench(groups: dict[str, GroupWrench]) -> dict[str, np.ndarray]:
    """Aggregates the per-group breakdown into left-hand-total,
    right-hand-total, robot-object-total, table-total, and grand-total
    force/torque -- the specific rollups Section 5 asks be recorded every
    substep."""
    left_force = np.zeros(3)
    left_torque = np.zeros(3)
    right_force = np.zeros(3)
    right_torque = np.zeros(3)
    for k, g in groups.items():
        if k.startswith("left_"):
            left_force += g.force
            left_torque += g.torque
        elif k.startswith("right_"):
            right_force += g.force
            right_torque += g.torque
    table_force = groups["table"].force.copy() if "table" in groups else np.zeros(3)
    table_torque = groups["table"].torque.copy() if "table" in groups else np.zeros(3)
    robot_force = left_force + right_force
    robot_torque = left_torque + right_torque
    return {
        "left_force": left_force, "left_torque": left_torque,
        "right_force": right_force, "right_torque": right_torque,
        "robot_object_force": robot_force, "robot_object_torque": robot_torque,
        "table_force": table_force, "table_torque": table_torque,
        "total_force": robot_force + table_force,
        "total_torque": robot_torque + table_torque,
    }
