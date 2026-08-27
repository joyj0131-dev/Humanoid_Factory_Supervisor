"""Builds the Phase 1 MuJoCo model from the vendored Unitree G1 with-hands MJCF.

The vendored file at assets/robots/g1/g1_with_hands.xml (mujoco_menagerie,
BSD-3-Clause) is never edited directly. All Phase-1-specific structure is
added programmatically via mujoco.MjSpec, so the original asset stays
untouched and can be diffed/updated independently:

- Lower-body fixing: the floating_base_joint (a freejoint on the pelvis) is
  deleted, making the pelvis a static zero-DOF body welded to the world by
  construction.

  An <equality> weld constraint (pelvis kept as a free body, held by a soft
  constraint) was tried first and empirically produced a persistent ~0.3m /
  0.2rad oscillation over 2000 steps even with all actuators at zero
  control, because the constraint is a spring-damper fighting gravity and
  the legs' own position-actuator gains rather than an exact restriction.
  Deleting the joint outright removes the degree of freedom entirely: zero
  drift, zero oscillation, verified over 2000 steps (see Phase 1 report).

- End-effector sites: the stock model has no site at the hands. One is
  added per wrist at the same local offset as the existing
  left/right_hand_palm_link geom, so it sits at the palm center.

- Table + manipulation object: not present in the stock model, added as new
  worldbody children.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from humanoid_learning.envs import task_config as tc


def build_model(config) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(config.g1_xml_path))

    # --- lower-body fixing -------------------------------------------------
    freejoint = spec.joint(tc.FLOATING_BASE_JOINT)
    spec.delete(freejoint)

    # The "stand" keyframe was authored for the 50-dof floating-base model
    # (7 free-joint qpos + 43 actuated joint qpos). Drop the leading 7
    # entries so it matches the new 43-dof fixed-base model.
    stand_key = spec.key(tc.STAND_KEYFRAME)
    stand_key.qpos = np.asarray(stand_key.qpos)[7:].tolist()

    # --- end-effector sites --------------------------------------------------
    spec.body("left_wrist_yaw_link").add_site(
        name=tc.LEFT_EE_SITE, pos=list(tc.LEFT_EE_OFFSET), size=[0.01, 0.01, 0.01]
    )
    spec.body("right_wrist_yaw_link").add_site(
        name=tc.RIGHT_EE_SITE, pos=list(tc.RIGHT_EE_OFFSET), size=[0.01, 0.01, 0.01]
    )

    # --- floor (not present in the bare robot XML) --------------------------
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        pos=[0, 0, 0],
        rgba=[0.28, 0.32, 0.37, 1],
    )

    # --- table ---------------------------------------------------------------
    table = spec.worldbody.add_body(name=tc.TABLE_BODY, pos=list(config.table_pos))
    table.add_geom(
        name=tc.TABLE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(config.table_half_size),
        rgba=[0.55, 0.4, 0.25, 1],
    )

    # --- object (free body; pose is set explicitly every reset(), not via
    # the pos below -- this initial pos is only the compile-time default) ---
    obj = spec.worldbody.add_body(
        name=tc.OBJECT_BODY,
        pos=[config.table_pos[0], config.table_pos[1], config.object_z],
    )
    obj.add_freejoint(name=tc.OBJECT_JOINT)
    obj.add_geom(
        name=tc.OBJECT_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[tc.OBJECT_HALF_SIZE] * 3,
        rgba=[0.85, 0.15, 0.15, 1],
        mass=0.1,
    )

    return spec.compile()
