"""Compile the two-workcell factory scene: floating-base G1 + Sharpa plus two
independent scripted automation cells.

Reuses ``model_builder``'s Sharpa attachment, EE/grasp sites and floor verbatim,
so the robot half of this model is the same robot the whole-body and grasp envs
already use. Only the scene around it is new. Every workcell body, joint,
actuator and geom is prefixed ``wc{index}_`` so the two cells can never share
state by accident.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from humanoid_learning.envs import factory_config as fcfg
from humanoid_learning.envs import model_builder
from humanoid_learning.envs import task_config as tc


def _pad10(values: list[float]) -> list[float]:
    return list(values) + [0.0] * (10 - len(values))


def _add_position_actuator(spec, name: str, joint: str, kp: float, kv: float, ctrlrange) -> None:
    spec.add_actuator(
        name=name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target=joint,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED,
        gainprm=_pad10([kp]),
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=_pad10([0.0, -kp, -kv]),
        ctrllimited=True,
        ctrlrange=list(ctrlrange),
    )


def _yaw_quat(angle: float) -> list[float]:
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def _add_automation_arm(spec, pose: fcfg.WorkcellPose) -> None:
    """A 3-DoF primitive arm on a static column.

    The column is mounted rotated so that ``base_yaw = 0`` already points the
    arm from its column toward the table centre; that keeps the scripted cycle
    waypoints small and readable instead of all sitting near -pi/2.
    """
    index = pose.index
    base_xy = pose.arm_base_xy
    # Arm +x must point from the column toward the table centre, i.e. along the
    # workcell-local -y direction -> world yaw of (heading - pi/2).
    mount_yaw = pose.heading_rad - math.pi / 2.0

    column = spec.worldbody.add_body(
        name=fcfg.arm_body_name(index, "column"),
        pos=[float(base_xy[0]), float(base_xy[1]), 0.0],
        quat=_yaw_quat(mount_yaw),
    )
    column.add_geom(
        name=fcfg.arm_body_name(index, "column_geom"),
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.07, fcfg.ARM_COLUMN_HALF_HEIGHT],
        pos=[0.0, 0.0, fcfg.ARM_COLUMN_HALF_HEIGHT],
        rgba=[0.35, 0.38, 0.42, 1],
    )

    turret = column.add_body(
        name=fcfg.arm_body_name(index, "turret"),
        pos=[0.0, 0.0, 2.0 * fcfg.ARM_COLUMN_HALF_HEIGHT],
    )
    turret.add_joint(
        name=fcfg.arm_joint_name(index, "base_yaw"),
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        range=list(fcfg.ARM_JOINT_RANGES[0]),
    )
    turret.add_geom(
        name=fcfg.arm_body_name(index, "turret_geom"),
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.06, 0.06, 0.05],
        rgba=[0.45, 0.48, 0.52, 1],
    )

    upper = turret.add_body(name=fcfg.arm_body_name(index, "upper"), pos=[0.0, 0.0, 0.0])
    upper.add_joint(
        name=fcfg.arm_joint_name(index, "shoulder_pitch"),
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
        range=list(fcfg.ARM_JOINT_RANGES[1]),
    )
    upper.add_geom(
        name=fcfg.arm_body_name(index, "upper_geom"),
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.0, 0.0, 0.0, fcfg.ARM_UPPER_LENGTH, 0.0, 0.0],
        size=[fcfg.ARM_LINK_RADIUS],
        rgba=[0.80, 0.55, 0.15, 1],
    )

    fore = upper.add_body(name=fcfg.arm_body_name(index, "fore"), pos=[fcfg.ARM_UPPER_LENGTH, 0.0, 0.0])
    fore.add_joint(
        name=fcfg.arm_joint_name(index, "elbow_pitch"),
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
        range=list(fcfg.ARM_JOINT_RANGES[2]),
    )
    fore.add_geom(
        name=fcfg.arm_body_name(index, "fore_geom"),
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.0, 0.0, 0.0, fcfg.ARM_FORE_LENGTH, 0.0, 0.0],
        size=[fcfg.ARM_LINK_RADIUS * 0.85],
        rgba=[0.85, 0.65, 0.25, 1],
    )
    fore.add_site(
        name=fcfg.arm_body_name(index, "tip"),
        pos=[fcfg.ARM_FORE_LENGTH, 0.0, 0.0],
        size=[0.02, 0.02, 0.02],
        rgba=[0.1, 0.9, 0.3, 1],
    )

    for suffix, ctrlrange, kp, kv in zip(
        fcfg.ARM_JOINT_SUFFIXES,
        fcfg.ARM_JOINT_RANGES,
        (fcfg.ARM_KP,) * 3,
        (fcfg.ARM_KV,) * 3,
    ):
        joint = fcfg.arm_joint_name(index, suffix)
        _add_position_actuator(spec, joint, joint, kp, kv, ctrlrange)


def _add_workcell(spec, pose: fcfg.WorkcellPose, config: fcfg.FactoryConfig) -> None:
    index = pose.index
    table_pos = pose.table_pos

    table = spec.worldbody.add_body(
        name=fcfg.table_body_name(index),
        pos=[float(table_pos[0]), float(table_pos[1]), float(table_pos[2])],
        quat=_yaw_quat(pose.heading_rad),
    )
    table.add_geom(
        name=fcfg.table_geom_name(index),
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(fcfg.LOCAL_TABLE_HALF_SIZE),
        rgba=[0.55, 0.4, 0.25, 1] if index == 0 else [0.42, 0.34, 0.26, 1],
    )

    # Floor marker showing where the G1 must stand to manipulate this cell.
    manip = np.asarray(pose.manipulation_xy, dtype=float)
    marker = spec.worldbody.add_body(
        name=f"wc{index}_manip_marker", pos=[float(manip[0]), float(manip[1]), 0.0]
    )
    marker.add_geom(
        name=f"wc{index}_manip_marker_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.18, 0.004],
        pos=[0.0, 0.0, 0.004],
        rgba=[0.20, 0.70, 0.95, 0.55] if index == 0 else [0.95, 0.75, 0.20, 0.55],
        contype=0,
        conaffinity=0,
    )

    part_xy = pose.canonical_part_xy
    half = config.part_half_size
    part = spec.worldbody.add_body(
        name=fcfg.part_body_name(index),
        pos=[float(part_xy[0]), float(part_xy[1]), fcfg.TABLE_TOP_Z + half + tc.OBJECT_TABLE_GAP],
    )
    part.add_freejoint(name=fcfg.part_joint_name(index))
    part.add_geom(
        name=fcfg.part_geom_name(index),
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half, half, half],
        rgba=[0.85, 0.15, 0.15, 1],
        mass=config.part_mass,
        friction=list(config.part_friction),
    )

    _add_automation_arm(spec, pose)


def build_factory_model(config: fcfg.FactoryConfig) -> mujoco.MjModel:
    """G1 + Sharpa on its real floating base, plus two independent workcells.

    The floating_base_joint is KEPT (as in ``build_whole_body_model``): the base
    is free, so any navigation between cells has to come from the legs.
    """
    spec = mujoco.MjSpec.from_file(str(config.g1_xml_path))
    model_builder.attach_sharpa_hands(spec, visual_style="g1")
    model_builder._add_ee_sites(spec)
    model_builder._add_sharpa_grasp_sites(spec)
    model_builder._add_floor(spec)

    for pose in config.workcells:
        _add_workcell(spec, pose, config)

    model = spec.compile()
    model_builder._restore_named_keyframe(model, config.g1_xml_path)
    return model
