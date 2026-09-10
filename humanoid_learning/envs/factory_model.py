"""Compile the conveyor-line factory scene: floating-base G1 + Sharpa plus one
conveyor carrying two independent scripted arm stations.

Reuses ``model_builder``'s Sharpa attachment, EE/grasp sites and floor verbatim,
so the robot half of this model is the same robot the whole-body and grasp envs
already use. Only the scene around it is new. Every station body, joint,
actuator and geom is prefixed ``wc{index}_`` so the two stations can never share
state by accident.

Each arm carries a real two-jaw gripper on slide joints with friction, so it
physically picks the part up rather than miming over it.
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


# MuJoCo's filterparent flag does NOT exclude an arm link from its own column:
# the column has no joint, so it is welded to the world, and the pair is treated
# as link-vs-world, which filterparent deliberately never filters. Left alone,
# the turret collided with its own column at 2.6e17 N and jammed base_yaw.
# Putting every arm geom in contype=2/conaffinity=1 makes the arm's own links
# transparent to each other while still colliding with the part, the belt, the
# floor and the robot -- which are all in the default group.
ARM_CONTYPE = 2
ARM_CONAFFINITY = 1


def _add_gripper(spec, wrist, index: int) -> None:
    """Two parallel jaws on slide joints, closing along the wrist's local y.

    Both jaws take the same commanded half-opening: the right jaw's slide axis
    is negated so one control value means "how far open" for both.
    """
    wrist.add_geom(
        name=fcfg.arm_body_name(index, "palm_geom"),
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.030, 0.045, 0.030],
        pos=[0.035, 0.0, 0.0],
        rgba=[0.30, 0.33, 0.38, 1],
        contype=ARM_CONTYPE,
        conaffinity=ARM_CONAFFINITY,
    )
    wrist.add_site(
        name=fcfg.arm_body_name(index, "grasp"),
        pos=[fcfg.GRIPPER_LENGTH, 0.0, 0.0],
        size=[0.012, 0.012, 0.012],
        rgba=[0.1, 0.9, 0.3, 1],
    )
    for suffix, sign in zip(fcfg.GRIPPER_JOINT_SUFFIXES, (1.0, -1.0)):
        jaw = wrist.add_body(name=fcfg.arm_body_name(index, suffix), pos=[fcfg.GRIPPER_LENGTH, 0.0, 0.0])
        jaw.add_joint(
            name=fcfg.arm_joint_name(index, suffix),
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0.0, sign, 0.0],
            range=list(fcfg.GRIPPER_JOINT_RANGE),
        )
        jaw.add_geom(
            name=fcfg.arm_body_name(index, f"{suffix}_geom"),
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(fcfg.GRIPPER_JAW_HALF_SIZE),
            rgba=[0.20, 0.22, 0.26, 1],
            friction=list(fcfg.GRIPPER_FRICTION),
            solref=list(fcfg.GRIPPER_SOLREF),
            contype=ARM_CONTYPE,
            conaffinity=ARM_CONAFFINITY,
            mass=0.05,
        )
        joint = fcfg.arm_joint_name(index, suffix)
        _add_position_actuator(spec, joint, joint, fcfg.GRIPPER_KP, fcfg.GRIPPER_KV, fcfg.GRIPPER_JOINT_RANGE)


def _add_automation_arm(spec, pose: fcfg.WorkcellPose) -> None:
    """4-DoF arm (base yaw, shoulder, elbow, wrist) plus a two-jaw gripper.

    The column is mounted so that ``base_yaw = 0`` points the arm back across
    the belt toward its pick spot, and ``wrist_pitch`` holds the gripper
    vertical so the jaws descend onto a part from above.
    """
    index = pose.index
    base_xy = pose.arm_base_xy
    mount_yaw = pose.heading_rad + math.pi

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
        contype=ARM_CONTYPE,
        conaffinity=ARM_CONAFFINITY,
    )

    turret = column.add_body(name=fcfg.arm_body_name(index, "turret"), pos=[0.0, 0.0, fcfg.ARM_TURRET_Z])
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
        contype=ARM_CONTYPE,
        conaffinity=ARM_CONAFFINITY,
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
        contype=ARM_CONTYPE,
        conaffinity=ARM_CONAFFINITY,
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
        contype=ARM_CONTYPE,
        conaffinity=ARM_CONAFFINITY,
    )

    wrist = fore.add_body(name=fcfg.arm_body_name(index, "wrist"), pos=[fcfg.ARM_FORE_LENGTH, 0.0, 0.0])
    wrist.add_joint(
        name=fcfg.arm_joint_name(index, "wrist_pitch"),
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 1, 0],
        range=list(fcfg.ARM_JOINT_RANGES[3]),
    )
    _add_gripper(spec, wrist, index)

    for suffix, ctrlrange in zip(fcfg.ARM_JOINT_SUFFIXES, fcfg.ARM_JOINT_RANGES):
        joint = fcfg.arm_joint_name(index, suffix)
        _add_position_actuator(spec, joint, joint, fcfg.ARM_KP, fcfg.ARM_KV, ctrlrange)


def _add_conveyor(spec) -> None:
    belt = spec.worldbody.add_body(name=fcfg.BELT_BODY, pos=[fcfg.CONVEYOR_CENTRE_X, 0.0, fcfg.BELT_BODY_Z])
    belt.add_geom(
        name=fcfg.BELT_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(fcfg.BELT_HALF_SIZE),
        rgba=[0.30, 0.32, 0.36, 1],
        friction=[1.0, 0.01, 0.0001],
    )
    # Side rails, so the line reads as a conveyor rather than a long table.
    for sign, tag in ((-1.0, "m"), (1.0, "p")):
        rail = spec.worldbody.add_body(
            name=f"conveyor_rail_{tag}",
            pos=[
                fcfg.CONVEYOR_CENTRE_X + sign * (fcfg.BELT_HALF_SIZE[0] + 0.02),
                0.0,
                fcfg.TABLE_TOP_Z + 0.03,
            ],
        )
        rail.add_geom(
            name=f"conveyor_rail_{tag}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.02, fcfg.BELT_HALF_SIZE[1], 0.03],
            rgba=[0.55, 0.57, 0.60, 1],
        )


def _add_station(spec, pose: fcfg.WorkcellPose, config: fcfg.FactoryConfig) -> None:
    index = pose.index

    manip = np.asarray(pose.manipulation_xy, dtype=float)
    marker = spec.worldbody.add_body(name=f"wc{index}_manip_marker", pos=[float(manip[0]), float(manip[1]), 0.0])
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

    stopper_xy = pose.stopper_xy
    stopper = spec.worldbody.add_body(
        name=fcfg.stopper_body_name(index),
        pos=[float(stopper_xy[0]), float(stopper_xy[1]), fcfg.TABLE_TOP_Z + fcfg.STOPPER_HALF_SIZE[2]],
        quat=_yaw_quat(pose.heading_rad),
    )
    stopper.add_geom(
        name=fcfg.stopper_geom_name(index),
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(fcfg.STOPPER_HALF_SIZE),
        rgba=[0.70, 0.72, 0.75, 1],
    )

    beacon = spec.worldbody.add_body(
        name=fcfg.beacon_body_name(index),
        pos=[float(pose.arm_base_xy[0]), float(pose.arm_base_xy[1]), fcfg.ARM_TURRET_Z + 0.25],
    )
    beacon.add_geom(
        name=fcfg.beacon_geom_name(index),
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.09],
        rgba=list(fcfg.BEACON_RUNNING_RGBA),
        contype=0,
        conaffinity=0,
    )

    _add_automation_arm(spec, pose)


def build_factory_model(config: fcfg.FactoryConfig) -> mujoco.MjModel:
    """G1 + Sharpa on its real floating base, plus a conveyor with two stations.

    The floating_base_joint is KEPT (as in ``build_whole_body_model``): the base
    is free, so any navigation along the line has to come from the legs.
    """
    spec = mujoco.MjSpec.from_file(str(config.g1_xml_path))
    model_builder.attach_sharpa_hands(spec, visual_style="g1")
    model_builder._add_ee_sites(spec)
    model_builder._add_sharpa_grasp_sites(spec)
    model_builder._add_floor(spec)

    _add_conveyor(spec)
    for pose in config.workcells:
        _add_station(spec, pose, config)

    # The line spans several metres, so the stock 640x480 offscreen framebuffer
    # is too narrow to show both stations at once when rendering headless.
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    model = spec.compile()
    model_builder._restore_named_keyframe(model, config.g1_xml_path)
    return model
