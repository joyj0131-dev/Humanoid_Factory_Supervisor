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
from humanoid_learning.envs import whole_body_config as wbc


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


# ---------------------------------------------------------------------------
# Phase 4: whole-body (floating-base) and planar-debug builders.
#
# Both start from the same vendored g1_with_hands.xml and add the same
# floor/EE-site/table/object pieces as build_model() above, but unlike
# build_model() they do NOT delete floating_base_joint -- see
# PROJECT_CONTEXT.md Phase 4, Section B. build_model() itself is untouched
# above (Foundation preserved byte-for-byte; verified by regression tests).
# ---------------------------------------------------------------------------


def _add_ee_sites(spec: "mujoco.MjSpec") -> None:
    spec.body("left_wrist_yaw_link").add_site(
        name=tc.LEFT_EE_SITE, pos=list(tc.LEFT_EE_OFFSET), size=[0.01, 0.01, 0.01]
    )
    spec.body("right_wrist_yaw_link").add_site(
        name=tc.RIGHT_EE_SITE, pos=list(tc.RIGHT_EE_OFFSET), size=[0.01, 0.01, 0.01]
    )


def _add_grasp_sites(spec: "mujoco.MjSpec") -> None:
    """Palm-frame and fingertip reference sites -- see whole_body_config.py
    for the empirically-derived convention (approach/closing/lateral axes).
    Never modifies the stock XML; added via MjSpec like _add_ee_sites."""
    spec.body("left_wrist_yaw_link").add_site(
        name=wbc.LEFT_PALM_SITE,
        pos=list(wbc.LEFT_PALM_LOCAL_POS),
        quat=list(wbc.LEFT_PALM_LOCAL_QUAT),
        size=[0.008, 0.008, 0.008],
    )
    spec.body("right_wrist_yaw_link").add_site(
        name=wbc.RIGHT_PALM_SITE,
        pos=list(wbc.RIGHT_PALM_LOCAL_POS),
        quat=list(wbc.RIGHT_PALM_LOCAL_QUAT),
        size=[0.008, 0.008, 0.008],
    )
    for site_name, body_name in wbc.FINGERTIP_SITE_BODIES.items():
        spec.body(body_name).add_site(
            name=site_name, pos=list(wbc.FINGERTIP_SITE_LOCAL_POS[site_name]), size=[0.004, 0.004, 0.004]
        )


def _add_floor(spec: "mujoco.MjSpec") -> None:
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        pos=[0, 0, 0],
        rgba=[0.28, 0.32, 0.37, 1],
    )


def _add_table_and_object(spec: "mujoco.MjSpec", config) -> None:
    table = spec.worldbody.add_body(name=tc.TABLE_BODY, pos=list(config.table_pos))
    table.add_geom(
        name=tc.TABLE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(config.table_half_size),
        rgba=[0.55, 0.4, 0.25, 1],
    )
    obj = spec.worldbody.add_body(
        name=tc.OBJECT_BODY,
        pos=[
            config.table_pos[0],
            config.table_pos[1],
            config.table_pos[2] + config.table_half_size[2] + tc.OBJECT_HALF_SIZE + tc.OBJECT_TABLE_GAP,
        ],
    )
    obj.add_freejoint(name=tc.OBJECT_JOINT)
    obj.add_geom(
        name=tc.OBJECT_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[tc.OBJECT_HALF_SIZE] * 3,
        rgba=[0.85, 0.15, 0.15, 1],
        mass=0.1,
        # default friction (1 0.005 0.0001) is too low for a fingertip-sphere
        # grasp to hold the box without slipping -- see Phase 4 grasp report.
        friction=[1.5, 0.01, 0.0001],
    )


def build_whole_body_model(config, include_object: bool = False) -> mujoco.MjModel:
    """Floating-base whole-body model: floating_base_joint is kept (NOT
    deleted), so legs/waist/pelvis are all free to move under their own
    position actuators + gravity + foot contact, exactly like the real
    robot. The stand keyframe qpos is used as-authored (no 7-entry strip --
    it already has the right length for a 50-dof floating-base model)."""
    spec = mujoco.MjSpec.from_file(str(config.g1_xml_path))
    _add_ee_sites(spec)
    _add_grasp_sites(spec)
    _add_floor(spec)
    if include_object:
        _add_table_and_object(spec, config)
    return spec.compile()


def build_grasp_model(config, hard_fixed_waist: bool = False) -> mujoco.MjModel:
    """Fixed-base grasp validation model: same lower-body-fixing approach as
    build_model() (Foundation, proven stable), but the object's mass/
    friction/size are configurable (GraspEnvConfig) instead of hard-coded,
    and object placement is a single deterministic point (not a randomized
    range) -- this env validates grasp PHYSICS, not reaching generalization.

    ``hard_fixed_waist`` (Net-Torque Root Cause Isolation session, default
    False -- the existing, unchanged behavior): when True, adds a physical
    <equality joint> constraint pinning each of the 3 waist joints to their
    stand-keyframe value, via the constraint solver (not a per-step qpos
    teleport -- the waist actuators/kp are untouched, this is a genuine
    holonomic constraint MuJoCo enforces every physics substep). Used ONLY
    by the grasp-only diagnostic/experimental path (--grasp-experimental,
    scripts/test_grasp.py's waist A/B comparison) to test whether the
    small (~0.088 rad measured) waist drift under compliant position
    control is a meaningful contributor to object net torque -- the
    default fixed-base grasp model (hard_fixed_waist=False, what --grasp
    and every existing test still use) is completely unaffected."""
    spec = mujoco.MjSpec.from_file(str(config.g1_xml_path))

    freejoint = spec.joint(tc.FLOATING_BASE_JOINT)
    spec.delete(freejoint)
    stand_key = spec.key(tc.STAND_KEYFRAME)
    stand_qpos = np.asarray(stand_key.qpos)[7:].tolist()
    stand_key.qpos = stand_qpos

    _add_ee_sites(spec)
    _add_grasp_sites(spec)
    _add_floor(spec)

    table = spec.worldbody.add_body(name=tc.TABLE_BODY, pos=list(config.table_pos))
    table.add_geom(
        name=tc.TABLE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(config.table_half_size),
        rgba=[0.55, 0.4, 0.25, 1],
    )
    obj_z = config.table_pos[2] + config.table_half_size[2] + config.object_half_size + tc.OBJECT_TABLE_GAP
    obj = spec.worldbody.add_body(
        name=tc.OBJECT_BODY,
        pos=[config.object_pos[0], config.object_pos[1], obj_z],
    )
    obj.add_freejoint(name=tc.OBJECT_JOINT)
    obj.add_geom(
        name=tc.OBJECT_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[config.object_half_size] * 3,
        rgba=[0.85, 0.15, 0.15, 1],
        mass=config.object_mass,
        friction=list(config.object_friction),
    )

    if hard_fixed_waist:
        # spec.joint(name).qpos0 order matches tc's own qpos layout (post
        # freejoint deletion) -- find each waist joint's stand-keyframe
        # value by its position in the (now-headless) stand_qpos list via
        # the SAME joint-name-to-index walk _apply_compliant_kp/the env
        # itself use elsewhere (jnt_qposadr), done here at MjSpec level by
        # re-deriving it from the compiled joint order below instead
        # (simpler: pin to the joint's own default qpos0, which for a
        # freshly-compiled spec still reflects the stand keyframe only if
        # explicitly set -- so pin to the ACTUAL stand_qpos value looked
        # up by name, computed after a throwaway compile pass would be
        # circular; instead locate each waist joint's index directly in
        # tc's known qpos ordering for this fixed-base spec).
        all_joint_names_in_order = [j.name for j in spec.joints]
        for wj in wbc.WAIST_JOINTS:
            idx = all_joint_names_in_order.index(wj)
            pin_value = float(stand_qpos[idx])
            spec.add_equality(
                type=mujoco.mjtEq.mjEQ_JOINT,
                name1=wj,
                objtype=mujoco.mjtObj.mjOBJ_JOINT,
                data=[pin_value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )

    model = spec.compile()
    _apply_compliant_kp(model, tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS, config.arm_kp)
    finger_joints = [n for n, _, _ in wbc.LEFT_HAND_SYNERGY_TARGETS + wbc.RIGHT_HAND_SYNERGY_TARGETS]
    _apply_compliant_kp(model, finger_joints, config.hand_kp)
    return model


def _apply_compliant_kp(model: mujoco.MjModel, joint_names: list[str], kp: float) -> None:
    """Lowers the given actuators' position gain from whatever the stock
    XML set (kp=500 for both arms and hands) to ``kp`` (grasp-model-only
    compliance -- see grasp_config.py and PROJECT_CONTEXT.md Phase 4,
    Section 6). kv is rescaled by sqrt(kp / old_kp) to keep the same
    dampratio=1 critical-damping relationship the stock XML's
    <position dampratio="1"/> establishes at kp=500, rather than becoming
    under- or over-damped at the new kp.

    Originally applied only to the 14 arm joints; extended to the finger
    joints too after diagnosing (Phase 4 Grasp Track redesign) that a
    STIFF kp=500 finger actuator was the actual cause of an 8-13N force
    spike appearing within a single control step the instant a fingertip
    first touched the object -- the compliant arm was never the
    bottleneck for that particular spike, the untouched finger actuators
    were."""
    for name in joint_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, f"actuator not found: {name}"
        old_kp = model.actuator_gainprm[aid, 0]
        old_kv = -model.actuator_biasprm[aid, 2]
        scale = np.sqrt(kp / old_kp)
        model.actuator_gainprm[aid, 0] = kp
        model.actuator_biasprm[aid, 1] = -kp
        model.actuator_biasprm[aid, 2] = -(old_kv * scale)


def build_planar_debug_model(config) -> mujoco.MjModel:
    """DEBUG/FALLBACK ONLY -- see PROJECT_CONTEXT.md Section 8. Deletes
    floating_base_joint (like the fixed-base builder) and replaces it with
    3 new actuated joints directly on the pelvis body: planar_x (slide),
    planar_y (slide), planar_yaw (hinge). Legs/waist/arms are held rigid at
    the stand pose (their position actuators simply hold the stand ctrl,
    exactly like the fixed-base env's _held_act_ids mechanism) -- nothing
    about this is a walking gait. z height is not an actuated DOF: the
    pelvis body's static height offset (0.793m, same as the fixed-base
    model) is used, so the whole rigid body "slides" at a constant height
    while x/y/yaw are tracked by high-gain position actuators (near-teleport
    tracking, not physically grounded locomotion)."""
    spec = mujoco.MjSpec.from_file(str(config.g1_xml_path))

    freejoint = spec.joint(tc.FLOATING_BASE_JOINT)
    spec.delete(freejoint)
    stand_key = spec.key(tc.STAND_KEYFRAME)
    stand_key.qpos = np.asarray(stand_key.qpos)[7:].tolist()

    pelvis = spec.body("pelvis")
    pelvis.add_joint(name="planar_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
    pelvis.add_joint(name="planar_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
    pelvis.add_joint(name="planar_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])

    _add_ee_sites(spec)
    _add_floor(spec)

    def pad10(values: list[float]) -> list[float]:
        return list(values) + [0.0] * (10 - len(values))

    kp_xy, kv_xy = 3000.0, 300.0
    kp_yaw, kv_yaw = 1000.0, 100.0
    for name, kp, kv, ctrlrange in [
        ("planar_x", kp_xy, kv_xy, list(config.x_range)),
        ("planar_y", kp_xy, kv_xy, list(config.y_range)),
        ("planar_yaw", kp_yaw, kv_yaw, [-3.1416, 3.1416]),
    ]:
        spec.add_actuator(
            name=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=name,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=pad10([kp]),
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=pad10([0, -kp, -kv]),
            ctrllimited=True,
            ctrlrange=ctrlrange,
        )

    return spec.compile()
