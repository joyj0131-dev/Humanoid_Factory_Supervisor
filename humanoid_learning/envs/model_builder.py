"""Canonical MuJoCo builders for bare Unitree G1 + Sharpa Wave.

Vendored assets are never edited directly.  Sharpa, sites and task objects
are attached programmatically with :class:`mujoco.MjSpec`.

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

- End-effector sites are added on each wrist for the common reach contract.

- Table + manipulation object: not present in the stock model, added as new
  worldbody children.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc


def _restore_named_keyframe(
    model: mujoco.MjModel,
    source_xml_path: str | Path,
    key_name: str = tc.STAND_KEYFRAME,
) -> None:
    """Rebuild an attached model's keyframe by joint/actuator *name*.

    ``MjSpec.attach`` inserts child joints at the attachment body's position
    in the kinematic tree.  A pre-existing flat ``key.qpos`` array is only
    padded, however; it is not reliably remapped to that new order.  With the
    bare G1 this silently assigned right-arm stand values to the newly
    inserted left Sharpa thumb and left the object freejoint at the origin.

    Joints absent from the source (Sharpa and the task object) retain their
    compiled ``qpos0``.  Actuators absent from the source retain zero control.
    """
    source = mujoco.MjModel.from_xml_path(str(source_xml_path))
    source_key = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_KEY, key_name)
    target_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if source_key < 0 or target_key < 0:
        raise ValueError(f"missing keyframe {key_name!r} while rebuilding attached model")

    rebuilt_qpos = model.qpos0.copy()
    for target_jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, target_jid)
        source_jid = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_JOINT, name)
        if source_jid < 0:
            continue
        target_type = model.jnt_type[target_jid]
        source_type = source.jnt_type[source_jid]
        if target_type != source_type:
            continue
        width = {
            mujoco.mjtJoint.mjJNT_FREE: 7,
            mujoco.mjtJoint.mjJNT_BALL: 4,
            mujoco.mjtJoint.mjJNT_SLIDE: 1,
            mujoco.mjtJoint.mjJNT_HINGE: 1,
        }[mujoco.mjtJoint(target_type)]
        target_adr = model.jnt_qposadr[target_jid]
        source_adr = source.jnt_qposadr[source_jid]
        rebuilt_qpos[target_adr : target_adr + width] = source.key_qpos[source_key, source_adr : source_adr + width]
    model.key_qpos[target_key] = rebuilt_qpos

    rebuilt_ctrl = np.zeros(model.nu, dtype=float)
    for target_aid in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, target_aid)
        source_aid = mujoco.mj_name2id(source, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if source_aid >= 0:
            rebuilt_ctrl[target_aid] = source.key_ctrl[source_key, source_aid]
    model.key_ctrl[target_key] = rebuilt_ctrl


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

    attach_sharpa_hands(spec, visual_style="g1")

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

    model = spec.compile()
    _restore_named_keyframe(model, config.g1_xml_path)
    return model


# ---------------------------------------------------------------------------
# Phase 4: whole-body (floating-base) and planar-debug builders.
#
# Both start from the same vendored bare G1 and attach the canonical Sharpa
# hands. They add the same floor/EE-site/table/object pieces as build_model()
# above, but unlike build_model() they do NOT delete floating_base_joint --
# see PROJECT_CONTEXT.md Phase 4, Section B. The Foundation's public
# observation/action contract is preserved and verified by regression tests;
# its internal robot model is intentionally migrated to Sharpa.
# ---------------------------------------------------------------------------


def _add_ee_sites(spec: "mujoco.MjSpec") -> None:
    spec.body("left_wrist_yaw_link").add_site(
        name=tc.LEFT_EE_SITE, pos=list(tc.LEFT_EE_OFFSET), size=[0.01, 0.01, 0.01]
    )
    spec.body("right_wrist_yaw_link").add_site(
        name=tc.RIGHT_EE_SITE, pos=list(tc.RIGHT_EE_OFFSET), size=[0.01, 0.01, 0.01]
    )


def _add_palm_sites(spec: "mujoco.MjSpec") -> None:
    """The wrist "palm frame" sites ONLY -- see whole_body_config.py for
    the empirically-derived convention (approach/closing/lateral axes).
    This frame is defined on {side}_wrist_yaw_link itself, so it is
    reusable independently of the Sharpa mesh attachment."""
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
    attach_sharpa_hands(spec, visual_style="g1")
    _add_ee_sites(spec)
    _add_sharpa_grasp_sites(spec)
    _add_floor(spec)
    if include_object:
        _add_table_and_object(spec, config)
    model = spec.compile()
    _restore_named_keyframe(model, config.g1_xml_path)
    return model


def _apply_compliant_kp(model: mujoco.MjModel, joint_names: list[str], kp: float) -> None:
    """Lowers the given actuators' position gain from whatever the stock
    XML set (kp=500 for the G1 arms) to ``kp`` (grasp-model-only
    compliance -- see grasp_config.py and PROJECT_CONTEXT.md Phase 4,
    Section 6). kv is rescaled by sqrt(kp / old_kp) to keep the same
    dampratio=1 critical-damping relationship the stock XML's
    <position dampratio="1"/> establishes at kp=500, rather than becoming
    under- or over-damped at the new kp.

    Sharpa finger gains remain those authored by the vendor model; this
    helper is currently called only for the 14 G1 arm joints."""
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

    attach_sharpa_hands(spec, visual_style="g1")
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

    model = spec.compile()
    _restore_named_keyframe(model, config.g1_xml_path)
    return model


# ---------------------------------------------------------------------------
# Sharpa Wave end-effector attachment.
#
# Mount-transform derivation (Stage 2): the existing, empirically-validated
# "palm frame" convention (whole_body_config.py) defines, for a site on
# {side}_wrist_yaw_link: approach axis = wrist local +X (fingers point
# outward), closing axis = wrist local -Y (left) / +Y (right), lateral
# axis = approach x closing. The vendored Sharpa MJCF has its OWN local
# convention instead (measured directly from the compiled standalone
# model, see assets/robots/sharpa_wave/README.md): fingers extend along
# the hand root body's own local +Z, and the four non-thumb fingers are
# laterally spread along the root's own local +Y. To reuse the validated
# palm frame rather than inventing a new one, SHARPA_TO_PALM_AXES maps
# Sharpa's local frame onto the palm frame's axes (Z_sharpa -> X_palm
# "approach", Y_sharpa -> Z_palm "lateral", and X_sharpa -> Y_palm
# "closing" follows from requiring a proper right-handed rotation) --
# composed with each side's own already-mirrored LEFT/RIGHT_PALM_LOCAL_QUAT
# to get the final wrist-local mount quaternion for that hand.
#
# Verified this session (scripts/test_sharpa_g1_integration.py): compiles,
# no NaN, left/right are exact mirrors (within ~10 micrometers of solver
# noise), ZERO real self-collision anywhere in the model at the stand
# pose (table<->object resting contact excluded -- see that test file's
# docstring), and the wrist drifts <8mm over 3 simulated seconds even at
# this env's COMPLIANT grasp-model gain (arm_kp=120, softer than the
# stock XML's kp=500) holding the stand pose against the added
# ~1.25kg-per-hand weight -- i.e. no arm-sag problem was found.
_SHARPA_TO_PALM_AXES = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])  # columns: R@ex_sharpa=(0,1,0)=Y_palm(closing), R@ey_sharpa=(0,0,1)=Z_palm(lateral), R@ez_sharpa=(1,0,0)=X_palm(approach)

_SHARPA_MOUNT_POS = (0.0, 0.0, 0.0)  # at the wrist_yaw_link's own origin -- the
# _with_wrist Sharpa variant already models the physical wrist-adapter
# standoff as part of its own geometry (see assets/robots/sharpa_wave/README.md)

def _sharpa_xml_path(side: str, mount: str = "wrist") -> Path:
    """``mount``: "wrist" (default, existing behavior) or "flange" -- see
    Session 40's mount A/B audit (docs/history/PHASE4_GRASP_SESSION_40.md).
    Measured directly from the vendored XML: both variants add a RIGID
    (zero extra DoF) hand-base standoff of nearly identical thickness
    (with_wrist: +29.0mm; with_flange: +29.5mm, i.e. with_flange is 0.5mm
    MORE, via an additional thin flange plate mesh, not less) -- neither
    is a "no adapter" bare-mount option, and the difference between them
    is not the wrist-duplication most of the visual asymmetry was
    hypothesized to come from."""
    project_root = Path(__file__).resolve().parents[2]
    suffix = "with_flange" if mount == "flange" else "with_wrist"
    return project_root / "assets" / "robots" / "sharpa_wave" / f"{side}_sharpa_wave" / f"{side}_sharpa_wave_{suffix}.xml"


def _sharpa_mount_quat(side: str) -> list[float]:
    palm_quat = wbc.LEFT_PALM_LOCAL_QUAT if side == "left" else wbc.RIGHT_PALM_LOCAL_QUAT
    r_wrist_palm = np.zeros(9)
    mujoco.mju_quat2Mat(r_wrist_palm, np.array(palm_quat, dtype=float))
    r_wrist_palm = r_wrist_palm.reshape(3, 3)
    r_wrist_sharpa = r_wrist_palm @ _SHARPA_TO_PALM_AXES
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, r_wrist_sharpa.flatten())
    return q.tolist()


# Reused from the bare G1 material definitions.
# <material> definitions (rgba only -- see that file's "metal"/"black"
# entries) rather than inventing new colors, so the Sharpa hands visually
# match the rest of G1 instead of the vendored light-lavender
# (0.79216 0.81961 0.93333) shell and bright-green (0.2 1 0.2) elastomer.
_G1_METAL_RGBA = [0.7, 0.7, 0.7, 1.0]
_G1_BLACK_RGBA = [0.2, 0.2, 0.2, 1.0]


def _apply_sharpa_visual_style(spec: "mujoco.MjSpec", side: str, style: str) -> None:
    """[Session 41 rewrite] VISUAL geoms only (contype==0==conaffinity, the
    vendored XML's own "no collision participation" convention for its
    *_visual duplicates -- see attach_sharpa_hands's docstring). Never
    touches a geom that participates in collision (contype/conaffinity,
    friction, solref) or mass/inertia -- recoloring is purely cosmetic,
    verified by a physics-invariance regression test.

    Session 40's version only set geom.rgba to a value CLOSE to G1's own
    metal (0.7,0.7,0.7) -- visually almost indistinguishable from the
    vendored lavender (0.79,0.82,0.93) under the viewer's lighting (user-
    confirmed: "still looks the same"). This version instead ASSIGNS the
    geom to G1's own actual named material ("black"/"metal", already
    defined in the bare G1 spec) --
    verifiable post-compile via geom_matid, not just an approximately-
    matching rgba -- and, per the user's explicit palette split, uses
    "black" (not "metal") for the LARGE housing/base surfaces (the
    dominant visible area) so the change is unmistakable, not just the
    small elastomer pads:
      - "wrist"/"C_MC" mesh (wrist adapter + palm/hand-base housing,
        Sharpa's own naming -- see assets/robots/sharpa_wave's vendored
        mesh list) -> G1 "black"
      - "elastomer" mesh (fingertip contact pads) -> G1 "black"
      - everything else (VL/PP/MP/DP finger structural links) -> G1
        "metal"
    Also sets geom.rgba to the SAME material's own rgba explicitly (not
    left at the vendored per-geom rgba, which would otherwise win over an
    assigned material at render time) so the visible result does not
    depend on MuJoCo's material-vs-rgba precedence rule."""
    if style != "g1":
        return
    prefix = f"{side}_{side}_"
    for body in spec.bodies:
        if not body.name.startswith(prefix):
            continue
        for g in body.geoms:
            if g.contype != 0 or g.conaffinity != 0:
                continue  # collision-participating geom -- never recolored
            mesh = (g.meshname or "").lower()
            name = (g.name or "").lower()
            is_housing = "wrist" in mesh or "c_mc" in mesh or "wrist" in name or "c_mc" in name
            is_elastomer = "elastomer" in mesh or "elastomer" in name
            if is_housing or is_elastomer:
                g.material = "black"
                g.rgba = _G1_BLACK_RGBA
            else:
                g.material = "metal"
                g.rgba = _G1_METAL_RGBA


def attach_sharpa_hands(spec: "mujoco.MjSpec", mount: str = "wrist", visual_style: str = "upstream") -> None:
    """Attach a vendored Sharpa Wave hand to each bare-G1 wrist.

    The stock rubber-hand visual is removed; the real wrist joint housing
    remains.  Sharpa's own
    shared (non-side-prefixed) mesh names (e.g. "MCP_VL", "elastomer")
    collide between left/right if both are attached to the same spec, so
    MjSpec.attach's ``prefix`` is used -- this makes the resulting body/
    joint names "{side}_{side}_..." (e.g. "left_left_thumb_CMC_FE"), a
    known cosmetic redundancy from the already-side-prefixed source XML,
    not a bug (verified: does not collide with any other name)."""
    for side in ("left", "right"):
        wrist = spec.body(f"{side}_wrist_yaw_link")
        for g in list(wrist.geoms):
            if g.meshname == f"{side}_rubber_hand":
                spec.delete(g)
        mount_site = wrist.add_site(
            name=f"{side}_sharpa_mount", pos=list(_SHARPA_MOUNT_POS), quat=_sharpa_mount_quat(side)
        )
        sharpa_spec = mujoco.MjSpec.from_file(str(_sharpa_xml_path(side, mount)))
        spec.attach(sharpa_spec, prefix=f"{side}_", site=mount_site)
        _apply_sharpa_visual_style(spec, side, visual_style)


def _add_sharpa_grasp_sites(spec: "mujoco.MjSpec") -> None:
    """[35th session, Stage 4] The palm-frame sites (reused unchanged --
    see _add_palm_sites) plus a fingertip reference site for each of the
    10 Sharpa fingertips, placed at that fingertip's real elastomer
    collision geom's own local position."""
    from humanoid_learning.envs import sharpa_config as sc

    _add_palm_sites(spec)
    for side in sc.SIDES:
        for finger in sc.FINGERS:
            dp_body = sc.sharpa_body(side, finger, "DP")
            elastomer_geom_name = sc.sharpa_elastomer_geom(side, finger)
            geom = next(g for g in spec.body(dp_body).geoms if g.name == elastomer_geom_name)
            spec.body(dp_body).add_site(
                name=f"{side}_{finger}_sharpa_tip", pos=list(geom.pos), size=[0.004, 0.004, 0.004]
            )


def build_grasp_model_sharpa(config) -> mujoco.MjModel:
    """Build the canonical fixed-base G1 + Sharpa grasp model."""
    spec = mujoco.MjSpec.from_file(str(config.g1_xml_path))

    freejoint = spec.joint(tc.FLOATING_BASE_JOINT)
    spec.delete(freejoint)
    stand_key = spec.key(tc.STAND_KEYFRAME)
    stand_key.qpos = np.asarray(stand_key.qpos)[7:].tolist()

    attach_sharpa_hands(spec, mount=config.sharpa_mount, visual_style=config.sharpa_visual_style)
    _add_ee_sites(spec)
    _add_sharpa_grasp_sites(spec)
    _add_floor(spec)

    table = spec.worldbody.add_body(name=tc.TABLE_BODY, pos=list(config.table_pos))
    table.add_geom(
        name=tc.TABLE_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(config.table_half_size),
        rgba=[0.55, 0.4, 0.25, 1],
    )
    object_half_extents = config.effective_object_half_extents
    obj_z = config.table_pos[2] + config.table_half_size[2] + object_half_extents[2] + tc.OBJECT_TABLE_GAP
    obj = spec.worldbody.add_body(
        name=tc.OBJECT_BODY,
        pos=[config.object_pos[0], config.object_pos[1], obj_z],
    )
    obj.add_freejoint(name=tc.OBJECT_JOINT)
    obj.add_geom(
        name=tc.OBJECT_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(object_half_extents),
        rgba=[0.85, 0.15, 0.15, 1],
        mass=config.object_mass,
        friction=list(config.object_friction),
    )

    model = spec.compile()
    _restore_named_keyframe(model, config.g1_xml_path)
    _apply_compliant_kp(model, tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS, config.arm_kp)
    return model
