"""Whole-body G1 facts and config, confirmed by directly inspecting
``assets/robots/g1/g1_with_hands.xml`` with the MuJoCo API on 2026-08-28
(see PROJECT_CONTEXT.md Phase 4 report for the full inspection output) --
none of the names/ranges below are guessed.

This module is additive to ``task_config.py`` (Foundation, fixed-base,
14-dim arm-only contract -- left untouched). It defines the joint groups
needed once the floating base is no longer deleted: legs, waist, and the
per-hand finger joints used by hand synergy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from humanoid_learning.envs import task_config as tc

# ---------------------------------------------------------------------------
# Robot facts (confirmed via mujoco.MjModel.from_xml_path + mj_id2name /
# jnt_range / actuator_ctrlrange, not guessed)
# ---------------------------------------------------------------------------

# Joint order below matches the order joints are declared in the XML
# (confirmed via mujoco.mj_id2name over range(model.njnt)).
LEFT_LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
RIGHT_LEG_JOINTS = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
LEG_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS  # 12

WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]  # 3

# 7 finger joints per hand: thumb has 3 (opposition + 2 curl), middle/index
# have 2 each (metacarpal + distal curl) -- no ring/pinky in this model.
LEFT_HAND_JOINTS = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]
RIGHT_HAND_JOINTS = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

# Actuator names are identical to joint names (each <position name="X"
# joint="X"/>), same convention as task_config.py's arm actuators.
LEFT_LEG_ACTUATORS = LEFT_LEG_JOINTS
RIGHT_LEG_ACTUATORS = RIGHT_LEG_JOINTS
WAIST_ACTUATORS = WAIST_JOINTS
LEFT_HAND_ACTUATORS = LEFT_HAND_JOINTS
RIGHT_HAND_ACTUATORS = RIGHT_HAND_JOINTS

PELVIS_BODY = "pelvis"
TORSO_BODY = "torso_link"

# ---------------------------------------------------------------------------
# Palm frame / fingertip sites (added via MjSpec, not in the stock XML --
# see model_builder._add_grasp_sites). Confirmed empirically (Phase 4 Grasp
# Track, 3rd/4th session): finger root bodies attach to the wrist body with
# identity local quat, extending along the wrist's own local +X, and the
# dominant CLOSING motion (open synergy=0 -> closed synergy=1) is along the
# wrist's local Y axis -- NEGATIVE for the left hand, POSITIVE for the right
# hand (measured on both index and middle fingertips, mirrored as expected).
#
# Palm frame convention (right-handed, orthonormal, det=+1):
#   approach axis = wrist local +X  (direction the fingers extend outward)
#   closing axis  = wrist local -Y (left) / +Y (right) (direction fingers
#                   curl toward when synergy goes 0 -> 1)
#   lateral axis  = approach x closing
# Encoded below as a fixed local quat (wxyz) for a site added directly on
# each wrist body, so site_xmat at runtime IS the palm frame in world coords.
# ---------------------------------------------------------------------------

LEFT_PALM_SITE = "left_palm_frame"
RIGHT_PALM_SITE = "right_palm_frame"
LEFT_PALM_LOCAL_POS = (0.05, 0.003, 0.0)
RIGHT_PALM_LOCAL_POS = (0.05, -0.003, 0.0)
# quat for R = columns [approach=+X, closing=-Y, lateral=-Z] (180 deg about X)
LEFT_PALM_LOCAL_QUAT = (0.0, 1.0, 0.0, 0.0)
# quat for R = columns [approach=+X, closing=+Y, lateral=+Z] (identity)
RIGHT_PALM_LOCAL_QUAT = (1.0, 0.0, 0.0, 0.0)

LEFT_THUMB_TIP_SITE = "left_thumb_tip"
LEFT_INDEX_TIP_SITE = "left_index_tip"
LEFT_MIDDLE_TIP_SITE = "left_middle_tip"
RIGHT_THUMB_TIP_SITE = "right_thumb_tip"
RIGHT_INDEX_TIP_SITE = "right_index_tip"
RIGHT_MIDDLE_TIP_SITE = "right_middle_tip"

# Distal link each fingertip site is attached to, at that link's own origin
# (a reasonable fingertip proxy -- these ARE the most distal link bodies).
FINGERTIP_SITE_BODIES = {
    LEFT_THUMB_TIP_SITE: "left_hand_thumb_2_link",
    LEFT_INDEX_TIP_SITE: "left_hand_index_1_link",
    LEFT_MIDDLE_TIP_SITE: "left_hand_middle_1_link",
    RIGHT_THUMB_TIP_SITE: "right_hand_thumb_2_link",
    RIGHT_INDEX_TIP_SITE: "right_hand_index_1_link",
    RIGHT_MIDDLE_TIP_SITE: "right_hand_middle_1_link",
}

# Sites already authored in the stock XML (not added by us): local offset
# (0,0,0) on each ankle_roll_link body.
LEFT_FOOT_SITE = "left_foot"
RIGHT_FOOT_SITE = "right_foot"

# Foot contact geoms are the 4 unnamed collision spheres per foot (contype=1
# conaffinity=1, friction=0.6, condim=3) attached directly to
# left/right_ankle_roll_link -- identified by body id at runtime, not by
# geom name (the stock XML gives them no name).
LEFT_FOOT_CONTACT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_CONTACT_BODY = "right_ankle_roll_link"

FLOATING_BASE_JOINT = tc.FLOATING_BASE_JOINT  # "floating_base_joint", body=pelvis
STAND_KEYFRAME = tc.STAND_KEYFRAME  # "stand": pelvis (0,0,0.79) identity quat, all joint qpos=0

# ---------------------------------------------------------------------------
# Hand synergy: open/close finger targets (radians), derived from the
# authored "stand" keyframe ctrl (used as OPEN -- it is the model's own
# resting hand pose: all finger joints at 0 except thumb_1) and each
# joint's own ctrlrange extreme in the flexion direction (used as CLOSE).
# This is a design decision, not a guessed fact -- validated empirically by
# the Phase 4 grasp smoke test (Section I).
#
# thumb_1 CORRECTED (Thumb Opposition + Claw-Style Bimanual Grasp session):
# direct single-joint measurement (sweeping thumb_1 alone, full range, with
# index/middle held at a representative mid-curl pose) found thumb_1 --
# NOT thumb_0 -- is the dominant thumb-to-finger OPPOSITION lever: its full
# range moves the thumb tip's distance to index/middle by ~10cm, roughly
# 2-3x thumb_0's or thumb_2's own range. The OLD open/close pair here was
# IDENTICAL (1.0472, 1.0472 / -1.0472, -1.0472) -- frozen, contributing
# NOTHING to any synergy command regardless of value -- and that frozen
# value happened to sit exactly at thumb_1's CLOSEST-to-fingers extreme
# (confirmed by the same sweep), not a neutral rest position as the old
# comment assumed. This directly explains the observed "thumb never fully
# opens/abducts, and never gets an independently adjustable close" failure
# mode: the single largest lever for thumb opposition distance was pinned
# permanently at one partially-closed value.
#
# OPEN is the TRANSPORT/REST pose (+0.45 left / -0.45 right), NOT
# thumb_1's raw kinematic range extreme (-0.724312/+0.724312) -- REVERTED
# back from a same-day attempt to widen this GLOBALLY (Thumb Opposition +
# Claw-Style Bimanual Grasp session, interactive-viewer follow-up,
# 2026-08-29). That attempt DID fix a real problem (thumb reaching the
# object at the same depth as index/middle, since +0.45 alone isn't
# enough clearance) but widening it HERE made it the default REST/
# TRANSPORT pose everywhere (WholeBodyEnv, any non-grasp-expert synergy
# usage, and this env's own STABLE_START), which self-collides with the
# hip at the STAND arm pose (~3.8cm thumb_2_link/hip_pitch_link
# penetration, confirmed directly) and broke an unrelated low-level IK
# test that holds the arm near the body. That was a genuine REST-pose
# regression, not a test-fixture bug -- per project policy this is fixed
# at the source (this table stays the safe TRANSPORT/REST pose) rather
# than by loosening the test. The full-clearance abduct pose thumb needs
# during an actual grasp attempt is instead applied by
# grasp_expert.py as a PHASE-SPECIFIC direct actuator override (see
# BimanualSidePinchExpert's THUMB_ABDUCT/THUMB_OPPOSE handling and
# GraspExpertConfig.thumb1_abduct_pose_left/right) -- active only from
# THUMB_ABDUCT (after the arm has already left the body and WRIST_ALIGN
# is done) through just before THUMB_OPPOSE closes it again, never as
# this module's own default.
#
# thumb_1 joint facts (measured directly, both hands, 2026-08-29):
#   left_hand_thumb_1_joint:  axis=local Z, range=[-0.724312, +1.0472]
#   right_hand_thumb_1_joint: axis=local Z, range=[-1.0472, +0.724312]
#   Mirrored as expected (right's range is left's negated and swapped).
#   Increasing left's value (toward +1.0472) / decreasing right's value
#   (toward -1.0472) moves the thumb CLOSER to index/middle (opposition);
#   the opposite bound on each side is the furthest/most-abducted point.
# ---------------------------------------------------------------------------

# (joint_name, open_rad, close_rad) -- order matches LEFT/RIGHT_HAND_JOINTS.
LEFT_HAND_SYNERGY_TARGETS = [
    ("left_hand_thumb_0_joint", 0.0, 0.6),  # lateral swing: neutral -> biased toward middle finger
    ("left_hand_thumb_1_joint", 0.45, 1.0472),  # opposition: TRANSPORT/REST (self-collision-safe) -> opposed (close), full range [-0.724312, 1.0472]
    ("left_hand_thumb_2_joint", 0.0, 1.74533),  # range [0, 1.74533]
    ("left_hand_middle_0_joint", 0.0, -1.5708),  # range [-1.5708, 0]
    ("left_hand_middle_1_joint", 0.0, -1.74533),  # range [-1.74533, 0]
    ("left_hand_index_0_joint", 0.0, -1.5708),
    ("left_hand_index_1_joint", 0.0, -1.74533),
]
RIGHT_HAND_SYNERGY_TARGETS = [
    ("right_hand_thumb_0_joint", 0.0, -0.6),  # mirrored sign vs left
    ("right_hand_thumb_1_joint", -0.45, -1.0472),  # mirrored: TRANSPORT/REST (self-collision-safe) -> opposed (close), full range [-1.0472, 0.724312]
    ("right_hand_thumb_2_joint", 0.0, -1.74533),
    ("right_hand_middle_0_joint", 0.0, 1.5708),  # range [0, 1.5708]
    ("right_hand_middle_1_joint", 0.0, 1.74533),
    ("right_hand_index_0_joint", 0.0, 1.5708),
    ("right_hand_index_1_joint", 0.0, 1.74533),
]

# ---------------------------------------------------------------------------
# Per-finger-group synergy targets (Dynamic-Aware Multi-Start IK + Multi-
# Finger Grasp/Lift/Hold session): the SAME (joint_name, open_rad,
# close_rad) triples as *_HAND_SYNERGY_TARGETS above, split into the 3
# independently-controllable groups direct measurement found necessary --
# a single scalar per hand could not guarantee the thumb actually made
# opposing contact (measured directly: the right hand's thumb never
# touched at all through a 60-step force-regulation window while index/
# middle flickered on/off, letting the object rotate/slip out under
# otherwise-plausible-looking aggregate force readings). Order within each
# group matches the joint's position in *_HAND_SYNERGY_TARGETS.
LEFT_THUMB_SYNERGY_TARGETS = LEFT_HAND_SYNERGY_TARGETS[0:3]
LEFT_MIDDLE_SYNERGY_TARGETS = LEFT_HAND_SYNERGY_TARGETS[3:5]
LEFT_INDEX_SYNERGY_TARGETS = LEFT_HAND_SYNERGY_TARGETS[5:7]
RIGHT_THUMB_SYNERGY_TARGETS = RIGHT_HAND_SYNERGY_TARGETS[0:3]
RIGHT_MIDDLE_SYNERGY_TARGETS = RIGHT_HAND_SYNERGY_TARGETS[3:5]
RIGHT_INDEX_SYNERGY_TARGETS = RIGHT_HAND_SYNERGY_TARGETS[5:7]

# ---------------------------------------------------------------------------
# Whole-body environment config
# ---------------------------------------------------------------------------


@dataclass
class StandingThresholds:
    """Standing/posture success criteria. Never hard-coded into env logic --
    always read from here (see PROJECT_CONTEXT.md Coding Rules)."""

    min_pelvis_height: float = 0.5  # stand key height is 0.79; a fall drops well below this
    max_roll: float = 0.6  # rad (~34 deg)
    max_pitch: float = 0.6
    max_xy_drift: float = 0.15  # meters, over the evaluation horizon
    max_joint_velocity: float = 30.0  # rad/s, matches Foundation's explosion guard order of magnitude
    horizon_steps: int = 1000  # at frame_skip=5, timestep=0.002 -> 10s


@dataclass
class WholeBodyConfig:
    seed: int | None = None
    frame_skip: int = 5
    max_episode_steps: int = 1000

    # Joint Position Delta action scales, per group (legs get a smaller
    # scale than arms -- a 0.05 rad/step delta on a loaded knee is a much
    # larger physical disturbance than the same delta on an unloaded arm).
    leg_action_scale: float = 0.02
    waist_action_scale: float = 0.02
    arm_action_scale: float = 0.05  # matches Foundation's EnvConfig.action_scale
    hand_synergy_action_scale: float = 0.05  # synergy units per step (synergy in [0,1])

    standing: StandingThresholds = field(default_factory=StandingThresholds)

    table_pos: tuple[float, float, float] = tc.DEFAULT_TABLE_POS
    table_half_size: tuple[float, float, float] = tc.DEFAULT_TABLE_HALF_SIZE

    g1_xml_path: str = str(tc.G1_XML_PATH)


@dataclass
class PlanarDebugConfig:
    """Debug/fallback-only kinematic planar base. NOT the final locomotion
    target -- see PROJECT_CONTEXT.md Section 8. x/y/yaw are tracked by
    high-gain position actuators on dedicated planar joints while legs,
    waist, and arms are held rigid at the stand pose (they do not move at
    all), so this is not a walking gait of any kind."""

    seed: int | None = None
    frame_skip: int = 5
    max_episode_steps: int = 500

    xy_action_scale: float = 0.05  # meters per step, per unit action
    yaw_action_scale: float = 0.05  # radians per step, per unit action
    x_range: tuple[float, float] = (-2.0, 2.0)
    y_range: tuple[float, float] = (-2.0, 2.0)

    g1_xml_path: str = str(tc.G1_XML_PATH)
