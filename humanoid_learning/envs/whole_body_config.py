"""Whole-body facts and configuration for the canonical G1 + Sharpa model."""

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

# G1 actuator names are identical to their joint names.
LEFT_LEG_ACTUATORS = LEFT_LEG_JOINTS
RIGHT_LEG_ACTUATORS = RIGHT_LEG_JOINTS
WAIST_ACTUATORS = WAIST_JOINTS

PELVIS_BODY = "pelvis"
TORSO_BODY = "torso_link"

# ---------------------------------------------------------------------------
# Palm sites are attached to the G1 wrist and therefore remain independent
# of the hand mesh.  Sharpa fingertip sites are derived from its real
# elastomer geoms in model_builder._add_sharpa_grasp_sites().
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
