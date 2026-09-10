"""Two-workcell factory layout, scripted automation arms, fault injection and
the Navigation Gate.

Layout convention (measured against the existing canonical grasp, not invented):
the fixed-base grasp env stands the G1 pelvis at the world origin facing +X with
its table at ``tc.DEFAULT_TABLE_POS`` = (0.30, 0, 0.70) and the canonical object
at x=0.27. This module therefore defines each workcell in a LOCAL frame whose
origin is the pelvis stand spot for canonical manipulation and whose +x is the
robot heading, so every workcell pose is that same validated relationship moved
by a rigid transform. Workcell 0 and workcell 1 differ only by that transform.

Nothing here teleports the base. ``NavigationGate`` states what a future
locomotion Expert/BC must achieve; it does not itself move the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc

N_WORKCELLS = 2

# ---------------------------------------------------------------------------
# Conveyor line geometry.
#
# One straight conveyor running along world +Y at x = CONVEYOR_CENTRE_X, with
# two arm stations on it. Both stations face the same way, so the G1 approaches
# each of them facing world +X exactly as in the canonical fixed-base grasp.
# That is the point of the straight-line layout: the grasp Expert's world-axis
# approach offsets stay valid, instead of being wrong by 169 mm as they were
# under the earlier +-35 degree workcell placement.
#
# Belt top sits at 0.75 m, the same working height as the canonical grasp
# table, so the validated table-height relationship carries over unchanged.
# ---------------------------------------------------------------------------
CONVEYOR_CENTRE_X = 2.0
BELT_HALF_SIZE = (0.25, 2.5, 0.05)  # 0.5 m wide, 5 m long
BELT_BODY_Z = 0.70
TABLE_TOP_Z = BELT_BODY_Z + BELT_HALF_SIZE[2]  # 0.75, the belt surface
BELT_SPEED = 0.25  # m/s while the line is running
# MuJoCo has no conveyor primitive. The belt is modelled by pushing parts that
# rest on it toward the belt velocity -- a viscous surface-drive model, not a
# simulated belt mechanism. Stated so the fidelity is not overread.
#
# The gain has to beat the friction holding the part down, not just look
# plausible: mu*m*g = 1.5 * 0.1 * 9.81 = 1.47 N. A gain of 12 gave 0.3 N and the
# part simply never moved. The drive is clipped so a part that is briefly
# pinched or jammed cannot be shoved through anything.
BELT_DRIVE_GAIN = 200.0
BELT_MAX_DRIVE_N = 5.0
BELT_DIRECTION = 1.0  # parts flow toward +Y

# Side-rail height, measured against the robot rather than chosen for looks.
# The G1's torso bottom sits at z=0.817 while standing at the manipulation pose,
# and its front reaches x=1.779 -- straight through the near rail's x span. The
# first version's rail top was at 0.810, leaving 7 mm, which walking bob closed:
# the torso struck the rail and the Navigation Gate correctly failed the run.
# A 0.030 m rail (top at 0.780) still catches the part, whose underside is at
# 0.752, while clearing the torso by 37 mm.
RAIL_HALF_HEIGHT = 0.015

# Station-local frame: origin on the floor at the pelvis stand spot, +x = robot
# heading (world +X), +y = robot left. Identical to the canonical grasp
# relationship, so both stations are the same scene under a pure translation.
LOCAL_TABLE_POS = tc.DEFAULT_TABLE_POS  # (0.30, 0.0, 0.70) -> belt centre ahead
LOCAL_CANONICAL_PART_XY = (0.27, 0.0)  # GraspEnvConfig.object_pos default
LOCAL_OBSERVATION_XY = (-0.80, 0.0)
# The arm column stands on the far side of the belt, facing the G1 across it,
# so the robot and the automation never contend for the same floor space.
LOCAL_ARM_BASE_XY = (0.85, 0.0)
# Where the arm sets a finished part down: upstream of the pick spot, so the
# running belt carries it back and the line forms a real repeating loop.
LOCAL_OUTFEED_XY = (0.27, -0.32)

# Physical stop blade just downstream of the pick spot. Real lines index parts
# against a stop like this; having one means the belt can keep running while a
# part waits to be picked, with no scripted "hold the part still" logic.
LOCAL_STOPPER_XY = (0.27, 0.09)
STOPPER_HALF_SIZE = (0.06, 0.015, 0.05)

# Dropped-part zone: past the stop blade, where the arm's cycle never reaches.
# The line stops, and only the G1 can put the part back. Deliberately still ON
# the belt, not on the floor -- floor-level picking needs body lowering and a
# different grasp topology, neither of which exists yet.
LOCAL_DROP_ZONE_XY = (0.27, 0.28)
DROP_ZONE_JITTER_XY = (0.03, 0.03)

# Recovery is judged on the part's measured pose, never on a flag the G1 sets.
RECOVERY_POSITION_TOLERANCE_M = 0.06
RECOVERY_SETTLE_SPEED = 0.02
RECOVERY_HOLD_STEPS = 50


@dataclass(frozen=True)
class LayoutPreset:
    """Spacing of the two arm stations along the conveyor.

    Station 0 sits upstream at -spacing/2 and station 1 downstream at
    +spacing/2, both facing world +X.
    """

    name: str
    station_spacing_m: float


LAYOUT_PRESETS: dict[str, LayoutPreset] = {
    "short": LayoutPreset("short", 1.6),
    "medium": LayoutPreset("medium", 2.2),
    "long": LayoutPreset("long", 3.0),
}
DEFAULT_LAYOUT = "medium"


@dataclass(frozen=True)
class WorkcellPose:
    """One workcell's rigid placement. Every scene pose is derived from this,
    so no world coordinate for a workcell is written down twice."""

    index: int
    manipulation_xy: tuple[float, float]
    heading_rad: float

    @property
    def heading_dir(self) -> np.ndarray:
        return np.array([math.cos(self.heading_rad), math.sin(self.heading_rad)])

    @property
    def left_dir(self) -> np.ndarray:
        return np.array([-math.sin(self.heading_rad), math.cos(self.heading_rad)])

    def to_world_xy(self, local_xy) -> np.ndarray:
        """Rigid-transform a workcell-local (x, y) into world (x, y)."""
        lx, ly = float(local_xy[0]), float(local_xy[1])
        return np.asarray(self.manipulation_xy, dtype=float) + lx * self.heading_dir + ly * self.left_dir

    def to_local_xy(self, world_xy) -> np.ndarray:
        """Inverse of :meth:`to_world_xy`."""
        delta = np.asarray(world_xy, dtype=float)[:2] - np.asarray(self.manipulation_xy, dtype=float)
        return np.array([float(delta @ self.heading_dir), float(delta @ self.left_dir)])

    @property
    def table_pos(self) -> np.ndarray:
        """Belt centre in front of this station (the canonical table position)."""
        xy = self.to_world_xy(LOCAL_TABLE_POS[:2])
        return np.array([xy[0], xy[1], LOCAL_TABLE_POS[2]])

    @property
    def outfeed_xy(self) -> np.ndarray:
        return self.to_world_xy(LOCAL_OUTFEED_XY)

    @property
    def stopper_xy(self) -> np.ndarray:
        return self.to_world_xy(LOCAL_STOPPER_XY)

    @property
    def arm_base_xy(self) -> np.ndarray:
        return self.to_world_xy(LOCAL_ARM_BASE_XY)

    @property
    def canonical_part_xy(self) -> np.ndarray:
        return self.to_world_xy(LOCAL_CANONICAL_PART_XY)

    @property
    def drop_zone_xy(self) -> np.ndarray:
        return self.to_world_xy(LOCAL_DROP_ZONE_XY)

    @property
    def observation_xy(self) -> np.ndarray:
        return self.to_world_xy(LOCAL_OBSERVATION_XY)


def workcell_poses(layout: str | LayoutPreset = DEFAULT_LAYOUT) -> list[WorkcellPose]:
    """Both stations face world +X; only their Y along the belt differs."""
    preset = LAYOUT_PRESETS[layout] if isinstance(layout, str) else layout
    half = preset.station_spacing_m / 2.0
    pelvis_x = CONVEYOR_CENTRE_X - LOCAL_TABLE_POS[0]
    return [
        WorkcellPose(
            index=index,
            manipulation_xy=(pelvis_x, -half if index == 0 else +half),
            heading_rad=0.0,
        )
        for index in range(N_WORKCELLS)
    ]


# ---------------------------------------------------------------------------
# Automation arm: a deliberately minimal 3-DoF primitive-geom arm. It is
# background factory equipment and a fault-generation device, never a research
# subject (PROJECT_CONTEXT.md "Factory simplicity rule"), but it is driven by
# real joints and position actuators rather than per-frame geom teleportation.
# ---------------------------------------------------------------------------
ARM_JOINT_SUFFIXES = ("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch")
GRIPPER_JOINT_SUFFIXES = ("jaw_left", "jaw_right")

# Column top (2 * half height = 1.20 m) must clear the 0.75 m belt by more than
# the arm needs to fold down, and the links must reach from the column across
# the belt. Checked in scripts/test_factory.py against the compiled model.
ARM_COLUMN_HALF_HEIGHT = 0.60
ARM_UPPER_LENGTH = 0.40
ARM_FORE_LENGTH = 0.35
ARM_LINK_RADIUS = 0.035
ARM_TURRET_Z = 2.0 * ARM_COLUMN_HALF_HEIGHT

# wrist_pitch keeps the gripper pointing straight DOWN whatever the shoulder and
# elbow are doing, so the jaws descend onto a part from above instead of raking
# at it sideways. GRIPPER_LENGTH is wrist joint -> jaw centre.
GRIPPER_LENGTH = 0.10
# Half-opening per jaw, measured from the jaw body centre. The part is a 0.12 m
# cube and the jaw is 0.008 m thick, so the jaws first touch at 0.068.
#
# Squeeze depth was measured, not guessed. Commanding 0.050 (18 mm of squeeze)
# drove 9.3 mm of contact penetration, let the part slip 70 mm through the jaws
# during the lift, and then EJECTED it upward -- the same soft-contact "squirt"
# failure this project already documented for the Sharpa hand. A light squeeze
# against a stiff actuator holds far better than a deep one.
GRIPPER_CONTACT_HALF_OPENING = 0.068
GRIPPER_OPEN = 0.11
GRIPPER_CLOSED = 0.058  # 10 mm of squeeze; see the sweep in docs/FACTORY_ENVIRONMENT.md
GRIPPER_JOINT_RANGE = (0.03, 0.12)
GRIPPER_JAW_HALF_SIZE = (0.030, 0.008, 0.045)
GRIPPER_FRICTION = (2.0, 0.02, 0.001)
# MuJoCo's default contact softness. Stiffening it to (0.002, 1) did cut
# penetration but needed ~59 N of jaw force to still lift, which is worse.
GRIPPER_SOLREF = (0.02, 1.0)
GRIPPER_KP = 1200.0
GRIPPER_KV = 30.0

ARM_JOINT_RANGES = ((-3.1416, 3.1416), (-1.8, 1.8), (-2.4, 2.4), (-3.1416, 3.1416))
ARM_KP = 300.0
ARM_KV = 30.0


def arm_prefix(index: int) -> str:
    return f"wc{index}_arm"


def arm_joint_name(index: int, suffix: str) -> str:
    return f"{arm_prefix(index)}_{suffix}"


def arm_body_name(index: int, part: str) -> str:
    return f"{arm_prefix(index)}_{part}"


def part_body_name(index: int) -> str:
    return f"wc{index}_part"


def part_joint_name(index: int) -> str:
    return f"wc{index}_part_joint"


def part_geom_name(index: int) -> str:
    return f"wc{index}_part_geom"


def table_body_name(index: int) -> str:
    return "conveyor_belt"


def table_geom_name(index: int) -> str:
    return "conveyor_belt_geom"


BELT_BODY = "conveyor_belt"
BELT_GEOM = "conveyor_belt_geom"


def stopper_body_name(index: int) -> str:
    return f"wc{index}_stopper"


def stopper_geom_name(index: int) -> str:
    return f"wc{index}_stopper_geom"


def beacon_body_name(index: int) -> str:
    return f"wc{index}_status_beacon"


def beacon_geom_name(index: int) -> str:
    return f"wc{index}_status_beacon_geom"


BEACON_RUNNING_RGBA = (0.15, 0.85, 0.25, 1.0)
BEACON_FAULT_RGBA = (0.95, 0.15, 0.10, 1.0)


def arm_polar_from_station_local(local_xy) -> tuple[float, float]:
    """Convert a station-local (x, y) into the arm's (base_yaw, tip_radius).

    The arm column sits at LOCAL_ARM_BASE_XY facing back across the belt, so its
    own +x is station-local -x and its +y is station-local -y. Deriving the
    cycle this way means the waypoints follow the layout instead of being
    re-tuned by hand whenever the belt or column moves.
    """
    dx = float(local_xy[0]) - LOCAL_ARM_BASE_XY[0]
    dy = float(local_xy[1]) - LOCAL_ARM_BASE_XY[1]
    x_arm, y_arm = -dx, -dy
    return math.atan2(y_arm, x_arm), math.hypot(x_arm, y_arm)


def arm_joint_targets(base_yaw: float, tip_radius: float, tip_z: float) -> tuple[float, float, float, float]:
    """Closed-form 2-link IK for the wrist position, plus the wrist_pitch that
    holds the gripper vertical.

    Waypoints are authored in tip space (how far out and how high the wrist
    should be) rather than as raw joint angles: an early version hard-coded
    joint angles and drove the arm straight through the work surface.
    """
    l1, l2 = ARM_UPPER_LENGTH, ARM_FORE_LENGTH
    drop = ARM_TURRET_Z - tip_z
    reach = math.hypot(tip_radius, drop)
    if reach > l1 + l2 or reach < abs(l1 - l2):
        raise ValueError(f"tip target (r={tip_radius}, z={tip_z}) is outside the arm's reach")
    cos_elbow = (reach * reach - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    elbow = math.acos(max(-1.0, min(1.0, cos_elbow)))
    shoulder = math.atan2(drop, tip_radius) - math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))
    # Positive pitch tips a link downward, so the gripper points straight down
    # when the three pitches sum to +pi/2.
    wrist = math.pi / 2.0 - (shoulder + elbow)
    return (base_yaw, shoulder, elbow, wrist)


# Scripted indexing cycle. Each entry is (station-local xy of the wrist target,
# wrist height, jaw half-opening). The belt indexes a part into the pick spot,
# the line pauses, the arm picks and places, and the belt resumes.
PICK_APPROACH_Z = 1.05
PART_CENTRE_Z = TABLE_TOP_Z + 0.06 + tc.OBJECT_TABLE_GAP  # 0.812
GRASP_WRIST_Z = PART_CENTRE_Z + GRIPPER_LENGTH  # wrist height that centres the jaws

# Jaw state is symbolic ("open"/"closed") and resolved at call time, so the
# open/closed half-openings stay tunable in one place instead of being frozen
# into this table at import.
ARM_CYCLE_STEPS: tuple[tuple[tuple[float, float], float, str], ...] = (
    (LOCAL_CANONICAL_PART_XY, PICK_APPROACH_Z, "open"),    # 0 above the part
    (LOCAL_CANONICAL_PART_XY, GRASP_WRIST_Z, "open"),      # 1 descend around it
    (LOCAL_CANONICAL_PART_XY, GRASP_WRIST_Z, "closed"),    # 2 grip
    (LOCAL_CANONICAL_PART_XY, PICK_APPROACH_Z, "closed"),  # 3 lift
    (LOCAL_OUTFEED_XY, PICK_APPROACH_Z, "closed"),         # 4 traverse
    (LOCAL_OUTFEED_XY, GRASP_WRIST_Z, "closed"),           # 5 lower
    (LOCAL_OUTFEED_XY, GRASP_WRIST_Z, "open"),             # 6 release
    (LOCAL_OUTFEED_XY, PICK_APPROACH_Z, "open"),           # 7 retract
)
ARM_CYCLE_HOLD_STEPS = 60

# Where a faulted arm freezes: just after it should have closed on the part, so
# a stalled arm sits over the belt with its jaws open and the part not taken.
ARM_FAULT_WAYPOINT_INDEX = 2


def arm_cycle_waypoints() -> tuple[tuple[float, float, float, float], ...]:
    """The scripted cycle as arm joint targets, derived from the layout."""
    out = []
    for local_xy, wrist_z, _jaw in ARM_CYCLE_STEPS:
        base_yaw, radius = arm_polar_from_station_local(local_xy)
        out.append(arm_joint_targets(base_yaw, radius, wrist_z))
    return tuple(out)


def arm_cycle_jaw_targets() -> tuple[float, ...]:
    return tuple(GRIPPER_OPEN if step[2] == "open" else GRIPPER_CLOSED for step in ARM_CYCLE_STEPS)


@dataclass
class FaultConfig:
    """Deterministic Dropped-Part / pick-failure injection."""

    enabled: bool = True
    scenario: str = "dropped_part"  # or misplaced_part
    misplaced_offset_y: float = 0.24
    misplaced_yaw_rad: float = 0.30
    drop_min_lift_m: float = 0.025
    drop_min_transfer_m: float = 0.10
    detection_fall_m: float = 0.02
    # Env step at which the chosen workcell faults. Sampled in
    # [min_step, max_step] from the episode seed when randomize_time is on.
    min_step: int = 120
    max_step: int = 360
    randomize_time: bool = False
    fixed_step: int = 200


@dataclass
class NavigationGate:
    """What a future locomotion Expert/BC must achieve to count as having
    reached a workcell. Thresholds are stated up front, before any policy
    exists, so they cannot be relaxed afterwards to manufacture a pass.

    ``max_position_error_m`` is deliberately looser than the grasp envelope
    that has actually been validated (+-0.010 m of object XY offset). That gap
    is real and is recorded in docs/FACTORY_ENVIRONMENT.md rather than hidden:
    arriving inside this gate is NOT by itself enough for the current grasp
    Expert to succeed.
    """

    max_position_error_m: float = 0.10
    max_heading_error_rad: float = 0.15  # ~8.6 deg
    min_hold_steps: int = 100  # 1.0 s at frame_skip=5, timestep=0.002
    max_steps: int = 1500
    # Any single env step moving the base further than this is not walking.
    max_base_step_m: float = 0.05
    forbidden_contact_bodies: tuple[str, ...] = ("pelvis", "torso_link")


def _factory_whole_body_config() -> wbc.WholeBodyConfig:
    """G1 control settings for the factory. Composed from WholeBodyConfig rather
    than re-declared, so the supervisor's action semantics cannot silently
    diverge from the whole-body env the rest of the project already uses. Only
    the episode length differs: navigating between cells needs a longer horizon
    than a standing test."""
    return wbc.WholeBodyConfig(max_episode_steps=2000)


@dataclass
class FactoryConfig:
    layout: str = DEFAULT_LAYOUT

    # G1 whole-body control (action scales, frame_skip, standing thresholds).
    whole_body: wbc.WholeBodyConfig = field(default_factory=_factory_whole_body_config)

    part_half_size: float = 0.06  # 12 cm cube, the validated canonical size
    part_mass: float = 0.1
    part_friction: tuple[float, float, float] = (1.5, 0.01, 0.0001)

    fault: FaultConfig = field(default_factory=FaultConfig)
    navigation: NavigationGate = field(default_factory=NavigationGate)

    g1_xml_path: str = str(tc.G1_XML_PATH)

    @property
    def workcells(self) -> list[WorkcellPose]:
        return workcell_poses(self.layout)
