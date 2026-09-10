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
# Workcell-local geometry, in the local frame described in the module docstring
# (+x = robot heading, +y = robot left, z = world up, origin on the floor at the
# pelvis stand spot). The table entry is DEFAULT_TABLE_POS verbatim so that a
# G1 standing at the local origin sees exactly the canonical grasp scene.
# ---------------------------------------------------------------------------
LOCAL_TABLE_POS = tc.DEFAULT_TABLE_POS  # (0.30, 0.0, 0.70)
LOCAL_TABLE_HALF_SIZE = tc.DEFAULT_TABLE_HALF_SIZE  # (0.15, 0.35, 0.05)
TABLE_TOP_Z = LOCAL_TABLE_POS[2] + LOCAL_TABLE_HALF_SIZE[2]  # 0.75

# Canonical grasp spot: GraspEnvConfig.object_pos default is (0.27, 0.0).
LOCAL_CANONICAL_PART_XY = (0.27, 0.0)

# Where the G1 supervises the cell from before stepping in to manipulate.
LOCAL_OBSERVATION_XY = (-0.80, 0.0)

# Automation arm column, clear of the table's 0.35 lateral half-width
# (column radius 0.07 -> its edge sits 0.10 m outside the table edge).
LOCAL_ARM_BASE_XY = (0.30, 0.55)

# Dropped-part zone: still on the tabletop, laterally offset from the canonical
# grasp spot. Deliberately NOT on the floor -- floor-level picking needs body
# lowering and a different grasp topology, neither of which exists yet (see
# docs/FACTORY_ENVIRONMENT.md). Kept configurable so a floor preset can be
# evaluated later without another layout change.
LOCAL_DROP_ZONE_XY = (0.27, 0.08)
DROP_ZONE_JITTER_XY = (0.03, 0.03)


@dataclass(frozen=True)
class LayoutPreset:
    """Polar placement of the two workcells around the G1 home pose.

    ``bearing_deg`` is measured from world +X; workcell 0 sits at -bearing and
    workcell 1 at +bearing, each facing outward along its own bearing, so the
    robot must both turn and translate to reach either one.
    """

    name: str
    home_to_cell_m: float
    bearing_deg: float


LAYOUT_PRESETS: dict[str, LayoutPreset] = {
    "short": LayoutPreset("short", 1.5, 35.0),
    "medium": LayoutPreset("medium", 2.5, 35.0),
    "long": LayoutPreset("long", 4.0, 35.0),
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
        xy = self.to_world_xy(LOCAL_TABLE_POS[:2])
        return np.array([xy[0], xy[1], LOCAL_TABLE_POS[2]])

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
    preset = LAYOUT_PRESETS[layout] if isinstance(layout, str) else layout
    poses = []
    for index in range(N_WORKCELLS):
        # workcell 0 -> -bearing, workcell 1 -> +bearing
        bearing = math.radians(preset.bearing_deg) * (1.0 if index == 1 else -1.0)
        distance = preset.home_to_cell_m
        poses.append(
            WorkcellPose(
                index=index,
                manipulation_xy=(distance * math.cos(bearing), distance * math.sin(bearing)),
                heading_rad=bearing,
            )
        )
    return poses


# ---------------------------------------------------------------------------
# Automation arm: a deliberately minimal 3-DoF primitive-geom arm. It is
# background factory equipment and a fault-generation device, never a research
# subject (PROJECT_CONTEXT.md "Factory simplicity rule"), but it is driven by
# real joints and position actuators rather than per-frame geom teleportation.
# ---------------------------------------------------------------------------
ARM_JOINT_SUFFIXES = ("base_yaw", "shoulder_pitch", "elbow_pitch")
# Column top (2 * half height = 1.20 m) must clear the 0.75 m tabletop by more
# than the arm needs to fold down, and the two link lengths must reach from the
# column to the table centre: horizontal 0.55, vertical 0.45 -> 0.711 m, inside
# the 0.75 m total reach. These numbers are checked in scripts/test_factory.py.
ARM_COLUMN_HALF_HEIGHT = 0.60
ARM_UPPER_LENGTH = 0.40
ARM_FORE_LENGTH = 0.35
ARM_LINK_RADIUS = 0.035
ARM_TURRET_Z = 2.0 * ARM_COLUMN_HALF_HEIGHT

ARM_JOINT_RANGES = ((-3.1416, 3.1416), (-1.8, 1.8), (-2.4, 2.4))
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


def beacon_body_name(index: int) -> str:
    return f"wc{index}_status_beacon"


def beacon_geom_name(index: int) -> str:
    return f"wc{index}_status_beacon_geom"


BEACON_RUNNING_RGBA = (0.15, 0.85, 0.25, 1.0)
BEACON_FAULT_RGBA = (0.95, 0.15, 0.10, 1.0)


def table_body_name(index: int) -> str:
    return f"wc{index}_table"


def table_geom_name(index: int) -> str:
    return f"wc{index}_table_geom"


def arm_joint_targets(base_yaw: float, tip_radius: float, tip_z: float) -> tuple[float, float, float]:
    """Closed-form 2-link IK for the automation arm, elbow-down solution.

    Waypoints are authored in tip space (how far out and how high the gripper
    should be) rather than as raw joint angles, so a change to the link lengths
    or column height cannot silently drive the arm through the tabletop -- the
    first attempt at this file hard-coded joint angles and did exactly that.
    ``tip_radius`` is measured horizontally from the turret axis and ``tip_z``
    in world height.
    """
    l1, l2 = ARM_UPPER_LENGTH, ARM_FORE_LENGTH
    drop = ARM_TURRET_Z - tip_z
    reach = math.hypot(tip_radius, drop)
    if reach > l1 + l2 or reach < abs(l1 - l2):
        raise ValueError(f"tip target (r={tip_radius}, z={tip_z}) is outside the arm's reach")
    cos_elbow = (reach * reach - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    elbow = math.acos(max(-1.0, min(1.0, cos_elbow)))
    shoulder = math.atan2(drop, tip_radius) - math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))
    return (base_yaw, shoulder, elbow)


# Scripted normal-production cycle in TIP space: (base_yaw, tip_radius, tip_z).
# base_yaw 0 points the arm from its column toward the table centre; the table
# centre sits at tip_radius = 0.55 (LOCAL_ARM_BASE_XY[1]) and the tabletop at
# z = 0.75.
#
# The "down" waypoints must clear the PART, not just the tabletop. Measured
# geometry (mj_geomDistance against the compiled model, not arithmetic -- two
# earlier guesses at this number were both wrong):
#   part rests centred at z = 0.812, half size 0.06  ->  top at 0.872
#   the forearm is a capsule reaching ARM_LINK_RADIUS*0.85 (~0.030 m) below the
#   tip, so a tip at z clears the part by roughly (z - 0.90)
# tip_z = 0.82 gave 98 arm/part contacts and 62 mm of part drift per 800 steps
# of "normal" production; tip_z = 0.88 was worse (247 contacts, 136 mm). The arm
# only mimics pick/place, so it must not disturb the part at all.
# tip_z = 0.95 leaves ~0.05 m of measured clearance.
ARM_CYCLE_TIP_TARGETS: tuple[tuple[float, float, float], ...] = (
    (0.00, 0.45, 1.08),  # home, raised
    (0.00, 0.55, 0.95),  # reach down over the table (pick)
    (0.00, 0.50, 1.04),  # lift clear
    (0.60, 0.50, 1.04),  # traverse to the outfeed side
    (0.60, 0.55, 0.95),  # place
    (0.00, 0.45, 1.08),  # return home
)


def arm_cycle_waypoints() -> tuple[tuple[float, float, float], ...]:
    """The scripted cycle as joint targets, derived from the tip targets."""
    return tuple(arm_joint_targets(*target) for target in ARM_CYCLE_TIP_TARGETS)
ARM_CYCLE_HOLD_STEPS = 60

# Where in the cycle a faulted arm freezes: the "pick" waypoint, so a stalled
# arm visibly sits over the table with its part missing.
ARM_FAULT_WAYPOINT_INDEX = 1


@dataclass
class FaultConfig:
    """Deterministic Dropped-Part / pick-failure injection."""

    enabled: bool = True
    # Env step at which the chosen workcell faults. Sampled in
    # [min_step, max_step] from the episode seed when randomize_time is on.
    min_step: int = 120
    max_step: int = 360
    randomize_time: bool = False
    fixed_step: int = 200
    randomize_drop_xy: bool = False
    # Height above the support surface the part is released from, so it
    # actually falls and settles under gravity rather than being pasted onto
    # the surface at its final resting pose.
    release_height_m: float = 0.05


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
