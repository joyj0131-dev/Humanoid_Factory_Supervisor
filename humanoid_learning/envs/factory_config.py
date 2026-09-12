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

import dataclasses
from dataclasses import dataclass, field
import math

import numpy as np

from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc

N_WORKCELLS = 2
N_LINES = N_WORKCELLS

# ---------------------------------------------------------------------------
# Two independent lines either side of one walking corridor.
#
# Both lines read TABLE | ARM | BELT from world -X to +X -- the same order, not
# a mirror image. That is what makes the corridor work: line 0 presents its BELT
# to the corridor and line 1 presents its TABLE, so the supervisor can reach
# line 0's jammed part and line 1's dropped part without ever walking around the
# outside of either line.
#
# Belt top sits at 0.75 m and the table top matches it, so the canonical grasp's
# validated working height carries over to both surfaces unchanged.
# ---------------------------------------------------------------------------
# The G1 spawns at the world origin, so the corridor is centred there: anything
# else drops the robot inside a table at reset.
CORRIDOR_CENTRE_X = 0.0
CORRIDOR_HALF_WIDTH_M = 0.65  # 1.3 m of clear floor; the G1 stance is 0.24 m wide
# Both work areas sit downstream of the spawn, so reaching either one is a walk
# along the corridor rather than a step sideways.
WORK_Y = 1.20

BELT_HALF_SIZE = (0.25, 2.5, 0.05)  # 0.5 m wide, 5 m long
BELT_BODY_Z = 0.70
TABLE_TOP_Z = BELT_BODY_Z + BELT_HALF_SIZE[2]  # 0.75, the belt surface
TABLE_HALF_SIZE = (0.20, 0.30, TABLE_TOP_Z / 2.0)  # top flush with the belt
# Clear floor between a surface edge and the arm column, so the column never
# sits under the part it is carrying.
ARM_CLEARANCE_M = 0.20
# The table sits slightly downstream of the pick spot, which turns the arm's
# transfer into a real yaw sweep instead of a 180 degree flip over its own column.
TABLE_DOWNSTREAM_M = -0.18

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


def _line_x_positions() -> tuple[dict[str, float], dict[str, float]]:
    """World x of each line's belt centre, arm column and table centre.

    Derived from the corridor edges outward so that changing the corridor width
    moves the whole factory consistently instead of needing every number
    retuned by hand.
    """
    belt_half, table_half = BELT_HALF_SIZE[0], TABLE_HALF_SIZE[0]
    left = CORRIDOR_CENTRE_X - CORRIDOR_HALF_WIDTH_M
    right = CORRIDOR_CENTRE_X + CORRIDOR_HALF_WIDTH_M
    # Line 0: belt against the corridor, then arm, then table further out.
    belt0 = left - belt_half
    arm0 = belt0 - belt_half - ARM_CLEARANCE_M
    table0 = arm0 - ARM_CLEARANCE_M - table_half
    # Line 1: same left-to-right order, so its table is the one facing in.
    table1 = right + table_half
    arm1 = table1 + table_half + ARM_CLEARANCE_M
    belt1 = arm1 + ARM_CLEARANCE_M + belt_half
    return ({"belt": belt0, "arm": arm0, "table": table0},
            {"belt": belt1, "arm": arm1, "table": table1})


LINE_X = _line_x_positions()

# Station-local frame: origin on the floor at the pelvis stand spot, +x = robot
# heading, +y = robot left. The canonical grasp relationship (part 0.27 m ahead
# of the pelvis on a 0.75 m surface) is reproduced in this frame on both lines,
# so each line is that same validated scene under a rigid transform.
LOCAL_CANONICAL_PART_XY = (0.27, 0.0)  # GraspEnvConfig.object_pos default
LOCAL_TABLE_POS = tc.DEFAULT_TABLE_POS  # (0.30, 0.0, 0.70)
LOCAL_OBSERVATION_XY = (-0.80, 0.0)
# The canonical grasp has the work surface 0.30 m ahead of the pelvis and the
# part at 0.27 m, so the part sits 30 mm in from the surface centreline, on the
# robot's side. Reproducing that inset is what keeps the validated grasp
# geometry exactly intact on both lines.
SURFACE_INSET_M = LOCAL_TABLE_POS[0] - LOCAL_CANONICAL_PART_XY[0]

# Physical stop blade just downstream of the pick spot. Real lines index parts
# against a stop like this; having one means the belt can keep running while a
# part waits to be picked, with no scripted "hold the part still" logic.
# The blade only has to catch the part's lower half, and every millimetre above
# that is something the arm has to lift the part over: at a 0.05 half-height its
# top stood at 0.85, the carried part's underside passed at 0.83, and the part
# was raked out of the jaws on every transfer.
STOPPER_HALF_SIZE = (0.06, 0.015, 0.03)
STOPPER_DOWNSTREAM_M = 0.09
PART_HALF_SIZE_M = 0.06  # the validated 0.12 m cube; FactoryConfig mirrors it
# The pick spot is where a part actually comes to REST against the blade, not
# the nominal centre of the cell. Authoring it as the latter left the part 15 mm
# downstream of the jaws: only one jaw reached it, and the closing gripper shoved
# the part out of the cell instead of gripping it (measured, jaw contacts 1).
PICK_Y = STOPPER_DOWNSTREAM_M - STOPPER_HALF_SIZE[1] - PART_HALF_SIZE_M

# Line 0 fault: the part jams upstream of the stop blade and never indexes into
# the pick spot. Deliberately still ON the belt -- the supervisor has to put it
# back at the pick spot, which is the belt-height task that already works.
JAM_UPSTREAM_M = -0.34  # relative to the belt origin, well short of the blade
JAM_JITTER_M = (0.03, 0.03)

# Line 1 fault: the arm carries the part past the table and opens its jaws, so
# the part topples off the corridor-facing edge onto the open floor. Nothing is
# teleported -- the gripper actually releases at the wrong place and gravity
# does the rest. The overshoot is measured from the table edge in the block's
# own half-width, so more than half of it hangs over open floor.
RELEASE_OVERSHOOT_M = 0.03

# Recovery is judged on the part's measured pose, never on a flag the G1 sets.
RECOVERY_POSITION_TOLERANCE_M = 0.06
RECOVERY_SETTLE_SPEED = 0.02
RECOVERY_HOLD_STEPS = 50


@dataclass(frozen=True)
class LayoutPreset:
    """How far apart the two lines sit, measured as corridor width."""

    name: str
    corridor_half_width_m: float


LAYOUT_PRESETS: dict[str, LayoutPreset] = {
    "narrow": LayoutPreset("narrow", 0.50),
    "medium": LayoutPreset("medium", CORRIDOR_HALF_WIDTH_M),
    "wide": LayoutPreset("wide", 0.90),
}
DEFAULT_LAYOUT = "medium"


@dataclass(frozen=True)
class WorkcellPose:
    """One line's rigid placement plus the world x of its three surfaces.

    Every scene pose is derived from this, so no world coordinate is written
    down twice. ``heading_rad`` differs by pi between the lines: the corridor
    runs between them, so the supervisor necessarily faces one line from one
    side and the other from the other.
    """

    index: int
    manipulation_xy: tuple[float, float]
    heading_rad: float
    belt_x: float
    arm_x: float
    table_x: float
    #: Which surface the supervisor's canonical stand pose is measured against.
    work_surface: str

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

    # -- world positions of the fixed furniture -------------------------------
    @property
    def belt_y(self) -> float:
        """Belts run the length of the hall, so only their x distinguishes them."""
        return 0.0

    @property
    def work_y(self) -> float:
        """Where along the line the arm works. The belt itself is centred on 0."""
        return WORK_Y

    @property
    def arm_base_xy(self) -> np.ndarray:
        return np.array([self.arm_x, self.work_y])

    @property
    def table_xy(self) -> np.ndarray:
        return np.array([self.table_x, self.work_y + TABLE_DOWNSTREAM_M])

    @property
    def table_pos(self) -> np.ndarray:
        """The surface the supervisor's canonical grasp relationship is on."""
        xy = self.to_world_xy(LOCAL_TABLE_POS[:2])
        return np.array([xy[0], xy[1], LOCAL_TABLE_POS[2]])

    @property
    def inward(self) -> np.ndarray:
        """Unit vector from this line toward the corridor."""
        return -self.heading_dir

    @property
    def pick_xy(self) -> np.ndarray:
        """Where the belt indexes a part for the arm to pick: against the blade."""
        return np.array([self.belt_x + self.inward[0] * SURFACE_INSET_M, self.work_y + PICK_Y])

    @property
    def canonical_part_xy(self) -> np.ndarray:
        """Where a part belongs when this line is running normally.

        Line 0 works at its belt, so that is the pick spot. Line 1 works at its
        table, so that is where a recovered part has to end up.
        """
        return self.pick_xy if self.work_surface == "belt" else self.table_place_xy

    @property
    def table_place_xy(self) -> np.ndarray:
        """Where the arm sets a finished part down on the table."""
        return np.array([self.table_x + self.inward[0] * SURFACE_INSET_M,
                         self.work_y + TABLE_DOWNSTREAM_M])

    @property
    def release_fault_xy(self) -> np.ndarray:
        """Where a faulted arm opens its jaws: past the corridor-facing edge.

        Measured from the table edge so more than half the block overhangs and
        it topples onto the open floor instead of sitting there. The overshoot
        is small on purpose -- the arm has to stay inside its own reach, and a
        part shoved off from height scatters instead of landing where the
        supervisor can find it.
        """
        inward = -1.0 if self.index == 1 else 1.0
        edge = self.table_x + inward * TABLE_HALF_SIZE[0]
        return np.array([edge + inward * RELEASE_OVERSHOOT_M, self.work_y + TABLE_DOWNSTREAM_M])

    @property
    def stopper_xy(self) -> np.ndarray:
        return np.array([self.belt_x, self.work_y + STOPPER_DOWNSTREAM_M])

    @property
    def jam_xy(self) -> np.ndarray:
        """Where a part jams on the belt: upstream, short of the stop blade."""
        return np.array([self.belt_x, self.work_y + JAM_UPSTREAM_M])

    @property
    def observation_xy(self) -> np.ndarray:
        return self.to_world_xy(LOCAL_OBSERVATION_XY)


def workcell_poses(layout: str | LayoutPreset = DEFAULT_LAYOUT) -> list[WorkcellPose]:
    """Line 0 faces world -X across its belt; line 1 faces +X across its table."""
    preset = LAYOUT_PRESETS[layout] if isinstance(layout, str) else layout
    belt_half, table_half = BELT_HALF_SIZE[0], TABLE_HALF_SIZE[0]
    left = CORRIDOR_CENTRE_X - preset.corridor_half_width_m
    right = CORRIDOR_CENTRE_X + preset.corridor_half_width_m
    belt0 = left - belt_half
    arm0 = belt0 - belt_half - ARM_CLEARANCE_M
    table0 = arm0 - ARM_CLEARANCE_M - table_half
    table1 = right + table_half
    arm1 = table1 + table_half + ARM_CLEARANCE_M
    belt1 = arm1 + ARM_CLEARANCE_M + belt_half
    stand_off = LOCAL_CANONICAL_PART_XY[0]
    poses = [
        WorkcellPose(index=0, heading_rad=math.pi, work_surface="belt",
                     manipulation_xy=(0.0, 0.0), belt_x=belt0, arm_x=arm0, table_x=table0),
        WorkcellPose(index=1, heading_rad=0.0, work_surface="table",
                     manipulation_xy=(0.0, 0.0), belt_x=belt1, arm_x=arm1, table_x=table1),
    ]
    # The stand pose is DERIVED from where the part has to be, so the canonical
    # "part 0.27 m straight ahead" relationship holds exactly on both lines
    # rather than being reproduced by hand and drifting.
    return [
        dataclasses.replace(pose, manipulation_xy=tuple(pose.canonical_part_xy + stand_off * pose.inward))
        for pose in poses
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
GRIPPER_CLOSED = 0.052  # 16 mm of squeeze against the taller jaw
GRIPPER_JOINT_RANGE = (0.03, 0.12)
# Thin along the closing axis; tall enough to actually hold a 0.12 m cube. At a
# 0.030 half-height the jaw face was 60 mm against the part's 120 mm, the part
# slid down through the jaws during every lift (measured: 53 mm of slip, the
# overlap shrinking to 37 mm before it fell out) and no squeeze depth from 0.058
# down to 0.040 changed that. The jaw now spans 110 mm and wraps the cube.
GRIPPER_JAW_HALF_SIZE = (0.055, 0.045, 0.008)
GRIPPER_FRICTION = (2.0, 0.02, 0.001)
# Stiffer than MuJoCo's default 0.02. The jaws close along the sweep's RADIAL
# direction, so centrifugal force in the transfer presses the part into the
# outer jaw and unloads the inner one; at 0.02 the outer jaw sank 13 mm into
# the part while the inner held 0.4 mm, and the part slid straight out. A
# harder contact keeps both faces loaded. Going all the way to 0.002 was tried
# earlier and needed ~59 N of jaw force to still lift, which is worse.
GRIPPER_SOLREF = (0.008, 1.0)
GRIPPER_KP = 1200.0
GRIPPER_KV = 30.0

ARM_JOINT_RANGES = ((-3.1416, 3.1416), (-1.8, 1.8), (-2.4, 2.4), (-3.1416, 3.1416))
# The tip sagged 8-9 mm off its commanded waypoint at kp=300, which is most of
# the gripper's 10 mm of squeeze: only one jaw reached the part and the grip was
# lost during the swing to the table. This is background factory equipment, not
# a compliant manipulator under study, so it gets a stiff actuator.
ARM_KP = 1200.0
ARM_KV = 70.0


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


def belt_body_name(index: int) -> str:
    return f"wc{index}_belt"


def belt_geom_name(index: int) -> str:
    return f"wc{index}_belt_geom"


def table_body_name(index: int) -> str:
    return f"wc{index}_table"


def table_geom_name(index: int) -> str:
    return f"wc{index}_table_geom"


def work_surface_geom_name(pose: "WorkcellPose") -> str:
    """The surface the supervisor's grasp is measured against on this line."""
    return (belt_geom_name(pose.index) if pose.work_surface == "belt"
            else table_geom_name(pose.index))


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


def arm_polar_from_world(pose: "WorkcellPose", world_xy) -> tuple[float, float]:
    """Convert a world (x, y) into this arm's (base_yaw, tip_radius).

    The two lines are NOT translations of one another -- line 0 presents its
    belt to the corridor and line 1 its table -- so the cycle is authored in
    world coordinates and projected into each column's own frame here, rather
    than in a shared station-local frame that no longer exists.
    """
    mount_yaw = pose.heading_rad + math.pi
    dx = float(world_xy[0]) - float(pose.arm_x)
    dy = float(world_xy[1]) - float(pose.work_y)
    cos, sin = math.cos(-mount_yaw), math.sin(-mount_yaw)
    x_arm = cos * dx - sin * dy
    y_arm = sin * dx + cos * dy
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


# High enough that a carried part clears the stop blade and the side rails with
# margin, including the ~60 mm it hangs below the jaw centre.
PICK_APPROACH_Z = 1.12
PART_CENTRE_Z = TABLE_TOP_Z + PART_HALF_SIZE_M + tc.OBJECT_TABLE_GAP  # 0.812
PART_TOP_Z = PART_CENTRE_Z + PART_HALF_SIZE_M
# The palm box hangs from the wrist along the arm: pos 0.035 plus half-size
# 0.030 puts its underside 0.065 below the wrist joint.
PALM_UNDERSIDE_FROM_WRIST = 0.065
PALM_PART_CLEARANCE_M = 0.008
# Centring the JAWS on the part (wrist at part centre + GRIPPER_LENGTH) also
# drove the PALM 25 mm into the part's top face: measured, the palm underside
# sat at 0.847 against a part top of 0.872. The descending palm pressed the part
# 8 mm into the belt and shoved it sideways, so only one jaw ever reached it and
# the grip was lost the moment the arm swung. The wrist is therefore set by the
# palm's clearance over the part, not by jaw centring; the jaws are 90 mm tall
# and still overlap the part by 57 mm at this height.
GRASP_WRIST_Z = PART_TOP_Z + PALM_PART_CLEARANCE_M + PALM_UNDERSIDE_FROM_WRIST

# Scripted indexing cycle: pick off the belt, swing, set down on the table.
# Each entry is (which world spot, wrist height, jaw state). Jaw state is
# symbolic and resolved at call time, so the open/closed half-openings stay
# tunable in one place instead of being frozen into this table at import.
# The fourth field scales that step's dwell. The belt and the table sit on
# opposite sides of the column, so the transfer is a ~200 degree sweep; at the
# uniform 60-step dwell the tip crossed it at 2.7 m/s and the part was flung out
# of the jaws every cycle. The two sweeping steps get the time that rotation
# actually needs, and the short vertical moves keep the short dwell.
ARM_CYCLE_STEPS: tuple[tuple[str, float, str, int], ...] = (
    ("pick", PICK_APPROACH_Z, "open", 7),      # 0 swing back over the part
    ("pick", GRASP_WRIST_Z, "open", 2),        # 1 descend around it
    ("pick", GRASP_WRIST_Z, "closed", 2),      # 2 grip, and let the squeeze settle
    ("pick", PICK_APPROACH_Z, "closed", 3),    # 3 lift
    ("place", PICK_APPROACH_Z, "closed", 7),   # 4 sweep to the table
    ("place", GRASP_WRIST_Z, "closed", 3),     # 5 lower
    ("place", GRASP_WRIST_Z, "open", 1),       # 6 release
    ("place", PICK_APPROACH_Z, "open", 2),     # 7 retract
)
ARM_CYCLE_HOLD_STEPS = 60


def arm_cycle_dwells() -> tuple[int, ...]:
    return tuple(ARM_CYCLE_HOLD_STEPS * step[3] for step in ARM_CYCLE_STEPS)

# The step whose "place" spot a drop-fault replaces. Steps 4-7 all use it, so
# a faulted arm carries the part past the table and lets go there, rather than
# teleporting anything: the jaws really open at the wrong place.
ARM_PLACE_STEP_INDICES = (4, 5, 6, 7)
ARM_RELEASE_STEP_INDEX = 6

# Where a faulted arm freezes instead, for the jam case: just after it should
# have closed on the part, so a stalled arm sits over the belt with its jaws
# open and the part not taken.
ARM_FAULT_WAYPOINT_INDEX = 2


def arm_cycle_waypoints(pose: "WorkcellPose", *, drop_fault: bool = False):
    """The scripted cycle as arm joint targets, derived from the layout.

    ``drop_fault`` swings the arm past the table's corridor-facing edge before
    it opens, which is the whole of line 1's failure: no special-case physics,
    just a place spot that is not over the table any more.
    """
    place = pose.release_fault_xy if drop_fault else pose.table_place_xy
    spots = {"pick": pose.pick_xy, "place": place}
    out = []
    for spot, wrist_z, _jaw, _dwell in ARM_CYCLE_STEPS:
        base_yaw, radius = arm_polar_from_world(pose, spots[spot])
        out.append(arm_joint_targets(base_yaw, radius, wrist_z))
    return tuple(out)


def arm_cycle_jaw_targets() -> tuple[float, ...]:
    return tuple(GRIPPER_OPEN if step[2] == "open" else GRIPPER_CLOSED for step in ARM_CYCLE_STEPS)


#: Line 0 presents its belt to the corridor, so its part jams there; line 1
#: presents its table, so its part is dropped there. One failure mode per line.
SCENARIOS = ("jam", "arm_drop")
DEFAULT_SCENARIOS = {0: "jam", 1: "arm_drop"}


@dataclass
class FaultConfig:
    """Deterministic jam / dropped-part injection."""

    enabled: bool = True
    scenario: str = "jam"
    jam_yaw_rad: float = 0.55
    #: How far below the belt surface a part has to be to count as on the floor.
    floor_clearance_m: float = 0.30
    #: A dropped part is only "landed" once it has stopped bouncing.
    landed_speed_m_s: float = 0.05
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
