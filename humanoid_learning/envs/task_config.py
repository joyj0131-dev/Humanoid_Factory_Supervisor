"""Robot facts and the configurable environment definition for Phase 1.

The joint/site names below were confirmed by directly opening
``assets/robots/g1/g1_with_hands.xml`` (Unitree G1 with hands, from
mujoco_menagerie) -- they are not guessed. See PROJECT_CONTEXT.md Phase 1
report for the inspection notes. If the vendored model file is ever updated,
re-check these names before trusting them again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Robot facts (confirmed from g1_with_hands.xml)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
G1_XML_PATH = PROJECT_ROOT / "assets" / "robots" / "g1" / "g1_with_hands.xml"

FLOATING_BASE_JOINT = "floating_base_joint"
STAND_KEYFRAME = "stand"

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Actuator names are identical to joint names in g1_with_hands.xml
# (each <position name="X" joint="X"/>), so the same lists index actuators.
LEFT_ARM_ACTUATORS = LEFT_ARM_JOINTS
RIGHT_ARM_ACTUATORS = RIGHT_ARM_JOINTS

# End-effector reference sites. Not present in the stock model; added at
# build time (see model_builder.py) at the same offset as the existing
# left/right_hand_palm_link geom, so they sit at the palm center.
LEFT_EE_SITE = "left_ee"
RIGHT_EE_SITE = "right_ee"
LEFT_EE_OFFSET = (0.0415, 0.003, 0.0)
RIGHT_EE_OFFSET = (0.0415, -0.003, 0.0)

TABLE_BODY = "table"
TABLE_GEOM = "table_geom"
OBJECT_BODY = "object"
OBJECT_GEOM = "object_geom"
OBJECT_JOINT = "object_joint"

OBJECT_HALF_SIZE = 0.03  # 6cm cube
OBJECT_TABLE_GAP = 0.002  # small clearance to avoid initial interpenetration


# ---------------------------------------------------------------------------
# Verified workspace (see Phase 1 report): with the pelvis fixed at
# z=0.793 and legs/waist held in the "stand" pose, the left shoulder sits at
# approximately (x=0.0, y=0.10, z=1.08). Sweeping shoulder_pitch/elbow with
# shoulder_roll=0 showed reachable end-effector points such as
# (x=0.25, y=0.13, z=0.77) and (x=0.34, y=0.10, z=0.92) using joint angles
# well inside their limits. The table/object ranges below were chosen to
# stay clearly inside that verified envelope.
# ---------------------------------------------------------------------------

DEFAULT_TABLE_POS = (0.30, 0.0, 0.70)
DEFAULT_TABLE_HALF_SIZE = (0.15, 0.35, 0.05)  # top surface at z = 0.75


@dataclass
class EnvConfig:
    """All Phase 1 environment parameters. Nothing here is hard-coded in env code."""

    seed: int | None = None
    max_episode_steps: int = 200
    frame_skip: int = 5

    action_scale: float = 0.05  # radians of joint-angle delta per control step, per unit action

    object_x_range: tuple[float, float] = (0.22, 0.32)
    object_y_range: tuple[float, float] = (-0.08, 0.08)
    object_z: float = (
        DEFAULT_TABLE_POS[2] + DEFAULT_TABLE_HALF_SIZE[2] + OBJECT_HALF_SIZE + OBJECT_TABLE_GAP
    )

    left_target_offset: tuple[float, float, float] = (0.0, 0.08, 0.0)
    right_target_offset: tuple[float, float, float] = (0.0, -0.08, 0.0)

    success_threshold: float = 0.05
    success_bonus: float = 1.0

    table_pos: tuple[float, float, float] = DEFAULT_TABLE_POS
    table_half_size: tuple[float, float, float] = DEFAULT_TABLE_HALF_SIZE

    g1_xml_path: Path = field(default_factory=lambda: G1_XML_PATH)

    @staticmethod
    def from_yaml(path: str | Path) -> "EnvConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        def tup(x):
            return tuple(x) if x is not None else None

        kwargs = {}
        if "seed" in raw:
            kwargs["seed"] = raw["seed"]
        if "max_episode_steps" in raw:
            kwargs["max_episode_steps"] = raw["max_episode_steps"]
        if "simulation" in raw and "frame_skip" in raw["simulation"]:
            kwargs["frame_skip"] = raw["simulation"]["frame_skip"]
        if "action" in raw and "scale" in raw["action"]:
            kwargs["action_scale"] = raw["action"]["scale"]
        if "object" in raw:
            obj = raw["object"]
            if "x_range" in obj:
                kwargs["object_x_range"] = tup(obj["x_range"])
            if "y_range" in obj:
                kwargs["object_y_range"] = tup(obj["y_range"])
            if "z" in obj:
                kwargs["object_z"] = obj["z"]
        if "target" in raw:
            tgt = raw["target"]
            if "left_offset" in tgt:
                kwargs["left_target_offset"] = tup(tgt["left_offset"])
            if "right_offset" in tgt:
                kwargs["right_target_offset"] = tup(tgt["right_offset"])
            if "success_threshold" in tgt:
                kwargs["success_threshold"] = tgt["success_threshold"]
            if "success_bonus" in tgt:
                kwargs["success_bonus"] = tgt["success_bonus"]

        return EnvConfig(**kwargs)
