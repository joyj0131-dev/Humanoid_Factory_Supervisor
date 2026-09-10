"""Configuration for the canonical fixed-base Sharpa grasp task."""

from __future__ import annotations

from dataclasses import dataclass

from humanoid_learning.envs import task_config as tc

SIZE_12_HALF = tc.OBJECT_HALF_SIZE * 2  # 0.06m half-size = 12cm cube


@dataclass
class GraspEnvConfig:
    seed: int | None = None
    frame_skip: int = 5
    max_episode_steps: int = 600

    # Phase 5 recovery work: the factory G1 stands on its own two feet, so the
    # grasp has to survive its own arm reaction forces instead of pushing
    # against a welded pelvis. Default stays True so every existing grasp
    # result, demo and gate keeps its original fixed-base meaning.
    fixed_base: bool = True

    arm_action_scale: float = 0.05  # matches Foundation EnvConfig.action_scale
    # Waist action channel, added for the coupled bilateral waist-aware IK
    # session -- same scale as the arms so a coupled-IK-derived joint delta
    # converts to an action the same way regardless of which of the 17
    # coupled DoF it applies to.
    waist_action_scale: float = 0.05
    hand_synergy_action_scale: float = 0.05
    # Grasp-specific compliant arm control; Foundation reach remains kp=500.
    arm_kp: float = 120.0

    # Force safety is evaluated at every physics substep. warning_ratio *
    # finger_force_safety_limit is the "slow down" threshold; the limit
    # itself is a hard "stop closing this group any further this tick"
    # ceiling, not a threshold to relax.
    finger_force_safety_limit: float = 8.0
    finger_force_warning_ratio: float = 0.7
    # True: an over-limit group is actively unloaded by setting its
    # position target to the current joint pose.  False preserves the old
    # "previous target" rollback for controlled comparison.
    unload_finger_on_force_limit: bool = False
    force_unload_fraction: float = 1.0
    persist_safety_synergy_rollback: bool = False
    # Experimental corrected metric; False preserves the calibrated
    # legacy max-single-contact convention until its thresholds are
    # re-derived end-to-end.
    use_net_group_force: bool = False
    # Compliant kp=120 arm actuators have a real
    # steady-state gravity/load droop (measured up to ~6.9cm for Sharpa at
    # FINGERTIP_PRECONTACT reach). A Cartesian re-solve was tried and
    # causally made this worse. This flag
    # instead adds a direct, physically
    # -grounded feedforward at the actuator: data.qfrc_bias (gravity +
    # Coriolis, already computed by MuJoCo every step) divided by arm_kp,
    # added to the arm/waist ctrl target every tick -- NOT a kp increase,
    # NOT a kinematic guess, a bounded correction sized by the ACTUAL
    # measured physical load. False remains the explicit baseline.
    arm_gravity_compensation: bool = False

    # "wrist" (default) or "flange" -- see model_builder._sharpa_xml_path's
    # docstring and docs/history/PHASE4_GRASP_SESSION_40.md's mount A/B
    # audit for why the two variants measured almost identically (0.5mm
    # hand-base offset difference, not a meaningful structural change).
    sharpa_mount: str = "wrist"
    # [Session 40] "upstream" (default, existing appearance unchanged) or
    # "g1" (recolor visual-only Sharpa geoms to reuse the G1 model's
    # own metal/black material rgba -- see model_builder._apply_sharpa_
    # visual_style docstring; never touches collision/mass/inertia).
    sharpa_visual_style: str = "upstream"

    object_pos: tuple[float, float, float] = (0.27, 0.0, 0.0)  # z resolved at reset (on table surface)
    # 2x every linear dimension of the Foundation object (0.03 -> 0.06,
    # i.e. 6cm cube -> 12cm cube) -- see PROJECT_CONTEXT.md Phase 4 Grasp
    # Track: a larger object gives real finger contact more surface to
    # engage instead of relying on a bimanual palm squeeze.
    object_half_size: float = tc.OBJECT_HALF_SIZE * 2  # 0.06
    object_half_extents: tuple[float, float, float] | None = None
    object_yaw_rad: float = 0.0
    object_xy_randomization_m: tuple[float, float] = (0.0, 0.0)
    object_yaw_randomization_rad: float = 0.0

    @property
    def effective_object_half_extents(self) -> tuple[float, float, float]:
        """Box half-extents in metres; explicit XYZ overrides the cubic default."""
        return self.object_half_extents if self.object_half_extents is not None else (self.object_half_size,) * 3

    def __post_init__(self):
        import math
        half = self.effective_object_half_extents
        if len(half) != 3 or any(not math.isfinite(v) or v <= 0 for v in half):
            raise ValueError("object half extents must contain three finite positive lengths in metres")
        if len(self.object_xy_randomization_m) != 2 or any(
                not math.isfinite(v) or v < 0 for v in self.object_xy_randomization_m):
            raise ValueError("XY randomization must contain two finite nonnegative radii in metres")
        if not math.isfinite(self.object_yaw_rad) or not math.isfinite(self.object_yaw_randomization_rad) or self.object_yaw_randomization_rad < 0:
            raise ValueError("yaw must be finite and yaw randomization nonnegative (radians)")

    # Mass is DELIBERATELY NOT density-scaled by default (condition A --
    # "controller validation": isolates whether the controller/contact
    # geometry works at all, holding mass fixed at the Foundation value).
    # Condition B ("physical sensitivity", mass = 0.1 * 2**3 = 0.8, i.e.
    # density held constant while volume grows 8x) is a SEPARATE,
    # explicitly-configured run -- never silently mixed with condition A.
    object_mass: float = 0.1  # condition A default
    # Default MuJoCo friction (1.0, 0.005, 0.0001) was found (Phase 4 whole-
    # body grasp report) too low for a fingertip-based grasp to hold an
    # object without slipping out under squeeze -- raised sliding/torsional
    # friction, matching the value already used in build_whole_body_model's
    # object.
    object_friction: tuple[float, float, float] = (1.5, 0.01, 0.0001)

    table_pos: tuple[float, float, float] = tc.DEFAULT_TABLE_POS
    table_half_size: tuple[float, float, float] = tc.DEFAULT_TABLE_HALF_SIZE

    # Active Sharpa grasp starts from the bare G1; model_builder attaches the
    # hand once, without inherited end-effector mass.
    g1_xml_path: str = str(tc.G1_XML_PATH)
