"""Config for the fixed-base grasp validation environment -- see
PROJECT_CONTEXT.md Phase 4 (Grasp Track). Kept separate from
task_config.EnvConfig (Foundation, untouched) and whole_body_config.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from humanoid_learning.envs import task_config as tc

# Condition B ("physical sensitivity"): mass consistent with the SAME
# density as the original 0.1kg / 6cm-cube object, at 2x linear scale
# (volume scales as 2**3 = 8x). Used only in explicit condition-B runs.
DENSITY_SCALED_MASS_8X = 0.1 * 8  # 0.8 kg

# Named object-size presets for the multi-size grasp track (Section 2):
# both required test sizes (6cm, 12cm) plus a 9cm interpolation-only probe
# (Section 12). Mass held equal across sizes (Condition A, "controller
# validation") -- density-scaled Condition B is a separate, later
# experiment, never blended with these.
SIZE_6_HALF = tc.OBJECT_HALF_SIZE  # 0.03 -- the ORIGINAL Foundation size,
# also matching Unitree's own official PickPlace-RedBlock-Dex3 Isaac Lab
# task object (0.06 full side length in that repo's CuboidCfg convention),
# confirmed via direct official-source comparison (see PROJECT_CONTEXT.md).
SIZE_9_HALF = 0.045  # interpolation-only diagnostic, not a required Gate
SIZE_12_HALF = tc.OBJECT_HALF_SIZE * 2  # 0.06 -- the size used since the
# earlier "2x" grasp-track session; kept as a required size, not replaced.


@dataclass
class GraspSuccessThresholds:
    min_lift_height: float = 0.02  # meters above the object's resting height
    min_contact_steps: int = 20  # consecutive control steps with hand-object contact
    min_hold_steps: int = 40  # control steps held at lift height without dropping


@dataclass
class GraspEnvConfig:
    seed: int | None = None
    frame_skip: int = 5
    max_episode_steps: int = 600

    arm_action_scale: float = 0.05  # matches Foundation EnvConfig.action_scale
    # Waist action channel, added for the coupled bilateral waist-aware IK
    # session -- same scale as the arms so a coupled-IK-derived joint delta
    # converts to an action the same way regardless of which of the 17
    # coupled DoF it applies to.
    waist_action_scale: float = 0.05
    hand_synergy_action_scale: float = 0.05
    # Foundation's arm actuators are kp=500 (rigid) -- fine for reaching in
    # free space, but under sustained bimanual contact this position
    # controller generates 30N+ squeeze forces for millimeter-scale errors,
    # popping a light (0.1kg) object out of grip (Phase 4 report). This
    # applies ONLY inside the grasp-specific model (build_grasp_model);
    # Foundation's build_model() and its kp=500 are never touched.
    arm_kp: float = 120.0
    # Finger actuators were left at the stock kp=500 through several
    # earlier sessions; diagnosed during the redesign as the actual source
    # of an 8-13N force spike appearing within a SINGLE control step the
    # instant a fingertip first touched the object (the compliant arm_kp
    # only softens the arm, not the fingers making the actual contact).
    hand_kp: float = 80.0

    # Dex3 Final Control Feasibility session: direct measurement found
    # per-group raw force reaching 25-30N despite grasp_expert.py's own
    # 8N safety regulation, because that regulation only reads/reacts to
    # contact force ONCE per outer env.step() call, AFTER all frame_skip
    # (5) physics substeps have already run -- a spike that builds up and
    # even partially decays entirely WITHIN that 5-substep window is
    # invisible to the control-tick-level check. These two fields let
    # step() itself watch and react to force at EVERY physics substep,
    # inside this grasp-specific env only (Foundation's BimanualReachEnv/
    # WholeBodyEnv step() are untouched). finger_force_warning_ratio *
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
    # Optional grasp-only end-effector collision model: retain the real
    # Dex3 kinematics/actuators but replace irregular hand-mesh contacts
    # with compliant high-friction pads at the six measured fingertips.
    use_fingertip_collision_pads: bool = False
    fingertip_pad_radius: float = 0.012
    fingertip_pad_friction: tuple[float, float, float] = (2.0, 0.02, 0.001)

    # [Session 39, Sharpa track] Compliant kp=120 arm actuators have a real
    # steady-state gravity/load droop (documented in coupled_ik.py's module
    # docstring for Dex3, ~0.04m there; measured up to ~6.9cm for Sharpa's
    # heavier hands at FINGERTIP_PRECONTACT reach -- see
    # docs/history/PHASE4_GRASP_SESSION_39.md). A Cartesian-space IK-resolve
    # compensation (grasp_expert.py's proven `_coupled_maybe_resolve`
    # recipe for Dex3) was tried for Sharpa and causally measured to make
    # this WORSE at this reach. This flag instead adds a direct, physically
    # -grounded feedforward at the actuator: data.qfrc_bias (gravity +
    # Coriolis, already computed by MuJoCo every step) divided by arm_kp,
    # added to the arm/waist ctrl target every tick -- NOT a kp increase,
    # NOT a kinematic guess, a bounded correction sized by the ACTUAL
    # measured physical load. False (default) preserves every existing
    # Sharpa/Dex3 test's behavior unchanged; only SharpaBimanualGraspExpert's
    # env enables it (sharpa_grasp_env.py's step()).
    arm_gravity_compensation: bool = False

    # [Session 40, Sharpa track only -- read by build_grasp_model_sharpa,
    # never by Dex3's build_grasp_model] "wrist" (default, existing
    # behavior) or "flange" -- see model_builder._sharpa_xml_path's
    # docstring and docs/history/PHASE4_GRASP_SESSION_40.md's mount A/B
    # audit for why the two variants measured almost identically (0.5mm
    # hand-base offset difference, not a meaningful structural change).
    sharpa_mount: str = "wrist"
    # [Session 40] "upstream" (default, existing appearance unchanged) or
    # "g1" (recolor VISUAL-only Sharpa geoms to reuse g1_with_hands.xml's
    # own metal/black material rgba -- see model_builder._apply_sharpa_
    # visual_style docstring; never touches collision/mass/inertia).
    sharpa_visual_style: str = "upstream"

    object_pos: tuple[float, float, float] = (0.27, 0.0, 0.0)  # z resolved at reset (on table surface)
    # 2x every linear dimension of the Foundation object (0.03 -> 0.06,
    # i.e. 6cm cube -> 12cm cube) -- see PROJECT_CONTEXT.md Phase 4 Grasp
    # Track: a larger object gives real finger contact more surface to
    # engage instead of relying on a bimanual palm squeeze.
    object_half_size: float = tc.OBJECT_HALF_SIZE * 2  # 0.06

    @property
    def effective_object_half_extents(self) -> tuple[float, float, float]:
        """Return canonical cubic half-extents for model builders.

        Sharpa model construction has consumed this shape-aware interface
        since its integration commit.  The official Phase 4.5 branch still
        uses a cube, so keep the compatibility API independent from the
        uncommitted rectangular-object viewer experiment.
        """
        return (self.object_half_size,) * 3

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

    success: GraspSuccessThresholds = field(default_factory=GraspSuccessThresholds)

    g1_xml_path: str = str(tc.G1_XML_PATH)

    # Net-Torque Root Cause Isolation session: False (default) reproduces
    # the existing --grasp/every current test's model exactly (waist held
    # only by its own compliant position actuator, ~0.088 rad measured
    # drift under grasp-time arm reaction forces). True builds the SAME
    # model with an added physical <equality joint> constraint (see
    # model_builder.build_grasp_model's hard_fixed_waist docstring) pinning
    # each waist joint to its stand value -- a real, substep-enforced
    # constraint, not a per-step qpos teleport -- used ONLY by this
    # session's grasp-only A/B waist experiment.
    hard_fixed_waist: bool = False


def make_grasp_env_config(object_half_size: float, **overrides) -> GraspEnvConfig:
    """Builds a GraspEnvConfig for a given object size, holding everything
    else (friction, kp, timestep/solver via the shared model builder, seed,
    table) identical -- the only input that should differ between the
    SIZE_6/SIZE_9/SIZE_12 conditions (Section 2)."""
    return GraspEnvConfig(object_half_size=object_half_size, **overrides)
