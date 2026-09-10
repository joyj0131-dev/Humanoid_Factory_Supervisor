"""Two-workcell factory supervisor environment.

Extends :class:`WholeBodyEnv` so the G1's 37-dim whole-body action convention,
index resolution, fall detection and step semantics are shared rather than
reimplemented -- only the scene, the scripted automation cells, the fault
injection and the observation block are new.

What this env does NOT do, stated plainly because it is easy to overclaim:
it contains no walking controller. The base is free (``floating_base_joint`` is
kept), so navigating between the two cells has to come from the legs, and no
policy in this repository can currently do that. :class:`NavigationTracker`
therefore scores an *externally supplied* trajectory against the Navigation
Gate; it never moves the robot, and reaching the gate is not claimed anywhere.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from humanoid_learning.envs import factory_config as fcfg
from humanoid_learning.envs import factory_model
from humanoid_learning.envs import frames
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs.whole_body_env import ACTION_DIM, WholeBodyEnv

# Fraction of each dwell spent ramping to the next waypoint; the rest is settle
# time. See ScriptedArm.target().
RAMP_FRACTION = 0.6

# Observation block appended after WholeBodyEnv's own pieces.
PER_WORKCELL_OBS = 10
FACTORY_OBS_DIM = fcfg.N_WORKCELLS * PER_WORKCELL_OBS + 11

FACTORY_OBS_LAYOUT = (
    "per workcell k in 0..1: manipulation pose in base frame (x,y), heading error "
    "(cos,sin), part position in base frame (x,y,z), arm cycle phase (cos,sin), "
    "arm fault flag; then: fault_active, target one-hot (2), target part in base "
    "frame (x,y,z), target manipulation pose in base frame (x,y), target heading "
    "error (cos,sin), belt_running"
)


class ScriptedArm:
    """Deterministic joint-space cycle for one automation arm and its gripper.

    Drives real position actuators; it never writes geom or body poses. A
    faulted arm stops advancing its phase and holds the fault waypoint, so the
    stall is visible in the scene and in the observation.
    """

    def __init__(self, index: int, model: mujoco.MjModel):
        self.index = index
        self.waypoints = fcfg.arm_cycle_waypoints()
        self.jaw_targets = fcfg.arm_cycle_jaw_targets()
        self.hold_steps = fcfg.ARM_CYCLE_HOLD_STEPS

        def act(suffix):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, fcfg.arm_joint_name(index, suffix))
            assert aid >= 0, f"actuator {suffix} missing on arm {index}"
            return aid

        def qadr(suffix):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fcfg.arm_joint_name(index, suffix))
            assert jid >= 0, f"joint {suffix} missing on arm {index}"
            return model.jnt_qposadr[jid]

        self.act_ids = np.array([act(s) for s in fcfg.ARM_JOINT_SUFFIXES])
        self.qpos_adr = np.array([qadr(s) for s in fcfg.ARM_JOINT_SUFFIXES])
        self.jaw_act_ids = np.array([act(s) for s in fcfg.GRIPPER_JOINT_SUFFIXES])
        self.jaw_qpos_adr = np.array([qadr(s) for s in fcfg.GRIPPER_JOINT_SUFFIXES])
        self.reset()

    def reset(self, phase_offset: int = 0) -> None:
        self._tick = int(phase_offset)
        self.faulted = False

    @property
    def cycle_length(self) -> int:
        return len(self.waypoints) * self.hold_steps

    @property
    def waypoint_index(self) -> int:
        if self.faulted:
            return fcfg.ARM_FAULT_WAYPOINT_INDEX
        return (self._tick // self.hold_steps) % len(self.waypoints)

    @property
    def phase(self) -> float:
        """Position in the cycle, 0..1. Frozen while faulted."""
        if self.faulted:
            return fcfg.ARM_FAULT_WAYPOINT_INDEX / len(self.waypoints)
        return (self._tick % self.cycle_length) / self.cycle_length

    def target(self) -> np.ndarray:
        """Smoothly ramp from the previous waypoint, then dwell.

        Holding each waypoint as a step change made the position actuators slam
        toward it; that was survivable on the lift but flung the gripped part
        off the line during the rotation to the outfeed. The ramp reaches the
        waypoint at RAMP_FRACTION of the dwell and holds for the remainder, so
        the arm still settles before the next stage begins.
        """
        current = np.asarray(self.waypoints[self.waypoint_index], dtype=np.float64)
        if self.faulted:
            return current
        index = self.waypoint_index
        previous = np.asarray(self.waypoints[(index - 1) % len(self.waypoints)], dtype=np.float64)
        phase = (self._tick % self.hold_steps) / self.hold_steps
        alpha = min(1.0, phase / RAMP_FRACTION)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep: zero velocity at both ends
        return previous + alpha * (current - previous)

    def jaw_target(self) -> float:
        """A faulted arm releases: the jaws open, which is what dropped the part."""
        if self.faulted:
            return fcfg.GRIPPER_OPEN
        return float(self.jaw_targets[self.waypoint_index])

    def apply(self, data: mujoco.MjData) -> None:
        data.ctrl[self.act_ids] = self.target()
        data.ctrl[self.jaw_act_ids] = self.jaw_target()

    def advance(self) -> None:
        if not self.faulted:
            self._tick += 1


class ConveyorBelt:
    """Friction-drive model of a running belt.

    MuJoCo has no conveyor primitive. Parts resting on the belt are pushed
    toward the belt velocity with a viscous force; parts the arm has lifted are
    out of contact and therefore undriven, which falls out of the physics rather
    than needing a special case. Stopping the line simply stops applying the
    drive -- the parts then coast to a halt against friction.
    """

    def __init__(self, model: mujoco.MjModel):
        self.belt_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, fcfg.BELT_GEOM)
        assert self.belt_geom >= 0, "conveyor belt geom missing"
        self.running = True

    def parts_on_belt(self, env) -> set[int]:
        touching = set()
        for i in range(env.data.ncon):
            contact = env.data.contact[i]
            geoms = {contact.geom1, contact.geom2}
            if self.belt_geom not in geoms:
                continue
            other = (geoms - {self.belt_geom}).pop()
            body = env.model.geom_bodyid[other]
            if body in env._part_body_ids:
                touching.add(int(body))
        return touching

    def drive(self, env) -> None:
        env.data.xfrc_applied[:] = 0.0
        if not self.running:
            return
        target_velocity = fcfg.BELT_SPEED * fcfg.BELT_DIRECTION
        on_belt = self.parts_on_belt(env)
        for index, body in enumerate(env._part_body_ids):
            if body not in on_belt:
                continue
            dof = env._part_dof_adr[index]
            velocity_y = float(env.data.qvel[dof + 1])
            mass = float(env.model.body_mass[body])
            force = float(
                np.clip(
                    fcfg.BELT_DRIVE_GAIN * mass * (target_velocity - velocity_y),
                    -fcfg.BELT_MAX_DRIVE_N,
                    fcfg.BELT_MAX_DRIVE_N,
                )
            )
            env.data.xfrc_applied[body, 1] = force
            # A surface drive acts at the contact patch, not the centre of mass.
            # Applying it at the COM alone tipped the part and made it hop along
            # the belt (z wandering 0.809-0.832); the matching torque r x F, with
            # r the COM-to-underside offset, puts the push back where the belt
            # actually touches it.
            env.data.xfrc_applied[body, 3] = env.factory.part_half_size * force


class LineState(Enum):
    """Task-manager states for the line."""

    RUNNING = auto()
    FAULT_RAISED = auto()   # line stopped, humanoid called, waiting for recovery
    RECOVERY_VERIFIED = auto()  # part is back; the line is about to restart


class FactoryTaskManager:
    """The signalling layer between the automation and the supervisor.

    Fault  -> stop the belt, stall the faulted arm, raise a call for the G1.
    Recovery -> judged from the part's MEASURED pose (never a flag the G1 sets),
                then the line restarts and the arm resumes.

    Recovery cannot happen on its own: no policy in this repository can move the
    part back. The state machine exists so the loop is closed once one does, and
    so it can be exercised deliberately in tests.
    """

    def __init__(self, config: fcfg.FactoryConfig):
        self.config = config
        self.reset(0, config.fault.fixed_step)

    def reset(self, fault_station: int, fault_step: int) -> None:
        self.state = LineState.RUNNING
        self.fault_station = int(fault_station)
        self.fault_step = int(fault_step)
        self.fault_triggered_step: int | None = None
        self.recovered_step: int | None = None
        self._recovery_hold = 0

    @property
    def fault_active(self) -> bool:
        return self.state is LineState.FAULT_RAISED

    @property
    def belt_should_run(self) -> bool:
        return self.state is not LineState.FAULT_RAISED

    @property
    def target_station(self) -> int:
        """Which station the G1 is being called to, or -1 when none."""
        return self.fault_station if self.fault_active else -1

    def recovery_progress(self, env) -> float:
        return self._recovery_hold / max(1, fcfg.RECOVERY_HOLD_STEPS)

    def update(self, env, step_count: int) -> None:
        if self.state is LineState.RUNNING:
            if env.factory.fault.enabled and self.fault_triggered_step is None and step_count >= self.fault_step:
                env._trigger_fault(self.fault_station)
                self.state = LineState.FAULT_RAISED
                self.fault_triggered_step = step_count
            return

        if self.state is LineState.FAULT_RAISED:
            if self._part_is_back(env):
                self._recovery_hold += 1
                if self._recovery_hold >= fcfg.RECOVERY_HOLD_STEPS:
                    self.state = LineState.RECOVERY_VERIFIED
                    self.recovered_step = step_count
            else:
                self._recovery_hold = 0
            return

        if self.state is LineState.RECOVERY_VERIFIED:
            # Signal back: restart the belt and let the stalled arm resume.
            env.arms[self.fault_station].faulted = False
            self.state = LineState.RUNNING

    def _part_is_back(self, env) -> bool:
        """Measured: the part sits at its pick spot and has stopped moving."""
        index = self.fault_station
        pose = env.poses[index]
        part = env.part_position(index)
        local = pose.to_local_xy(part[:2])
        if float(np.linalg.norm(local - np.asarray(fcfg.LOCAL_CANONICAL_PART_XY))) > fcfg.RECOVERY_POSITION_TOLERANCE_M:
            return False
        if abs(float(part[2]) - env._part_rest_pos(index, fcfg.LOCAL_CANONICAL_PART_XY)[2]) > 0.05:
            return False
        dof = env._part_dof_adr[index]
        return float(np.abs(env.data.qvel[dof : dof + 6]).max()) < fcfg.RECOVERY_SETTLE_SPEED


class NavigationTracker:
    """Scores a trajectory against :class:`fcfg.NavigationGate`.

    Every criterion is evaluated from measured state. ``teleported`` exists so a
    kinematic oracle that slides the base cannot be mistaken for walking: the
    tracker reports it, and :meth:`result` fails the gate when it is set.
    """

    def __init__(self, config: fcfg.FactoryConfig, target_index: int, poses: list[fcfg.WorkcellPose]):
        self.gate = config.navigation
        self.target_index = target_index
        self.poses = poses
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.hold_steps = 0
        self.max_hold_steps = 0
        self.max_base_step_m = 0.0
        self.teleported = False
        self.fell = False
        self.forbidden_contact = False
        self.entered_wrong_first = False
        self._entered_any = False
        self._prev_xy: np.ndarray | None = None
        self.position_error_m = float("inf")
        self.heading_error_rad = float("inf")

    def update(self, base_xy: np.ndarray, base_yaw: float, fallen: bool, forbidden_contact: bool) -> None:
        base_xy = np.asarray(base_xy, dtype=float)[:2]
        if self._prev_xy is not None:
            step = float(np.linalg.norm(base_xy - self._prev_xy))
            self.max_base_step_m = max(self.max_base_step_m, step)
            if step > self.gate.max_base_step_m:
                self.teleported = True
        self._prev_xy = base_xy.copy()
        self.steps += 1
        self.fell = self.fell or bool(fallen)
        self.forbidden_contact = self.forbidden_contact or bool(forbidden_contact)

        target = self.poses[self.target_index]
        self.position_error_m = float(np.linalg.norm(base_xy - np.asarray(target.manipulation_xy)))
        self.heading_error_rad = float(abs(_wrap_angle(base_yaw - target.heading_rad)))

        # "Entered" a cell means standing inside its manipulation marker area.
        entry_radius = 3.0 * self.gate.max_position_error_m
        for pose in self.poses:
            if float(np.linalg.norm(base_xy - np.asarray(pose.manipulation_xy))) <= entry_radius:
                if not self._entered_any and pose.index != self.target_index:
                    self.entered_wrong_first = True
                self._entered_any = True

        at_target = (
            self.position_error_m <= self.gate.max_position_error_m
            and self.heading_error_rad <= self.gate.max_heading_error_rad
            and not fallen
        )
        self.hold_steps = self.hold_steps + 1 if at_target else 0
        self.max_hold_steps = max(self.max_hold_steps, self.hold_steps)

    def result(self) -> dict[str, Any]:
        reached = self.max_hold_steps >= self.gate.min_hold_steps
        passed = bool(
            reached
            and not self.fell
            and not self.forbidden_contact
            and not self.teleported
            and not self.entered_wrong_first
            and self.steps <= self.gate.max_steps
        )
        return {
            "passed": passed,
            "reached_and_held": bool(reached),
            "target_workcell": int(self.target_index),
            "final_position_error_m": self.position_error_m,
            "final_heading_error_rad": self.heading_error_rad,
            "max_hold_steps": int(self.max_hold_steps),
            "required_hold_steps": int(self.gate.min_hold_steps),
            "steps": int(self.steps),
            "max_steps": int(self.gate.max_steps),
            "fell": bool(self.fell),
            "forbidden_contact": bool(self.forbidden_contact),
            "teleported": bool(self.teleported),
            "max_base_step_m": self.max_base_step_m,
            "entered_wrong_workcell_first": bool(self.entered_wrong_first),
        }


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class FactoryEnv(WholeBodyEnv):
    """G1 supervisor over two independent scripted automation cells."""

    def __init__(self, config: fcfg.FactoryConfig | None = None, render_mode: str | None = None):
        self.factory = config or fcfg.FactoryConfig()
        self.poses = self.factory.workcells
        super().__init__(config=self.factory.whole_body, include_object=False, render_mode=render_mode)

    # ------------------------------------------------------------------
    def _build_model(self) -> mujoco.MjModel:
        return factory_model.build_factory_model(self.factory)

    def _resolve_indices(self) -> None:
        super()._resolve_indices()
        model = self.model
        self.arms = [ScriptedArm(k, model) for k in range(fcfg.N_WORKCELLS)]
        self.belt = ConveyorBelt(model)
        self.task_manager = FactoryTaskManager(self.factory)
        self._part_qpos_adr = []
        self._part_dof_adr = []
        self._part_body_ids = []
        for k in range(fcfg.N_WORKCELLS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fcfg.part_joint_name(k))
            assert jid >= 0, f"part joint {k} not found"
            self._part_qpos_adr.append(model.jnt_qposadr[jid])
            self._part_dof_adr.append(model.jnt_dofadr[jid])
            self._part_body_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fcfg.part_body_name(k)))
        self._forbidden_body_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.factory.navigation.forbidden_contact_bodies
        }
        self._forbidden_body_ids.discard(-1)
        self._floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # Scenario defaults so _get_obs/_get_info are well defined during the
        # base class's constructor-time reset, before reset() draws a scenario.
        self.fault_workcell = 0
        self.fault_step = int(self.factory.fault.fixed_step)
        self.drop_local_xy = np.asarray(fcfg.LOCAL_DROP_ZONE_XY, dtype=float)

    # ------------------------------------------------------------------
    def _part_rest_pos(self, index: int, local_xy) -> np.ndarray:
        xy = self.poses[index].to_world_xy(local_xy)
        z = fcfg.TABLE_TOP_Z + self.factory.part_half_size + tc.OBJECT_TABLE_GAP
        return np.array([xy[0], xy[1], z])

    def _set_part(self, index: int, position: np.ndarray) -> None:
        adr = self._part_qpos_adr[index]
        self.data.qpos[adr : adr + 3] = position
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        dof = self._part_dof_adr[index]
        self.data.qvel[dof : dof + 6] = 0.0

    def part_position(self, index: int) -> np.ndarray:
        adr = self._part_qpos_adr[index]
        return self.data.qpos[adr : adr + 3].copy()

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        options = options or {}
        # Draw the scenario from a dedicated stream so the fault choice depends
        # only on the episode seed, never on how many physics steps have run.
        scenario_rng = np.random.default_rng(0 if seed is None else int(seed))
        fault_cfg = self.factory.fault

        self.fault_workcell = int(options.get("fault_workcell", scenario_rng.integers(0, fcfg.N_WORKCELLS)))
        if not 0 <= self.fault_workcell < fcfg.N_WORKCELLS:
            raise ValueError(f"fault_workcell must be 0..{fcfg.N_WORKCELLS - 1}")
        if "fault_step" in options:
            self.fault_step = int(options["fault_step"])
        elif fault_cfg.randomize_time:
            self.fault_step = int(scenario_rng.integers(fault_cfg.min_step, fault_cfg.max_step + 1))
        else:
            self.fault_step = int(fault_cfg.fixed_step)

        jitter = np.zeros(2)
        if fault_cfg.randomize_drop_xy:
            jitter = scenario_rng.uniform(-1.0, 1.0, size=2) * np.asarray(fcfg.DROP_ZONE_JITTER_XY)
        self.drop_local_xy = np.asarray(fcfg.LOCAL_DROP_ZONE_XY, dtype=float) + jitter

        obs, info = super().reset(seed=seed, options=options)

        for k, arm in enumerate(self.arms):
            # Offset the two cells so they are visibly out of phase and a test
            # cannot pass by accidentally comparing two synchronised arms.
            arm.reset(phase_offset=k * arm.cycle_length // 3)
            arm.apply(self.data)
            self.data.qpos[arm.qpos_adr] = arm.target()
            self.data.qpos[arm.jaw_qpos_adr] = arm.jaw_target()
            self._set_part(k, self._part_rest_pos(k, fcfg.LOCAL_CANONICAL_PART_XY))

        self.task_manager.reset(self.fault_workcell, self.fault_step)
        self.belt.running = True
        self.data.xfrc_applied[:] = 0.0
        self.navigation = NavigationTracker(self.factory, self.fault_workcell, self.poses)

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = self._get_info()
        self._update_navigation(info)
        return obs, info

    # ------------------------------------------------------------------
    @property
    def fault_active(self) -> bool:
        return self.task_manager.fault_active

    @property
    def line_state(self) -> "LineState":
        return self.task_manager.state

    def _trigger_fault(self, index: int) -> None:
        """Pick failure: the arm's jaws open and the part ends up past the stop
        blade, where the arm's own cycle never reaches. The release point is
        scripted fault injection (the arms are fault-generation devices, not
        research subjects); the landing is real physics and is measured."""
        position = self._part_rest_pos(index, self.drop_local_xy)
        position[2] += self.factory.fault.release_height_m
        self._set_part(index, position)
        self.arms[index].faulted = True

    def step(self, action: np.ndarray):
        self.task_manager.update(self, self._step_count)
        self.belt.running = self.task_manager.belt_should_run
        self.belt.drive(self)

        for arm in self.arms:
            arm.apply(self.data)

        obs, reward, terminated, truncated, info = super().step(action)

        for arm in self.arms:
            arm.advance()
        if not info.get("unstable", False):
            info = self._get_info()
            self._update_navigation(info)
            obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _forbidden_contact(self) -> bool:
        """A pelvis or torso touching anything is a fall-grade collision. Foot
        and hand contacts are normal and are not counted."""
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            bodies = {self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]}
            if bodies & self._forbidden_body_ids:
                return True
        return False

    def _update_navigation(self, info: dict[str, Any]) -> None:
        pelvis_quat = self.data.xquat[self._pelvis_body_id]
        self.navigation.update(
            base_xy=info["pelvis_pos"][:2],
            base_yaw=frames.yaw_from_quat(pelvis_quat),
            fallen=info["fallen"],
            forbidden_contact=self._forbidden_contact(),
        )

    def navigation_result(self) -> dict[str, Any]:
        return self.navigation.result()

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()
        if not hasattr(self, "arms"):
            return base  # during WholeBodyEnv.__init__, before _resolve_indices
        pelvis_pos = self.data.xpos[self._pelvis_body_id]
        yaw = frames.yaw_from_quat(self.data.xquat[self._pelvis_body_id])

        pieces: list[np.ndarray] = []
        per_cell: list[np.ndarray] = []
        for k, pose in enumerate(self.poses):
            manip_world = np.array([pose.manipulation_xy[0], pose.manipulation_xy[1], 0.0])
            manip_base = frames.world_to_base(manip_world, pelvis_pos, yaw)[:2]
            heading_error = _wrap_angle(pose.heading_rad - yaw)
            part_base = frames.world_to_base(self.part_position(k), pelvis_pos, yaw)
            phase = 2.0 * np.pi * self.arms[k].phase
            cell = np.concatenate(
                [
                    manip_base,
                    [np.cos(heading_error), np.sin(heading_error)],
                    part_base,
                    [np.cos(phase), np.sin(phase)],
                    [1.0 if self.arms[k].faulted else 0.0],
                ]
            )
            per_cell.append(cell)
        pieces.extend(per_cell)

        target_onehot = np.zeros(fcfg.N_WORKCELLS)
        # Before the fault fires there is no target to point at; the one-hot
        # stays all-zero so the learner cannot read the answer early.
        if self.fault_active:
            target_onehot[self.fault_workcell] = 1.0
            target_cell = per_cell[self.fault_workcell]
            target_part = target_cell[4:7]
            target_manip = target_cell[0:2]
            target_heading = target_cell[2:4]
        else:
            target_part = np.zeros(3)
            target_manip = np.zeros(2)
            target_heading = np.zeros(2)
        pieces.append(
            np.concatenate(
                [
                    [1.0 if self.fault_active else 0.0],
                    target_onehot,
                    target_part,
                    target_manip,
                    target_heading,
                    [1.0 if self.belt.running else 0.0],
                ]
            )
        )
        return np.concatenate([base, *pieces]).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        info.update(
            {
                "fault_active": bool(getattr(self, "fault_active", False)),
                "fault_workcell": int(getattr(self, "fault_workcell", -1)),
                "fault_step": int(getattr(self, "fault_step", -1)),
                "fault_triggered_step": self.task_manager.fault_triggered_step,
                "recovered_step": self.task_manager.recovered_step,
                "line_state": self.task_manager.state.name,
                "belt_running": bool(self.belt.running),
                "target_workcell": self.task_manager.target_station,
            }
        )
        if hasattr(self, "arms"):
            info["arm_phase"] = [float(a.phase) for a in self.arms]
            info["arm_waypoint"] = [int(a.waypoint_index) for a in self.arms]
            info["arm_faulted"] = [bool(a.faulted) for a in self.arms]
            info["part_position"] = [self.part_position(k) for k in range(fcfg.N_WORKCELLS)]
            info["jaw_opening"] = [float(self.data.qpos[a.jaw_qpos_adr[0]]) for a in self.arms]
            info["recovery_progress"] = self.task_manager.recovery_progress(self)
            info["forbidden_contact"] = self._forbidden_contact()
        return info
