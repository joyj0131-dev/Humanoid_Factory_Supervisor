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

from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from humanoid_learning.envs import factory_config as fcfg
from humanoid_learning.envs import factory_model
from humanoid_learning.envs import frames
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs.whole_body_env import ACTION_DIM, WholeBodyEnv

# Observation block appended after WholeBodyEnv's own pieces.
PER_WORKCELL_OBS = 10
FACTORY_OBS_DIM = fcfg.N_WORKCELLS * PER_WORKCELL_OBS + 10

FACTORY_OBS_LAYOUT = (
    "per workcell k in 0..1: manipulation pose in base frame (x,y), heading error "
    "(cos,sin), part position in base frame (x,y,z), arm cycle phase (cos,sin), "
    "arm fault flag; then: fault_active, target one-hot (2), target part in base "
    "frame (x,y,z), target manipulation pose in base frame (x,y) and heading "
    "error (cos,sin)"
)


class ScriptedArm:
    """Deterministic joint-space cycle for one automation arm.

    Drives real position actuators; it never writes geom or body poses. A
    faulted arm stops advancing its phase and holds the fault waypoint, so the
    stall is visible in the scene and in the observation.
    """

    def __init__(self, index: int, model: mujoco.MjModel):
        self.index = index
        self.waypoints = fcfg.arm_cycle_waypoints()
        self.hold_steps = fcfg.ARM_CYCLE_HOLD_STEPS
        self.act_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, fcfg.arm_joint_name(index, s))
                for s in fcfg.ARM_JOINT_SUFFIXES
            ]
        )
        assert (self.act_ids >= 0).all(), f"automation arm {index} actuators not found"
        self.qpos_adr = np.array(
            [
                model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fcfg.arm_joint_name(index, s))]
                for s in fcfg.ARM_JOINT_SUFFIXES
            ]
        )
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
        return np.asarray(self.waypoints[self.waypoint_index], dtype=np.float64)

    def apply(self, data: mujoco.MjData) -> None:
        data.ctrl[self.act_ids] = self.target()

    def advance(self) -> None:
        if not self.faulted:
            self._tick += 1


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
        self.fault_active = False
        self.fault_triggered_step = None
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
            self._set_part(k, self._part_rest_pos(k, fcfg.LOCAL_CANONICAL_PART_XY))

        self.fault_active = False
        self.fault_triggered_step: int | None = None
        self.navigation = NavigationTracker(self.factory, self.fault_workcell, self.poses)

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = self._get_info()
        self._update_navigation(info)
        return obs, info

    # ------------------------------------------------------------------
    def _trigger_fault(self) -> None:
        """Release the part above the drop zone so it falls and settles under
        gravity. The release itself is scripted fault injection (the arms are
        fault-generation devices, not research subjects); the landing is real
        physics and is measured, not assumed."""
        index = self.fault_workcell
        position = self._part_rest_pos(index, self.drop_local_xy)
        position[2] += self.factory.fault.release_height_m
        self._set_part(index, position)
        self.arms[index].faulted = True
        self.fault_active = True
        self.fault_triggered_step = self._step_count

    def step(self, action: np.ndarray):
        if self.factory.fault.enabled and not self.fault_active and self._step_count >= self.fault_step:
            self._trigger_fault()

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
                [[1.0 if self.fault_active else 0.0], target_onehot, target_part, target_manip, target_heading]
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
                "fault_triggered_step": getattr(self, "fault_triggered_step", None),
                "target_workcell": int(self.fault_workcell) if getattr(self, "fault_active", False) else -1,
            }
        )
        if hasattr(self, "arms"):
            info["arm_phase"] = [float(a.phase) for a in self.arms]
            info["arm_waypoint"] = [int(a.waypoint_index) for a in self.arms]
            info["arm_faulted"] = [bool(a.faulted) for a in self.arms]
            info["part_position"] = [self.part_position(k) for k in range(fcfg.N_WORKCELLS)]
            info["forbidden_contact"] = self._forbidden_contact()
        return info
