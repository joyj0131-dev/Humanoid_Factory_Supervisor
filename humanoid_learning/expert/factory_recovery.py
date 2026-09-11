"""Experimental fault-to-lift supervisor in FactoryEnv's single physics world.

Only actuator commands move G1. A lift is not a completed repair: production
stays stopped until a separate place-and-verification skill restores the part.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from humanoid_learning.envs import factory_config as fc, model_builder, task_config as tc
from humanoid_learning.envs.grasp_config import GraspEnvConfig
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.g1_walk_policy import G1WalkPolicy, stand_pose_for_part
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SharpaBimanualGraspExpert
from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift
from humanoid_learning.expert.stance_stabilizer import StanceGains, StanceStabilizer


@dataclass
class RecoveryConfig:
    motion_profile: str = 'baseline'
    stand_off_m: float = 0.27
    arrival_radius_m: float = 0.03
    settle_steps: int = 200
    max_steps: int = 12000
    lift_clearance_m: float = 0.05
    lift_hold_seconds: float = 5.0
    palm_first_closure: bool = False
    hold_squeeze_m: float = 0.006
    contact_settle_grace_seconds: float = 2.5
    # Raised hands change this pretrained gait's zero-speed operating point.
    # A positive bias caused forward overshoot into the conveyor rail.
    forward_command_bias: float = -0.08


class PrecisionApproach:
    """Slow near the actual part; hand off measured targets at double support."""

    def __init__(self, walker, config):
        self.walker, self.config = walker, config
        self.integral = np.zeros(2)
        self.velocity = np.zeros(2)
        self.near = self.arrived = False
        self.error = np.full(2, np.inf)

    def step(self, part_position):
        w = self.walker
        position, yaw = w.base_pose()
        goal = stand_pose_for_part(part_position, 0.0,
                                   canonical_local_xy=(self.config.stand_off_m, 0.0))
        self.error = goal - position
        dt = w.env.model.opt.timestep * w.env.config.frame_skip
        self.integral = np.clip(self.integral + self.error * dt, -0.5, 0.5)
        if np.linalg.norm(self.error) < 0.35 and not self.near:
            self.near = True
            self.integral[:] = 0.0
        self.velocity = 0.95 * self.velocity + 0.05 * w.env.data.qvel[:2]
        c, s = np.cos(yaw), np.sin(yaw)
        command = np.array([[c, s], [-s, c]]) @ (
            1.4 * self.error + 0.6 * self.integral - 0.6 * self.velocity)
        command[0] += self.config.forward_command_bias
        if np.linalg.norm(self.error) < 0.2:
            command[1] = np.clip(command[1], -0.06, 0.06)
        if (np.linalg.norm(self.error) < self.config.arrival_radius_m
                and abs(yaw) < 0.06 and np.linalg.norm(self.velocity) < 0.07
                and w.in_double_support()):
            w.hold_stance()
            self.arrived = True
        w.step(np.r_[np.clip(command, -0.3, 0.3), np.clip(-2 * yaw, -0.3, 0.3)])
        w.env.data.ctrl[w.act_ids] = w.env._leg_target


class FactoryRecovery:
    """One tick advances physics exactly once, including the scripted factory."""

    TERMINAL = ('LIFTED', 'FAILED')

    def __init__(self, env, config: RecoveryConfig | None = None):
        self.env, self.config = env, config or RecoveryConfig()
        self.used_direct_approach = None
        if self.config.motion_profile not in ('baseline', 'compact', 'direct'):
            raise ValueError('motion_profile must be baseline, compact or direct')
        gain, bias = env.model.actuator_gainprm.copy(), env.model.actuator_biasprm.copy()
        self._original_gain, self._original_bias = gain, bias
        self._original_noslip = env.model.opt.noslip_iterations
        self.walker = G1WalkPolicy(env, initialize_pose=False)
        # Wait/prepare on the original standing plant, not soft gait gains.
        env.model.actuator_gainprm[:] = gain
        env.model.actuator_biasprm[:] = bias
        self.state, self.station = 'WAIT_FAULT', None
        self.ticks = self.total_steps = 0
        self.failure = None
        self.events = []
        self.hold_steps = self.max_hold_steps = 0
        self.max_clearance = 0.0
        self.max_object_hand_penetration_m = 0.0
        self.forbidden_contact_ticks = 0
        self.forbidden_contact_pairs = {}
        self._last_info = env._get_info()
        self.grasp = self._grasp_view(0)
        model_builder._apply_compliant_kp(
            env.model, tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS, self.grasp.config.arm_kp)
        self.stabilizer = StanceStabilizer(
            self.grasp, StanceGains(pitch_kp=1.0, pitch_kd=0.1, roll_kp=0.7, roll_kd=0.07))

    def _grasp_view(self, station):
        e = self.env
        config = GraspEnvConfig(fixed_base=False, arm_gravity_compensation=True,
                                max_episode_steps=self.config.max_steps)
        return SharpaGraspEnv(
            config, shared_model=e.model, shared_data=e.data,
            object_joint=fc.part_joint_name(station), object_body=fc.part_body_name(station),
            object_geom=fc.part_geom_name(station), support_geom=fc.BELT_GEOM)

    def _transition(self, state):
        self.events.append({'step': self.env._step_count, 'state': state})
        self.state, self.ticks = state, 0

    def close(self):
        """Restore controller tuning before a viewer reset; never reset poses."""
        self.env.model.actuator_gainprm[:] = self._original_gain
        self.env.model.actuator_biasprm[:] = self._original_bias
        self.env.model.opt.noslip_iterations = self._original_noslip
        self.grasp.close()

    def lift_evidence(self):
        e = self.env
        obj, belt = (e.model.geom(name).id for name in
                     (fc.part_geom_name(self.station), fc.BELT_GEOM))
        half_height = abs(e.data.geom_xmat[obj].reshape(3, 3)[2]) @ e.model.geom_size[obj]
        top = e.data.geom_xpos[belt, 2] + abs(e.data.geom_xmat[belt].reshape(3, 3)[2]) @ e.model.geom_size[belt]
        clearance = float(e.data.geom_xpos[obj, 2] - half_height - top)
        bodies = SharpaContactLift.hand_wrist_body_ids(self.grasp)
        forces = SharpaContactLift.measure_support_forces(self.grasp, bodies)
        supported = (all(np.linalg.norm(f) > 0.5 for f in forces.values())
                     and np.dot(forces['left'][:2], forces['right'][:2]) < 0)
        allowed = bodies['left'] | bodies['right']
        for contact in e.data.contact:
            b1, b2 = e.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if self.grasp._object_body_id in (b1, b2):
                other = b2 if b1 == self.grasp._object_body_id else b1
                if other not in allowed:
                    supported = False
        return clearance, supported

    def _standing_feedback(self):
        self.stabilizer.apply()
        self.env.data.ctrl[self.stabilizer.pitch_ids] += np.clip(
            0.3 * self.env.data.qvel[0], -0.08, 0.08)
        self.env.data.ctrl[self.stabilizer.roll_ids] -= np.clip(
            0.3 * self.env.data.qvel[1], -0.08, 0.08)

    def step(self):
        if self.state in self.TERMINAL:
            return self._last_info
        e = self.env
        action = np.zeros(25)
        if self.state == 'WAIT_FAULT':
            self.stabilizer.apply()
            if e.fault_active:
                self.station = e.fault_workcell
                e.task_manager.signal('accept', self.station, e._step_count)
                self.grasp = self._grasp_view(self.station)
                self.stabilizer.env = self.grasp
                self.expert = SharpaBimanualGraspExpert(self.grasp)
                self._transition('PREPARE_HANDS')
        elif self.state == 'PREPARE_HANDS':
            action = self.expert.step()
            self.stabilizer.apply()
            if self.expert.state.name == 'FOREARM_FORWARD_REACH':
                self.walker.holding = False
                self.walker._apply_policy_gains()
                self.navigator = PrecisionApproach(self.walker, self.config)
                self._transition('WALK')
        elif self.state == 'WALK':
            self.navigator.step(e.part_position(self.station))
            if self.navigator.arrived:
                # The last gait ctrl is NOT the measured handoff target.
                self.leg_hold = self.walker.leg_target.copy()
                roll, pitch, _, _ = self.stabilizer.tilt()
                self.stabilizer._neutral_pitch = self.leg_hold[[4, 10]] - pitch
                self.stabilizer._neutral_roll = self.leg_hold[[5, 11]] - 0.7 * roll
                self._transition('SETTLE')
        elif self.state == 'SETTLE':
            e.data.ctrl[self.walker.act_ids] = self.leg_hold
            self._standing_feedback()
            if self.ticks >= self.config.settle_steps:
                self.expert = SharpaBimanualGraspExpert(self.grasp)
                self.expert.config.palm_first_closure = self.config.palm_first_closure
                self.expert.config.hold_squeeze_m = self.config.hold_squeeze_m
                self.expert.config.contact_settle_grace_seconds = self.config.contact_settle_grace_seconds
                if self.config.motion_profile == 'compact':
                    self.expert.resume_prepared_approach()
                elif self.config.motion_profile == 'direct':
                    # From the pose walking left the arms in, straight at the
                    # block. Falls back to the full entry if the arms are not
                    # settled, and that fallback is recorded rather than hidden.
                    self.used_direct_approach = self.expert.begin_direct_approach()
                self._transition('GRASP')
        elif self.state == 'GRASP':
            action = self.expert.step()
            self._standing_feedback()

        e.task_manager.update(e, e._step_count)
        e.belt.running = e.task_manager.belt_should_run
        e.belt.drive(e)
        for arm in e.arms:
            arm.apply(e.data)
        self.grasp.step(action)
        e._step_count += 1
        for arm in e.arms:
            arm.advance()
        self.total_steps += 1
        self.ticks += 1
        info = e._get_info()
        e._update_navigation(info)
        self.forbidden_contact_ticks += int(info.get('forbidden_contact', False))
        if info.get('forbidden_contact', False):
            for index, contact in enumerate(e.data.contact):
                bodies = e.model.geom_bodyid[[contact.geom1, contact.geom2]]
                if set(bodies) & e._forbidden_body_ids:
                    pair = ':'.join([self.state, *(e.model.body(int(b)).name for b in bodies)])
                    record = self.forbidden_contact_pairs.setdefault(
                        pair, {'samples': 0, 'peak_force_n': 0.0, 'max_penetration_m': 0.0})
                    force = np.zeros(6)
                    mujoco.mj_contactForce(e.model, e.data, index, force)
                    record['samples'] += 1
                    record['peak_force_n'] = max(record['peak_force_n'], float(np.linalg.norm(force[:3])))
                    record['max_penetration_m'] = max(record['max_penetration_m'], -float(contact.dist))
        if self.state == 'GRASP':
            clearance, supported = self.lift_evidence()
            hands = SharpaContactLift.hand_wrist_body_ids(self.grasp)
            hand_ids = hands['left'] | hands['right']
            for contact in e.data.contact:
                bodies = set(e.model.geom_bodyid[[contact.geom1, contact.geom2]])
                if self.grasp._object_body_id in bodies and bodies & hand_ids:
                    self.max_object_hand_penetration_m = max(
                        self.max_object_hand_penetration_m, -float(contact.dist))
            self.max_clearance = max(self.max_clearance, clearance)
            self.hold_steps = self.hold_steps + 1 if (
                clearance >= self.config.lift_clearance_m and supported) else 0
            self.max_hold_steps = max(self.max_hold_steps, self.hold_steps)
            dt = e.model.opt.timestep * e.config.frame_skip
            if self.expert.state.name == 'FAILURE':
                self.failure = self.expert.failure_reason.name
                self._transition('FAILED')
            elif self.hold_steps * dt >= self.config.lift_hold_seconds:
                self._transition('LIFTED')
        if info.get('fallen'):
            self.failure = 'FALL'
            self._transition('FAILED')
        elif not np.isfinite(e.data.qpos).all() or not np.isfinite(e.data.qvel).all():
            self.failure = 'NONFINITE_PHYSICS'
            self._transition('FAILED')
        elif self.total_steps >= self.config.max_steps and self.state not in self.TERMINAL:
            self.failure = self.state + '_TIMEOUT'
            self._transition('FAILED')
        info.update(recovery_state=self.state, recovery_failure=self.failure,
                    recovery_station=self.station, lift_clearance_max_m=self.max_clearance,
                    lift_hold_steps=self.max_hold_steps,
                    object_hand_penetration_max_m=self.max_object_hand_penetration_m,
                    forbidden_contact_ticks=self.forbidden_contact_ticks,
                    grasp_state=(self.expert.state.name if hasattr(self, 'expert') else None))
        self._last_info = info
        return info
