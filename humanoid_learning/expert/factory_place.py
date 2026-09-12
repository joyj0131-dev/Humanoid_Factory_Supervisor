"""Actuator-only place/retract prototype; never moves the free object directly.

Known limitation: seed-0 station-0 loaded TRANSFER falls before reaching the
pick pose. LOWER/RELEASE/RETRACT/restart are wired but not end-to-end validated.
This must remain opt-in until loaded transfer and balance are solved.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from humanoid_learning.envs import factory_config as fc
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import _quintic_scale, _slerp_R
from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift


@dataclass
class PlaceConfig:
    transfer_ticks: int = 600
    lower_ticks: int = 400
    settle_ticks: int = 200
    release_ticks: int = 150
    retract_ticks: int = 300
    approach_height_m: float = 0.08
    position_tolerance_m: float = 0.015


class FactoryPlace:
    def __init__(self, recovery, config=None):
        self.recovery = recovery
        self.env, self.expert = recovery.env, recovery.expert
        self.view = recovery.grasp
        self.config = config or PlaceConfig()
        self.stage, self.tick, self.failure = 'TRANSFER', 0, None
        self.target = self.env._part_rest_pos(recovery.station, fc.LOCAL_CANONICAL_PART_XY)
        self.above = self.target + [0., 0., self.config.approach_height_m]
        self.start = self.env.part_position(recovery.station).copy()
        self.start_R = self.env.data.xmat[self.view._object_body_id].reshape(3, 3).copy()
        self.offset = {}
        self.hand_R = {}
        for side in ('left', 'right'):
            p, R = self.view.palm_pose(side)
            self.offset[side] = self.start_R.T @ (p - self.start)
            self.hand_R[side] = self.start_R.T @ R
        self.unsupported_ticks = 0
        self.final_error_m = float('inf')
        self.hands_clear = False

    def _advance(self, stage):
        self.stage, self.tick = stage, 0

    def _move(self, pos, R):
        targets = {s: pos + R @ self.offset[s] for s in self.offset}
        rotations = {s: R @ self.hand_R[s] for s in self.hand_R}
        self._solve(targets, rotations)

    def _solve(self, targets, rotations):
        result = self.expert._solve_both(
            targets, rotations, require_orientation=True, ori_task_weight=1.,
            rest_q=self.expert._current_rest_q(), rest_gain=0.05, pos_tol=0.001,
            joint_weight=self.expert.DESCEND_JOINT_WEIGHT)
        self.expert._apply_ik_result(result)

    def _object_supported_by_belt(self):
        return self.view._object_body_id in self.env.belt.parts_on_belt(self.env)

    def _hands_clear(self):
        # Actual geometry distances, not commanded hand positions.
        hands = SharpaContactLift.hand_wrist_body_ids(self.view)
        hand_ids = hands['left'] | hands['right']
        object_geom = self.env.model.geom(fc.part_geom_name(self.recovery.station)).id
        arm_prefix = fc.arm_body_name(self.recovery.station, '')
        obstacles = [object_geom] + [i for i in range(self.env.model.ngeom)
                    if (self.env.model.geom(i).name or '').startswith(arm_prefix)]
        min_distance = 1.
        for geom in range(self.env.model.ngeom):
            if int(self.env.model.geom_bodyid[geom]) in hand_ids:
                for obstacle in obstacles:
                    min_distance = min(min_distance, float(mujoco.mj_geomDistance(
                        self.env.model, self.env.data, geom, obstacle, 1., None)))
        return min_distance > 0.04

    def step(self):
        cfg = self.config
        action = np.zeros(25)
        self.tick += 1
        actual = self.env.part_position(self.recovery.station)
        if self.stage == 'TRANSFER':
            f = _quintic_scale(self.tick / cfg.transfer_ticks)
            self._move((1-f)*self.start + f*self.above, _slerp_R(self.start_R, np.eye(3), f))
            self.final_error_m = float(np.linalg.norm(actual - self.above))
            if self.tick >= cfg.transfer_ticks:
                if self.final_error_m < cfg.position_tolerance_m:
                    self._advance('LOWER')
                elif self.tick >= cfg.transfer_ticks + cfg.settle_ticks:
                    self.failure = 'PLACE_TRANSFER_NOT_REACHED'
        elif self.stage == 'LOWER':
            f = _quintic_scale(self.tick / cfg.lower_ticks)
            self._move((1-f)*self.above + f*self.target, np.eye(3))
            self.final_error_m = float(np.linalg.norm(actual - self.target))
            if self.tick >= cfg.lower_ticks:
                if self.final_error_m < cfg.position_tolerance_m and self._object_supported_by_belt():
                    self.release_start = {s: self.view.palm_pose(s)[0].copy() for s in self.offset}
                    self.release_R = {s: self.view.palm_pose(s)[1].copy() for s in self.offset}
                    self._advance('RELEASE')
                elif self.tick >= cfg.lower_ticks + cfg.settle_ticks:
                    self.failure = 'PLACE_SUPPORT_NOT_ESTABLISHED'
        elif self.stage in ('RELEASE', 'RETRACT'):
            action[17:25] = -1.
            if self.stage == 'RELEASE':
                f = _quintic_scale(self.tick / cfg.release_ticks)
                delta = lambda side: np.array([0., 0.05 if side == 'left' else -0.05, 0.]) * f
            else:
                f = _quintic_scale(self.tick / cfg.retract_ticks)
                delta = lambda side: np.array([-0.08*f, 0.05 if side == 'left' else -0.05, 0.08*f])
            self._solve({s: self.release_start[s] + delta(s) for s in self.offset}, self.release_R)
            if self.stage == 'RELEASE' and self.tick >= cfg.release_ticks:
                self._advance('RETRACT')
            elif self.stage == 'RETRACT' and self.tick >= cfg.retract_ticks:
                self.hands_clear = self._hands_clear()
                if self.hands_clear and self.env.task_manager._part_is_back(self.env):
                    self._advance('READY_TO_VERIFY')
                elif self.tick >= cfg.retract_ticks + cfg.settle_ticks:
                    self.failure = 'PLACE_RETRACT_OR_SETTLE_NOT_ACHIEVED'
        if self.stage in ('TRANSFER', 'LOWER'):
            forces = SharpaContactLift.measure_support_forces(self.view)
            supported = all(np.linalg.norm(f) > 0.5 for f in forces.values())
            self.unsupported_ticks = 0 if supported or self._object_supported_by_belt() else self.unsupported_ticks + 1
            if self.unsupported_ticks * self.env.config.frame_skip * self.env.model.opt.timestep > 0.5:
                self.failure = 'PLACE_CONTACT_LOST'
        action[:3] = self.expert._waist_action_toward_target()
        action[3:17] = self.expert._arm_action_toward_target()
        return action
