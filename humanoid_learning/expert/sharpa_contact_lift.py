"""Contact-driven bilateral lift, independent of the historical tripod Gate A.

Only actuator targets are commanded. The free object is never attached or
repositioned. MuJoCo's optional no-slip *solver iterations* reduce soft-contact
creep; friction coefficients and collision geometry are not changed.
"""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import sharpa_config as sc


class SharpaContactLift:
    def __init__(self, expert):
        self.expert = expert
        self.env = expert.env
        self.dt = self.env.model.opt.timestep * self.env.config.frame_skip
        self.ticks = 0
        self.start = {s: self.env.palm_pose(s)[0].copy() for s in sc.SIDES}
        self.rotation = {s: self.env.palm_pose(s)[1].copy() for s in sc.SIDES}
        self.start_object = expert._object_pos().copy()
        self.height = 0.0
        self.table_geom = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, tc.TABLE_GEOM)
        self.object_geom = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, tc.OBJECT_GEOM)
        self.env.model.opt.noslip_iterations = expert.config.hold_noslip_iterations

    def support_forces(self):
        """Actual object forces from each whole hand, not historical touch flags."""
        result = {s: np.zeros(3) for s in sc.SIDES}
        bodies = {"left": self.env._left_hand_body_ids, "right": self.env._right_hand_body_ids}
        model, data = self.env.model, self.env.data
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if self.env._object_body_id not in (b1, b2):
                continue
            other = b2 if b1 == self.env._object_body_id else b1
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f)
            world = c.frame.reshape(3, 3).T @ f[:3]
            if b1 == self.env._object_body_id:
                world = -world
            for side in sc.SIDES:
                if other in bodies[side]:
                    result[side] += world
        return result

    def supported(self):
        f = self.support_forces()
        return (all(np.linalg.norm(f[s]) > 0.5 for s in sc.SIDES)
                and np.dot(f['left'][:2], f['right'][:2]) < 0)

    def clearance(self):
        """Signed mesh distance to the table, valid also for a tilted block."""
        return float(mujoco.mj_geomDistance(self.env.model, self.env.data,
                                           self.object_geom, self.table_geom, 1.0, None))

    def step(self, height):
        cfg = self.expert.config
        # Keep the achieved finger posture under load. Chasing a non-contacting
        # finger's force target caused the loaded ring/pinky to roll off the box.
        action = np.zeros(25)
        if self.ticks % 10 == 0:
            squeeze = cfg.hold_squeeze_m * min(self.ticks * self.dt / 2.0, 1.0)
            targets = {s: self.start[s] + np.array([
                0.0, -squeeze if s == 'left' else squeeze, height]) for s in sc.SIDES}
            result = self.expert._solve_both(
                targets, self.rotation, require_orientation=True, ori_task_weight=1.0,
                rest_q=self.expert._current_rest_q(), rest_gain=0.05,
                pos_tol=0.0005, joint_weight=self.expert.DESCEND_JOINT_WEIGHT)
            self.expert._apply_ik_result(result)
        action[:3] = self.expert._waist_action_toward_target()
        action[3:17] = self.expert._arm_action_toward_target()
        self.ticks += 1
        return action
