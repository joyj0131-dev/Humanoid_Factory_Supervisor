"""Foot-anchored quasi-static posture IK. Only joint targets reach live physics.

The floating-base coordinates are optimization variables on scratch MjData,
never commands. Gravity, floor contact and the actuated legs move the real base.
"""
from __future__ import annotations

import copy
import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from humanoid_learning.envs import whole_body_config as wc, task_config as tc
from humanoid_learning.expert.pose_ik import orientation_error, so3_exp


class WholeBodyPosture:
    def __init__(self, env, clearance_m=0.0, self_collision_only=True):
        self.env = env
        m, d = env.model, env.data
        self.names = wc.LEG_JOINTS + wc.WAIST_JOINTS + tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS
        self.jids = np.array([m.joint(n).id for n in self.names])
        self.qadr = m.jnt_qposadr[self.jids]
        self.dadr = m.jnt_dofadr[self.jids]
        self.act = np.array([m.actuator(n).id for n in self.names])
        base = m.joint(wc.FLOATING_BASE_JOINT).id
        self.base_qadr = int(m.jnt_qposadr[base])
        self.cols = np.r_[np.arange(m.jnt_dofadr[base], m.jnt_dofadr[base] + 6), self.dadr]
        self.pelvis = m.body(wc.PELVIS_BODY).id
        self.feet = [m.site(n).id for n in (wc.LEFT_FOOT_SITE, wc.RIGHT_FOOT_SITE)]
        self.palms = [m.site(n).id for n in (wc.LEFT_PALM_SITE, wc.RIGHT_PALM_SITE)]
        self.foot_positions = [d.site_xpos[s].copy() for s in self.feet]
        self.foot_rotations = [d.site_xmat[s].reshape(3, 3).copy() for s in self.feet]
        # A gait handoff can leave one sole tilted on its edge. That measured
        # orientation is NOT a suitable fixed constraint for a deep squat.
        for i, sid in enumerate(self.feet):
            old = self.foot_rotations[i]
            yaw = np.arctan2(old[1, 0], old[0, 0])
            level = so3_exp(np.array([0., 0., yaw]))
            body = m.site_bodyid[sid]
            heights = []
            for gid in range(m.ngeom):
                if m.geom_bodyid[gid] == body and m.geom_contype[gid] and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    offset = old.T @ (d.geom_xpos[gid] - d.site_xpos[sid])
                    heights.append(m.geom_size[gid, 0] - (level @ offset)[2])
            if heights:
                self.foot_positions[i][2] = max(heights)
            self.foot_rotations[i] = level
        self.start_height = float(d.xpos[self.pelvis, 2])
        old_base = d.xmat[self.pelvis].reshape(3, 3)
        yaw = np.arctan2(old_base[1, 0], old_base[0, 0])
        self.base_rotation = so3_exp(np.array([0., 0., yaw]))
        self.com_xy = np.mean(self.foot_positions, axis=0)[:2] + .025*self.base_rotation[:2, 0]
        self.rest = d.qpos[self.qadr].copy()
        # Planning-only clearance: detect self proximity before penetration.
        # A separate model preserves ALL live collision/physical parameters.
        self.clearance_m = float(clearance_m)
        self.ik_model = copy.copy(m) if clearance_m > 0. or self_collision_only else m
        robot = m.body_rootid[m.geom_bodyid] == self.pelvis
        if clearance_m > 0.:
            self.ik_model.geom_margin[robot] = np.maximum(m.geom_margin[robot], clearance_m)
        if self_collision_only:
            # The objective below uses robot self contacts ONLY. Avoid narrow
            # phase work for conveyor/parts/floor that would be discarded.
            # This is a private planning model; live contacts remain unchanged.
            self.ik_model.geom_contype[~robot] = 0
            self.ik_model.geom_conaffinity[~robot] = 0
            active = robot & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))
            if np.all(m.geom_contype[active] == 1) and np.all(m.geom_conaffinity[active] == 1):
                # Finger joints are not optimization variables here. Contacts
                # wholly within one hand have zero Jacobian in these columns:
                # they cannot be relieved by any leg/waist/arm/base motion.
                # Keep hand-to-arm, hand-to-body and opposite-hand contacts.
                self.ik_model.geom_contype[active] = 1
                self.ik_model.geom_conaffinity[active] = 7
                for side, bit, affinity in (('left', 2, 5), ('right', 4, 3)):
                    ids = getattr(env, f'_{side}_hand_body_ids')
                    hand = active & np.isin(m.geom_bodyid, list(ids))
                    self.ik_model.geom_contype[hand] = bit
                    self.ik_model.geom_conaffinity[hand] = affinity
        self.geom_body = m.geom_bodyid.copy()
        self.geom_is_robot = robot.copy()
        self.scratch = mujoco.MjData(self.ik_model)
        self.solution = None
        self.last_error = {}
        self.last_command = d.qpos[self.qadr].copy()
        self.command_velocity = np.zeros(len(self.names))

    def solve(self, height, palms, rotations=None, pelvis_pitch=0.0, iterations=12,
              locked_upper_q=None):
        m, live, d = self.ik_model, self.env.data, self.scratch
        d.qpos[:] = live.qpos
        if self.solution is not None:
            adr = self.base_qadr
            d.qpos[adr:adr+7] = self.solution[adr:adr+7]
            d.qpos[self.qadr] = self.solution[self.qadr]
        else:
            # Straight knees are a singular branch. Seed a small positive bend
            # on scratch only; the live targets still ramp from measured pose.
            for hip, knee, ankle in ((0,3,4),(6,9,10)):
                if d.qpos[self.qadr[knee]] < .1:
                    d.qpos[self.qadr[hip]] -= .1
                    d.qpos[self.qadr[knee]] += .2
                    d.qpos[self.qadr[ankle]] -= .1
        n = len(self.cols)
        if locked_upper_q is not None:
            d.qpos[self.qadr[12:]] = locked_upper_q
        desired_pelvis_R = self.base_rotation @ so3_exp(np.array([0., pelvis_pitch, 0.]))
        for _ in range(iterations):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rows, errors = [], []
            def add(j, err, weight):
                rows.append(weight * j[:, self.cols])
                errors.append(weight * np.atleast_1d(err))
            for sid, pos, rotation in zip(self.feet, self.foot_positions, self.foot_rotations):
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jacSite(m, d, jp, jr, sid)
                add(jp, pos - d.site_xpos[sid], 20.)
                add(jr, orientation_error(d.site_xmat[sid].reshape(3, 3), rotation), 3.)
            for i, (sid, pos) in enumerate(zip(self.palms, palms)):
                if locked_upper_q is not None:
                    continue  # carry the held arm posture with the torso
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jacSite(m, d, jp, jr, sid)
                add(jp, pos - d.site_xpos[sid], 5.)
                if rotations is not None:
                    add(jr, orientation_error(d.site_xmat[sid].reshape(3, 3), rotations[i]), .5)
            jc = np.zeros((3, m.nv))
            mujoco.mj_jacSubtreeCom(m, d, jc, self.pelvis)
            add(jc[:2], self.com_xy - d.subtree_com[self.pelvis, :2], 10.)
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jr, self.pelvis)
            add(jp[2:3], [height - d.xpos[self.pelvis, 2]], 3.)
            add(jr, orientation_error(d.xmat[self.pelvis].reshape(3, 3), desired_pelvis_R), .5)
            mujoco.mj_collision(m, d)
            for contact in d.contact:
                g1, g2 = contact.geom1, contact.geom2
                # Foot support is intended; all robot self contact is not.
                if not (self.geom_is_robot[g1] and self.geom_is_robot[g2]):
                    continue
                clearance_error = max(.003, self.clearance_m) - contact.dist
                if clearance_error <= 0.:
                    continue
                b1, b2 = self.geom_body[g1], self.geom_body[g2]
                jp1, jp2 = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp1, None, contact.pos, int(b1))
                mujoco.mj_jac(m, d, jp2, None, contact.pos, int(b2))
                normal = contact.frame.reshape(3, 3)[0]
                add((normal @ (jp2 - jp1))[None, :], [clearance_error], 30.)
            # Weak posture regularization prevents arbitrary knee/arm branches.
            regularizer = np.zeros((len(self.names), n))
            regularizer[:, 6:] = np.eye(len(self.names)) * .015
            rows.append(regularizer)
            errors.append(.015 * (self.rest - d.qpos[self.qadr]))
            j, error = np.vstack(rows), np.concatenate(errors)
            low = np.full(n, -.04)
            high = np.full(n, .04)
            joint_low = m.jnt_range[self.jids, 0].copy() + .005
            joint_low[[3, 9]] = .03
            low[6:] = np.maximum(low[6:], joint_low - d.qpos[self.qadr])
            high[6:] = np.minimum(high[6:], m.jnt_range[self.jids, 1] - .005 - d.qpos[self.qadr])
            if locked_upper_q is not None:
                low[18:], high[18:] = -1e-10, 1e-10
            # Solve WITH bounds: clipping an unconstrained step afterwards
            # invalidates the foot/CoM solution and can drive the base away.
            delta = lsq_linear(np.vstack([j, np.sqrt(.002)*np.eye(n)]),
                               np.r_[error, np.zeros(n)], bounds=(low, high),
                               method='bvls', tol=1e-6, max_iter=50).x
            velocity = np.zeros(m.nv)
            velocity[self.cols] = delta
            mujoco.mj_integratePos(m, d.qpos, velocity, 1.)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr], m.jnt_range[self.jids, 0] + .005,
                                      m.jnt_range[self.jids, 1] - .005)
            # Stay on the forward-bending knee branch through full extension.
            # The asset permits slight hyperextension, which is not a squat.
            d.qpos[self.qadr[[3, 9]]] = np.maximum(d.qpos[self.qadr[[3, 9]]], .03)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        self.last_error = {'palm_m': [float(np.linalg.norm(p - d.site_xpos[s]))
                                     for p, s in zip(palms, self.palms)],
                           'com_m': float(np.linalg.norm(d.subtree_com[self.pelvis, :2] - self.com_xy)),
                           'height_m': float(d.xpos[self.pelvis, 2])}
        self.solution = d.qpos.copy()
        # Contact-consistent static feedforward. For a standing robot the legs
        # support the torso; bare qfrc_bias treats them as hanging from a free
        # pelvis and omits the floor reaction needed at the knees/hips.
        mujoco.mj_comVel(m, d)
        bias = np.zeros(m.nv)
        mujoco.mj_rne(m, d, 0, bias)
        contact_jacobians = []
        for sid in self.feet:
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, sid)
            contact_jacobians.append(np.vstack([jp, jr]))
        contact_j = np.vstack(contact_jacobians)
        scale = np.tile([1., 1., 1., .04, .04, .01], 2)
        base_balance = contact_j[:, self.cols[:6]].T * scale
        wrench = scale * np.linalg.lstsq(base_balance, bias[self.cols[:6]], rcond=None)[0]
        self.static_torque = bias[self.dadr] - contact_j[:, self.dadr].T @ wrench
        return d.qpos[self.qadr].copy()

    def command(self, q):
        """Set leg targets and return the ordinary 25D manipulation action."""
        e, m, d = self.env, self.env.model, self.env.data
        # Collision-active IK can switch branches abruptly even along a smooth
        # Cartesian path. Bound acceleration as well as velocity at the actual
        # actuator command boundary, especially for loaded knees and ankles.
        dt = m.opt.timestep * e.config.frame_skip
        error = q - self.last_command
        requested = np.sign(error) * np.minimum(.6, np.sqrt(2. * np.abs(error)))
        self.command_velocity += np.clip(requested-self.command_velocity, -dt, dt)
        q = np.clip(self.last_command + dt*self.command_velocity,
                    m.jnt_range[self.jids, 0]+.005, m.jnt_range[self.jids, 1]-.005)
        self.last_command = q.copy()
        gains = m.actuator_gainprm[self.act[:12], 0]
        gravity = self.static_torque[:12] / np.maximum(gains, 1.)
        commanded = q[:12] + gravity
        rotation = d.xmat[self.pelvis].reshape(3, 3)
        planned = self.scratch.xmat[self.pelvis].reshape(3, 3)
        tilt = rotation.T @ orientation_error(planned, rotation)
        angular = d.qvel[self.cols[3:6]]
        velocity = rotation.T @ d.qvel[self.cols[:3]]
        com_error = rotation.T @ np.r_[d.subtree_com[self.pelvis, :2] - self.com_xy, 0.]
        commanded[[4, 10]] += np.clip(tilt[1] + .1*angular[1] + .3*velocity[0] + 1.5*com_error[0], -.25, .25)
        commanded[[5, 11]] += np.clip(.7*tilt[0] + .07*angular[0] - .3*velocity[1] - 1.5*com_error[1], -.15, .15)
        d.ctrl[self.act[:12]] = np.clip(commanded,
                                       m.actuator_ctrlrange[self.act[:12], 0],
                                       m.actuator_ctrlrange[self.act[:12], 1])
        action = np.zeros(25)
        action[:3] = np.clip((q[12:15] - e._waist_target) / e.config.waist_action_scale, -1., 1.)
        action[3:17] = np.clip((q[15:] - e._arm_target) / e.config.arm_action_scale, -1., 1.)
        return action
