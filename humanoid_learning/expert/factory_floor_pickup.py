"""Experimental foot-supported floor pickup; no object/base pose writes."""
import numpy as np
import mujoco
from humanoid_learning.expert.whole_body_posture import WholeBodyPosture
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    _quintic_scale, _slerp_R, _wahba_R, LOCAL_APPROACH_VEC, LOCAL_CLOSING_VEC,
)
from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift
from humanoid_learning.expert.pose_ik import orientation_error
from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.expert.sharpa_pad_grasp import SharpaPadGrasp, upper_side_patch


class FactoryFloorPickup:
    def __init__(self, recovery):
        self.recovery, self.env = recovery, recovery.grasp
        self.four_finger = getattr(getattr(recovery, 'config', None), 'floor_four_finger_grip', False)
        self.posture_iterations = getattr(getattr(recovery, 'config', None), 'floor_posture_iterations', 8)
        if not isinstance(self.posture_iterations, int) or self.posture_iterations < 1:
            raise ValueError('floor posture iterations must be a positive integer')
        self.pad_grasp = (SharpaPadGrasp(self.env) if getattr(getattr(recovery, 'config', None), 'floor_pad_grip', False)
                          else None)
        if self.pad_grasp is not None:
            self.four_finger = True
        self.load_low, self.load_high = getattr(getattr(recovery, 'config', None), 'floor_grip_load_n', (3., 6.))
        if not 0. < self.load_low < self.load_high:
            raise ValueError('floor grip load band must be positive and ordered')
        self.finger_bodies = {side: {bid for bid in getattr(self.env, f'_{side}_hand_body_ids')
            if any(f'_{finger}_' in (self.env.model.body(bid).name or '')
                   for finger in ('index', 'middle', 'ring', 'pinky'))}
            for side in sc.SIDES}
        self.thumb_bodies = {side: {bid for bid in getattr(self.env, f'_{side}_hand_body_ids')
            if '_thumb_' in (self.env.model.body(bid).name or '')} for side in sc.SIDES}
        self.thumb_peak_force_n = {side: 0. for side in sc.SIDES}
        self.env.config.per_actuator_gravity_compensation = True
        self.ik_clearance_m = .008
        self.posture = WholeBodyPosture(self.env, self.ik_clearance_m)
        self.standing_height = self.posture.start_height
        self.stage, self.tick, self.failure = 'CROUCH', 0, None
        self.height = .45
        self.pitch = .40
        self.lower_start_height = self.posture.start_height
        self.lower_start_pitch = 0.
        self.start = np.array([self.env.palm_pose(s)[0] for s in ('left', 'right')])
        self.start_R = [self.env.palm_pose(s)[1] for s in ('left', 'right')]
        self.obj = recovery.env.part_position(recovery.station).copy()
        expert = recovery.expert
        # Floor-specific grasp: fingers descend around the side faces while
        # the wrists stay higher. The belt's horizontal wrap put the wrists
        # directly in the crouched knees' path.
        targets = expert._mirrored_targets(-.03, .18, .13)
        self.goal = np.array([targets[s] for s in ('left', 'right')])
        self.goal_R = [_wahba_R([LOCAL_APPROACH_VEC, LOCAL_CLOSING_VEC],
                               [np.array([0., 0., -1.]), expert._inward_direction(s)], [1., 1.])
                       for s in ('left', 'right')]
        if self.pad_grasp is not None:
            self.goal, self.goal_R = self.pad_grasp.reach_targets(expert)
        self.support_ticks = 0
        self.side_entry_ready = {'left': False, 'right': False}
        self.peak_hand_force_n = {'left': 0., 'right': 0.}
        self.target_error_m = float('inf')
        self.last_q = None
        self.close_height = self.height
        self.close_pitch = self.pitch
        for side in ('left', 'right'):
            self.env.set_preshape(side, 1.)

    def step(self):
        if self.stage == 'CROUCH' and self.tick == 0:
            # The factory constructs this controller before the final SETTLE
            # physics tick. Anchor on the actual first controlled tick, not on
            # that stale handoff pose (nor on a separately restored checkpoint).
            # Synchronize derived sites/CoM with the integrated qpos, just as
            # a checkpoint restore does. This changes no pose or velocity.
            mujoco.mj_forward(self.env.model, self.env.data)
            self.posture = WholeBodyPosture(self.env, self.ik_clearance_m)
            self.standing_height = self.posture.start_height
            self.lower_start_height = self.posture.start_height
            self.start = np.array([self.env.palm_pose(s)[0] for s in ('left', 'right')])
            self.start_R = [self.env.palm_pose(s)[1] for s in ('left', 'right')]
        self.tick += 1
        expert = self.recovery.expert
        progress = _quintic_scale(min(self.tick / 1400., 1.)) if self.stage in ('CROUCH','LOWER') else 1.
        height = self.lower_start_height*(1-progress) + self.height*progress
        pitch = self.lower_start_pitch*(1-progress)+self.pitch*progress
        if self.stage == 'CROUCH':
            palms = self.start + np.array([0., 0., height-self.lower_start_height])
            rotations = self.start_R
        else:
            palms = self.start * (1-progress) + self.goal * progress
            rotations = [_slerp_R(a, b, progress) for a, b in zip(self.start_R, self.goal_R)]
        forces = (self.pad_grasp.forces() if self.pad_grasp is not None else
                  SharpaContactLift.measure_support_forces(
                      self.env, self.finger_bodies if self.four_finger else None))
        thumb_forces = SharpaContactLift.measure_support_forces(self.env, self.thumb_bodies)
        for side in sc.SIDES:
            self.thumb_peak_force_n[side] = max(self.thumb_peak_force_n[side], float(np.linalg.norm(thumb_forces[side])))
        for side, force in forces.items():
            self.peak_hand_force_n[side] = max(self.peak_hand_force_n[side], float(np.linalg.norm(force)))
        normal_loads = {side: float(np.dot(force, -self._side_face(side)[2]))
                        for side, force in forces.items()}
        supported = (all(load>1.5 for load in normal_loads.values())
                     and np.dot(forces['left'][:2], forces['right'][:2]) < 0.)
        if self.stage == 'CLOSE':
            height, pitch = self.close_height, self.close_pitch
            self.support_ticks = self.support_ticks+1 if supported else 0
            if self.support_ticks >= 30:
                self.stage, self.tick = 'PICK_CLEAR', 0
                self.pick_rotations = [self.env.palm_pose(s)[1] for s in ('left','right')]
                self.clear_ticks = 0
                self.env.model.opt.noslip_iterations = expert.config.hold_noslip_iterations
            elif self.tick >= 800:
                self.failure = 'FLOOR_CONTACT_NOT_ESTABLISHED'
        if self.stage == 'PICK_CLEAR':
            clearance, held = self.recovery.lift_evidence()
            self.clear_ticks = self.clear_ticks+1 if clearance >= .03 and held else 0
            if self.clear_ticks >= 30:
                self.stage, self.tick = 'RISE', 0
                self.lift_start = np.array([self.env.palm_pose(s)[0] for s in ('left', 'right')])
                self.lift_R = [self.env.palm_pose(s)[1] for s in ('left', 'right')]
                self.rise_base_height = float(self.env.data.xpos[self.posture.pelvis, 2])
                self.rise_pitch = float(self.recovery.stabilizer.tilt()[1])
                self.rise_contact_gap = 0
                self.rise_leg_ctrl = self.env.data.ctrl[self.posture.act[:12]].copy()
                self.posture = WholeBodyPosture(self.env, self.ik_clearance_m)
            elif self.tick >= 1200:
                self.failure = 'FLOOR_INITIAL_LIFT_NOT_ACHIEVED'
        if self.stage in ('RISE', 'HOLD'):
            f = _quintic_scale(min(self.tick/1600., 1.)) if self.stage == 'RISE' else 1.
            height = self.rise_base_height + f*(self.standing_height-self.rise_base_height)
            pitch = self.rise_pitch*(1-f)
            palms = self.lift_start + np.array([0., 0., .55*f])
            rotations = self.lift_R
            self.rise_contact_gap = 0 if supported else self.rise_contact_gap+1
            if self.tick >= 200 and self.rise_contact_gap > 50:
                self.failure = 'FLOOR_LIFT_CONTACT_LOST'
            if self.stage == 'RISE' and self.tick >= 1600:
                if (abs(self.env.data.xpos[self.posture.pelvis, 2]-self.standing_height) < .02
                        and abs(self.recovery.stabilizer.tilt()[1]) < .10):
                    self.stage, self.tick = 'HOLD', 0
                elif self.tick >= 2600:
                    self.failure = 'FLOOR_STAND_NOT_ACHIEVED'
        if self.stage in ('CLOSE', 'PICK_CLEAR'):
            action = self._contact_action(forces)
            if self.stage == 'PICK_CLEAR' and supported:
                action[3:17] += self._pick_clear_action()
                action[3:17] = np.clip(action[3:17], -.003/self.env.config.arm_action_scale,
                                       .003/self.env.config.arm_action_scale)
        else:
            locked = (self.env.data.qpos[self.posture.qadr[12:]].copy()
                      if self.stage in ('RISE', 'HOLD') else None)
            self.last_q = self.posture.solve(height, palms, rotations, pitch, iterations=self.posture_iterations,
                                            locked_upper_q=locked)
            action = self.posture.command(self.last_q)
            if self.stage == 'RISE' and self.tick == 0:
                self.rise_leg_target_jump_rad = float(np.max(np.abs(
                    self.env.data.ctrl[self.posture.act[:12]] - self.rise_leg_ctrl)))
            if self.stage in ('RISE', 'HOLD'):
                # Preserve the loaded arm targets; only contact feedback may
                # adjust them. Re-solving their world pose released the grip.
                action[3:17] = self._contact_action(forces, hold_legs=False)[3:17]
                action[:3] = 0.  # preserve the loaded waist target too
        synergy = (.8 - .55*_quintic_scale(np.clip((progress-.6)/.4, 0., 1.))
                   if self.stage == 'LOWER' and self.pad_grasp is None else .8)
        if self.stage not in ('RISE', 'HOLD'):
            action[17:25] = np.clip((synergy - self.env._group_synergy)/self.env.config.hand_synergy_action_scale, -.1, .1)
        if self.four_finger and self.stage != 'CROUCH':
            # Keep the proven compact crouch. Legacy grip opens the thumb on
            # the reach; the pad clamp parks it inside to clear the table wall.
            opening = (_quintic_scale(np.clip((progress-.6)/.4, 0., 1.))
                       if self.stage == 'LOWER' else 1.)
            if self.pad_grasp is not None:
                self.pad_grasp.apply_shape(opening)
            parked_thumb = .7 if self.pad_grasp is not None else 0.
            for i in (0, 4):
                action[17+i] = np.clip((.8*(1-opening)+parked_thumb*opening-self.env._group_synergy[i])/self.env.config.hand_synergy_action_scale, -.1, .1)
            for side in sc.SIDES:
                for suffix in sc.preshape_suffixes('thumb'):
                    aid = self.env.model.actuator(sc.sharpa_actuator(side, 'thumb', suffix)).id
                    thumb_preshape = 1. if self.pad_grasp is not None else 1-opening
                    self.env.data.ctrl[aid] = np.clip(thumb_preshape*sc.PRESHAPE_TARGETS['thumb'][suffix],
                                                    *self.env.model.actuator_ctrlrange[aid])
        self.target_error_m = max(float(np.linalg.norm(self.env.palm_pose(s)[0]-palms[i]))
                                  for i, s in enumerate(('left','right')))
        if self.stage == 'CROUCH' and self.tick >= 1400:
            if abs(self.env.data.xpos[self.posture.pelvis,2]-self.height)<.03:
                self.stage, self.tick = 'LOWER', 0
                self.start = np.array([self.env.palm_pose(s)[0] for s in ('left','right')])
                self.start_R = [self.env.palm_pose(s)[1] for s in ('left','right')]
                self.lower_start_height = float(self.env.data.xpos[self.posture.pelvis,2])
                self.lower_start_pitch = self.pitch
                # Start the reach from the actual settled crouch, not from a
                # standing-pose null-space seed or an obsolete scratch base.
                self.posture = WholeBodyPosture(self.env, self.ik_clearance_m)
            elif self.tick >= 2200:
                self.failure = 'FLOOR_CROUCH_NOT_ACHIEVED'
        if self.stage == 'LOWER' and self.tick >= 1400:
            # Cartesian residual is diagnostic, not proof that a grip is
            # impossible. Try closure plus inward adjustment from the actual
            # pose; only measured contact/lift can establish grasp success.
            self.begin_contact()
        return action

    def begin_contact(self):
        self.stage, self.tick = 'CLOSE', 0
        d = self.env.data
        self.close_height = float(d.xpos[self.posture.pelvis, 2])
        self.close_pitch = float(self.recovery.stabilizer.tilt()[1])
        self.close_leg_ctrl = d.ctrl[self.posture.act[:12]].copy()
        self.close_R = d.xmat[self.posture.pelvis].reshape(3, 3).copy()
        self.close_com = d.subtree_com[self.posture.pelvis].copy()
        self.close_hand_rotations = [self.env.palm_pose(s)[1] for s in ('left', 'right')]
        self.last_q = d.qpos[self.posture.qadr].copy()

    def _pick_clear_action(self):
        """Lift clear with arms before rising; pad grip retracts from the table."""
        m, d = self.env.model, self.env.data
        action = np.zeros(14)
        # With long exposed pads, advancing toward the table traps the thumb
        # and then the index finger on its vertical side. Clear upward/back
        # toward the worker instead; preserve the legacy hook-grip trajectory.
        forward = -.00004 if self.pad_grasp is not None else .00004
        translation = self.recovery.expert.task_rotation[:,0]*forward + np.array([0.,0.,.00008])
        for i, side in enumerate(('left','right')):
            sid = self.posture.palms[i]
            jp, jr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
            mujoco.mj_jacSite(m,d,jp,jr,sid)
            cols = self.env._arm_dof_adr[7*i:7*i+7]
            jac = np.vstack([jp[:,cols], .2*jr[:,cols]])
            rotation_error = orientation_error(self.env.palm_pose(side)[1], self.pick_rotations[i])
            delta = np.r_[translation, .2*np.clip(.05*rotation_error,-.002,.002)]
            # Use the same joint preference as the squeeze servo: summing an
            # unrestricted lift with a shoulder-averse squeeze defeats the
            # latter and can drive shoulder roll back into the torso.
            weights = (np.ones(7) if self.pad_grasp is None else
                       np.array([1., .2, .3, 1., .5, 1., .5]))
            weighted = jac*weights
            dq = weights*(weighted.T @ np.linalg.solve(
                weighted@weighted.T+1e-4*np.eye(6), delta))
            action[7*i:7*i+7] = dq/self.env.config.arm_action_scale
        return action

    def _contact_action(self, forces, hold_legs=True):
        """Resolved-rate surface servo, integrated into arm position targets.

        Hold the supported crouch while permitting wrist/arm adjustment. No
        world-base command, object force or pose substitution is used.
        """
        m, d = self.env.model, self.env.data
        action = np.zeros(25)
        rotation = d.xmat[self.posture.pelvis].reshape(3, 3)
        tilt = rotation.T @ orientation_error(self.close_R, rotation)
        angular = d.qvel[self.posture.cols[3:6]]
        com = rotation.T @ (d.subtree_com[self.posture.pelvis]-self.close_com)
        legs = self.close_leg_ctrl.copy()
        legs[[4,10]] += np.clip(tilt[1]+.1*angular[1]+1.5*com[0], -.2, .2)
        legs[[5,11]] += np.clip(.7*tilt[0]+.07*angular[0]-1.5*com[1], -.15, .15)
        if hold_legs:
            d.ctrl[self.posture.act[:12]] = np.clip(legs, m.actuator_ctrlrange[self.posture.act[:12],0],
                                                   m.actuator_ctrlrange[self.posture.act[:12],1])
        obj = m.geom(self.env.object_geom_name).id
        object_R = d.geom_xmat[obj].reshape(3,3)
        half = m.geom_size[obj]
        for i, side in enumerate(('left', 'right')):
            axis, sign, normal = self._side_face(side)
            load = float(np.dot(forces[side], -normal))
            sid = self.env._fingertip_site[(side, 'middle')]
            point = (self.pad_grasp.point(side, 'middle') if self.pad_grasp is not None
                     else d.site_xpos[sid].copy())
            local_point = object_R.T @ (point-d.geom_xpos[obj])
            patch = np.clip(local_point, -.7*half, .7*half)
            # A dropped cube can rest on ANY face. In this factory rollout
            # local X, not local Z, is vertical. Treating local Z as height
            # drove the pads forward along the table rather than down its side.
            if self.pad_grasp is not None:
                patch, up_axis = upper_side_patch(object_R, half, local_point)
                tangent = [j for j in range(3) if j not in (up_axis, axis)]
                if len(tangent) == 1:
                    j = tangent[0]
                    # Choose the forward strip of the side face explicitly:
                    # wrists must pass in front of, not through, crouched knees.
                    patch[j] = np.sign(object_R[:, j] @ self.recovery.expert.task_rotation[:, 0])*.5*half[j]
            else:
                up_axis = 2
                patch[2] = .5*half[2]
            if load > 1.5 or (not self.side_entry_ready[side]
                    and abs(local_point[up_axis]-patch[up_axis]) < .005
                    and sign*local_point[axis] > half[axis]+.012):
                self.side_entry_ready[side] = True
            entering = not self.side_entry_ready[side]
            patch[axis] = sign*(half[axis]+.02 if entering else half[axis]-.001)
            in_band = not entering and self.load_low <= load <= self.load_high
            if in_band:
                continue
            # Opposite side-face patches, not a shared nearest top corner.
            # Back off an overloaded hand while the other establishes contact.
            if in_band:
                difference = np.zeros(3)
            elif self.stage in ('PICK_CLEAR','RISE','HOLD'):
                difference = normal*(.0002 if load > self.load_high else -.0002)
            else:
                difference = (normal*.0005 if load > self.load_high
                              else d.geom_xpos[obj]+object_R@patch-point)
            distance = np.linalg.norm(difference)
            direction = (difference/distance if distance > 1e-8
                         else self.recovery.expert._inward_direction(side))
            travel = min(.001, distance)
            jp, jr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
            if self.pad_grasp is None:
                mujoco.mj_jacSite(m,d,jp,jr,sid)
            else:
                mujoco.mj_jac(m, d, jp, jr, point, self.pad_grasp.bodies[side, 'middle'])
            columns = self.env._arm_dof_adr[7*i:7*i+7]
            # A point task alone can rotate the wrist around that point and
            # turn a side grasp into pressure on the upper edge of the block.
            # Keep the measured hand frame while translating toward the face.
            reference = self.close_hand_rotations[i]
            if self.pad_grasp is not None and self.stage in ('RISE', 'HOLD'):
                # Preserve the loaded arm targets while the torso rises.
                # Pressure corrections minimize instantaneous wrist rotation;
                # do not chase a stored world/body pose through the chest.
                reference = self.env.palm_pose(side)[1]
            rotation_error = orientation_error(self.env.palm_pose(side)[1], reference)
            jac = np.vstack([jp[:, columns], .1*jr[:, columns]])
            target = np.r_[travel*direction, .1*np.clip(.1*rotation_error, -.003, .003)]
            weights = np.ones(7)
            if self.pad_grasp is not None:
                # Prefer elbow/forward reach adjustments to adducting the
                # shoulder into the chest. These are soft costs, not locks.
                weights = np.array([1., .2, .3, 1., .5, 1., .5])
            weighted = jac*weights
            dq = weights*(weighted.T @ np.linalg.solve(weighted@weighted.T+1e-4*np.eye(6),target))
            action[3+7*i:10+7*i] = np.clip(dq,-.003,.003)/self.env.config.arm_action_scale
        return action

    def _side_face(self, side):
        m, d = self.env.model, self.env.data
        obj = m.geom(self.env.object_geom_name).id
        rotation = d.geom_xmat[obj].reshape(3,3)
        outward = -self.recovery.expert._inward_direction(side)
        candidates = rotation if self.pad_grasp is not None else rotation[:, :2]
        axis = int(np.argmax(np.abs(candidates.T @ outward)))
        sign = 1. if np.dot(rotation[:,axis],outward) > 0. else -1.
        return axis, sign, sign*rotation[:,axis]

    def geometry_report(self):
        """Actual collision-surface distances, not palm-site proxy distances."""
        m, d = self.env.model, self.env.data
        obj = m.geom(self.env.object_geom_name).id
        hands = SharpaContactLift.hand_wrist_body_ids(self.env)
        report = {}
        for side, bodies in hands.items():
            nearest = None
            for gid in range(m.ngeom):
                if m.geom_bodyid[gid] not in bodies or not (m.geom_contype[gid] or m.geom_conaffinity[gid]):
                    continue
                segment = np.zeros(6)
                distance = mujoco.mj_geomDistance(m, d, gid, obj, 1., segment)
                if nearest is None or distance < nearest['distance_m']:
                    nearest = {'distance_m': float(distance), 'geom': m.geom(gid).name, 'geom_id': gid,
                               'hand_point': segment[:3].tolist(), 'object_point': segment[3:].tolist()}
            report[side] = {'palm': self.env.palm_pose(side)[0].tolist(),
                            'nearest_surface': nearest}
            if self.pad_grasp is not None:
                inward = -self._side_face(side)[2]
                report[side]['pads'] = {
                    finger: {
                        'face_angle_deg': float(np.degrees(np.arccos(np.clip(
                            self.pad_grasp.normal(side, finger) @ inward, -1., 1.)))),
                        'surface_distance_m': float(mujoco.mj_geomDistance(
                            m, d, self.pad_grasp.geoms[side, finger], obj, 1., None)),
                        'point': self.pad_grasp.point(side, finger).tolist(),
                        'curl_joint_degrees': {suffix: float(np.degrees(d.qpos[
                            m.joint(sc.sharpa_joint(side, finger, suffix)).qposadr[0]]))
                            for suffix in sc.curl_suffixes(finger)},
                    } for finger in ('index', 'middle', 'ring', 'pinky')}
        report['object'] = d.geom_xpos[obj].tolist()
        return report
