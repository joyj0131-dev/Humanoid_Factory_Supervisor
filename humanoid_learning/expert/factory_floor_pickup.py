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
from humanoid_learning.envs import factory_config as fc
from humanoid_learning.expert.sharpa_pad_grasp import SharpaPadGrasp, upper_side_patch


# G1 elbow: straight at q~1.3; q > 1.3 bends it BACKWARD (limit 2.094 is ~45
# deg of hyperextension). Pulling a near-straight arm in is singular, and the
# solver was measured to resolve it into hyperextension up to the limit.
# Arm servos below add a weak preference for ordinary flexion instead.
ELBOW_FLEX_START = 1.1


def _with_elbow_flexion(jac, target, q_elbow, column=3, weight=.3):
    """Append a DLS row nudging a straightish/hyperextended elbow to flex.

    Below ELBOW_FLEX_START no row is added: a zero-rate row would instead
    resist the flexion the hand task itself asks for."""
    if q_elbow <= ELBOW_FLEX_START:
        return jac, target
    row = np.zeros(jac.shape[1])
    row[column] = weight
    step = -min(.002, .02*(q_elbow-ELBOW_FLEX_START))
    return np.vstack([jac, row]), np.r_[target, weight*step]


# Standing carry pose of each palm relative to its shoulder (see _begin_carry_rise).
# Measured shoulder-palm distance vs elbow q in this posture: .35 m at q=0
# (90 deg), .40 m at q=.5, .42 m straight. .38 m keeps the elbow comfortably
# bent (q~.3); a .29 m carry forced q<-.1, wrist self-contact and a turned block.
# Forward/down split: with .30/-.24 the held block (hanging ~.19 m along the
# fingers) sat .47 m ahead of the pelvis and touched the table face at .58 m;
# .22/-.30 keeps ~.37 m of reach and carries it ~8 cm closer to the body.
CARRY_PALM_FORWARD_M = .22
CARRY_PALM_DOWN_M = -.30
CARRY_REACH_M = float(np.hypot(CARRY_PALM_FORWARD_M, CARRY_PALM_DOWN_M))
# Minimum clearance kept between the held block's leading edge and the table.
CARRY_TABLE_MARGIN_M = .06


class FactoryFloorPickup:
    def __init__(self, recovery):
        self.recovery, self.env = recovery, recovery.grasp
        self.four_finger = getattr(getattr(recovery, 'config', None), 'floor_four_finger_grip', False)
        self.posture_iterations = getattr(getattr(recovery, 'config', None), 'floor_posture_iterations', 8)
        if not isinstance(self.posture_iterations, int) or self.posture_iterations < 1:
            raise ValueError('floor posture iterations must be a positive integer')
        # Re-plan the whole-body posture every N control ticks. Commanding,
        # contact feedback and balance feedback still run every tick; a stage
        # change or a new posture planner always re-plans immediately.
        self.posture_solve_interval = getattr(getattr(recovery, 'config', None), 'floor_posture_solve_interval', 1)
        if not isinstance(self.posture_solve_interval, int) or self.posture_solve_interval < 1:
            raise ValueError('floor posture solve interval must be a positive integer')
        self._planned_with = None  # (posture planner, stage) of the last solve
        self._plan_age = 0
        self.posture_solves = 0
        self.pad_grasp = (SharpaPadGrasp(self.env) if getattr(getattr(recovery, 'config', None), 'floor_pad_grip', False)
                          else None)
        if self.pad_grasp is not None:
            self.four_finger = True
        self.frog_stance = bool(getattr(recovery.config, 'floor_frog_stance', False))
        if self.frog_stance and self.pad_grasp is None:
            raise ValueError('frog stance requires the pad grip')
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
        self.stance_metrics = {}
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
            self.stance_knee_lateral = self.posture.knee_lateral.copy()
            self.stance_foot_rotations = [R.copy() for R in self.posture.foot_rotations]
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
        moving_support = (all(load > .15 for load in normal_loads.values())
                          if self.frog_stance else supported)
        if self.stage == 'CLOSE':
            height, pitch = self.close_height, self.close_pitch
            self.support_ticks = self.support_ticks+1 if supported else 0
            if self.support_ticks >= 30:
                self.stage, self.tick = 'PICK_CLEAR', 0
                self.pick_rotations = [self.env.palm_pose(s)[1] for s in ('left','right')]
                self.pick_object_R = self.env.data.geom_xmat[
                    self.env.model.geom(self.env.object_geom_name).id].reshape(3, 3).copy()
                self.clear_ticks = 0
                self.env.model.opt.noslip_iterations = expert.config.hold_noslip_iterations
            elif self.tick >= 800:
                self.failure = 'FLOOR_CONTACT_NOT_ESTABLISHED'
        if self.stage == 'PICK_CLEAR':
            clearance, held = self.recovery.lift_evidence()
            # Frog: no separate draw-in before rising. Measured, that 4 cm pull
            # (hands back while crouched) straightened both elbows to their
            # extension limit and rotated the block ~26 deg; the carry rise
            # brings the block in with the shoulders instead.
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
                if self.frog_stance:
                    self._begin_carry_rise()
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
            if self.stage == 'PICK_CLEAR' and moving_support:
                action[3:17] += self._pick_clear_action()
                action[3:17] = np.clip(action[3:17], -.003/self.env.config.arm_action_scale,
                                       .003/self.env.config.arm_action_scale)
        elif self.frog_stance and self.stage in ('RISE', 'HOLD'):
            action = self._carry_rise_action(height, pitch, f, normal_loads)
        else:
            locked = (self.env.data.qpos[self.posture.qadr[12:]].copy()
                      if self.stage in ('RISE', 'HOLD') else None)
            spread = 0.
            if self.frog_stance:
                # Clear the hips with the proven reach first. Opening the
                # knees from the start blocks that reach with the thighs.
                if self.stage == 'LOWER':
                    spread = .06*_quintic_scale(np.clip((progress-.75)/.25, 0., 1.))
                self.posture.knee_lateral = self.stance_knee_lateral.copy()
                self.posture.foot_rotations = [R.copy() for R in self.stance_foot_rotations]
            planned = self._planned_with
            if (planned is None or planned[0] is not self.posture or planned[1] != self.stage
                    or self._plan_age >= self.posture_solve_interval):
                # command() reads static_torque and the planned pelvis frame
                # from this same solve, so a reused plan stays self-consistent.
                self.last_q = self.posture.solve(height, palms, rotations, pitch, iterations=self.posture_iterations,
                                                locked_upper_q=locked, knee_outward_m=spread)
                self._planned_with, self._plan_age = (self.posture, self.stage), 0
                self.posture_solves += 1
            self._plan_age += 1
            action = self.posture.command(self.last_q)
            if self.stage == 'RISE' and self.tick == 0:
                self.rise_leg_target_jump_rad = float(np.max(np.abs(
                    self.env.data.ctrl[self.posture.act[:12]]-self.rise_leg_ctrl)))
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
        lateral = expert.task_rotation[:, 1]
        forward = expert.task_rotation[:, 0]
        d = self.env.data
        sample = {
            'knee_width_m': float((d.xpos[self.posture.knees[0]]-d.xpos[self.posture.knees[1]])@lateral),
            'object_forward_from_pelvis_m': float((self.recovery.env.part_position(self.recovery.station)-
                                                  d.xpos[self.posture.pelvis])@forward),
        }
        row = self.stance_metrics.setdefault(self.stage, {})
        for key, value in sample.items():
            row[key+'_min'] = min(row.get(key+'_min', value), value)
            row[key+'_max'] = max(row.get(key+'_max', value), value)
            row[key+'_last'] = value
        return action

    def _begin_carry_rise(self):
        """Anchor the rise on the measured pose: commanded hands, CoM, knees.

        Every RISE target starts at its measured value, so the first plan does
        not jump away from the live body (the old targets put the planned
        pelvis 9.3 deg off the live one and kicked both ankles by .17 rad).
        """
        m, d, p = self.env.model, self.env.data, self.posture
        self.carry_palm0 = self._commanded_palms()
        # Hand orientation is carried in the TORSO frame: holding it fixed in
        # the world while the torso straightens drove both wrists into their
        # own link geometry (roll/yaw self-contact) before the grip was lost.
        self.carry_torso = m.body('torso_link').id
        torso_R = d.xmat[self.carry_torso].reshape(3, 3)
        self.carry_R_torso = [torso_R.T @ self._carry_fk.site_xmat[sid].reshape(3, 3) for sid in p.palms]
        self.carry_shoulders = [m.body(f'{side}_shoulder_pitch_link').id for side in ('left', 'right')]
        R = p.base_rotation
        # Hand targets live in the frame of the LIVE shoulders: measured here,
        # both elbows are at their extension limit (shoulder-palm .36 m = full
        # reach), so a world- or pelvis-fixed hand path leaves reach as soon as
        # the torso straightens. Standing, hold the palms CARRY_PALM_* from the
        # shoulders (elbows bent): the block rides in front of the body and is
        # drawn in as the shoulders move back.
        self.carry_v0 = [R.T @ (self.carry_palm0[i] - d.xpos[b]) for i, b in enumerate(self.carry_shoulders)]
        self.carry_v_stand = [np.array([CARRY_PALM_FORWARD_M, v[1], CARRY_PALM_DOWN_M]) for v in self.carry_v0]
        mid = .5*(self.carry_v0[0][1] + self.carry_v0[1][1])
        for v in self.carry_v_stand:
            v[1] -= mid  # centre the pair on the body without changing its width
        self.carry_squeeze = 0.
        self.carry_forward_trim = 0.
        # The line's own table: the held block must stay clear of its face.
        station = self.recovery.station
        self.carry_obstacle = m.geom(fc.work_surface_geom_name(self.recovery.env.poses[station])).id
        self.rise_com0 = d.subtree_com[p.pelvis, :2].copy()
        self.rise_com_stand = p.com_xy.copy()
        self.rise_knee0 = np.array(p.knee_lateral)
        # Standing knee WIDTH as before the crouch, re-centred on the feet as
        # they are now: they slide ~1-2 cm while crouched, and the stale centre
        # was measured to leave the standing pelvis rolled ~5 deg.
        lateral = p.base_rotation[:, 1]
        feet_centre = float(np.mean([pos @ lateral for pos in p.foot_positions]))
        stance = np.array(self.stance_knee_lateral)
        self.rise_knee_stand = stance + (feet_centre - stance.mean())

    def _block_table_gap(self):
        """Clearance (m) from the block's leading edge to the table face,
        along the robot's forward axis."""
        m, d = self.env.model, self.env.data
        forward = self.posture.base_rotation[:, 0]
        obj = m.geom(self.env.object_geom_name).id
        lead = d.geom_xpos[obj] @ forward + np.abs(d.geom_xmat[obj].reshape(3, 3).T @ forward) @ m.geom_size[obj]
        t = self.carry_obstacle
        face = d.geom_xpos[t] @ forward - np.abs(d.geom_xmat[t].reshape(3, 3).T @ forward) @ m.geom_size[t]
        return float(face - lead)

    def _commanded_palms(self):
        """Palm positions the current arm/waist TARGETS produce on the live body."""
        m, d, e = self.env.model, self.env.data, self.env
        if not hasattr(self, '_carry_fk'):
            self._carry_fk = mujoco.MjData(m)
        s = self._carry_fk
        s.qpos[:] = d.qpos
        s.qpos[e._arm_qpos_adr] = e._arm_target
        s.qpos[e._waist_qpos_adr] = e._waist_target
        mujoco.mj_kinematics(m, s)
        mujoco.mj_comPos(m, s)
        return np.array([s.site_xpos[sid].copy() for sid in self.posture.palms])

    def _carry_rise_action(self, height, pitch, f, normal_loads):
        """Legs/torso from the whole-body plan; hands carried with the shoulders.

        The plan owns balance and height with the upper body at its present
        configuration. Arms are servoed in the LIVE frame: a floating-base plan
        is ~1 cm off the real base, more than a pad squeeze.
        """
        p, d = self.posture, self.env.data
        # Squeeze is common to both hands: opposing pad forces on a free block
        # are equal, so a per-hand loop only shoves the block sideways.
        load = min(normal_loads.values())
        if load < self.load_low:
            self.carry_squeeze = min(self.carry_squeeze + .0001, .01)
        elif load > self.load_high:
            self.carry_squeeze = max(self.carry_squeeze - .0001, -.01)
        R = p.base_rotation
        expert = self.recovery.expert
        # Where the robot ended up standing sets how close the carried block
        # comes to the table: from a start offset by 5-10 cm it touched the
        # face and the grip rolled onto the pinkies. Draw the carry pose in
        # while the block's leading edge is within CARRY_TABLE_MARGIN_M.
        if self._block_table_gap() < CARRY_TABLE_MARGIN_M:
            self.carry_forward_trim = max(self.carry_forward_trim - .0002, -CARRY_PALM_FORWARD_M + .08)
        forward = CARRY_PALM_FORWARD_M + self.carry_forward_trim
        down = -np.sqrt(CARRY_REACH_M**2 - forward**2)
        stand = [np.array([forward, v[1], down]) for v in self.carry_v_stand]
        palms = [d.xpos[b] + R @ ((1-f)*self.carry_v0[i] + f*stand[i])
                 + self.carry_squeeze*expert._inward_direction(side)
                 for i, (b, side) in enumerate(zip(self.carry_shoulders, ('left', 'right')))]
        self.carry_target_block = .5*(palms[0] + palms[1])
        p.com_xy = (1-f)*self.rise_com0 + f*self.rise_com_stand
        knees = (1-f)*self.rise_knee0 + f*self.rise_knee_stand
        locked = d.qpos[p.qadr[12:]].copy()
        self.last_q = p.solve(height, np.array(palms), None, pitch, iterations=self.posture_iterations,
                              locked_upper_q=locked, knee_lateral_targets=knees)
        self.posture_solves += 1
        action = p.command(self.last_q)
        if self.stage == 'RISE' and self.tick == 0:
            self.rise_leg_target_jump_rad = float(np.max(np.abs(
                d.ctrl[p.act[:12]]-self.rise_leg_ctrl)))
        action[:3] = 0.  # the loaded waist target is kept
        action[3:17] = self._carry_arm_action(palms)
        return action

    def _carry_arm_action(self, palms):
        """Resolved-rate step moving the COMMANDED hands toward the targets."""
        m, e = self.env.model, self.env
        commanded = self._commanded_palms()
        s = self._carry_fk
        action = np.zeros(14)
        weights = np.array([1., .2, .3, 1., .5, 1., .5])  # shoulder-averse, as the squeeze servo
        for i, sid in enumerate(self.posture.palms):
            error = palms[i] - commanded[i]
            distance = float(np.linalg.norm(error))
            step = error if distance <= .001 else error*(.001/distance)
            reference = self.env.data.xmat[self.carry_torso].reshape(3, 3) @ self.carry_R_torso[i]
            rotation_error = orientation_error(s.site_xmat[sid].reshape(3, 3), reference)
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, s, jp, jr, sid)
            cols = e._arm_dof_adr[7*i:7*i+7]
            jac, target = _with_elbow_flexion(np.vstack([jp[:, cols], .1*jr[:, cols]]),
                                              np.r_[step, .1*np.clip(.1*rotation_error, -.003, .003)],
                                              e._arm_target[7*i+3])
            jac, target = self._with_torso_clearance(jac, target, i, cols)
            weighted = jac*weights
            dq = weights*(weighted.T @ np.linalg.solve(weighted@weighted.T+1e-4*np.eye(len(target)), target))
            action[7*i:7*i+7] = np.clip(dq, -.003, .003)/e.config.arm_action_scale
        return action

    def _with_torso_clearance(self, jac, target, side_index, cols, rate=.0005):
        """Rows pushing this arm's links off the torso/pelvis along live contacts.

        The whole-body posture IK keeps self-clearance, but the carry arm servo
        is separate; without this the right shoulder was measured pressing the
        torso at up to 232 N while the hands followed their path."""
        m, d = self.env.model, self.env.data
        if not hasattr(self, '_arm_link_bodies'):
            self._arm_link_bodies = [
                {m.body(n).id for n in (f'{side}_shoulder_pitch_link', f'{side}_shoulder_roll_link',
                                         f'{side}_shoulder_yaw_link', f'{side}_elbow_link')}
                for side in ('left', 'right')]
            self._trunk_bodies = {m.body('torso_link').id, m.body('pelvis').id}
        mine = self._arm_link_bodies[side_index]
        rows, values = [], []
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 in mine and b2 in self._trunk_bodies:
                arm, away = b1, -c.frame[:3]
            elif b2 in mine and b1 in self._trunk_bodies:
                arm, away = b2, c.frame[:3].copy()
            else:
                continue
            jp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, None, c.pos, arm)
            rows.append(away @ jp[:, cols])
            values.append(rate)
        if not rows:
            return jac, target
        return np.vstack([jac, *rows]), np.r_[target, values]

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

    def _pick_clear_action(self, translation=None, preserve_rotation=True):
        """Lift clear with arms before rising; pad grip retracts from the table."""
        m, d = self.env.model, self.env.data
        action = np.zeros(14)
        # With long exposed pads, advancing toward the table traps the thumb
        # and then the index finger on its vertical side. Clear upward/back
        # toward the worker instead; preserve the legacy hook-grip trajectory.
        forward = -.00004 if self.pad_grasp is not None else .00004
        if translation is None:
            translation = self.recovery.expert.task_rotation[:,0]*forward + np.array([0.,0.,.00008])
        for i, side in enumerate(('left','right')):
            sid = self.posture.palms[i]
            step = translation
            if self.frog_stance and self.stage == 'PICK_CLEAR':
                # Lift along the palm->shoulder line. A straight-up world lift
                # was measured to extend both elbows to their limit (a reach
                # singularity) before the rise; this path bends them instead,
                # lifting the block and bringing it slightly toward the body.
                toward = d.xpos[m.body(f'{side}_shoulder_pitch_link').id] - d.site_xpos[sid]
                step = .0001*toward/np.linalg.norm(toward)
            jp, jr = np.zeros((3,m.nv)), np.zeros((3,m.nv))
            mujoco.mj_jacSite(m,d,jp,jr,sid)
            cols = self.env._arm_dof_adr[7*i:7*i+7]
            jac = np.vstack([jp[:,cols], .2*jr[:,cols]])
            reference = self.pick_rotations[i]
            if self.frog_stance:
                object_R = d.geom_xmat[m.geom(self.env.object_geom_name).id].reshape(3, 3)
                reference = object_R @ self.pick_object_R.T @ reference
            rotation_error = (orientation_error(self.env.palm_pose(side)[1], reference)
                              if preserve_rotation else np.zeros(3))
            delta = np.r_[step, .2*np.clip(.05*rotation_error,-.002,.002)]
            # Use the same joint preference as the squeeze servo: summing an
            # unrestricted lift with a shoulder-averse squeeze defeats the
            # latter and can drive shoulder roll back into the torso.
            weights = (np.ones(7) if self.pad_grasp is None else
                       np.array([1., .2, .3, 1., .5, 1., .5]))
            if self.frog_stance and self.stage == 'PICK_CLEAR':
                jac, delta = _with_elbow_flexion(jac, delta, self.env._arm_target[7*i+3])
            weighted = jac*weights
            dq = weights*(weighted.T @ np.linalg.solve(
                weighted@weighted.T+1e-4*np.eye(len(delta)), delta))
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
            contact_load = .15 if self.frog_stance else 1.5
            if load > contact_load or (not self.side_entry_ready[side]
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
            elif self.stage in ('PICK_CLEAR','RISE','HOLD') or (
                    self.frog_stance and self.stage == 'CLOSE' and load > contact_load):
                # Once rubber touches, stop chasing the tangential patch.
                # That position servo pushed the 0.1 kg block toward the
                # table before bilateral pressure was established. This is a
                # controller switch only; support/success thresholds stay put.
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
            if self.frog_stance and self.stage in ('PICK_CLEAR', 'RISE', 'HOLD'):
                # Track the two measured frames relative to the same live
                # object, not two independently frozen world orientations.
                reference = object_R @ self.pick_object_R.T @ self.pick_rotations[i]
            elif self.pad_grasp is not None and self.stage in ('RISE', 'HOLD'):
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
