"""Line 2 floor pickup -> table place in one stance (experimental, opt-in).

Status (measured, seed 0): this does NOT yet complete. The observe/select/
verify loop (recovery_agent) now refuses to start the pick, with the cause,
because no registered pad pinch links the floor grasp to the table: the
crouched reach needs the fingers >= ~55 deg below horizontal, while carrying
the block over the table's front edge from any single stance needs <= ~50 deg
(the hands must sit behind, not above, a block held high near the body).
Before that transfer check existed, four executed rises lost the grip where
the arm joints reached their limits.

Why no walking: with this pad pinch the hands hang ~.17 m beyond the palms, so
any held block keeps both arms forward (CoM ~.06-.09 m ahead of the pelvis).
Measured with the block dropped, the pretrained gait then did not track: a
constant .3 m/s command for 1.5 s moved ~0 m, then ~.15 m of UNcommanded drift
at 12 deg pelvis pitch, never reaching a still double support. With the arms
down it walked normally. (Carrying an actual block was not tested.)

Sequence (all through actuator targets; the block is never written):
  CROUCH        inherited frog crouch; then the pick DECISION (recovery_agent)
  LOWER         the selected entry path (possibly via e39d7ba's reach pose)
  CLOSE         pads close onto the two side faces nearest the robot's lateral
  ALIGN         both hands turn the still-supported block onto the table axes
  PICK_CLEAR    lift a few cm while crouched
  RISE/HOLD     transfer path: up short of the table face, then over the spot
  DOWN          lower until the table carries the load
  RELEASE       pads back off sideways only after the table has the load
  RETRACT/STAND hands up and back, torso upright, arms home
  BACKOFF       on a monitored failure: open, replay the entry path backward,
                re-observe and re-decide (budgeted); ABORT holds still.
"""
import mujoco
import numpy as np

from humanoid_learning.envs import factory_config as fc, sharpa_config as sc
from humanoid_learning.expert.factory_floor_pickup import FactoryFloorPickup, _with_elbow_flexion
from humanoid_learning.expert.pose_ik import orientation_error, so3_exp
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import _quintic_scale, _slerp_R, _wahba_R
from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift
from humanoid_learning.expert.whole_body_posture import WholeBodyPosture
from humanoid_learning.expert.recovery_agent import (
    CandidateEvaluator, DecisionLog, GraspPlan, RuleSelector, candidate_plans, observe)

# Pinch geometry in the block frame [forward, left, up] (see _grip_offsets).
# Straight fingers: a curled finger tilts its pad ~35 deg toward the palm and
# the two palms then converge onto each other. Distal pads land first with the
# fingertips angled 12 deg into the face (toe-in) so the proximal shells stand
# off. The finger angle and pad position along the face are NOT fixed here:
# they are part of the GraspPlan the selector picks for the observed scene.
PINCH_TOE_IN_DEG = 16.   # 12 left the middle-phalanx shells pressing the face's top edge
PINCH_HEIGHT_M = .01     # pad above the block centre: the block hangs below the pinch
# Pick-point decisions (initial + replans after a backed-off attempt).
MAX_PICK_DECISIONS = 3
PROVEN_TICKS = 1400  # e39d7ba LOWER duration
LOCAL_TICKS = 700    # proven reach pose -> this plan's entry pose


# The plan that completes the seed-0 Line 2 cycle end to end (fault -> walk ->
# floor pad pick -> rise -> over the table edge -> place -> retract -> verified
# restart). RecoveryConfig.floor_table_plan=None uses it; 'evaluate' runs the
# candidate evaluator instead.
DEFAULT_TABLE_PLAN = dict(
    phi_deg=70., forward_m=0., crouch_pitch=1.2, crouch_height=.5, knee_spread_m=.1, entry_gap_m=.05,
    via_proven=True, carry_height=.78, carry_pitch=0., carry_com_m=0., rise_block_z=.45,
    cross_clearance_m=.05, place_height=.77, place_pitch=.5, place_com_m=.08, place_backoff_m=.03)


def _yaw_R(angle):
    return so3_exp(np.array([0., 0., angle]))


class FloorTableCycle(FactoryFloorPickup):
    owns_legs = True

    def __init__(self, recovery):
        super().__init__(recovery)
        if self.pad_grasp is None or not self.frog_stance:
            raise ValueError('floor table place requires the pad grip and frog stance')
        # e39d7ba's reach pose (computed by the parent with its own finger shape).
        self.proven_goal = (self.goal.copy(), [R.copy() for R in self.goal_R])
        self.pad_grasp.extend_fingers()
        self.squeeze_rate, self.squeeze_cap = .0001, .025
        # The rubber pads were modelled with sliding friction only (condim 3),
        # so a pinched block pivoted freely about the pad axis. A pad is a
        # patch contact: give the elastomer geoms torsional friction.
        m = self.env.model
        for gid in self.pad_grasp.geoms.values():
            m.geom_condim[gid] = 4
            m.geom_friction[gid, 1] = .01
        from humanoid_learning.envs.sharpa_grasp_env import _ARM_JOINTS
        self._arm_jids = np.array([self.env.model.joint(n).id for n in _ARM_JOINTS])
        self.plan = GraspPlan()
        self.agent_log = DecisionLog()
        self.selector = RuleSelector()
        self.failed_plans = set()
        self.pick_decisions = 0
        self.squeeze = {'left': -self.plan.entry_gap_m, 'right': -self.plan.entry_gap_m}
        self.offsets = self._grip_offsets()
        self.goal, self.goal_R = self.entry_targets(self.plan)
        self.touched = {'left': False, 'right': False}
        self.normal_loads = {'left': 0., 'right': 0.}
        self.final_error_m = float('inf')
        self.commanded_rotation_deg = 0.
        self.tilt_deg = 0.
        self.hands_clear = False
        self.table_supported_ticks = 0
        self.metrics = {'align_rotation_deg': None, 'table_force_peak_n': 0.,
                        'max_block_tilt_deg_held': 0.}

    # ----- geometry -------------------------------------------------------
    def _block(self):
        m, d = self.env.model, self.env.data
        obj = m.geom(self.env.object_geom_name).id
        return d.geom_xpos[obj].copy(), d.geom_xmat[obj].reshape(3, 3).copy(), m.geom_size[obj].copy()

    def _block_frame(self, R):
        """[forward, left, up] of the block's own axes nearest the robot's."""
        heading = self.recovery.expert.task_rotation
        up_axis = int(np.argmax(np.abs(R[2])))
        rest = [j for j in range(3) if j != up_axis]
        lat_axis = max(rest, key=lambda j: abs(R[:, j] @ heading[:, 1]))
        fwd_axis = [j for j in rest if j != lat_axis][0]
        lat = R[:, lat_axis]*np.sign(R[:, lat_axis] @ heading[:, 1])
        fwd = R[:, fwd_axis]*np.sign(R[:, fwd_axis] @ heading[:, 0])
        lat[2] = fwd[2] = 0.
        lat /= np.linalg.norm(lat)
        fwd = fwd - (fwd @ lat)*lat
        fwd /= np.linalg.norm(fwd)
        return np.column_stack([fwd, lat, np.cross(fwd, lat)])

    def _grip_offsets(self, plan=None):
        """Palm pose in the block frame for each side, from the pad geometry of
        the extended hand (scratch FK, live base)."""
        plan = self.plan if plan is None else plan
        m, e = self.env.model, self.env
        s = mujoco.MjData(m)
        s.qpos[:] = e.data.qpos
        for side in sc.SIDES:
            for group in range(1, 4):
                s.qpos[e._group_qpos_adr[side][group]] = self.pad_grasp.pad_close[side][group]
        mujoco.mj_kinematics(m, s)
        half = self._block()[2][1]
        phi, alpha = np.radians(plan.phi_deg), np.radians(PINCH_TOE_IN_DEG)
        offsets = {}
        for side, sgn in (('left', 1.), ('right', -1.)):
            palm = e._left_palm_site if side == 'left' else e._right_palm_site
            pR = s.site_xmat[palm].reshape(3, 3)
            b = self.pad_grasp.bodies[side, 'middle']
            ln = pR.T @ self.pad_grasp.normal(side, 'middle', s)
            ll = pR.T @ s.xmat[b].reshape(3, 3)[:, 0]
            lp = pR.T @ (self.pad_grasp.point(side, 'middle', s) - s.site_xpos[palm])
            inward = np.array([0., -sgn, 0.])
            longi = np.array([np.cos(phi), 0., -np.sin(phi)])
            longi2 = np.cos(alpha)*longi + np.sin(alpha)*inward
            normal2 = np.cos(alpha)*inward - np.sin(alpha)*longi
            R = _wahba_R([ln, ll], [normal2, longi2], [1., 1.])
            pad = np.array([plan.forward_m, sgn*half, PINCH_HEIGHT_M])
            offsets[side] = (pad - R @ lp, R, inward)
        return offsets

    def _palm_targets(self, centre, frame, squeeze=None, offsets=None):
        squeeze = self.squeeze if squeeze is None else squeeze
        offsets = self.offsets if offsets is None else offsets
        palms, rotations = [], []
        for side in ('left', 'right'):
            offset, R, inward = offsets[side]
            palms.append(centre + frame @ (offset + inward*squeeze[side]))
            rotations.append(frame @ R)
        return np.array(palms), rotations

    def entry_targets(self, plan):
        """Entry palms for ``plan`` on the block AS IT IS NOW (re-read each call)."""
        centre, R, _ = self._block()
        gap = {'left': -plan.entry_gap_m, 'right': -plan.entry_gap_m}
        return self._palm_targets(centre, self._block_frame(R), gap, self._grip_offsets(plan))

    def aligned_frame(self, frame=None):
        """Block frame turned onto the nearest quarter turn of the (world-aligned) table."""
        if frame is None:
            frame = self._block_frame(self._block()[1])
        yaw = float(np.arctan2(frame[1, 0], frame[0, 0]))
        target = np.round(yaw/(np.pi/2))*(np.pi/2)
        return _yaw_R(float(np.arctan2(np.sin(target-yaw), np.cos(target-yaw)))) @ frame

    def hand_shape_qpos(self, qpos):
        """Write the executed pinch hand shape (straight fingers, parked
        thumb as _finger_and_thumb_commands drives it) into a qpos COPY."""
        e, m = self.env, self.env.model
        for side in sc.SIDES:
            for group in range(1, 4):
                qpos[e._group_qpos_adr[side][group]] = self.pad_grasp.pad_close[side][group]
            qpos[e._group_qpos_adr[side][0]] = (.3*self.pad_grasp.pad_open[side][0]
                                                + .7*self.pad_grasp.pad_close[side][0])
            for suffix in sc.preshape_suffixes('thumb'):
                j = m.joint(sc.sharpa_joint(side, 'thumb', suffix))
                qpos[j.qposadr[0]] = np.clip(sc.PRESHAPE_TARGETS['thumb'][suffix], *m.jnt_range[j.id])
        return qpos

    def place_target_for(self, plan, heading=None):
        forward = (self.posture.base_rotation if heading is None else heading)[:, 0]
        target = self.recovery.env._part_rest_pos(self.recovery.station, fc.LOCAL_CANONICAL_PART_XY).copy()
        target[:2] -= plan.place_backoff_m*forward[:2]
        return target

    def place_target_estimate(self):
        """Place spot and planted-feet frame, known before the pick: the feet
        do not move between pick and place in this cycle."""
        p = self.posture
        forward = p.base_rotation[:, 0]
        target = self.recovery.env._part_rest_pos(self.recovery.station, fc.LOCAL_CANONICAL_PART_XY).copy()
        target[:2] -= getattr(self.recovery.config, 'floor_table_place_backoff_m', .045)*forward[:2]
        return target, np.mean(p.foot_positions, axis=0), p.base_rotation

    # ----- decisions ------------------------------------------------------
    def _decide_pick(self, replan, cause=None):
        """Observe, evaluate the registered pick plans, select, configure LOWER."""
        obs = observe(self)
        if replan:
            self.agent_log.close('FAILED', cause, obs)
        self.pick_decisions += 1
        target, feet_mid, heading = self.place_target_estimate()
        evaluator = CandidateEvaluator(self)
        fixed = getattr(self.recovery.config, 'floor_table_plan', None)
        if fixed is None:
            fixed = DEFAULT_TABLE_PLAN
        if fixed != 'evaluate' and not replan:
            from humanoid_learning.expert.recovery_agent import Decision
            reports = []
            decision = Decision('EXECUTE', GraspPlan(**fixed), 'fixed plan (RecoveryConfig.floor_table_plan or the default)')
        elif self.pick_decisions > MAX_PICK_DECISIONS:
            reports, decision = [], None
            from humanoid_learning.expert.recovery_agent import Decision
            decision = Decision('ABORT', None, f'pick decision budget ({MAX_PICK_DECISIONS}) spent; last cause {cause}')
        else:
            reports = evaluator.evaluate(candidate_plans(), target, feet_mid, heading, excluded=self.failed_plans)
            decision = self.selector.select(obs, reports, self.agent_log.records)
        record = self.agent_log.add('PICK', obs, reports, decision, replan)
        record['path_reference'] = getattr(evaluator, 'reference', None)
        if decision.kind != 'EXECUTE':
            self.stage, self.tick = 'ABORT', 0
            self.abort_reason = 'PICK_' + decision.reason
            return
        plan = decision.plan
        self.plan = plan
        self.offsets = self._grip_offsets(plan)
        self.squeeze = {'left': -plan.entry_gap_m, 'right': -plan.entry_gap_m}
        self.touched = {'left': False, 'right': False}
        self.goal, self.goal_R = self.entry_targets(plan)
        self.pitch, self.height = plan.crouch_pitch, plan.crouch_height
        self.lower_knee_spread_m = plan.knee_spread_m
        self.entry_block = self._block()
        self.entry_bad_ticks = 0

    def _block_disturbed(self, pos_tol=.015, yaw_tol_deg=6.):
        c0, R0, _ = self.entry_block
        c, R, _ = self._block()
        f0, f = self._block_frame(R0), self._block_frame(R)
        yaw = abs(np.degrees(np.arctan2(f0[0, 0]*f[1, 0]-f0[1, 0]*f[0, 0], f0[:, 0] @ f[:, 0])))
        moved = float(np.linalg.norm((c-c0)[:2]))
        if moved > pos_tol or yaw > yaw_tol_deg:
            return f'BLOCK_DISTURBED(moved {moved*1000:.0f} mm, turned {yaw:.1f} deg)'
        return None

    def _hand_leg_contact(self, peak_n=15., sustained_n=3., sustained_ticks=30):
        """Hand pushing a leg, not brushing it. Calibrated on e39d7ba's own
        LOWER from the same crouch: a right wrist-knee brush of up to 2.4 N
        over 4 ticks is part of that physically successful reach."""
        m, d = self.env.model, self.env.data
        hands = SharpaContactLift.hand_wrist_body_ids(self.env)
        hand_ids = hands['left'] | hands['right']
        worst, pair = 0., None
        for i, c in enumerate(d.contact[:d.ncon]):
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            other = b2 if b1 in hand_ids else b1 if b2 in hand_ids else None
            if other is None or other in hand_ids:
                continue
            if any(k in (m.body(other).name or '') for k in ('hip', 'knee', 'ankle')):
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                if abs(f[0]) > worst:
                    worst, pair = abs(f[0]), f'{m.body(b1).name}|{m.body(b2).name}'
        self.hand_leg_ticks = getattr(self, 'hand_leg_ticks', 0)+1 if worst > sustained_n else 0
        self.metrics['hand_leg_peak_n'] = max(self.metrics.get('hand_leg_peak_n', 0.), worst)
        if worst > peak_n or self.hand_leg_ticks > sustained_ticks:
            return f'HAND_LEG_CONTACT({pair} {worst:.0f} N, {self.hand_leg_ticks} ticks)'
        return None

    # ----- low-level servos ----------------------------------------------
    def _servo_arms(self, palms, rotations, max_step=None, max_rot=.004, reference_q=None):
        """Resolved-rate step of the COMMANDED palms (live base, arm targets).

        Local rates alone followed the rise into both wrist-pitch limits
        (margin 1.1 -> .03 rad) and the right pad lost the block. Joints near
        a limit are pushed inward, and when the whole-body plan's arm
        configuration is given (it respects limits and self-clearance) the
        arms drift toward that branch in the hand task's null space."""
        m, e = self.env.model, self.env
        max_step = getattr(self, 'arm_step_m', .001) if max_step is None else max_step
        commanded = self._commanded_palms()
        s = self._carry_fk
        action = np.zeros(14)
        weights = np.array([1., .2, .3, 1., .5, 1., .5])
        for i, sid in enumerate(self.posture.palms):
            error = palms[i] - commanded[i]
            distance = float(np.linalg.norm(error))
            step = error if distance <= max_step else error*(max_step/distance)
            rotation = orientation_error(s.site_xmat[sid].reshape(3, 3), rotations[i])
            angle = float(np.linalg.norm(rotation))
            if angle > max_rot:
                rotation *= max_rot/angle
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, s, jp, jr, sid)
            cols = e._arm_dof_adr[7*i:7*i+7]
            jac, target = _with_elbow_flexion(np.vstack([jp[:, cols], .3*jr[:, cols]]),
                                              np.r_[step, .3*rotation], e._arm_target[7*i+3])
            jac, target = self._with_torso_clearance(jac, target, i, cols,
                                                     rate=self.__dict__.get('torso_clear_rate', .0005))
            q = e._arm_target[7*i:7*i+7]
            low, high = m.jnt_range[self._arm_jids[7*i:7*i+7]].T
            rows, values = [], []
            for j in range(7):
                if q[j] - low[j] < .15:
                    rows.append(np.eye(7)[j]); values.append(.002)
                elif high[j] - q[j] < .15:
                    rows.append(np.eye(7)[j]); values.append(-.002)
            if rows:
                jac, target = np.vstack([jac, *rows]), np.r_[target, values]
            weighted = jac*weights
            A = weighted@weighted.T+1e-4*np.eye(len(target))
            dq = weights*(weighted.T @ np.linalg.solve(A, target))
            if reference_q is not None:
                pull = np.clip(self.__dict__.get('arm_pull_gain', .02)*(reference_q[7*i:7*i+7] - q), -.004, .004)
                null = np.eye(7) - weights[:, None]*(weighted.T @ np.linalg.solve(A, jac))  # J @ null ~ 0
                dq = dq + null @ pull
            action[7*i:7*i+7] = np.clip(dq, -self.__dict__.get('arm_dq_max', .004), self.__dict__.get('arm_dq_max', .004))/e.config.arm_action_scale
        return action

    def _hold_crouch_legs(self):
        """Crouched legs as closed, with the same tilt/CoM feedback as CLOSE."""
        m, d = self.env.model, self.env.data
        rotation = d.xmat[self.posture.pelvis].reshape(3, 3)
        tilt = rotation.T @ orientation_error(self.close_R, rotation)
        angular = d.qvel[self.posture.cols[3:6]]
        com = rotation.T @ (d.subtree_com[self.posture.pelvis]-self.close_com)
        legs = self.close_leg_ctrl.copy()
        legs[[4, 10]] += np.clip(tilt[1]+.1*angular[1]+1.5*com[0], -.2, .2)
        legs[[5, 11]] += np.clip(.7*tilt[0]+.07*angular[0]-1.5*com[1], -.15, .15)
        d.ctrl[self.posture.act[:12]] = np.clip(legs, m.actuator_ctrlrange[self.posture.act[:12], 0],
                                               m.actuator_ctrlrange[self.posture.act[:12], 1])

    def _measure_loads(self, frame):
        forces = self.pad_grasp.forces()
        self.normal_loads = {side: float(abs(forces[side] @ frame[:, 1])) for side in forces}
        return self.normal_loads

    def _squeeze_servo(self):
        """Common squeeze once both pads touch: equal and opposite on a free block."""
        load = min(self.normal_loads.values())
        rate = 0.
        if load < self.load_low:
            rate = self.squeeze_rate
        elif load > self.load_high:
            rate = -self.squeeze_rate
        for side in self.squeeze:
            # Measured at CLOSE: all eight pads loaded but the arms' compliance
            # leaves the palms 5-7 mm short of the command; a 12 mm cap held
            # the pinch at 2.7-3.0 N, just under the band.
            self.squeeze[side] = float(np.clip(self.squeeze[side] + rate, -.01, self.squeeze_cap))

    def _table(self, heading=None):
        m, d = self.env.model, self.env.data
        t = m.geom(fc.work_surface_geom_name(self.recovery.env.poses[self.recovery.station])).id
        R = d.geom_xmat[t].reshape(3, 3)
        top = d.geom_xpos[t, 2] + np.abs(R[2]) @ m.geom_size[t]
        forward = (self.heading if heading is None else heading)[:, 0]
        face = d.geom_xpos[t] @ forward - np.abs(R.T @ forward) @ m.geom_size[t]
        return t, float(top), float(face)

    def _table_force(self):
        m, d = self.env.model, self.env.data
        t = self._table()[0]
        obj = m.geom(self.env.object_geom_name).id
        total = 0.
        for i, c in enumerate(d.contact[:d.ncon]):
            if {c.geom1, c.geom2} == {t, obj}:
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                total += f[0]
        return float(total)

    def _hands_clear(self):
        m, d = self.env.model, self.env.data
        hands = SharpaContactLift.hand_wrist_body_ids(self.env)
        hand_ids = hands['left'] | hands['right']
        station = self.recovery.station
        obstacles = [m.geom(fc.part_geom_name(station)).id] + [
            i for i in range(m.ngeom) if (m.geom(i).name or '').startswith(fc.arm_body_name(station, ''))]
        nearest = 1.
        for geom in range(m.ngeom):
            if int(m.geom_bodyid[geom]) in hand_ids and (m.geom_contype[geom] or m.geom_conaffinity[geom]):
                for obstacle in obstacles:
                    nearest = min(nearest, float(mujoco.mj_geomDistance(m, d, geom, obstacle, 1., None)))
        return nearest > .04

    # ----- stages ---------------------------------------------------------
    def step(self):
        if self.stage == 'CROUCH' and self.tick == 0:
            self.home_arm_q = self.env._arm_target.copy()
        if self.stage == 'CROUCH':
            action = super().step()
            if self.stage == 'LOWER':
                self._decide_pick(replan=False)
            return action
        if self.stage == 'LOWER':
            action = self._lower_step()
            if self.stage == 'LOWER':
                cause = self._block_disturbed() or self._hand_leg_contact()
                error = max(self.posture.last_error.get('palm_m', [0.]))
                self.entry_bad_ticks = self.entry_bad_ticks+1 if error > .02 else 0
                if cause is None and self.entry_bad_ticks >= 25:
                    cause = f'ENTRY_TRACKING(plan palm error {error*1000:.0f} mm)'
                # A plan that changes arm branch mid-path drags the real arm
                # through configurations nobody checked (measured: a 1.2 rad
                # swing swept the left hand into the block ~50 ticks later).
                history = self.__dict__.setdefault('arm_plan_history', [])
                history.append(self.last_q[15:29].copy())
                if len(history) > 20:
                    history.pop(0)
                    swing = float(np.abs(history[-1] - history[0]).max())
                    if cause is None and swing > .5:
                        cause = f'PLAN_BRANCH_SWITCH(arm joint swing {swing:.2f} rad in 20 ticks)'
                if cause:
                    self._begin_backoff(cause)
            if self.stage == 'CLOSE':
                self._begin_close()
            return action
        if self.stage in ('BACKOFF', 'ABORT'):
            return self._backoff_step()
        self.tick += 1
        centre, R, half = self._block()
        frame = self._block_frame(R)
        loads = self._measure_loads(frame)
        action = np.zeros(25)
        if self.stage in ('CLOSE', 'ALIGN', 'PICK_CLEAR'):
            self._hold_crouch_legs()
            if self.stage == 'CLOSE':
                for side in self.squeeze:
                    if not self.touched[side]:
                        self.touched[side] = loads[side] > .15
                        self.squeeze[side] = min(self.squeeze[side] + .0005, 0.)
                if all(self.touched.values()):
                    self._squeeze_servo()
                    # Pinch against the block pose at first bilateral touch.
                    # Re-reading the live pose let the hands chase a block
                    # they were pushing (measured: shoved 13 cm in 5 s).
                    self.__dict__.setdefault('touch_pose', (centre.copy(), frame.copy()))
                in_band = all(self.load_low <= v <= self.load_high for v in loads.values())
                self.support_ticks = self.support_ticks+1 if in_band else 0
                pinch_centre, pinch_frame = getattr(self, 'touch_pose', (centre, frame))
                palms, rotations = self._palm_targets(pinch_centre, pinch_frame)
                # Squeezing a free block nudges it; only a real shove is a failure.
                disturbed = self._block_disturbed(pos_tol=.04, yaw_tol_deg=20.)
                if self.support_ticks >= 30:
                    self._begin_align(centre, frame)
                elif disturbed:
                    self._begin_backoff(disturbed)
                elif self.tick >= 800:
                    self._begin_backoff(f'CONTACT_NOT_ESTABLISHED(loads {loads["left"]:.1f}/{loads["right"]:.1f} N)')
            else:
                self._squeeze_servo()
                f = _quintic_scale(min(self.tick/self.stage_ticks, 1.))
                if self.stage == 'ALIGN':
                    frame_goal = _yaw_R(f*self.align_yaw) @ self.align_frame
                    goal = self.align_centre
                else:
                    frame_goal = self.hold_frame
                    goal = self.clear_start + np.array([0., 0., .035*f])
                palms, rotations = self._palm_targets(goal, frame_goal)
                if self.stage == 'ALIGN' and self.tick >= self.stage_ticks + 50:
                    self.metrics['align_rotation_deg'] = float(np.degrees(self.align_yaw))
                    self.metrics['align_residual_deg'] = float(np.degrees(np.arctan2(
                        frame[1, 0]*self.hold_frame[0, 0] - frame[0, 0]*self.hold_frame[1, 0],
                        frame[:, 0] @ self.hold_frame[:, 0])))
                    self.stage, self.tick = 'PICK_CLEAR', 0
                    self.stage_ticks = 350
                    self.clear_start = centre.copy()
                    self.clear_ticks = 0
                    self.env.model.opt.noslip_iterations = self.recovery.expert.config.hold_noslip_iterations
                elif self.stage == 'PICK_CLEAR':
                    clearance, held = self.recovery.lift_evidence()
                    self.clear_ticks = self.clear_ticks+1 if clearance >= .025 and held else 0
                    if self.clear_ticks >= 30:
                        self._begin_rise(centre)
                        self.agent_log.close('SUCCEEDED', 'pad grip established, block aligned and lifted clear',
                                             observe(self))
                    elif self.tick >= 1200:
                        self._begin_backoff(f'INITIAL_LIFT_NOT_ACHIEVED(clearance {clearance*1000:.0f} mm, held {held})')
                if self.stage == 'ALIGN' and max(loads.values()) < .15 and self.tick > 50:
                    self._begin_backoff('GRIP_LOST_IN_ALIGN')
            if self.stage == 'BACKOFF':
                return self._backoff_step()
            action[3:17] = self._servo_arms(palms, rotations)
        else:
            action = self._whole_body_stage(centre, frame, half, loads)
        self._finger_and_thumb_commands(action)
        held = self.stage in ('PICK_CLEAR', 'RISE', 'HOLD', 'DOWN')
        if held:
            tilt = float(np.degrees(np.arccos(np.clip(np.max(np.abs(R[2])), -1., 1.))))
            self.metrics['max_block_tilt_deg_held'] = max(self.metrics['max_block_tilt_deg_held'], tilt)
        return action

    def entry_waypoint(self, plan, s, goal=None):
        """Hand entry path of LOWER at progress s in [0, 1] (shared with the
        candidate path check, so what is checked is what is executed)."""
        goal_palms, goal_R = self.entry_targets(plan) if goal is None else goal
        if plan.via_proven:
            # Leg 1: e39d7ba's LOWER verbatim (pitch .40, height .45, knees
            # opened only in its last quarter). Leg 2: local move to the entry.
            split = PROVEN_TICKS/(PROVEN_TICKS + LOCAL_TICKS)
            pg, pR = self.proven_goal
            if s <= split:
                q = _quintic_scale(s/split)
                palms = self.start*(1-q) + pg*q
                rotations = [_slerp_R(a, b, q) for a, b in zip(self.start_R, pR)]
                return (palms, rotations, self.lower_start_height*(1-q) + .45*q,
                        self.lower_start_pitch*(1-q) + .40*q,
                        .06*_quintic_scale(np.clip((q-.75)/.25, 0., 1.)))
            q = _quintic_scale((s-split)/(1-split))
            if plan.via_height_m > 0.:
                # Local leg through a point above (and outside) the entry, then
                # straight down beside the faces: a direct local move was
                # measured clipping a block corner (13 mm shove, 6 deg turn).
                via = goal_palms.copy()
                for i, side in enumerate(('left', 'right')):
                    via[i] += np.array([0., 0., plan.via_height_m]) - plan.via_out_m*self.recovery.expert._inward_direction(side)
                palms = pg + (via-pg)*min(q/.6, 1.) if q < .6 else via + (goal_palms-via)*(q-.6)/.4
                rotations = [_slerp_R(a, b, min(q/.6, 1.)) for a, b in zip(pR, goal_R)]
            else:
                palms = pg*(1-q) + goal_palms*q
                rotations = [_slerp_R(a, b, q) for a, b in zip(pR, goal_R)]
            return (palms, rotations, .45*(1-q) + plan.crouch_height*q, .40*(1-q) + plan.crouch_pitch*q,
                    .06*(1-q) + plan.knee_spread_m*q)
        q = _quintic_scale(float(np.clip(s, 0., 1.)))
        rotations = [_slerp_R(a, b, min(q/.6, 1.) if plan.via_height_m > 0. else q)
                     for a, b in zip(self.start_R, goal_R)]
        if plan.via_height_m > 0.:
            via = goal_palms.copy()
            for i, side in enumerate(('left', 'right')):
                via[i] += np.array([0., 0., plan.via_height_m]) - plan.via_out_m*self.recovery.expert._inward_direction(side)
            palms = (self.start + (via-self.start)*min(q/.6, 1.)) if q < .6 else via + (goal_palms-via)*(q-.6)/.4
        else:
            palms = self.start*(1-q) + goal_palms*q
        height = self.lower_start_height*(1-q) + plan.crouch_height*q
        pitch = self.lower_start_pitch*(1-q) + plan.crouch_pitch*q
        spread = plan.knee_spread_m*_quintic_scale(np.clip((q-.75)/.25, 0., 1.))
        return palms, rotations, height, pitch, spread

    def proven_waypoint(self, s):
        """e39d7ba's LOWER from the current crouch (the calibration reference)."""
        q = _quintic_scale(float(np.clip(s, 0., 1.)))
        pg, pR = self.proven_goal
        return (self.start*(1-q) + pg*q, [_slerp_R(a, b, q) for a, b in zip(self.start_R, pR)],
                self.lower_start_height*(1-q) + .45*q, self.lower_start_pitch*(1-q) + .40*q,
                .06*_quintic_scale(np.clip((q-.75)/.25, 0., 1.)))

    def lower_ticks(self, plan=None):
        plan = self.plan if plan is None else plan
        return PROVEN_TICKS + LOCAL_TICKS if plan.via_proven else PROVEN_TICKS

    def _lower_step(self):
        self.tick += 1
        total = self.lower_ticks()
        palms, rotations, height, pitch, spread = self.entry_waypoint(self.plan, self.tick/total, (self.goal, self.goal_R))
        self.commanded_reach = (height, pitch, spread)
        action = self._reach_step(palms, rotations, height, pitch, spread)
        self.target_error_m = max(float(np.linalg.norm(self.env.palm_pose(s)[0]-palms[i]))
                                  for i, s in enumerate(('left', 'right')))
        self._finger_and_thumb_commands(action)
        if self.tick >= total:
            self.begin_contact()
        return action

    def _reach_step(self, palms, rotations, height, pitch, spread):
        """Crouched whole-body reach, exactly as LOWER plans it."""
        p = self.posture
        p.knee_lateral = self.stance_knee_lateral.copy()
        p.foot_rotations = [R.copy() for R in self.stance_foot_rotations]
        self.last_q = p.solve(height, palms, rotations, pitch, iterations=self.posture_iterations,
                              knee_outward_m=spread)
        self.posture_solves += 1
        return p.command(self.last_q)

    def _begin_backoff(self, cause):
        """Stop the attempt and retreat along motion already executed.

        A fresh retreat path (lift, then straight back) was measured to tip
        the crouched robot over backward. The entry path was just traversed
        stably, so: open the pinch sideways at the entry pose, then replay
        LOWER's own waypoints backward to the crouch start."""
        self.failed_plans.add(self.plan.key())
        self.backoff_cause = cause
        at_entry = self.stage != 'LOWER'
        self.backoff_s = 1. if at_entry else min(self.tick/self.lower_ticks(), 1.)
        self.backoff_open_ticks = 150 if at_entry else 0
        self.backoff_back_ticks = max(1, int(self.backoff_s*self.lower_ticks()))
        self.backoff_squeeze0 = dict(self.squeeze)
        self.backoff_palms0 = self._commanded_palms()
        self.backoff_R0 = [self.env.palm_pose(s)[1] for s in ('left', 'right')]
        self.__dict__.pop('touch_pose', None)
        self.stage, self.tick = 'BACKOFF', 0
        self.metrics.setdefault('backoffs', []).append(dict(tick=self.recovery.total_steps, cause=cause,
                                                            progress=round(self.backoff_s, 3)))

    def _backoff_step(self):
        """BACKOFF: open (at the entry pose only), replay LOWER backward,
        settle 100 ticks, then re-observe and re-decide. ABORT: hold still
        with the hands where they are and report the cause."""
        self.tick += 1
        if self.stage == 'ABORT':
            if self.tick == 1:
                self.abort_hold = ([self.env.palm_pose(s)[0] for s in ('left', 'right')],
                                   [self.env.palm_pose(s)[1] for s in ('left', 'right')],
                                   float(self.env.data.xpos[self.posture.pelvis, 2]), self.pitch)
            palms, rotations, height, pitch = self.abort_hold
            action = self._reach_step(np.array(palms), rotations, height, pitch, 0.)
            if self.tick >= 50:
                self.failure = self.abort_reason
        else:
            goal = (self.goal, self.goal_R)
            if self.tick <= self.backoff_open_ticks:
                # Open from where the hands ARE commanded, outward along each
                # hand's own side -- never relative to the block: a block that
                # was shoved away made that target a 100 mm reach that pulled
                # the crouched robot over.
                f = _quintic_scale(self.tick/self.backoff_open_ticks)
                opening = self.plan.entry_gap_m + max(self.backoff_squeeze0.values())
                palms = np.array([self.backoff_palms0[i] - f*opening*self.recovery.expert._inward_direction(side)
                                  for i, side in enumerate(('left', 'right'))])
                rotations = self.backoff_R0
                _, _, height, pitch, spread = self.entry_waypoint(self.plan, 1., goal)
                if self.tick == self.backoff_open_ticks:
                    self.goal, self.goal_R = palms.copy(), rotations  # replay back from here
            else:
                k = self.tick - self.backoff_open_ticks
                f = _quintic_scale(min(k/self.backoff_back_ticks, 1.))
                palms, rotations, height, pitch, spread = self.entry_waypoint(
                    self.plan, self.backoff_s*(1-f), goal)
            action = self._reach_step(palms, rotations, height, pitch, spread)
            if self.tick >= self.backoff_open_ticks + self.backoff_back_ticks + 100:
                self._restart_lower()
        self._finger_and_thumb_commands(action)
        return action

    def _restart_lower(self):
        d = self.env.data
        self.start = np.array([self.env.palm_pose(s)[0] for s in ('left', 'right')])
        self.start_R = [self.env.palm_pose(s)[1] for s in ('left', 'right')]
        self.lower_start_height = float(d.xpos[self.posture.pelvis, 2])
        self.lower_knee_spread_m = .06
        self.posture = WholeBodyPosture(self.env, self.ik_clearance_m)
        self._planned_with = None
        self.arm_plan_history = []
        self._decide_pick(replan=True, cause=self.backoff_cause)
        if self.stage != 'ABORT':
            self.stage, self.tick = 'LOWER', 0

    def _finger_and_thumb_commands(self, action):
        e = self.env
        opening = 1.
        if self.stage in ('RELEASE', 'RETRACT', 'STAND', 'READY_TO_VERIFY'):
            action[17:25] = 0.  # the extended pad shape is already open
        else:
            action[17:25] = np.clip((.8 - e._group_synergy)/e.config.hand_synergy_action_scale, -.1, .1)
        self.pad_grasp.apply_shape(opening)
        for i in (0, 4):
            action[17+i] = np.clip((.7-e._group_synergy[i])/e.config.hand_synergy_action_scale, -.1, .1)
        for side in sc.SIDES:
            for suffix in sc.preshape_suffixes('thumb'):
                aid = e.model.actuator(sc.sharpa_actuator(side, 'thumb', suffix)).id
                e.data.ctrl[aid] = np.clip(sc.PRESHAPE_TARGETS['thumb'][suffix], *e.model.actuator_ctrlrange[aid])

    def _begin_close(self):
        # super().begin_contact() already latched the crouched legs/hands.
        self.support_ticks = 0
        self.offsets = self._grip_offsets()

    def _begin_align(self, centre, frame):
        self.stage, self.tick = 'ALIGN', 0
        heading = self.recovery.expert.task_rotation
        # Nearest quarter turn of the table (world-aligned line) to the block's
        # forward axis; the robot heading only chose which faces to hold.
        yaw = float(np.arctan2(frame[1, 0], frame[0, 0]))
        line = float(np.arctan2(heading[1, 0], heading[0, 0]))
        del line  # Line 2's table is world-aligned: its quarter turns are world ones.
        target = np.round(yaw/(np.pi/2))*(np.pi/2)
        self.align_yaw = float(np.arctan2(np.sin(target-yaw), np.cos(target-yaw)))
        self.align_frame = frame.copy()
        self.align_centre = centre.copy()
        self.hold_frame = _yaw_R(self.align_yaw) @ frame
        self.stage_ticks = max(200, int(abs(self.align_yaw)/.0012))
        self.pick_rotations = [self.env.palm_pose(s)[1] for s in ('left', 'right')]
        self.pick_object_R = self._block()[1]

    def standing_knees(self, posture):
        """Standing knee width as before the crouch, re-centred on the feet as
        they are now (they slide ~1-2 cm while crouched)."""
        lateral = posture.base_rotation[:, 1]
        feet_centre = float(np.mean([pos @ lateral for pos in posture.foot_positions]))
        stance = np.array(self.stance_knee_lateral)
        return stance + (feet_centre - stance.mean())

    def transfer_frame(self, plan, lift, base_height, base_pitch, com0, knee0, knee_stand, feet_mid, heading):
        """Block path from the lift to the table, shared by RISE/HOLD/DOWN and
        by the candidate transfer check: L lift -> A up to the crossing
        height, still short of the table face -> O just over the table ->
        B above the place spot -> C down onto the table."""
        _, top, face = self._table(heading)
        half = float(self.env.model.geom_size[self.env.model.geom(self.env.object_geom_name).id][0])
        forward = heading[:, 0]
        rest_z = top + half
        cross_z = rest_z + plan.cross_clearance_m
        L = np.asarray(lift, dtype=float).copy()
        # H: low in front of the body while it rises.
        # Keep the block's front face >= 3.5 cm off the table face while it
        # goes up: the held block sways ~2 cm fore-aft, and a 1-2 cm margin
        # let it brush the table face (lift evidence lost).
        short_of_face = face - half - .025
        H = L + forward*(min(L @ forward + .03, short_of_face) - L @ forward)
        H[2] = max(L[2], plan.rise_block_z)
        A = L + forward*(short_of_face - L @ forward)
        A[2] = cross_z
        target = self.place_target_for(plan, heading)
        O = target.copy()
        O += forward*((face + plan.over_x_m) - target @ forward)
        O[2] = cross_z
        if O @ forward > target @ forward:
            O = target.copy(); O[2] = cross_z
        B, C = target.copy(), target.copy()
        B[2], C[2] = cross_z, rest_z - .004
        return dict(L=L, H=H, A=A, O=O, B=B, C=C, base=(float(base_height), float(base_pitch)),
                    com0=np.asarray(com0, dtype=float)[:2].copy(), knee0=np.asarray(knee0, dtype=float),
                    knee_stand=np.asarray(knee_stand, dtype=float), feet_mid=np.asarray(feet_mid, dtype=float),
                    heading=heading, carry=(plan.carry_height, plan.carry_pitch, plan.carry_com_m),
                    place=(plan.place_height, plan.place_pitch, plan.place_com_m), rise_lag=plan.rise_lag)

    # Transfer progress breakpoints:
    # rise (block low) | block up to the crossing | over the edge | lean to spot | down.
    TRANSFER_S = (.3, .45, .6, .85)

    @staticmethod
    def transfer_waypoint(tf, s):
        """Block centre and whole-body posture at transfer progress s in [0, 1]:
        [0, .3] L->H rising from the crouch into the carry posture, block low,
        [.3, .45] H->A block up to the crossing height, short of the table face,
        [.45, .6] A->O over the table edge while hinging into the place lean,
        [.6, .85] O->B to the place spot, leaning,
        [.85, 1] B->C down onto the table."""
        s0, s1, s2, s3 = FloorTableCycle.TRANSFER_S
        s = float(np.clip(s, 0., 1.))
        base = (*tf['base'], tf['com0'])
        carry_h, carry_p, carry_c = tf['carry']
        place_h, place_p, place_c = tf['place']
        forward = tf['heading'][:2, 0]
        def posture(a, b, f):
            (h0, p0, c0), (h1, p1, c1) = a, b
            c0 = c0 if np.ndim(c0) else tf['feet_mid'][:2] + c0*forward
            c1 = tf['feet_mid'][:2] + c1*forward
            return h0*(1-f) + h1*f, p0*(1-f) + p1*f, c0*(1-f) + c1*f
        if s <= s0:
            q = _quintic_scale(s/s0)
            lag = tf.get('rise_lag', 0.)
            blend = _quintic_scale(float(np.clip((s/s0 - lag)/(1. - lag), 0., 1.)))
            centre = tf['L'] + (tf['H']-tf['L'])*q
            height, pitch, com = posture(base, tf['carry'], blend)
            knees = tf['knee0']*(1-blend) + tf['knee_stand']*blend
            return centre, height, pitch, com, knees
        if s <= s1:
            q = _quintic_scale((s-s0)/(s1-s0))
            centre = tf['H'] + (tf['A']-tf['H'])*q
            height, pitch, com = posture((carry_h, carry_p, carry_c), tf['carry'], 1.)
        elif s <= s2:
            # Hinge fully into the place lean while crossing the edge: carried
            # forward over the table upright, the body backed away to balance
            # the reaching arms (pelvis -12 cm) until the hands ran out of reach.
            q = _quintic_scale((s-s1)/(s2-s1))
            centre = tf['A'] + (tf['O']-tf['A'])*q
            height, pitch, com = posture((carry_h, carry_p, carry_c), tf['place'], q)
        elif s <= s3:
            q = _quintic_scale((s-s2)/(s3-s2))
            centre = tf['O'] + (tf['B']-tf['O'])*q
            height, pitch, com = posture((place_h, place_p, place_c), tf['place'], 1.)
        else:
            q = _quintic_scale((s-s3)/(1-s3))
            centre = tf['B'] + (tf['C']-tf['B'])*q
            height, pitch, com = posture((place_h, place_p, place_c), tf['place'], 1.)
        return centre, height, pitch, com, tf['knee_stand']

    def _begin_rise(self, centre):
        d = self.env.data
        self.stage, self.tick = 'RISE', 0
        self.posture = WholeBodyPosture(self.env, self.ik_clearance_m)
        p = self.posture
        self.heading = p.base_rotation
        self.feet_mid = np.mean(p.foot_positions, axis=0)
        self.rise_knee_stand = self.standing_knees(p)
        self.tf = self.transfer_frame(self.plan, centre, float(d.xpos[p.pelvis, 2]),
                                      float(self.recovery.stabilizer.tilt()[1]), d.subtree_com[p.pelvis],
                                      np.array(p.knee_lateral), self.rise_knee_stand, self.feet_mid, self.heading)
        self.hold_point = self.tf['B']
        self.place_target = self.tf['C']
        self.carry_posture = self.tf['carry']
        self.place_posture_full = self.tf['place']
        # Loaded rise: stiffen the arms (as the walking carry does) and let the
        # common squeeze react faster and deeper. At the grasp's compliant
        # kp=120 the pads unloaded to ~2 N while the body straightened.
        from humanoid_learning.envs import model_builder, task_config as tc
        kp = getattr(self.recovery.config, 'carry_arm_kp', 300.)
        model_builder._apply_compliant_kp(self.env.model, tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS, kp)
        self.env.config.arm_kp = kp
        self.squeeze_rate, self.squeeze_cap = .0003, .04
        # The block path peaked at ~1.6 mm/tick while the arm servo was capped
        # at 1 mm/tick: the commanded hands fell 10 -> 150 mm behind the block
        # frame and both pads unloaded at block height ~.4 m. Slower path,
        # faster servo.
        self.arm_step_m, self.arm_dq_max, self.stage_ticks = .0025, .008, 2400
        # Follow the whole-body plan's arm branch (it keeps joint margin) much
        # more firmly: a .02 null-space pull left the servo in a local branch
        # that ran the wrist pitch into its limit with the block near the chest.
        self.arm_pull_gain = .1
        self.torso_clear_rate = .002  # .0005 let the left shoulder press the torso at 140 N

    def begin_place(self):
        """Called by FactoryRecovery once the held block over the table has
        been verified: lower it onto the table along the same transfer path."""
        self.stage, self.tick = 'DOWN', 0
        self.stage_ticks = 2200

    def _plan(self, palms, rotations, height, pitch, com_forward, knees=None):
        p = self.posture
        p.com_xy = self.feet_mid[:2] + com_forward*self.heading[:2, 0]
        self.last_q = p.solve(height, palms, rotations, pitch, iterations=self.posture_iterations,
                              knee_lateral_targets=knees)
        self.posture_solves += 1
        return p.command(self.last_q)

    def _whole_body_stage(self, centre, frame, half, loads):
        d = self.env.data
        f = _quintic_scale(min(self.tick/max(self.stage_ticks, 1), 1.))
        knees = self.rise_knee_stand
        grip = self.stage in ('RISE', 'HOLD', 'DOWN')
        if grip:
            self._squeeze_servo()
        if self.stage in ('RISE', 'HOLD', 'DOWN'):
            over = self.TRANSFER_S[1]  # RISE/HOLD end at the crossing point A
            if self.stage == 'RISE':
                progress = over*min(self.tick/self.stage_ticks, 1.)
            elif self.stage == 'HOLD':
                progress = over
            else:
                progress = over + (1.-over)*min(self.tick/self.stage_ticks, 1.)
            goal, height, pitch, com, knees = self.transfer_waypoint(self.tf, progress)
            # Close the loop on the block itself: the compliant arms let the held
            # block drift ~3 cm ahead of its path as the body moved, into the
            # table face. Aim the hands past the error (bounded).
            if self.stage in ('RISE', 'HOLD') or progress < self.TRANSFER_S[3]:
                error = goal - centre
                self.block_bias = np.clip(getattr(self, 'block_bias', np.zeros(3)) + .02*error, -.04, .04)
                aim = goal + np.clip(error, -.03, .03) + self.block_bias
            else:
                aim = goal
            palms, rotations = self._palm_targets(aim, self.hold_frame)
            p = self.posture
            p.com_xy = com
            self.last_q = p.solve(height, palms, rotations, pitch, iterations=self.posture_iterations,
                                  knee_lateral_targets=knees)
            self.posture_solves += 1
            action = p.command(self.last_q)
            self.hold_goal = goal
            if self.stage in ('RISE', 'HOLD'):
                clearance, held = self.recovery.lift_evidence()
                self.rise_contact_gap = 0 if held else getattr(self, 'rise_contact_gap', 0)+1
                if self.tick >= 200 and self.rise_contact_gap > 50:
                    self.failure = 'FLOOR_LIFT_CONTACT_LOST'
            if self.stage == 'RISE' and self.tick >= self.stage_ticks:
                carry_height, carry_pitch, _ = self.carry_posture
                if (abs(d.xpos[p.pelvis, 2]-carry_height) < .03
                        and abs(self.recovery.stabilizer.tilt()[1]-carry_pitch) < .15):
                    self.stage, self.tick = 'HOLD', 0
                elif self.tick >= self.stage_ticks + 1000:
                    self.failure = 'FLOOR_STAND_NOT_ACHIEVED'
            elif self.stage == 'DOWN':
                force = self._table_force()
                weight = float(self.env.model.body_subtreemass[self.env._object_body_id])*9.81
                self.metrics['table_force_peak_n'] = max(self.metrics['table_force_peak_n'], force)
                # The table must carry the block before the pads may let go.
                # Only the final descent onto the spot counts: crossing over, a
                # sagging block that grazed the table's front edge was taken
                # for support and let go there (it fell off the edge).
                on_spot = progress >= self.TRANSFER_S[3]
                self.table_supported_ticks = self.table_supported_ticks+1 if (on_spot and force > .5*weight) else 0
                if self.table_supported_ticks >= 40:
                    self.stage, self.tick = 'RELEASE', 0
                    self.stage_ticks = 250
                    self.release_frame = self.hold_frame.copy()
                    self.release_centre = goal.copy()
                    self.release_squeeze = dict(self.squeeze)
                elif self.tick >= self.stage_ticks + 400:
                    self.failure = 'PLACE_SUPPORT_NOT_ESTABLISHED'
        else:
            action = self._release_stage(f, knees)
            palms = rotations = None
        if palms is not None:
            action[3:17] = self._servo_arms(palms, rotations, reference_q=self.last_q[15:29])
        target_error = self.recovery.env.part_position(self.recovery.station) - self.recovery.env._part_rest_pos(
            self.recovery.station, fc.LOCAL_CANONICAL_PART_XY)
        self.final_error_m = float(np.linalg.norm(target_error))
        R = self._block()[1]
        self.tilt_deg = float(np.degrees(np.arccos(np.clip(np.max(np.abs(R[2])), -1., 1.))))
        return action

    def _release_stage(self, f, knees):
        if self.stage == 'RELEASE':
            # Sideways only: each pad backs off its face; no push along the table.
            squeeze = {s: self.release_squeeze[s] - .035*f for s in self.release_squeeze}
            palms, rotations = self._palm_targets(self.release_centre, self.release_frame, squeeze)
            action = self._plan(palms, rotations, *self.place_posture_full, knees)
            action[3:17] = self._servo_arms(palms, rotations)
            if self.tick >= self.stage_ticks:
                self.stage, self.tick = 'RETRACT', 0
                self.stage_ticks = 500
                self.retract_start = palms.copy()
                self.retract_R = rotations
        elif self.stage == 'RETRACT':
            lift = np.array([0., 0., .10*f]) - self.heading[:, 0]*.12*f
            palms = self.retract_start + lift
            lean = 1-f
            carry_height, carry_pitch, carry_com = self.place_posture_full
            action = self._plan(palms, self.retract_R, self.standing_height + (carry_height-self.standing_height)*lean,
                                carry_pitch*lean, .025 + (carry_com-.025)*lean, knees)
            action[3:17] = self._servo_arms(palms, self.retract_R)
            if self.tick >= self.stage_ticks:
                self.stage, self.tick = 'STAND', 0
                self.stage_ticks = 300
        else:
            # Stand upright holding the retracted hands where RETRACT left them.
            # Homing the arms in joint space swung them back through the body
            # and tipped the robot over backward (and swept the block off).
            palms = self.retract_start + np.array([0., 0., .10]) - self.heading[:, 0]*.12
            action = self._plan(palms, self.retract_R, self.standing_height, 0., .025, knees)
            action[3:17] = self._servo_arms(palms, self.retract_R)
            if self.stage == 'STAND' and self.tick >= self.stage_ticks:
                self.hands_clear = self._hands_clear()
                if self.hands_clear and self.recovery.env.task_manager._part_is_back(self.recovery.env):
                    self.stage, self.tick = 'READY_TO_VERIFY', 0
                elif self.tick >= self.stage_ticks + 400:
                    self.failure = 'PLACE_RETRACT_OR_SETTLE_NOT_ACHIEVED'
            elif self.stage == 'READY_TO_VERIFY':
                self.hands_clear = self._hands_clear()
        return action


