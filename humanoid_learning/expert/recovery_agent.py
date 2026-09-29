"""Observe -> select -> execute -> verify -> replan for the floor->table recovery.

This is NOT a new controller stack. The physical skills stay in
FloorTableCycle / WholeBodyPosture / the pad servos, which keep running every
tick. This module only owns the slow loop around them:

* observe(cycle)          measured state (block, balance, contacts, clearances)
* CandidateEvaluator      scratch reach checks that link one stage to the next
                          (place reach -> pick end pose -> hand entry PATH)
* RuleSelector            picks an executable candidate or returns an explicit
                          ABORT; it never sees joint commands
* DecisionLog             what was observed, considered, chosen and why, and
                          what the physics then did

A scratch IK success is only evidence that a candidate is worth trying; the
decision record keeps that apart from the physical outcome, which is written
back when the skill reports success or a failure reason. The selector is an
interface (select(observation, reports, history)) so another chooser can be
swapped in and compared on the same records.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from humanoid_learning.expert.whole_body_posture import WholeBodyPosture


@dataclass(frozen=True)
class GraspPlan:
    """One executable way to pick: pinch geometry + crouch reach posture."""
    phi_deg: float = 75.       # finger angle below the block forward axis
    forward_m: float = -.02    # middle pad along the block forward axis
    crouch_pitch: float = .40  # pelvis pitch the LOWER reach ends at
    crouch_height: float = .45
    knee_spread_m: float = .06
    entry_gap_m: float = .05
    # Entry path: 0 = straight from the crouch pose (as e39d7ba's LOWER);
    # > 0 = over the top: via a point this high above (and via_out_m outside)
    # the entry pose, then down beside the faces.
    via_height_m: float = 0.
    via_out_m: float = 0.
    # Through e39d7ba's proven reach pose first (its exact LOWER), then a
    # local move to this plan's entry pose.
    via_proven: bool = False
    # Place: CoM lead over the planted feet (sole contact reaches +.125 m
    # ahead of the foot sites) and how far short of the canonical spot to aim
    # (the verification gate accepts 60 mm).
    place_com_m: float = .07
    place_backoff_m: float = .045
    # Carry posture while the block crosses the table edge. Measured from the
    # standing pick stance: UPRIGHT (pitch 0) holds the pinched block high
    # near the body with .32 rad joint margin, while a .5 rad forward lean
    # ran the same crossing into the arm joint limits.
    carry_height: float = .76
    carry_pitch: float = 0.
    cross_clearance_m: float = .03
    rise_lag: float = 0.   # fraction of the lift the body waits before rising
    # Lean only for the final reach onto the place spot (hip hinge).
    place_height: float = .77
    place_pitch: float = .5
    over_x_m: float = .06  # block beyond the table face where the lean starts
    # Block height held while the body rises (low and close, where this pinch
    # keeps wide joint margins); it is raised to the crossing height only once
    # the body is up.
    rise_block_z: float = .45
    carry_com_m: float = .03   # CoM lead while carrying upright (place_com_m only for the place lean)

    def key(self):
        return (self.phi_deg, self.forward_m, self.crouch_pitch, self.crouch_height,
                self.knee_spread_m, self.entry_gap_m, self.via_height_m, self.via_out_m, self.via_proven,
                self.place_com_m, self.place_backoff_m, self.carry_height, self.carry_pitch, self.cross_clearance_m,
                self.rise_lag, self.place_height, self.place_pitch, self.over_x_m, self.rise_block_z,
                self.carry_com_m)

    def end_key(self):
        return self.key()[:6]

    def path_key(self):
        return self.key()[:9]

    def place_key(self):
        return (self.phi_deg, self.forward_m, self.place_com_m, self.place_backoff_m,
                self.place_height, self.place_pitch)

    def transfer_key(self):
        return self.place_key() + (self.cross_clearance_m, self.rise_lag, self.carry_height,
                                   self.carry_pitch, self.over_x_m)


@dataclass
class Observation:
    tick: int
    stage: str
    block_centre: list
    block_yaw_rel_deg: float      # block face frame vs robot heading
    block_tilt_deg: float
    block_speed: float
    pelvis: list
    pelvis_roll_pitch_deg: list
    foot_loads_n: list
    pad_loads_n: dict
    hand_leg_clearance_m: float
    plan_palm_error_m: float

    def summary(self):
        return {k: (np.round(v, 3).tolist() if isinstance(v, (list, np.ndarray)) else
                    (round(v, 3) if isinstance(v, float) else v)) for k, v in asdict(self).items()}


def _leg_geoms(model, pelvis):
    names = ('hip', 'knee', 'ankle', 'pelvis')
    return [g for g in range(model.ngeom)
            if model.body_rootid[model.geom_bodyid[g]] == pelvis
            and any(n in (model.body(int(model.geom_bodyid[g])).name or '') for n in names)
            and (model.geom_contype[g] or model.geom_conaffinity[g])]


def hand_leg_clearance(env, model, data, pelvis):
    """Nearest hand/wrist-to-leg distance, measured on actual geometry."""
    from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift
    hands = SharpaContactLift.hand_wrist_body_ids(env)
    hand_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in (hands['left'] | hands['right'])
                  and (model.geom_contype[g] or model.geom_conaffinity[g])]
    nearest = 1.
    for h in hand_geoms:
        for leg in _leg_geoms(model, pelvis):
            nearest = min(nearest, float(mujoco.mj_geomDistance(model, data, h, leg, .2, None)))
    return nearest


def hand_block_clearance(env, model, data, block_geom):
    """Nearest hand/wrist-to-block distance (the block as it lies now)."""
    from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift
    hands = SharpaContactLift.hand_wrist_body_ids(env)
    nearest = 1.
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in (hands['left'] | hands['right']) and (model.geom_contype[g] or model.geom_conaffinity[g]):
            nearest = min(nearest, float(mujoco.mj_geomDistance(model, data, g, block_geom, .2, None)))
    return nearest


def observe(cycle) -> Observation:
    env, m, d = cycle.env, cycle.env.model, cycle.env.data
    centre, R, _ = cycle._block()
    frame = cycle._block_frame(R)
    heading = cycle.recovery.expert.task_rotation
    rel = float(np.degrees(np.arctan2(frame[:, 0] @ heading[:, 1], frame[:, 0] @ heading[:, 0])))
    tilt = float(np.degrees(np.arccos(np.clip(np.max(np.abs(R[2])), -1., 1.))))
    dof = env.model.jnt_dofadr[env.model.body_jntadr[env._object_body_id]]
    roll, pitch, _, _ = cycle.recovery.stabilizer.tilt()
    loads = [0., 0.]
    floor = m.geom('floor').id
    for i, c in enumerate(d.contact[:d.ncon]):
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            for side, body in enumerate(cycle.recovery.walker.foot_bodies):
                if m.geom_bodyid[other] == body:
                    f = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, f)
                    loads[side] += float(f[0])
    plan_error = max(cycle.posture.last_error.get('palm_m', [0.])) if cycle.posture.last_error else 0.
    return Observation(
        tick=cycle.recovery.total_steps, stage=cycle.stage, block_centre=centre.tolist(),
        block_yaw_rel_deg=rel, block_tilt_deg=tilt,
        block_speed=float(np.linalg.norm(d.qvel[dof:dof+3])),
        pelvis=d.xpos[cycle.posture.pelvis].tolist(),
        pelvis_roll_pitch_deg=[float(np.degrees(roll)), float(np.degrees(pitch))],
        foot_loads_n=loads, pad_loads_n=dict(getattr(cycle, 'normal_loads', {})),
        hand_leg_clearance_m=hand_leg_clearance(env, m, d, cycle.posture.pelvis),
        plan_palm_error_m=float(plan_error))


@dataclass
class CandidateReport:
    plan: GraspPlan
    place_error_m: float = float('inf')
    pick_end_error_m: float = float('inf')
    path_worst_error_m: float = float('inf')
    path_min_clearance_m: float = float('inf')
    place_posture: tuple | None = None
    path_trace: list | None = None
    path_block_clearance_m: float = float('inf')
    path_branch_jump_rad: float = float('inf')
    transfer_worst_m: float = float('inf')
    transfer_low_margin_fraction: float = float('inf')
    transfer_jump_rad: float = float('inf')
    transfer_trace: list | None = None
    rejected_at: str | None = None

    @property
    def feasible(self):
        return self.rejected_at is None

    def summary(self):
        out = dict(plan=asdict(self.plan))
        for k in ('place_error_m', 'pick_end_error_m', 'path_worst_error_m', 'path_min_clearance_m',
                  'path_block_clearance_m', 'path_branch_jump_rad', 'transfer_worst_m',
                  'transfer_low_margin_fraction', 'transfer_jump_rad'):
            v = getattr(self, k)
            out[k] = None if not np.isfinite(v) else round(float(v), 4)
        out['rejected_at'] = self.rejected_at
        if self.path_trace:
            worst = min(self.path_trace, key=lambda t: t[1])
            out['min_clearance_at_progress'] = worst[0]
        return out


class CandidateEvaluator:
    """Scratch-only reach checks. Nothing here touches live data or commands."""

    PLACE_TOL_M = .006
    PICK_TOL_M = .006
    # Calibrated on e39d7ba's own LOWER from the same crouch (physically
    # successful): its scratch path check reads 8-19 mm worst error.
    PATH_TOL_M = .025
    PATH_CLEARANCE_M = .004
    # Hands may not approach the block closer than this before the pads'
    # entry pose (measured: a local entry move that clipped a block corner
    # shoved it 15 mm and turned it; self-collision-only checks missed it).
    BLOCK_CLEARANCE_M = .008
    # Largest arm-joint change between consecutive path samples (~22 ticks).
    # Measured: the plan switched the left arm branch mid-path (elbow 1.30 ->
    # 2.09, wrist roll -.15 -> -.88 in ~25 ticks) while its palm error stayed
    # small; the real arm swept through the block on its way there.
    BRANCH_JUMP_RAD = .3

    def __init__(self, cycle):
        self.cycle = cycle
        self._shaped = None
        # Uncalibrated defaults; evaluate() re-derives them from the reference.
        self.path_tol, self.path_clearance = self.PATH_TOL_M, self.PATH_CLEARANCE_M

    def shaped_env(self):
        """A read-only stand-in for the live env whose data COPY has the hand
        shape LOWER actually executes (straight fingers, parked thumbs).
        Checking with the crouch's curled fingers overstated the hand-block
        clearance: a path read 28 mm clear while the straightened fingertips
        clipped the block. Live data is never written."""
        if self._shaped is None:
            import types
            c = self.cycle
            m = c.env.model
            data = mujoco.MjData(m)
            data.qpos[:] = c.env.data.qpos
            c.hand_shape_qpos(data.qpos)
            mujoco.mj_forward(m, data)
            self._shaped = types.SimpleNamespace(
                model=m, data=data, config=c.env.config,
                _left_hand_body_ids=c.env._left_hand_body_ids, _right_hand_body_ids=c.env._right_hand_body_ids)
        return self._shaped

    def _posture(self):
        c = self.cycle
        p = WholeBodyPosture(self.shaped_env(), c.ik_clearance_m)
        if c.frog_stance:
            p.knee_lateral = c.stance_knee_lateral.copy()
            p.foot_rotations = [R.copy() for R in c.stance_foot_rotations]
        return p

    def place_error(self, plan, target, feet_mid, heading):
        """Hip-hinge reach from the planted feet to the place pose (block level,
        on the table axes). Best of a few lean postures."""
        c = self.cycle
        target = c.place_target_for(plan)
        offsets = c._grip_offsets(plan)
        frame = c.aligned_frame()
        palms, rotations = c._palm_targets(target, frame, {'left': 0., 'right': 0.}, offsets)
        best, posture = float('inf'), None
        for height, pitch in ((plan.place_height, plan.place_pitch),):
            p = WholeBodyPosture(self.shaped_env(), c.ik_clearance_m)
            p.com_xy = feet_mid[:2] + plan.place_com_m*heading[:2, 0]
            p.solve(height, palms, rotations, pitch, iterations=150)
            error = max(p.last_error['palm_m']) + p.last_error['com_m']
            if error < best:
                best, posture = error, (height, pitch)
            if best < self.PLACE_TOL_M:
                break
        return best, posture

    def pick_end_error(self, plan):
        c = self.cycle
        palms, rotations = c.entry_targets(plan)
        p = self._posture()
        p.solve(plan.crouch_height, palms, rotations, plan.crouch_pitch, iterations=150,
                knee_outward_m=plan.knee_spread_m)
        return max(p.last_error['palm_m'])

    def path(self, plan, samples=96, iterations=16):
        """Follow the SAME entry path LOWER executes (cycle.entry_waypoint) on
        one warm-started scratch plan and record the worst tracking error and
        the nearest hand-leg distance along the way."""
        c = self.cycle
        goal = c.entry_targets(plan) if plan is not None else c.proven_goal
        p = self._posture()
        worst, clearance = 0., 1.
        self.last_path_trace = []
        self.last_block_clearance = 1.
        self.last_branch_jump = 0.
        previous_arms = None
        model = p.ik_model
        block = c.env.model.geom(c.env.object_geom_name).id
        full = mujoco.MjData(c.env.model)
        for i in range(1, samples+1):
            if plan is None:  # reference: e39d7ba's own LOWER, same checker
                palms, rotations, height, pitch, spread = c.proven_waypoint(i/samples)
            else:
                palms, rotations, height, pitch, spread = c.entry_waypoint(plan, i/samples, goal)
            q = p.solve(height, palms, rotations, pitch, iterations=iterations, knee_outward_m=spread)
            worst = max(worst, max(p.last_error['palm_m']))
            if previous_arms is not None:
                self.last_branch_jump = max(self.last_branch_jump, float(np.abs(q[15:] - previous_arms).max()))
            previous_arms = q[15:].copy()
            if i % 3 == 0 or i == samples:
                gap = hand_leg_clearance(c.env, model, p.scratch, p.pelvis)
                clearance = min(clearance, gap)
                self.last_path_trace.append((round(i/samples, 2), round(gap*1000, 1)))
                if plan is not None:
                    # Block geometry is disabled in the planning model: measure on the live model.
                    full.qpos[:] = p.scratch.qpos
                    mujoco.mj_kinematics(c.env.model, full)
                    self.last_block_clearance = min(self.last_block_clearance,
                                                    hand_block_clearance(c.env, c.env.model, full, block))
            if worst > 2*self.path_tol:
                break
        return worst, clearance

    TRANSFER_TOL_M = .006
    TRANSFER_LOW_MARGIN_RAD = .02
    TRANSFER_LOW_MARGIN_FRACTION = .1

    def transfer(self, plan, samples=70, iterations=16):
        """Follow the SAME transfer path RISE/HOLD/DOWN execute, starting from
        this plan's grasp crouch (scratch). Endpoint checks alone passed a hold
        pose that sat on the arm joint limits and a table-edge crossing the
        pinch could not reach; this walks the whole path."""
        c = self.cycle
        offsets = c._grip_offsets(plan)
        centre, R, _ = c._block()
        grasp = c._palm_targets(centre, c._block_frame(R), {'left': 0., 'right': 0.}, offsets)
        crouch = self._posture()
        crouch.solve(plan.crouch_height, grasp[0], grasp[1], plan.crouch_pitch, iterations=150,
                     knee_outward_m=plan.knee_spread_m)
        # Execution re-measures the body when RISE starts (new posture: feet,
        # toe-out, knees as they are). Do the same from the grasp solution.
        import types
        env = self.shaped_env()
        data = mujoco.MjData(env.model)
        data.qpos[:] = crouch.scratch.qpos
        c.hand_shape_qpos(data.qpos)
        mujoco.mj_forward(env.model, data)
        p = WholeBodyPosture(types.SimpleNamespace(**{**vars(env), 'data': data}), c.ik_clearance_m)
        _, feet_mid, heading = c.place_target_estimate()
        lateral = heading[:, 1]
        knee0 = np.array([float(data.xpos[b] @ lateral) for b in p.knees])
        pitch0 = float(np.arcsin(-data.xmat[p.pelvis][6]))
        tf = c.transfer_frame(plan, centre + np.array([0., 0., .035]), float(data.xpos[p.pelvis, 2]), pitch0,
                              data.subtree_com[p.pelvis], knee0, c.standing_knees(p), feet_mid, heading)
        frame = c.aligned_frame()
        rng = p.ik_model.jnt_range[p.jids]
        worst, low, jump, previous = 0., 0, 0., None
        self.last_transfer_trace = []
        for i in range(1, samples+1):
            goal, height, pitch, com, knees = c.transfer_waypoint(tf, i/samples)
            palms, rotations = c._palm_targets(goal, frame, {'left': 0., 'right': 0.}, offsets)
            p.com_xy = com
            q = p.solve(height, palms, rotations, pitch, iterations=iterations, knee_lateral_targets=knees)
            error = max(p.last_error['palm_m']) + p.last_error['com_m']
            worst = max(worst, error)
            margin = float(np.minimum(q-rng[:, 0], rng[:, 1]-q)[15:].min())
            low += margin < self.TRANSFER_LOW_MARGIN_RAD
            if previous is not None:
                jump = max(jump, float(np.abs(q[15:]-previous).max()))
            previous = q[15:].copy()
            if i % 7 == 0:
                self.last_transfer_trace.append((round(i/samples, 2), round(error*1000, 1), round(margin, 2)))
        return worst, low/samples, jump

    def calibrate(self):
        """Measure e39d7ba's physically successful LOWER with THIS checker,
        from THIS state. A candidate path is not asked to be cleaner than
        that reference (scratch planner lag reads as a few mm of overlap)."""
        self.path_tol, self.path_clearance = self.PATH_TOL_M, self.PATH_CLEARANCE_M
        worst, clearance = self.path(None, samples=64)
        self.reference = dict(worst_m=worst, clearance_m=clearance)
        self.path_tol = max(self.PATH_TOL_M, worst + .005)
        self.path_clearance = min(self.PATH_CLEARANCE_M, clearance - .001)
        self.reference.update(path_tol_m=self.path_tol, clearance_floor_m=self.path_clearance)

    def evaluate(self, plans, target, feet_mid, heading, excluded=(), path_budget=12, wanted=2,
                 transfer_budget=10):
        self.calibrate()
        reports, place_cache, end_cache = [], {}, {}
        for plan in plans:
            r = CandidateReport(plan)
            reports.append(r)
            if plan.key() in excluded:
                r.rejected_at = 'failed_before'
                continue
            key = plan.place_key()
            if key not in place_cache:
                place_cache[key] = self.place_error(plan, target, feet_mid, heading)
            r.place_error_m, r.place_posture = place_cache[key]
            if r.place_error_m > self.PLACE_TOL_M:
                r.rejected_at = 'place_reach'
                continue
            if plan.end_key() not in end_cache:
                end_cache[plan.end_key()] = self.pick_end_error(plan)
            r.pick_end_error_m = end_cache[plan.end_key()]
            if r.pick_end_error_m > self.PICK_TOL_M:
                r.rejected_at = 'pick_end'
        # Transfer (lift -> table) for the best surviving grasp/carry choices.
        transfer_cache, done = {}, 0
        # Round-robin over finger angles: variants of one grasp must not use
        # up the whole budget (measured: ten phi=65 variants, 55/60 unchecked).
        by_phi = {}
        for r in sorted([r for r in reports if r.feasible], key=RuleSelector.static_cost):
            by_phi.setdefault(r.plan.phi_deg, []).append(r)
        interleaved = []
        while any(by_phi.values()):
            for phi in sorted(by_phi):
                if by_phi[phi]:
                    interleaved.append(by_phi[phi].pop(0))
        for r in interleaved:
            key = r.plan.transfer_key()
            if key not in transfer_cache:
                if done >= transfer_budget:
                    r.rejected_at = 'not_transfer_checked'
                    continue
                done += 1
                transfer_cache[key] = self.transfer(r.plan) + (self.last_transfer_trace,)
            worst, low, jump, trace = transfer_cache[key]
            r.transfer_worst_m, r.transfer_low_margin_fraction, r.transfer_jump_rad = worst, low, jump
            r.transfer_trace = trace
            if worst > self.TRANSFER_TOL_M:
                r.rejected_at = 'transfer_reach'
            elif low > self.TRANSFER_LOW_MARGIN_FRACTION:
                r.rejected_at = 'transfer_joint_limits'
            elif jump > self.BRANCH_JUMP_RAD:
                r.rejected_at = 'transfer_branch_switch'
        # Path checks are the expensive link: end poses in cost order, and for
        # each end pose its entry paths in order of added motion (straight,
        # through the proven reach pose, over the top) -- a blocked path
        # changes the path before it changes the grasp. Stop once `wanted`
        # pass or the budget is spent; the rest are marked, not assumed.
        survivors = [r for r in reports if r.feasible]
        order = {}
        for r in survivors:
            order.setdefault(r.plan.end_key(), []).append(r)
        mode = lambda r: (r.plan.via_proven, r.plan.via_height_m)
        groups = sorted(order.values(), key=lambda g: min(RuleSelector.static_cost(r) for r in g))
        survivors = [r for g in groups for r in sorted(g, key=mode)]
        passed = checked = 0
        path_cache = {}
        for r in survivors:
            key = r.plan.path_key()
            if key not in path_cache:
                if passed >= wanted or checked >= path_budget:
                    r.rejected_at = 'not_path_checked'
                    continue
                checked += 1
                worst, clearance = self.path(r.plan)
                path_cache[key] = (worst, clearance, self.last_path_trace, self.last_block_clearance,
                                   self.last_branch_jump)
            worst, clearance, trace, block_gap, jump = path_cache[key]
            r.path_worst_error_m, r.path_min_clearance_m, r.path_trace = worst, clearance, trace
            r.path_block_clearance_m, r.path_branch_jump_rad = block_gap, jump
            if worst > self.path_tol:
                r.rejected_at = 'entry_path'
            elif jump > self.BRANCH_JUMP_RAD:
                r.rejected_at = 'entry_branch_switch'
            elif clearance < self.path_clearance:
                r.rejected_at = 'entry_clearance'
            elif block_gap < self.BLOCK_CLEARANCE_M:
                r.rejected_at = 'entry_block_contact'
            else:
                passed += 1
        return reports


@dataclass
class Decision:
    kind: str                       # 'EXECUTE' | 'ABORT'
    plan: GraspPlan | None
    reason: str


class RuleSelector:
    """Priority: complete chain feasible > margins > less motion > compute."""

    @staticmethod
    def static_cost(r: CandidateReport):
        p = r.plan
        # quality first (reach errors), then less extra posture (pitch/spread
        # beyond the proven crouch), then entry distance.
        return (r.place_error_m + r.pick_end_error_m + .002*abs(p.phi_deg-60.)/10.
                + .01*max(0., p.crouch_pitch-.4) + .05*max(0., p.knee_spread_m-.06)
                + .2*max(0., p.place_com_m-.07) + .2*max(0., p.place_backoff_m-.045)
                + .02*p.via_height_m + .002*p.via_proven)

    def select(self, observation, reports, history):
        ok = [r for r in reports if r.feasible]
        if not ok:
            causes = sorted({r.rejected_at for r in reports})
            return Decision('ABORT', None, f'no candidate links place, pick and entry path ({", ".join(causes)})')
        best = min(ok, key=lambda r: (r.path_worst_error_m + self.static_cost(r), -r.path_min_clearance_m))
        return Decision('EXECUTE', best.plan,
                        f'all links feasible: place {best.place_error_m*1000:.1f} mm, pick {best.pick_end_error_m*1000:.1f} mm, '
                        f'path worst {best.path_worst_error_m*1000:.1f} mm, clearance {best.path_min_clearance_m*1000:.0f} mm')


def candidate_plans():
    """The small registered set: every entry is executable by FloorTableCycle."""
    # The pinch axis must pass (nearly) over the block's centre of mass. With
    # the middle pad 2 cm behind it (and 2 cm above) the held block was
    # measured pivoting in the pads toward its 45 deg equilibrium (atan(.02/
    # .02)) and falling during the rise; pad torsion does not hold it.
    # Finger angle is a trade-off measured on this scene: fingers near
    # vertical (70-75 deg) enter the crouched reach but cannot hold the block
    # high near the body, so the table-edge crossing is out of reach; 50 deg
    # crosses easily but did not enter that crouch (27+ mm short). The transfer
    # and entry checks decide per scene.
    plans = []
    for phi in (50., 55., 60., 65., 70.):
        for pitch, height, spread in itertools.product((.8, 1., 1.2), (.47, .50), (.06, .10, .13)):
            for via_h, via_out, proven in ((0., 0., False), (0., 0., True), (.08, .02, True)):
                for carry_h, carry_p, com in ((.77, .5, .07), (.72, .5, .07), (.77, .5, .09), (.77, .3, .07)):
                    for cross, lag in ((.03, 0.), (.015, 0.), (.03, .4)):
                        plans.append(GraspPlan(phi, 0., pitch, height, spread, .05, via_h, via_out, proven,
                                               com, .045, carry_h, carry_p, cross, lag))
    return plans


@dataclass
class DecisionLog:
    records: list = field(default_factory=list)

    def add(self, point, observation, reports, decision, replan):
        self.records.append(dict(
            point=point, replan=replan, observation=observation.summary(),
            candidates=sorted([r.summary() for r in reports if r.rejected_at not in ('place_reach', 'pick_end', 'failed_before')],
                              key=lambda c: (c['rejected_at'] is not None, c['rejected_at'] or ''))[:12],
            rejected=_count([r.rejected_at for r in reports]),
            decision=dict(kind=decision.kind, plan=asdict(decision.plan) if decision.plan else None,
                          reason=decision.reason),
            result=None))
        return self.records[-1]

    def close(self, outcome, reason=None, observation=None):
        if self.records and self.records[-1]['result'] is None:
            self.records[-1]['result'] = dict(outcome=outcome, reason=reason,
                                              observation=observation.summary() if observation else None)


def _count(items):
    out = {}
    for i in items:
        out[str(i)] = out.get(str(i), 0) + 1
    return out
