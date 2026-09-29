#!/usr/bin/env python3
"""Contracts of the observe/select/execute/verify loop around the floor->table
cycle. Selection and scratch checks only: not a physical pick/place test."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from humanoid_learning.envs.factory_env import FactoryEnv
from humanoid_learning.expert.factory_recovery import FactoryRecovery, RecoveryConfig
from humanoid_learning.expert.factory_floor_table import FloorTableCycle, PROVEN_TICKS, LOCAL_TICKS
from humanoid_learning.expert.recovery_agent import (
    CandidateEvaluator, CandidateReport, DecisionLog, GraspPlan, RuleSelector, candidate_plans, observe)
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SharpaBimanualGraspExpert


def _cycle(env):
    recovery = FactoryRecovery(env, RecoveryConfig(floor_pad_grip=True, floor_frog_stance=True,
                                                   floor_table_place=True))
    recovery.station, recovery.floor_task = 1, True
    recovery.grasp = recovery._grasp_view(1)
    recovery.stabilizer.env = recovery.grasp
    recovery.expert = SharpaBimanualGraspExpert(recovery.grasp)
    cycle = FloorTableCycle(recovery)
    cycle.stance_knee_lateral = list(cycle.posture.knee_lateral)
    cycle.stance_foot_rotations = [R.copy() for R in cycle.posture.foot_rotations]
    cycle.lower_start_height, cycle.lower_start_pitch = cycle.posture.start_height, .4
    return recovery, cycle


def main():
    plans = candidate_plans()
    assert len({p.key() for p in plans}) == len(plans)
    # Measured: a pinch axis 1-2 cm off the block's centre of mass let the
    # held block pivot in the pads; only (nearly) centred pinches are offered.
    assert all(abs(p.forward_m) <= .01 for p in plans)
    print(f'PASS {len(plans)} registered plans, unique, pinch axis over the centre of mass')

    selector = RuleSelector()
    bad = [CandidateReport(GraspPlan(phi_deg=70.), rejected_at='transfer_reach'),
           CandidateReport(GraspPlan(phi_deg=55.), rejected_at='pick_end')]
    decision = selector.select(None, bad, [])
    assert decision.kind == 'ABORT' and decision.plan is None
    assert 'transfer_reach' in decision.reason and 'pick_end' in decision.reason
    good = CandidateReport(GraspPlan(phi_deg=60.), .002, .002, .01, .005)
    better_but_rejected = CandidateReport(GraspPlan(phi_deg=65.), .0, .0, .0, .01, rejected_at='entry_branch_switch')
    decision = selector.select(None, [better_but_rejected, good], [])
    assert decision.kind == 'EXECUTE' and decision.plan == good.plan
    print('PASS selector executes only fully linked candidates and aborts with the causes otherwise')

    env = FactoryEnv()
    try:
        env.reset(seed=0, options={'fault_workcell': 1})
        m, d = env.model, env.data
        recovery, cycle = _cycle(env)
        before = {name: getattr(m, name).copy() for name in
                  ('geom_contype', 'geom_conaffinity', 'geom_margin', 'geom_friction', 'body_mass')}
        qpos, qvel, ctrl = d.qpos.copy(), d.qvel.copy(), d.ctrl.copy()

        plan = GraspPlan(60., 0., 1.2, .5, .1, .05, via_proven=True)
        goal = cycle.entry_targets(plan)
        palms0, _, _, _, _ = cycle.entry_waypoint(plan, 0., goal)
        palms_split, _, _, _, _ = cycle.entry_waypoint(plan, PROVEN_TICKS/(PROVEN_TICKS+LOCAL_TICKS), goal)
        palms1, rot1, height1, pitch1, spread1 = cycle.entry_waypoint(plan, 1., goal)
        np.testing.assert_allclose(palms0, cycle.start, atol=1e-12)
        np.testing.assert_allclose(palms_split, cycle.proven_goal[0], atol=1e-12)
        np.testing.assert_allclose(palms1, goal[0], atol=1e-12)
        assert (height1, pitch1, spread1) == (plan.crouch_height, plan.crouch_pitch, plan.knee_spread_m)
        print('PASS entry path starts at the crouch, passes the proven reach pose, ends at the entry pose')

        _, feet, heading = cycle.place_target_estimate()
        tf = cycle.transfer_frame(plan, cycle._block()[0], .5, 1.2, feet[:2], np.zeros(2), np.ones(2), feet, heading)
        s_rise, s_cross, s_over, s_spot = FloorTableCycle.TRANSFER_S
        start, h0, p0, _, k0 = cycle.transfer_waypoint(tf, 0.)
        risen, hr, pr, _, kr = cycle.transfer_waypoint(tf, s_rise)
        cross, _, _, _, _ = cycle.transfer_waypoint(tf, s_cross)
        over, ho, po, _, _ = cycle.transfer_waypoint(tf, s_over)
        spot, _, _, _, _ = cycle.transfer_waypoint(tf, s_spot)
        end, h1, p1, _, k1 = cycle.transfer_waypoint(tf, 1.)
        for got, key in ((start, 'L'), (risen, 'H'), (cross, 'A'), (over, 'O'), (spot, 'B'), (end, 'C')):
            np.testing.assert_allclose(got, tf[key], atol=1e-12)
        assert (h0, p0) == (.5, 1.2)
        _, hc, pc, _, _ = cycle.transfer_waypoint(tf, s_cross)
        assert (hr, pr) == (hc, pc) == (plan.carry_height, plan.carry_pitch)   # upright up to the crossing
        assert (ho, po) == (h1, p1) == (plan.place_height, plan.place_pitch)   # leaning once over the table
        np.testing.assert_allclose(k0, 0.) and np.testing.assert_allclose(k1, 1.)
        _, top, face = cycle._table(heading)
        assert tf['A'] @ heading[:, 0] + .06 <= face + 1e-9   # still short of the table face
        assert tf['A'][2] - .06 >= top                       # bottom above the table top
        print('PASS transfer path: rise with the block low, raise short of the table face, over, lean, down')

        evaluator = CandidateEvaluator(cycle)
        evaluator.place_error(plan, None, feet, heading)
        evaluator.pick_end_error(plan)
        evaluator.path(plan, samples=4, iterations=2)
        evaluator.transfer(plan, samples=4, iterations=2)
        evaluator.calibrate()
        for name, old in before.items():
            np.testing.assert_array_equal(getattr(m, name), old)
        np.testing.assert_array_equal(d.qpos, qpos)
        np.testing.assert_array_equal(d.qvel, qvel)
        np.testing.assert_array_equal(d.ctrl, ctrl)
        print('PASS candidate checks run on scratch copies: live pose, commands and model unchanged')

        log = DecisionLog()
        record = log.add('PICK', observe(cycle), [CandidateReport(plan)], selector.select(None, [], []), False)
        assert record['decision']['kind'] == 'ABORT' and record['result'] is None
        log.close('FAILED', 'TEST_CAUSE')
        assert log.records[-1]['result']['reason'] == 'TEST_CAUSE'
        np.testing.assert_array_equal(d.qpos, qpos)
        print('PASS decision records keep observation, candidates, choice and the later outcome')
        recovery.close()
    finally:
        env.close()

    # Verification treats the unmarked cube by symmetry: any face flat counts,
    # a real tilt still does not. (Test-only pose writes on a fresh env.)
    import mujoco
    from humanoid_learning.envs import factory_config as fc
    from humanoid_learning.expert.pose_ik import so3_exp
    env = FactoryEnv()
    try:
        env.reset(seed=0, options={'fault_workcell': 1})
        m, d = env.model, env.data
        adr = m.jnt_qposadr[m.joint(fc.part_joint_name(1)).id]
        rest = env._part_rest_pos(1) - np.array([0., 0., .003])  # rest height sits 2 mm above the top
        def place(R):
            q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel())
            d.qpos[adr:adr+3] = rest; d.qpos[adr+3:adr+7] = q
            d.qvel[:] = 0.; mujoco.mj_forward(m, d)
            return env.task_manager._part_is_back(env)
        env.task_manager.fault_station = 1
        assert place(np.eye(3))
        assert place(so3_exp(np.array([np.pi, 0., 0.])))          # upside down: another flat face
        assert place(so3_exp(np.array([0., np.pi/2, 0.])))        # on its side
        assert not place(so3_exp(np.array([np.radians(20.), 0., 0.])))  # tilted
        print('PASS cube verification accepts any flat face, rejects a tilted cube')
    finally:
        env.close()


if __name__ == '__main__':
    main()
