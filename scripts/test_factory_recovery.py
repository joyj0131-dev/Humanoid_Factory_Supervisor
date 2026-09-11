#!/usr/bin/env python3
"""Fast integration-contract checks; these do NOT claim a successful mission."""
from pathlib import Path
import sys
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from humanoid_learning.envs.factory_env import FactoryEnv
from humanoid_learning.envs import factory_config as fc
from humanoid_learning.expert.factory_recovery import FactoryRecovery
from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    BimanualGraspConfig, BimanualGraspState, SharpaBimanualGraspExpert,
)


def test_prepared_resume_guard_without_physics_writes():
    # Synthetic state tests the transition guard only, not physical reachability.
    expert = object.__new__(SharpaBimanualGraspExpert)
    expert.config = BimanualGraspConfig()
    target = np.r_[expert._clearance_arm_vector('left'),
                   expert._clearance_arm_vector('right')]
    data = SimpleNamespace(qpos=target.copy(), qvel=np.zeros(17))
    expert.env = SimpleNamespace(
        data=data, _waist_target=np.zeros(3), _arm_qpos_adr=np.arange(14),
        _arm_dof_adr=np.arange(14), _waist_dof_adr=np.arange(14, 17),
        _torso_arm_collision_force=lambda: 0., _hand_hand_contact_force=lambda: 0.)
    before = data.qpos.copy()
    with patch.object(expert, '_advance') as advance:
        assert expert.resume_prepared_approach()
        advance.assert_called_once_with(BimanualGraspState.FOREARM_FORWARD_REACH)
        np.testing.assert_array_equal(data.qpos, before)
        advance.reset_mock()
        data.qvel[0] = 0.1
        assert not expert.resume_prepared_approach()
        data.qvel[0] = 0.
        data.qpos[0] += 1.
        assert not expert.resume_prepared_approach()
        data.qpos[:] = before
        expert.env._torso_arm_collision_force = lambda: 100.
        assert not expert.resume_prepared_approach()
        advance.assert_not_called()


def test_shared_view_is_read_only_at_construction():
    env = FactoryEnv()
    try:
        env.reset(seed=0)
        qpos, qvel, time = env.data.qpos.copy(), env.data.qvel.copy(), env.data.time
        recovery = FactoryRecovery(env)
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.qvel, qvel)
        assert env.data.time == time
        view = recovery._grasp_view(1)
        assert view.model is env.model and view.data is env.data
        assert view._object_body_id == env.model.body(fc.part_body_name(1)).id
        try:
            view.reset()
        except RuntimeError:
            pass
        else:
            raise AssertionError('shared reset must be rejected')
        recovery.close()
    finally:
        env.close()


def test_one_physics_advance_and_no_pose_writes():
    env = FactoryEnv()
    try:
        env.reset(seed=0)
        recovery = FactoryRecovery(env)
        before = env.data.qpos.copy()
        calls = 0
        original_step = mujoco.mj_step

        def checked_step(model, data, *args, **kwargs):
            nonlocal before, calls
            np.testing.assert_array_equal(data.qpos, before)
            original_step(model, data, *args, **kwargs)
            before = data.qpos.copy()
            calls += 1

        start = env.data.time
        with patch.object(mujoco, 'mj_step', checked_step):
            for _ in range(20):
                recovery.step()
        assert calls == 20 * env.config.frame_skip
        assert env._step_count == recovery.grasp._step_count == 20
        assert abs(env.data.time - start - calls * env.model.opt.timestep) < 1e-10
        assert recovery.state == 'WAIT_FAULT' and recovery.station is None
        recovery.close()
    finally:
        env.close()


def test_on_belt_is_not_a_lift_and_terminal_does_not_step():
    env = FactoryEnv()
    try:
        env.reset(seed=0)
        recovery = FactoryRecovery(env)
        recovery.station = 0
        clearance, supported = recovery.lift_evidence()
        assert clearance < recovery.config.lift_clearance_m
        assert not supported
        # Even invented opposing hand-force readings cannot bypass clearance.
        with patch.object(SharpaContactLift, 'measure_support_forces', return_value={
                'left': np.array([0., 2., 0.]), 'right': np.array([0., -2., 0.])}):
            clearance, _ = recovery.lift_evidence()
            assert clearance < recovery.config.lift_clearance_m
        recovery._transition('FAILED')
        t = env.data.time
        recovery.step()
        assert env.data.time == t
        recovery.close()
    finally:
        env.close()


def test_reset_restores_tuning_and_preserves_baseline_config():
    env = FactoryEnv()
    try:
        gain, bias = env.model.actuator_gainprm.copy(), env.model.actuator_biasprm.copy()
        recovery = FactoryRecovery(env)
        recovery.walker._apply_policy_gains()
        recovery.close()
        np.testing.assert_array_equal(env.model.actuator_gainprm, gain)
        np.testing.assert_array_equal(env.model.actuator_biasprm, bias)
        assert BimanualGraspConfig().palm_first_closure is True
        assert BimanualGraspConfig().hold_squeeze_m == 0.003
        assert BimanualGraspConfig().contact_settle_grace_seconds == 0.5
    finally:
        env.close()


if __name__ == '__main__':
    tests = [v for k, v in globals().copy().items() if k.startswith('test_')]
    failed = 0
    for test in tests:
        try:
            test()
            print('PASS', test.__name__, flush=True)
        except Exception as exc:
            failed += 1
            print('FAIL', test.__name__, repr(exc), flush=True)
    print(f'{len(tests) - failed}/{len(tests)} contract tests passed; mission success is evaluated separately.')
    raise SystemExit(bool(failed))
