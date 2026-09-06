#!/usr/bin/env python3
"""Command/probe isolation and archive integrity tests, with optional real ablation."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from humanoid_learning.envs.grasp_config import GraspEnvConfig
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.envs.sharpa_command import SharpaGraspCommand
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SharpaBimanualGraspExpert
from humanoid_learning.data.sharpa_demo import collect_episode, replay_episode, targets

CASE = dict(name='short-test', x=.27, y=0., yaw_deg=0., size=[.12]*3, seed=0)


def test_probe_has_no_live_physics_or_controller_effect():
    envs = [SharpaGraspEnv(GraspEnvConfig(arm_gravity_compensation=True)) for _ in range(2)]
    try:
        for env in envs:
            env.reset(seed=0)
            for _ in range(10):
                env.step(np.zeros(25))
        expert = SharpaBimanualGraspExpert(envs[0])
        expert._empirical_group_closure_world('left', (1, 2, 3), .15)
        for name in ('qpos', 'qvel', 'ctrl', 'qacc', 'qacc_warmstart', 'qfrc_bias'):
            np.testing.assert_array_equal(getattr(envs[0].data, name), getattr(envs[1].data, name))
        assert envs[0].data.time == envs[1].data.time
        assert envs[0]._step_count == envs[1]._step_count
        np.testing.assert_array_equal(targets(envs[0]), targets(envs[1]))
        for _ in range(5):
            for env in envs:
                env.step(np.zeros(25))
            np.testing.assert_array_equal(envs[0].data.qpos, envs[1].data.qpos)
    finally:
        for env in envs:
            env.close()


def test_complete_command_and_validation():
    env = SharpaGraspEnv()
    try:
        env.reset(seed=0)
        env.set_preshape('left', 1.)
        env.model.opt.noslip_iterations = 10
        command = env.capture_command(np.zeros(25))
        assert command.preshape_targets.shape == (16,)
        env.reset(seed=0)
        env.step_command(command)
        assert env.model.opt.noslip_iterations == 10
        ids = np.concatenate([env._preshape_act_ids[s] for s in ('left','right')])
        np.testing.assert_array_equal(env.data.ctrl[ids], command.preshape_targets)
        try:
            command.action[0] = 1
        except ValueError:
            pass
        else:
            raise AssertionError('recorded command must be immutable')
        before = env.data.ctrl.copy()
        for bad in (SharpaGraspCommand(np.zeros(25), np.zeros(1), 0),
                    SharpaGraspCommand(np.zeros(25), np.full(16, 100.), 0)):
            try:
                env.step_command(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid command accepted')
            np.testing.assert_array_equal(env.data.ctrl, before)
    finally:
        env.close()


def test_short_archive_replays_without_expert_and_cannot_claim_success():
    with tempfile.TemporaryDirectory(prefix='sharpa-demo-test-') as directory:
        path = Path(directory) / 'episode.npz'
        result = collect_episode(path, CASE, max_steps=8)
        assert not result['physical_success']
        with np.load(path, allow_pickle=False) as archive:
            assert archive['observations'].shape == (9,129)
            assert archive['actions'].shape == (8,25)
            assert archive['preshape'].shape == (8,16)
        with patch.object(SharpaBimanualGraspExpert, '__init__', side_effect=AssertionError('Expert must not be used')):
            replay = replay_episode(path)
        assert replay['replay_matches'] and not replay['passed']
        try:
            collect_episode(path, CASE, max_steps=8)
        except FileExistsError:
            pass
        else:
            raise AssertionError('existing demonstration overwritten')


def test_full_episode_and_missing_commands(episode):
    # Remove only auxiliary hand controls from a copy of a real successful demo.
    # This is deliberately corrupted test data, never a training demonstration.
    with patch.object(SharpaBimanualGraspExpert, '__init__', side_effect=AssertionError('Expert must not be used')):
        result = replay_episode(episode)
    assert result['passed'], result
    with np.load(episode, allow_pickle=False) as archive:
        data = {k:archive[k] for k in archive.files}
    data['preshape'] = np.zeros_like(data['preshape'])
    with tempfile.TemporaryDirectory(prefix='sharpa-demo-ablation-') as directory:
        path = Path(directory) / 'missing_preshape.npz'
        np.savez_compressed(path, **data)
        result = replay_episode(path)
    assert not result['replay_matches'] and not result['passed'], result
    print('Missing-command ablation correctly rejected:', json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path)
    args = parser.parse_args()
    for test in (test_probe_has_no_live_physics_or_controller_effect, test_complete_command_and_validation,
                 test_short_archive_replays_without_expert_and_cannot_claim_success):
        test()
        print('PASS', test.__name__, flush=True)
    if args.episode:
        test_full_episode_and_missing_commands(args.episode)
        print('PASS test_full_episode_and_missing_commands', flush=True)
