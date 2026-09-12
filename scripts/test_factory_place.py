"""Place prototype contract tests; these are not completed-repair claims."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_learning.envs.factory_env import FactoryEnv
from humanoid_learning.expert.factory_recovery import FactoryRecovery, RecoveryConfig
from humanoid_learning.expert.factory_place import FactoryPlace
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SharpaBimanualGraspExpert


def test_place_frame_roundtrip_and_no_pose_writes():
    env = FactoryEnv()
    try:
        env.reset(seed=0)
        recovery = FactoryRecovery(env)
        recovery.station = 0
        recovery.expert = SharpaBimanualGraspExpert(recovery.grasp)
        before = env.data.qpos.copy()
        place = FactoryPlace(recovery)
        with patch.object(place, '_solve') as solve:
            place._move(place.start, place.start_R)
            targets, rotations = solve.call_args.args
            for side in ('left', 'right'):
                p, R = recovery.grasp.palm_pose(side)
                np.testing.assert_allclose(targets[side], p, atol=1e-12)
                np.testing.assert_allclose(rotations[side], R, atol=1e-12)
        np.testing.assert_array_equal(env.data.qpos, before)
        assert not env.task_manager.completion_requested
        # Even an exact target position must not trigger release in mid-air.
        place.stage, place.tick = 'LOWER', place.config.lower_ticks - 1
        with patch.object(place, '_move'), patch.object(place, '_object_supported_by_belt', return_value=False), \
                patch.object(env, 'part_position', return_value=place.target.copy()):
            place.step()
            assert place.stage == 'LOWER'
            place.tick = place.config.lower_ticks + place.config.settle_ticks
            place.step()
            assert place.failure == 'PLACE_SUPPORT_NOT_ESTABLISHED'
        assert not env.task_manager.completion_requested
        np.testing.assert_array_equal(env.data.qpos, before)
        assert RecoveryConfig().place_after_lift is False
        recovery.close()
    finally:
        env.close()


if __name__ == '__main__':
    test_place_frame_roundtrip_and_no_pose_writes()
    print('PASS place frame roundtrip / live-state preservation / opt-in contract')
