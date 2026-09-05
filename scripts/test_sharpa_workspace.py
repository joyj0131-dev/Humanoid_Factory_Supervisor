#!/usr/bin/env python3
"""Fast geometry/reset tests; physical success is evaluated separately."""
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.envs import task_config as tc
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SharpaBimanualGraspExpert
from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift


def test_invalid_geometry():
    for kwargs in [dict(object_half_extents=(.05, -.1, .05)),
                   dict(object_half_extents=(.05, .1)),
                   dict(object_yaw_rad=float('nan')),
                   dict(object_xy_randomization_m=(-.01, .01))]:
        try:
            GraspEnvConfig(**kwargs)
        except ValueError:
            continue
        raise AssertionError(kwargs)


def test_seeded_reset_and_explicit_override():
    config = GraspEnvConfig(object_xy_randomization_m=(.01, .02),
                           object_yaw_randomization_rad=.1)
    env = SharpaGraspEnv(config)
    try:
        _, first = env.reset(seed=17)
        _, repeat = env.reset(seed=17)
        _, other = env.reset(seed=18)
        key = 'reset_object_xy_yaw'
        np.testing.assert_array_equal(first[key], repeat[key])
        assert not np.array_equal(first[key], other[key]), 'Different seeds must change the scene'
        assert abs(first[key][0] - config.object_pos[0]) <= .01
        assert abs(first[key][1] - config.object_pos[1]) <= .02
        assert abs(first[key][2]) <= .1
        _, explicit = env.reset(seed=99, options={'object_xy': [.275, -.006], 'object_yaw_rad': .08})
        np.testing.assert_allclose(explicit[key], [.275, -.006, .08])
        q = env.data.qpos[env._object_qpos_adr:env._object_qpos_adr + 7]
        np.testing.assert_allclose(q[:2], [.275, -.006])
        np.testing.assert_allclose(q[3:], [np.cos(.04), 0, 0, np.sin(.04)])
    finally:
        env.close()


def test_rectangular_geometry_mass_and_resting_height():
    config = GraspEnvConfig(object_half_extents=(.05, .075, .05), object_yaw_rad=.2)
    env = SharpaGraspEnv(config)
    try:
        _, info = env.reset(seed=0)
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, tc.OBJECT_GEOM)
        np.testing.assert_allclose(env.model.geom_size[gid], [.05, .075, .05])
        np.testing.assert_allclose(info['object_half_extents'], [.05, .075, .05])
        assert env.model.body_mass[env._object_body_id] == config.object_mass
        expected_z = config.table_pos[2] + config.table_half_size[2] + .05 + tc.OBJECT_TABLE_GAP
        assert abs(env.data.qpos[env._object_qpos_adr+2] - expected_z) < 1e-12
        expected_inertia = config.object_mass / 3 * np.array([.075**2+.05**2, .05**2+.05**2, .05**2+.075**2])
        np.testing.assert_allclose(env.model.body_inertia[env._object_body_id], expected_inertia)
    finally:
        env.close()


def test_rotated_separation_and_actual_wrist_support():
    env = SharpaGraspEnv(GraspEnvConfig(object_half_extents=(.05, .075, .05), object_yaw_rad=.3))
    try:
        env.reset(seed=0)
        expert = SharpaBimanualGraspExpert(env)
        pos = expert._object_pos()
        rotation = env.data.xmat[env._object_body_id].reshape(3, 3)
        for side, sign in [('left', 1), ('right', -1)]:
            point = pos + rotation @ np.array([0., sign * (.075 + .012), 0.])
            with patch.object(expert, '_nonthumb_tip_centroid', return_value=point):
                assert abs(expert._precontact_separation_m(side) - .012) < 1e-12
        original_solver = env.model.opt.noslip_iterations
        before = SharpaContactLift.measure_support_forces(env)
        assert all(np.linalg.norm(f) == 0. for f in before.values())
        assert env.model.opt.noslip_iterations == original_solver, 'Census must not change physics'
        wrist = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'left_wrist_yaw_link')
        # Controlled collision fixture only: put the test box around the wrist.
        # No rollout/controller ever assigns the object's pose this way.
        env.data.qpos[env._object_qpos_adr:env._object_qpos_adr+3] = env.data.xpos[wrist]
        mujoco.mj_forward(env.model, env.data)
        bodies = SharpaContactLift.hand_wrist_body_ids(env)
        for side in ('left', 'right'):
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f'{side}_wrist_yaw_link') in bodies[side]
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f'{side}_elbow_link') not in bodies[side]
        whole = SharpaContactLift.measure_support_forces(env)
        hands = SharpaContactLift.measure_support_forces(env, {
            'left': env._left_hand_body_ids, 'right': env._right_hand_body_ids})
        assert np.linalg.norm(whole['left'] - hands['left']) > .01, 'Real wrist force was omitted'
    finally:
        env.close()


if __name__ == '__main__':
    for test in (test_invalid_geometry, test_seeded_reset_and_explicit_override,
                 test_rectangular_geometry_mass_and_resting_height,
                 test_rotated_separation_and_actual_wrist_support):
        test()
        print(f'PASS {test.__name__}')
