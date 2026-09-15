#!/usr/bin/env python3
"""Planning/filter and pad-surface contracts, not a physical lift success test."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from humanoid_learning.envs.factory_env import FactoryEnv
from humanoid_learning.expert.factory_recovery import FactoryRecovery, RecoveryConfig
from humanoid_learning.expert.factory_floor_pickup import FactoryFloorPickup
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SharpaBimanualGraspExpert
from humanoid_learning.expert.pose_ik import so3_exp
from humanoid_learning.expert.whole_body_posture import WholeBodyPosture
from humanoid_learning.expert.sharpa_pad_grasp import SharpaPadGrasp, upper_side_patch


def main():
    # Six resting faces: "up" must come from world geometry, not local Z.
    for up in range(3):
        for sign in (-1., 1.):
            R = np.roll(np.eye(3), 2-up, axis=0)
            R[2] *= sign
            if np.linalg.det(R) < 0.:
                R[0] *= -1
            point, axis = upper_side_patch(R, np.full(3, .06), R.T @ [0., 0., .08])
            assert axis == up
            np.testing.assert_allclose(R @ point, [0., 0., .042], atol=1e-12)
    print('PASS upper grasp patch follows world up for all six resting faces')
    env = FactoryEnv()
    try:
        env.reset(seed=0)
        recovery = FactoryRecovery(env)
        grasp = recovery.grasp
        m, d = env.model, env.data
        before = {name: getattr(m, name).copy() for name in
                  ('geom_contype', 'geom_conaffinity', 'geom_margin', 'geom_friction',
                   'body_mass', 'body_inertia')}
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
        planning = WholeBodyPosture(grasp, .008)
        pads = SharpaPadGrasp(grasp)
        left, right = pads.geoms['left', 'middle'], pads.geoms['right', 'middle']
        other = pads.geoms['left', 'index']
        torso = next(i for i in range(m.ngeom) if
                     m.geom_bodyid[i] == m.body('torso_link').id and m.geom_contype[i])
        def allowed(model, a, b):
            return bool((model.geom_contype[a] & model.geom_conaffinity[b]) or
                        (model.geom_contype[b] & model.geom_conaffinity[a]))
        assert not allowed(planning.ik_model, left, other)
        assert allowed(planning.ik_model, left, right)
        assert allowed(planning.ik_model, left, torso)
        assert allowed(m, left, other)
        # At the SAME contact point rigid hand motion has identical Jacobians:
        # the removed constraint cannot be changed by any optimized variable.
        for side in ('left', 'right'):
            point = pads.point(side, 'middle')
            ja, jb = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, ja, None, point, pads.bodies[side, 'middle'])
            mujoco.mj_jac(m, d, jb, None, point, pads.bodies[side, 'index'])
            np.testing.assert_allclose((ja-jb)[:, planning.cols], 0., atol=1e-12)
            assert np.linalg.norm(ja-jb) > 0.  # finger variables do differ
            for body in getattr(grasp, f'_{side}_hand_body_ids'):
                mujoco.mj_jac(m, d, jb, None, point, body)
                np.testing.assert_allclose((ja-jb)[:, planning.cols], 0., atol=1e-12)
        print('PASS only uncontrollable same-hand collision rows omitted from scratch IK')
        pads.apply_shape(0.)
        for side in ('left', 'right'):
            for group in range(1, 4):
                np.testing.assert_array_equal(grasp._group_close[side][group], pads.original_close[side][group])
        pads.apply_shape(1.)
        for side in ('left', 'right'):
            for group in range(1, 4):
                assert np.max(grasp._group_close[side][group]) <= .60
                np.testing.assert_allclose(grasp._group_close[side][group].reshape(-1, 3)[:, 1:],
                                           np.tile([.10, .08], (len(grasp._group_close[side][group])//3, 1)))
            for finger in ('index', 'middle', 'ring', 'pinky'):
                assert pads.points[side, finger][1] > 0.
                assert np.isclose(np.linalg.norm(pads.normal(side, finger)), 1.)
                assert m.geom(pads.geoms[side, finger]).name.endswith('_elastomer')
        print('PASS actual rubber-face geometry and near-straight finger command mapping')
        # A large hard-shell contact must not masquerade as rubber support.
        obj = m.geom(grasp.object_geom_name).id
        hard = m.geom('left_left_middle_DP').id
        contacts = [SimpleNamespace(geom1=obj, geom2=gid, frame=np.eye(3).ravel())
                    for gid in (hard, left)]
        pads.env = SimpleNamespace(model=m, data=SimpleNamespace(contact=contacts),
                                   _object_body_id=grasp._object_body_id)
        def force(model, data, index, output):
            output[:] = 0.
            output[0] = 100. if index == 0 else 3.
        with patch.object(mujoco, 'mj_contactForce', side_effect=force):
            np.testing.assert_allclose(pads.forces()['left'], [-3., 0., 0.])
            np.testing.assert_array_equal(pads.forces()['right'], np.zeros(3))
        pads.env = grasp
        print('PASS hard-shell pressure cannot satisfy the rubber-pad support signal')
        for name, old in before.items():
            np.testing.assert_array_equal(getattr(m, name), old)
        np.testing.assert_array_equal(d.qpos, qpos)
        np.testing.assert_array_equal(d.qvel, qvel)
        print('PASS live collision masks/materials/mass/pose unchanged')
        recovery.close()
        recovery = FactoryRecovery(env, RecoveryConfig(floor_pad_grip=True))
        grasp = recovery.grasp
        recovery.station = 0
        recovery.expert = SharpaBimanualGraspExpert(recovery.grasp)
        floor = FactoryFloorPickup(recovery)
        recovery.grasp._group_synergy[:] = .8
        floor.begin_contact()
        action = floor.step()
        # The unused thumb is parked, not extended into the table; the other
        # fingers remain near-straight and do NOT inherit this .7 thumb curl.
        assert action[17] < 0. and action[21] < 0.
        for side in ('left', 'right'):
            for group in range(1, 4):
                assert np.max(recovery.grasp._group_close[side][group]) <= .60
        print('PASS parked thumb is independent of extended gripping fingers')
        floor.side_entry_ready = {'left': True, 'right': True}
        forces = {side: -4.*floor._side_face(side)[2] for side in ('left', 'right')}
        floor.stage = 'RISE'
        floor.lift_R = floor.close_hand_rotations
        forward = recovery.expert.task_rotation[:, 0]
        action = floor._contact_action(forces)
        np.testing.assert_allclose(action[3:17], 0., atol=1e-10)
        # A stored orientation must not move a loaded hand inside its force
        # band. Outside that band, physical pressure feedback remains active.
        floor.close_hand_rotations = [so3_exp(np.array([.02, 0., 0.]))@R
                                       for R in floor.close_hand_rotations]
        np.testing.assert_allclose(floor._contact_action(forces)[3:17], 0., atol=1e-10)
        corrected = floor._contact_action({s: .1*f for s, f in forces.items()})
        assert np.linalg.norm(corrected[3:17]) > 1e-5
        assert np.max(np.abs(corrected[3:17])) <= .003/recovery.grasp.config.arm_action_scale + 1e-12
        print('PASS loaded pad targets hold in-band and correct low pressure')
        # Initial lift must use the same soft joint costs as pad squeezing.
        # Otherwise the two summed controllers oppose each other's posture.
        floor.pick_rotations = [grasp.palm_pose(s)[1] for s in ('left', 'right')]
        lifted = floor._pick_clear_action()
        weights = np.array([1., .2, .3, 1., .5, 1., .5])
        delta = np.r_[forward*(-.00004)+[0., 0., .00008], np.zeros(3)]
        for i in range(2):
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, floor.posture.palms[i])
            columns = grasp._arm_dof_adr[7*i:7*i+7]
            jac = np.vstack([jp[:, columns], .2*jr[:, columns]])*weights
            expected = weights*(jac.T @ np.linalg.solve(jac@jac.T+1e-4*np.eye(6), delta))
            np.testing.assert_allclose(lifted[7*i:7*i+7], expected/grasp.config.arm_action_scale,
                                       atol=1e-12)
        print('PASS initial pad lift uses the squeeze controller joint preference')
        recovery.close()
    finally:
        env.close()


if __name__ == '__main__':
    main()
