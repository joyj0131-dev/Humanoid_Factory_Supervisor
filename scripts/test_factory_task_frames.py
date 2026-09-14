#!/usr/bin/env python3
"""Frame and access contracts; physical mission results are evaluated separately."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch
from humanoid_learning.envs.factory_env import FactoryEnv
from humanoid_learning.envs import factory_config as fc
from humanoid_learning.expert.factory_recovery import FactoryRecovery
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    SharpaBimanualGraspExpert, LOCAL_APPROACH_VEC, LOCAL_CLOSING_VEC,
)
from humanoid_learning.expert.whole_body_posture import WholeBodyPosture
from humanoid_learning.expert.factory_floor_pickup import FactoryFloorPickup
from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift


def main():
    env = FactoryEnv()
    try:
        env.reset(seed=0)
        recovery = FactoryRecovery(env)
        base = SharpaBimanualGraspExpert(recovery.grasp)
        obj = base._object_pos()
        p = base._mirrored_targets(.1, .08, .15)
        for heading in (np.pi, -1.1, 1.13):
            rotated = SharpaBimanualGraspExpert(recovery.grasp, heading=heading)
            r = rotated.task_rotation
            actual = rotated._mirrored_targets(.1, .08, .15)
            for side in ('left', 'right'):
                np.testing.assert_allclose(actual[side], obj + r @ (p[side] - obj), atol=1e-12)
                np.testing.assert_allclose(rotated._inward_direction(side), r @ base._inward_direction(side))
                original_R = base._facing_rotation(side, p[side], obj, np.array([1., 0., 0.]))
                rotated_R = rotated._facing_rotation(side, actual[side], obj, r[:, 0])
                np.testing.assert_allclose(rotated_R, r @ original_R, atol=1e-12)
        print('PASS heading-equivariant position, inward servo and orientation')
        pose = env.poses[0]
        gid = [i for i in range(env.model.ngeom) if (env.model.geom(i).name or '').startswith('wc0_rail_p_geom')]
        assert len(gid) == 2
        for i in gid:
            center, half = env.data.geom_xpos[i, 1], env.model.geom_size[i, 1]
            assert center + half <= pose.infeed_xy[1] - .39 or center - half >= pose.pick_xy[1] + .44
            assert env.model.geom_contype[i] != 0
        assert env.model.geom('wc0_rail_m_geom').id >= 0
        print('PASS physical access opening, other guards retained and collidable')
        posture = WholeBodyPosture(recovery.grasp)
        before = env.data.qpos.copy()
        target = [recovery.grasp.palm_pose(s)[0] for s in ('left','right')]
        q = posture.solve(posture.start_height-.02, target, iterations=30)
        np.testing.assert_array_equal(before, env.data.qpos)
        assert q[3] > 0 and q[9] > 0
        print('PASS posture IK uses scratch only and forward-bending knees')
        physical_margin = env.model.geom_margin.copy()
        planned = WholeBodyPosture(recovery.grasp, clearance_m=.008)
        np.testing.assert_array_equal(physical_margin, env.model.geom_margin)
        assert planned.ik_model is not env.model
        assert np.max(planned.ik_model.geom_margin-physical_margin) >= .008
        print('PASS anticipatory clearance belongs to planning model only, live collisions unchanged')
        # Re-anchoring IK at the crouch must not redefine standing height.
        floor = FactoryFloorPickup(SimpleNamespace(grasp=recovery.grasp, env=env,
                                                   station=0, expert=base,
                                                   stabilizer=recovery.stabilizer))
        for i, side in enumerate(('left', 'right')):
            assert np.dot(floor.goal_R[i] @ LOCAL_APPROACH_VEC, [0., 0., -1.]) > .98
            assert np.dot(floor.goal_R[i] @ LOCAL_CLOSING_VEC, base._inward_direction(side)) > .98
            assert floor.goal[i,2] > floor.obj[2] + .15
        print('PASS floor-specific high-wrist approach, fingertips down and closure inward')
        before_q, before_v = env.data.qpos.copy(), env.data.qvel.copy()
        before_force, before_external = env.data.qfrc_applied.copy(), env.data.xfrc_applied.copy()
        measured_palms = np.array([recovery.grasp.palm_pose(s)[0] for s in ('left', 'right')])
        floor.start[:] += .01  # stale handoff cache, not a live robot pose edit
        action = floor.step()
        np.testing.assert_array_equal(floor.start, measured_palms)
        assert action.shape == (25,) and np.isfinite(action).all()
        floor.begin_contact()
        action = floor._contact_action({'left': np.zeros(3), 'right': np.zeros(3)})
        assert np.max(np.abs(action[3:17])) <= .003/recovery.grasp.config.arm_action_scale + 1e-12
        np.testing.assert_array_equal(before_q, env.data.qpos)
        np.testing.assert_array_equal(before_v, env.data.qvel)
        np.testing.assert_array_equal(before_force, env.data.qfrc_applied)
        np.testing.assert_array_equal(before_external, env.data.xfrc_applied)
        print('PASS floor control commands joints only, no live pose or external-force substitution')
        standing_height = floor.standing_height
        captured = {}
        def solve(height, *args, **kwargs):
            captured['height'] = height
            return q
        floor.posture = SimpleNamespace(start_height=.30, solve=solve,
                                        command=lambda q: np.zeros(25), qadr=posture.qadr,
                                        pelvis=posture.pelvis)
        floor.stage, floor.tick = 'RISE', 1599
        floor.rise_base_height, floor.rise_pitch = .30, .20
        floor.lift_start, floor.lift_R = floor.start.copy(), floor.start_R
        with patch.object(SharpaContactLift, 'measure_support_forces',
                          return_value={'left': np.array([0., -2., 0.]),
                                        'right': np.array([0., 2., 0.])}), \
             patch.object(floor, '_contact_action', return_value=np.zeros(25)):
            floor.step()
        assert abs(captured['height'] - standing_height) < 1e-12
        print('PASS floor rise retains original standing height after crouch re-anchor')
        assert recovery._floor_support_continuity([200., 200.])
        assert recovery._floor_support_continuity([0., 400.])
        for _ in range(5):
            supported = recovery._floor_support_continuity([0., 400.])
        assert not supported
        assert recovery._floor_support_continuity([200., 200.])
        assert recovery._floor_support_continuity([0., 0.])
        for _ in range(5):
            supported = recovery._floor_support_continuity([0., 0.])
        assert not supported
        print('PASS floor support permits brief contact chatter, rejects sustained loss')
        recovery.close()
    finally:
        env.close()


if __name__ == '__main__':
    main()
