#!/usr/bin/env python3
"""End-to-end physical lift regression, including release (no hidden attachment).

OPENBLAS_NUM_THREADS=1 python3 scripts/test_sharpa_grasp_lift.py --seeds 0 1 2
MUJOCO_GL=egl python3 scripts/test_sharpa_grasp_lift.py --seeds 0 --render-dir results/sharpa_lift
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    SharpaBimanualGraspExpert, BimanualGraspState as State)
from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift


def snapshot(env, directory, label):
    if directory is None:
        return
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(env.model, height=480, width=640) as renderer:
        for name, elevation in [('front', -15), ('top', -55)]:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [.27, 0, .86]
            camera.distance, camera.azimuth, camera.elevation = 1.1, 180, elevation
            renderer.update_scene(env.data, camera)
            Image.fromarray(renderer.render()).save(directory / f'{label}_{name}.png')


def verify(seed, render_dir=None):
    env = SharpaGraspEnv(GraspEnvConfig(arm_gravity_compensation=True,
                                      sharpa_visual_style='g1', max_episode_steps=8000))
    env.reset(seed=seed)
    expert = SharpaBimanualGraspExpert(env)
    original_solver = env.model.opt.noslip_iterations
    diagnostic = SharpaContactLift(expert)
    assert not diagnostic.supported(), 'An untouched box is not a grasp'
    assert abs(diagnostic.clearance()) < .003, 'Reset box must be on the table'
    env.reset(seed=seed)
    assert env.model.opt.noslip_iterations == original_solver
    expert = SharpaBimanualGraspExpert(env)
    mass = env.model.body_mass.copy()
    friction = env.model.geom_friction.copy()
    constraints = env.model.neq
    initial_z = expert._object_pos()[2]
    previous = None
    peak_torso = 0.0
    max_penetration = 0.0
    air_table_force = 0.0
    for tick in range(8000):
        _, _, term, trunc, info = env.step(expert.step())
        assert not term and not trunc and not info.get('unstable'), (tick, info)
        if expert._contact_lift is not None:
            peak_torso = max(peak_torso, env._torso_arm_collision_force())
            for i in range(env.data.ncon):
                c = env.data.contact[i]
                if env._object_body_id in (env.model.geom_bodyid[c.geom1], env.model.geom_bodyid[c.geom2]):
                    max_penetration = max(max_penetration, -float(c.dist))
                if expert.state == State.AIR_HOLD and set((c.geom1, c.geom2)) == set((
                        expert._contact_lift.table_geom, expert._contact_lift.object_geom)):
                    f = np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, i, f)
                    air_table_force = max(air_table_force, np.linalg.norm(f[:3]))
        if expert.state != previous:
            print(f'seed={seed} tick={tick} {expert.state.name}', flush=True)
            if expert.state in (State.FORCE_SETTLE, State.LIFT, State.AIR_HOLD, State.SUCCESS, State.FAILURE):
                snapshot(env, render_dir, expert.state.name.lower())
            previous = expert.state
        if expert.state in (State.SUCCESS, State.FAILURE):
            break
    assert expert.state == State.SUCCESS, expert.failure_reason
    lift = expert._contact_lift
    assert lift.supported() and lift.clearance() >= .05
    assert expert._air_hold_steps * lift.dt >= 5.0
    assert expert._min_air_clearance >= .05
    assert air_table_force == 0.0, 'Table must not support the airborne box'
    assert np.array_equal(mass, env.model.body_mass)
    assert np.array_equal(friction, env.model.geom_friction)
    assert env.model.neq == constraints, 'No weld/equality attachment may be introduced'
    result = dict(seed=seed, terminal=expert.state.name, steps=tick+1,
                  object_center_rise_m=float(expert._object_pos()[2]-initial_z),
                  object_table_clearance_m=lift.clearance(),
                  minimum_air_hold_clearance_m=expert._min_air_clearance,
                  air_hold_seconds=expert._air_hold_steps*lift.dt,
                  air_table_force_n=float(air_table_force),
                  hold_torso_peak_n=float(peak_torso),
                  hold_max_object_penetration_m=max_penetration,
                  strict_gate_a_streak=expert._max_bilateral_streak,
                  support_forces={s:f.tolist() for s,f in lift.support_forces().items()},
                  noslip_iterations=env.model.opt.noslip_iterations)
    result['object_contacts'] = []
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        b1, b2 = env.model.geom_bodyid[c.geom1], env.model.geom_bodyid[c.geom2]
        if env._object_body_id in (b1, b2):
            f = np.zeros(6)
            mujoco.mj_contactForce(env.model, env.data, i, f)
            if np.linalg.norm(f[:3]) > .05:
                other = b2 if b1 == env._object_body_id else b1
                result['object_contacts'].append(dict(body=env.model.body(other).name,
                    force_n=float(np.linalg.norm(f[:3])), penetration_m=max(0., -float(c.dist))))
    # Five more seconds after SUCCESS: do not let a terminal flag freeze a fall.
    for _ in range(500):
        env.step(lift.step(lift.height))
        assert lift.supported() and lift.clearance() >= .05
    result['extended_hold_seconds'] = 5.0
    snapshot(env, render_dir, 'extended_hold')
    # Open and move hands apart with actuators. Do not reset or move the object.
    for t in range(400):
        for side, sign in [('left', 1), ('right', -1)]:
            if t < 100:
                lift.start[side][1] += sign * .0005
        action = lift.step(lift.height)
        action[17:25] = -1
        env.step(action)
    assert abs(lift.clearance()) < .003, ('Object did not fall back onto table', lift.clearance())
    assert not lift.supported(), 'Open hands must not continue to support the object'
    result['release_returned_to_table'] = True
    snapshot(env, render_dir, 'released')
    env.reset(seed=seed)
    assert env.model.opt.noslip_iterations == original_solver
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--render-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = [verify(s, args.render_dir / f'seed{s}' if args.render_dir else None) for s in args.seeds]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2)+'\n')
    print(f'PASS: {len(results)} physical lift/extended-hold/release rollouts')
