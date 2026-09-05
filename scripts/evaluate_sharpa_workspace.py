#!/usr/bin/env python3
"""Evaluate real grasp/lift success on explicit object variations, including failures."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from humanoid_learning.envs.grasp_config import GraspEnvConfig
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SharpaBimanualGraspExpert, BimanualGraspConfig, BimanualGraspState as State


def cases(suite):
    base = dict(x=.27, y=0., yaw_deg=0., size=[.12, .12, .12], seed=0)
    result = [dict(base, name='canonical')]
    if suite in ('position', 'all'):
        for axis in ('x', 'y'):
            for delta in (-.01, -.005, .005, .01):
                result.append(dict(base, name=f'{axis}{delta:+.3f}', **{axis:base[axis]+delta}))
    if suite in ('shape', 'all'):
        for size in (.11, .13):
            result.append(dict(base, name=f'cube{size:.2f}', size=[size]*3))
        result.append(dict(base, name='rectangular', size=[.10, .15, .10]))
        for yaw in (-5., 5.):
            result.append(dict(base, name=f'yaw{yaw:+.0f}', yaw_deg=yaw))
    if suite == 'heldout':
        result = [dict(base, name=f'heldout{i}', x=x, y=y, yaw_deg=yaw, size=size) for i,(x,y,yaw,size) in enumerate([
            (.267, -.003, -2., [.12]*3), (.273, .003, 2., [.12]*3),
            (.266, .004, 0., [.115]*3), (.274, -.004, 0., [.125]*3)])]
    return result


def evaluate(payload):
    case, overrides = payload
    started = time.monotonic()
    config = GraspEnvConfig(arm_gravity_compensation=True, max_episode_steps=8000,
        object_pos=(case['x'], case['y'], 0.), object_half_extents=tuple(v/2 for v in case['size']),
        object_yaw_rad=math.radians(case['yaw_deg']))
    env = SharpaGraspEnv(config)
    env.reset(seed=case['seed'])
    expert = SharpaBimanualGraspExpert(env, BimanualGraspConfig(**overrides))
    transitions = []
    support_trace = []
    initial_mass = env.model.body_mass.copy()
    initial_friction = env.model.geom_friction.copy()
    initial_neq = env.model.neq
    peak_torso = 0.0
    peak_penetration = 0.0
    peak_air_table_force = 0.0
    previous = None
    for tick in range(config.max_episode_steps):
        _, _, term, trunc, info = env.step(expert.step())
        if expert._contact_lift is not None:
            peak_torso = max(peak_torso, env._torso_arm_collision_force())
            lift = expert._contact_lift
            for i in range(env.data.ncon):
                c = env.data.contact[i]
                if env._object_body_id in (env.model.geom_bodyid[c.geom1], env.model.geom_bodyid[c.geom2]):
                    peak_penetration = max(peak_penetration, -float(c.dist))
                if expert.state == State.AIR_HOLD and {c.geom1, c.geom2} == {lift.table_geom, lift.object_geom}:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, i, force)
                    peak_air_table_force = max(peak_air_table_force, float(np.linalg.norm(force[:3])))
        if expert.state != previous:
            transitions.append(dict(tick=tick, state=expert.state.name))
            previous = expert.state
        if expert._contact_lift is not None and tick % 20 == 0:
            lift = expert._contact_lift
            support_trace.append(dict(tick=tick, state=expert.state.name,
                force={s:f.tolist() for s,f in lift.support_forces().items()},
                supported=bool(lift.supported()), object_pos=expert._object_pos().tolist(),
                palm={s:env.palm_pose(s)[0].tolist() for s in ('left','right')}))
        if expert.state in (State.SUCCESS, State.FAILURE) or term or trunc:
            break
    lift = expert._contact_lift
    clearance = lift.clearance() if lift else None
    success = bool(expert.state == State.SUCCESS and lift and lift.supported()
                   and clearance >= expert.config.lift_height_m
                   and expert._air_hold_steps * lift.dt >= expert.config.air_hold_seconds
                   and peak_air_table_force == 0.0)
    assert np.array_equal(initial_mass, env.model.body_mass)
    assert np.array_equal(initial_friction, env.model.geom_friction)
    assert env.model.neq == initial_neq, 'No new attachment/equality constraint allowed'
    result = dict(case=case, success=success, state=expert.state.name,
        failure=str(expert.failure_reason), transitions=transitions, steps=tick+1,
        clearance_m=clearance, air_hold_steps=expert._air_hold_steps,
        air_hold_seconds=expert._air_hold_steps * (lift.dt if lift else 0.),
        min_air_clearance_m=expert._min_air_clearance if np.isfinite(expert._min_air_clearance) else None,
        peak_air_table_force_n=peak_air_table_force, hold_peak_torso_force_n=peak_torso,
        hold_max_object_penetration_m=peak_penetration,
        object_mass_kg=config.object_mass, object_friction=list(config.object_friction),
        forces={s:[env._group_contact_force(s,g)[1] for g in ('thumb','index','middle','wrap')] for s in ('left','right')},
        elapsed_seconds=time.monotonic()-started, support_trace=support_trace)
    env.close()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite', choices=['position','shape','all','heldout'], default='position')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--only', nargs='+')
    p.add_argument('--expert-overrides', default='{}')
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    selected=[c for c in cases(args.suite) if not args.only or c['name'] in args.only]
    if not selected:
        p.error('No matching cases')
    overrides=json.loads(args.expert_overrides)
    if args.workers < 1:
        p.error('--workers must be positive')
    BimanualGraspConfig(**overrides)  # Reject unknown settings before spawning workers.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  suite=args.suite, expert_overrides=overrides, case_count=len(selected),
                  mujoco_version=mujoco.__version__, expert_config=asdict(BimanualGraspConfig(**overrides)),
                  git_status=subprocess.check_output(['git','status','--short'],text=True),
                  source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(Path('humanoid_learning').rglob('*.py'))},
                  evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    results=[]
    with args.output.open('w') as f, ProcessPoolExecutor(max_workers=args.workers) as pool:
        f.write(json.dumps(dict(metadata=metadata))+'\n');f.flush()
        for result in pool.map(evaluate, [(c,overrides) for c in selected]):
            results.append(result)
            f.write(json.dumps(result)+'\n');f.flush()
            print(f"{result['case']['name']}: {result['state']} {result['failure']} clearance={result['clearance_m']}",flush=True)
    print(f"SUCCESS {sum(r['success'] for r in results)}/{len(results)}",flush=True)


if __name__ == '__main__':
    main()
