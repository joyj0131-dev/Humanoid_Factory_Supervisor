#!/usr/bin/env python3
"""Run the live fault -> walk -> lift attempt and save an honest result."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import hashlib

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from humanoid_learning.envs.factory_env import FactoryEnv
from humanoid_learning.expert.factory_recovery import FactoryRecovery, RecoveryConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--station', type=int, choices=(0, 1), required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scenario', choices=('dropped_part', 'misplaced_part'), default='dropped_part')
    parser.add_argument('--stand-off', type=float, default=0.27)
    parser.add_argument('--max-steps', type=int, default=12000)
    parser.add_argument('--squeeze', type=float, default=0.006)
    parser.add_argument('--forward-bias', type=float, default=-0.08)
    parser.add_argument('--motion-profile', choices=('baseline', 'compact', 'direct', 'smooth'), default='baseline')
    parser.add_argument('--out', default='results/factory/recovery.json')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--kinematic-ik', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    env = FactoryEnv()
    try:
        env.reset(seed=args.seed, options={'fault_workcell': args.station, 'scenario': args.scenario})
        recovery = FactoryRecovery(env, RecoveryConfig(stand_off_m=args.stand_off, max_steps=args.max_steps,
                                                       hold_squeeze_m=args.squeeze,
                                                       motion_profile=args.motion_profile,
                                                       kinematic_ik=args.kinematic_ik,
                                                       forward_command_bias=args.forward_bias))
        peak_step = 0.0
        previous = env.data.qpos[:3].copy()
        last_state = None
        phase_events = []
        motion_samples = {}
        previous_palms = None
        previous_motion_state = None
        step_times = []
        phase_times = {}
        state_digest = hashlib.sha256()
        loop_started = time.perf_counter()
        sim_started = env.data.time
        for step in range(args.max_steps):
            phase = (recovery.expert.state.name if recovery.state == 'GRASP' else recovery.state)
            started = time.perf_counter()
            info = recovery.step()
            step_times.append(time.perf_counter() - started)
            phase_times.setdefault(phase, []).append(step_times[-1])
            for values in (env.data.qpos, env.data.qvel, env.data.ctrl):
                state_digest.update(values.tobytes())
            peak_step = max(peak_step, float(np.linalg.norm(env.data.qpos[:3] - previous)))
            previous = env.data.qpos[:3].copy()
            state = (recovery.state, getattr(getattr(recovery, 'expert', None), 'state', None))
            if state[0] == 'GRASP' and state[1] is not None:
                palms = np.stack([recovery.grasp.palm_pose(s)[0].copy() for s in ('left', 'right')])
                if previous_motion_state == state and previous_palms is not None:
                    dt = env.config.frame_skip * env.model.opt.timestep
                    speed = np.linalg.norm(palms - previous_palms, axis=1) / dt
                    motion_samples.setdefault(state[1].name, []).append(speed.tolist())
                previous_palms = palms
            previous_motion_state = state
            if state != last_state:
                phase_events.append({'step': step, 'phase': recovery.state,
                                     'grasp_state': state[1].name if state[1] is not None else None})
                print(step, *state, 'base', np.round(previous, 4), flush=True)
                last_state = state
            if recovery.state in recovery.TERMINAL:
                break
        elapsed = time.perf_counter() - loop_started
        simulated = env.data.time - sim_started
        motion_metrics = {}
        for name, samples in motion_samples.items():
            speeds = np.asarray(samples)
            # Interior 80% excludes intended phase-boundary acceleration/settling.
            interior = speeds[len(speeds)//10:max(len(speeds)//10 + 1, 9*len(speeds)//10)]
            motion_metrics[name] = {
                'samples': len(samples),
                'interior_slow_fraction_per_hand': np.mean(interior < 0.002, axis=0).tolist(),
                'interior_speed_std_m_s_per_hand': np.std(interior, axis=0).tolist(),
                'peak_speed_m_s_per_hand': speeds.max(axis=0).tolist(),
            }
        result = {
            'station': args.station, 'seed': args.seed, 'scenario': args.scenario,
            'success': recovery.state == 'LIFTED', 'state': recovery.state,
            'failure': recovery.failure, 'steps': recovery.total_steps,
            'max_clearance_m': recovery.max_clearance,
            'max_supported_hold_seconds': recovery.max_hold_steps * env.config.frame_skip * env.model.opt.timestep,
            'max_base_step_m': peak_step, 'fallen': bool(info.get('fallen')),
            'object_hand_penetration_max_m': recovery.max_object_hand_penetration_m,
            'forbidden_contact_ticks': recovery.forbidden_contact_ticks,
            'forbidden_contact_pairs': recovery.forbidden_contact_pairs,
            'collision_free_lift_success': recovery.state == 'LIFTED' and recovery.forbidden_contact_ticks == 0,
            'events': recovery.events, 'config': vars(recovery.config),
            'phase_events': phase_events,
            'used_direct_approach': recovery.used_direct_approach,
            'motion_metrics': motion_metrics,
            'physics_dt_s': env.config.frame_skip * env.model.opt.timestep,
            'controller_step_wall_p50_p95_s': np.percentile(step_times, [50, 95]).tolist(),
            'headless_loop_wall_seconds': elapsed,
            'simulated_seconds': simulated,
            'headless_loop_rtf': simulated / elapsed,
            'qpos_qvel_ctrl_trajectory_sha256': state_digest.hexdigest(),
            'phase_timing': {name: {'ticks': len(values), 'wall_seconds': sum(values),
                                  'mean_step_ms': 1000 * float(np.mean(values))}
                             for name, values in phase_times.items()},
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
        if args.render:
            import mujoco
            from PIL import Image
            from humanoid_learning.envs import factory_config as fc
            for station, arm in enumerate(env.arms):
                env.model.geom_rgba[env.model.geom(fc.beacon_geom_name(station)).id] = (
                    (1.0, 0.65, 0.05, 1.0) if arm.faulted else fc.BEACON_RUNNING_RGBA)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = env.part_position(args.station)
            camera.distance, camera.azimuth, camera.elevation = 1.6, 190, -15
            with mujoco.Renderer(env.model, height=720, width=1280) as renderer:
                renderer.update_scene(env.data, camera)
                Image.fromarray(renderer.render()).save(out.with_suffix('.png'))
        return 0 if result['success'] else 1
    finally:
        if 'recovery' in locals():
            recovery.close()
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
