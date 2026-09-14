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
import mujoco

from humanoid_learning.envs import factory_config as fcfg
from humanoid_learning.envs.factory_env import FactoryEnv
from humanoid_learning.expert.factory_recovery import FactoryRecovery, RecoveryConfig
from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--station', type=int, choices=(0, 1), required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--scenario', choices=list(fcfg.SCENARIOS), default=None,
                        help='default: jam on line 0, arm_drop on line 1')
    parser.add_argument('--stand-off', type=float, default=0.27)
    parser.add_argument('--max-steps', type=int, default=12000)
    parser.add_argument('--squeeze', type=float, default=0.006)
    parser.add_argument('--forward-bias', type=float, default=-0.08)
    parser.add_argument('--motion-profile', choices=('baseline', 'compact', 'direct', 'smooth'), default='baseline')
    parser.add_argument('--out', default='results/factory/recovery.json')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--place', action='store_true', help='continue after lift into experimental place/restart')
    parser.add_argument('--carry', action=argparse.BooleanOptionalAction, default=True,
                        help='walk the held part to the canonical spot before placing')
    parser.add_argument('--kinematic-ik', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    env = FactoryEnv()
    try:
        env.reset(seed=args.seed, options={k: v for k, v in (('fault_workcell', args.station), ('scenario', args.scenario)) if v is not None})
        recovery = FactoryRecovery(env, RecoveryConfig(stand_off_m=args.stand_off, max_steps=args.max_steps,
                                                       hold_squeeze_m=args.squeeze,
                                                       motion_profile=args.motion_profile,
                                                       kinematic_ik=args.kinematic_ik,
                                                       place_after_lift=args.place,
                                                       carry_by_walking=args.carry,
                                                       forward_command_bias=args.forward_bias))
        peak_step = 0.0
        foot_part_ticks, foot_part_peak_n = 0, 0.
        extra_contact_ticks, interference_ticks, extra_pairs = 0, 0, {}
        hands = SharpaContactLift.hand_wrist_body_ids(recovery.grasp)
        hand_ids = hands['left'] | hands['right']
        pelvis_id = recovery.stabilizer.pelvis_body
        floor_geom = env.model.geom('floor').id
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
            if recovery.station is not None:
                part_body = env.model.body(fcfg.part_body_name(recovery.station)).id
                touched, extra, interference = False, False, False
                for index, contact in enumerate(env.data.contact):
                    bodies = env.model.geom_bodyid[[contact.geom1, contact.geom2]]
                    if part_body in bodies and any(body in bodies for body in recovery.walker.foot_bodies):
                        force = np.zeros(6)
                        mujoco.mj_contactForce(env.model, env.data, index, force)
                        foot_part_peak_n = max(foot_part_peak_n, float(np.linalg.norm(force[:3])))
                        touched = True
                    robot = [env.model.body_rootid[b] == pelvis_id for b in bodies]
                    if not any(robot):
                        continue
                    allowed_foot = (floor_geom in (contact.geom1, contact.geom2)
                                    and any(b in recovery.walker.foot_bodies for b in bodies))
                    allowed_grip = part_body in bodies and any(b in hand_ids for b in bodies)
                    if allowed_foot or allowed_grip:
                        continue
                    force = np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, index, force)
                    magnitude = float(np.linalg.norm(force[:3]))
                    if magnitude <= .5:
                        continue
                    internal_hand = any(all(b in ids for b in bodies) for ids in hands.values())
                    category = 'hand_internal' if internal_hand else ('robot_self' if all(robot) else 'environment')
                    extra, interference = True, interference or not internal_hand
                    phase_name = info.get('floor_stage') or recovery.state
                    pair = ':'.join([phase_name, category, *(env.model.body(int(b)).name for b in bodies)])
                    row = extra_pairs.setdefault(pair, {'samples': 0, 'peak_force_n': 0., 'max_penetration_m': 0.})
                    row['samples'] += 1
                    row['peak_force_n'] = max(row['peak_force_n'], magnitude)
                    row['max_penetration_m'] = max(row['max_penetration_m'], -float(contact.dist))
                foot_part_ticks += int(touched)
                extra_contact_ticks += int(extra)
                interference_ticks += int(interference)
            state = (recovery.state, getattr(getattr(recovery, 'expert', None), 'state', None),
                     info.get('floor_stage'))
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
                                     'grasp_state': state[1].name if state[1] is not None else None,
                                     'floor_stage': state[2]})
                print(step, *state, 'base', np.round(previous, 4), flush=True)
                last_state = state
            elif recovery.state == 'FLOOR_PICKUP' and step % 500 == 0:
                print(step, 'FLOOR_PICKUP', info.get('floor_stage'),
                      'palm error', info.get('floor_target_error_m'),
                      'hand forces', info.get('floor_peak_hand_force_n'),
                      'base', np.round(previous, 4), flush=True)
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
            'success': recovery.state == ('RECOVERED' if args.place else 'LIFTED'), 'state': recovery.state,
            'place_stage': info.get('place_stage'), 'place_error_m': info.get('place_error_m'),
            'carry_error_m': info.get('carry_error_m'),
            'place_commanded_rotation_deg': info.get('place_commanded_rotation_deg'),
            'place_tilt_deg': info.get('place_tilt_deg'),
            'place_start_position': recovery.placer.start.tolist() if hasattr(recovery, 'placer') else None,
            'place_target_position': recovery.placer.target.tolist() if hasattr(recovery, 'placer') else None,
            'place_actual_position': env.part_position(recovery.station).tolist() if hasattr(recovery, 'placer') else None,
            'line_state': info.get('line_state'), 'mission_events': info.get('mission_events'),
            'failure': recovery.failure, 'steps': recovery.total_steps,
            'foot_part_contact_ticks': foot_part_ticks, 'foot_part_peak_force_n': foot_part_peak_n,
            'floor_stage': info.get('floor_stage'),
            'floor_peak_hand_force_n': info.get('floor_peak_hand_force_n'),
            'floor_geometry': recovery.floor_pickup.geometry_report() if hasattr(recovery, 'floor_pickup') else None,
            'floor_target_error_m': info.get('floor_target_error_m'),
            'intentional_crouch': info.get('intentional_crouch', False),
            'final_pelvis_position_m': env.data.xpos[recovery.stabilizer.pelvis_body].tolist(),
            'final_tilt_rad': np.asarray(recovery.stabilizer.tilt()).tolist(),
            'final_double_support_15n': recovery.walker.in_double_support(15.),
            'floor_ground_foot_loads_n': info.get('floor_ground_foot_loads_n'),
            'max_clearance_m': recovery.max_clearance,
            'max_supported_hold_seconds': recovery.max_hold_steps * env.config.frame_skip * env.model.opt.timestep,
            'max_base_step_m': peak_step,
            'fallen': bool(info.get('recovery_fallen', info.get('fallen'))),
            'raw_env_fallen': bool(info.get('fallen')),
            'object_hand_penetration_max_m': recovery.max_object_hand_penetration_m,
            'forbidden_contact_ticks': recovery.forbidden_contact_ticks,
            'forbidden_contact_pairs': recovery.forbidden_contact_pairs,
            'contact_audit_force_threshold_n': .5,
            'additional_contact_ticks': extra_contact_ticks,
            'interference_contact_ticks': interference_ticks,
            'additional_contact_pairs': extra_pairs,
            'pelvis_torso_contact_free_lift_success': recovery.state in ('LIFTED', 'RECOVERED') and recovery.forbidden_contact_ticks == 0,
            'collision_free_lift_success': recovery.state in ('LIFTED', 'RECOVERED') and extra_contact_ticks == 0,
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
            from PIL import Image
            from humanoid_learning.envs import factory_config as fc
            state_type = mujoco.mjtState.mjSTATE_INTEGRATION
            snapshot = np.zeros(mujoco.mj_stateSize(env.model, state_type))
            mujoco.mj_getState(env.model, env.data, snapshot, state_type)
            np.savez(out.with_suffix('.npz'), diagnostic_only=True, learner_ready=False,
                     state=snapshot, gain=env.model.actuator_gainprm,
                     bias=env.model.actuator_biasprm, noslip_iterations=env.model.opt.noslip_iterations,
                     qpos=env.data.qpos,
                     qvel=env.data.qvel, ctrl=env.data.ctrl)
            for station, arm in enumerate(env.arms):
                env.model.geom_rgba[env.model.geom(fc.beacon_geom_name(station)).id] = (
                    (1.0, 0.65, 0.05, 1.0) if arm.faulted else fc.BEACON_RUNNING_RGBA)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = env.part_position(recovery.station) + [0., 0., .15]
            camera.distance = 1.9
            camera.azimuth, camera.elevation = 110., -45.
            if recovery.floor_task:
                camera.azimuth, camera.elevation = 110., -65.
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
