#!/usr/bin/env python3
"""Capture a REAL walk handoff, or probe floor control from that initial state.

Restored entry probes are subsystem experiments, NEVER end-to-end successes.
"""
import argparse
import copy
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from humanoid_learning.envs.factory_env import FactoryEnv
from humanoid_learning.expert.factory_recovery import FactoryRecovery, RecoveryConfig
from humanoid_learning.expert.factory_floor_pickup import FactoryFloorPickup
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SharpaBimanualGraspExpert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', default='results/factory/floor_entry.npz')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--compare-factory', action='store_true',
                        help='with --capture: compare live factory versus paused-props diagnostic physics')
    parser.add_argument('--contact-snapshot', default=None,
                        help='probe CLOSE from an evaluated physical terminal pose; not an end-to-end run')
    parser.add_argument('--height', type=float, default=None)
    parser.add_argument('--pitch', type=float, default=None)
    parser.add_argument('--steps', type=int, default=2300)
    parser.add_argument('--render', default=None)
    parser.add_argument('--body-lift', action='store_true',
                        help='subsystem candidate: start standing after bilateral contact, without arm-only lift')
    parser.add_argument('--horizontal-wrap', action='store_true',
                        help='A/B only: old low-wrist belt grasp reused on the floor')
    parser.add_argument('--azimuth', type=float, default=110.)
    parser.add_argument('--elevation', type=float, default=-65.)
    args = parser.parse_args()
    e = FactoryEnv()
    e.reset(seed=0, options={'fault_workcell': 1})
    r = FactoryRecovery(e, RecoveryConfig(motion_profile='smooth'))
    state_type = mujoco.mjtState.mjSTATE_INTEGRATION
    if args.capture:
        for _ in range(12000):
            r.step()
            if r.state == 'FLOOR_PICKUP':
                state = np.zeros(mujoco.mj_stateSize(e.model, state_type))
                mujoco.mj_getState(e.model, e.data, state, state_type)
                Path(args.entry).parent.mkdir(parents=True, exist_ok=True)
                np.savez(args.entry, state=state, gain=e.model.actuator_gainprm,
                         bias=e.model.actuator_biasprm, arm=r.grasp._arm_target,
                         waist=r.grasp._waist_target, synergy=r.grasp._group_synergy,
                         heading=r.pick_heading, diagnostic_only=True, learner_ready=False,
                         noslip_iterations=e.model.opt.noslip_iterations)
                print('Captured real walk-to-floor entry', r.total_steps, flush=True)
                break
            if r.state in r.TERMINAL:
                raise RuntimeError(r.failure)
        else:
            raise RuntimeError('no floor entry')
    else:
        entry = np.load(args.contact_snapshot or args.entry)
        e.model.actuator_gainprm[:] = entry['gain']
        e.model.actuator_biasprm[:] = entry['bias']
        if 'noslip_iterations' in entry:
            e.model.opt.noslip_iterations = int(entry['noslip_iterations'])
        mujoco.mj_setState(e.model, e.data, entry['state'], state_type)
        mujoco.mj_forward(e.model, e.data)
        r.station, r.floor_task = 1, True
        r.pick_heading = float(entry['heading']) if 'heading' in entry else 0.
        r.grasp = r._grasp_view(1)
        r.grasp._arm_target[:] = entry['arm'] if 'arm' in entry else e.data.qpos[r.grasp._arm_qpos_adr]
        r.grasp._waist_target[:] = entry['waist'] if 'waist' in entry else e.data.qpos[r.grasp._waist_qpos_adr]
        r.grasp._group_synergy[:] = entry['synergy'] if 'synergy' in entry else .8
        r.expert = SharpaBimanualGraspExpert(r.grasp, heading=r.pick_heading)
        r.floor_pickup = FactoryFloorPickup(r)
    floor = r.floor_pickup
    if args.compare_factory:
        if not args.capture:
            raise ValueError('--compare-factory requires a real --capture walk')
        compare_factory(r)
        e.close()
        return
    if args.height is not None:
        floor.height = args.height
    if args.pitch is not None:
        floor.pitch = args.pitch
    if args.horizontal_wrap:
        targets = r.expert._mirrored_targets(.02, .08, .15)
        floor.goal = np.array([targets[s] for s in ('left', 'right')])
        floor.goal_R = [r.expert._facing_rotation(s, targets[s], floor.obj, floor.start_R[i][:,0])
                        for i, s in enumerate(('left', 'right'))]
    if args.contact_snapshot:
        floor.begin_contact()
        floor.standing_height = .763  # target only; never written into live state
    for t in range(args.steps):
        action = floor.step()
        r.grasp.step(action)
        if args.body_lift and floor.stage == 'PICK_CLEAR':
            from humanoid_learning.expert.whole_body_posture import WholeBodyPosture
            floor.stage, floor.tick = 'RISE', 0
            floor.lift_start = np.array([r.grasp.palm_pose(s)[0] for s in ('left', 'right')])
            floor.lift_R = [r.grasp.palm_pose(s)[1] for s in ('left', 'right')]
            floor.rise_base_height = float(e.data.xpos[floor.posture.pelvis, 2])
            floor.rise_pitch = float(r.stabilizer.tilt()[1])
            floor.rise_contact_gap = 0
            floor.posture = WholeBodyPosture(r.grasp, floor.ik_clearance_m)
        if t % 100 == 0:
            p = floor.posture
            actual_com = e.data.subtree_com[p.pelvis, :2]
            print(t, floor.stage, 'base', np.round(e.data.xpos[p.pelvis], 3),
                  'tilt', np.round(r.stabilizer.tilt()[:2], 3),
                  'com',np.round(actual_com-p.com_xy,3),
                  'feet',r.walker.in_double_support(15.),
                  'palms',np.round([r.grasp.palm_pose(s)[0][2] for s in ('left','right')],3),
                  'error',round(floor.target_error_m,3), 'IK',p.last_error,flush=True)
            print('peak hand forces',floor.peak_hand_force_n,'object',e.part_position(1),
                  'lift evidence',r.lift_evidence(), flush=True)
            from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift
            print('current forces', {s: np.round(v, 3).tolist() for s, v in
                  SharpaContactLift.measure_support_forces(r.grasp).items()}, flush=True)
        if abs(r.stabilizer.tilt()[0]) > .6 or abs(r.stabilizer.tilt()[1]) > 1.2 or floor.failure:
            print('STOP', floor.failure or 'TILT',flush=True)
            break
    if args.render:
        from PIL import Image
        camera = mujoco.MjvCamera()
        camera.lookat[:] = e.data.xpos[floor.posture.pelvis]+[0,0,.15]
        camera.distance, camera.azimuth, camera.elevation = 2., args.azimuth, args.elevation
        with mujoco.Renderer(e.model, height=720, width=960) as renderer:
            renderer.update_scene(e.data, camera)
            Image.fromarray(renderer.render()).save(args.render)
    contacts = []
    for i, c in enumerate(e.data.contact):
        force = np.zeros(6)
        mujoco.mj_contactForce(e.model, e.data, i, force)
        contacts.append((float(np.linalg.norm(force[:3])),
                         e.model.body(int(e.model.geom_bodyid[c.geom1])).name,
                         e.model.body(int(e.model.geom_bodyid[c.geom2])).name))
    print('contacts', sorted(contacts, reverse=True)[:15], flush=True)
    print('surface geometry',floor.geometry_report(),flush=True)
    p = floor.posture
    print('joint_errors', [(n, round(float(q-a),3)) for n,q,a in
                           zip(p.names, floor.last_q, e.data.qpos[p.qadr])],flush=True)
    e.close()


def compare_factory(live):
    """Find first divergence; never use either branch as a mission success."""
    e = live.env
    other = FactoryEnv()
    other.reset(seed=0, options={'fault_workcell': 1})
    shadow = FactoryRecovery(other, RecoveryConfig(motion_profile='smooth'))
    other.model.actuator_gainprm[:] = e.model.actuator_gainprm
    other.model.actuator_biasprm[:] = e.model.actuator_biasprm
    other.model.opt.noslip_iterations = e.model.opt.noslip_iterations
    state_type = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(e.model, state_type))
    mujoco.mj_getState(e.model, e.data, state, state_type)
    mujoco.mj_setState(other.model, other.data, state, state_type)
    mujoco.mj_forward(e.model, e.data)
    mujoco.mj_forward(other.model, other.data)
    shadow.station, shadow.floor_task, shadow.pick_heading = 1, True, live.pick_heading
    shadow.grasp = shadow._grasp_view(1)
    shadow.grasp.config = copy.deepcopy(live.grasp.config)
    for name in ('_arm_target', '_waist_target', '_group_synergy'):
        getattr(shadow.grasp, name)[:] = getattr(live.grasp, name)
    shadow.expert = SharpaBimanualGraspExpert(shadow.grasp, heading=shadow.pick_heading)
    shadow.expert.config = copy.deepcopy(live.expert.config)
    shadow.floor_pickup = FactoryFloorPickup(shadow)
    live.floor_pickup = FactoryFloorPickup(live)
    shadow.floor_pickup.ik_clearance_m = live.floor_pickup.ik_clearance_m = 0.
    robot_q = []
    pelvis = e.model.body('pelvis').id
    for jid in range(e.model.njnt):
        if e.model.body_rootid[e.model.jnt_bodyid[jid]] == pelvis:
            start = e.model.jnt_qposadr[jid]
            end = e.model.jnt_qposadr[jid+1] if jid+1 < e.model.njnt else e.model.nq
            robot_q.extend(range(start, end))
    for tick in range(2500):
        live.step()
        other_action = shadow.floor_pickup.step()
        shadow.grasp.step(other_action)
        error = np.max(np.abs(e.data.qpos[robot_q]-other.data.qpos[robot_q]))
        if tick % 100 == 0 or error > 1e-8:
            target_error = np.max(np.abs(live.grasp._arm_target-shadow.grasp._arm_target))
            print('COMPARE', tick, 'robot q error', error, 'arm target error',target_error,
                  'live',live.floor_pickup.stage,'shadow',shadow.floor_pickup.stage,flush=True)
        if error > 1e-8:
            print('ctrl difference ids',np.where(np.abs(e.data.ctrl-other.data.ctrl)>1e-8)[0],flush=True)
            break
    other.close()


if __name__ == '__main__':
    main()
