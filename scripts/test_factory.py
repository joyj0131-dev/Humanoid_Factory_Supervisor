#!/usr/bin/env python3
"""Factory environment tests: model structure, independent scripted cells,
deterministic fault injection, dropped-part physics, and the Navigation Gate.

Run directly (the system pytest CLI is broken in this project):
    OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python3 scripts/test_factory.py
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from humanoid_learning.envs import factory_config as fcfg
from humanoid_learning.envs import frames
from humanoid_learning.envs.factory_env import ACTION_DIM, FACTORY_OBS_DIM, FactoryEnv
from humanoid_learning.envs.factory_model import build_factory_model

ZERO = np.zeros(ACTION_DIM)


def wait_for_fault(env, limit=2200):
    for _ in range(limit):
        info = env.step(ZERO)[4]
        if info["fault_active"]:
            return info
    raise AssertionError(f"physical fault not detected: {env.scenario_status}")

# The compiled G1+Sharpa actuator contract this project has locked in.
G1_ACTUATOR_COUNT = 73


def test_model_compiles_with_two_independent_lines():
    model = build_factory_model(fcfg.FactoryConfig())
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert len(names) == len(set(names)), "duplicate actuator names"
    station_actuators = [n for n in names if n and n.startswith("wc")]
    expected = fcfg.N_WORKCELLS * (len(fcfg.ARM_JOINT_SUFFIXES) + len(fcfg.GRIPPER_JOINT_SUFFIXES))
    assert len(station_actuators) == expected, station_actuators
    assert model.nu - len(station_actuators) == G1_ACTUATOR_COUNT, "G1 actuator contract changed"

    # Two of everything: each line owns a belt, a table, a part and an arm.
    for namer in (fcfg.belt_body_name, fcfg.table_body_name,
                  fcfg.part_body_name, fcfg.stopper_body_name, fcfg.beacon_body_name):
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, namer(k)) for k in range(fcfg.N_WORKCELLS)]
        assert all(i >= 0 for i in ids), namer(0)
        assert len(set(ids)) == len(ids), f"stations share {namer(0)}"

    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    assert base >= 0 and model.jnt_type[base] == mujoco.mjtJoint.mjJNT_FREE
    print(f"    nq/nv/nu = {model.nq}/{model.nv}/{model.nu}; {len(station_actuators)} station actuators, "
          f"{G1_ACTUATOR_COUNT} G1 actuators")


def test_arm_links_do_not_collide_with_each_other():
    """MuJoCo's filterparent does NOT protect an arm link from its own column:
    the column has no joint, so it is welded to the world and the pair reads as
    link-vs-world. Left unfixed the turret hit its column at 2.6e17 N and jammed
    base_yaw entirely."""
    env = FactoryEnv()
    try:
        env.reset(seed=0)
        arm_bodies = {
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, fcfg.arm_body_name(k, part))
            for k in range(fcfg.N_WORKCELLS)
            for part in ("column", "turret", "upper", "fore", "wrist", "jaw_left", "jaw_right")
        }
        worst = 0.0
        for _ in range(400):
            env.step(ZERO)
            for c in range(env.data.ncon):
                contact = env.data.contact[c]
                b1 = env.model.geom_bodyid[contact.geom1]
                b2 = env.model.geom_bodyid[contact.geom2]
                if b1 in arm_bodies and b2 in arm_bodies:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, c, force)
                    worst = max(worst, float(np.linalg.norm(force[:3])))
        assert worst == 0.0, f"arm self-collision at up to {worst:.1f} N"
        print("    400 steps: zero arm self-collision contacts")
    finally:
        env.close()


def test_each_line_reproduces_the_canonical_grasp_relationship():
    config = fcfg.FactoryConfig()
    poses = config.workcells
    assert len(poses) == fcfg.N_WORKCELLS

    for pose in poses:
        # Round-trip: world <-> local must be exact to floating point.
        for local in [(0.27, 0.0), (0.3, 0.55), (-0.8, 0.0)]:
            back = pose.to_local_xy(pose.to_world_xy(local))
            assert np.allclose(back, local, atol=1e-12), (local, back)
        # The canonical grasp relationship (table 0.30 ahead, part at 0.27)
        # must hold identically in every cell's own frame.
        assert np.allclose(pose.to_local_xy(pose.table_pos[:2]), fcfg.LOCAL_TABLE_POS[:2], atol=1e-12)
        assert np.allclose(pose.to_local_xy(pose.canonical_part_xy), fcfg.LOCAL_CANONICAL_PART_XY, atol=1e-12)

    # The corridor runs BETWEEN the lines, so the supervisor necessarily works
    # them from opposite sides. Line 0 presents its belt to the corridor and
    # line 1 its table, which is what makes both faults reachable from inside.
    assert abs(abs(poses[0].heading_rad - poses[1].heading_rad) - math.pi) < 1e-12
    assert poses[0].work_surface == "belt" and poses[1].work_surface == "table"
    for pose in poses:
        offset = float(np.linalg.norm(np.asarray(pose.manipulation_xy) - fcfg.CORRIDOR_CENTRE_X * np.array([1.0, 0.0])))
        assert abs(pose.manipulation_xy[0] - fcfg.CORRIDOR_CENTRE_X) < fcfg.CORRIDOR_HALF_WIDTH_M, (
            f"line {pose.index}'s stand pose is outside the walking corridor")
    separation = float(np.linalg.norm(np.asarray(poses[0].manipulation_xy) - np.asarray(poses[1].manipulation_xy)))
    assert separation > 1.0, f"lines too close for locomotion to be required: {separation:.3f} m"
    print(f"    both stand poses inside the corridor, {separation:.3f} m apart, facing opposite ways")


def test_arm_cycle_stays_above_the_belt_and_reaches_the_part():
    config = fcfg.FactoryConfig()
    model = build_factory_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand"))
    worst = math.inf
    grasp_error = None
    for k, pose in enumerate(config.workcells):
        adr = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fcfg.arm_joint_name(k, s))]
            for s in fcfg.ARM_JOINT_SUFFIXES
        ]
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, fcfg.arm_body_name(k, "grasp"))
        part = np.r_[pose.pick_xy, fcfg.PART_CENTRE_Z]
        for index, waypoint in enumerate(fcfg.arm_cycle_waypoints(pose)):
            for a, v in zip(adr, waypoint):
                data.qpos[a] = v
            mujoco.mj_forward(model, data)
            grasp = data.site_xpos[site]
            # The gripper centre may sit at part height, but never below the belt.
            worst = min(worst, float(grasp[2]) - fcfg.TABLE_TOP_Z)
            if index == fcfg.ARM_FAULT_WAYPOINT_INDEX:
                # Only the horizontal alignment has to be exact: the wrist is
                # deliberately high enough to keep the palm off the part's top.
                error = float(np.linalg.norm(grasp[:2] - part[:2]))
                grasp_error = error if grasp_error is None else max(grasp_error, error)
    assert worst > 0.0, f"gripper reaches {worst:.3f} m relative to the belt surface"
    assert grasp_error < 0.005, f"grasp pose misses the part centre by {grasp_error * 1000:.1f} mm"
    print(f"    lowest gripper height above the belt = {worst:+.3f} m; "
          f"grasp pose lands within {grasp_error * 1000:.2f} mm of the part centre")


def test_arm_actually_picks_the_part_up_and_places_it():
    """The arm must physically grip and carry the part, not mime over it.

    Tuning this needed measurement, not reasoning. A shallow jaw target gave
    only ~1 N (a position actuator's grip force is kp x (target - actual), so a
    jaw that stops 1 mm short pushes barely at all) and the part was dropped; an
    18 mm squeeze at low stiffness drove 9.3 mm of contact penetration and
    EJECTED the part upward -- the same soft-contact "squirt" failure this
    project already recorded for the Sharpa hand.
    """
    env = FactoryEnv()
    try:
        env.reset(seed=0, options={"fault_step": 10 ** 9})  # normal production only
        pose = env.poses[0]
        jaw_geoms = {
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, fcfg.arm_body_name(0, f"{s}_geom"))
            for s in fcfg.GRIPPER_JOINT_SUFFIXES
        }
        part_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, fcfg.part_geom_name(0))
        peak_z = 0.0
        peak_force = 0.0
        carried = False
        for _ in range(sum(fcfg.arm_cycle_dwells())):
            env.step(ZERO)
            part = env.part_position(0)
            peak_z = max(peak_z, float(part[2]))
            force = 0.0
            for c in range(env.data.ncon):
                contact = env.data.contact[c]
                geoms = {contact.geom1, contact.geom2}
                if part_geom in geoms and (geoms & jaw_geoms):
                    value = np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, c, value)
                    force += float(np.linalg.norm(value[:3]))
            peak_force = max(peak_force, force)
            if float(np.linalg.norm(part[:2] - pose.table_place_xy)) < 0.06 and part[2] > fcfg.TABLE_TOP_Z:
                carried = True

        rest_z = float(env._part_rest_pos(0)[2])
        assert peak_force > 2.0, f"the jaws never really gripped (peak {peak_force:.2f} N)"
        assert peak_z > rest_z + 0.02, f"the part was never lifted (peak z {peak_z:.3f}, rest {rest_z:.3f})"
        assert carried, "the part never reached the outfeed table"
        assert float(np.linalg.norm(env.part_position(0)[:2] - pose.table_place_xy)) < 0.06, (
            "the part did not end up on the table")
        assert peak_z < 1.30, f"the part was flung (peak z {peak_z:.3f})"
        print(f"    peak jaw force {peak_force:.1f} N, lifted to z={peak_z:.3f} (rest {rest_z:.3f}), "
              f"delivered to the outfeed table")
    finally:
        env.close()


def test_running_belt_carries_a_part_to_the_stop_blade():
    env = FactoryEnv()
    try:
        env.reset(seed=0, options={"fault_step": 10 ** 9})
        pose = env.poses[0]
        upstream = pose.pick_xy - np.array([0.0, 0.45])
        env._set_part(0, env._rest_pos(upstream))
        mujoco.mj_forward(env.model, env.data)
        start_y = float(env.part_position(0)[1])
        for _ in range(500):
            env.step(ZERO)
        part = env.part_position(0)
        speed = float(np.abs(env.data.qvel[env._part_dof_adr[0] : env._part_dof_adr[0] + 3]).max())
        # The blade holds the part's centre roughly at the pick spot.
        assert part[1] > start_y + 0.20, f"the belt barely moved the part ({start_y:.3f} -> {part[1]:.3f})"
        assert part[1] < float(pose.stopper_xy[1]), "the part went past the stop blade"
        assert speed < 0.05, f"the part never settled against the blade (|v|={speed:.3f})"
        assert part[2] > fcfg.TABLE_TOP_Z, "the part left the belt"
        print(f"    belt carried the part from y={start_y:+.3f} to {part[1]:+.3f} and the blade held it "
              f"(|v|={speed:.4f} m/s)")
    finally:
        env.close()


def test_belt_stops_on_fault_and_restarts_only_after_verified_recovery():
    """The full signalling loop the factory is built around.

    The recovery step is performed by the TEST HARNESS, not by a policy: nothing
    in this repository can walk to a station and move a part yet. The point is
    that the task manager judges recovery from the part's measured pose and
    closes the loop, not that the robot did it.
    """
    env = FactoryEnv()
    try:
        _, info = env.reset(seed=0)
        station = info["fault_workcell"]
        healthy = 1 - station
        assert info["line_state"] == "RUNNING" and info["belt_running"]

        wait_for_fault(env)
        _, _, _, _, info = env.step(ZERO)
        assert info["line_state"] == "FAULT_RAISED"
        assert not info["belt_running"], "the line kept running through a fault"
        assert info["target_workcell"] == station
        assert info["arm_faulted"][station] and not info["arm_faulted"][healthy]

        # It must NOT recover by itself while the part is still in the drop zone.
        for _ in range(fcfg.RECOVERY_HOLD_STEPS * 3):
            _, _, _, _, info = env.step(ZERO)
        assert info["line_state"] == "FAULT_RAISED", "the line restarted without recovery"
        assert not info["belt_running"]

        # --- TEST HARNESS: stand in for the missing recovery policy ---
        env._set_part(station, env._part_rest_pos(station))
        mujoco.mj_forward(env.model, env.data)
        # Position alone must not restart the line; the supervisor must reply.
        for _ in range(fcfg.RECOVERY_HOLD_STEPS * 2):
            info = env.step(ZERO)[4]
        assert info["line_state"] == "FAULT_RAISED" and not info["belt_running"]
        info = env.step(ZERO, supervisor_signal=("accept", healthy))[4]
        assert info["supervisor_signal_accepted"] is False
        info = env.step(ZERO, supervisor_signal=("complete", station))[4]
        assert info["supervisor_signal_accepted"] is False
        info = env.step(ZERO, supervisor_signal=("accept", station))[4]
        assert info["supervisor_signal_accepted"] is True
        info = env.step(ZERO, supervisor_signal=("complete", station))[4]
        assert info["supervisor_signal_accepted"] is True
        for _ in range(fcfg.RECOVERY_HOLD_STEPS * 4):
            _, _, _, _, info = env.step(ZERO)
            if info["line_state"] == "RUNNING":
                break
        assert info["line_state"] == "RUNNING", f"the line never restarted ({info['line_state']})"
        assert info["belt_running"], "the belt did not restart"
        assert not any(info["arm_faulted"]), "the stalled arm did not resume"
        assert info["target_workcell"] == -1, "the G1 is still being called after recovery"
        assert info["recovered_step"] is not None
        assert [e["event"] for e in info["mission_events"]] == [
            "fault_raised", "request_accepted", "verification_requested",
            "recovery_verified", "line_restarted"]
        print(f"    fault at step {info['fault_triggered_step']} stopped the line; "
              f"verified recovery at step {info['recovered_step']} restarted belt and arm")
    finally:
        env.close()


def test_the_line_visibly_runs_and_then_visibly_stops():
    """A part is only convincing as "stopped" if it was first seen moving.

    The part enters upstream and rides the belt in. The jam takes the belt's
    traction away mid-run, so it coasts down and stops short of the blade --
    measured on the part itself, not on the flag that stopped it.
    """
    env = FactoryEnv()
    try:
        env.reset(seed=0, options={"fault_workcell": 0, "scenario": "jam", "fault_step": 200})
        start_y = float(env.part_position(0)[1])
        running = 0.0
        for _ in range(150):
            _, _, _, _, info = env.step(ZERO)
            running = max(running, info["line_throughput_m_s"][0])
        travelled = float(env.part_position(0)[1]) - start_y
        assert running > 0.1, f"the part never got moving ({running:.3f} m/s)"
        assert travelled > 0.15, f"the part barely travelled ({travelled * 1000:.0f} mm)"

        info = wait_for_fault(env)
        assert env.scenario_status == "jammed"
        stopped_at = env.part_position(0).copy()
        after = 0.0
        for _ in range(400):
            _, _, _, _, info = env.step(ZERO)
            after = max(after, info["line_throughput_m_s"][0])
        drift = float(np.linalg.norm(env.part_position(0) - stopped_at))
        short = float(np.linalg.norm(env.part_position(0)[:2] - env.poses[0].pick_xy))

        assert after < 0.02, f"the stopped line's part kept moving at {after:.3f} m/s"
        assert drift < 0.01, f"the stopped part drifted {drift * 1000:.0f} mm"
        assert short > fcfg.RECOVERY_POSITION_TOLERANCE_M, (
            f"the part stopped close enough to the pick spot to count as indexed ({short * 1000:.0f} mm)")
        assert info["lines_running"][1], "the healthy line stopped too"
        print(f"    part ran {travelled * 1000:.0f} mm at up to {running:.2f} m/s, then stopped "
              f"{short * 1000:.0f} mm short of the blade and held within {drift * 1000:.1f} mm "
              f"while line 1 kept running")
    finally:
        env.close()


def test_two_arms_move_independently():
    env = FactoryEnv()
    try:
        env.reset(seed=0)
        traces = [[], []]
        for _ in range(240):
            env.step(ZERO)
            for k in range(fcfg.N_WORKCELLS):
                traces[k].append(env.data.qpos[env.arms[k].qpos_adr].copy())
        a, b = np.array(traces[0]), np.array(traces[1])
        assert np.abs(a - a[0]).max() > 0.1, "workcell 0 arm never moved"
        assert np.abs(b - b[0]).max() > 0.1, "workcell 1 arm never moved"
        assert np.abs(a - b).max() > 0.1, "the two arms moved identically -- they are not independent"
        print(f"    arm travel wc0={np.abs(a - a[0]).max():.3f} rad, wc1={np.abs(b - b[0]).max():.3f} rad, "
              f"max divergence={np.abs(a - b).max():.3f} rad")
    finally:
        env.close()


def test_completion_signal_cannot_fake_recovery_and_reset_clears_messages():
    env = FactoryEnv()
    try:
        env.reset(seed=1, options={"fault_step": 5})
        for _ in range(5):
            env.step(ZERO)
        assert env.navigation.steps == 0
        info = wait_for_fault(env)
        station = info["target_workcell"]
        assert env.navigation.steps == 1
        env.step(ZERO, supervisor_signal=("accept", station))
        env.step(ZERO, supervisor_signal=("complete", station))
        for _ in range(fcfg.RECOVERY_HOLD_STEPS * 2):
            obs, _, _, _, info = env.step(ZERO)
        assert info["line_state"] == "RECOVERING" and not info["belt_running"]
        assert info["recovery_progress"] == 0 and info["recovered_step"] is None
        assert obs[-3] == 1.0 and obs[-4] == 0.0
        assert obs[-8:-4].sum() == 1
        assert len(info["mission_events"]) == 3
        # Duplicate complete requests must neither duplicate events nor reset progress.
        env.step(ZERO, supervisor_signal=("complete", station))
        assert len(env.task_manager.events) == 3
        env.reset(seed=1)
        info = env._get_info()
        assert info["mission_events"] == [] and not info["completion_requested"]
        assert env.navigation.steps == 0
    finally:
        env.close()


def test_verification_is_rechecked_before_restart():
    from humanoid_learning.envs.factory_env import LineState
    env = FactoryEnv()
    try:
        env.reset(seed=0, options={"fault_step": 0})
        info = wait_for_fault(env)
        station = info["target_workcell"]
        manager = env.task_manager
        manager.signal("accept", station, 1)
        manager.signal("complete", station, 1)
        # Test-only restoration; this is not a G1 recovery controller.
        env._set_part(station, env._part_rest_pos(station))
        mujoco.mj_forward(env.model, env.data)
        for _ in range(250):
            env.step(ZERO)
            if manager.state is LineState.RECOVERY_VERIFIED:
                break
        assert manager.state is LineState.RECOVERY_VERIFIED
        assert not manager.belt_should_run and manager.target_station == station
        env._set_part(station, env._rest_pos(env.poses[station].infeed_xy))
        mujoco.mj_forward(env.model, env.data)
        manager.update(env, 100)
        assert manager.state is LineState.RECOVERING
        assert not manager.belt_should_run
        assert manager.recovered_step is None
        assert manager.events[-1]["event"] == "verification_lost"
    finally:
        env.close()


def test_seed_reproduces_the_same_scenario_and_physics():
    def rollout(seed):
        env = FactoryEnv()
        try:
            _, info = env.reset(seed=seed)
            scenario = (info["fault_workcell"], info["fault_step"], info["scenario"])
            for _ in range(600):
                env.step(ZERO)
            return scenario, env.data.qpos.copy(), env.navigation_result()
        finally:
            env.close()

    a_scenario, a_qpos, a_nav = rollout(3)
    b_scenario, b_qpos, b_nav = rollout(3)
    assert a_scenario == b_scenario, (a_scenario, b_scenario)
    np.testing.assert_array_equal(a_qpos, b_qpos)
    assert a_nav == b_nav
    print(f"    seed 3 -> workcell {a_scenario[0]}, fault step {a_scenario[1]}, bitwise-identical qpos")


def test_different_seeds_select_both_workcells():
    seen = {}
    for seed in range(12):
        env = FactoryEnv()
        try:
            _, info = env.reset(seed=seed)
            seen.setdefault(info["fault_workcell"], []).append(seed)
        finally:
            env.close()
    assert set(seen) == set(range(fcfg.N_WORKCELLS)), f"only workcells {sorted(seen)} ever fault"
    print("    seeds 0..11 ->", {k: v for k, v in sorted(seen.items())})


def test_fault_stalls_only_its_own_cell():
    env = FactoryEnv()
    try:
        _, info = env.reset(seed=0)
        faulted = info["fault_workcell"]
        healthy = 1 - faulted
        wait_for_fault(env)
        # Let the stalled arm finish parking first: when the fault fires it
        # steps from wherever it was in the cycle to the stall waypoint, and
        # that one-off transient is real motion (~0.6 rad) that would otherwise
        # be mistaken for continued production.
        for _ in range(fcfg.ARM_CYCLE_HOLD_STEPS * 2):
            env.step(ZERO)
        # Measure the RANGE each arm sweeps, not its endpoint displacement: the
        # cycle is 360 steps long, so an arm sampled a full period later is back
        # where it started and would look motionless.
        traces = {healthy: [], faulted: []}
        for _ in range(300):
            _, _, _, _, info = env.step(ZERO)
            for k in traces:
                traces[k].append(env.data.qpos[env.arms[k].qpos_adr].copy())
        assert info["arm_faulted"][faulted] and not info["arm_faulted"][healthy]
        healthy_travel = float(np.ptp(np.array(traces[healthy]), axis=0).max())
        stalled_travel = float(np.ptp(np.array(traces[faulted]), axis=0).max())
        assert healthy_travel > 0.1, f"healthy cell stopped working too ({healthy_travel:.3f} rad)"
        assert stalled_travel < 0.05, f"faulted cell kept moving ({stalled_travel:.3f} rad)"
        # The healthy cell's part must stay where production put it.
        assert env.part_position(healthy)[2] > fcfg.TABLE_TOP_Z
        print(f"    after fault on wc{faulted}: healthy arm swept {healthy_travel:.3f} rad, "
              f"stalled arm swept {stalled_travel:.3f} rad")
    finally:
        env.close()


def test_arm_drop_puts_the_part_on_the_corridor_floor():
    """Line 1's fault must leave the part somewhere the supervisor can reach.

    The arm is not teleporting anything: it runs the same cycle with a place
    spot past the table's corridor-facing edge, opens its jaws there, and the
    part topples onto open floor. This test checks the outcome is physical --
    at rest, in real contact, inside the corridor -- not that a flag was set.
    """
    env = FactoryEnv()
    try:
        env.reset(seed=0, options={"fault_workcell": 1, "scenario": "arm_drop"})
        info = wait_for_fault(env)
        for _ in range(200):
            _, _, _, _, info = env.step(ZERO)
        part = env.part_position(1)
        dof = env._part_dof_adr[1]
        velocity = float(np.abs(env.data.qvel[dof : dof + 6]).max())

        assert env.scenario_status == "dropped_to_floor", env.scenario_status
        assert velocity < 0.02, f"part never came to rest (|qvel|max={velocity:.4f})"
        assert part[2] < fcfg.TABLE_TOP_Z - 0.3, f"the part is not on the floor: z={part[2]:.3f}"
        assert abs(part[2] - env.factory.part_half_size) < 0.01, f"part not resting on the floor: z={part[2]:.4f}"

        # Real contact, not hovering.
        support = [i for i in range(env.data.ncon)
                   if env._part_body_ids[1] in (env.model.geom_bodyid[env.data.contact[i].geom1],
                                                env.model.geom_bodyid[env.data.contact[i].geom2])]
        assert support, "dropped part has no contacts -- it is floating"
        penetration = max(-float(env.data.contact[i].dist) for i in support)
        assert penetration < 0.005, f"dropped part is sunk into the floor by {penetration * 1000:.2f} mm"

        # Inside the walking corridor, where the supervisor can stand over it.
        assert abs(part[0] - fcfg.CORRIDOR_CENTRE_X) < fcfg.CORRIDOR_HALF_WIDTH_M, (
            f"the part landed outside the corridor at x={part[0]:.3f}")
        clearance = fcfg.CORRIDOR_HALF_WIDTH_M - abs(part[0] - fcfg.CORRIDOR_CENTRE_X)
        print(f"    arm released past the table edge; part rests on the floor at "
              f"{np.round(part, 3)}, {clearance * 1000:.0f} mm inside the corridor edge, "
              f"penetration {penetration * 1000:.2f} mm")
    finally:
        env.close()


def test_observation_exposes_target_workcell_relative_pose():
    env = FactoryEnv()
    try:
        obs, info = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape
        base_dim = obs.shape[0] - FACTORY_OBS_DIM
        assert base_dim > 0

        # Before the fault the target must not be readable from the observation.
        block = obs[base_dim:]
        assert block[fcfg.N_WORKCELLS * 10] == 0.0, "fault_active set before the fault"
        assert np.all(block[fcfg.N_WORKCELLS * 10 + 1 : fcfg.N_WORKCELLS * 10 + 3] == 0.0), "target leaked early"

        wait_for_fault(env)
        obs = env._get_obs()
        block = obs[base_dim:]
        target = env.fault_workcell
        onehot = block[fcfg.N_WORKCELLS * 10 + 1 : fcfg.N_WORKCELLS * 10 + 3]
        assert onehot[target] == 1.0 and onehot.sum() == 1.0, onehot

        # Cross-check the relative pose against an independent computation.
        pelvis = env.data.xpos[env._pelvis_body_id]
        yaw = frames.yaw_from_quat(env.data.xquat[env._pelvis_body_id])
        pose = env.poses[target]
        expected_manip = frames.world_to_base(
            np.array([pose.manipulation_xy[0], pose.manipulation_xy[1], 0.0]), pelvis, yaw
        )[:2]
        expected_part = frames.world_to_base(env.part_position(target), pelvis, yaw)
        got_part = block[fcfg.N_WORKCELLS * 10 + 3 : fcfg.N_WORKCELLS * 10 + 6]
        got_manip = block[fcfg.N_WORKCELLS * 10 + 6 : fcfg.N_WORKCELLS * 10 + 8]
        np.testing.assert_allclose(got_manip, expected_manip, atol=1e-5)
        np.testing.assert_allclose(got_part, expected_part, atol=1e-5)
        print(f"    target wc{target}: manipulation pose in base frame = {np.round(got_manip, 3)} m, "
              f"part = {np.round(got_part, 3)} m")
    finally:
        env.close()


def test_reset_clears_fault_and_leaks_no_state():
    env = FactoryEnv()
    try:
        _, info = env.reset(seed=0)
        wait_for_fault(env)
        assert env.fault_active, "precondition: the episode should have faulted"

        _, info = env.reset(seed=0)
        assert not env.fault_active and info["fault_triggered_step"] is None
        assert not any(info["arm_faulted"]), info["arm_faulted"]
        assert env.navigation.steps == 0, "waiting must not consume navigation budget"
        for k in range(fcfg.N_WORKCELLS):
            # Production starts with every part entering upstream, whichever
            # surface that line's recovery target happens to be on.
            expected = env._rest_pos(env.poses[k].infeed_xy)
            np.testing.assert_allclose(env.part_position(k), expected, atol=1e-9)
        # A fresh env reset to the same seed must match this one exactly.
        other = FactoryEnv()
        try:
            other.reset(seed=0)
            np.testing.assert_array_equal(env.data.qpos, other.data.qpos)
            np.testing.assert_array_equal(env.data.qvel, other.data.qvel)
        finally:
            other.close()
        print("    reset restored both parts, cleared both arms, and matched a fresh env bit-for-bit")
    finally:
        env.close()


def test_no_forbidden_collision_while_standing():
    env = FactoryEnv()
    try:
        _, info = env.reset(seed=2)
        assert not info["forbidden_contact"], "pelvis/torso already in contact at reset"
        for _ in range(500):
            _, _, _, _, info = env.step(ZERO)
            assert not info["forbidden_contact"], "pelvis/torso collided while merely standing"
            assert not info["fallen"], "robot fell while merely standing"
        result = env.navigation_result()
        assert not result["teleported"] and result["max_base_step_m"] < 1e-3
        print(f"    500 steps standing: no forbidden contact, no fall, "
              f"max base step {result['max_base_step_m'] * 1000:.4f} mm")
    finally:
        env.close()


def test_navigation_gate_rejects_a_teleporting_oracle_and_scores_a_gradual_one():
    """The gate must not be satisfiable by sliding the base.

    Both movers below are TEST-ONLY KINEMATIC ORACLES: they write base qpos
    directly. Neither is a walking controller and neither result may be
    reported as locomotion. The point of the test is precisely that the gate
    distinguishes them from a physical trajectory.
    """
    env = FactoryEnv()
    try:
        _, info = env.reset(seed=0)
        target = env.poses[env.fault_workcell]
        gate = env.factory.navigation

        # (a) One-shot teleport straight onto the target pose.
        env.reset(seed=0)
        env.navigation.reset()
        start = env.data.xpos[env._pelvis_body_id][:2].copy()
        env.navigation.update(start, 0.0, False, False)
        env.navigation.update(np.asarray(target.manipulation_xy), target.heading_rad, False, False)
        for _ in range(gate.min_hold_steps + 5):
            env.navigation.update(np.asarray(target.manipulation_xy), target.heading_rad, False, False)
        teleport = env.navigation.result()
        assert teleport["teleported"], "a one-shot base jump was not detected"
        assert not teleport["passed"], "teleport passed the Navigation Gate"

        # (b) Gradual kinematic oracle, under the per-step displacement limit.
        env.navigation.reset()
        goal = np.asarray(target.manipulation_xy)
        n = int(np.ceil(np.linalg.norm(goal - start) / (0.5 * gate.max_base_step_m))) + 1
        for i in range(n + 1):
            t = i / n
            env.navigation.update(start + t * (goal - start), t * target.heading_rad, False, False)
        for _ in range(gate.min_hold_steps + 5):
            env.navigation.update(goal, target.heading_rad, False, False)
        gradual = env.navigation.result()
        assert not gradual["teleported"], gradual
        assert gradual["passed"], gradual
        assert not gradual["entered_wrong_workcell_first"]

        # (c) Visiting the wrong cell first must fail even if the target is reached.
        env.navigation.reset()
        wrong = np.asarray(env.poses[1 - env.fault_workcell].manipulation_xy)
        env.navigation.update(wrong, target.heading_rad, False, False)
        for _ in range(gate.min_hold_steps + 5):
            env.navigation.update(goal, target.heading_rad, False, False)
        detour = env.navigation.result()
        assert detour["entered_wrong_workcell_first"] and not detour["passed"], detour

        # (d) Falling must fail.
        env.navigation.reset()
        for i in range(n + 1):
            t = i / n
            env.navigation.update(start + t * (goal - start), t * target.heading_rad, i == n // 2, False)
        for _ in range(gate.min_hold_steps + 5):
            env.navigation.update(goal, target.heading_rad, False, False)
        assert not env.navigation.result()["passed"], "a fall passed the gate"

        print(f"    gate: teleport rejected (max step {teleport['max_base_step_m']:.3f} m > "
              f"{gate.max_base_step_m} m), gradual oracle scored pass={gradual['passed']}, "
              f"wrong-cell detour rejected, fall rejected")
    finally:
        env.close()


def test_manipulation_pose_reproduces_the_canonical_grasp_geometry():
    """STATIC GEOMETRY CHECK ONLY -- this writes the base pose directly to ask
    "if the robot were standing here, would the scene match the canonical grasp
    scene?". It is not locomotion and no navigation result is derived from it.
    """
    env = FactoryEnv()
    try:
        env.reset(seed=0)
        base_adr = env.model.jnt_qposadr[
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        ]
        # Use the env's own resting-pose expression so the 2 mm
        # OBJECT_TABLE_GAP cannot be forgotten here and re-derived wrongly.
        canonical_rest_z = float(env._part_rest_pos(0)[2])
        canonical_part_local = np.array([*fcfg.LOCAL_CANONICAL_PART_XY, canonical_rest_z])
        for pose in env.poses:
            yaw = pose.heading_rad
            env.data.qpos[base_adr : base_adr + 2] = pose.manipulation_xy
            env.data.qpos[base_adr + 3 : base_adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            mujoco.mj_forward(env.model, env.data)

            pelvis = env.data.xpos[env._pelvis_body_id]
            measured_yaw = frames.yaw_from_quat(env.data.xquat[env._pelvis_body_id])
            # Where the part BELONGS on this line -- line 0's belt pick spot,
            # line 1's table -- not wherever production has it at this instant.
            target = np.r_[pose.canonical_part_xy, canonical_rest_z]
            part_base = frames.world_to_base(target, pelvis, measured_yaw)
            # Height is relative to the floor, not the pelvis, so compare xy
            # against the canonical offset and z against the work surface.
            np.testing.assert_allclose(part_base[:2], canonical_part_local[:2], atol=1e-6)
            assert abs(target[2] - canonical_part_local[2]) < 1e-6

        print("    both lines reproduce the canonical (0.27, 0) part offset exactly, "
              "line 0 against its belt and line 1 against its table")
    finally:
        env.close()


def test_the_two_lines_face_opposite_ways_and_that_is_recorded():
    """Records the cost of putting the walking corridor BETWEEN the two lines.

    ``SharpaBimanualGraspExpert._mirrored_targets`` builds approach targets as
    ``object_pos + [-standoff, +-y_offset, height]`` -- offsets along WORLD
    axes. The previous single-belt layout had both stations facing world +X, so
    those offsets were exact. They cannot be exact for both lines now: the
    corridor runs between them, so the supervisor works line 0 facing -X and
    line 1 facing +X, and a world-axis offset that is right for one is mirrored
    for the other. This test measures that error rather than letting it be
    discovered later as a mystery 500 mm miss, and it is why the grasp side
    still needs its Cartesian offsets expressed in the robot's own frame.
    """
    standoff, y_offset = 0.25, 0.13
    errors = {}
    for pose in fcfg.FactoryConfig().workcells:
        world_axis = np.array([-standoff, y_offset])
        heading_frame = -standoff * pose.heading_dir + y_offset * pose.left_dir
        errors[pose.index] = float(np.linalg.norm(world_axis - heading_frame))
    assert errors[0] < 1e-9 or errors[1] < 1e-9, "neither line matches the world-axis convention"
    assert max(errors.values()) > 0.1, "expected one line to be mirrored; the layout may have changed"
    print(f"    world-axis approach offsets: line 0 off by {errors[0] * 1000:.0f} mm, "
          f"line 1 by {errors[1] * 1000:.0f} mm -- the grasp Expert needs heading-relative offsets")


def test_each_lines_own_fault_is_physical_and_recoverable():
    """Only the DESIGNED pairings. Each line has one failure mode because each
    presents a different surface to the corridor: line 0's belt faces in, so a
    part jams there; line 1's table faces in, so a dropped part lands there.
    Running line 0's arm-drop would put the part in the 0.2 m slot between its
    table and its arm column, which is exactly the unreachable spot this layout
    exists to avoid -- so that combination is not a scenario, and is not tested
    as though it were one."""
    assert dict(fcfg.DEFAULT_SCENARIOS) == {0: "jam", 1: "arm_drop"}
    for station, scenario in sorted(fcfg.DEFAULT_SCENARIOS.items()):
        if True:
            env = FactoryEnv()
            try:
                obs, info = env.reset(seed=station, options={"scenario": scenario, "fault_workcell": station})
                assert np.all(obs[-2:] == 0), "fault label exposed before detection"
                setter = env._set_part
                def forbid_teleport(*args, **kwargs):
                    raise AssertionError("runtime scenario teleported the part")
                env._set_part = forbid_teleport
                info = wait_for_fault(env)
                assert info["recovery_task"]["fault_type"] == scenario
                assert info["recovery_task"]["station"] == station
                assert not info["belt_running"]
                assert env._get_obs()[-2:].sum() == 1
                if scenario == "arm_drop":
                    assert info["scenario_status"] == "dropped_to_floor"
                    # Released while still carried above the table, then fell.
                    assert info["release_position"][2] > fcfg.TABLE_TOP_Z
                    assert env.part_position(station)[2] < fcfg.TABLE_TOP_Z - 0.3
                    assert env._jaw_contacts(station) == 0
                else:
                    assert info["scenario_status"] == "jammed"
                    assert env._jaw_contacts(station) == 0
                    short = float(np.linalg.norm(
                        env.part_position(station)[:2] - env.poses[station].pick_xy))
                    assert short > fcfg.RECOVERY_POSITION_TOLERANCE_M, (
                        f"a jam that is already at the pick spot is not a jam ({short * 1000:.0f} mm)")
                for _ in range(120):
                    info = env.step(ZERO)[4]
                assert info["line_state"] == "FAULT_RAISED"
                # Test harness restores the part; G1 has no recovery controller.
                env._set_part = setter
                setter(station, env._part_rest_pos(station))
                adr = env._part_qpos_adr[station]
                env.data.qpos[adr + 3:adr + 7] = [np.cos(.15), 0, 0, np.sin(.15)]
                mujoco.mj_forward(env.model, env.data)
                env.step(ZERO, supervisor_signal=("accept", station))
                env.step(ZERO, supervisor_signal=("complete", station))
                for _ in range(120):
                    info = env.step(ZERO)[4]
                assert not info["belt_running"], "misaligned returned part restarted the line"
                setter(station, env._part_rest_pos(station))
                mujoco.mj_forward(env.model, env.data)
                for _ in range(300):
                    info = env.step(ZERO)[4]
                    if info["line_state"] == "RUNNING":
                        break
                assert info["line_state"] == "RUNNING", (scenario, station, info["recovery_progress"])
                assert info["belt_running"] and info["recovery_task"] is None
                assert not any(info["arm_faulted"])
                assert env.arms[station].park_target is None
                assert not env.arms[station].release_override
                print(f"    {scenario} station={station}: fault={info['fault_triggered_step']}, "
                      f"verified={info['recovered_step']}, restarted (test-harness restoration)")
            finally:
                env.close()


def main() -> int:
    tests = [
        test_model_compiles_with_two_independent_lines,
        test_arm_links_do_not_collide_with_each_other,
        test_each_line_reproduces_the_canonical_grasp_relationship,
        test_arm_cycle_stays_above_the_belt_and_reaches_the_part,
        test_arm_actually_picks_the_part_up_and_places_it,
        test_running_belt_carries_a_part_to_the_stop_blade,
        test_belt_stops_on_fault_and_restarts_only_after_verified_recovery,
        test_completion_signal_cannot_fake_recovery_and_reset_clears_messages,
        test_verification_is_rechecked_before_restart,
        test_each_lines_own_fault_is_physical_and_recoverable,
        test_the_line_visibly_runs_and_then_visibly_stops,
        test_two_arms_move_independently,
        test_seed_reproduces_the_same_scenario_and_physics,
        test_different_seeds_select_both_workcells,
        test_fault_stalls_only_its_own_cell,
        test_arm_drop_puts_the_part_on_the_corridor_floor,
        test_observation_exposes_target_workcell_relative_pose,
        test_reset_clears_fault_and_leaks_no_state,
        test_no_forbidden_collision_while_standing,
        test_navigation_gate_rejects_a_teleporting_oracle_and_scores_a_gradual_one,
        test_manipulation_pose_reproduces_the_canonical_grasp_geometry,
        test_the_two_lines_face_opposite_ways_and_that_is_recorded,
    ]
    passed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report, do not abort the suite
            print(f"FAIL  {test.__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS  {test.__name__}", flush=True)
    print(f"\n{passed} passed, {len(tests) - passed} failed out of {len(tests)}")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
