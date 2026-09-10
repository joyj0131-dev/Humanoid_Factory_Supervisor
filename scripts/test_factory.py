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

# The compiled G1+Sharpa actuator contract this project has locked in.
G1_ACTUATOR_COUNT = 73


def test_model_compiles_with_two_independent_workcells():
    model = build_factory_model(fcfg.FactoryConfig())
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    assert len(names) == len(set(names)), "duplicate actuator names"
    workcell_actuators = [n for n in names if n and n.startswith("wc")]
    assert len(workcell_actuators) == fcfg.N_WORKCELLS * len(fcfg.ARM_JOINT_SUFFIXES)
    assert model.nu - len(workcell_actuators) == G1_ACTUATOR_COUNT, "G1 actuator contract changed"

    for kind, namer in (
        (mujoco.mjtObj.mjOBJ_BODY, lambda k: fcfg.part_body_name(k)),
        (mujoco.mjtObj.mjOBJ_BODY, lambda k: fcfg.table_body_name(k)),
        (mujoco.mjtObj.mjOBJ_GEOM, lambda k: fcfg.table_geom_name(k)),
        (mujoco.mjtObj.mjOBJ_JOINT, lambda k: fcfg.part_joint_name(k)),
    ):
        ids = [mujoco.mj_name2id(model, kind, namer(k)) for k in range(fcfg.N_WORKCELLS)]
        assert all(i >= 0 for i in ids), f"missing {[namer(k) for k in range(fcfg.N_WORKCELLS)]}"
        assert len(set(ids)) == len(ids), "workcells share an element"

    # The floating base must still be free: navigation has to come from legs.
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    assert base >= 0 and model.jnt_type[base] == mujoco.mjtJoint.mjJNT_FREE


def test_workcell_poses_are_one_rigid_transform_of_the_canonical_scene():
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

    separation = float(np.linalg.norm(np.asarray(poses[0].manipulation_xy) - np.asarray(poses[1].manipulation_xy)))
    assert separation > 2.0, f"workcells too close for locomotion to be required: {separation:.3f} m"
    print(f"    workcell manipulation-pose separation = {separation:.3f} m")


def test_arm_cycle_never_drives_the_tip_through_the_table():
    config = fcfg.FactoryConfig()
    model = build_factory_model(config)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand"))
    waypoints = fcfg.arm_cycle_waypoints()
    worst = math.inf
    pick_local = None
    for k, pose in enumerate(config.workcells):
        adr = [
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fcfg.arm_joint_name(k, s))]
            for s in fcfg.ARM_JOINT_SUFFIXES
        ]
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, fcfg.arm_body_name(k, "tip"))
        for index, waypoint in enumerate(waypoints):
            for a, v in zip(adr, waypoint):
                data.qpos[a] = v
            mujoco.mj_forward(model, data)
            tip = data.site_xpos[site]
            worst = min(worst, float(tip[2]) - fcfg.TABLE_TOP_Z)
            if index == fcfg.ARM_FAULT_WAYPOINT_INDEX:
                local = pose.to_local_xy(tip[:2])
                if pick_local is None:
                    pick_local = local
                # Both cells must reach the same spot in their own frames.
                assert np.allclose(local, pick_local, atol=1e-6)
    assert worst > 0.0, f"arm cycle reaches {worst:.3f} m relative to the tabletop"
    # The pick waypoint should actually be over the table, not off to one side.
    assert np.allclose(pick_local, fcfg.LOCAL_CANONICAL_PART_XY, atol=0.05), pick_local
    print(f"    worst tabletop clearance over the cycle = {worst:+.3f} m; pick tip local = {np.round(pick_local, 3)}")


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


def test_seed_reproduces_the_same_scenario_and_physics():
    def rollout(seed):
        env = FactoryEnv()
        try:
            _, info = env.reset(seed=seed)
            scenario = (info["fault_workcell"], info["fault_step"], tuple(np.round(env.drop_local_xy, 12)))
            for _ in range(260):
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
        while not env.step(ZERO)[4]["fault_active"]:
            pass
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


def test_dropped_part_settles_on_a_real_surface():
    env = FactoryEnv()
    try:
        _, info = env.reset(seed=0)
        faulted = info["fault_workcell"]
        for _ in range(600):
            _, _, _, _, info = env.step(ZERO)
        part = env.part_position(faulted)
        dof = env._part_dof_adr[faulted]
        velocity = float(np.abs(env.data.qvel[dof : dof + 6]).max())
        resting_z = fcfg.TABLE_TOP_Z + env.factory.part_half_size

        assert velocity < 0.02, f"part never came to rest (|qvel|max={velocity:.4f})"
        assert abs(part[2] - resting_z) < 0.01, f"part not resting on the tabletop: z={part[2]:.4f}"
        assert part[2] > 0.1, "part fell through the world"

        # It must be supported by real contact, not hovering.
        support = [
            i
            for i in range(env.data.ncon)
            if env._part_body_ids[faulted]
            in (env.model.geom_bodyid[env.data.contact[i].geom1], env.model.geom_bodyid[env.data.contact[i].geom2])
        ]
        assert support, "dropped part has no contacts -- it is floating"
        penetration = max(-float(env.data.contact[i].dist) for i in support)
        assert penetration < 0.005, f"dropped part is sunk into its support by {penetration * 1000:.2f} mm"

        local = env.poses[faulted].to_local_xy(part[:2])
        assert np.allclose(local, env.drop_local_xy, atol=0.02), (local, env.drop_local_xy)
        print(f"    dropped part rest z={part[2]:.4f} (expected {resting_z:.4f}), |qvel|max={velocity:.5f}, "
              f"penetration={penetration * 1000:.2f} mm, local xy={np.round(local, 3)}")
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

        while not env.step(ZERO)[4]["fault_active"]:
            pass
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
        for _ in range(400):
            env.step(ZERO)
        assert env.fault_active, "precondition: the episode should have faulted"

        _, info = env.reset(seed=0)
        assert not env.fault_active and info["fault_triggered_step"] is None
        assert not any(info["arm_faulted"]), info["arm_faulted"]
        assert env.navigation.steps == 1, "navigation tracker carried over"
        for k in range(fcfg.N_WORKCELLS):
            expected = env._part_rest_pos(k, fcfg.LOCAL_CANONICAL_PART_XY)
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
        canonical_rest_z = float(env._part_rest_pos(0, fcfg.LOCAL_CANONICAL_PART_XY)[2])
        canonical_part_local = np.array([*fcfg.LOCAL_CANONICAL_PART_XY, canonical_rest_z])
        for pose in env.poses:
            yaw = pose.heading_rad
            env.data.qpos[base_adr : base_adr + 2] = pose.manipulation_xy
            env.data.qpos[base_adr + 3 : base_adr + 7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            mujoco.mj_forward(env.model, env.data)

            pelvis = env.data.xpos[env._pelvis_body_id]
            measured_yaw = frames.yaw_from_quat(env.data.xquat[env._pelvis_body_id])
            part_base = frames.world_to_base(env.part_position(pose.index), pelvis, measured_yaw)
            # Height is relative to the floor, not the pelvis, so compare xy
            # against the canonical offset and z against the tabletop.
            np.testing.assert_allclose(part_base[:2], canonical_part_local[:2], atol=1e-6)
            assert abs(env.part_position(pose.index)[2] - canonical_part_local[2]) < 1e-6

            table_base = frames.world_to_base(pose.table_pos, pelvis, measured_yaw)
            np.testing.assert_allclose(table_base[:2], fcfg.LOCAL_TABLE_POS[:2], atol=1e-6)
        print("    both cells reproduce the canonical (0.27, 0) part offset and (0.30, 0) table offset exactly")
    finally:
        env.close()


def test_existing_grasp_expert_offsets_are_world_axis_and_do_not_transfer():
    """Quantifies a real blocker rather than assuming transfer works.

    ``SharpaBimanualGraspExpert._mirrored_targets`` builds its approach targets
    as ``object_pos + [-standoff, +-y_offset, height]`` -- offsets along WORLD
    axes, not along the robot's heading. At a workcell rotated by 35 degrees
    that points the approach in the wrong direction. This test measures the
    resulting error so the number is on record; it is expected to be large.
    """
    standoff, y_offset = 0.25, 0.13
    worst = 0.0
    for pose in fcfg.FactoryConfig().workcells:
        heading = pose.heading_rad
        # What the expert would command (world axes) vs what the same offset
        # means in the robot's own heading frame.
        world_axis = np.array([-standoff, y_offset])
        heading_frame = -standoff * pose.heading_dir + y_offset * pose.left_dir
        worst = max(worst, float(np.linalg.norm(world_axis - heading_frame)))
    assert worst > 0.05, "offsets unexpectedly agree -- re-check the layout"
    print(f"    world-axis vs heading-frame approach offset differs by up to {worst * 1000:.0f} mm "
          f"at 35 deg -- the grasp Expert needs a heading-frame rewrite before it can run in a cell")


def main() -> int:
    tests = [
        test_model_compiles_with_two_independent_workcells,
        test_workcell_poses_are_one_rigid_transform_of_the_canonical_scene,
        test_arm_cycle_never_drives_the_tip_through_the_table,
        test_two_arms_move_independently,
        test_seed_reproduces_the_same_scenario_and_physics,
        test_different_seeds_select_both_workcells,
        test_fault_stalls_only_its_own_cell,
        test_dropped_part_settles_on_a_real_surface,
        test_observation_exposes_target_workcell_relative_pose,
        test_reset_clears_fault_and_leaks_no_state,
        test_no_forbidden_collision_while_standing,
        test_navigation_gate_rejects_a_teleporting_oracle_and_scores_a_gradual_one,
        test_manipulation_pose_reproduces_the_canonical_grasp_geometry,
        test_existing_grasp_expert_offsets_are_world_axis_and_do_not_transfer,
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
