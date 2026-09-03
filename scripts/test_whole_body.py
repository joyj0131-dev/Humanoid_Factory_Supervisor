"""Phase 4 smoke tests: whole-body model, standing balance, posture,
planar debug mode, base-relative observation, hand synergy, grasp physics.

Run with:
    python scripts/test_whole_body.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs import frames
from humanoid_learning.envs import model_builder
from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.envs.planar_debug_env import PlanarDebugEnv
from humanoid_learning.envs.whole_body_env import ACTION_DIM, N_ARMS, N_LEGS, N_WAIST, WholeBodyEnv

ZERO_ACTION = np.zeros(ACTION_DIM, dtype=np.float32)


def make_wb_env(include_object: bool = False) -> WholeBodyEnv:
    return WholeBodyEnv(wbc.WholeBodyConfig(), include_object=include_object)


# ---------------------------------------------------------------------
# Model / builder
# ---------------------------------------------------------------------


def test_whole_body_model_loads_and_keeps_floating_base():
    env = make_wb_env()
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, wbc.FLOATING_BASE_JOINT)
    assert jid >= 0, "floating_base_joint must exist in the whole-body model"
    assert env.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE


def test_whole_body_nq_nv_nu():
    env = make_wb_env()
    # 29 G1 + 44 Sharpa hinges and one floating base.
    assert env.model.nq == 80
    assert env.model.nv == 79
    assert env.model.nu == 73


def test_fixed_base_and_whole_body_builders_are_independent():
    fixed_config = tc.EnvConfig.from_yaml(PROJECT_ROOT / "configs" / "environment.yaml")
    fixed_model = model_builder.build_model(fixed_config)
    wb_model = model_builder.build_whole_body_model(wbc.WholeBodyConfig())
    assert fixed_model.nq == 80  # 29 G1 + 44 Sharpa + 7 object freejoint
    assert wb_model.nq == 80  # 29 G1 + 44 Sharpa + 7 floating-base qpos
    jid_fixed = mujoco.mj_name2id(fixed_model, mujoco.mjtObj.mjOBJ_JOINT, wbc.FLOATING_BASE_JOINT)
    assert jid_fixed < 0, "fixed-base model must not have floating_base_joint"
    jid_wb = mujoco.mj_name2id(wb_model, mujoco.mjtObj.mjOBJ_JOINT, wbc.FLOATING_BASE_JOINT)
    assert jid_wb >= 0


def test_leg_waist_arm_hand_actuator_mapping():
    env = make_wb_env()
    for name in wbc.LEG_JOINTS + wbc.WAIST_JOINTS + tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS:
        aid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert aid >= 0, f"actuator missing: {name}"
    for side in sc.SIDES:
        for finger in sc.FINGERS:
            for suffix in sc.curl_suffixes(finger) + sc.preshape_suffixes(finger):
                name = sc.sharpa_actuator(side, finger, suffix)
                aid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                assert aid >= 0, f"hand actuator missing: {name}"
    assert N_LEGS == 12 and N_WAIST == 3 and N_ARMS == 14
    assert ACTION_DIM == 37


# ---------------------------------------------------------------------
# Reset / step / spaces
# ---------------------------------------------------------------------


def test_reset_step_shapes():
    env = make_wb_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(ZERO_ACTION)
    assert env.observation_space.contains(obs)
    assert isinstance(term, bool) and isinstance(trunc, bool)


def test_deterministic_reset():
    env = make_wb_env()
    obs1, info1 = env.reset(seed=5)
    obs2, info2 = env.reset(seed=5)
    np.testing.assert_allclose(obs1, obs2)
    np.testing.assert_allclose(info1["pelvis_pos"], info2["pelvis_pos"])


def test_no_nan_over_random_actions():
    env = make_wb_env()
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.uniform(-0.3, 0.3, size=ACTION_DIM).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(obs).all()
        if term or trunc:
            env.reset(seed=1)


# ---------------------------------------------------------------------
# Standing balance (numeric, not visual)
# ---------------------------------------------------------------------


def test_standing_balance_1000_steps():
    env = make_wb_env()
    obs, info = env.reset(seed=0)
    start_xy = info["pelvis_pos"][:2].copy()
    thresholds = env.config.standing

    min_h = float("inf")
    max_roll = max_pitch = max_drift = max_qvel = 0.0
    left_contacts = right_contacts = 0
    for _ in range(thresholds.horizon_steps):
        obs, r, term, trunc, info = env.step(ZERO_ACTION)
        assert np.isfinite(obs).all()
        min_h = min(min_h, info["pelvis_height"])
        max_roll = max(max_roll, abs(info["roll"]))
        max_pitch = max(max_pitch, abs(info["pitch"]))
        max_drift = max(max_drift, float(np.linalg.norm(info["pelvis_pos"][:2] - start_xy)))
        max_qvel = max(max_qvel, float(np.abs(env.data.qvel).max()))
        left_contacts += info["left_foot_contact"]
        right_contacts += info["right_foot_contact"]

    print(
        f"    standing: min_h={min_h:.4f} max_roll={max_roll:.4f} max_pitch={max_pitch:.4f} "
        f"max_drift={max_drift:.4f} max_qvel={max_qvel:.3f} "
        f"left_contact={left_contacts}/{thresholds.horizon_steps} right_contact={right_contacts}/{thresholds.horizon_steps}"
    )
    assert min_h > thresholds.min_pelvis_height
    assert max_roll < thresholds.max_roll
    assert max_pitch < thresholds.max_pitch
    assert max_drift < thresholds.max_xy_drift
    assert max_qvel < thresholds.max_joint_velocity
    assert left_contacts == thresholds.horizon_steps
    assert right_contacts == thresholds.horizon_steps
    assert not info["fallen"]


def test_fall_detection_triggers_on_large_disturbance():
    # A large, sustained knee-flex disturbance is known (Phase 4 report) to
    # topple this passive position-controller stance -- used here purely to
    # verify the fall-detection logic itself fires, not to validate posture.
    env = make_wb_env()
    env.reset(seed=0)
    a = ZERO_ACTION.copy()
    a[3] = 1.0
    a[9] = 1.0
    fallen = False
    for _ in range(150):
        obs, r, term, trunc, info = env.step(a)
        if info["fallen"]:
            fallen = True
            break
    assert fallen, "fall detection did not trigger for a known-destabilizing disturbance"


# ---------------------------------------------------------------------
# Posture control (small, verified-stable commands)
# ---------------------------------------------------------------------


def test_posture_small_knee_bend_and_return_no_fall():
    # Empirically swept (Phase 4 report): knee-only magnitude 0.15 for 15
    # steps LOOKS stable for ~150 steps but is actually a slow-motion fall
    # in progress (independent-joint PD control has no explicit whole-body
    # CoM compensation, so any offset from the stand target droops slowly);
    # magnitude 0.08 is genuinely stable over 400+ steps and is used here.
    env = make_wb_env()
    env.reset(seed=0)
    a = ZERO_ACTION.copy()
    for _ in range(15):
        a2 = a.copy()
        a2[3] = 0.08
        a2[9] = 0.08
        obs, r, term, trunc, info = env.step(a2)
    for _ in range(200):
        obs, r, term, trunc, info = env.step(a)  # hold; verifies a genuine new equilibrium, not a slow fall
    assert not info["fallen"]
    assert info["pelvis_height"] < 0.793 - 0.001, "expected a (small) measurable knee-bend height drop"
    assert info["left_foot_contact"] and info["right_foot_contact"]

    for _ in range(15):
        a2 = a.copy()
        a2[3] = -0.08
        a2[9] = -0.08
        obs, r, term, trunc, info = env.step(a2)
    for _ in range(50):
        obs, r, term, trunc, info = env.step(a)
    assert not info["fallen"]
    assert abs(info["pelvis_height"] - 0.793) < 0.01, "expected recovery back near stand height"
    assert info["left_foot_contact"] and info["right_foot_contact"]


def test_posture_waist_lean_and_rotate_stable():
    env = make_wb_env()
    env.reset(seed=0)
    a = ZERO_ACTION.copy()
    for waist_idx in (N_LEGS + 1, N_LEGS + 0):  # waist_roll, then waist_yaw
        env.reset(seed=0)
        for _ in range(30):
            a2 = ZERO_ACTION.copy()
            a2[waist_idx] = 0.5
            obs, r, term, trunc, info = env.step(a2)
        assert not info["fallen"]
        for _ in range(30):
            a2 = ZERO_ACTION.copy()
            a2[waist_idx] = -0.5
            obs, r, term, trunc, info = env.step(a2)
        assert not info["fallen"]
        assert info["left_foot_contact"] and info["right_foot_contact"]
        assert np.isfinite(obs).all()


def test_posture_respects_joint_limits():
    env = make_wb_env()
    env.reset(seed=0)
    a = ZERO_ACTION.copy()
    a[:N_LEGS] = 1.0  # push legs hard toward their limits
    for _ in range(60):
        obs, r, term, trunc, info = env.step(a)
        if info.get("unstable"):
            break
    if not info.get("unstable"):
        leg_qpos = env.data.qpos[env._leg_qpos_adr]
        # MuJoCo's joint-limit constraint is a soft solver constraint, not an
        # exact clamp -- under sustained saturated actuation (all legs
        # commanded to +1.0 simultaneously) a few mrad of transient
        # penetration was observed empirically (Phase 4 report); 0.01 rad
        # (~0.6 deg) tolerance distinguishes that from a real mapping bug.
        assert np.all(leg_qpos >= env._leg_ctrl_low - 1e-2)
        assert np.all(leg_qpos <= env._leg_ctrl_high + 1e-2)


# ---------------------------------------------------------------------
# Planar debug mode (NOT real locomotion -- see module docstrings)
# ---------------------------------------------------------------------


def test_planar_debug_model_independent_of_whole_body():
    planar_model = model_builder.build_planar_debug_model(wbc.PlanarDebugConfig())
    for name in ("planar_x", "planar_y", "planar_yaw"):
        assert mujoco.mj_name2id(planar_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0
    jid = mujoco.mj_name2id(planar_model, mujoco.mjtObj.mjOBJ_JOINT, wbc.FLOATING_BASE_JOINT)
    assert jid < 0, "planar debug model must not have a floating base"


def test_planar_debug_tracks_xy_yaw_commands():
    env = PlanarDebugEnv(wbc.PlanarDebugConfig())
    env.reset(seed=0)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(200):
        obs, r, term, trunc, info = env.step(a)
    assert abs(info["planar_pose"][0] - 2.0) < 0.1  # x_range upper bound
    assert abs(info["pelvis_pos"][2] - 0.793) < 1e-3, "z must stay fixed (debug mode has no z DOF)"
    assert np.isfinite(obs).all()


def test_planar_debug_deterministic_seed():
    env1 = PlanarDebugEnv(wbc.PlanarDebugConfig())
    obs1, _ = env1.reset(seed=3)
    env2 = PlanarDebugEnv(wbc.PlanarDebugConfig())
    obs2, _ = env2.reset(seed=3)
    np.testing.assert_allclose(obs1, obs2)


# ---------------------------------------------------------------------
# Base-relative observation
# ---------------------------------------------------------------------


def test_world_to_base_translation_invariance():
    world_pos = np.array([1.0, 2.0, 0.5])
    rel_a = frames.world_to_base(world_pos, base_pos=np.array([0.0, 0.0, 0.0]), base_yaw=0.0)
    rel_b = frames.world_to_base(world_pos + np.array([5.0, -3.0, 0.0]), base_pos=np.array([5.0, -3.0, 0.0]), base_yaw=0.0)
    np.testing.assert_allclose(rel_a, rel_b, atol=1e-9)


def test_world_to_base_yaw_rotation():
    world_pos = np.array([1.0, 0.0, 0.0])
    rel = frames.world_to_base(world_pos, base_pos=np.zeros(3), base_yaw=np.pi / 2)
    # rotating the world point by -90 degrees: (1,0) -> (0,-1)
    np.testing.assert_allclose(rel[:2], [0.0, -1.0], atol=1e-9)


def test_whole_body_obs_ee_relative_to_pelvis_matches_frames():
    env = make_wb_env()
    obs, info = env.reset(seed=0)
    pelvis_pos = info["pelvis_pos"]
    yaw = frames.yaw_from_quat(env.data.xquat[env._pelvis_body_id])
    expected_left_rel = frames.world_to_base(env.left_ee_pos, pelvis_pos, yaw)
    # left EE relative position occupies a fixed slice of the obs vector
    # (roll,pitch,z=3 + linvel=3 + angvel=3 + legs qpos/qvel=24 + waist
    # qpos/qvel=6 + arms qpos/qvel=28 = 67, then left_ee_rel at [67:70))
    left_rel_from_obs = obs[67:70]
    np.testing.assert_allclose(left_rel_from_obs, expected_left_rel, atol=1e-4)


# ---------------------------------------------------------------------
# Hand synergy
# ---------------------------------------------------------------------


def test_hand_synergy_open_close_within_ctrlrange():
    env = make_wb_env()
    for synergy in (0.0, 0.5, 1.0):
        for side in sc.SIDES:
            for group_idx in range(len(sc.GROUPS)):
                act_ids = env._hand_group_act_ids[side][group_idx]
                open_q = env._hand_group_open[side][group_idx]
                close_q = env._hand_group_close[side][group_idx]
                targets = open_q + synergy * (close_q - open_q)
                low = env.model.actuator_ctrlrange[act_ids, 0]
                high = env.model.actuator_ctrlrange[act_ids, 1]
                assert np.all(targets >= low - 1e-6) and np.all(targets <= high + 1e-6)


def test_hand_synergy_actuates_real_finger_joints():
    env = make_wb_env()
    env.reset(seed=0)
    left_joint_names = [
        sc.sharpa_joint("left", finger, suffix)
        for finger in sc.FINGERS for suffix in sc.curl_suffixes(finger)
    ]
    left_qpos_adr = np.array([
        env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in left_joint_names
    ])
    open_qpos = env.data.qpos[left_qpos_adr].copy()

    a = ZERO_ACTION.copy()
    a[29:33] = 1.0  # all four left Sharpa closing groups
    for _ in range(30):
        obs, r, term, trunc, info = env.step(a)
    closed_qpos = env.data.qpos[left_qpos_adr].copy()

    assert np.isfinite(closed_qpos).all()
    assert np.linalg.norm(closed_qpos - open_qpos) > 0.3, "closing synergy should visibly move the fingers"
    assert not info["fallen"]


# ---------------------------------------------------------------------
# Object-enabled whole-body smoke test.  Grasp Gate A belongs to the
# fixed-base Sharpa grasp suite; this only verifies the common floating-base
# environment remains stable when the task object and real hand actuators
# coexist.
# ---------------------------------------------------------------------


def test_object_enabled_sharpa_whole_body_runs_without_crash():
    env = make_wb_env(include_object=True)
    obs, _ = env.reset(seed=0)
    action = ZERO_ACTION.copy()
    action[29:37] = 0.3
    for _ in range(30):
        obs, _, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
        assert not terminated
        if truncated:
            break
    assert not info["fallen"]


# ---------------------------------------------------------------------
if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
