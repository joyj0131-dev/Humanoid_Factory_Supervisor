"""Phase 1 smoke tests for BimanualReachEnv.

Run with:
    python scripts/test_env.py
or:
    pytest scripts/test_env.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from humanoid_learning.envs import model_builder
from humanoid_learning.envs.humanoid_reach_env import BimanualReachEnv
from humanoid_learning.envs.task_config import EnvConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "environment.yaml"


def make_env() -> BimanualReachEnv:
    config = EnvConfig.from_yaml(CONFIG_PATH)
    return BimanualReachEnv(config)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


def test_model_loads():
    config = EnvConfig.from_yaml(CONFIG_PATH)
    model = model_builder.build_model(config)
    # 29 G1 hinges + 44 Sharpa hinges + 7 object freejoint.
    assert model.nq == 80
    assert model.nu == 73


def test_simulation_forward_runs():
    env = make_env()
    env.reset(seed=1)
    import mujoco

    mujoco.mj_forward(env.model, env.data)


# ---------------------------------------------------------------------
# Environment creation / reset / step
# ---------------------------------------------------------------------


def test_env_creation():
    env = make_env()
    assert env.observation_space.shape == (43,)
    assert env.action_space.shape == (14,)


def test_reset_returns_valid_obs_and_info():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "success" in info and "left_ee_error" in info and "right_ee_error" in info


def test_step_zero_action():
    env = make_env()
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_step_random_action():
    env = make_env()
    env.reset(seed=0)
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            env.reset(seed=0)


# ---------------------------------------------------------------------
# Observation / action space contracts
# ---------------------------------------------------------------------


def test_observation_dtype_shape_finite():
    env = make_env()
    obs, _ = env.reset(seed=3)
    assert obs.dtype == np.float32
    assert obs.shape == (43,)
    assert np.isfinite(obs).all()


def test_action_space_contains_extremes():
    env = make_env()
    env.reset(seed=0)
    assert env.action_space.contains(np.full(14, -1.0, dtype=np.float32))
    assert env.action_space.contains(np.full(14, 1.0, dtype=np.float32))


# ---------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------


def test_same_seed_same_object_pose():
    env = make_env()
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)
    np.testing.assert_allclose(info1["object_position"], info2["object_position"])
    np.testing.assert_allclose(obs1, obs2)


def test_different_seed_different_object_pose():
    env = make_env()
    _, info1 = env.reset(seed=1)
    _, info2 = env.reset(seed=2)
    assert not np.allclose(info1["object_position"], info2["object_position"])


# ---------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------


def test_stability_zero_action_no_nan_no_drift():
    env = make_env()
    env.reset(seed=0)
    pelvis_start = env.data.body("pelvis").xpos.copy()
    action = np.zeros(14, dtype=np.float32)
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break
    pelvis_end = env.data.body("pelvis").xpos.copy()
    drift = np.linalg.norm(pelvis_end - pelvis_start)
    assert drift < 1e-6, f"pelvis drift detected: {drift}"


def test_stability_random_action_no_explosion():
    env = make_env()
    env.reset(seed=7)
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        assert np.abs(env.data.qvel).max() < 100.0, "simulation velocity exploded"
        if terminated or truncated:
            env.reset(seed=7)


def test_object_does_not_fly_off():
    env = make_env()
    _, info = env.reset(seed=0)
    obj_start = info["object_position"].copy()
    action = np.zeros(14, dtype=np.float32)
    for _ in range(100):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    obj_end = info["object_position"]
    assert np.linalg.norm(obj_end - obj_start) < 0.05, "object moved unexpectedly with zero arm action"


# ---------------------------------------------------------------------
# End-effector
# ---------------------------------------------------------------------


def test_ee_positions_finite():
    env = make_env()
    env.reset(seed=0)
    left_ee = env.data.site_xpos[env._left_ee_site]
    right_ee = env.data.site_xpos[env._right_ee_site]
    assert np.isfinite(left_ee).all()
    assert np.isfinite(right_ee).all()


def test_left_arm_motion_moves_left_ee_only():
    # Moderate action (not the full -1.0 extreme, which drives the arm into
    # self-collision with the torso and confounds this check with contact
    # forces). Note: the waist is held by a finite-gain position actuator,
    # not a rigid weld, so driving one arm does couple a small amount of
    # motion into the other arm's EE through the shared torso (see Phase 1
    # report, Known Limitations). This test checks index-mapping
    # correctness (the intended arm dominates), not zero coupling.
    env = make_env()
    env.reset(seed=0)
    left_ee_start = env.data.site_xpos[env._left_ee_site].copy()
    right_ee_start = env.data.site_xpos[env._right_ee_site].copy()

    action = np.zeros(14, dtype=np.float32)
    action[:7] = -0.4  # left arm only (first 7 = LEFT_ARM_JOINTS)
    for _ in range(15):
        env.step(action)

    left_ee_end = env.data.site_xpos[env._left_ee_site].copy()
    right_ee_end = env.data.site_xpos[env._right_ee_site].copy()

    left_moved = np.linalg.norm(left_ee_end - left_ee_start)
    right_moved = np.linalg.norm(right_ee_end - right_ee_start)
    assert left_moved > 0.02, f"left EE barely moved: {left_moved}"
    assert right_moved < 0.2 * left_moved, (
        f"right EE moved too much relative to left ({right_moved} vs {left_moved}) "
        "-- suggests wrong actuator index mapping, not just waist coupling"
    )


def test_right_arm_motion_moves_right_ee_only():
    env = make_env()
    env.reset(seed=0)
    left_ee_start = env.data.site_xpos[env._left_ee_site].copy()
    right_ee_start = env.data.site_xpos[env._right_ee_site].copy()

    action = np.zeros(14, dtype=np.float32)
    action[7:] = -0.4  # right arm only (last 7 = RIGHT_ARM_JOINTS)
    for _ in range(15):
        env.step(action)

    left_ee_end = env.data.site_xpos[env._left_ee_site].copy()
    right_ee_end = env.data.site_xpos[env._right_ee_site].copy()

    left_moved = np.linalg.norm(left_ee_end - left_ee_start)
    right_moved = np.linalg.norm(right_ee_end - right_ee_start)
    assert right_moved > 0.02, f"right EE barely moved: {right_moved}"
    assert left_moved < 0.2 * right_moved, (
        f"left EE moved too much relative to right ({left_moved} vs {right_moved}) "
        "-- suggests wrong actuator index mapping, not just waist coupling"
    )


# ---------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------


def test_target_offset_from_object():
    env = make_env()
    _, info = env.reset(seed=0)
    object_pos = info["object_position"]
    expected_left = object_pos + np.array(env.config.left_target_offset)
    expected_right = object_pos + np.array(env.config.right_target_offset)
    np.testing.assert_allclose(env._left_target, expected_left, atol=1e-8)
    np.testing.assert_allclose(env._right_target, expected_right, atol=1e-8)


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
