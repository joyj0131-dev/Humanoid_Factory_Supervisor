"""Phase 2 smoke tests: Expert, IK, Recorder, dataset alignment, replay, collection.

Run with:
    python scripts/test_expert.py
or:
    pytest scripts/test_expert.py -v
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import collect_demos
from humanoid_learning.data import demo_dataset
from humanoid_learning.data.recorder import EpisodeRecorder
from humanoid_learning.envs.humanoid_reach_env import BimanualReachEnv
from humanoid_learning.envs.task_config import EnvConfig
from humanoid_learning.expert.scripted_expert import ExpertConfig, ScriptedExpert

CONFIG_PATH = PROJECT_ROOT / "configs" / "environment.yaml"


def make_env() -> BimanualReachEnv:
    return BimanualReachEnv(EnvConfig.from_yaml(CONFIG_PATH))


def make_expert(env) -> ScriptedExpert:
    return ScriptedExpert(env, ExpertConfig(damping=0.05, gain=1.0))


# ---------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------


def test_expert_creation():
    env = make_env()
    env.reset(seed=0)
    expert = make_expert(env)
    assert expert is not None


def test_expert_action_shape_matches_env():
    env = make_env()
    env.reset(seed=0)
    expert = make_expert(env)
    action = expert.act()
    assert action.shape == env.action_space.shape


def test_expert_action_finite_and_in_range():
    env = make_env()
    env.reset(seed=0)
    expert = make_expert(env)
    for _ in range(20):
        action = expert.act()
        assert np.isfinite(action).all()
        assert env.action_space.contains(action)
        env.step(action)


def test_expert_left_right_arm_mapping():
    # left-arm-only error should not influence the right-arm action slice
    # (validates the expert reads env.arm_jacobians()'s left/right split
    # correctly, matching the env's own left/right actuator index mapping).
    env = make_env()
    env.reset(seed=0)
    expert = make_expert(env)
    action = expert.act()
    left_action, right_action = action[:7], action[7:]
    # with fresh reset both arms have nonzero error, so both halves should
    # generally be nonzero -- a degenerate all-zero half would indicate a
    # broken jacobian/index mapping.
    assert np.abs(left_action).sum() > 0
    assert np.abs(right_action).sum() > 0


# ---------------------------------------------------------------------
# IK
# ---------------------------------------------------------------------


def test_ik_reduces_error_over_episode():
    env = make_env()
    obs, info = env.reset(seed=0)
    expert = make_expert(env)
    initial_error = info["mean_ee_error"]
    for _ in range(env.config.max_episode_steps):
        action = expert.act()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info["mean_ee_error"] < initial_error


def test_ik_no_nan_and_respects_joint_limits():
    env = make_env()
    env.reset(seed=1)
    expert = make_expert(env)
    for _ in range(env.config.max_episode_steps):
        action = expert.act()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break
    left_q, right_q = env.arm_qpos()
    qpos = np.concatenate([left_q, right_q])
    assert np.all(qpos >= env._arm_ctrl_low - 1e-6)
    assert np.all(qpos <= env._arm_ctrl_high + 1e-6)


def test_ik_unreachable_target_fails_gracefully_no_crash():
    env = make_env()
    env.reset(seed=0)
    # force a target far outside the robot's physical workspace
    env._left_target = np.array([5.0, 5.0, 5.0])
    env._right_target = np.array([5.0, 5.0, 5.0])
    expert = make_expert(env)
    info = {"success": False}
    for _ in range(env.config.max_episode_steps):
        action = expert.act()
        assert np.isfinite(action).all()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break
    assert info["success"] is False


# ---------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------


def test_recorder_save_and_reload():
    with tempfile.TemporaryDirectory() as tmp:
        recorder = EpisodeRecorder()
        obs_dim, action_dim = 43, 14
        for t in range(5):
            recorder.record(np.full(obs_dim, t, dtype=np.float32), np.full(action_dim, -t, dtype=np.float32))
        path = Path(tmp) / "episode_000001.npz"
        recorder.save(path, success=True, seed=7)

        reloaded = demo_dataset.load_episode(path)
        assert reloaded["observations"].shape == (5, obs_dim)
        assert reloaded["actions"].shape == (5, action_dim)
        assert reloaded["observations"].dtype == np.float32
        assert reloaded["actions"].dtype == np.float32
        assert np.isfinite(reloaded["observations"]).all()
        assert bool(reloaded["success"]) is True
        assert int(reloaded["seed"]) == 7


def test_index_csv_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index.csv"
        row = {
            "episode_id": 1,
            "file": "episode_000001.npz",
            "success": True,
            "length": 42,
            "seed": 0,
            "object_x": 0.25,
            "object_y": 0.0,
            "object_z": 0.78,
            "final_left_error": 0.01,
            "final_right_error": 0.02,
        }
        demo_dataset.append_index_row(index_path, row)
        import csv

        with open(index_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["file"] == "episode_000001.npz"


# ---------------------------------------------------------------------
# Alignment / Replay
# ---------------------------------------------------------------------


def _collect_one_episode(seed: int):
    env = make_env()
    expert = make_expert(env)
    recorder = EpisodeRecorder()
    info, spawn_pos = collect_demos.run_one_attempt(env, expert, recorder, seed)
    return env, recorder, info, spawn_pos


def test_observation_action_alignment():
    env, recorder, info, spawn_pos = _collect_one_episode(seed=0)
    assert info["success"], "expected this seed to succeed for the alignment test"

    replay_env = make_env()
    obs, reset_info = replay_env.reset(seed=0)
    np.testing.assert_allclose(reset_info["object_position"], spawn_pos)

    stored_obs = np.stack(recorder._observations)
    stored_actions = np.stack(recorder._actions)
    assert stored_obs.shape[0] == stored_actions.shape[0]

    for t in range(len(stored_actions)):
        np.testing.assert_allclose(obs, stored_obs[t], atol=1e-5)
        obs, reward, terminated, truncated, replay_info = replay_env.step(stored_actions[t])
        if terminated or truncated:
            break


def test_replay_reproduces_success():
    with tempfile.TemporaryDirectory() as tmp:
        env = make_env()
        expert = make_expert(env)
        recorder = EpisodeRecorder()
        info, spawn_pos = collect_demos.run_one_attempt(env, expert, recorder, seed=0)
        assert info["success"]

        path = Path(tmp) / "episode_000001.npz"
        recorder.save(path, success=True, seed=0, object_position=spawn_pos)
        ep = demo_dataset.load_episode(path)

        replay_env = make_env()
        replay_env.reset(seed=int(ep["seed"]))
        success = False
        for t in range(len(ep["actions"])):
            obs, reward, terminated, truncated, replay_info = replay_env.step(ep["actions"][t])
            if terminated:
                success = True
                break
        assert success == bool(ep["success"])


# ---------------------------------------------------------------------
# Collection (staged: 1 / 5 / 10 attempts)
# ---------------------------------------------------------------------


def _run_collect_cli(tmp_dir: str, attempts: int, seed: int):
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "collect_demos.py"),
            "--attempts",
            str(attempts),
            "--seed",
            str(seed),
            "--output-dir",
            tmp_dir,
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return result


def test_collection_1_attempt():
    with tempfile.TemporaryDirectory() as tmp:
        result = _run_collect_cli(tmp, attempts=1, seed=0)
        assert result.returncode == 0, result.stderr
        assert (Path(tmp) / "index.csv").exists()
        npz_files = list(Path(tmp).glob("episode_*.npz"))
        assert len(npz_files) >= 1
        ep = demo_dataset.load_episode(npz_files[0])
        assert ep["observations"].shape[1] == 43
        assert ep["actions"].shape[1] == 14


def test_collection_5_attempts():
    with tempfile.TemporaryDirectory() as tmp:
        result = _run_collect_cli(tmp, attempts=5, seed=0)
        assert result.returncode == 0, result.stderr
        npz_files = list(Path(tmp).glob("episode_*.npz"))
        assert len(npz_files) >= 1
        assert "Total attempts:        5" in result.stdout


def test_collection_10_attempts():
    with tempfile.TemporaryDirectory() as tmp:
        result = _run_collect_cli(tmp, attempts=10, seed=0)
        assert result.returncode == 0, result.stderr
        npz_files = list(Path(tmp).glob("episode_*.npz"))
        assert len(npz_files) >= 5, "expected most of 10 attempts to succeed given ~90% measured rate"


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
