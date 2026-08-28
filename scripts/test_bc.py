"""Phase 3 smoke tests: dataset split, normalization, policy, training,
checkpoint, env inference, evaluation.

Run with:
    python scripts/test_bc.py
or:
    pytest scripts/test_bc.py -v
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from humanoid_learning.envs.humanoid_reach_env import BimanualReachEnv
from humanoid_learning.envs.task_config import EnvConfig
from humanoid_learning.evaluation import evaluator
from humanoid_learning.evaluation.config import ConditionConfig
from humanoid_learning.imitation import dataset as bc_dataset
from humanoid_learning.imitation import policy as bc_policy
from humanoid_learning.imitation.normalizer import Normalizer
from humanoid_learning.imitation.trainer import BCConfig, train_bc

ENV_CONFIG_PATH = PROJECT_ROOT / "configs" / "environment.yaml"
DATASET_DIR = PROJECT_ROOT / "datasets"


def make_env() -> BimanualReachEnv:
    return BimanualReachEnv(EnvConfig.from_yaml(ENV_CONFIG_PATH))


# ---------------------------------------------------------------------
# Dataset loading / split
# ---------------------------------------------------------------------


def test_dataset_reload():
    episodes = bc_dataset.load_episodes(DATASET_DIR)
    assert len(episodes) == 50
    for ep in episodes:
        assert ep.observations.ndim == 2 and ep.observations.shape[1] == 43
        assert ep.actions.ndim == 2 and ep.actions.shape[1] == 14
        assert ep.observations.shape[0] == ep.actions.shape[0]
        assert np.isfinite(ep.observations).all()
        assert np.isfinite(ep.actions).all()


def test_split_is_deterministic():
    episodes = bc_dataset.load_episodes(DATASET_DIR)
    train1, val1 = bc_dataset.split_episodes(episodes, val_fraction=0.2, seed=42)
    train2, val2 = bc_dataset.split_episodes(episodes, val_fraction=0.2, seed=42)
    assert [e.episode_id for e in train1] == [e.episode_id for e in train2]
    assert [e.episode_id for e in val1] == [e.episode_id for e in val2]


def test_split_different_seed_different_split():
    episodes = bc_dataset.load_episodes(DATASET_DIR)
    _, val_a = bc_dataset.split_episodes(episodes, val_fraction=0.2, seed=1)
    _, val_b = bc_dataset.split_episodes(episodes, val_fraction=0.2, seed=2)
    ids_a = {e.episode_id for e in val_a}
    ids_b = {e.episode_id for e in val_b}
    assert ids_a != ids_b


def test_split_no_episode_leakage():
    episodes = bc_dataset.load_episodes(DATASET_DIR)
    train, val = bc_dataset.split_episodes(episodes, val_fraction=0.2, seed=42)
    train_ids = {e.episode_id for e in train}
    val_ids = {e.episode_id for e in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {e.episode_id for e in episodes}
    assert len(val_ids) >= 1
    assert len(train_ids) >= 1


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------


def test_normalizer_round_trip():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=(200, 10)).astype(np.float32)
    normalizer = Normalizer.compute(data)
    normalized = normalizer.normalize(data)
    recovered = normalizer.denormalize(normalized)
    np.testing.assert_allclose(recovered, data, atol=1e-4)
    # roughly zero-mean, unit-std after normalization
    assert np.abs(normalized.mean(axis=0)).max() < 1e-3
    assert np.abs(normalized.std(axis=0) - 1.0).max() < 1e-2


def test_normalizer_constant_column_safe():
    data = np.zeros((50, 3), dtype=np.float32)
    data[:, 0] = 1.0  # constant, nonzero
    data[:, 1] = 0.0  # constant, zero
    data[:, 2] = np.linspace(-1, 1, 50)  # normal varying column
    normalizer = Normalizer.compute(data)
    normalized = normalizer.normalize(data)
    assert np.isfinite(normalized).all()
    assert np.abs(normalized[:, 0]).max() < 1e-3
    assert np.abs(normalized[:, 1]).max() < 1e-3


def test_normalizer_near_constant_column_not_amplified():
    # Regression test for the Phase 3 Noise-evaluation bug: a column that is
    # constant up to float32 storage noise (real std ~1.5e-5, like this
    # project's left/right_target z) must NOT amplify a small real-world
    # perturbation into a huge normalized value.
    rng = np.random.default_rng(0)
    data = np.full((200, 1), 0.872, dtype=np.float32)
    data += rng.normal(0, 1.5e-5, size=data.shape).astype(np.float32)
    normalizer = Normalizer.compute(data)
    assert normalizer.std[0] < 1e-4  # confirms this is the near-constant regime

    perturbed = np.array([[0.872 + 0.001]], dtype=np.float32)  # 1mm perturbation
    normalized = normalizer.normalize(perturbed)
    assert np.isfinite(normalized).all()
    assert np.abs(normalized).max() < 1.0, (
        f"near-constant column amplified a 1mm perturbation to {normalized} -- "
        "dividing by the tiny real std instead of treating it as constant"
    )


# ---------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------


def test_mlp_policy_shape_and_range():
    net = bc_policy.MLPPolicy(obs_dim=43, action_dim=14, hidden_sizes=(32, 32))
    obs = torch.randn(5, 43)
    action = net(obs)
    assert action.shape == (5, 14)
    assert torch.isfinite(action).all()
    assert action.min().item() >= -1.0 - 1e-6
    assert action.max().item() <= 1.0 + 1e-6


def test_bcpolicy_act_shape_and_range():
    net = bc_policy.MLPPolicy(obs_dim=43, action_dim=14, hidden_sizes=(16, 16))
    normalizer = Normalizer.compute(np.random.default_rng(0).normal(size=(100, 43)).astype(np.float32))
    policy = bc_policy.BCPolicy(net, normalizer)
    obs = np.random.default_rng(1).normal(size=43).astype(np.float32)
    action = policy.act(obs)
    assert action.shape == (14,)
    assert np.isfinite(action).all()
    assert (action >= -1.0 - 1e-6).all() and (action <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------
# Training smoke run
# ---------------------------------------------------------------------


def _smoke_train_config() -> BCConfig:
    return BCConfig(seed=0, val_fraction=0.2, hidden_sizes=(16, 16), batch_size=32, learning_rate=1e-3, epochs=3)


def test_training_smoke_run_executes():
    episodes = bc_dataset.load_episodes(DATASET_DIR)
    result = train_bc(episodes, _smoke_train_config())
    assert len(result.history) == 3
    for row in result.history:
        assert np.isfinite(row["train_loss"])
        assert np.isfinite(row["val_loss"])
    assert result.train_episodes + result.val_episodes == len(episodes)


def test_checkpoint_save_load_same_output():
    episodes = bc_dataset.load_episodes(DATASET_DIR)
    result = train_bc(episodes, _smoke_train_config())
    net = result.build_net(use_best=True)
    policy_before = bc_policy.BCPolicy(net, result.obs_normalizer)

    obs = episodes[0].observations[0]
    action_before = policy_before.act(obs)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.pt"
        bc_policy.save_checkpoint(
            path, net, result.obs_normalizer, result.hidden_sizes, bc_policy.OBS_DIM, bc_policy.ACTION_DIM
        )
        loaded_policy = bc_policy.load_checkpoint(path)

    action_after = loaded_policy.act(obs)
    np.testing.assert_allclose(action_before, action_after, atol=1e-6)


# ---------------------------------------------------------------------
# MuJoCo environment inference
# ---------------------------------------------------------------------


def test_env_inference_runs():
    episodes = bc_dataset.load_episodes(DATASET_DIR)
    result = train_bc(episodes, _smoke_train_config())
    policy = bc_policy.BCPolicy(result.build_net(use_best=True), result.obs_normalizer)

    env = make_env()
    obs, info = env.reset(seed=0)
    for _ in range(20):
        action = policy.act(obs)
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break


# ---------------------------------------------------------------------
# Evaluation structure / reproducibility
# ---------------------------------------------------------------------


def _random_policy() -> bc_policy.BCPolicy:
    net = bc_policy.MLPPolicy(obs_dim=43, action_dim=14, hidden_sizes=(16, 16))
    normalizer = Normalizer.compute(np.random.default_rng(0).normal(size=(100, 43)).astype(np.float32))
    return bc_policy.BCPolicy(net, normalizer)


def test_evaluation_result_structure():
    env = make_env()
    policy = _random_policy()
    results = evaluator.run_evaluation(env, policy, seeds=[0, 1, 2])
    assert len(results) == 3
    for r in results:
        assert isinstance(r.success, bool)
        assert isinstance(r.length, int)
        assert np.isfinite(r.left_ee_error)
        assert np.isfinite(r.right_ee_error)
        assert np.isfinite(r.mean_ee_error)
        assert r.failure_reason is None or isinstance(r.failure_reason, str)


def test_evaluation_seed_reproducibility_no_noise():
    policy = _random_policy()
    seeds = [10, 11, 12]

    env_a = make_env()
    results_a = evaluator.run_evaluation(env_a, policy, seeds)
    env_b = make_env()
    results_b = evaluator.run_evaluation(env_b, policy, seeds)

    for ra, rb in zip(results_a, results_b):
        assert ra.success == rb.success
        assert ra.length == rb.length
        assert ra.left_ee_error == rb.left_ee_error
        assert ra.right_ee_error == rb.right_ee_error


def test_evaluation_seed_reproducibility_with_noise():
    policy = _random_policy()
    seeds = [10, 11, 12]

    env_a = make_env()
    results_a = evaluator.run_evaluation(env_a, policy, seeds, obs_noise_std=0.01, noise_seed=7)
    env_b = make_env()
    results_b = evaluator.run_evaluation(env_b, policy, seeds, obs_noise_std=0.01, noise_seed=7)

    for ra, rb in zip(results_a, results_b):
        assert ra.success == rb.success
        assert ra.length == rb.length
        assert ra.left_ee_error == rb.left_ee_error
        assert ra.right_ee_error == rb.right_ee_error


def test_make_env_for_condition_overrides_object_range():
    base_config = EnvConfig.from_yaml(ENV_CONFIG_PATH)
    cond = ConditionConfig(object_x_range=(0.32, 0.36), object_y_range=(0.09, 0.15))
    env = evaluator.make_env_for_condition(base_config, cond.object_x_range, cond.object_y_range)
    xs, ys = [], []
    for seed in range(10):
        _, info = env.reset(seed=seed)
        xs.append(info["object_position"][0])
        ys.append(info["object_position"][1])
    assert min(xs) >= 0.32 and max(xs) <= 0.36
    assert min(ys) >= 0.09 and max(ys) <= 0.15


# ---------------------------------------------------------------------
# CLI integration (train_bc.py + evaluate_bc.py actually executed)
# ---------------------------------------------------------------------


def test_train_and_evaluate_cli_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        checkpoint_dir = tmp_path / "checkpoints"
        results_dir = tmp_path / "results"

        train_result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_bc.py"),
                "--epochs",
                "3",
                "--hidden-sizes",
                "16",
                "16",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--results-dir",
                str(results_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert train_result.returncode == 0, train_result.stderr
        assert (checkpoint_dir / "best.pt").exists()
        assert (checkpoint_dir / "last.pt").exists()
        assert (results_dir / "train_log.csv").exists()

        eval_config_path = tmp_path / "evaluation.yaml"
        eval_config_path.write_text(
            yaml.safe_dump(
                {
                    "episodes_per_condition": 3,
                    "seed_base": 1000,
                    "seen": {"object_x_range": [0.22, 0.32], "object_y_range": [-0.08, 0.08]},
                    "unseen": {"object_x_range": [0.32, 0.34], "object_y_range": [-0.05, 0.05]},
                    "noise": {
                        "object_x_range": [0.22, 0.32],
                        "object_y_range": [-0.08, 0.08],
                        "obs_noise_std": 0.01,
                        "noise_seed": 500,
                    },
                }
            )
        )

        eval_result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_bc.py"),
                "--checkpoint",
                str(checkpoint_dir / "best.pt"),
                "--eval-config",
                str(eval_config_path),
                "--results-dir",
                str(results_dir),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert eval_result.returncode == 0, eval_result.stderr
        assert "[Seen]" in eval_result.stdout
        assert "[Unseen]" in eval_result.stdout
        assert "[Noise]" in eval_result.stdout
        assert (results_dir / "eval_results.csv").exists()


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
