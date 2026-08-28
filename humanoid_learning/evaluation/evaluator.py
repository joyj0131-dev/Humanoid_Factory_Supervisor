"""Runs a policy inside the shared BimanualReachEnv (no BC-specific
environment variant -- see PROJECT_CONTEXT.md, Shared Environment Rule) and
records per-episode outcome.

EE error here is a subsystem debugging metric, not the project's final
metric (PROJECT_CONTEXT.md, Evaluation) -- it is kept because Phase 3 has no
higher-level "mission" concept yet.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from humanoid_learning.envs.humanoid_reach_env import BimanualReachEnv
from humanoid_learning.envs.task_config import EnvConfig


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    length: int
    left_ee_error: float
    right_ee_error: float
    mean_ee_error: float
    failure_reason: str | None


def _classify_failure(info: dict, success: bool, truncated: bool) -> str | None:
    if success:
        return None
    if info.get("unstable"):
        return "NUMERICAL_ERROR"
    if truncated:
        return "MAX_STEPS"
    return "UNKNOWN"


def make_env_for_condition(
    base_config: EnvConfig,
    object_x_range: tuple[float, float] | None = None,
    object_y_range: tuple[float, float] | None = None,
) -> BimanualReachEnv:
    """Builds the shared env with an object-pose-range override for a given
    evaluation condition (e.g. Unseen). Never a separate env class."""
    overrides = {}
    if object_x_range is not None:
        overrides["object_x_range"] = tuple(object_x_range)
    if object_y_range is not None:
        overrides["object_y_range"] = tuple(object_y_range)
    config = dataclasses.replace(base_config, **overrides) if overrides else base_config
    return BimanualReachEnv(config)


def run_episode(
    env: BimanualReachEnv,
    policy,
    seed: int,
    obs_noise_std: float = 0.0,
    noise_rng: np.random.Generator | None = None,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    truncated = False
    for _ in range(env.config.max_episode_steps):
        obs_in = obs
        if obs_noise_std > 0.0:
            assert noise_rng is not None, "noise_rng is required when obs_noise_std > 0"
            obs_in = obs + noise_rng.normal(0.0, obs_noise_std, size=obs.shape).astype(np.float32)
        action = policy.act(obs_in)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    success = bool(info["success"])
    return EpisodeResult(
        seed=seed,
        success=success,
        length=int(info["step_count"]),
        left_ee_error=float(info["left_ee_error"]),
        right_ee_error=float(info["right_ee_error"]),
        mean_ee_error=float(info["mean_ee_error"]),
        failure_reason=_classify_failure(info, success, truncated),
    )


def run_evaluation(
    env: BimanualReachEnv,
    policy,
    seeds: list[int],
    obs_noise_std: float = 0.0,
    noise_seed: int | None = None,
) -> list[EpisodeResult]:
    """Same fixed seed list for every policy/condition comparison (no
    cherry-picking) -- see PROJECT_CONTEXT.md, Fair Evaluation. Deterministic
    given (seeds, obs_noise_std, noise_seed): a single Generator is created
    once and consumed in seed order, so a rerun reproduces identical noise
    draws and thus identical results for a deterministic policy."""
    noise_rng = np.random.default_rng(noise_seed) if obs_noise_std > 0.0 else None
    return [run_episode(env, policy, seed, obs_noise_std, noise_rng) for seed in seeds]
