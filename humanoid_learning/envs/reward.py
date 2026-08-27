"""Reward and success-condition computation, owned by the environment.

env.step() always returns a reward computed here, even though Behavior
Cloning does not use it. This keeps a single definition of "success" and
"reward" shared by Expert, BC, and PPO -- see PROJECT_CONTEXT.md, Reward
Ownership.
"""

from __future__ import annotations

import numpy as np


def compute_errors(
    left_ee: np.ndarray, right_ee: np.ndarray, left_target: np.ndarray, right_target: np.ndarray
) -> tuple[float, float, float]:
    left_error = float(np.linalg.norm(left_ee - left_target))
    right_error = float(np.linalg.norm(right_ee - right_target))
    mean_error = 0.5 * (left_error + right_error)
    return left_error, right_error, mean_error


def compute_success(left_error: float, right_error: float, threshold: float) -> bool:
    return left_error < threshold and right_error < threshold


def compute_reward(left_error: float, right_error: float, success: bool, success_bonus: float) -> float:
    reward = -(left_error + right_error)
    if success:
        reward += success_bonus
    return reward
