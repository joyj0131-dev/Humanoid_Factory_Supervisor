"""Hand synergy: maps one open<->close scalar per hand to the 7 real finger
joint targets for that hand, instead of exposing raw 14-dim finger control
to a policy -- see PROJECT_CONTEXT.md Phase 4, Section I.

synergy=0.0 -> the authored "stand" keyframe hand pose (used as OPEN).
synergy=1.0 -> each finger joint's flexion-direction ctrlrange extreme
(used as CLOSE). Values are linearly interpolated and clipped to each
joint's real ctrlrange, so an out-of-[0,1] synergy value cannot command an
out-of-limit target. Which physical grasp pose this produces is validated
empirically by the Phase 4 grasp smoke test, not assumed.
"""

from __future__ import annotations

import numpy as np

from humanoid_learning.envs import whole_body_config as wbc


def synergy_to_targets(synergy: float, targets: list[tuple[str, float, float]]) -> np.ndarray:
    """targets: list of (joint_name, open_rad, close_rad), e.g.
    wbc.LEFT_HAND_SYNERGY_TARGETS. Returns the 7 joint targets in the same
    order as ``targets``."""
    s = float(np.clip(synergy, 0.0, 1.0))
    return np.array([open_rad + s * (close_rad - open_rad) for _, open_rad, close_rad in targets], dtype=np.float64)


def left_hand_targets(synergy: float) -> np.ndarray:
    return synergy_to_targets(synergy, wbc.LEFT_HAND_SYNERGY_TARGETS)


def right_hand_targets(synergy: float) -> np.ndarray:
    return synergy_to_targets(synergy, wbc.RIGHT_HAND_SYNERGY_TARGETS)
