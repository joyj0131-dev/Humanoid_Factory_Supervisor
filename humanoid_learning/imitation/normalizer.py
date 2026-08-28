"""Z-score normalization statistics, computed once on a train split and
reused unchanged for validation/inference and (later) PPO -- see
PROJECT_CONTEXT.md, Normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_EPS = 1e-4


@dataclass
class Normalizer:
    """mean/std over the last axis.

    A dimension whose training-set std falls below ``eps`` is treated as
    effectively constant: it is normalized by subtracting the mean only
    (std treated as 1.0), instead of dividing by its tiny real std.

    This matters beyond avoiding NaN/Inf at std=0: in this project's
    Foundation dataset, ``left_target``/``right_target`` z (obs dims 39, 42)
    are mathematically constant across every episode (fixed object_z +
    fixed offset) but their measured std is ~1.5e-5, not exactly 0, due to
    float32 storage noise. Flooring std at a naive tiny epsilon (e.g. 1e-6)
    does NOT protect against this -- ``max(1.5e-5, 1e-6) == 1.5e-5`` still
    divides by the tiny real std, amplifying any real-world perturbation on
    that dimension (observation noise, later PPO exploration/observation
    noise) by ~10^4-10^5x, pushing the network far outside its training
    input range. This was discovered empirically: adding obs_noise_std as
    small as 0.001 (1mm) during Phase 3 slim evaluation collapsed success
    from 100% to 0% -- traced to exactly this amplification, not a real
    robustness finding. See PROJECT_CONTEXT.md Phase 3 report.
    """

    mean: np.ndarray
    std: np.ndarray
    eps: float = DEFAULT_EPS

    @staticmethod
    def compute(data: np.ndarray, eps: float = DEFAULT_EPS) -> "Normalizer":
        mean = data.mean(axis=0).astype(np.float64)
        std = data.std(axis=0).astype(np.float64)
        return Normalizer(mean=mean, std=std, eps=eps)

    def _safe_std(self) -> np.ndarray:
        return np.where(self.std < self.eps, 1.0, self.std)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self._safe_std()).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return (x * self._safe_std() + self.mean).astype(np.float32)
