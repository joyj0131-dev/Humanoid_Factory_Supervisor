"""Versioned complete command for Sharpa recording/replay (not a BC policy yet)."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SharpaGraspCommand:
    action: np.ndarray
    preshape_targets: np.ndarray
    noslip_iterations: int

    def __post_init__(self):
        for name in ('action', 'preshape_targets'):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError(f'{name} must be a finite vector')
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.action.shape != (25,):
            raise ValueError('Sharpa action must have 25 components')
        if not isinstance(self.noslip_iterations, (int, np.integer)) or self.noslip_iterations < 0:
            raise ValueError('noslip_iterations must be a nonnegative integer')
