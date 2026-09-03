"""Simulation-time helpers shared by grasp controllers."""

from __future__ import annotations

import numpy as np


def sim_time_to_steps(env, seconds: float) -> int:
    """Convert seconds to outer control steps using the real model timing."""
    control_dt = float(env.model.opt.timestep) * int(env.config.frame_skip)
    if control_dt <= 0.0:
        raise ValueError(f"invalid control timestep: {control_dt}")
    return int(np.ceil(float(seconds) / control_dt))
