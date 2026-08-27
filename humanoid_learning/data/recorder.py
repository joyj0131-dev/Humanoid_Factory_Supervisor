"""Records one episode's observation/action trajectory to an NPZ file.

Alignment contract (see PROJECT_CONTEXT.md, Observation / Action Alignment):
    record(obs_t, action_t) is called BEFORE env.step(action_t), so
    observations[t] is always the state action[t] was computed from, never
    the state that resulted from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EpisodeRecorder:
    def __init__(self):
        self._observations: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []

    def reset(self) -> None:
        self._observations = []
        self._actions = []

    def record(self, obs: np.ndarray, action: np.ndarray) -> None:
        self._observations.append(np.asarray(obs, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.float32))

    def __len__(self) -> int:
        return len(self._observations)

    def save(self, path: str | Path, **metadata) -> None:
        if len(self) == 0:
            raise ValueError("cannot save an empty episode")
        observations = np.stack(self._observations, axis=0)
        actions = np.stack(self._actions, axis=0)
        timesteps = np.arange(len(self), dtype=np.int64)
        np.savez(
            path,
            observations=observations,
            actions=actions,
            timesteps=timesteps,
            **metadata,
        )
