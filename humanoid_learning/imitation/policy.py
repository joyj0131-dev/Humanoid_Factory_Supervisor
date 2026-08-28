"""MLP Behavior Cloning policy for BimanualReachEnv (43-dim obs -> 14-dim
action), plus checkpoint save/load.

Action normalization is intentionally NOT applied. Stored actions are
already the environment's canonical Joint Position Delta in [-1, 1]
(env.action_space contract, shared by Expert/BC/PPO per PROJECT_CONTEXT.md's
Shared Environment Rule) -- normalizing them again would be a no-op at best
and would risk a scale mismatch when PPO reuses this same action space
later. The policy's Tanh output layer enforces the [-1, 1] range directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from humanoid_learning.imitation.normalizer import Normalizer

OBS_DIM = 43
ACTION_DIM = 14


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM, hidden_sizes: tuple[int, ...] = (256, 256)):
        super().__init__()
        sizes = [obs_dim, *hidden_sizes]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-1], action_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class BCPolicy:
    """observation -> action wrapper: normalizes the raw observation, runs
    the MLP, returns a raw numpy action -- matches the project's
    ``bc_policy.act(obs)`` calling convention (PROJECT_CONTEXT.md, Policy
    Calling Convention), same shape as ``ScriptedExpert.act()``'s output."""

    def __init__(self, net: MLPPolicy, obs_normalizer: Normalizer, device: str = "cpu"):
        self.net = net.to(device)
        self.obs_normalizer = obs_normalizer
        self.device = device

    def act(self, obs: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            norm_obs = self.obs_normalizer.normalize(np.asarray(obs, dtype=np.float32))
            obs_t = torch.from_numpy(norm_obs).float().unsqueeze(0).to(self.device)
            action_t = self.net(obs_t)
        return action_t.squeeze(0).cpu().numpy().astype(np.float32)


def save_checkpoint(
    path: str | Path,
    net: MLPPolicy,
    obs_normalizer: Normalizer,
    hidden_sizes: tuple[int, ...],
    obs_dim: int,
    action_dim: int,
    extra: dict[str, Any] | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": net.state_dict(),
        "obs_mean": torch.from_numpy(obs_normalizer.mean.astype(np.float32)),
        "obs_std": torch.from_numpy(obs_normalizer.std.astype(np.float32)),
        "obs_eps": float(obs_normalizer.eps),
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "hidden_sizes": tuple(int(h) for h in hidden_sizes),
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> BCPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    net = MLPPolicy(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_sizes=checkpoint["hidden_sizes"],
    )
    net.load_state_dict(checkpoint["model_state_dict"])
    normalizer = Normalizer(
        mean=checkpoint["obs_mean"].numpy().astype(np.float64),
        std=checkpoint["obs_std"].numpy().astype(np.float64),
        eps=checkpoint["obs_eps"],
    )
    return BCPolicy(net, normalizer, device=device)
