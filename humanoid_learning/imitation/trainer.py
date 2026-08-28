"""Training loop for the BC MLP policy: episode-level split, normalization
fit on train only, MSE loss, best/last checkpoint tracking.

Config (seed, batch size, LR, epochs, hidden sizes) is never hard-coded --
see PROJECT_CONTEXT.md, Coding Rules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from humanoid_learning.imitation.dataset import Episode, TimestepDataset, split_episodes
from humanoid_learning.imitation.normalizer import Normalizer
from humanoid_learning.imitation.policy import ACTION_DIM, OBS_DIM, MLPPolicy


@dataclass
class BCConfig:
    seed: int = 42
    val_fraction: float = 0.2
    hidden_sizes: tuple[int, ...] = (256, 256)
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 200

    @staticmethod
    def from_yaml(path: str | Path) -> "BCConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        kwargs = {}
        if "seed" in raw:
            kwargs["seed"] = raw["seed"]
        if "val_fraction" in raw:
            kwargs["val_fraction"] = raw["val_fraction"]
        if "model" in raw and "hidden_sizes" in raw["model"]:
            kwargs["hidden_sizes"] = tuple(raw["model"]["hidden_sizes"])
        if "training" in raw:
            tr = raw["training"]
            if "batch_size" in tr:
                kwargs["batch_size"] = tr["batch_size"]
            if "learning_rate" in tr:
                kwargs["learning_rate"] = tr["learning_rate"]
            if "weight_decay" in tr:
                kwargs["weight_decay"] = tr["weight_decay"]
            if "epochs" in tr:
                kwargs["epochs"] = tr["epochs"]
        return BCConfig(**kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainingResult:
    obs_normalizer: Normalizer
    hidden_sizes: tuple[int, ...]
    best_state_dict: dict
    last_state_dict: dict
    history: list[dict]
    best_epoch: int
    best_val_loss: float
    train_episodes: int
    val_episodes: int
    train_samples: int
    val_samples: int

    def build_net(self, use_best: bool = True) -> MLPPolicy:
        net = MLPPolicy(OBS_DIM, ACTION_DIM, self.hidden_sizes)
        net.load_state_dict(self.best_state_dict if use_best else self.last_state_dict)
        return net


def train_bc(episodes: list[Episode], config: BCConfig) -> TrainingResult:
    set_seed(config.seed)

    train_eps, val_eps = split_episodes(episodes, config.val_fraction, config.seed)
    obs_normalizer = Normalizer.compute(np.concatenate([ep.observations for ep in train_eps], axis=0))

    train_ds = TimestepDataset(train_eps, obs_normalizer)
    val_ds = TimestepDataset(val_eps, obs_normalizer)

    print(
        f"Episodes: {len(episodes)} total -> "
        f"train {len(train_eps)} episodes ({len(train_ds)} samples), "
        f"val {len(val_eps)} episodes ({len(val_ds)} samples)"
    )
    print(f"Train episode ids: {sorted(ep.episode_id for ep in train_eps)}")
    print(f"Val episode ids:   {sorted(ep.episode_id for ep in val_eps)}")

    loader_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    net = MLPPolicy(OBS_DIM, ACTION_DIM, config.hidden_sizes)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = torch.nn.MSELoss()

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state = {k: v.clone() for k, v in net.state_dict().items()}
    log_every = max(1, config.epochs // 10)

    for epoch in range(config.epochs):
        net.train()
        train_loss_sum, train_count = 0.0, 0
        for obs_batch, act_batch in train_loader:
            optimizer.zero_grad()
            pred = net(obs_batch)
            loss = loss_fn(pred, act_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * obs_batch.shape[0]
            train_count += obs_batch.shape[0]
        train_loss = train_loss_sum / train_count

        net.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                pred = net(obs_batch)
                loss = loss_fn(pred, act_batch)
                val_loss_sum += loss.item() * obs_batch.shape[0]
                val_count += obs_batch.shape[0]
        val_loss = val_loss_sum / val_count

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if epoch % log_every == 0 or epoch == config.epochs - 1:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    last_state = {k: v.clone() for k, v in net.state_dict().items()}

    return TrainingResult(
        obs_normalizer=obs_normalizer,
        hidden_sizes=config.hidden_sizes,
        best_state_dict=best_state,
        last_state_dict=last_state,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_episodes=len(train_eps),
        val_episodes=len(val_eps),
        train_samples=len(train_ds),
        val_samples=len(val_ds),
    )
