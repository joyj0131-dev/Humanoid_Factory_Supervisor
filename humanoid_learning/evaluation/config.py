"""Evaluation condition config: Seen / Unseen / Noise, each an override on
top of the training EnvConfig -- see PROJECT_CONTEXT.md, Evaluation.

The same seed list is used across all three conditions (and would be reused
across policies being compared): a seed always selects the same draw from
whichever object-pose distribution the condition defines, so no condition or
policy is given an easier seed set than another -- see PROJECT_CONTEXT.md,
Fair Evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConditionConfig:
    object_x_range: tuple[float, float] | None = None
    object_y_range: tuple[float, float] | None = None
    obs_noise_std: float = 0.0
    noise_seed: int | None = None


@dataclass
class EvalConfig:
    episodes_per_condition: int = 20
    seed_base: int = 1000
    seen: ConditionConfig = None
    unseen: ConditionConfig = None
    noise: ConditionConfig = None

    def seeds(self) -> list[int]:
        return list(range(self.seed_base, self.seed_base + self.episodes_per_condition))

    @staticmethod
    def from_yaml(path: str | Path) -> "EvalConfig":
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        def condition(name: str) -> ConditionConfig:
            c = raw.get(name, {}) or {}
            return ConditionConfig(
                object_x_range=tuple(c["object_x_range"]) if "object_x_range" in c else None,
                object_y_range=tuple(c["object_y_range"]) if "object_y_range" in c else None,
                obs_noise_std=float(c.get("obs_noise_std", 0.0)),
                noise_seed=c.get("noise_seed"),
            )

        return EvalConfig(
            episodes_per_condition=int(raw.get("episodes_per_condition", 20)),
            seed_base=int(raw.get("seed_base", 1000)),
            seen=condition("seen"),
            unseen=condition("unseen"),
            noise=condition("noise"),
        )
