"""Damped-least-squares scripted Expert for BimanualReachEnv.

Runs as a resolved-rate controller: act() is called once per env.step(),
does one small DLS-IK step per arm toward the target the Environment
already computed (env.left_target / env.right_target -- the Expert never
recomputes a target from object pose, see PROJECT_CONTEXT.md Target
Calculation), and returns a 14-dim Joint Position Delta action in the same
[-1, 1] space the policy uses. Convergence over an episode emerges from
repeated small steps through env.step(), the same way BC/PPO will act.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from humanoid_learning.envs.humanoid_reach_env import BimanualReachEnv
from humanoid_learning.expert import ik_solver


@dataclass
class ExpertConfig:
    damping: float = 0.05
    # gain=1.0 (raw DLS step, no scaling) saturated the action at +-1 for
    # ~70% of steps across a 15-seed sweep -- the controller was essentially
    # bang-bang most of the episode. gain=0.5 measured higher success
    # (97.5% vs ~93% over 40 seeds), shorter episodes, and ~55% saturation
    # instead of 70%, so demonstrations are smoother without sacrificing
    # reliability (see Phase 2 report).
    gain: float = 0.5


class ScriptedExpert:
    def __init__(self, env: BimanualReachEnv, config: ExpertConfig | None = None):
        self.env = env
        self.config = config or ExpertConfig()

    def act(self) -> np.ndarray:
        left_jac, right_jac = self.env.arm_jacobians()

        left_error = self.env.left_target - self.env.left_ee_pos
        right_error = self.env.right_target - self.env.right_ee_pos

        left_dq = ik_solver.dls_step(left_jac, left_error, self.config.damping) * self.config.gain
        right_dq = ik_solver.dls_step(right_jac, right_error, self.config.damping) * self.config.gain

        delta_q = np.concatenate([left_dq, right_dq])
        action = np.clip(delta_q / self.env.config.action_scale, -1.0, 1.0)
        return action.astype(np.float32)
