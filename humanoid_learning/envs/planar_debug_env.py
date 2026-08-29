"""DEBUG/FALLBACK-ONLY planar-base environment -- see PROJECT_CONTEXT.md
Section 8. This is NOT the final locomotion target. It exists only for:
    - Factory architecture / layout verification
    - Supervisor state-machine verification
    - Manipulation subsystem debugging

Mechanism: legs, waist, and arms are held rigid at the stand pose (they
never move -- this is not a walking gait of any kind). Base x/y/yaw are
tracked by high-gain position actuators on 3 dedicated joints added directly
to the pelvis body in place of the deleted floating_base_joint (see
model_builder.build_planar_debug_model). Because the tracking gain is high
relative to the base's own inertia, a commanded (x, y, yaw) is reached in a
fraction of a second -- i.e. this behaves like a near-teleport, not physics-
grounded locomotion. A success reported by this env's tests must never be
reported as a real-locomotion result (PROJECT_CONTEXT.md Phase 4 rule).

Action (3-dim, Box(-1, 1)): [dx, dy, dyaw], each a delta on the tracked
(x, y, yaw) target -- xy_action_scale (meters/step) and yaw_action_scale
(radians/step). This 3-dim action is a property of THIS debug env only and
is never merged into WholeBodyEnv's 31-dim action space.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from humanoid_learning.envs import model_builder
from humanoid_learning.envs import whole_body_config as wbc

ACTION_DIM = 3


class PlanarDebugEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: wbc.PlanarDebugConfig | None = None, render_mode: str | None = None):
        self.config = config or wbc.PlanarDebugConfig()
        self.render_mode = render_mode

        self.model = model_builder.build_planar_debug_model(self.config)
        self.data = mujoco.MjData(self.model)
        self._resolve_indices()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self._target = np.zeros(3, dtype=np.float64)  # x, y, yaw
        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None

    def _resolve_indices(self) -> None:
        model = self.model
        self._planar_act_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("planar_x", "planar_y", "planar_yaw")]
        )
        assert (self._planar_act_ids >= 0).all()
        self._planar_qpos_adr = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("planar_x", "planar_y", "planar_yaw")]
        )
        self._planar_low = model.actuator_ctrlrange[self._planar_act_ids, 0].copy()
        self._planar_high = model.actuator_ctrlrange[self._planar_act_ids, 1].copy()

        all_act_ids = np.arange(model.nu)
        self._held_act_ids = np.array([i for i in all_act_ids if i not in set(self._planar_act_ids)])

        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, wbc.STAND_KEYFRAME)
        assert key_id >= 0
        self._stand_ctrl = model.key_ctrl[key_id].copy()
        n_robot_qpos = model.nq - 3  # 3 planar dofs appended after the 43 robot hinge qpos
        self._stand_qpos = model.key_qpos[key_id][:n_robot_qpos].copy()

        self._pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wbc.PELVIS_BODY)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        qpos = np.zeros(self.model.nq)
        qpos[: len(self._stand_qpos)] = self._stand_qpos
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._stand_ctrl

        self._target = np.zeros(3, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        scale = np.array([self.config.xy_action_scale, self.config.xy_action_scale, self.config.yaw_action_scale])
        self._target = np.clip(self._target + scale * action, self._planar_low, self._planar_high)

        ctrl = self.data.ctrl.copy()
        ctrl[self._held_act_ids] = self._stand_ctrl[self._held_act_ids]
        ctrl[self._planar_act_ids] = self._target
        self.data.ctrl[:] = ctrl

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        unstable = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        if unstable:
            return np.zeros(3, dtype=np.float32), 0.0, True, False, {"unstable": True, "step_count": self._step_count}

        truncated = self._step_count >= self.config.max_episode_steps
        return self._get_obs(), 0.0, False, truncated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        return self.data.qpos[self._planar_qpos_adr].astype(np.float32).copy()

    def _get_info(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
            "planar_pose": self.data.qpos[self._planar_qpos_adr].copy(),
            "target": self._target.copy(),
            "pelvis_pos": self.data.xpos[self._pelvis_body_id].copy(),
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
