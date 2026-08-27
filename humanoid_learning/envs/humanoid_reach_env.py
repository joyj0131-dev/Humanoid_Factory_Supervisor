"""Bimanual Pre-grasp Reaching environment for the Unitree G1 (with hands).

Shared by Expert, Behavior Cloning, and PPO (see PROJECT_CONTEXT.md, Shared
Environment Rule) -- no policy-specific variant of this class is created.

Observation (43-dim, raw/unnormalized physical units):
    left arm joint pos (7), right arm joint pos (7),
    left arm joint vel (7), right arm joint vel (7),
    left EE pos (3), right EE pos (3),
    object pos (3), left target pos (3), right target pos (3)

Action (14-dim, Box(-1, 1)):
    joint position delta for the 14 active arm joints (7 left + 7 right).
    next_target = clip(current_target + action_scale * action, joint_range)

Normalization is intentionally NOT applied here -- BC and PPO each apply
their own normalization wrapper around this same raw-observation env, using
normalization statistics that must be kept in sync between them (see
PROJECT_CONTEXT.md, Observation Design).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from humanoid_learning.envs import model_builder
from humanoid_learning.envs import reward as reward_fns
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs.task_config import EnvConfig

_ARM_JOINT_NAMES = tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS  # 14, left then right
_N_ARM = len(_ARM_JOINT_NAMES)
_N_PER_ARM = len(tc.LEFT_ARM_JOINTS)

ACTION_DIM = _N_ARM


class BimanualReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        self.config = config or EnvConfig()
        self.render_mode = render_mode

        self.model = model_builder.build_model(self.config)
        self.data = mujoco.MjData(self.model)

        self._resolve_indices()

        # left_qpos(7) + right_qpos(7) + left_qvel(7) + right_qvel(7)
        # + left_ee(3) + right_ee(3) + object(3) + left_target(3) + right_target(3)
        obs_dim = 2 * _N_ARM + 5 * 3
        assert obs_dim == 43, f"unexpected observation dim {obs_dim}"

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        self._arm_target = np.zeros(_N_ARM, dtype=np.float64)
        self._left_target = np.zeros(3, dtype=np.float64)
        self._right_target = np.zeros(3, dtype=np.float64)
        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _resolve_indices(self) -> None:
        model = self.model

        def act_id(name: str) -> int:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert aid >= 0, f"actuator not found: {name}"
            return aid

        def joint_qpos_adr(name: str) -> int:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"joint not found: {name}"
            return model.jnt_qposadr[jid]

        def joint_dof_adr(name: str) -> int:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return model.jnt_dofadr[jid]

        self._arm_act_ids = np.array([act_id(n) for n in _ARM_JOINT_NAMES])
        self._arm_qpos_adr = np.array([joint_qpos_adr(n) for n in _ARM_JOINT_NAMES])
        self._arm_dof_adr = np.array([joint_dof_adr(n) for n in _ARM_JOINT_NAMES])
        self._arm_ctrl_low = model.actuator_ctrlrange[self._arm_act_ids, 0].copy()
        self._arm_ctrl_high = model.actuator_ctrlrange[self._arm_act_ids, 1].copy()

        all_act_ids = np.arange(model.nu)
        self._held_act_ids = np.array([i for i in all_act_ids if i not in set(self._arm_act_ids)])

        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, tc.STAND_KEYFRAME)
        assert key_id >= 0, "stand keyframe not found"
        # the compiled model pads the keyframe's qpos to model.nq (robot +
        # object); only the first n_robot_qpos entries (the robot) are the
        # actual authored "stand" values, the rest is object-slot padding
        # that reset() overwrites explicitly every episode.
        n_robot_qpos = model.nq - 7
        self._stand_qpos = model.key_qpos[key_id][:n_robot_qpos].copy()
        self._stand_ctrl = model.key_ctrl[key_id].copy()

        self._left_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tc.LEFT_EE_SITE)
        self._right_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tc.RIGHT_EE_SITE)
        assert self._left_ee_site >= 0 and self._right_ee_site >= 0

        obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tc.OBJECT_JOINT)
        assert obj_jid >= 0
        self._object_qpos_adr = model.jnt_qposadr[obj_jid]

        self._n_robot_qpos = model.nq - 7  # object freejoint contributes 7

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        qpos = np.zeros(self.model.nq)
        qpos[: self._n_robot_qpos] = self._stand_qpos
        ctrl = self._stand_ctrl.copy()

        x = self.np_random.uniform(*self.config.object_x_range)
        y = self.np_random.uniform(*self.config.object_y_range)
        z = self.config.object_z
        qpos[self._object_qpos_adr : self._object_qpos_adr + 3] = [x, y, z]
        qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = ctrl

        self._arm_target = ctrl[self._arm_act_ids].copy()

        object_pos = np.array([x, y, z])
        self._left_target = object_pos + np.array(self.config.left_target_offset)
        self._right_target = object_pos + np.array(self.config.right_target_offset)

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        obs = self._get_obs()
        info = self._get_info(obs_computed=True)
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._arm_target = np.clip(
            self._arm_target + self.config.action_scale * action,
            self._arm_ctrl_low,
            self._arm_ctrl_high,
        )

        ctrl = self.data.ctrl.copy()
        ctrl[self._held_act_ids] = self._stand_ctrl[self._held_act_ids]
        ctrl[self._arm_act_ids] = self._arm_target
        self.data.ctrl[:] = ctrl

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        unstable = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        if unstable:
            # defensive fallback: NaN/Inf should not occur given the verified
            # stable configuration, but a corrupted state must never be
            # returned to a caller.
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {
                "success": False,
                "left_ee_error": float("nan"),
                "right_ee_error": float("nan"),
                "mean_ee_error": float("nan"),
                "object_position": np.full(3, np.nan),
                "step_count": self._step_count,
                "unstable": True,
            }
            return obs, 0.0, True, False, info

        obs = self._get_obs()
        info = self._get_info(obs_computed=True)

        left_error = info["left_ee_error"]
        right_error = info["right_ee_error"]
        success = info["success"]
        reward = reward_fns.compute_reward(
            left_error, right_error, success, self.config.success_bonus
        )

        self._step_count += 1
        terminated = bool(success)
        truncated = self._step_count >= self.config.max_episode_steps

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # observation / info
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        left_qpos = self.data.qpos[self._arm_qpos_adr[:_N_PER_ARM]]
        right_qpos = self.data.qpos[self._arm_qpos_adr[_N_PER_ARM:]]
        left_qvel = self.data.qvel[self._arm_dof_adr[:_N_PER_ARM]]
        right_qvel = self.data.qvel[self._arm_dof_adr[_N_PER_ARM:]]

        left_ee = self.data.site_xpos[self._left_ee_site]
        right_ee = self.data.site_xpos[self._right_ee_site]
        object_pos = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3]

        obs = np.concatenate(
            [
                left_qpos,
                right_qpos,
                left_qvel,
                right_qvel,
                left_ee,
                right_ee,
                object_pos,
                self._left_target,
                self._right_target,
            ]
        ).astype(np.float32)
        return obs

    def _get_info(self, obs_computed: bool) -> dict[str, Any]:
        left_ee = self.data.site_xpos[self._left_ee_site]
        right_ee = self.data.site_xpos[self._right_ee_site]
        left_error, right_error, mean_error = reward_fns.compute_errors(
            left_ee, right_ee, self._left_target, self._right_target
        )
        success = reward_fns.compute_success(left_error, right_error, self.config.success_threshold)
        object_pos = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3].copy()
        return {
            "success": success,
            "left_ee_error": left_error,
            "right_ee_error": right_error,
            "mean_ee_error": mean_error,
            "object_position": object_pos,
            "step_count": self._step_count,
        }

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
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
