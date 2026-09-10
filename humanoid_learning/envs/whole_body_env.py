"""Floating-base whole-body G1 environment (Phase 4).

Shares the Foundation's Joint Position Delta convention and Gymnasium API,
but is a separate class from BimanualReachEnv (whose 43-dim/14-dim fixed-base
contract is left completely unchanged -- see PROJECT_CONTEXT.md, Foundation
Preservation). This env is the whole-body successor used for standing,
posture, and grasp validation; it is not yet wired into Expert/BC/PPO.

Action (37-dim, Box(-1, 1)), each slice a delta on its own internally
tracked target -- NOT one uniform physical unit, documented explicitly
(PROJECT_CONTEXT.md Phase 4, Section H):
    [0:12)  legs         -- Joint Position Delta, radians, leg_action_scale
    [12:15) waist        -- Joint Position Delta, radians, waist_action_scale
    [15:29) arms         -- Joint Position Delta, radians, arm_action_scale
                             (order: left 7, right 7, same as task_config)
    [29:37) Sharpa groups -- unitless open(0)<->close(1) deltas for
                             thumb/index/middle/wrap, left then right

legs are exposed as a directly-commandable slice here because Phase 4's own
standing/posture experiments need to drive them directly (see Section C/D);
this does not by itself decide which layer computes leg commands at
Mission time -- see PROJECT_CONTEXT.md Phase 4 report for that decision.

Observation is built from named pieces and concatenated -- its dimension is
computed at construction time, never hard-coded (see PROJECT_CONTEXT.md
Coding Rules): base roll/pitch/height, base linear/angular velocity
(base-yaw-frame), legs qpos/qvel, waist qpos/qvel, arms qpos/qvel, left/right
EE position (base-yaw-frame), hand synergy state, foot contact, and --
only when ``include_object=True`` -- object position (base-yaw-frame).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from humanoid_learning.envs import frames
from humanoid_learning.envs import model_builder
from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc

_ARM_JOINTS = tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS  # 14, left then right
N_LEGS = len(wbc.LEG_JOINTS)  # 12
N_WAIST = len(wbc.WAIST_JOINTS)  # 3
N_ARMS = len(_ARM_JOINTS)  # 14
N_HAND_SYNERGY = 2 * len(sc.GROUPS)  # 8
ACTION_DIM = N_LEGS + N_WAIST + N_ARMS + N_HAND_SYNERGY  # 37

_LEG_SLICE = slice(0, N_LEGS)
_WAIST_SLICE = slice(N_LEGS, N_LEGS + N_WAIST)
_ARM_SLICE = slice(N_LEGS + N_WAIST, N_LEGS + N_WAIST + N_ARMS)
_HAND_SLICE = slice(N_LEGS + N_WAIST + N_ARMS, ACTION_DIM)


class WholeBodyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: wbc.WholeBodyConfig | None = None, include_object: bool = False, render_mode: str | None = None):
        self.config = config or wbc.WholeBodyConfig()
        self.include_object = include_object
        self.render_mode = render_mode

        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self._resolve_indices()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        self._leg_target = np.zeros(N_LEGS, dtype=np.float64)
        self._waist_target = np.zeros(N_WAIST, dtype=np.float64)
        self._arm_target = np.zeros(N_ARMS, dtype=np.float64)
        self._hand_synergy = np.zeros(N_HAND_SYNERGY, dtype=np.float64)
        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None

        self.reset(seed=0)
        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------
    def _build_model(self) -> mujoco.MjModel:
        """Subclass hook: FactoryEnv builds the same G1 inside a two-workcell
        scene. Overriding this is the only model-side change a subclass needs;
        the action convention, index resolution and step semantics below are
        then shared rather than reimplemented."""
        return model_builder.build_whole_body_model(self.config, include_object=self.include_object)

    # ------------------------------------------------------------------
    def _resolve_indices(self) -> None:
        model = self.model

        def act_id(name: str) -> int:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert aid >= 0, f"actuator not found: {name}"
            return aid

        def qpos_adr(name: str) -> int:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"joint not found: {name}"
            return model.jnt_qposadr[jid]

        def dof_adr(name: str) -> int:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return model.jnt_dofadr[jid]

        self._leg_act_ids = np.array([act_id(n) for n in wbc.LEG_JOINTS])
        self._leg_qpos_adr = np.array([qpos_adr(n) for n in wbc.LEG_JOINTS])
        self._leg_dof_adr = np.array([dof_adr(n) for n in wbc.LEG_JOINTS])
        self._leg_ctrl_low = model.actuator_ctrlrange[self._leg_act_ids, 0].copy()
        self._leg_ctrl_high = model.actuator_ctrlrange[self._leg_act_ids, 1].copy()

        self._waist_act_ids = np.array([act_id(n) for n in wbc.WAIST_JOINTS])
        self._waist_qpos_adr = np.array([qpos_adr(n) for n in wbc.WAIST_JOINTS])
        self._waist_dof_adr = np.array([dof_adr(n) for n in wbc.WAIST_JOINTS])
        self._waist_ctrl_low = model.actuator_ctrlrange[self._waist_act_ids, 0].copy()
        self._waist_ctrl_high = model.actuator_ctrlrange[self._waist_act_ids, 1].copy()

        self._arm_act_ids = np.array([act_id(n) for n in _ARM_JOINTS])
        self._arm_qpos_adr = np.array([qpos_adr(n) for n in _ARM_JOINTS])
        self._arm_dof_adr = np.array([dof_adr(n) for n in _ARM_JOINTS])
        self._arm_ctrl_low = model.actuator_ctrlrange[self._arm_act_ids, 0].copy()
        self._arm_ctrl_high = model.actuator_ctrlrange[self._arm_act_ids, 1].copy()

        self._hand_group_act_ids: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        self._hand_group_open: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        self._hand_group_close: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        for side in sc.SIDES:
            for group in sc.GROUPS:
                aids, opens, closes = [], [], []
                for finger in sc.GROUP_FINGERS[group]:
                    for suffix in sc.curl_suffixes(finger):
                        jname = sc.sharpa_joint(side, finger, suffix)
                        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                        lo, hi = model.jnt_range[jid]
                        aids.append(act_id(sc.sharpa_actuator(side, finger, suffix)))
                        opens.append(lo)
                        closes.append(lo + sc.CLOSE_FRACTION * (hi - lo))
                self._hand_group_act_ids[side].append(np.asarray(aids, dtype=int))
                self._hand_group_open[side].append(np.asarray(opens, dtype=float))
                self._hand_group_close[side].append(np.asarray(closes, dtype=float))

        self._pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wbc.PELVIS_BODY)
        self._left_foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wbc.LEFT_FOOT_CONTACT_BODY)
        self._right_foot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wbc.RIGHT_FOOT_CONTACT_BODY)
        self._floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        assert self._floor_geom_id >= 0

        self._left_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tc.LEFT_EE_SITE)
        self._right_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tc.RIGHT_EE_SITE)
        assert self._left_ee_site >= 0 and self._right_ee_site >= 0

        self._stand_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, wbc.STAND_KEYFRAME)
        assert self._stand_key_id >= 0

        if self.include_object:
            obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tc.OBJECT_JOINT)
            assert obj_jid >= 0
            self._object_qpos_adr = model.jnt_qposadr[obj_jid]
            self._object_dof_adr = model.jnt_dofadr[obj_jid]

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._stand_key_id)

        if self.include_object:
            x, y, z = self.config.table_pos
            object_pos = np.array(
                [x, y, z + self.config.table_half_size[2] + tc.OBJECT_HALF_SIZE + tc.OBJECT_TABLE_GAP]
            )
            self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3] = object_pos
            self.data.qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)

        self._leg_target = self.data.ctrl[self._leg_act_ids].copy()
        self._waist_target = self.data.ctrl[self._waist_act_ids].copy()
        self._arm_target = self.data.ctrl[self._arm_act_ids].copy()
        self._hand_synergy = np.zeros(N_HAND_SYNERGY, dtype=np.float64)
        for side in sc.SIDES:
            for group_idx in range(len(sc.GROUPS)):
                self.data.ctrl[self._hand_group_act_ids[side][group_idx]] = self._hand_group_open[side][group_idx]
        self._step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._leg_target = np.clip(
            self._leg_target + self.config.leg_action_scale * action[_LEG_SLICE],
            self._leg_ctrl_low,
            self._leg_ctrl_high,
        )
        self._waist_target = np.clip(
            self._waist_target + self.config.waist_action_scale * action[_WAIST_SLICE],
            self._waist_ctrl_low,
            self._waist_ctrl_high,
        )
        self._arm_target = np.clip(
            self._arm_target + self.config.arm_action_scale * action[_ARM_SLICE],
            self._arm_ctrl_low,
            self._arm_ctrl_high,
        )
        self._hand_synergy = np.clip(
            self._hand_synergy + self.config.hand_synergy_action_scale * action[_HAND_SLICE],
            0.0,
            1.0,
        )

        ctrl = self.data.ctrl.copy()
        ctrl[self._leg_act_ids] = self._leg_target
        ctrl[self._waist_act_ids] = self._waist_target
        ctrl[self._arm_act_ids] = self._arm_target
        for side_idx, side in enumerate(sc.SIDES):
            for group_idx in range(len(sc.GROUPS)):
                synergy = self._hand_synergy[side_idx * len(sc.GROUPS) + group_idx]
                open_q = self._hand_group_open[side][group_idx]
                close_q = self._hand_group_close[side][group_idx]
                ctrl[self._hand_group_act_ids[side][group_idx]] = open_q + synergy * (close_q - open_q)
        self.data.ctrl[:] = ctrl

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        unstable = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        if unstable:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"unstable": True, "fallen": True, "step_count": self._step_count}
            return obs, 0.0, True, False, info

        obs = self._get_obs()
        info = self._get_info()
        terminated = False
        truncated = self._step_count >= self.config.max_episode_steps
        return obs, 0.0, terminated, truncated, info

    # ------------------------------------------------------------------
    # observation / info
    # ------------------------------------------------------------------
    def _foot_contacts(self) -> np.ndarray:
        left = right = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            bodies = {b1, b2}
            geoms = {c.geom1, c.geom2}
            if self._floor_geom_id not in geoms:
                continue
            if self._left_foot_body_id in bodies:
                left = 1.0
            if self._right_foot_body_id in bodies:
                right = 1.0
        return np.array([left, right], dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        pelvis_pos = self.data.xpos[self._pelvis_body_id].copy()
        pelvis_quat = self.data.xquat[self._pelvis_body_id].copy()
        yaw = frames.yaw_from_quat(pelvis_quat)
        roll, pitch = frames.roll_pitch_from_quat(pelvis_quat)

        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._pelvis_body_id, vel6, 0)
        base_ang_vel = frames.world_vel_to_base(vel6[:3], yaw)
        base_lin_vel = frames.world_vel_to_base(vel6[3:], yaw)

        legs_qpos = self.data.qpos[self._leg_qpos_adr]
        legs_qvel = self.data.qvel[self._leg_dof_adr]
        waist_qpos = self.data.qpos[self._waist_qpos_adr]
        waist_qvel = self.data.qvel[self._waist_dof_adr]
        arms_qpos = self.data.qpos[self._arm_qpos_adr]
        arms_qvel = self.data.qvel[self._arm_dof_adr]

        left_ee_rel = frames.world_to_base(self.data.site_xpos[self._left_ee_site], pelvis_pos, yaw)
        right_ee_rel = frames.world_to_base(self.data.site_xpos[self._right_ee_site], pelvis_pos, yaw)

        pieces = [
            np.array([roll, pitch, pelvis_pos[2]]),
            base_lin_vel,
            base_ang_vel,
            legs_qpos,
            legs_qvel,
            waist_qpos,
            waist_qvel,
            arms_qpos,
            arms_qvel,
            left_ee_rel,
            right_ee_rel,
            self._hand_synergy,
            self._foot_contacts(),
        ]
        if self.include_object:
            obj_world = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3]
            pieces.append(frames.world_to_base(obj_world, pelvis_pos, yaw))

        return np.concatenate(pieces).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        pelvis_pos = self.data.xpos[self._pelvis_body_id].copy()
        pelvis_quat = self.data.xquat[self._pelvis_body_id].copy()
        roll, pitch = frames.roll_pitch_from_quat(pelvis_quat)
        foot_contact = self._foot_contacts()
        thresholds = self.config.standing
        fallen = (
            pelvis_pos[2] < thresholds.min_pelvis_height
            or abs(roll) > thresholds.max_roll
            or abs(pitch) > thresholds.max_pitch
        )
        info: dict[str, Any] = {
            "step_count": self._step_count,
            "pelvis_pos": pelvis_pos,
            "pelvis_height": float(pelvis_pos[2]),
            "roll": roll,
            "pitch": pitch,
            "left_foot_contact": bool(foot_contact[0]),
            "right_foot_contact": bool(foot_contact[1]),
            "fallen": bool(fallen),
            "unstable": False,
        }
        if self.include_object:
            info["object_position"] = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3].copy()
            info["object_velocity"] = self.data.qvel[self._object_dof_adr : self._object_dof_adr + 6].copy()
        return info

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

    # ------------------------------------------------------------------
    # public accessors (mirrors BimanualReachEnv's Expert-facing accessors)
    # ------------------------------------------------------------------
    @property
    def left_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._left_ee_site].copy()

    @property
    def right_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._right_ee_site].copy()

    def arm_jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._left_ee_site)
        left_jac = jacp[:, self._arm_dof_adr[:7]].copy()
        jacp[:] = 0.0
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._right_ee_site)
        right_jac = jacp[:, self._arm_dof_adr[7:]].copy()
        return left_jac, right_jac

    def arm_qpos(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[self._arm_qpos_adr[:7]].copy(), self.data.qpos[self._arm_qpos_adr[7:]].copy()
