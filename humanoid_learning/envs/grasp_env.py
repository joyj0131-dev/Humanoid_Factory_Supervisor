"""Fixed-base grasp validation environment (Phase 4, Grasp Track).

Does NOT touch BimanualReachEnv's 43-dim observation / 14-dim action
contract -- this is a separate class for validating grasp physics only.

Fixes the bug diagnosed in PROJECT_CONTEXT.md Phase 4 (Grasp Track,
Section 1): BimanualReachEnv.step() resets every actuator NOT in the arm
action (including the 14 finger actuators) back to the stand-pose ctrl on
every call, via ``ctrl[self._held_act_ids] = self._stand_ctrl[...]``. That
meant a finger "close" target set right after ``env.step()`` returned was
immediately overwritten back to "open" on the very next ``env.step()``
call, before it had a chance to persist across the 5-substep frame_skip.
Legs/waist/hands here are simply never written after reset() instead of
being reset-to-a-snapshot every step, so whatever this env's own action
sets (arms + hand synergy) is the only thing that changes, and it persists.

Action (23-dim, Box(-1, 1)) -- EXTENDED AGAIN this session (Dynamic-Aware
Multi-Start IK + Multi-Finger Grasp/Lift/Hold) from the 19-dim (waist3 +
arms14 + hand2) contract to replace the 2 per-HAND scalar synergy channels
with 6 per-FINGER-GROUP (thumb/index/middle) scalar channels:
    [0:3)   waist        -- Joint Position Delta, radians, waist_action_scale
                            (order: yaw, roll, pitch, matching whole_body_
                            config.WAIST_JOINTS)
    [3:17)  arms         -- Joint Position Delta, radians, arm_action_scale
                            (order: left 7, right 7, same as task_config)
    [17:20) left hand groups  -- delta on a unitless open(0)<->close(1)
                            scalar per group, hand_synergy_action_scale
                            (order: thumb, index, middle)
    [20:23) right hand groups -- same, right hand (order: thumb, index,
                            middle)

Why groups instead of one scalar per hand: direct measurement (see
PROJECT_CONTEXT.md, Forearm Descent Before Wrist Alignment session and
this one) found a single per-hand scalar cannot guarantee the thumb
actually makes opposing contact -- the right hand's thumb never touched
at all through a 60-step force-regulation window while index/middle
flickered on/off, letting the object rotate/slip out from under
otherwise-plausible aggregate force readings, right before a
CONTACT_LOST failure. Splitting into 3 independently-driven groups (still
each a single open<->close scalar over that group's own joints, not raw
per-joint control) lets the controller hold a touching group steady while
a still-approaching group keeps closing, instead of one shared value
being torn between "this group needs to open" and "that group needs to
close".

Why extend the action instead of driving the waist some other way: the
coupled bilateral IK solver (humanoid_learning.expert.coupled_ik) targets
waist+both-arms as ONE 17-DoF system precisely because the waist was found
(direct A/B/D causal experiment, see PROJECT_CONTEXT.md) to be a real,
measured coupling channel between the two arms' IK -- each arm's own
Jacobian has no waist columns, so if the waist moves under reaction
torque, neither arm's per-step correction accounts for it. Solving the
waist as part of the SAME task means the resulting q_target has to be
actually reachable through this env's own control interface, the same way
arm targets already are; keeping the waist un-actuated (as before) would
mean the solver's waist_q is dead information the env can never act on.
Only THIS grasp-specific env changes -- Foundation's build_model()/
BimanualReachEnv, WholeBodyEnv, and existing 50-episode dataset/BC
checkpoints (all built against the 16-dim/no-waist contract) are
byte-for-byte untouched; nothing outside this file's own action assembly
was modified for this.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from humanoid_learning.envs import grasp_config as gc
from humanoid_learning.envs import hand_synergy
from humanoid_learning.envs import model_builder
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc

_ARM_JOINTS = tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS  # 14
_WAIST_JOINTS = wbc.WAIST_JOINTS  # 3: yaw, roll, pitch
N_WAIST = len(_WAIST_JOINTS)
N_ARMS = len(_ARM_JOINTS)
N_HAND_GROUPS = 6  # (thumb, index, middle) x (left, right)
ACTION_DIM = N_WAIST + N_ARMS + N_HAND_GROUPS  # 23
_WAIST_SLICE = slice(0, N_WAIST)
_ARM_SLICE = slice(N_WAIST, N_WAIST + N_ARMS)
_LEFT_HAND_GROUP_SLICE = slice(N_WAIST + N_ARMS, N_WAIST + N_ARMS + 3)
_RIGHT_HAND_GROUP_SLICE = slice(N_WAIST + N_ARMS + 3, ACTION_DIM)
# Fixed per-hand group order used throughout this file and by every caller
# (grasp_expert.py): thumb, index, middle.
_LEFT_GROUP_TARGETS = [wbc.LEFT_THUMB_SYNERGY_TARGETS, wbc.LEFT_INDEX_SYNERGY_TARGETS, wbc.LEFT_MIDDLE_SYNERGY_TARGETS]
_RIGHT_GROUP_TARGETS = [wbc.RIGHT_THUMB_SYNERGY_TARGETS, wbc.RIGHT_INDEX_SYNERGY_TARGETS, wbc.RIGHT_MIDDLE_SYNERGY_TARGETS]


class FixedBaseGraspEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: gc.GraspEnvConfig | None = None, render_mode: str | None = None):
        self.config = config or gc.GraspEnvConfig()
        self.render_mode = render_mode

        self.model = model_builder.build_grasp_model(self.config)
        self.data = mujoco.MjData(self.model)
        self._resolve_indices()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_ARMS * 2 + 6 + 3,), dtype=np.float32)

        self._arm_target = np.zeros(N_ARMS, dtype=np.float64)
        # Order: [left_thumb, left_index, left_middle, right_thumb,
        # right_index, right_middle] -- matches _LEFT/RIGHT_HAND_GROUP_SLICE.
        self._hand_group_synergy = np.zeros(N_HAND_GROUPS, dtype=np.float64)
        self._step_count = 0
        self._contact_streak = 0
        self._renderer: mujoco.Renderer | None = None

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

        self._arm_act_ids = np.array([act_id(n) for n in _ARM_JOINTS])
        self._arm_qpos_adr = np.array([qpos_adr(n) for n in _ARM_JOINTS])
        self._arm_dof_adr = np.array([dof_adr(n) for n in _ARM_JOINTS])
        self._arm_ctrl_low = model.actuator_ctrlrange[self._arm_act_ids, 0].copy()
        self._arm_ctrl_high = model.actuator_ctrlrange[self._arm_act_ids, 1].copy()

        self._waist_act_ids = np.array([act_id(n) for n in _WAIST_JOINTS])
        self._waist_qpos_adr = np.array([qpos_adr(n) for n in _WAIST_JOINTS])
        self._waist_dof_adr = np.array([dof_adr(n) for n in _WAIST_JOINTS])
        self._waist_ctrl_low = model.actuator_ctrlrange[self._waist_act_ids, 0].copy()
        self._waist_ctrl_high = model.actuator_ctrlrange[self._waist_act_ids, 1].copy()

        self._left_hand_act_ids = np.array([act_id(n) for n, _, _ in wbc.LEFT_HAND_SYNERGY_TARGETS])
        self._right_hand_act_ids = np.array([act_id(n) for n, _, _ in wbc.RIGHT_HAND_SYNERGY_TARGETS])
        self._left_finger_qpos_adr = np.array([qpos_adr(n) for n, _, _ in wbc.LEFT_HAND_SYNERGY_TARGETS])
        self._right_finger_qpos_adr = np.array([qpos_adr(n) for n, _, _ in wbc.RIGHT_HAND_SYNERGY_TARGETS])
        # Per-finger-group actuator ids, in the fixed [thumb, index, middle]
        # order every group-indexed array in this file/grasp_expert.py uses.
        self._left_group_act_ids = [np.array([act_id(n) for n, _, _ in g]) for g in _LEFT_GROUP_TARGETS]
        self._right_group_act_ids = [np.array([act_id(n) for n, _, _ in g]) for g in _RIGHT_GROUP_TARGETS]

        self._left_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tc.LEFT_EE_SITE)
        self._right_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tc.RIGHT_EE_SITE)
        assert self._left_ee_site >= 0 and self._right_ee_site >= 0

        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, tc.STAND_KEYFRAME)
        assert key_id >= 0
        n_robot_qpos = model.nq - 7  # object freejoint contributes 7
        self._stand_qpos = model.key_qpos[key_id][:n_robot_qpos].copy()
        self._stand_ctrl = model.key_ctrl[key_id].copy()

        obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tc.OBJECT_JOINT)
        assert obj_jid >= 0
        self._object_qpos_adr = model.jnt_qposadr[obj_jid]
        self._object_dof_adr = model.jnt_dofadr[obj_jid]
        self._object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tc.OBJECT_BODY)

        left_hand_body_names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)}
        self._left_hand_body_ids = {
            b for b in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("left_hand")
        }
        self._right_hand_body_ids = {
            b for b in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("right_hand")
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        qpos = np.zeros(self.model.nq)
        qpos[: len(self._stand_qpos)] = self._stand_qpos
        obj_x, obj_y, _ = self.config.object_pos
        obj_z = (
            self.config.table_pos[2]
            + self.config.table_half_size[2]
            + self.config.object_half_size
            + tc.OBJECT_TABLE_GAP
        )
        qpos[self._object_qpos_adr : self._object_qpos_adr + 3] = [obj_x, obj_y, obj_z]
        qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._stand_ctrl

        self._arm_target = self._stand_ctrl[self._arm_act_ids].copy()
        self._waist_target = self._stand_ctrl[self._waist_act_ids].copy()
        self._hand_group_synergy = np.zeros(N_HAND_GROUPS, dtype=np.float64)
        self._step_count = 0
        self._contact_streak = 0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, self.action_space.low, self.action_space.high)

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
        group_delta = np.concatenate([action[_LEFT_HAND_GROUP_SLICE], action[_RIGHT_HAND_GROUP_SLICE]])
        self._hand_group_synergy = np.clip(
            self._hand_group_synergy + self.config.hand_synergy_action_scale * group_delta, 0.0, 1.0
        )

        # Legs/everything-else besides waist+arms+hands is simply never
        # written after reset(), so it stays exactly at stand ctrl -- no
        # per-step "held" reset that could clobber a target set elsewhere
        # (see module docstring).
        self.data.ctrl[self._waist_act_ids] = self._waist_target
        self.data.ctrl[self._arm_act_ids] = self._arm_target
        for g in range(3):
            self.data.ctrl[self._left_group_act_ids[g]] = hand_synergy.synergy_to_targets(
                self._hand_group_synergy[g], _LEFT_GROUP_TARGETS[g]
            )
            self.data.ctrl[self._right_group_act_ids[g]] = hand_synergy.synergy_to_targets(
                self._hand_group_synergy[3 + g], _RIGHT_GROUP_TARGETS[g]
            )

        for _ in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        unstable = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        if unstable:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {"unstable": True, "step_count": self._step_count}

        info = self._get_info()
        if info["hand_object_contact"]:
            self._contact_streak += 1
        else:
            self._contact_streak = 0
        info["contact_streak"] = self._contact_streak

        truncated = self._step_count >= self.config.max_episode_steps
        return self._get_obs(), 0.0, False, truncated, info

    # ------------------------------------------------------------------
    def _hand_object_contact(self) -> tuple[bool, float]:
        max_force = 0.0
        touched = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            bodies = {self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]}
            if self._object_body_id not in bodies:
                continue
            other = (bodies - {self._object_body_id})
            other_id = other.pop() if other else self._object_body_id
            if other_id in self._left_hand_body_ids or other_id in self._right_hand_body_ids:
                touched = True
                force6 = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force6)
                max_force = max(max_force, float(np.linalg.norm(force6[:3])))
        return touched, max_force

    def _get_obs(self) -> np.ndarray:
        arm_qpos = self.data.qpos[self._arm_qpos_adr]
        arm_qvel = self.data.qvel[self._arm_dof_adr]
        left_ee = self.data.site_xpos[self._left_ee_site]
        right_ee = self.data.site_xpos[self._right_ee_site]
        obj_pos = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3]
        return np.concatenate([arm_qpos, arm_qvel, left_ee, right_ee, obj_pos]).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        touched, max_force = self._hand_object_contact()
        obj_pos = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3].copy()
        obj_vel = self.data.qvel[self._object_dof_adr : self._object_dof_adr + 6].copy()
        return {
            "step_count": self._step_count,
            "object_position": obj_pos,
            "object_velocity": obj_vel,
            "hand_object_contact": touched,
            "max_contact_force": max_force,
            "unstable": False,
        }

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
