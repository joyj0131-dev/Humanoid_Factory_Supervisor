"""Sharpa Wave fixed-base grasp environment (Phase 4, 35th session, Stage 4).

A SEPARATE env from FixedBaseGraspEnv (Dex3) -- not a subclass, not a
hand-model branch inside it. FixedBaseGraspEnv's action/observation
contract, hand_synergy group tables, and force-safety bookkeeping are all
keyed to Dex3's 3-finger-group body-name convention
("left_hand_thumb"/"left_hand_index"/"left_hand_middle" prefixes); Sharpa
has 5 fingers, a different body-name prefix ("left_left_<finger>"), and a
4-group (thumb/index/middle/wrap) closing scheme (see sharpa_config.py) --
forcing that through FixedBaseGraspEnv's existing per-Dex3-group code
would either silently rely on Dex3 names (breaking) or need enough
conditionals to obscure both paths. Keeping them separate means
FixedBaseGraspEnv is provably unaffected (verified: test_env/test_expert/
test_whole_body/test_grasp all still pass byte-for-byte) and this file
can be read on its own.

Action (25-dim, Box(-1,1)):
    [0:3)   waist        -- same convention as FixedBaseGraspEnv
    [3:17)  arms         -- same convention as FixedBaseGraspEnv
    [17:21) left hand groups  -- CURL-only synergy delta, order
                            (thumb, index, middle, wrap), matching
                            sharpa_config.GROUPS
    [21:25) right hand groups -- same, right hand

Preshape (spread/thumb-opposition) joints are NOT part of the action --
they are a "set once, then held" concern (see sharpa_config.py's role
split), written directly via set_preshape(), the same "write once, never
reset-to-stand every tick" convention FixedBaseGraspEnv already uses for
its own thumb1_ctrl_override.
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
from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc

_ARM_JOINTS = tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS  # 14
_WAIST_JOINTS = wbc.WAIST_JOINTS  # 3
N_WAIST = len(_WAIST_JOINTS)
N_ARMS = len(_ARM_JOINTS)
N_GROUPS_PER_HAND = len(sc.GROUPS)  # 4
N_HAND_GROUPS = 2 * N_GROUPS_PER_HAND  # 8
ACTION_DIM = N_WAIST + N_ARMS + N_HAND_GROUPS  # 25
_WAIST_SLICE = slice(0, N_WAIST)
_ARM_SLICE = slice(N_WAIST, N_WAIST + N_ARMS)
_LEFT_GROUP_SLICE = slice(N_WAIST + N_ARMS, N_WAIST + N_ARMS + N_GROUPS_PER_HAND)
_RIGHT_GROUP_SLICE = slice(N_WAIST + N_ARMS + N_GROUPS_PER_HAND, ACTION_DIM)


class SharpaGraspEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config: gc.GraspEnvConfig | None = None, render_mode: str | None = None):
        self.config = config or gc.GraspEnvConfig()
        self.render_mode = render_mode

        self.model = model_builder.build_grasp_model_sharpa(self.config)
        self.data = mujoco.MjData(self.model)
        self._resolve_indices()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        # [36th session correction] The single-hand prototype's 37-dim
        # observation (arm qpos/qvel + EE pos + object pos only) omitted
        # hand joint state, object orientation/angular velocity, and
        # per-group contact/force -- an independent audit found this
        # would block sharing this SAME env across a future Expert/BC/PPO
        # pipeline for this recovery skill (project rule: same env,
        # same observation contract for all of them). Dimension is
        # computed from the REAL compiled model, never hard-coded, and
        # locked by a test (test_sharpa_bimanual_grasp.py).
        self._n_curl_joints_per_side = sum(len(self._group_qpos_adr["left"][g]) for g in range(N_GROUPS_PER_HAND))
        obs_dim = (
            N_ARMS * 2  # arm qpos + qvel, both sides
            + 2 * self._n_curl_joints_per_side * 2  # curl qpos + qvel, both sides
            + 3 + 9  # left palm pos + xmat(9)
            + 3 + 9  # right palm pos + xmat(9)
            + 3 + 4  # object pos + quat
            + 3 + 3  # object linear + angular velocity
            + N_HAND_GROUPS  # per-(side,group) net contact force
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._arm_target = np.zeros(N_ARMS, dtype=np.float64)
        self._waist_target = np.zeros(N_WAIST, dtype=np.float64)
        # Order: [left_thumb, left_index, left_middle, left_wrap,
        # right_thumb, right_index, right_middle, right_wrap].
        self._group_synergy = np.zeros(N_HAND_GROUPS, dtype=np.float64)
        self._step_count = 0
        self._contact_streak = 0
        self._renderer: mujoco.Renderer | None = None

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

        # Per-group curl-joint (open, close) targets, built from the REAL
        # compiled joint ranges (never hard-coded) -- see sharpa_config's
        # CLOSE_FRACTION for why close is not the hard upper limit.
        self._group_act_ids: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        self._group_qpos_adr: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        self._group_dof_adr: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        self._group_open: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        self._group_close: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        self._preshape_act_ids: dict[str, np.ndarray] = {}
        self._preshape_targets: dict[str, np.ndarray] = {}
        for side in sc.SIDES:
            for group in sc.GROUPS:
                aids, qadrs, dadrs, opens, closes = [], [], [], [], []
                for finger in sc.GROUP_FINGERS[group]:
                    for suffix in sc.curl_suffixes(finger):
                        jname = sc.sharpa_joint(side, finger, suffix)
                        aname = sc.sharpa_actuator(side, finger, suffix)
                        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                        lo, hi = model.jnt_range[jid]
                        aids.append(act_id(aname))
                        qadrs.append(qpos_adr(jname))
                        dadrs.append(dof_adr(jname))
                        opens.append(lo)
                        closes.append(lo + sc.CLOSE_FRACTION * (hi - lo))
                self._group_act_ids[side].append(np.array(aids))
                self._group_qpos_adr[side].append(np.array(qadrs))
                self._group_dof_adr[side].append(np.array(dadrs))
                self._group_open[side].append(np.array(opens))
                self._group_close[side].append(np.array(closes))

            preshape_aids, preshape_vals = [], []
            for finger in sc.FINGERS:
                for suffix in sc.preshape_suffixes(finger):
                    jname = sc.sharpa_joint(side, finger, suffix)
                    aname = sc.sharpa_actuator(side, finger, suffix)
                    preshape_aids.append(act_id(aname))
                    preshape_vals.append(sc.PRESHAPE_TARGETS[finger][suffix])
            self._preshape_act_ids[side] = np.array(preshape_aids)
            self._preshape_targets[side] = np.array(preshape_vals)

        self._left_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tc.LEFT_EE_SITE)
        self._right_ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tc.RIGHT_EE_SITE)
        self._left_palm_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_PALM_SITE)
        self._right_palm_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.RIGHT_PALM_SITE)
        assert self._left_ee_site >= 0 and self._right_ee_site >= 0
        assert self._left_palm_site >= 0 and self._right_palm_site >= 0

        self._fingertip_site = {
            (side, finger): mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_sharpa_tip")
            for side in sc.SIDES for finger in sc.FINGERS
        }

        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, tc.STAND_KEYFRAME)
        assert key_id >= 0
        n_robot_qpos = model.nq - 7
        self._stand_qpos = model.key_qpos[key_id][:n_robot_qpos].copy()
        # The stand keyframe's key_ctrl only covers the ORIGINAL (Dex3)
        # model's actuators -- Sharpa's actuators didn't exist when the
        # keyframe was authored, so this model's ctrl0 (all zero) is used
        # for the hand actuators instead; only arm/waist/leg ctrl come
        # from the keyframe.
        self._stand_ctrl_arms_waist = model.key_ctrl[key_id][: len(model.key_ctrl[key_id])].copy()

        obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tc.OBJECT_JOINT)
        assert obj_jid >= 0
        self._object_qpos_adr = model.jnt_qposadr[obj_jid]
        self._object_dof_adr = model.jnt_dofadr[obj_jid]
        self._object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tc.OBJECT_BODY)

        self._left_hand_body_ids = {
            b for b in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("left_left_")
        }
        self._right_hand_body_ids = {
            b for b in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("right_right_")
        }

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        qpos = np.zeros(self.model.nq)
        qpos[: len(self._stand_qpos)] = self._stand_qpos
        obj_x, obj_y, _ = self.config.object_pos
        obj_z = (
            self.config.table_pos[2]
            + self.config.table_half_size[2]
            + self.config.effective_object_half_extents[2]
            + tc.OBJECT_TABLE_GAP
        )
        qpos[self._object_qpos_adr : self._object_qpos_adr + 3] = [obj_x, obj_y, obj_z]
        qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self._arm_act_ids] = self._stand_qpos[self._arm_qpos_adr]
        self.data.ctrl[self._waist_act_ids] = self._stand_qpos[self._waist_qpos_adr]
        for side in sc.SIDES:
            for g, group in enumerate(sc.GROUPS):
                self.data.ctrl[self._group_act_ids[side][g]] = self._group_open[side][g]

        self._arm_target = self.data.ctrl[self._arm_act_ids].copy()
        self._waist_target = self.data.ctrl[self._waist_act_ids].copy()
        self._group_synergy = np.zeros(N_HAND_GROUPS, dtype=np.float64)
        self._step_count = 0
        self._contact_streak = 0
        self.last_safety_events: list[tuple[str, int, float]] = []

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def set_preshape(self, side: str, fraction: float = 1.0) -> None:
        """Writes that side's preshape (spread/thumb-opposition) joint
        ctrl directly toward sharpa_config.PRESHAPE_TARGETS, scaled by
        ``fraction`` (0=neutral/open, 1=full preshape target) -- called
        by the expert once per FIVE_FINGER_PRESHAPE state entry, then
        never touched again (same "write once, persists" convention as
        FixedBaseGraspEnv's thumb1_ctrl_override)."""
        self.data.ctrl[self._preshape_act_ids[side]] = fraction * self._preshape_targets[side]

    # ------------------------------------------------------------------
    def _group_contact_force(self, side: str, group: str) -> tuple[float, float]:
        """Returns (peak_single_contact_N, net_group_force_N) between
        this finger GROUP's real bodies and the object -- both recorded
        (Stage 4 requirement), unlike FixedBaseGraspEnv which returns
        only one depending on a config flag."""
        model, data = self.model, self.data
        obj_body = self._object_body_id
        prefixes = tuple(sc.sharpa_body(side, f, "") for f in sc.GROUP_FINGERS[group])
        force_sum = np.zeros(3, dtype=np.float64)
        peak = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if obj_body not in (b1, b2):
                continue
            other = b2 if b1 == obj_body else b1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if not any(name.startswith(p) for p in prefixes):
                continue
            force6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force6)
            peak = max(peak, float(np.linalg.norm(force6[:3])))
            contact_R = np.asarray(c.frame, dtype=np.float64).reshape(3, 3)
            force_on_geom2 = contact_R.T @ force6[:3]
            force_sum += -force_on_geom2 if b1 == obj_body else force_on_geom2
        return peak, float(np.linalg.norm(force_sum))

    def _torso_arm_collision_force(self) -> float:
        """[Session 40] G1's own torso/upper-body vs either arm/wrist/hand
        -- a category the existing hand-hand and proximal-object checks
        never covered. Added after the object-facing wrist orientation
        change was found (causally, A/B) to drive the right wrist into
        torso_link at up to 109N (docs/history/PHASE4_GRASP_SESSION_40.md)."""
        model, data = self.model, self.data
        peak = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
            arm_prefixes = ("left_left_", "right_right_", "left_wrist", "right_wrist",
                             "left_elbow", "right_elbow", "left_shoulder", "right_shoulder")
            is_torso_arm = (
                ("torso" in b1 and any(b2.startswith(p) or p in b2 for p in arm_prefixes)) or
                ("torso" in b2 and any(b1.startswith(p) or p in b1 for p in arm_prefixes))
            )
            if not is_torso_arm:
                continue
            force6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force6)
            peak = max(peak, float(np.linalg.norm(force6[:3])))
        return peak

    def _hand_hand_contact_force(self) -> float:
        model, data = self.model, self.data
        peak = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
            if not ((b1.startswith("left_left_") and b2.startswith("right_right_")) or
                    (b2.startswith("left_left_") and b1.startswith("right_right_"))):
                continue
            force6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force6)
            peak = max(peak, float(np.linalg.norm(force6[:3])))
        return peak

    def _proximal_object_penetration(self) -> float:
        """Max penetration between any NON-fingertip Sharpa link (i.e. any
        hand body that is not a *_DP fingertip body) and the object --
        Stage 4 explicitly requires checking proximal/palm collision, not
        just fingertip contact."""
        model, data = self.model, self.data
        obj_body = self._object_body_id
        max_pen = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0:
                continue
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if obj_body not in (b1, b2):
                continue
            other = b2 if b1 == obj_body else b1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            is_hand = name.startswith("left_left_") or name.startswith("right_right_")
            is_fingertip = name.endswith("_DP")
            if is_hand and not is_fingertip:
                max_pen = max(max_pen, -float(c.dist))
        return max_pen

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._waist_target = np.clip(
            self._waist_target + self.config.waist_action_scale * action[_WAIST_SLICE],
            self._waist_ctrl_low, self._waist_ctrl_high,
        )
        self._arm_target = np.clip(
            self._arm_target + self.config.arm_action_scale * action[_ARM_SLICE],
            self._arm_ctrl_low, self._arm_ctrl_high,
        )
        prev_group_synergy = self._group_synergy.copy()
        group_delta = np.concatenate([action[_LEFT_GROUP_SLICE], action[_RIGHT_GROUP_SLICE]])
        self._group_synergy = np.clip(
            self._group_synergy + self.config.hand_synergy_action_scale * group_delta, 0.0, 1.0
        )

        prev_group_ctrl = {
            side: [self.data.ctrl[self._group_act_ids[side][g]].copy() for g in range(N_GROUPS_PER_HAND)]
            for side in sc.SIDES
        }

        self.data.ctrl[self._waist_act_ids] = self._waist_target
        self.data.ctrl[self._arm_act_ids] = self._arm_target
        if self.config.arm_gravity_compensation:
            # [Session 39] cancel the compliant kp=120 position actuators'
            # steady-state gravity/load droop with a direct feedforward:
            # data.qfrc_bias (gravity+Coriolis bias force, already computed
            # by the PREVIOUS mj_step/mj_forward call) divided by arm_kp is
            # the actuator-space position offset needed to hold the CURRENT
            # load at the CURRENT target -- see grasp_config.py's
            # arm_gravity_compensation docstring. Uses last tick's qfrc_bias
            # (one-tick feedback delay, standard and stable for a load that
            # changes slowly relative to the control tick) rather than an
            # extra mj_forward call.
            grav_arm = self.data.qfrc_bias[self._arm_dof_adr] / self.config.arm_kp
            grav_waist = self.data.qfrc_bias[self._waist_dof_adr] / self.config.arm_kp
            self.data.ctrl[self._arm_act_ids] = np.clip(
                self._arm_target + grav_arm, self._arm_ctrl_low, self._arm_ctrl_high
            )
            self.data.ctrl[self._waist_act_ids] = np.clip(
                self._waist_target + grav_waist, self._waist_ctrl_low, self._waist_ctrl_high
            )
        for side_idx, side in enumerate(sc.SIDES):
            for g in range(N_GROUPS_PER_HAND):
                syn = self._group_synergy[side_idx * N_GROUPS_PER_HAND + g]
                open_t, close_t = self._group_open[side][g], self._group_close[side][g]
                self.data.ctrl[self._group_act_ids[side][g]] = open_t + syn * (close_t - open_t)
        # preshape ctrl is NOT touched here -- set_preshape() writes it
        # directly and it is never reset-to-stand every tick (module docstring).

        # Physics-substep force safety, generalized to Sharpa's 4 groups x
        # 2 hands (Stage 4 requirement: substep-granularity net GROUP
        # force, not a once-per-tick check -- same reasoning as
        # FixedBaseGraspEnv's Dex3 version, written fresh here since the
        # body-name prefixes and group set differ).
        limit = self.config.finger_force_safety_limit
        warn = limit * self.config.finger_force_warning_ratio
        self.last_safety_events = []
        for _substep_idx in range(self.config.frame_skip):
            mujoco.mj_step(self.model, self.data)
            for side_idx, side in enumerate(sc.SIDES):
                for g, group in enumerate(sc.GROUPS):
                    peak, net = self._group_contact_force(side, group)
                    force = net if self.config.use_net_group_force else peak
                    ids = self._group_act_ids[side][g]
                    qadr = self._group_qpos_adr[side][g]
                    unload_target = np.clip(
                        self.data.qpos[qadr], self.model.actuator_ctrlrange[ids, 0], self.model.actuator_ctrlrange[ids, 1]
                    )
                    safe_target = unload_target if self.config.unload_finger_on_force_limit else prev_group_ctrl[side][g]
                    if force > limit:
                        frac = self.config.force_unload_fraction if self.config.unload_finger_on_force_limit else 1.0
                        self.data.ctrl[ids] = (1.0 - frac) * self.data.ctrl[ids] + frac * safe_target
                        if self.config.persist_safety_synergy_rollback:
                            self._group_synergy[side_idx * N_GROUPS_PER_HAND + g] = prev_group_synergy[side_idx * N_GROUPS_PER_HAND + g]
                        self.last_safety_events.append((side, g, force))
                    elif force > warn:
                        frac = 0.5 * (self.config.force_unload_fraction if self.config.unload_finger_on_force_limit else 1.0)
                        self.data.ctrl[ids] = (1.0 - frac) * self.data.ctrl[ids] + frac * safe_target
            hh_force = self._hand_hand_contact_force()
            if hh_force > limit:
                for side in sc.SIDES:
                    thumb_idx = sc.GROUPS.index("thumb")
                    self.data.ctrl[self._group_act_ids[side][thumb_idx]] = prev_group_ctrl[side][thumb_idx]
                self.last_safety_events.append(("hand_hand", 0, hh_force))

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
            other = bodies - {self._object_body_id}
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
        parts = [arm_qpos, arm_qvel]
        for side in sc.SIDES:
            for g in range(N_GROUPS_PER_HAND):
                parts.append(self.data.qpos[self._group_qpos_adr[side][g]])
                parts.append(self.data.qvel[self._group_dof_adr[side][g]])
        left_pos, left_R = self.palm_pose("left")
        right_pos, right_R = self.palm_pose("right")
        parts += [left_pos, left_R.flatten(), right_pos, right_R.flatten()]
        obj_pos = self.data.qpos[self._object_qpos_adr : self._object_qpos_adr + 3]
        obj_quat = self.data.qpos[self._object_qpos_adr + 3 : self._object_qpos_adr + 7]
        obj_linvel = self.data.qvel[self._object_dof_adr : self._object_dof_adr + 3]
        obj_angvel = self.data.qvel[self._object_dof_adr + 3 : self._object_dof_adr + 6]
        parts += [obj_pos, obj_quat, obj_linvel, obj_angvel]
        group_forces = [self._group_contact_force(side, group)[1] for side in sc.SIDES for group in sc.GROUPS]
        parts.append(np.array(group_forces))
        return np.concatenate(parts).astype(np.float32)

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
            "proximal_object_penetration": self._proximal_object_penetration(),
            "hand_hand_force": self._hand_hand_contact_force(),
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

    def palm_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        sid = self._left_palm_site if side == "left" else self._right_palm_site
        return self.data.site_xpos[sid].copy(), self.data.site_xmat[sid].reshape(3, 3).copy()

    def fingertip_pos(self, side: str, finger: str) -> np.ndarray:
        return self.data.site_xpos[self._fingertip_site[(side, finger)]].copy()

    def arm_jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._left_ee_site)
        left_jac = jacp[:, self._arm_dof_adr[:7]].copy()
        jacp[:] = 0.0
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._right_ee_site)
        right_jac = jacp[:, self._arm_dof_adr[7:]].copy()
        return left_jac, right_jac
