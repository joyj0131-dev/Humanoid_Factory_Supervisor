"""Session 41, Stage 3: free-space Sharpa Wave bimanual hand open/close
DIAGNOSTIC (not the official grasp controller -- no object interaction,
no Gate A/precontact semantics, never affects any Gate metric). Exists
so a user can directly verify the Sharpa finger actuators and MuJoCo
physics work correctly, independent of whether the official
SharpaBimanualGraspExpert's approach/grasp logic currently succeeds.

Moves both arms to the presentation pose via a DIRECT joint target --
NOT Cartesian IK. [Session 41, measured] a Cartesian IK move to a fixed
world-frame target here drove the wrist into a real, reproducible
wrist_roll_link<->wrist_yaw_link self-collision (the exact known
low-inertia/zero-damping wrist instability documented in
sharpa_bimanual_grasp_expert.py's module docstring). Since this demo's
whole point is to move the arm somewhere SAFE and VISIBLE, not to reach
a precise Cartesian point, it reuses the OFFICIAL controller's own
validated ARM_LATERAL_CLEARANCE joint posture
(BimanualGraspConfig.clearance_shoulder_pitch/roll/elbow -- a bounded
3-candidate FK+physics sweep found 0 real self-collisions, elbow well
below shoulder, near-perfect actuator tracking; see
docs/history/PHASE4_GRASP_SESSION_41.md) directly, sidestepping the
whole redundant-IK-branch-selection problem entirely.

Then drives the REAL Sharpa finger actuators (group synergy, the same
action channel SharpaBimanualGraspExpert uses for CONTACT_ACQUIRE/
THUMB_OPPOSE/ENVELOPING_CLOSE) through open -> sequential per-finger
curl -> thumb opposition -> full close -> hold -> reopen, entirely
through real env.step() physics (no qpos teleport).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import mujoco
import numpy as np

from humanoid_learning.envs import sharpa_config as sc

SIDES = ("left", "right")
Y_SIGN = {"left": 1.0, "right": -1.0}


class HandDemoState(Enum):
    MOVE_TO_PRESENT = auto()
    OPEN_HOLD = auto()
    CURL_INDEX = auto()
    CURL_MIDDLE = auto()
    CURL_WRAP = auto()
    CURL_THUMB = auto()
    CLOSE_HOLD = auto()
    REOPEN = auto()
    DONE = auto()


@dataclass
class HandDemoConfig:
    # [Session 41] Direct joint target reusing SharpaBimanualGraspExpert's
    # own validated ARM_LATERAL_CLEARANCE posture (same numbers, see that
    # module's BimanualGraspConfig docstring) -- NOT an independently
    # chosen Cartesian point (see this module's docstring for why a
    # Cartesian IK move here caused a real wrist self-collision).
    present_shoulder_pitch: float = -0.1
    present_shoulder_roll: float = 1.1
    present_elbow: float = 0.8
    present_joint_tol_rad: float = 0.03
    present_stable_streak_required: int = 15
    open_hold_ticks: int = 60          # ~1s at typical control rate
    curl_stage_ticks: int = 60
    close_hold_ticks: int = 120        # ~2s
    reopen_ticks: int = 60
    close_rate_per_step: float = 0.03
    arm_action_scale_ramp: float = 1.0


@dataclass
class HandDemoStatus:
    state: HandDemoState
    state_step: int
    group_synergy: dict  # {side: {group: float in [0,1]}}
    left_palm_pos_error_m: float
    right_palm_pos_error_m: float
    forbidden_collision: bool


class SharpaHandDemo:
    """Drives a SharpaGraspEnv through the free-space open/close demo.
    Call .step() once per control tick (mirrors SharpaBimanualGraspExpert's
    own calling convention); returns the 25-dim action to pass to
    env.step()."""

    def __init__(self, env, config: HandDemoConfig | None = None):
        self.env = env
        self.config = config or HandDemoConfig()
        self.state = HandDemoState.MOVE_TO_PRESENT
        self._state_step = 0
        self._just_advanced = False
        self._group_synergy = {s: {g: 0.0 for g in sc.GROUPS} for s in SIDES}
        self._curl_order = ["index", "middle", "wrap", "thumb"]
        self._curl_stage_idx = 0

        self._arm_ik_target = env._arm_target.copy()
        self._waist_ik_target = env._waist_target.copy()
        cfg = self.config
        self._present_target = np.concatenate([
            self._waist_ik_target,  # waist stays neutral/current
            self._present_arm_vector("left"),
            self._present_arm_vector("right"),
        ])
        self._present_stable_streak = 0
        self._last_pos_error = {"left": float("inf"), "right": float("inf")}

    def _present_arm_vector(self, side: str) -> np.ndarray:
        cfg = self.config
        return np.array([
            cfg.present_shoulder_pitch, Y_SIGN[side] * cfg.present_shoulder_roll, 0.0,
            cfg.present_elbow, 0.0, 0.0, 0.0,
        ])

    # ------------------------------------------------------------------
    def _arm_action_toward_target(self) -> np.ndarray:
        env = self.env
        scale = env.config.arm_action_scale * self.config.arm_action_scale_ramp
        delta = np.clip(self._arm_ik_target - env._arm_target, -scale, scale)
        return delta / max(env.config.arm_action_scale, 1e-9)

    def _waist_action_toward_target(self) -> np.ndarray:
        env = self.env
        scale = env.config.waist_action_scale
        delta = np.clip(self._waist_ik_target - env._waist_target, -scale, scale)
        return delta / max(env.config.waist_action_scale, 1e-9)

    def _zero_action(self) -> np.ndarray:
        from humanoid_learning.envs.sharpa_grasp_env import ACTION_DIM
        return np.zeros(ACTION_DIM)

    def _group_action(self, deltas: dict) -> np.ndarray:
        vec = np.zeros(2 * len(sc.GROUPS))
        for side_idx, side in enumerate(SIDES):
            for g_idx, group in enumerate(sc.GROUPS):
                vec[side_idx * len(sc.GROUPS) + g_idx] = deltas.get(side, {}).get(group, 0.0) / max(
                    self.env.config.hand_synergy_action_scale, 1e-9
                )
        return vec

    def _advance(self, next_state: HandDemoState) -> None:
        self.state = next_state
        self._state_step = 0
        self._just_advanced = True

    def _forbidden_collision(self) -> bool:
        """hand-hand, hand-table, or torso-arm -- must be zero throughout
        this diagnostic (Stage 3 requirement)."""
        env = self.env
        if env._hand_hand_contact_force() > 1.0:
            return True
        if env._torso_arm_collision_force() > 1.0:
            return True
        model, data = env.model, env.data
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0:
                continue
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
            is_hand = b1.startswith("left_left_") or b1.startswith("right_right_") or b2.startswith("left_left_") or b2.startswith("right_right_")
            is_table = "table" in b1 or "table" in b2
            if is_hand and is_table:
                return True
        return False

    # ------------------------------------------------------------------
    def step(self) -> np.ndarray:
        action = self._zero_action()
        cfg = self.config
        state = self.state
        self._just_advanced = False

        if state == HandDemoState.MOVE_TO_PRESENT:
            if self._state_step == 0:
                self._waist_ik_target = self._present_target[:3].copy()
                self._arm_ik_target = self._present_target[3:].copy()
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            actual_arm_q = np.concatenate([
                self.env.data.qpos[self.env._arm_qpos_adr[:7]], self.env.data.qpos[self.env._arm_qpos_adr[7:]]
            ])
            joint_err = float(np.max(np.abs(actual_arm_q - self._arm_ik_target)))
            qvel_ok = float(np.max(np.abs(self.env.data.qvel[np.concatenate([self.env._arm_dof_adr, self.env._waist_dof_adr])]))) < 0.05
            no_collision = not self._forbidden_collision()
            for side in SIDES:
                self._last_pos_error[side] = 0.0  # direct joint target -- no Cartesian target to report
            stable_now = joint_err <= cfg.present_joint_tol_rad and qvel_ok and no_collision
            self._present_stable_streak = self._present_stable_streak + 1 if stable_now else 0
            if self._present_stable_streak >= cfg.present_stable_streak_required:
                self.env.set_preshape("left", 1.0)
                self.env.set_preshape("right", 1.0)
                self._advance(HandDemoState.OPEN_HOLD)

        elif state in (HandDemoState.OPEN_HOLD, HandDemoState.CLOSE_HOLD):
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            required = cfg.open_hold_ticks if state == HandDemoState.OPEN_HOLD else cfg.close_hold_ticks
            if self._state_step >= required:
                if state == HandDemoState.OPEN_HOLD:
                    self._curl_stage_idx = 0
                    self._advance(HandDemoState.CURL_INDEX)
                else:
                    self._advance(HandDemoState.REOPEN)

        elif state in (HandDemoState.CURL_INDEX, HandDemoState.CURL_MIDDLE,
                       HandDemoState.CURL_WRAP, HandDemoState.CURL_THUMB):
            group = self._curl_order[self._curl_stage_idx]
            deltas = {"left": {group: cfg.close_rate_per_step}, "right": {group: cfg.close_rate_per_step}}
            for side in SIDES:
                self._group_synergy[side][group] = min(1.0, self._group_synergy[side][group] + cfg.close_rate_per_step)
            action[17:25] = self._group_action(deltas)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            if self._state_step >= cfg.curl_stage_ticks:
                self._curl_stage_idx += 1
                if self._curl_stage_idx < len(self._curl_order):
                    next_group = self._curl_order[self._curl_stage_idx]
                    next_state = {
                        "middle": HandDemoState.CURL_MIDDLE, "wrap": HandDemoState.CURL_WRAP,
                        "thumb": HandDemoState.CURL_THUMB,
                    }[next_group]
                    self._advance(next_state)
                else:
                    self._advance(HandDemoState.CLOSE_HOLD)

        elif state == HandDemoState.REOPEN:
            deltas = {"left": {}, "right": {}}
            for side in SIDES:
                for group in sc.GROUPS:
                    if self._group_synergy[side][group] > 0.0:
                        deltas[side][group] = -cfg.close_rate_per_step
                        self._group_synergy[side][group] = max(0.0, self._group_synergy[side][group] - cfg.close_rate_per_step)
            action[17:25] = self._group_action(deltas)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            all_open = all(self._group_synergy[s][g] <= 1e-6 for s in SIDES for g in sc.GROUPS)
            if all_open or self._state_step >= cfg.reopen_ticks * 3:
                self._advance(HandDemoState.DONE)

        elif state == HandDemoState.DONE:
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()

        if not self._just_advanced:
            self._state_step += 1
        return action

    def status(self) -> HandDemoStatus:
        return HandDemoStatus(
            state=self.state, state_step=self._state_step,
            group_synergy={s: dict(self._group_synergy[s]) for s in SIDES},
            left_palm_pos_error_m=self._last_pos_error["left"],
            right_palm_pos_error_m=self._last_pos_error["right"],
            forbidden_collision=self._forbidden_collision(),
        )

    def loop_if_done(self) -> None:
        """Restart the open/curl/close/reopen cycle (NOT the MOVE_TO_
        PRESENT arm motion) once DONE, for a --restart (looping) demo."""
        if self.state == HandDemoState.DONE:
            self._advance(HandDemoState.OPEN_HOLD)
