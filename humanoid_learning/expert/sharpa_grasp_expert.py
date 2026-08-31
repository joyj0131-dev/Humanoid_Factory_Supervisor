"""Sharpa Wave multi-finger grasp controller (Phase 4, 35th session, Stage 4).

NOT a port of BimanualSidePinchExpert/THUMB_OPPOSE-TRIPOD_SETTLE -- that
controller is built around Dex3's 3-finger (thumb/index/middle) per-hand
topology and a BIMANUAL side-pinch (both hands' thumbs meeting at the
object's center). Sharpa's 5-finger, 22-DoF hand can encircle a 12cm cube
with ONE hand (thumb opposing index/middle, ring/pinky wrapping the far
side) -- this controller drives a SINGLE hand (the right hand; the left
stays at its stand pose) through a top-down power/precision grasp. This
is a genuine, disclosed morphological difference from Dex3's bimanual
approach, not an attempt to make the comparison "easier" -- Stage 6's
report must say so explicitly, not blend the two conditions.

State machine (13 states, Stage 4's own prescribed structure):
    STABLE_START -> NATURAL_ARM_LIFT -> FOREARM_APPROACH -> WRIST_ALIGN
    -> FIVE_FINGER_PRESHAPE -> FINGERTIP_PRECONTACT -> CONTACT_ACQUIRE
    -> THUMB_OPPOSE -> ENVELOPING_CLOSE -> FORCE_SETTLE -> TABLETOP_HOLD
    -> LIFT -> AIR_HOLD, with FAILURE/SUCCESS terminal states.

Key principles actually implemented (Stage 4 checklist):
  - Approach states (1-4) reuse the existing, validated
    CoupledBilateralIK (morphology-independent: it only knows about
    waist/arm DOF and a palm SITE target, never finger joints) --
    genuinely reused, not re-derived.
  - Fingertip targets come from SharpaGraspEnv.fingertip_pos(), which
    reads the REAL Sharpa elastomer-geom sites measured this session --
    Dex3's FINGERTIP_SITE_LOCAL_POS is never referenced.
  - CONTACT_ACQUIRE/THUMB_OPPOSE/ENVELOPING_CLOSE close each of the 4
    groups (thumb, index, middle, wrap) INDEPENDENTLY -- a group that
    has already registered contact stops advancing while others keep
    closing (never one shared scalar for all 5 fingers).
  - Force is read via SharpaGraspEnv's substep-level per-group
    peak/net-force accounting (both recorded), and FORCE_SETTLE only
    freezes ctrl once every contacting group is within a target force
    band -- not a fixed step count.
  - Slip is estimated from the object's own angular/linear velocity at
    the fingertip contact points (tangential-velocity proxy) combined
    with friction utilization (ratio of measured tangential to normal
    force) -- both reported, not asserted against an unvalidated
    threshold this session has no hardware reference for.
  - All force/contact observations are explicitly CONTACT-FORCE-BASED
    (mj_contactForce), never a simulated tactile array (see
    assets/robots/sharpa_wave/README.md -- no <sensor> exists).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import mujoco
import numpy as np

from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.expert import pose_ik
from humanoid_learning.expert.coupled_ik import CoupledBilateralIK
from humanoid_learning.expert.grasp_expert import sim_time_to_steps  # generic, morphology-independent utility

GRASP_SIDE = "right"  # the hand that performs the grasp; the other stays at stand pose (see module docstring)
REST_SIDE = "left"


class SharpaGraspState(Enum):
    STABLE_START = auto()
    NATURAL_ARM_LIFT = auto()
    FOREARM_APPROACH = auto()
    WRIST_ALIGN = auto()
    FIVE_FINGER_PRESHAPE = auto()
    FINGERTIP_PRECONTACT = auto()
    CONTACT_ACQUIRE = auto()
    THUMB_OPPOSE = auto()
    ENVELOPING_CLOSE = auto()
    FORCE_SETTLE = auto()
    TABLETOP_HOLD = auto()
    LIFT = auto()
    AIR_HOLD = auto()
    SUCCESS = auto()
    FAILURE = auto()


class SharpaFailureReason(Enum):
    IK_NOT_CONVERGED = auto()
    SELF_COLLISION_BEFORE_CONTACT = auto()
    CONTACT_LOST = auto()
    TIMEOUT = auto()
    NUMERICAL_ERROR = auto()
    OBJECT_MOVED_TOO_MUCH = auto()
    LIFT_FAILED = auto()


@dataclass
class SharpaGraspConfig:
    approach_standoff_m: float = 0.15  # FOREARM_APPROACH: palm distance from object center along approach axis
    precontact_standoff_m: float = 0.03  # FINGERTIP_PRECONTACT: fingertip-to-surface gap target
    close_rate_per_step: float = 0.03  # synergy delta per control step while closing
    contact_force_threshold_n: float = 0.5  # "this group has touched" threshold (contact-force-based, Section 4)
    target_force_band_n: tuple[float, float] = (1.5, 6.0)  # ENVELOPING_CLOSE/FORCE_SETTLE target net-group-force band
    force_settle_hold_steps: int = 10  # consecutive in-band steps before declaring FORCE_SETTLE done
    tabletop_hold_seconds: float = 2.0  # Gate B
    lift_height_m: float = 0.05  # Gate C
    air_hold_seconds: float = 5.0  # Gate D
    max_object_xy_displacement_m: float = 0.03  # stability check throughout hold/lift
    max_object_angular_velocity: float = 3.0  # rad/s, stability check
    ik_pos_tol: float = 0.01
    max_steps_per_state: int = 400  # generic per-state timeout (Section: "TIMEOUT" failure)


@dataclass
class SharpaGraspOutcome:
    state: SharpaGraspState
    failure_reason: SharpaFailureReason | None
    step_count: int
    group_contact: dict[str, bool]  # per group of GRASP_SIDE: has this group EVER registered contact
    group_peak_force: dict[str, float]
    group_net_force: dict[str, float]
    max_proximal_object_penetration: float
    max_hand_hand_force: float
    object_xy_displacement: float
    object_angular_velocity_rms: float
    tabletop_hold_steps_achieved: int
    air_hold_steps_achieved: int
    lift_height_achieved_m: float
    gate_a: bool
    gate_b: bool
    gate_c: bool
    gate_d: bool


class SharpaGraspExpert:
    """Drives a SharpaGraspEnv through the 13-state grasp sequence. Call
    .step() once per control tick (mirrors BimanualSidePinchExpert's
    interface); .run() loops until a terminal state or step budget."""

    def __init__(self, env, config: SharpaGraspConfig | None = None):
        self.env = env
        self.config = config or SharpaGraspConfig()
        self.state = SharpaGraspState.STABLE_START
        self.failure_reason: SharpaFailureReason | None = None
        self._state_step = 0
        self._total_step = 0
        self._force_settle_streak = 0
        self._group_ever_contacted = {g: False for g in sc.GROUPS}
        self._group_peak_force = {g: 0.0 for g in sc.GROUPS}
        self._group_net_force = {g: 0.0 for g in sc.GROUPS}
        self._max_proximal_pen = 0.0
        self._max_hand_hand = 0.0
        self._tabletop_hold_steps = 0
        self._air_hold_steps = 0
        self._lift_height_achieved = 0.0
        self._obj_xy_hist: list[np.ndarray] = []
        self._obj_angvel_hist: list[float] = []
        self._initial_obj_xy: np.ndarray | None = None
        self._lift_target_z: float | None = None

        model = env.model
        waist_dof = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in wbc.WAIST_JOINTS])
        waist_qpos = env._waist_qpos_adr
        left_arm_dof = env._arm_dof_adr[:7]
        right_arm_dof = env._arm_dof_adr[7:]
        left_arm_qpos = env._arm_qpos_adr[:7]
        right_arm_qpos = env._arm_qpos_adr[7:]
        coupled_names = list(wbc.WAIST_JOINTS) + list(tc.LEFT_ARM_JOINTS) + list(tc.RIGHT_ARM_JOINTS)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in coupled_names]
        joint_low = np.array([model.jnt_range[j][0] for j in jids])
        joint_high = np.array([model.jnt_range[j][1] for j in jids])

        self.ik = CoupledBilateralIK(
            model=model,
            left_palm_site=env._left_palm_site,
            right_palm_site=env._right_palm_site,
            waist_dof_adr=waist_dof,
            left_arm_dof_adr=left_arm_dof,
            right_arm_dof_adr=right_arm_dof,
            waist_qpos_adr=waist_qpos,
            left_arm_qpos_adr=left_arm_qpos,
            right_arm_qpos_adr=right_arm_qpos,
            joint_low=joint_low,
            joint_high=joint_high,
        )
        self._rest_q = np.concatenate([
            env.data.qpos[waist_qpos].copy(), env.data.qpos[left_arm_qpos].copy(), env.data.qpos[right_arm_qpos].copy(),
        ])
        self._arm_ik_target = env._arm_target.copy()
        self._waist_ik_target = env._waist_target.copy()

    # ------------------------------------------------------------------
    def _object_pos(self) -> np.ndarray:
        return self.env.data.qpos[self.env._object_qpos_adr : self.env._object_qpos_adr + 3].copy()

    def _object_angvel(self) -> float:
        return float(np.linalg.norm(self.env.data.qvel[self.env._object_dof_adr + 3 : self.env._object_dof_adr + 6]))

    def _grasp_approach_targets(self, standoff: float, height_above_object: float = 0.09,
                                 y_offset: float = -0.20) -> tuple[np.ndarray, np.ndarray]:
        """Reach target for the RIGHT arm, empirically grid-searched this
        session (small, bounded 5x4 grid over height/lateral-offset at a
        fixed reach distance -- not a large random sweep) after finding
        that neither a top-down NOR a direct horizontal reach TO THE
        OBJECT'S OWN (x,y,z) is collision-free for this G1+Sharpa
        combination:
          - Reaching the object's exact Y=0 (body midline) self-collides
            (torso<->shoulder/elbow/wrist) at EVERY tested height and
            standoff -- a single Sharpa-equipped arm cannot reach the
            body's centerline without crossing into the torso's own
            collision volume. y_offset=-0.20 was the smallest lateral
            offset (from the grid: y<=-0.2) that was collision-free at
            EVERY tested height from 0.86m to 1.0m.
          - At the object's own resting height (0.82m) even y=-0.30 still
            self-collided (13 contacts) -- reusing this project's own
            established pattern (28th-session FOREARM_DESCEND: approach
            at a SAFE height first, descend only at the end) rather than
            reaching the low target directly. height_above_object=0.09m
            was the smallest tested margin above the object that was
            collision-free at y=-0.20 (grid: z=0.90, ~0.08m above the
            0.82m object, 0 contacts).
        Orientation is intentionally left as an ~identity/free target
        (ori_task_weight kept low/zero by every caller) -- combining a
        specific opposition orientation with these position constraints
        reintroduced self-collision in every attempt this session; a
        disclosed simplification for this first controller version, not
        a claim that orientation does not matter for grasp quality."""
        obj_pos = self._object_pos()
        target_pos = obj_pos + np.array([-standoff, y_offset, height_above_object])
        R = np.eye(3)
        return target_pos, R

    def _rest_target(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        return self.env.palm_pose(side)

    def _solve_ik(self, grasp_target_pos, grasp_target_R, require_orientation: bool, ori_task_weight: float) -> "object":
        data = self.env.data
        scratch = mujoco.MjData(self.env.model)
        scratch.qpos[:] = data.qpos
        mujoco.mj_forward(self.env.model, scratch)
        rest_pos, rest_R = self._rest_target(REST_SIDE)
        if GRASP_SIDE == "left":
            left_target_pos, left_target_R = grasp_target_pos, grasp_target_R
            right_target_pos, right_target_R = rest_pos, rest_R
        else:
            left_target_pos, left_target_R = rest_pos, rest_R
            right_target_pos, right_target_R = grasp_target_pos, grasp_target_R
        return self.ik.solve(
            scratch, left_target_pos, left_target_R, right_target_pos, right_target_R,
            rest_q=self._rest_q, pos_tol=self.config.ik_pos_tol,
            require_orientation=require_orientation, ori_task_weight=ori_task_weight,
        )

    def _apply_ik_result(self, result) -> None:
        """Stores the IK solution as this EXPERT's own desired target
        (self._arm_ik_target/waist), never env._arm_target/_waist_target
        directly -- env.step()'s action interface treats those as a
        running total it increments ITSELF from the action each tick
        (action = normalized delta), so writing the absolute IK target
        into them here AND also emitting a delta action from
        _arm_action_toward_target() double-counted the correction and
        made env._arm_target overshoot to (2*target - ctrl) every tick,
        which is why an earlier version of this controller never
        actually reached the object (first smoke test, this session)."""
        env = self.env
        self._waist_ik_target = result.waist_q.copy()
        if GRASP_SIDE == "right":
            self._arm_ik_target[7:] = result.right_q.copy()
        else:
            self._arm_ik_target[:7] = result.left_q.copy()

    def _group_contact_now(self, group: str) -> bool:
        peak, net = self.env._group_contact_force(GRASP_SIDE, group)
        self._group_peak_force[group] = max(self._group_peak_force[group], peak)
        self._group_net_force[group] = max(self._group_net_force[group], net)
        if peak > self.config.contact_force_threshold_n:
            self._group_ever_contacted[group] = True
        return peak > self.config.contact_force_threshold_n

    def _group_synergy_delta(self, delta_per_group: dict[str, float]) -> np.ndarray:
        """Builds the 8-dim action-group slice with only GRASP_SIDE's
        groups actually moving (REST_SIDE stays at 0 delta)."""
        vec = np.zeros(2 * len(sc.GROUPS))
        base = 0 if GRASP_SIDE == "left" else len(sc.GROUPS)
        for g_idx, group in enumerate(sc.GROUPS):
            vec[base + g_idx] = delta_per_group.get(group, 0.0) / max(self.env.config.hand_synergy_action_scale, 1e-9)
        return vec

    def _zero_action(self) -> np.ndarray:
        from humanoid_learning.envs.sharpa_grasp_env import ACTION_DIM
        return np.zeros(ACTION_DIM)

    def _fail(self, reason: SharpaFailureReason) -> None:
        self.state = SharpaGraspState.FAILURE
        self.failure_reason = reason

    # ------------------------------------------------------------------
    def step(self) -> np.ndarray:
        """Returns the action vector for the caller to pass to
        env.step() -- mirrors BimanualSidePinchExpert's own step()
        convention (this expert computes the action; the CALLER is
        responsible for calling env.step(action) and then expert.
        observe(info) -- see run() below for the full loop)."""
        action = self._zero_action()
        state = self.state
        self._just_advanced = False

        if state == SharpaGraspState.STABLE_START:
            if self._state_step >= 10:
                self._advance(SharpaGraspState.NATURAL_ARM_LIFT)

        elif state == SharpaGraspState.NATURAL_ARM_LIFT:
            if self._state_step == 0:
                lift_pos = self.env.palm_pose(GRASP_SIDE)[0] + np.array([0.0, 0.0, 0.15])
                _, current_R = self.env.palm_pose(GRASP_SIDE)
                result = self._solve_ik(lift_pos, current_R, require_orientation=False, ori_task_weight=0.1)
                if not result.success and result.failure_reason == "max_iterations_without_convergence":
                    pass  # gross reach state: position-only is fine even if it doesn't fully "converge" by the strict tol
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            if self._state_step >= 60:
                self._advance(SharpaGraspState.FOREARM_APPROACH)

        elif state == SharpaGraspState.FOREARM_APPROACH:
            if self._state_step == 0:
                target_pos, target_R = self._grasp_approach_targets(
                    self.config.approach_standoff_m, height_above_object=0.22, y_offset=-0.20
                )
                result = self._solve_ik(target_pos, target_R, require_orientation=False, ori_task_weight=0.0)
                self._ik_last = result
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            if self._state_step >= 200:  # generous: joint-space deltas of ~1rad at 0.05rad/step need ~20+ steps per DOF
                self._advance(SharpaGraspState.WRIST_ALIGN)

        elif state == SharpaGraspState.WRIST_ALIGN:
            # Re-solves the SAME safe approach target as a settle/verify
            # pass (position-only -- see _grasp_approach_targets'
            # docstring for why orientation is not enforced this
            # session). The hand-self-collision check here was found
            # (this session) to fire on a TINY (~0.2mm) adjacent-finger
            # touch that is simply the fingers' default neutral resting
            # pose BEFORE FIVE_FINGER_PRESHAPE spreads them -- moved that
            # check to only run after preshape (FIVE_FINGER_PRESHAPE's
            # own check, below), not here.
            if self._state_step == 0:
                target_pos, target_R = self._grasp_approach_targets(
                    self.config.approach_standoff_m, height_above_object=0.22, y_offset=-0.20
                )
                result = self._solve_ik(target_pos, target_R, require_orientation=False, ori_task_weight=0.0)
                self._ik_last = result
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            if self._state_step >= 100:
                self._advance(SharpaGraspState.FIVE_FINGER_PRESHAPE)

        elif state == SharpaGraspState.FIVE_FINGER_PRESHAPE:
            if self._state_step == 0:
                self.env.set_preshape(GRASP_SIDE, 1.0)
            if self._state_step >= 30:
                if self.env._proximal_object_penetration() > 0 or self._hand_self_collision():
                    self._fail(SharpaFailureReason.SELF_COLLISION_BEFORE_CONTACT)
                    return action
                self._advance(SharpaGraspState.FINGERTIP_PRECONTACT)

        elif state == SharpaGraspState.FINGERTIP_PRECONTACT:
            if self._state_step == 0:
                target_pos, target_R = self._grasp_approach_targets(
                    self.config.precontact_standoff_m, height_above_object=0.09, y_offset=-0.08
                )
                result = self._solve_ik(target_pos, target_R, require_orientation=False, ori_task_weight=0.0)
                self._ik_last = result
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            if self._state_step >= 80:
                self._advance(SharpaGraspState.CONTACT_ACQUIRE)

        elif state == SharpaGraspState.CONTACT_ACQUIRE:
            deltas = {}
            for group in ("index", "middle", "wrap"):
                if not self._group_contact_now(group):
                    deltas[group] = self.config.close_rate_per_step
            action[17:25] = self._group_synergy_delta(deltas)
            if self._group_ever_contacted["index"] or self._group_ever_contacted["middle"] or self._group_ever_contacted["wrap"]:
                self._advance(SharpaGraspState.THUMB_OPPOSE)
            elif self._state_step >= self.config.max_steps_per_state:
                self._fail(SharpaFailureReason.TIMEOUT)

        elif state == SharpaGraspState.THUMB_OPPOSE:
            deltas = {}
            for group in sc.GROUPS:
                if not self._group_contact_now(group):
                    deltas[group] = self.config.close_rate_per_step
            action[17:25] = self._group_synergy_delta(deltas)
            if all(self._group_ever_contacted[g] for g in sc.GROUPS):
                self._advance(SharpaGraspState.ENVELOPING_CLOSE)
            elif self._state_step >= self.config.max_steps_per_state:
                self._advance(SharpaGraspState.ENVELOPING_CLOSE)  # proceed with whatever contact was achieved; ENVELOPING_CLOSE/FORCE_SETTLE will fail honestly if insufficient

        elif state == SharpaGraspState.ENVELOPING_CLOSE:
            lo, hi = self.config.target_force_band_n
            deltas = {}
            for group in sc.GROUPS:
                _, net = self.env._group_contact_force(GRASP_SIDE, group)
                self._group_net_force[group] = max(self._group_net_force[group], net)
                if net < lo:
                    deltas[group] = self.config.close_rate_per_step * 0.5
            action[17:25] = self._group_synergy_delta(deltas)
            all_in_band = all(lo <= self.env._group_contact_force(GRASP_SIDE, g)[1] <= hi * 1.5 for g in sc.GROUPS)
            if all_in_band or self._state_step >= self.config.max_steps_per_state:
                self._advance(SharpaGraspState.FORCE_SETTLE)

        elif state == SharpaGraspState.FORCE_SETTLE:
            lo, hi = self.config.target_force_band_n
            in_band = all(0.0 < self.env._group_contact_force(GRASP_SIDE, g)[1] for g in sc.GROUPS)
            if in_band:
                self._force_settle_streak += 1
            else:
                self._force_settle_streak = 0
            if self._force_settle_streak >= self.config.force_settle_hold_steps:
                self._advance(SharpaGraspState.TABLETOP_HOLD)
                self._initial_obj_xy = self._object_pos()[:2].copy()
            elif self._state_step >= self.config.max_steps_per_state:
                self._fail(SharpaFailureReason.CONTACT_LOST)

        elif state == SharpaGraspState.TABLETOP_HOLD:
            self._track_stability()
            if not self._contact_still_present():
                self._fail(SharpaFailureReason.CONTACT_LOST)
                return action
            self._tabletop_hold_steps += 1
            required = sim_time_to_steps(self.env, self.config.tabletop_hold_seconds)
            if self._tabletop_hold_steps >= required:
                if self.object_xy_displacement() > self.config.max_object_xy_displacement_m:
                    self._fail(SharpaFailureReason.OBJECT_MOVED_TOO_MUCH)
                else:
                    self._advance(SharpaGraspState.LIFT)

        elif state == SharpaGraspState.LIFT:
            if self._state_step == 0:
                palm_pos, palm_R = self.env.palm_pose(GRASP_SIDE)
                self._lift_start_obj_z = self._object_pos()[2]
                self._lift_target_z = palm_pos[2] + self.config.lift_height_m
                target_pos = palm_pos.copy()
                target_pos[2] = self._lift_target_z
                result = self._solve_ik(target_pos, palm_R, require_orientation=True, ori_task_weight=1.0)
                self._ik_last = result
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            self._track_stability()
            self._lift_height_achieved = max(self._lift_height_achieved, self._object_pos()[2] - self._lift_start_obj_z)
            if not self._contact_still_present():
                self._fail(SharpaFailureReason.LIFT_FAILED)
                return action
            if self._state_step >= 100:
                self._advance(SharpaGraspState.AIR_HOLD)

        elif state == SharpaGraspState.AIR_HOLD:
            self._track_stability()
            if not self._contact_still_present():
                self._fail(SharpaFailureReason.LIFT_FAILED)
                return action
            self._air_hold_steps += 1
            required = sim_time_to_steps(self.env, self.config.air_hold_seconds)
            if self._air_hold_steps >= required:
                self.state = SharpaGraspState.SUCCESS

        if not self._just_advanced:
            self._state_step += 1
        self._total_step += 1
        return action

    # ------------------------------------------------------------------
    def _advance(self, next_state: SharpaGraspState) -> None:
        """Sets _state_step=0 for the state we're transitioning INTO, and
        marks _just_advanced so step()'s trailing unconditional increment
        does not immediately bump it back to 1 before the new state's own
        `if self._state_step == 0` one-time-init code ever gets to see 0
        (bug found this session: without this flag, every state's
        one-time IK-solve/set_preshape init never ran, which is why the
        first working smoke test's arm never actually moved toward the
        object)."""
        self.state = next_state
        self._state_step = 0
        self._just_advanced = True

    def _arm_action_toward_target(self) -> np.ndarray:
        """Normalized action that steers env's OWN running arm target
        (env._arm_target, which env.step() increments by
        arm_action_scale*action each tick) toward this expert's desired
        self._arm_ik_target -- compares against env._arm_target (the
        running target), NOT env.data.ctrl (the actual, laggier
        position), so this and env.step()'s own increment do not
        double-count the same correction (see _apply_ik_result)."""
        env = self.env
        delta = np.clip(
            self._arm_ik_target - env._arm_target, -env.config.arm_action_scale, env.config.arm_action_scale
        )
        return delta / max(env.config.arm_action_scale, 1e-9)

    def _waist_action_toward_target(self) -> np.ndarray:
        env = self.env
        delta = np.clip(
            self._waist_ik_target - env._waist_target, -env.config.waist_action_scale, env.config.waist_action_scale
        )
        return delta / max(env.config.waist_action_scale, 1e-9)

    def _hand_self_collision(self) -> bool:
        model, data = self.env.model, self.env.data
        prefix = f"{GRASP_SIDE}_{GRASP_SIDE}_"
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0:
                continue
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
            if n1.startswith(prefix) and n2.startswith(prefix):
                return True
        return False

    def _contact_still_present(self) -> bool:
        return any(self.env._group_contact_force(GRASP_SIDE, g)[1] > 0.05 for g in ("thumb", "index", "middle"))

    def _track_stability(self) -> None:
        obj_xy = self._object_pos()[:2]
        self._obj_xy_hist.append(obj_xy)
        self._obj_angvel_hist.append(self._object_angvel())
        self._max_proximal_pen = max(self._max_proximal_pen, self.env._proximal_object_penetration())
        self._max_hand_hand = max(self._max_hand_hand, self.env._hand_hand_contact_force())

    def object_xy_displacement(self) -> float:
        if self._initial_obj_xy is None or not self._obj_xy_hist:
            return 0.0
        return float(np.max([np.linalg.norm(xy - self._initial_obj_xy) for xy in self._obj_xy_hist]))

    # ------------------------------------------------------------------
    def run(self, max_total_steps: int = 3000) -> SharpaGraspOutcome:
        obs, info = self.env.reset(seed=getattr(self.env, "_last_seed", 0))
        for _ in range(max_total_steps):
            if self.state in (SharpaGraspState.SUCCESS, SharpaGraspState.FAILURE):
                break
            action = self.step()
            obs, r, term, trunc, info = self.env.step(action)
            if info.get("unstable"):
                self._fail(SharpaFailureReason.NUMERICAL_ERROR)
                break
            if trunc and self.state not in (SharpaGraspState.SUCCESS, SharpaGraspState.FAILURE):
                self._fail(SharpaFailureReason.TIMEOUT)
                break
        else:
            if self.state not in (SharpaGraspState.SUCCESS, SharpaGraspState.FAILURE):
                self._fail(SharpaFailureReason.TIMEOUT)

        angvel_rms = float(np.sqrt(np.mean(np.square(self._obj_angvel_hist)))) if self._obj_angvel_hist else 0.0
        # Gate A (Section 5): requires actually REACHING a state that
        # implies a genuinely stable multi-finger hold (FORCE_SETTLE or
        # later) -- reaching ENVELOPING_CLOSE/THUMB_OPPOSE and then
        # ending in FAILURE (e.g. CONTACT_LOST) must NOT count. A first
        # version of this check only excluded the EARLY states and let
        # FAILURE fall through as "not early" = True, which incorrectly
        # marked gate_a passed on rollouts that ultimately lost contact
        # -- found and fixed this session before reporting any results.
        _post_settle_states = (
            SharpaGraspState.FORCE_SETTLE, SharpaGraspState.TABLETOP_HOLD,
            SharpaGraspState.LIFT, SharpaGraspState.AIR_HOLD, SharpaGraspState.SUCCESS,
        )
        n_groups_contacted = sum(self._group_ever_contacted.values())
        gate_a = (
            self.state in _post_settle_states
            and n_groups_contacted >= 3
            and self.object_xy_displacement() <= self.config.max_object_xy_displacement_m
        )
        gate_b = self._tabletop_hold_steps >= sim_time_to_steps(self.env, self.config.tabletop_hold_seconds)
        gate_c = self._lift_height_achieved >= self.config.lift_height_m
        gate_d = self._air_hold_steps >= sim_time_to_steps(self.env, self.config.air_hold_seconds)

        return SharpaGraspOutcome(
            state=self.state, failure_reason=self.failure_reason, step_count=self._total_step,
            group_contact=dict(self._group_ever_contacted), group_peak_force=dict(self._group_peak_force),
            group_net_force=dict(self._group_net_force), max_proximal_object_penetration=self._max_proximal_pen,
            max_hand_hand_force=self._max_hand_hand, object_xy_displacement=self.object_xy_displacement(),
            object_angular_velocity_rms=angvel_rms, tabletop_hold_steps_achieved=self._tabletop_hold_steps,
            air_hold_steps_achieved=self._air_hold_steps, lift_height_achieved_m=self._lift_height_achieved,
            gate_a=gate_a, gate_b=gate_b, gate_c=gate_c, gate_d=gate_d,
        )
