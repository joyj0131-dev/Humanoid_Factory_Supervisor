"""sharpa_bimanual: the Phase 4 OFFICIAL Sharpa Wave grasp controller
(36th session -- corrective milestone after an independent audit found
the 35th session's sharpa_grasp_expert.py grasped with a SINGLE hand
only, which is NOT the user-approved bimanual target; that controller is
kept, relabeled SharpaSingleHandGraspExpert, as an explicit exploratory
diagnostic -- see its module docstring).

Both hands grasp the SAME object from opposite lateral (Y) sides --
mirrored, not one hand crossing the body midline to reproduce a
single-hand grasp. CoupledBilateralIK solves BOTH palm targets in one
call at every approach state (it always did; the single-hand version
just fed one side a "stay put" target instead of a real one).

Root-cause geometry finding this session (small, bounded grid searches,
never a large random sweep):
  - Reaching the object's own Y=+-0.06 face directly self-collides
    (torso<->shoulder AND, at 12cm object width, the two hands' own
    fingertips collide with EACH OTHER -- a 12cm cube is not wide enough
    for two full 5-finger Sharpa envelopes to coexist without their
    ulnar-side (ring/pinky) links overlapping). Y offset >=0.10 (palm
    target) is collision-free (grid: 0 real self-collision at
    y_off in {0.10, 0.12, 0.15, 0.20}); at y_off=0.10 the FINGERTIPS
    (not the palm) land at y~=0.06-0.068 on their own, i.e. right at the
    object's actual face, without the palm itself needing to cross in.
  - Enforcing an independently-CHOSEN opposition orientation (e.g.
    approach=+-Y, closing=+X) reintroduces severe self-collision
    (200+ contacts) at ANY nonzero orientation task weight, exactly as
    found for the single-hand controller. Fix used here: solve
    position-only first (collision-free, verified), CAPTURE the
    orientation the solver naturally landed on, then LOCK that exact
    orientation as the explicit WRIST_ALIGN/FINGERTIP_PRECONTACT target
    -- this converges to <1 degree orientation error with only 4 minor
    residual penetrations (down from 200+), because the solver is never
    asked to fight its own collision-free position solution for an
    independently-invented orientation.

Gate A definition (Section 6, modeled directly on the EXISTING, approved
Dex3 bilateral-tripod definition in grasp_expert.py -- see
BimanualSidePinchExpert's `_bilateral_tripod_streak`/`max_bilateral_tripod_streak`,
`>=30` consecutive ticks, and `object_displacement_limit=0.03`):
  - Per side: thumb touching AND (index OR middle touching) AND wrap
    touching AND opposition (thumb's contact-force direction opposes
    the index/middle combined force direction: dot product < 0) AND NOT
    this side self-colliding with the OTHER hand.
  - Bilateral: BOTH sides simultaneously satisfy the above, SAME tick.
  - Streak: consecutive ticks bilateral-satisfied, reset to 0 otherwise
    (exactly BimanualSidePinchExpert's own pattern).
  - Gate A = max_bilateral_streak >= 30 AND object xy displacement
    <= 0.03m (Dex3's own object_displacement_limit, confirmed identical
    value in grasp_expert.py:254) AND object peak angular velocity
    <= 2.0 rad/s (DISCLOSED: no coded Dex3 angular-velocity gate exists
    to inherit -- grep of grasp_expert.py/test_grasp.py found none, only
    the qualitative PROJECT_CONTEXT.md requirement to consider it; 2.0
    rad/s is this session's own, deliberately conservative, choice, not
    a relaxation of an existing number) AND no forbidden penetration
    beyond the same numerical-tolerance band already established for
    Sharpa (1mm) AND the terminal state is not FAILURE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import mujoco
import numpy as np

from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.expert.coupled_ik import CoupledBilateralIK
from humanoid_learning.expert.grasp_expert import sim_time_to_steps

SIDES = ("left", "right")
Y_SIGN = {"left": 1.0, "right": -1.0}
REQUIRED_GROUPS = ("thumb", "index", "middle", "wrap")  # topology: thumb + (index or middle) + wrap, checked below
# Not copied from Dex3 (object_displacement_limit IS, see module docstring):
OBJECT_XY_DISPLACEMENT_LIMIT = 0.03  # matches grasp_expert.GraspExpertConfig.object_displacement_limit exactly
OBJECT_PEAK_ANGULAR_VELOCITY_LIMIT = 2.0  # rad/s -- disclosed, no Dex3-coded precedent exists (see module docstring)
BILATERAL_STREAK_REQUIRED = 30  # matches Dex3's approved max_bilateral_tripod_streak >= 30


class BimanualGraspState(Enum):
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


class BimanualFailureReason(Enum):
    IK_NOT_CONVERGED = auto()
    WRIST_ORIENTATION_NOT_STABLE = auto()
    SELF_COLLISION_BEFORE_CONTACT = auto()
    CONTACT_LOST = auto()
    TIMEOUT = auto()
    NUMERICAL_ERROR = auto()
    OBJECT_MOVED_TOO_MUCH = auto()
    OBJECT_ANGULAR_VELOCITY_EXCEEDED = auto()
    LIFT_FAILED = auto()
    PRECONTACT_TRACKING_NOT_ACHIEVED = auto()


@dataclass
class BimanualGraspConfig:
    approach_standoff_m: float = 0.15
    # 0.22, not the 0.15 used for the bimanual self-collision grid search:
    # that grid search only checked HAND<->HAND/HAND<->TORSO self-collision
    # pairs (see module docstring), not hand<->table: 0.15 was found to let
    # fingertips graze the table during transit (verified this session --
    # `table<->*_DP` contacts caused the wrist to visibly stick, unable to
    # reach its IK target through the resulting friction lock). 0.22 is
    # SharpaSingleHandGraspExpert's own already-proven table-clearance
    # margin (sharpa_grasp_expert.py's FOREARM_APPROACH/WRIST_ALIGN), reused
    # directly rather than re-deriving it.
    approach_height_m: float = 0.22
    approach_y_offset_m: float = 0.15
    precontact_standoff_m: float = 0.08
    precontact_height_m: float = 0.09
    precontact_y_offset_m: float = 0.10
    close_rate_per_step: float = 0.03
    contact_force_threshold_n: float = 0.5
    target_force_band_n: tuple[float, float] = (1.0, 6.0)
    force_settle_hold_steps: int = 10
    tabletop_hold_seconds: float = 2.0
    lift_height_m: float = 0.05
    air_hold_seconds: float = 5.0
    object_xy_displacement_limit_m: float = OBJECT_XY_DISPLACEMENT_LIMIT
    object_peak_angular_velocity_limit: float = OBJECT_PEAK_ANGULAR_VELOCITY_LIMIT
    bilateral_streak_required: int = BILATERAL_STREAK_REQUIRED
    proximal_penetration_tolerance_m: float = 0.001  # same 1mm numerical band as sharpa_config's tolerances
    hand_hand_force_limit_n: float = 8.0  # matches Dex3's own established hand-hand safety concern order of magnitude
    ik_pos_tol: float = 0.01
    max_steps_per_state: int = 400
    wrist_orientation_stability_tol_deg: float = 5.0  # max angular drift over the last 30 ticks to call WRIST_ALIGN settled
    # [Session 39] FINGERTIP_PRECONTACT Precontact Tracking Gate (see
    # docs/history/PHASE4_GRASP_SESSION_39.md): the ctrl register
    # converges EXACTLY to the IK-solved joint target --
    # joint_target_minus_ctrl_norm == 0 -- yet the actual physical palm
    # settles several cm short, a steady-state compliant-actuator (arm_kp
    # =120) gravity/load droop under the Sharpa hands' own weight, not a
    # kinematic or rate-limit error. A Cartesian-target-inflation resolve
    # (grasp_expert.py's own proven `_coupled_maybe_resolve` recipe for
    # the Dex3 track) was tried here and causally measured to make the
    # gap WORSE at this already-extreme precontact reach (see the same
    # history doc) -- not used. The actual fix is
    # SharpaGraspEnv's config-gated arm_gravity_compensation (see
    # grasp_config.py/sharpa_grasp_env.py); this state only HONESTLY
    # measures the actual settled pose and gates the transition on it.
    precontact_ori_tol_deg: float = 5.0  # Precontact Tracking Gate orientation tolerance (Session 39 spec)
    precontact_stable_streak_required: int = 15  # Precontact Tracking Gate: consecutive ticks required


@dataclass
class BimanualGraspOutcome:
    state: BimanualGraspState
    failure_reason: BimanualFailureReason | None
    step_count: int
    per_side_group_contact: dict  # {side: {group: bool ever-contacted}}
    per_side_group_peak_force: dict
    per_side_group_net_force: dict
    max_bilateral_stable_streak: int
    max_left_stable_streak: int
    max_right_stable_streak: int
    max_proximal_object_penetration: float
    max_hand_hand_force: float
    object_xy_displacement: float
    object_angular_velocity_peak: float
    object_angular_velocity_rms: float
    tabletop_hold_steps_achieved: int
    air_hold_steps_achieved: int
    lift_height_achieved_m: float
    gate_a: bool
    gate_b: bool
    gate_c: bool
    gate_d: bool
    precontact_final_pos_error_m: dict  # {side: m}, FINGERTIP_PRECONTACT actual-vs-target Cartesian error
    precontact_final_ori_error_deg: dict  # {side: deg}
    precontact_max_stable_streak: int
    precontact_gate: bool  # Session 39 Precontact Tracking Gate (see module docstring)


class SharpaBimanualGraspExpert:
    """Drives a SharpaGraspEnv with BOTH hands grasping the object
    simultaneously (mirrored across Y). Call .step() once per control
    tick; .run() loops to a terminal state or step budget."""

    # See _arm_action_toward_target's docstring: a plain ctrl-ramp-rate
    # reduction (not a gain/damping change) that empirically eliminates a
    # real wrist_pitch dynamics instability shared with the single-hand
    # controller.
    RAMP_FRACTION = 1.0
    APPROACH_STEPS = 200
    WAYPOINT_COUNT = 4
    WAYPOINT_TICKS = 60

    def __init__(self, env, config: BimanualGraspConfig | None = None):
        self.env = env
        self.config = config or BimanualGraspConfig()
        self.state = BimanualGraspState.STABLE_START
        self.failure_reason: BimanualFailureReason | None = None
        self._state_step = 0
        self._total_step = 0
        self._just_advanced = False
        self._force_settle_streak = 0
        self._group_ever_contacted = {s: {g: False for g in sc.GROUPS} for s in SIDES}
        self._group_peak_force = {s: {g: 0.0 for g in sc.GROUPS} for s in SIDES}
        self._group_net_force = {s: {g: 0.0 for g in sc.GROUPS} for s in SIDES}
        self._max_proximal_pen = 0.0
        self._max_hand_hand = 0.0
        self._tabletop_hold_steps = 0
        self._air_hold_steps = 0
        self._lift_height_achieved = 0.0
        self._obj_xy_hist: list[np.ndarray] = []
        self._obj_angvel_hist: list[float] = []
        self._initial_obj_xy: np.ndarray | None = None
        self._left_streak = 0
        self._right_streak = 0
        self._bilateral_streak = 0
        self._max_left_streak = 0
        self._max_right_streak = 0
        self._max_bilateral_streak = 0
        self._locked_R: dict | None = None  # set at WRIST_ALIGN entry, see module docstring
        self.wrist_orientation_drift_deg: float = float("inf")
        self._precontact_stable_streak = 0
        self._max_precontact_stable_streak = 0
        self._precontact_final_pos_error = {"left": float("inf"), "right": float("inf")}
        self._precontact_final_ori_error_deg = {"left": float("inf"), "right": float("inf")}

        model = env.model
        waist_dof = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in wbc.WAIST_JOINTS])
        coupled_names = list(wbc.WAIST_JOINTS) + list(tc.LEFT_ARM_JOINTS) + list(tc.RIGHT_ARM_JOINTS)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in coupled_names]
        joint_low = np.array([model.jnt_range[j][0] for j in jids])
        joint_high = np.array([model.jnt_range[j][1] for j in jids])
        self.ik = CoupledBilateralIK(
            model=model, left_palm_site=env._left_palm_site, right_palm_site=env._right_palm_site,
            waist_dof_adr=waist_dof, left_arm_dof_adr=env._arm_dof_adr[:7], right_arm_dof_adr=env._arm_dof_adr[7:],
            waist_qpos_adr=env._waist_qpos_adr, left_arm_qpos_adr=env._arm_qpos_adr[:7],
            right_arm_qpos_adr=env._arm_qpos_adr[7:], joint_low=joint_low, joint_high=joint_high,
        )
        self._rest_q = np.concatenate([
            env.data.qpos[env._waist_qpos_adr].copy(), env.data.qpos[env._arm_qpos_adr[:7]].copy(),
            env.data.qpos[env._arm_qpos_adr[7:]].copy(),
        ])
        self._arm_ik_target = env._arm_target.copy()
        self._waist_ik_target = env._waist_target.copy()

    # ------------------------------------------------------------------
    def _object_pos(self) -> np.ndarray:
        return self.env.data.qpos[self.env._object_qpos_adr:self.env._object_qpos_adr + 3].copy()

    def _object_angvel(self) -> float:
        return float(np.linalg.norm(self.env.data.qvel[self.env._object_dof_adr + 3:self.env._object_dof_adr + 6]))

    def _mirrored_targets(self, standoff: float, height: float, y_offset: float) -> dict:
        obj_pos = self._object_pos()
        return {side: obj_pos + np.array([-standoff, Y_SIGN[side] * y_offset, height]) for side in SIDES}

    def _solve_both(self, targets: dict, R: dict, require_orientation: bool, ori_task_weight: float,
                     rest_q: np.ndarray | None = None):
        data = self.env.data
        scratch = mujoco.MjData(self.env.model)
        scratch.qpos[:] = data.qpos
        mujoco.mj_forward(self.env.model, scratch)
        return self.ik.solve(
            scratch, targets["left"], R["left"], targets["right"], R["right"],
            rest_q=rest_q if rest_q is not None else self._rest_q, pos_tol=self.config.ik_pos_tol,
            require_orientation=require_orientation, ori_task_weight=ori_task_weight,
        )

    def _apply_ik_result(self, result) -> None:
        self._waist_ik_target = result.waist_q.copy()
        self._arm_ik_target[:7] = result.left_q.copy()
        self._arm_ik_target[7:] = result.right_q.copy()

    def _arm_action_toward_target(self) -> np.ndarray:
        # [This session's finding, verified against BOTH the bimanual AND
        # the pre-existing single-hand controller -- not a bimanual-only
        # bug] targeting a FIXED/independently-chosen wrist orientation
        # (e.g. np.eye(3) at any nonzero weight, or a "locked" orientation
        # captured elsewhere) reliably drives wrist_pitch (very low
        # armature=0.01, zero dof_damping -- see grasp_config.py/
        # model_builder.py, both protected/unmodified) into a real,
        # reproducible divergence to its own hard joint limit -- confirmed
        # identical in SharpaSingleHandGraspExpert (qpos ends ~1.7rad from
        # ctrl, pinned against a genuine wrist_roll_link<->wrist_yaw_link
        # self-collision), which never checked IK convergence after
        # FOREARM_APPROACH and so never surfaced it. Neither slowing the
        # ctrl ramp rate (tried: ineffective, only delays the divergence)
        # nor trimming the IK's joint-limit margin (tried: ineffective)
        # fixed this -- the actual, verified fix is at the IK CALL SITES
        # (see FOREARM_APPROACH/FINGERTIP_PRECONTACT below): always
        # soft-anchor the orientation task to whatever orientation the arm
        # is CURRENTLY, already stably holding (small ori_task_weight,
        # never a fixed/locked target), which keeps the solver from ever
        # choosing the unstable configuration in the first place.
        # RAMP_FRACTION is kept as a class constant (currently 1.0, i.e.
        # no slowdown) only as a documented, easy-to-flip knob if a
        # similar instability resurfaces elsewhere.
        env = self.env
        scale = env.config.arm_action_scale * self.RAMP_FRACTION
        delta = np.clip(self._arm_ik_target - env._arm_target, -scale, scale)
        return delta / max(env.config.arm_action_scale, 1e-9)

    def _waist_action_toward_target(self) -> np.ndarray:
        env = self.env
        scale = env.config.waist_action_scale * self.RAMP_FRACTION
        delta = np.clip(self._waist_ik_target - env._waist_target, -scale, scale)
        return delta / max(env.config.waist_action_scale, 1e-9)

    def _zero_action(self) -> np.ndarray:
        from humanoid_learning.envs.sharpa_grasp_env import ACTION_DIM
        return np.zeros(ACTION_DIM)

    def _group_action(self, deltas: dict) -> np.ndarray:
        """deltas: {side: {group: delta}} -> 8-dim [left x4, right x4] action slice."""
        vec = np.zeros(2 * len(sc.GROUPS))
        for side_idx, side in enumerate(SIDES):
            for g_idx, group in enumerate(sc.GROUPS):
                vec[side_idx * len(sc.GROUPS) + g_idx] = deltas.get(side, {}).get(group, 0.0) / max(
                    self.env.config.hand_synergy_action_scale, 1e-9
                )
        return vec

    def _group_contact_now(self, side: str, group: str) -> bool:
        peak, net = self.env._group_contact_force(side, group)
        self._group_peak_force[side][group] = max(self._group_peak_force[side][group], peak)
        self._group_net_force[side][group] = max(self._group_net_force[side][group], net)
        if peak > self.config.contact_force_threshold_n:
            self._group_ever_contacted[side][group] = True
        return peak > self.config.contact_force_threshold_n

    def _group_force_vector(self, side: str, group: str) -> np.ndarray:
        """Net world-frame force vector (not magnitude) this GROUP applies
        to the object, for the opposition dot-product check below."""
        model, data = self.env.model, self.env.data
        obj_body = self.env._object_body_id
        prefixes = tuple(sc.sharpa_body(side, f, "") for f in sc.GROUP_FINGERS[group])
        force_sum = np.zeros(3)
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
            contact_R = np.asarray(c.frame, dtype=np.float64).reshape(3, 3)
            force_on_geom2 = contact_R.T @ force6[:3]
            force_sum += -force_on_geom2 if b1 == obj_body else force_on_geom2
        return force_sum

    def _side_stable(self, side: str) -> bool:
        """Section 6/Dex3-analog topology check for ONE side: thumb
        touching AND (index or middle) touching AND wrap touching AND
        thumb's force genuinely OPPOSES the index/middle combined force
        (dot product < 0 -- ruling out both pressing from the same side,
        the exact Dex3 `_opposing_normal_score` failure mode this
        mirrors) AND this side is not in a hand-hand collision."""
        env = self.env
        thumb_peak, _ = env._group_contact_force(side, "thumb")
        index_peak, _ = env._group_contact_force(side, "index")
        middle_peak, _ = env._group_contact_force(side, "middle")
        wrap_peak, _ = env._group_contact_force(side, "wrap")
        thumb_touch = thumb_peak > self.config.contact_force_threshold_n
        finger_touch = (index_peak > self.config.contact_force_threshold_n) or (middle_peak > self.config.contact_force_threshold_n)
        wrap_touch = wrap_peak > self.config.contact_force_threshold_n
        if not (thumb_touch and finger_touch and wrap_touch):
            return False
        thumb_f = self._group_force_vector(side, "thumb")
        finger_f = self._group_force_vector(side, "index") + self._group_force_vector(side, "middle")
        if np.linalg.norm(thumb_f) < 1e-6 or np.linalg.norm(finger_f) < 1e-6:
            return False
        opposing = float(np.dot(thumb_f, finger_f)) < 0.0
        return opposing

    def _bilateral_stable_now(self) -> tuple[bool, bool, bool]:
        left_ok = self._side_stable("left")
        right_ok = self._side_stable("right")
        hh_force = self.env._hand_hand_contact_force()
        if hh_force > self.config.hand_hand_force_limit_n:
            left_ok = right_ok = False
        return left_ok, right_ok, left_ok and right_ok

    def _proximal_penetration_ok(self) -> bool:
        return self.env._proximal_object_penetration() <= self.config.proximal_penetration_tolerance_m

    def _fail(self, reason: BimanualFailureReason) -> None:
        self.state = BimanualGraspState.FAILURE
        self.failure_reason = reason

    def _advance(self, next_state: BimanualGraspState) -> None:
        self.state = next_state
        self._state_step = 0
        self._just_advanced = True

    def _track_stability(self) -> None:
        self._obj_xy_hist.append(self._object_pos()[:2])
        self._obj_angvel_hist.append(self._object_angvel())
        self._max_proximal_pen = max(self._max_proximal_pen, self.env._proximal_object_penetration())
        self._max_hand_hand = max(self._max_hand_hand, self.env._hand_hand_contact_force())
        left_ok, right_ok, both_ok = self._bilateral_stable_now()
        self._left_streak = self._left_streak + 1 if left_ok else 0
        self._right_streak = self._right_streak + 1 if right_ok else 0
        self._bilateral_streak = self._bilateral_streak + 1 if both_ok else 0
        self._max_left_streak = max(self._max_left_streak, self._left_streak)
        self._max_right_streak = max(self._max_right_streak, self._right_streak)
        self._max_bilateral_streak = max(self._max_bilateral_streak, self._bilateral_streak)

    def object_xy_displacement(self) -> float:
        if self._initial_obj_xy is None or not self._obj_xy_hist:
            return 0.0
        return float(np.max([np.linalg.norm(xy - self._initial_obj_xy) for xy in self._obj_xy_hist]))

    def _contact_still_present(self) -> bool:
        return any(
            self.env._group_contact_force(side, g)[1] > 0.05
            for side in SIDES for g in ("thumb", "index", "middle")
        )

    # ------------------------------------------------------------------
    def step(self) -> np.ndarray:
        action = self._zero_action()
        state = self.state
        self._just_advanced = False
        cfg = self.config

        if state == BimanualGraspState.STABLE_START:
            if self._state_step >= 10:
                self._advance(BimanualGraspState.NATURAL_ARM_LIFT)

        elif state == BimanualGraspState.NATURAL_ARM_LIFT:
            if self._state_step == 0:
                lp, lR = self.env.palm_pose("left")
                rp, rR = self.env.palm_pose("right")
                targets = {"left": lp + np.array([0, 0, 0.15]), "right": rp + np.array([0, 0, 0.15])}
                R = {"left": lR, "right": rR}
                result = self._solve_both(targets, R, require_orientation=False, ori_task_weight=0.1)
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            if self._state_step >= 60:
                self._advance(BimanualGraspState.FOREARM_APPROACH)

        elif state == BimanualGraspState.FOREARM_APPROACH:
            if self._state_step == 0:
                # [This session's real root-cause fix] targeting a FIXED
                # np.eye(3) orientation at ori_task_weight=0.0 (nominally
                # "orientation ignored") let the position-redundant DLS
                # solve drift wrist_pitch into a configuration where the
                # closed-loop PD (kp=120, dof_damping=0, armature=0.01 --
                # see grasp_config.py/model_builder.py, both protected)
                # is genuinely dynamically unstable: qvel diverges to
                # >3rad/s within ~50 ticks regardless of ramp speed, joint
                # margin, or null-space anchor (all independently tried
                # and all failed this session), and the joint is flung to
                # its own hard limit every time. Soft-anchoring to the
                # CURRENT achieved orientation at ori_task_weight=0.1 --
                # the exact recipe NATURAL_ARM_LIFT already used
                # successfully one state earlier -- removes the
                # instability entirely (verified: wrist_pitch converges
                # cleanly, max excursion 0.5rad, matching NATURAL_ARM_LIFT's
                # own already-stable configuration instead of drifting
                # into an unrelated one).
                targets = self._mirrored_targets(cfg.approach_standoff_m, cfg.approach_height_m, cfg.approach_y_offset_m)
                lR = self.env.palm_pose("left")[1].copy()
                rR = self.env.palm_pose("right")[1].copy()
                result = self._solve_both(targets, {"left": lR, "right": rR}, require_orientation=False, ori_task_weight=0.1)
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            if self._state_step >= self.APPROACH_STEPS:
                self._advance(BimanualGraspState.WRIST_ALIGN)

        elif state == BimanualGraspState.WRIST_ALIGN:
            # [This session's design, after the original re-solve-with-
            # hard-orientation-weight approach was found to freeze the IK
            # solver -- error_history flat across all 200 iterations,
            # joint_limit_margin pinned at ~0, a genuine degenerate/
            # saturated configuration, not a slow-convergence issue] does
            # NOT run a second IK solve at all. Instead it holds
            # FOREARM_APPROACH's own target and MEASURES whether the
            # achieved palm orientation has actually settled (max angular
            # drift over the last 30 ticks) -- a directly measurable,
            # non-fabricated criterion for "orientation error has been
            # reduced" that this controller can be honest about: if the
            # orientation is STILL changing by more than the tolerance,
            # this is reported as WRIST_ORIENTATION_NOT_STABLE, never
            # silently treated as WRIST_ALIGN succeeding.
            if self._state_step == 0:
                self._wrist_align_R_hist = {"left": [], "right": []}
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            lR = self.env.palm_pose("left")[1].copy()
            rR = self.env.palm_pose("right")[1].copy()
            self._wrist_align_R_hist["left"].append(lR)
            self._wrist_align_R_hist["right"].append(rR)
            if self._state_step >= 60:
                def _max_angle_dev_deg(hist: list[np.ndarray]) -> float:
                    r_final = hist[-1]
                    devs = []
                    for r in hist[-30:]:
                        r_delta = r_final.T @ r
                        cos_ang = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)
                        devs.append(np.degrees(np.arccos(cos_ang)))
                    return max(devs)

                left_dev = _max_angle_dev_deg(self._wrist_align_R_hist["left"])
                right_dev = _max_angle_dev_deg(self._wrist_align_R_hist["right"])
                self.wrist_orientation_drift_deg = max(left_dev, right_dev)
                self._locked_R = {"left": lR, "right": rR}
                if self.wrist_orientation_drift_deg > cfg.wrist_orientation_stability_tol_deg:
                    self._fail(BimanualFailureReason.WRIST_ORIENTATION_NOT_STABLE)
                    return action
                self._advance(BimanualGraspState.FIVE_FINGER_PRESHAPE)

        elif state == BimanualGraspState.FIVE_FINGER_PRESHAPE:
            if self._state_step == 0:
                self.env.set_preshape("left", 1.0)
                self.env.set_preshape("right", 1.0)
            if self._state_step >= 30:
                if not self._proximal_penetration_ok() or self.env._hand_hand_contact_force() > cfg.hand_hand_force_limit_n:
                    self._fail(BimanualFailureReason.SELF_COLLISION_BEFORE_CONTACT)
                    return action
                self._advance(BimanualGraspState.FINGERTIP_PRECONTACT)

        elif state == BimanualGraspState.FINGERTIP_PRECONTACT:
            # [This session's finding] a SINGLE IK jump straight from
            # WRIST_ALIGN's converged pose to the full precontact target
            # is kinematically valid (IK success=True, <1cm error) but
            # requires a large one-shot change in the low-inertia,
            # zero-damping wrist joints (e.g. wrist_roll by >1.5rad in one
            # solve) -- exactly the pattern already found to destabilize
            # wrist_pitch during FOREARM_APPROACH, now reproduced in
            # wrist_roll instead. Breaking the same Cartesian move into
            # WAYPOINT_COUNT smaller interpolated sub-targets, each
            # re-solved with the same CURRENT-orientation soft anchor and
            # held for WAYPOINT_TICKS before advancing, keeps every
            # individual joint delta small -- the same fix category as
            # FOREARM_APPROACH's, applied generically instead of
            # re-diagnosing per-joint each time.
            if self._state_step == 0:
                self._precontact_start = {"left": self.env.palm_pose("left")[0].copy(),
                                           "right": self.env.palm_pose("right")[0].copy()}
                self._precontact_final = self._mirrored_targets(
                    cfg.precontact_standoff_m, cfg.precontact_height_m, cfg.precontact_y_offset_m
                )
                # Target orientation for the gate check is WRIST_ALIGN's
                # already-locked, measured-stable orientation (module
                # docstring: "LOCK that exact orientation as the explicit
                # WRIST_ALIGN/FINGERTIP_PRECONTACT target"), not a freshly
                # re-measured one -- these should coincide closely since
                # every waypoint solve below only soft-anchors
                # (ori_task_weight=0.05) to whatever orientation is
                # currently held.
                self._precontact_final_R = {s: self._locked_R[s].copy() for s in SIDES}
                self._precontact_waypoint = 0
                self._precontact_stable_streak = 0
            if self._state_step % self.WAYPOINT_TICKS == 0 and self._precontact_waypoint < self.WAYPOINT_COUNT:
                self._precontact_waypoint += 1
                frac = self._precontact_waypoint / self.WAYPOINT_COUNT
                targets = {
                    side: (1 - frac) * self._precontact_start[side] + frac * self._precontact_final[side]
                    for side in SIDES
                }
                lR = self.env.palm_pose("left")[1].copy()
                rR = self.env.palm_pose("right")[1].copy()
                result = self._solve_both(targets, {"left": lR, "right": rR}, require_orientation=False, ori_task_weight=0.05)
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()

            if self._precontact_waypoint >= self.WAYPOINT_COUNT:
                # [Session 39 finding -- see docs/history/
                # PHASE4_GRASP_SESSION_39.md] the ctrl register/
                # rate-limited chase has nothing left to converge to at
                # this point (it reaches the IK-solved joint target
                # exactly). Two independent Cartesian-target-inflation
                # resolve variants (Dex3's own _coupled_maybe_resolve
                # recipe, both with and without a posture-hold rest_q)
                # were causally tested here and BOTH measured WORSE
                # (gap grew from ~6.9cm to 10-25cm, joint norm to
                # >1rad) -- this state's redundant 17-DOF solve, at this
                # already-extreme precontact reach, does not have the
                # locally-linear droop-vs-target relationship the Dex3
                # resolve assumes; extrapolating the Cartesian target
                # drives the IK into a qualitatively different, LESS
                # favorable arm configuration instead of compensating.
                # That IK-side compensation avenue is therefore not
                # used. What IS applied is a real, physically-grounded
                # feedforward: SharpaGraspEnv's arm_gravity_compensation
                # (config-gated, see grasp_config.py/sharpa_grasp_env.py)
                # cancels the actual measured qfrc_bias/kp steady-state
                # droop AT THE ACTUATOR, not via a kinematic guess. This
                # block does NOT re-solve IK -- it only measures the
                # ACTUAL settled pose and HONESTLY gates the transition
                # on it (Precontact Tracking Gate), never advancing on a
                # fixed tick count regardless of convergence (the
                # pre-Session-39 behavior).
                left_actual, left_R = self.env.palm_pose("left")
                right_actual, right_R = self.env.palm_pose("right")
                left_pos_err = float(np.linalg.norm(self._precontact_final["left"] - left_actual))
                right_pos_err = float(np.linalg.norm(self._precontact_final["right"] - right_actual))

                def _ang_deg(Ra, Rb):
                    r_delta = Ra.T @ Rb
                    cos_ang = np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)
                    return float(np.degrees(np.arccos(cos_ang)))

                left_ori_err = _ang_deg(left_R, self._precontact_final_R["left"])
                right_ori_err = _ang_deg(right_R, self._precontact_final_R["right"])
                self._precontact_final_pos_error = {"left": left_pos_err, "right": right_pos_err}
                self._precontact_final_ori_error_deg = {"left": left_ori_err, "right": right_ori_err}

                no_hand_hand = self.env._hand_hand_contact_force() <= cfg.hand_hand_force_limit_n
                stable_now = (
                    left_pos_err <= cfg.ik_pos_tol and right_pos_err <= cfg.ik_pos_tol
                    and left_ori_err <= cfg.precontact_ori_tol_deg and right_ori_err <= cfg.precontact_ori_tol_deg
                    and no_hand_hand
                )
                self._precontact_stable_streak = self._precontact_stable_streak + 1 if stable_now else 0
                self._max_precontact_stable_streak = max(self._max_precontact_stable_streak, self._precontact_stable_streak)
                if self._precontact_stable_streak >= cfg.precontact_stable_streak_required:
                    self._advance(BimanualGraspState.CONTACT_ACQUIRE)
                elif self._state_step >= cfg.max_steps_per_state:
                    self._fail(BimanualFailureReason.PRECONTACT_TRACKING_NOT_ACHIEVED)

        elif state == BimanualGraspState.CONTACT_ACQUIRE:
            # Both sides close INDEPENDENTLY/asynchronously: a side/group
            # that has already registered contact stops advancing while
            # the other side keeps closing (Section 4 requirement 7/8).
            deltas = {"left": {}, "right": {}}
            for side in SIDES:
                for group in ("index", "middle", "wrap"):
                    if not self._group_contact_now(side, group):
                        deltas[side][group] = cfg.close_rate_per_step
            action[17:25] = self._group_action(deltas)
            both_have_some_contact = all(
                any(self._group_ever_contacted[side][g] for g in ("index", "middle", "wrap")) for side in SIDES
            )
            if both_have_some_contact:
                self._advance(BimanualGraspState.THUMB_OPPOSE)
            elif self._state_step >= cfg.max_steps_per_state:
                self._fail(BimanualFailureReason.TIMEOUT)

        elif state == BimanualGraspState.THUMB_OPPOSE:
            deltas = {"left": {}, "right": {}}
            for side in SIDES:
                for group in sc.GROUPS:
                    if not self._group_contact_now(side, group):
                        deltas[side][group] = cfg.close_rate_per_step
            action[17:25] = self._group_action(deltas)
            if all(self._group_ever_contacted[side][g] for side in SIDES for g in sc.GROUPS):
                self._advance(BimanualGraspState.ENVELOPING_CLOSE)
            elif self._state_step >= cfg.max_steps_per_state:
                self._advance(BimanualGraspState.ENVELOPING_CLOSE)

        elif state == BimanualGraspState.ENVELOPING_CLOSE:
            lo, hi = cfg.target_force_band_n
            deltas = {"left": {}, "right": {}}
            for side in SIDES:
                for group in sc.GROUPS:
                    _, net = self.env._group_contact_force(side, group)
                    self._group_net_force[side][group] = max(self._group_net_force[side][group], net)
                    if net < lo:
                        deltas[side][group] = cfg.close_rate_per_step * 0.5
            action[17:25] = self._group_action(deltas)
            all_in_band = all(
                lo <= self.env._group_contact_force(side, g)[1] <= hi * 1.5 for side in SIDES for g in sc.GROUPS
            )
            if all_in_band or self._state_step >= cfg.max_steps_per_state:
                self._advance(BimanualGraspState.FORCE_SETTLE)

        elif state == BimanualGraspState.FORCE_SETTLE:
            # [Section 5 fix] REAL force-band check, not "> 0.0" -- every
            # required group on BOTH sides must sit within
            # target_force_band_n, peak below the safety limit, AND the
            # bilateral topology/opposition check must hold, for
            # force_settle_hold_steps CONSECUTIVE ticks.
            lo, hi = cfg.target_force_band_n
            in_band = all(
                lo <= self.env._group_contact_force(side, g)[1] <= hi for side in SIDES for g in sc.GROUPS
            )
            peak_ok = all(
                self.env._group_contact_force(side, g)[0] <= self.env.config.finger_force_safety_limit
                for side in SIDES for g in sc.GROUPS
            )
            _, _, bilateral_ok = self._bilateral_stable_now()
            if in_band and peak_ok and bilateral_ok:
                self._force_settle_streak += 1
            else:
                self._force_settle_streak = 0
            self._track_stability()
            if self._force_settle_streak >= cfg.force_settle_hold_steps:
                self._advance(BimanualGraspState.TABLETOP_HOLD)
                self._initial_obj_xy = self._object_pos()[:2].copy()
            elif self._state_step >= cfg.max_steps_per_state:
                self._fail(BimanualFailureReason.CONTACT_LOST)

        elif state == BimanualGraspState.TABLETOP_HOLD:
            self._track_stability()
            if not self._contact_still_present():
                self._fail(BimanualFailureReason.CONTACT_LOST)
                return action
            self._tabletop_hold_steps += 1
            required = sim_time_to_steps(self.env, cfg.tabletop_hold_seconds)
            if self._tabletop_hold_steps >= required:
                if self.object_xy_displacement() > cfg.object_xy_displacement_limit_m:
                    self._fail(BimanualFailureReason.OBJECT_MOVED_TOO_MUCH)
                elif max(self._obj_angvel_hist, default=0.0) > cfg.object_peak_angular_velocity_limit:
                    self._fail(BimanualFailureReason.OBJECT_ANGULAR_VELOCITY_EXCEEDED)
                else:
                    self._advance(BimanualGraspState.LIFT)

        elif state == BimanualGraspState.LIFT:
            if self._state_step == 0:
                lp, lR = self.env.palm_pose("left")
                rp, rR = self.env.palm_pose("right")
                self._lift_start_obj_z = self._object_pos()[2]
                lift_h = cfg.lift_height_m
                targets = {"left": lp + np.array([0, 0, lift_h]), "right": rp + np.array([0, 0, lift_h])}
                result = self._solve_both(targets, {"left": lR, "right": rR}, require_orientation=True, ori_task_weight=1.0)
                self._apply_ik_result(result)
            action[0:3] = self._waist_action_toward_target()
            action[3:17] = self._arm_action_toward_target()
            self._track_stability()
            self._lift_height_achieved = max(self._lift_height_achieved, self._object_pos()[2] - self._lift_start_obj_z)
            if not self._contact_still_present():
                self._fail(BimanualFailureReason.LIFT_FAILED)
                return action
            if self._state_step >= 100:
                self._advance(BimanualGraspState.AIR_HOLD)

        elif state == BimanualGraspState.AIR_HOLD:
            self._track_stability()
            if not self._contact_still_present():
                self._fail(BimanualFailureReason.LIFT_FAILED)
                return action
            self._air_hold_steps += 1
            required = sim_time_to_steps(self.env, cfg.air_hold_seconds)
            if self._air_hold_steps >= required:
                self.state = BimanualGraspState.SUCCESS

        if not self._just_advanced:
            self._state_step += 1
        self._total_step += 1
        return action

    # ------------------------------------------------------------------
    def run(self, max_total_steps: int = 8000) -> BimanualGraspOutcome:
        obs, info = self.env.reset(seed=getattr(self.env, "_last_seed", 0))
        for _ in range(max_total_steps):
            if self.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
                break
            action = self.step()
            obs, r, term, trunc, info = self.env.step(action)
            if info.get("unstable"):
                self._fail(BimanualFailureReason.NUMERICAL_ERROR)
                break
            if trunc and self.state not in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
                self._fail(BimanualFailureReason.TIMEOUT)
                break
        else:
            if self.state not in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
                self._fail(BimanualFailureReason.TIMEOUT)

        angvel_peak = float(np.max(self._obj_angvel_hist)) if self._obj_angvel_hist else 0.0
        angvel_rms = float(np.sqrt(np.mean(np.square(self._obj_angvel_hist)))) if self._obj_angvel_hist else 0.0
        gate_a = (
            self.state != BimanualGraspState.FAILURE
            and self._max_bilateral_streak >= self.config.bilateral_streak_required
            and self.object_xy_displacement() <= self.config.object_xy_displacement_limit_m
            and angvel_peak <= self.config.object_peak_angular_velocity_limit
            and self._max_proximal_pen <= self.config.proximal_penetration_tolerance_m
            and self._max_hand_hand <= self.config.hand_hand_force_limit_n
        )
        gate_b = self._tabletop_hold_steps >= sim_time_to_steps(self.env, self.config.tabletop_hold_seconds)
        gate_c = self._lift_height_achieved >= self.config.lift_height_m
        gate_d = self._air_hold_steps >= sim_time_to_steps(self.env, self.config.air_hold_seconds)
        precontact_gate = (
            self._max_precontact_stable_streak >= self.config.precontact_stable_streak_required
        )

        return BimanualGraspOutcome(
            state=self.state, failure_reason=self.failure_reason, step_count=self._total_step,
            per_side_group_contact={s: dict(self._group_ever_contacted[s]) for s in SIDES},
            per_side_group_peak_force={s: dict(self._group_peak_force[s]) for s in SIDES},
            per_side_group_net_force={s: dict(self._group_net_force[s]) for s in SIDES},
            max_bilateral_stable_streak=self._max_bilateral_streak, max_left_stable_streak=self._max_left_streak,
            max_right_stable_streak=self._max_right_streak, max_proximal_object_penetration=self._max_proximal_pen,
            max_hand_hand_force=self._max_hand_hand, object_xy_displacement=self.object_xy_displacement(),
            object_angular_velocity_peak=angvel_peak, object_angular_velocity_rms=angvel_rms,
            tabletop_hold_steps_achieved=self._tabletop_hold_steps, air_hold_steps_achieved=self._air_hold_steps,
            lift_height_achieved_m=self._lift_height_achieved, gate_a=gate_a, gate_b=gate_b, gate_c=gate_c, gate_d=gate_d,
            precontact_final_pos_error_m=dict(self._precontact_final_pos_error),
            precontact_final_ori_error_deg=dict(self._precontact_final_ori_error_deg),
            precontact_max_stable_streak=self._max_precontact_stable_streak,
            precontact_gate=precontact_gate,
        )
