"""Block Size/Position Feasibility Map session: runs the UNCHANGED
canonical BimanualSidePinchExpert/FixedBaseGraspEnv controller (no
experimental flags, all defaults) across a grid of object
half_size/x/y and records rich per-condition diagnostics, to separate
four hypotheses:

    A1. a size/position region exists where canonical succeeds
        (Gate A, no relaxed threshold) -> controller/target calibration
        is the real issue, not gripper geometry.
    A2. only a single or extremely narrow point succeeds -> a
        geometric sweet spot exists but the controller is fragile there.
    A3. static fingertip feasibility exists everywhere tried but every
        DYNAMIC rollout still fails -> the THUMB_OPPOSE dynamic path is
        the primary cause, not aperture/size.
    A4. static tripod contact is geometrically impossible everywhere
        tried -> a real Dex3 aperture/contact-geometry limit.

Read-only sweep: never imports into or is imported by the production
control loop. Every condition uses GraspExpertConfig()/GraspEnvConfig()
at their canonical defaults except object_pos/object_half_size -- no
kp/friction/timestep/frame_skip/solver/mass/seed/force-threshold/
vertical_stagger/yaw/table-height change, and every experimental flag
added in the 28th session (unload_finger_on_force_limit,
persist_safety_synergy_rollback, use_net_group_force,
use_fingertip_collision_pads, latch_thumb_after_first_contact,
lock_synergy_after_tripod, hold_opposing_fingers_during_thumb_oppose,
thumb_crossing_*, lead/trailing_thumb_*, freeze_contact_ref_during_
thumb_oppose, postcontact_x_*) stays at its default (off/neutral)."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig
from humanoid_learning.envs.grasp_env import FixedBaseGraspEnv
from humanoid_learning.expert.grasp_expert import BimanualSidePinchExpert, GraspExpertConfig, GraspState

SEED = 0
MAX_STEPS = 1500


def _net_group_force(env: FixedBaseGraspEnv, side: str, group_idx: int) -> float:
    """Physical net resultant force for one finger group, computed the
    SAME way grasp_env.py's use_net_group_force=True branch does (sum of
    world-frame contact forces, not the largest single point) -- kept as
    an independent read function here so the sweep never has to flip the
    env's own config.use_net_group_force flag (which would also change
    what the live substep safety loop sees mid-rollout)."""
    model, data = env.model, env.data
    obj_body = env._object_body_id
    group_name = ("thumb", "index", "middle")[group_idx]
    prefix = f"{side}_hand_{group_name}"
    force_sum = np.zeros(3, dtype=np.float64)
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if obj_body not in (b1, b2):
            continue
        other = b2 if b1 == obj_body else b1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        if not name.startswith(prefix):
            continue
        force6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force6)
        contact_R = np.asarray(c.frame, dtype=np.float64).reshape(3, 3)
        force_on_geom2 = contact_R.T @ force6[:3]
        force_sum += -force_on_geom2 if b1 == obj_body else force_on_geom2
    return float(np.linalg.norm(force_sum))


@dataclass
class ConditionResult:
    half_size: float
    full_size_cm: float
    obj_x: float
    obj_y: float
    seed: int

    final_state: str = ""
    failure_reason: str = ""
    steps_run: int = 0

    state_entry_step: dict = field(default_factory=dict)

    max_bilateral_tripod_streak: int = 0
    max_bilateral_multifinger_streak: int = 0
    left_group_contact: tuple = (False, False, False)
    right_group_contact: tuple = (False, False, False)

    object_xy_displacement: float = 0.0
    vertical_peak: float = 0.0
    angular_speed_peak: float = 0.0
    angular_speed_rms: float = 0.0

    peak_single_contact_force: float = 0.0
    peak_net_group_force: float = 0.0
    peak_thumb_single_force: float = 0.0

    hand_hand_peak_force: float = 0.0
    hand_hand_streak: int = 0
    thumb_thumb_collision: bool = False

    safety_events: int = 0
    force_recrossing_count: int = 0

    thumb_first_contact_obj_xy: tuple | None = None
    thumb_oppose_entry_obj_xy: tuple | None = None
    tripod_settle_entry_obj_xy: tuple | None = None
    thumb_oppose_to_tripod_push_distance: float | None = None

    min_thumb1_joint_margin: float = float("inf")

    gate_a_pass: bool = False


def run_condition(half_size: float, obj_x: float, obj_y: float, seed: int = SEED, max_steps: int = MAX_STEPS) -> ConditionResult:
    result = ConditionResult(
        half_size=half_size, full_size_cm=half_size * 2 * 100.0, obj_x=obj_x, obj_y=obj_y, seed=seed,
    )
    env = FixedBaseGraspEnv(GraspEnvConfig(object_pos=(obj_x, obj_y, 0.0), arm_kp=120.0, object_half_size=half_size))
    env.reset(seed=seed)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())  # every field at its canonical default
    model, data = env.model, env.data
    obj_qpos_adr = env._object_qpos_adr
    obj_dof_adr = env._object_dof_adr

    thumb1_qpos_adr = {"left": env._left_finger_qpos_adr[1], "right": env._right_finger_qpos_adr[1]}
    thumb1_jid = {
        "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_hand_thumb_1_joint"),
        "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_hand_thumb_1_joint"),
    }

    initial_xy = data.qpos[obj_qpos_adr:obj_qpos_adr + 2].copy()
    initial_z = float(data.qpos[obj_qpos_adr + 2])
    ang_speeds = []
    had_thumb_safety_prev = False
    seen_states: set[str] = set()
    outcome = None

    for i in range(max_steps):
        outcome = expert.step()
        state_name = outcome.state.name
        if state_name not in seen_states:
            seen_states.add(state_name)
            result.state_entry_step[state_name] = i

        obj_xy = data.qpos[obj_qpos_adr:obj_qpos_adr + 2].copy()
        obj_z = float(data.qpos[obj_qpos_adr + 2])
        ang_speed = float(np.linalg.norm(data.qvel[obj_dof_adr + 3:obj_dof_adr + 6]))
        ang_speeds.append(ang_speed)
        result.vertical_peak = max(result.vertical_peak, abs(obj_z - initial_z))

        result.peak_single_contact_force = max(
            result.peak_single_contact_force, outcome.left_max_force_raw, outcome.right_max_force_raw
        )
        result.peak_thumb_single_force = max(
            result.peak_thumb_single_force, outcome.left_group_force_raw[0], outcome.right_group_force_raw[0]
        )
        net_peak = max(_net_group_force(env, s, g) for s in ("left", "right") for g in range(3))
        result.peak_net_group_force = max(result.peak_net_group_force, net_peak)

        result.hand_hand_peak_force = max(result.hand_hand_peak_force, outcome.max_hand_hand_force_raw)
        if outcome.max_hand_hand_contact_streak > 0:
            result.thumb_thumb_collision = True

        if result.thumb_first_contact_obj_xy is None and (
            outcome.left_group_contact[0] or outcome.right_group_contact[0]
        ):
            result.thumb_first_contact_obj_xy = tuple(obj_xy)
        if result.thumb_oppose_entry_obj_xy is None and state_name == "THUMB_OPPOSE":
            result.thumb_oppose_entry_obj_xy = tuple(obj_xy)
        if result.tripod_settle_entry_obj_xy is None and state_name == "TRIPOD_SETTLE":
            result.tripod_settle_entry_obj_xy = tuple(obj_xy)

        for side in ("left", "right"):
            jq = data.qpos[thumb1_qpos_adr[side]]
            jlo, jhi = model.jnt_range[thumb1_jid[side]]
            result.min_thumb1_joint_margin = min(result.min_thumb1_joint_margin, float(min(jq - jlo, jhi - jq)))

        thumb_safety_now = any(ev[1] == 0 or ev[0] == "hand_hand" for ev in env.last_safety_events)
        if thumb_safety_now and not had_thumb_safety_prev:
            result.force_recrossing_count += 1
        had_thumb_safety_prev = thumb_safety_now

        if outcome.state in (GraspState.FAILURE, GraspState.SUCCESS):
            break

    result.steps_run = (i + 1) if outcome is not None else 0
    result.final_state = outcome.state.name if outcome else "NONE"
    result.failure_reason = str(outcome.failure_reason) if outcome else ""
    result.max_bilateral_tripod_streak = outcome.max_bilateral_tripod_streak if outcome else 0
    result.max_bilateral_multifinger_streak = outcome.max_bilateral_multifinger_streak if outcome else 0
    result.left_group_contact = outcome.left_group_contact if outcome else (False, False, False)
    result.right_group_contact = outcome.right_group_contact if outcome else (False, False, False)
    result.object_xy_displacement = outcome.object_xy_displacement if outcome else 0.0
    result.hand_hand_streak = outcome.max_hand_hand_contact_streak if outcome else 0
    result.safety_events = outcome.substep_safety_event_count if outcome else 0

    if ang_speeds:
        arr = np.array(ang_speeds)
        result.angular_speed_peak = float(arr.max())
        result.angular_speed_rms = float(np.sqrt(np.mean(arr ** 2)))

    if result.thumb_oppose_entry_obj_xy is not None and result.tripod_settle_entry_obj_xy is not None:
        a = np.array(result.thumb_oppose_entry_obj_xy)
        b = np.array(result.tripod_settle_entry_obj_xy)
        result.thumb_oppose_to_tripod_push_distance = float(np.linalg.norm(b - a))

    # Gate A, literal, not relaxed: >=30 consecutive bilateral tripod
    # steps AND the object must not be bouncing/drifting/spinning while
    # that streak is counted -- a streak alone (even >=30) is not treated
    # as passing if xy displacement or angular speed stayed large.
    result.gate_a_pass = (
        result.max_bilateral_tripod_streak >= 30
        and result.object_xy_displacement < 0.02
        and result.angular_speed_rms < 1.0
    )
    return result


def condition_to_row(r: ConditionResult) -> dict:
    return dict(
        full_size_cm=round(r.full_size_cm, 3), half_size=r.half_size, obj_x=r.obj_x, obj_y=r.obj_y, seed=r.seed,
        final_state=r.final_state, failure_reason=r.failure_reason, steps_run=r.steps_run,
        max_bilateral_tripod_streak=r.max_bilateral_tripod_streak,
        max_bilateral_multifinger_streak=r.max_bilateral_multifinger_streak,
        left_group_contact=r.left_group_contact, right_group_contact=r.right_group_contact,
        object_xy_displacement=round(r.object_xy_displacement, 5),
        vertical_peak=round(r.vertical_peak, 5),
        angular_speed_peak=round(r.angular_speed_peak, 4),
        angular_speed_rms=round(r.angular_speed_rms, 4),
        peak_single_contact_force=round(r.peak_single_contact_force, 3),
        peak_net_group_force=round(r.peak_net_group_force, 3),
        peak_thumb_single_force=round(r.peak_thumb_single_force, 3),
        hand_hand_peak_force=round(r.hand_hand_peak_force, 3),
        hand_hand_streak=r.hand_hand_streak,
        thumb_thumb_collision=r.thumb_thumb_collision,
        safety_events=r.safety_events,
        force_recrossing_count=r.force_recrossing_count,
        thumb_first_contact_obj_xy=r.thumb_first_contact_obj_xy,
        thumb_oppose_to_tripod_push_distance=(
            round(r.thumb_oppose_to_tripod_push_distance, 5) if r.thumb_oppose_to_tripod_push_distance is not None else None
        ),
        min_thumb1_joint_margin=round(r.min_thumb1_joint_margin, 4) if np.isfinite(r.min_thumb1_joint_margin) else None,
        state_entry_step=dict(r.state_entry_step),
        gate_a_pass=r.gate_a_pass,
    )
