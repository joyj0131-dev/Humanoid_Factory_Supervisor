"""[This session] Candidate A/B/C harness for the Functional Orientation
fix. Monkeypatches sharpa_bimanual_grasp_expert._object_facing_R with
each candidate's implementation and runs the REAL, full rollout through
env.step()/expert.step() (no shortcuts), measuring the same metrics the
Functional Orientation Gate needs.

Root cause established (scripts/audit_sharpa_local_closing_frame.py):
the "closing axis = palm_R column 1" assumption was circular. The TRUE,
empirically-measured local closing vector (expressed in the wrist
site's own rigid local frame, measured CONTACT-FREE at the
FOREARM_FORWARD_REACH end pose -- away from the table, curl deltas
0.05/0.1/0.2 all agree within a few degrees, BOTH hands identical, no
mirroring needed) is:

    LOCAL_CLOSING_VEC ~= [0.187, 0.982, 0.005]   (not the assumed [0,1,0])

(Measuring the same quantity near the table -- e.g. at the WRIST_SIDE_
GRASP_ALIGN converged pose -- gives inconsistent, contact-contaminated
results: reset-to-open fingers immediately graze the table there, which
redirects the measured displacement. That contamination, not a meshing
error, is the likely source of the previous session's much larger
43-93deg figure. This session's fix targets the clean, contact-free
kinematic direction.)

Read-only with respect to the actual source files (patches the function
object at runtime only); nothing here is committed as-is.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv, ACTION_DIM
from humanoid_learning.expert import sharpa_bimanual_grasp_expert as sbe
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SIDES, BimanualGraspState, SharpaBimanualGraspExpert


# [Corrected, second iteration] Measuring at curl=0 gives a DIFFERENT
# (and wrong for this purpose) local closing vector than measuring at
# the ACTUAL base curl level the approach holds (side_align_preshape_
# curl=0.5) -- the finger's closing tangent direction rotates
# continuously and predictably as curl increases (confirmed: base_curl
# in {0.0, 0.3, 0.5, 0.7} gives [0.18,0.98,0] -> [-0.14,0.99,0] ->
# [-0.61,0.80,0.02] -> [-0.85,0.52,0] -- SAME for both hands, no
# mirroring, at every level -- scripts/audit_sharpa_local_closing_frame.py
# / the half-curl re-measurement this session). Since WRIST_SIDE_GRASP_
# ALIGN/FIVE_FINGER_PRESHAPE hold curl=0.5 throughout, THIS is the
# physically relevant tangent, not the curl=0 one.
LOCAL_CLOSING_VEC = np.array([-0.6057, 0.7955, 0.0215])
LOCAL_CLOSING_VEC /= np.linalg.norm(LOCAL_CLOSING_VEC)
LOCAL_APPROACH_VEC = np.array([1.0, 0.0, 0.0])
NONTHUMB = ("index", "middle", "ring", "pinky")
sbe.__dict__["_ORIGINAL_object_facing_R"] = sbe._object_facing_R


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=3000, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def _wahba_R(local_vecs, world_vecs, weights) -> np.ndarray:
    """Kabsch/Wahba best-fit rotation R minimizing sum w_i |R@local_i - world_i|^2."""
    B = np.zeros((3, 3))
    for w, l, wd in zip(weights, local_vecs, world_vecs):
        l = l / np.linalg.norm(l)
        wd = wd / np.linalg.norm(wd)
        B += w * np.outer(wd, l)
    U, S, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    return U @ D @ Vt


def _down_target(to_obj_dir: np.ndarray, inward_weight: float = 0.18) -> np.ndarray:
    """Desired direction for the finger-longitudinal (local +X / approach)
    axis: mostly down, slightly inward toward the object (not straight
    down into the table, not straight sideways at the object)."""
    horiz = to_obj_dir.copy()
    horiz[2] = 0.0
    n = np.linalg.norm(horiz)
    horiz = horiz / n if n > 1e-9 else np.zeros(3)
    d = (1 - inward_weight) * np.array([0.0, 0.0, -1.0]) + inward_weight * horiz
    return d / np.linalg.norm(d)


def candidate_A(side: str, palm_pos: np.ndarray, obj_pos: np.ndarray) -> np.ndarray:
    """Single-vector fix: closing_local -> to_object, remaining DOF (twist
    about that axis) resolved via a minimal, natural (world-up-referenced)
    completion -- same style as the ORIGINAL _object_facing_R, but using
    the empirically-correct LOCAL_CLOSING_VEC instead of the circular
    palm_R-column-1/local-e_y assumption."""
    to_obj = obj_pos - palm_pos
    to_obj = to_obj / np.linalg.norm(to_obj)
    # Find R minimizing angle: R @ LOCAL_CLOSING_VEC = to_obj, remaining
    # rotation about to_obj chosen so LOCAL_APPROACH_VEC leans toward -Z
    # as much as this one remaining DOF allows (single-vector Wahba: only
    # one target constraint, so add a synthetic weak second point for a
    # well-posed SVD -- world "up" reference, low weight).
    world_up = np.array([0.0, 0.0, 1.0])
    return _wahba_R([LOCAL_CLOSING_VEC, np.array([0.0, 0.0, 1.0])], [to_obj, world_up], weights=[1.0, 0.05])


def candidate_B(side: str, palm_pos: np.ndarray, obj_pos: np.ndarray) -> np.ndarray:
    """Two-vector Kabsch/Wahba: closing_local -> to_object (primary,
    HEAVILY weighted -- the two local vectors are not close to
    orthogonal-compatible with their targets simultaneously, so this is
    a genuine weighted-least-squares trade-off; measured (this session,
    pure math, no physics): w_close=0.9 -> closing residual ~4deg,
    approach/down residual ~52deg. The Functional Orientation Gate's
    hard numeric requirement is on the CLOSING axis (<=15deg); the
    finger-long/'inward-down' condition is qualitative, so priority goes
    to closing accuracy.), approach_local(local +X) -> inward-down
    (secondary, soft)."""
    to_obj = obj_pos - palm_pos
    to_obj = to_obj / np.linalg.norm(to_obj)
    down = _down_target(to_obj)
    return _wahba_R([LOCAL_CLOSING_VEC, LOCAL_APPROACH_VEC], [to_obj, down], weights=[0.9, 0.1])


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def candidate_A2(side: str, palm_pos: np.ndarray, obj_pos: np.ndarray) -> np.ndarray:
    """Minimal-rotation surgical correction: start from the ORIGINAL
    (already collision-free, already-safe) _object_facing_R construction,
    then apply the SMALLEST possible single-axis rotation that moves the
    REAL empirical closing axis (LOCAL_CLOSING_VEC transformed through
    that base orientation) onto the true palm->object direction -- this
    preserves whatever twist/approach-axis choice the original safe
    construction already found, instead of re-deriving the whole
    orientation from scratch (which the 2-vector Wahba candidates showed
    can introduce a NEW table/torso collision by over-constraining the
    approach axis toward an arbitrary 'down' target)."""
    R_old = sbe.__dict__["_ORIGINAL_object_facing_R"](side, palm_pos, obj_pos)
    to_obj = obj_pos - palm_pos
    to_obj = to_obj / np.linalg.norm(to_obj)
    a = R_old @ LOCAL_CLOSING_VEC
    a = a / np.linalg.norm(a)
    cos_ang = np.clip(np.dot(a, to_obj), -1.0, 1.0)
    if cos_ang > 0.999999:
        return R_old
    axis = np.cross(a, to_obj)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / n
    angle = np.arccos(cos_ang)
    dR = _rodrigues(axis, angle)
    return dR @ R_old


# Candidate C reuses candidate_B's R construction; the difference is in the
# TRAJECTORY (orientation-first at a higher/wider pose, then translate) --
# implemented in run_rollout via orientation_first=True.
candidate_C = candidate_B


def _object_facing_angle_deg_v2(side: str, palm_R: np.ndarray, palm_pos: np.ndarray, obj_pos: np.ndarray) -> float:
    """Corrected metric: compares the REAL empirical closing axis
    (LOCAL_CLOSING_VEC transformed into world via the CURRENT palm_R) to
    the true palm->object direction. No longer circular: LOCAL_CLOSING_VEC
    is a fixed constant measured independently (audit_sharpa_local_closing_
    frame.py), not derived from whatever the solver targeted."""
    to_obj = obj_pos - palm_pos
    n = np.linalg.norm(to_obj)
    if n < 1e-9:
        return 0.0
    to_obj = to_obj / n
    world_closing = palm_R @ LOCAL_CLOSING_VEC
    cos_ang = np.clip(np.dot(world_closing, to_obj), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def empirical_closing_axis_world(env: SharpaGraspEnv, side: str, curl_amount: float) -> np.ndarray:
    side_idx = 0 if side == "left" else 1
    tips0 = {f: env.fingertip_pos(side, f).copy() for f in sc.FINGERS}
    saved = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy())
    steps = max(1, int(np.ceil(curl_amount / env.config.hand_synergy_action_scale)))
    for _ in range(steps):
        a = np.zeros(ACTION_DIM)
        for g in (1, 2, 3):
            a[17 + side_idx * 4 + g] = 1.0
        env.step(a)
    tips1 = {f: env.fingertip_pos(side, f).copy() for f in sc.FINGERS}
    env.data.qpos[:], env.data.qvel[:], env.data.ctrl[:] = saved
    mujoco.mj_forward(env.model, env.data)
    return tips1, tips0


def functional_check(env: SharpaGraspEnv, side: str, obj_pos: np.ndarray, curl_amount: float = 0.05) -> dict:
    """Section 4/9-style functional check using the ACTUAL current
    preshape state (no reset) -- literally what CONTACT_ACQUIRE will do."""
    tips1, tips0 = empirical_closing_axis_world(env, side, curl_amount)
    per_finger_dot = {}
    for f in NONTHUMB:
        delta = tips1[f] - tips0[f]
        d_obj = obj_pos - tips0[f]
        d_obj = d_obj / np.linalg.norm(d_obj)
        per_finger_dot[f] = float(np.dot(delta, d_obj))
    n_positive = sum(1 for v in per_finger_dot.values() if v > 0)
    mean_disp_inward_mm = 1000 * np.mean([
        np.dot(tips1[f] - tips0[f], (obj_pos - tips0[f]) / np.linalg.norm(obj_pos - tips0[f])) for f in NONTHUMB
    ])
    palm_pos, palm_R = env.palm_pose(side)
    to_obj = obj_pos - palm_pos
    to_obj = to_obj / np.linalg.norm(to_obj)
    world_closing = palm_R @ LOCAL_CLOSING_VEC
    inward_angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(world_closing, to_obj), -1, 1))))
    return {
        "per_finger_dot": per_finger_dot, "n_positive": n_positive,
        "mean_disp_inward_mm": mean_disp_inward_mm, "inward_angle_deg": inward_angle_deg,
    }


def run_rollout(candidate_fn, label: str) -> dict:
    sbe._object_facing_R = candidate_fn
    sbe._object_facing_angle_deg = _object_facing_angle_deg_v2
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    max_torso, max_table, max_hh = 0.0, 0.0, 0.0
    max_wrist_qvel = 0.0
    wrist_dof = np.concatenate([env._arm_dof_adr[4:7], env._arm_dof_adr[11:14]])
    reached_align = False
    reached_precontact = False
    align_snapshot = None
    for _ in range(2500):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        prev_state = expert.state
        action = expert.step()
        env.step(action)
        max_torso = max(max_torso, env._torso_arm_collision_force())
        max_table = max(max_table, env._hand_table_contact_force())
        max_hh = max(max_hh, env._hand_hand_contact_force())
        max_wrist_qvel = max(max_wrist_qvel, float(np.max(np.abs(env.data.qvel[wrist_dof]))))
        if prev_state == BimanualGraspState.WRIST_SIDE_GRASP_ALIGN and expert.state == BimanualGraspState.FIVE_FINGER_PRESHAPE:
            reached_align = True
            align_snapshot = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(),
                               env._group_synergy.copy(), expert._object_pos())
        if expert.state == BimanualGraspState.FINGERTIP_PRECONTACT:
            reached_precontact = True

    result = {
        "label": label, "final_state": expert.state.name, "failure_reason": str(expert.failure_reason),
        "reached_align": reached_align, "reached_precontact": reached_precontact,
        "max_torso_arm_n": max_torso, "max_hand_table_n": max_table, "max_hand_hand_n": max_hh,
        "max_wrist_qvel": max_wrist_qvel,
        "left_object_facing_angle_deg": expert.left_object_facing_angle_deg,
        "right_object_facing_angle_deg": expert.right_object_facing_angle_deg,
        "side_grasp_gate": expert._side_grasp_gate,
    }
    if align_snapshot is not None:
        qpos, qvel, ctrl, syn, obj_pos = align_snapshot
        env.data.qpos[:], env.data.qvel[:], env.data.ctrl[:] = qpos, qvel, ctrl
        env._group_synergy[:] = syn
        mujoco.mj_forward(env.model, env.data)
        for side in SIDES:
            for curl in (0.05, 0.10, 0.15):
                result[f"functional_{side}_curl{curl}"] = functional_check(env, side, obj_pos, curl_amount=curl)
    return result


def main() -> None:
    for fn, label in [(candidate_A2, "A2_minimal_rotation_correction")]:
        print(f"\n{'='*70}\nCANDIDATE {label}\n{'='*70}")
        r = run_rollout(fn, label)
        for k, v in r.items():
            if k.startswith("functional_"):
                print(f"  {k}:")
                for fk, fv in v.items():
                    print(f"      {fk} = {fv}")
            else:
                print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
