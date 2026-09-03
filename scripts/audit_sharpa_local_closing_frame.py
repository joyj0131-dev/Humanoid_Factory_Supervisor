"""[This session] Determines the REAL closing axis of the Sharpa hand
expressed in the wrist site's OWN local (body-fixed) frame -- the
quantity the previous session's audit (scripts/audit_sharpa_palm_orientation.py)
proved was NOT palm_R column 1 (that assumption was circular).

Key idea: `env.palm_pose(side)` returns (pos, R) where R = site_xmat, the
ACTUAL world-frame rotation of the rigid wrist_yaw_link-attached site.
Any vector rigidly attached to the hand mesh (e.g. "the direction
fingertips move when curled") transforms as world_vec = R @ local_vec,
so local_vec = R.T @ world_vec is a MESH PROPERTY -- invariant to
whatever orientation the arm/IK happens to be holding the wrist in.

This script measures local_vec = R.T @ empirical_closing_world_axis at
TWO structurally different poses (FOREARM_FORWARD_REACH's end pose, and
WRIST_SIDE_GRASP_ALIGN's converged pose) for both hands, and checks that
the LOCAL vector is consistent across pose and across curl-delta
magnitude -- if so, that local vector (not palm_R column 1) is the
correct, pose-invariant definition of "closing axis in the hand's own
frame" to build _object_facing_R from.

Read-only: never modifies SharpaBimanualGraspExpert or SharpaGraspEnv
source. Reuses the state-preserving empirical_closing_axis helper from
scripts/audit_sharpa_palm_orientation.py (both files intentionally
share the identical measurement recipe).

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/audit_sharpa_local_closing_frame.py
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
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SIDES, BimanualGraspState, SharpaBimanualGraspExpert

NONTHUMB = ("index", "middle", "ring", "pinky")


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=3000, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


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
    disp = [tips1[f] - tips0[f] for f in NONTHUMB]
    net = np.mean(disp, axis=0)
    return net / np.linalg.norm(net)


def ang_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def measure_local_closing(env: SharpaGraspEnv, side: str, curl_deltas=(0.05, 0.10, 0.20),
                           reset_to_open: bool = False) -> list[np.ndarray]:
    """[Fixed, see INVARIANCE CHECK below] Measuring the closing direction
    from a base pose that is ALREADY substantially curled (e.g. WRIST_
    SIDE_GRASP_ALIGN's own 0.5 protective preshape) confounds the
    "early-closure" direction with later, nonlinear curled-arc motion
    (Section 6 of this session's spec warns about exactly this). When
    reset_to_open=True, resets this side's index/middle/wrap synergy to
    0.0 (fully open) FIRST, so curl_deltas measure genuine early-closure
    from a known, comparable reference regardless of which FSM state the
    hand happened to be in when this function was called."""
    if reset_to_open:
        side_idx = 0 if side == "left" else 1
        for g in (1, 2, 3):
            env._group_synergy[side_idx * 4 + g] = 0.0
            env.data.ctrl[env._group_act_ids[side][g]] = env._group_open[side][g]
        mujoco.mj_forward(env.model, env.data)
    palm_pos, palm_R = env.palm_pose(side)
    locals_ = []
    for d in curl_deltas:
        world_axis = empirical_closing_axis_world(env, side, d)
        local_axis = palm_R.T @ world_axis
        local_axis /= np.linalg.norm(local_axis)
        locals_.append(local_axis)
    return locals_


def main() -> None:
    results = {}

    # Pose 1: FOREARM_FORWARD_REACH end pose (before the big reorientation).
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    for _ in range(1500):
        if expert.state == BimanualGraspState.WRIST_SIDE_GRASP_ALIGN:
            break
        action = expert.step()
        env.step(action)
    print(f"=== POSE 1: state={expert.state.name} (FOREARM_FORWARD_REACH end pose) ===")
    for side in SIDES:
        locs = measure_local_closing(env, side, reset_to_open=True)
        results[("pose1", side)] = locs
        for d, v in zip((0.05, 0.10, 0.20), locs):
            print(f"  side={side} curl_delta={d}: local_closing_vec={v.round(4)}")

    # Pose 2: WRIST_SIDE_GRASP_ALIGN converged pose (the Gate-evaluated pose).
    env2 = make_env()
    expert2 = SharpaBimanualGraspExpert(env2)
    env2.reset(seed=0)
    for _ in range(1500):
        if expert2.state == BimanualGraspState.FIVE_FINGER_PRESHAPE:
            break
        action = expert2.step()
        env2.step(action)
    print(f"\n=== POSE 2: state={expert2.state.name} (WRIST_SIDE_GRASP_ALIGN converged pose) ===")
    for side in SIDES:
        locs = measure_local_closing(env2, side, reset_to_open=True)
        results[("pose2", side)] = locs
        for d, v in zip((0.05, 0.10, 0.20), locs):
            print(f"  side={side} curl_delta={d}: local_closing_vec={v.round(4)}")

    print("\n" + "=" * 70)
    print("INVARIANCE CHECK: local_closing_vec should be ~consistent across pose/curl/side-mirror")
    for side in SIDES:
        p1 = results[("pose1", side)][1]  # curl_delta=0.10 as reference
        p2 = results[("pose2", side)][1]
        print(f"  side={side}: pose1 vs pose2 angle = {ang_deg(p1, p2):.2f}deg")
    l_ref = results[("pose2", "left")][1]
    r_ref = results[("pose2", "right")][1]
    # mirror check: local frames are defined identically per side (same convention),
    # so the local closing vector itself should be near-IDENTICAL (not mirrored) if
    # both hands are built with mirrored meshes but identical LOCAL-frame convention.
    print(f"  left vs right (pose2, same local vector expected under convention): angle = {ang_deg(l_ref, r_ref):.2f}deg")
    print(f"  left local vec = {l_ref.round(4)}")
    print(f"  right local vec = {r_ref.round(4)}")
    print(f"  (for reference, nominal assumed closing axis was local e_y = [0,1,0])")

    # Recommended fixed local closing vector: average across pose2 measurements (the
    # Gate-evaluated pose), both curl deltas, both sides (if left/right agree).
    all_vecs = [results[("pose2", s)][i] for s in SIDES for i in range(3)]
    mean_vec = np.mean(all_vecs, axis=0)
    mean_vec /= np.linalg.norm(mean_vec)
    print(f"\nRECOMMENDED fixed local_closing_vec (mean, pose2, all curls/sides) = {mean_vec.round(4)}")
    for s in SIDES:
        for i, d in enumerate((0.05, 0.10, 0.20)):
            v = results[("pose2", s)][i]
            print(f"  residual vs mean: side={s} curl={d}: {ang_deg(v, mean_vec):.2f}deg")


if __name__ == "__main__":
    main()
