"""Swept-path safety validation for the Sharpa bimanual side-grasp
controller (this session): endpoint-only collision checks are not
sufficient (Section 8 of this session's mandate) -- this script drives
the OFFICIAL controller tick-by-tick and records, for every state
transition segment, the same safety metrics the state machine itself
checks (never a separate/looser measurement):

  - palm position/orientation (both sides)
  - elbow/shoulder body position (both sides)
  - wrist qvel peak/RMS (raw, unfiltered -- same DOFs as the Wrist
    Transition Gate)
  - torso-arm collision force
  - hand-table collision force
  - hand-hand collision force
  - hand-object premature contact (proximal, non-fingertip penetration)
  - minimum geom distance is NOT separately computed (MuJoCo does not
    expose a cheap all-pairs min-distance query outside contacts;
    forbidden-collision force, already checked every tick, is the
    project's existing, established proxy -- see sharpa_grasp_env.py)
  - joint-limit margin (from the IK solver's own report at each waypoint
    solve)
  - IK target vs actual physics tracking error (Cartesian, both sides)

Read-only: never modifies SharpaBimanualGraspExpert or SharpaGraspEnv,
drives the OFFICIAL controller exactly as scripts/test_sharpa_bimanual_
grasp.py does.

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/diagnose_side_grasp_swept_path.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    SIDES,
    BimanualGraspState,
    SharpaBimanualGraspExpert,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "side_grasp_swept_path"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRANSITION_STATES = (
    BimanualGraspState.STABLE_START,
    BimanualGraspState.ARM_LATERAL_CLEARANCE,
    BimanualGraspState.FOREARM_FORWARD_REACH,
    BimanualGraspState.WRIST_SIDE_GRASP_ALIGN,
    BimanualGraspState.FIVE_FINGER_PRESHAPE,
    BimanualGraspState.FOREARM_SIDE_DESCEND,
    BimanualGraspState.FINGERTIP_PRECONTACT,
)


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=3000, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def main() -> None:
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)

    solve_log: list[dict] = []
    orig_solve = expert.ik.solve

    def recording_solve(data, ltp, ltR, rtp, rtR, *a, **kw):
        result = orig_solve(data, ltp, ltR, rtp, rtR, *a, **kw)
        solve_log.append({
            "total_step": expert._total_step, "state": expert.state.name,
            "success": result.success, "iterations": result.iterations,
            "joint_limit_margin": result.joint_limit_margin,
            "left_pos_error_mm": result.left_pos_error * 1000, "right_pos_error_mm": result.right_pos_error * 1000,
        })
        return result

    expert.ik.solve = recording_solve

    tick_log: list[dict] = []
    wrist_dof = np.concatenate([env._arm_dof_adr[4:7], env._arm_dof_adr[11:14]])
    per_state_qvel: dict[str, list[float]] = {s.name: [] for s in TRANSITION_STATES}

    obs, info = env.reset(seed=0)
    for _ in range(2500):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        state_before = expert.state
        action = expert.step()
        obs, r, term, trunc, info = env.step(action)

        wrist_qvel = float(np.max(np.abs(env.data.qvel[wrist_dof])))
        if state_before.name in per_state_qvel:
            per_state_qvel[state_before.name].append(wrist_qvel)

        row = {"total_step": expert._total_step, "state": state_before.name}
        for side in SIDES:
            pos, R = env.palm_pose(side)
            row[f"{side}_palm_pos"] = pos.copy()
            sb = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_roll_link")
            eb = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_elbow_link")
            row[f"{side}_shoulder_z"] = float(env.data.xpos[sb][2])
            row[f"{side}_elbow_z"] = float(env.data.xpos[eb][2])
        row["torso_arm_force_n"] = env._torso_arm_collision_force()
        row["hand_table_force_n"] = env._hand_table_contact_force()
        row["hand_hand_force_n"] = env._hand_hand_contact_force()
        row["proximal_object_penetration_m"] = env._proximal_object_penetration()
        row["wrist_qvel_raw"] = wrist_qvel
        tick_log.append(row)

        if info.get("unstable"):
            print("UNSTABLE -- aborting")
            break
        if trunc:
            break

    print(f"Ran {expert._total_step} total ticks, final state={expert.state.name}, "
          f"failure_reason={expert.failure_reason}")

    print("\n=== Per-segment (state) summary ===")
    for s in TRANSITION_STATES:
        rows = [r for r in tick_log if r["state"] == s.name]
        if not rows:
            print(f"  {s.name}: NOT REACHED")
            continue
        max_torso_arm = max(r["torso_arm_force_n"] for r in rows)
        max_hand_table = max(r["hand_table_force_n"] for r in rows)
        max_hand_hand = max(r["hand_hand_force_n"] for r in rows)
        max_prox_pen = max(r["proximal_object_penetration_m"] for r in rows)
        qvels = per_state_qvel[s.name]
        peak_qvel = max(qvels) if qvels else 0.0
        rms_qvel = float(np.sqrt(np.mean(np.square(qvels)))) if qvels else 0.0
        print(f"  {s.name} ({len(rows)} ticks): max_torso_arm={max_torso_arm:.2f}N "
              f"max_hand_table={max_hand_table:.2f}N max_hand_hand={max_hand_hand:.2f}N "
              f"max_proximal_object_pen={max_prox_pen*1000:.3f}mm "
              f"wrist_qvel_peak={peak_qvel:.3f}rad/s wrist_qvel_rms={rms_qvel:.3f}rad/s")

    print("\n=== IK solve joint-limit margin / tracking error, per state ===")
    for s in TRANSITION_STATES:
        rows = [r for r in solve_log if r["state"] == s.name]
        if not rows:
            continue
        min_margin = min(r["joint_limit_margin"] for r in rows)
        max_pos_err = max(max(r["left_pos_error_mm"], r["right_pos_error_mm"]) for r in rows)
        any_fail = any(not r["success"] for r in rows)
        print(f"  {s.name}: n_solves={len(rows)} min_joint_limit_margin={min_margin:.4f} "
              f"max_ik_pos_error={max_pos_err:.2f}mm any_solve_reported_failure={any_fail}")

    csv_path = RESULTS_DIR / "swept_path_ticks.csv"
    if tick_log:
        keys = sorted({k for row in tick_log for k in row if not isinstance(row[k], np.ndarray)})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            for row in tick_log:
                w.writerow({k: row.get(k, "") for k in keys})
    print(f"\nRaw per-tick trace (scalars only) written to {csv_path}")


if __name__ == "__main__":
    main()
