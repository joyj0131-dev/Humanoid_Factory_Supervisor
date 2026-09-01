"""Stage 1 diagnostic (Session 39): decomposes the FINGERTIP_PRECONTACT
5-6cm IK-vs-physics gap into its exact layer, for BOTH hands, at every
control tick of FINGERTIP_PRECONTACT (plus the tail of WRIST_ALIGN /
FIVE_FINGER_PRESHAPE for context).

Never modifies SharpaBimanualGraspExpert or SharpaGraspEnv -- it drives
the OFFICIAL controller exactly as scripts/test_sharpa_bimanual_grasp.py
does and only OBSERVES already-public attributes plus a monkeypatched
wrapper around CoupledBilateralIK.solve (records call args/results,
delegates to the real implementation unchanged).

Pipeline layers recorded per solve (waypoint) and per tick:
  L0 Cartesian target        -- the (left,right) palm target position/R
                                 handed to CoupledBilateralIK.solve()
  L1 IK solved FK pose       -- solve()'s OWN reported left/right_pos_error
                                 (scratch data, immediately after solve --
                                 this is what "IK success" currently means)
  L2 solved joint target     -- result.waist_q/left_q/right_q, i.e. what
                                 _apply_ik_result() writes into
                                 self._arm_ik_target/_waist_ik_target
  L3 ctrl register           -- env._arm_target/_waist_target, the
                                 rate-limited (arm_action_scale-clipped)
                                 register actually written to data.ctrl
                                 each tick (_arm_action_toward_target())
  L4 ctrl-implied palm pose  -- FK computed by setting a scratch qpos's
                                 arm/waist DOFs to the CURRENT ctrl
                                 register (L3) and mj_forward -- "where the
                                 palm would be if the actuator had zero
                                 tracking error", isolating L3->L5 droop
                                 from L2->L3 rate-limit lag
  L5 actual physics palm pose -- env.palm_pose(side) after mj_step

The gap between L1 and L4 (mediated by whether L3 has caught up to L2 in
the ticks given) versus the gap between L4 and L5 (pure actuator
tracking) is the causal split this stage exists to make -- see
docs/history/PHASE4_GRASP_SESSION_39.md.

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/diagnose_precontact_gap.py
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

RESULTS_DIR = PROJECT_ROOT / "results" / "precontact_gap"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_env(max_episode_steps: int = 2500) -> SharpaGraspEnv:
    config = GraspEnvConfig(
        object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
        max_episode_steps=max_episode_steps, arm_gravity_compensation=True,
    )
    return SharpaGraspEnv(config)


def ctrl_implied_palm_pose(env, waist_target: np.ndarray, arm_target: np.ndarray, scratch: mujoco.MjData):
    """L4: FK with the arm/waist qpos set to the CTRL REGISTER (not the
    live physics qpos) -- "where the palm would be with zero actuator
    tracking error", holding everything else (object, fingers) at the
    live state."""
    scratch.qpos[:] = env.data.qpos
    scratch.qpos[env._waist_qpos_adr] = waist_target
    scratch.qpos[env._arm_qpos_adr] = arm_target
    mujoco.mj_forward(env.model, scratch)
    out = {}
    for side in SIDES:
        sid = env._left_palm_site if side == "left" else env._right_palm_site
        out[side] = (scratch.site_xpos[sid].copy(), scratch.site_xmat[sid].reshape(3, 3).copy())
    return out


def main() -> None:
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    scratch = mujoco.MjData(env.model)

    solve_log: list[dict] = []
    tick_log: list[dict] = []

    orig_solve = expert.ik.solve

    def recording_solve(data, left_target_pos, left_target_R, right_target_pos, right_target_R, *a, **kw):
        result = orig_solve(data, left_target_pos, left_target_R, right_target_pos, right_target_R, *a, **kw)
        solve_log.append({
            "total_step": expert._total_step,
            "state": expert.state.name,
            "waypoint": getattr(expert, "_precontact_waypoint", -1),
            "left_target": left_target_pos.copy(),
            "right_target": right_target_pos.copy(),
            "success": result.success,
            "iterations": result.iterations,
            "left_pos_error": result.left_pos_error,
            "right_pos_error": result.right_pos_error,
            "left_ori_error": result.left_ori_error,
            "right_ori_error": result.right_ori_error,
            "left_q": result.left_q.copy(),
            "right_q": result.right_q.copy(),
            "waist_q": result.waist_q.copy(),
            "joint_limit_margin": result.joint_limit_margin,
        })
        return result

    expert.ik.solve = recording_solve

    obs, info = env.reset(seed=0)
    max_total_steps = 2400
    entered_precontact_at = None
    left_final_target = right_final_target = None

    for _ in range(max_total_steps):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        state_before = expert.state
        action = expert.step()
        if state_before == BimanualGraspState.FINGERTIP_PRECONTACT and entered_precontact_at is None:
            entered_precontact_at = expert._total_step
            left_final_target = expert._precontact_final["left"].copy()
            right_final_target = expert._precontact_final["right"].copy()

        obs, r, term, trunc, info = env.step(action)

        if state_before in (BimanualGraspState.WRIST_ALIGN, BimanualGraspState.FIVE_FINGER_PRESHAPE,
                             BimanualGraspState.FINGERTIP_PRECONTACT, BimanualGraspState.CONTACT_ACQUIRE):
            implied = ctrl_implied_palm_pose(env, env._waist_target, env._arm_target, scratch)
            row = {
                "total_step": expert._total_step,
                "state": state_before.name,
                "waypoint": getattr(expert, "_precontact_waypoint", -1),
            }
            for side in SIDES:
                actual_pos, actual_R = env.palm_pose(side)
                implied_pos, implied_R = implied[side]
                ik_joint_target = expert._arm_ik_target[:7] if side == "left" else expert._arm_ik_target[7:]
                actual_arm_q = (env.data.qpos[env._arm_qpos_adr[:7]] if side == "left"
                                else env.data.qpos[env._arm_qpos_adr[7:]])
                ctrl_arm_q = (env._arm_target[:7] if side == "left" else env._arm_target[7:])
                final_target = left_final_target if side == "left" else right_final_target
                row[f"{side}_actual_pos"] = actual_pos
                row[f"{side}_ctrl_implied_pos"] = implied_pos
                row[f"{side}_joint_target_minus_ctrl_norm"] = float(np.linalg.norm(ik_joint_target - ctrl_arm_q))
                row[f"{side}_ctrl_minus_actual_q_norm"] = float(np.linalg.norm(ctrl_arm_q - actual_arm_q))
                row[f"{side}_L4_L5_gap_m"] = float(np.linalg.norm(implied_pos - actual_pos))
                if final_target is not None:
                    row[f"{side}_actual_vs_final_target_m"] = float(np.linalg.norm(actual_pos - final_target))
                    row[f"{side}_ctrl_implied_vs_final_target_m"] = float(np.linalg.norm(implied_pos - final_target))
            tick_log.append(row)

        if info.get("unstable"):
            print("UNSTABLE -- aborting")
            break
        if trunc:
            break

    print(f"Ran {expert._total_step} total ticks, final state={expert.state.name}, "
          f"failure_reason={expert.failure_reason}")
    print(f"FINGERTIP_PRECONTACT entered at total_step={entered_precontact_at}")

    # ---- Summary: state entry/exit + waypoint transitions + max error ----
    print("\n=== Per-solve summary (every _solve_both call) ===")
    for s in solve_log:
        print(f"  step={s['total_step']:5d} state={s['state']:22s} wp={s['waypoint']} "
              f"success={s['success']} iters={s['iterations']:3d} "
              f"L1_pos_err(l,r)=({s['left_pos_error']:.4f},{s['right_pos_error']:.4f}) "
              f"ori_err(l,r)=({s['left_ori_error']:.4f},{s['right_ori_error']:.4f})")

    if tick_log:
        precontact_rows = [r for r in tick_log if r["state"] == "FINGERTIP_PRECONTACT"]
        print(f"\n=== FINGERTIP_PRECONTACT tick count: {len(precontact_rows)} ===")
        if precontact_rows:
            last = precontact_rows[-1]
            print("Last FINGERTIP_PRECONTACT tick (entering CONTACT_ACQUIRE):")
            for side in SIDES:
                print(f"  {side}: actual_vs_final_target={last[f'{side}_actual_vs_final_target_m']*100:.2f}cm  "
                      f"ctrl_implied_vs_final_target={last[f'{side}_ctrl_implied_vs_final_target_m']*100:.2f}cm  "
                      f"L4_L5(ctrl_implied vs actual)_gap={last[f'{side}_L4_L5_gap_m']*100:.2f}cm  "
                      f"joint_target_minus_ctrl_norm={last[f'{side}_joint_target_minus_ctrl_norm']:.4f}rad  "
                      f"ctrl_minus_actual_q_norm={last[f'{side}_ctrl_minus_actual_q_norm']:.4f}rad")

            # per-waypoint transition snapshot: last tick of each waypoint
            wp_last = {}
            for r in precontact_rows:
                wp_last[r["waypoint"]] = r
            print("\nPer-waypoint end-of-waypoint snapshot:")
            for wp in sorted(wp_last):
                r = wp_last[wp]
                print(f"  waypoint={wp} step={r['total_step']}: "
                      f"left actual_vs_final={r['left_actual_vs_final_target_m']*100:.2f}cm "
                      f"L4_L5_gap={r['left_L4_L5_gap_m']*100:.2f}cm | "
                      f"right actual_vs_final={r['right_actual_vs_final_target_m']*100:.2f}cm "
                      f"L4_L5_gap={r['right_L4_L5_gap_m']*100:.2f}cm")

            last_wp = max(wp_last)
            last_wp_rows = [r for r in precontact_rows if r["waypoint"] == last_wp]
            print(f"\nIntra-waypoint {last_wp} L4_L5 gap trajectory (steady-state droop check, every 5th tick):")
            for i, r in enumerate(last_wp_rows):
                if i % 5 == 0 or i == len(last_wp_rows) - 1:
                    print(f"    tick {i:3d} (step={r['total_step']}): left_gap={r['left_L4_L5_gap_m']*100:.2f}cm "
                          f"right_gap={r['right_L4_L5_gap_m']*100:.2f}cm "
                          f"ctrl_minus_actual_q_norm_left={r['left_ctrl_minus_actual_q_norm']:.4f}rad")

            max_l4l5_left = max(r["left_L4_L5_gap_m"] for r in precontact_rows)
            max_l4l5_right = max(r["right_L4_L5_gap_m"] for r in precontact_rows)
            print(f"\nMax L4-L5 (ctrl-implied vs actual physics) gap during FINGERTIP_PRECONTACT: "
                  f"left={max_l4l5_left*100:.2f}cm right={max_l4l5_right*100:.2f}cm")

    # ---- raw CSV (not committed) ----
    csv_path = RESULTS_DIR / "precontact_gap_ticks.csv"
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
