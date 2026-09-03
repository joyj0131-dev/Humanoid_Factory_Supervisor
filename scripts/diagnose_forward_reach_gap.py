"""Session diagnostic: decomposes the FOREARM_FORWARD_REACH ~10.165mm
settled palm error into layers, for BOTH hands, at every control tick from
the final (6th) waypoint solve onward, using the EXACT official config
(max_steps_per_state=400, FORWARD_REACH_WAYPOINTS=6 -- unmodified).

Never modifies SharpaBimanualGraspExpert or SharpaGraspEnv -- drives the
OFFICIAL controller exactly as scripts/test_sharpa_bimanual_grasp.py does
and only OBSERVES already-public attributes plus a monkeypatched wrapper
around CoupledBilateralIK.solve (records call args/results, delegates to
the real implementation unchanged).

Layers recorded per tick (from the 6th/final waypoint solve onward):
  L0 Cartesian target        -- self._forward_reach_final[side] (fixed
                                 after the final waypoint fires)
  L1 IK solved FK pose       -- solve()'s own reported pos_error (scratch,
                                 immediately after solve -- "IK success")
  L2 solved joint target     -- result.left_q/right_q/waist_q, i.e. what
                                 _apply_ik_result() writes into
                                 expert._arm_ik_target/_waist_ik_target
  L3 ctrl register           -- env._arm_target/_waist_target, the
                                 rate-limited register actually written to
                                 data.ctrl each tick
  L4 ctrl-implied palm pose  -- FK with a scratch qpos's arm/waist DOFs
                                 set to the CURRENT ctrl register (L3)
  L5 actual physics palm pose -- env.palm_pose(side) after mj_step

After the OFFICIAL controller declares FORWARD_REACH_NOT_ACHIEVED (tick
400), this script does NOT stop -- it keeps stepping the env with a
ZERO action (the controller's own state machine no longer issues arm/
waist commands once in FAILURE, so the ctrl register/IK target are held
exactly at whatever they were at the moment of failure; gravity-
compensation feedforward keeps recomputing from the live qfrc_bias every
tick, same as it would have inside the state). This isolates timing
(H3: would more ticks at the SAME final target have converged under
10mm?) from a genuine steady-state actuator-tracking plateau (H2), WITHOUT
altering FORWARD_REACH_WAYPOINTS/max_steps_per_state and therefore without
changing the waypoint schedule that produced the reported 10.165mm.

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/diagnose_forward_reach_gap.py
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
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv, ACTION_DIM
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    SIDES,
    BimanualGraspState,
    SharpaBimanualGraspExpert,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "forward_reach_gap"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EXTRA_SETTLE_TICKS = 600  # diagnostic-only continuation past official FAILURE


def make_env(max_episode_steps: int = 2500) -> SharpaGraspEnv:
    config = GraspEnvConfig(
        object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
        max_episode_steps=max_episode_steps, arm_gravity_compensation=True,
    )
    return SharpaGraspEnv(config)


def ctrl_implied_palm_pose(env, waist_target, arm_target, scratch: mujoco.MjData):
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
    expert = SharpaBimanualGraspExpert(env)  # official default BimanualGraspConfig, untouched
    scratch = mujoco.MjData(env.model)

    solve_log: list[dict] = []
    tick_log: list[dict] = []

    orig_solve = expert.ik.solve

    def recording_solve(data, left_target_pos, left_target_R, right_target_pos, right_target_R, *a, **kw):
        result = orig_solve(data, left_target_pos, left_target_R, right_target_pos, right_target_R, *a, **kw)
        solve_log.append({
            "total_step": expert._total_step,
            "waypoint": getattr(expert, "_forward_reach_waypoint", -1),
            "success": result.success,
            "iterations": result.iterations,
            "left_pos_error": result.left_pos_error,
            "right_pos_error": result.right_pos_error,
            "joint_limit_margin": result.joint_limit_margin,
        })
        return result

    expert.ik.solve = recording_solve

    obs, info = env.reset(seed=0)
    reached_final_wp_at = None
    official_failure_at = None
    zero_action = np.zeros(ACTION_DIM)

    def log_tick():
        final_target = getattr(expert, "_forward_reach_final", None)
        if final_target is None:
            return
        implied = ctrl_implied_palm_pose(env, env._waist_target, env._arm_target, scratch)
        row = {"total_step": expert._total_step, "waypoint": getattr(expert, "_forward_reach_waypoint", -1),
               "post_official_failure": official_failure_at is not None}
        for side in SIDES:
            actual_pos, _ = env.palm_pose(side)
            implied_pos, _ = implied[side]
            ik_joint_target = expert._arm_ik_target[:7] if side == "left" else expert._arm_ik_target[7:]
            actual_arm_q = (env.data.qpos[env._arm_qpos_adr[:7]] if side == "left"
                            else env.data.qpos[env._arm_qpos_adr[7:]])
            ctrl_arm_q = (env._arm_target[:7] if side == "left" else env._arm_target[7:])
            arm_qvel = env.data.qvel[env._arm_dof_adr[:7] if side == "left" else env._arm_dof_adr[7:]]
            row[f"{side}_actual_vs_final_target_m"] = float(np.linalg.norm(actual_pos - final_target[side]))
            row[f"{side}_ctrl_implied_vs_final_target_m"] = float(np.linalg.norm(implied_pos - final_target[side]))
            row[f"{side}_L4_L5_gap_m"] = float(np.linalg.norm(implied_pos - actual_pos))
            row[f"{side}_joint_target_minus_ctrl_norm"] = float(np.linalg.norm(ik_joint_target - ctrl_arm_q))
            row[f"{side}_ctrl_minus_actual_q_norm"] = float(np.linalg.norm(ctrl_arm_q - actual_arm_q))
            row[f"{side}_arm_qvel_norm"] = float(np.linalg.norm(arm_qvel))
        tick_log.append(row)

    # ---- Phase A: run the OFFICIAL controller to FAILURE/SUCCESS ----
    for _ in range(1200):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        state_before = expert.state
        wp_before = getattr(expert, "_forward_reach_waypoint", -1)
        action = expert.step()
        wp_after = getattr(expert, "_forward_reach_waypoint", -1)
        if state_before == BimanualGraspState.FOREARM_FORWARD_REACH and wp_before < 6 and wp_after >= 6 and reached_final_wp_at is None:
            reached_final_wp_at = expert._total_step
        obs, r, term, trunc, info = env.step(action)
        if state_before == BimanualGraspState.FOREARM_FORWARD_REACH:
            log_tick()
        if info.get("unstable"):
            print("UNSTABLE -- aborting")
            break
        if trunc:
            break

    print(f"Official controller stopped at total_step={expert._total_step}, state={expert.state.name}, "
          f"failure_reason={expert.failure_reason}")
    print(f"FOREARM_FORWARD_REACH final (6th) waypoint solved at total_step={reached_final_wp_at}")

    # ---- Phase B: diagnostic-only continuation past official FAILURE,
    # holding the SAME ctrl target the controller had at the moment of
    # failure (state machine no longer issues commands once in FAILURE,
    # so action stays zero; gravity-compensation feedforward keeps
    # recomputing from the live qfrc_bias every tick). ----
    if expert.state == BimanualGraspState.FAILURE and expert.failure_reason is not None:
        official_failure_at = expert._total_step
        for _ in range(EXTRA_SETTLE_TICKS):
            obs, r, term, trunc, info = env.step(zero_action)
            expert._total_step += 1
            log_tick()
            if info.get("unstable") or trunc:
                break

    print(f"\n=== Per-solve summary (every _solve_both call during FOREARM_FORWARD_REACH) ===")
    for s in solve_log:
        print(f"  step={s['total_step']:5d} wp={s['waypoint']} success={s['success']} iters={s['iterations']:3d} "
              f"L1_pos_err(l,r)=({s['left_pos_error']*1000:.3f}mm,{s['right_pos_error']*1000:.3f}mm) "
              f"joint_limit_margin={s['joint_limit_margin']:.4f}")

    if tick_log:
        last = tick_log[-1]
        print("\nLast tick on record (official + diagnostic continuation):")
        for side in SIDES:
            print(f"  {side}: actual_vs_final_target={last[f'{side}_actual_vs_final_target_m']*1000:.3f}mm  "
                  f"ctrl_implied_vs_final_target={last[f'{side}_ctrl_implied_vs_final_target_m']*1000:.3f}mm  "
                  f"L4_L5(ctrl_implied vs actual)_gap={last[f'{side}_L4_L5_gap_m']*1000:.3f}mm  "
                  f"joint_target_minus_ctrl_norm={last[f'{side}_joint_target_minus_ctrl_norm']:.6f}rad  "
                  f"ctrl_minus_actual_q_norm={last[f'{side}_ctrl_minus_actual_q_norm']:.6f}rad  "
                  f"arm_qvel_norm={last[f'{side}_arm_qvel_norm']:.6f}rad/s")

        post_final = [r for r in tick_log if r["waypoint"] >= 6]
        print(f"\nTicks on record from final(6th) waypoint solve onward: {len(post_final)} "
              f"(official failure at total_step={official_failure_at})")
        print("actual_vs_final_target_m trajectory (every 10th tick, both sides, plus flags):")
        for i, r in enumerate(post_final):
            if i % 10 == 0 or i == len(post_final) - 1:
                l = r["left_actual_vs_final_target_m"] * 1000
                rr = r["right_actual_vs_final_target_m"] * 1000
                lv = r["left_arm_qvel_norm"]
                rv = r["right_arm_qvel_norm"]
                flag = " [POST-FAILURE]" if r["post_official_failure"] else ""
                print(f"    tick {i:4d} (total_step={r['total_step']}): left={l:.3f}mm(qvel={lv:.4f}) "
                      f"right={rr:.3f}mm(qvel={rv:.4f}){flag}")

        # explicit answer to H3 vs H2: value AT official failure vs best
        # value reached during the diagnostic-only continuation
        at_failure = [r for r in tick_log if not r["post_official_failure"] and r["waypoint"] >= 6]
        after_failure = [r for r in tick_log if r["post_official_failure"]]
        if at_failure:
            f = at_failure[-1]
            print(f"\nAt OFFICIAL failure (total_step={f['total_step']}): "
                  f"left={f['left_actual_vs_final_target_m']*1000:.3f}mm right={f['right_actual_vs_final_target_m']*1000:.3f}mm")
        if after_failure:
            min_left = min(r["left_actual_vs_final_target_m"] for r in after_failure) * 1000
            min_right = min(r["right_actual_vs_final_target_m"] for r in after_failure) * 1000
            last_left = after_failure[-1]["left_actual_vs_final_target_m"] * 1000
            last_right = after_failure[-1]["right_actual_vs_final_target_m"] * 1000
            print(f"During {EXTRA_SETTLE_TICKS}-tick diagnostic continuation (ctrl target held fixed):")
            print(f"  min error reached: left={min_left:.3f}mm right={min_right:.3f}mm")
            print(f"  final error at continuation end: left={last_left:.3f}mm right={last_right:.3f}mm")

    csv_path = RESULTS_DIR / "forward_reach_gap_ticks.csv"
    if tick_log:
        keys = sorted({k for row in tick_log for k in row})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            for row in tick_log:
                w.writerow(row)
    print(f"\nRaw per-tick trace written to {csv_path}")


if __name__ == "__main__":
    main()
