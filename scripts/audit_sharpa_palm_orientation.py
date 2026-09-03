"""Re-audit of the Sharpa palm/closing axis, prompted by a real
discrepancy: numeric Side-Grasp Posture Gate PASS (inward ~14deg,
computed from palm_R column 1) vs a direct GUI observation that the
palm/finger-closure direction actually looks reversed.

ROOT CAUSE FOUND: the Gate's own "inward_angle" was measuring the SAME
axis the wrist was SOLVED to align with _object_facing_R's y_col --
i.e. the check was CIRCULAR (palm_R col1 points at the object nearly by
construction; the 14deg residual is only solver/tracking noise, not
independent verification). This script instead measures the REAL
closing direction directly: hold wrist/arm fixed, apply an ACTUAL curl
delta to index/middle/wrap, and read the resulting fingertip
displacement -- the state is restored afterward so this measurement
never permanently perturbs the rollout it's called from.

Measured at the WRIST_SIDE_GRASP_ALIGN end pose (seed=0, the exact pose
the Side-Grasp Posture Gate is evaluated at) -- NOT stand pose, which
was found (this session) to give a misleadingly different, and still
wrong, relationship due to configuration-dependent (nonlinear, redundant
17-DOF) kinematics.

Read-only: never modifies SharpaBimanualGraspExpert or SharpaGraspEnv.

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/audit_sharpa_palm_orientation.py
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


def ang_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def empirical_closing_axis(env: SharpaGraspEnv, side: str, curl_amount: float) -> np.ndarray:
    """State-preserving: applies curl, measures net NONTHUMB fingertip
    displacement, then restores qpos/qvel/ctrl exactly -- callable mid-
    rollout without side effects."""
    side_idx = 0 if side == "left" else 1
    tips0 = {f: env.fingertip_pos(side, f).copy() for f in sc.FINGERS}
    saved = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy())
    steps = max(1, int(np.ceil(curl_amount / env.config.hand_synergy_action_scale)))
    for _ in range(steps):
        a = np.zeros(ACTION_DIM)
        for g in (1, 2, 3):  # index, middle, wrap -- not thumb
            a[17 + side_idx * 4 + g] = 1.0
        env.step(a)
    tips1 = {f: env.fingertip_pos(side, f).copy() for f in sc.FINGERS}
    env.data.qpos[:], env.data.qvel[:], env.data.ctrl[:] = saved
    mujoco.mj_forward(env.model, env.data)
    disp = [tips1[f] - tips0[f] for f in NONTHUMB]
    net = np.mean(disp, axis=0)
    return net / np.linalg.norm(net)


def main() -> None:
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    for _ in range(1500):
        if expert.state == BimanualGraspState.FIVE_FINGER_PRESHAPE:
            break
        action = expert.step()
        env.step(action)
    print(f"Measured at state={expert.state.name} (WRIST_SIDE_GRASP_ALIGN's converged, "
          f"Side-Grasp-Posture-Gate-evaluated pose), group_synergy={env._group_synergy.round(3)}")
    obj_pos = expert._object_pos()
    print(f"object pos = {obj_pos.round(4)}")

    for side in SIDES:
        print(f"\n{'='*20} SIDE={side} {'='*20}")
        palm_pos, palm_R = env.palm_pose(side)
        obj_dir = obj_pos - palm_pos
        obj_dir /= np.linalg.norm(obj_dir)

        print(f"  palm_pos={palm_pos.round(4)}")
        print(f"  EXISTING palm_R col0(approach)={palm_R[:,0].round(3)} "
              f"col1(assumed closing)={palm_R[:,1].round(3)} col2(lateral)={palm_R[:,2].round(3)}")
        print(f"  direction to object = {obj_dir.round(4)}")
        print(f"  angle(EXISTING col1, to-object) = {ang_deg(palm_R[:,1], obj_dir):.2f}deg  "
              f"<- this is what the Side-Grasp Posture Gate reported as 'inward_angle' (~14deg) -- CIRCULAR,"
              f" since the wrist was SOLVED to make col1 point here.")

        for curl in (0.0, 0.2, 0.4, 0.6):  # 0.0 = fully reset to open before measuring
            if curl == 0.0:
                side_idx = 0 if side == "left" else 1
                for g in (1, 2, 3):
                    env._group_synergy[side_idx * 4 + g] = 0.0
                    env.data.ctrl[env._group_act_ids[side][g]] = env._group_open[side][g]
                mujoco.mj_forward(env.model, env.data)
                emp = empirical_closing_axis(env, side, 0.1)
                label = "curl_start=0, delta=0.1"
            else:
                emp = empirical_closing_axis(env, side, curl)
                label = f"curl_delta={curl}"
            ang = ang_deg(emp, obj_dir)
            ang_vs_col1 = ang_deg(emp, palm_R[:, 1])
            print(f"  [{label}] REAL empirical closing axis = {emp.round(4)}  "
                  f"angle_to_object={ang:.2f}deg  angle_vs_assumed_col1={ang_vs_col1:.2f}deg")

    print("\n" + "=" * 60)
    print("VERDICT: if angle_to_object for the REAL empirical closing axis is")
    print(">> 60-90deg (not ~14deg), the Side-Grasp Posture Gate's inward_angle")
    print("metric is INVALID (circular) and the previous PASS must be RETRACTED.")


if __name__ == "__main__":
    main()
