"""Decompose FINGERTIP_PRECONTACT's tracking failure into Actuator Tracking
Failure (target fine, physics doesn't converge) vs Grasp Geometry Failure
(target itself unreachable/insufficient remaining travel), per this
session's directive Section 11. Runs candidate C's (height=0.07,
curl=0.35) real physics to FINGERTIP_PRECONTACT and prints the full error
chain at the final waypoint tail.

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/probe_precontact_tracking_gap.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    SIDES, BimanualGraspConfig, BimanualGraspState, SharpaBimanualGraspExpert,
)


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=4000, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def main() -> None:
    env = make_env()
    cfg = replace(BimanualGraspConfig(), side_descend_height_m=0.07, side_descend_curl_target=0.35)
    expert = SharpaBimanualGraspExpert(env, cfg)
    env.reset(seed=0)

    printed_waypoints = 0
    for _ in range(4000):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            print(f"terminal: {expert.state.name} {expert.failure_reason}")
            break
        prev_state = expert.state
        action = expert.step()
        env.step(action)
        if expert.state == BimanualGraspState.FINGERTIP_PRECONTACT and getattr(expert, "_precontact_waypoint", 0) >= expert.WAYPOINT_COUNT:
            if printed_waypoints < 5 or expert._state_step % 40 == 0:
                printed_waypoints += 1
                for side in SIDES:
                    actual, actual_R = env.palm_pose(side)
                    target = expert._precontact_final[side]
                    pos_err = float(np.linalg.norm(target - actual))
                    arm_target_q = expert._arm_ik_target[:7] if side == "left" else expert._arm_ik_target[7:]
                    arm_qpos_adr = env._arm_qpos_adr[:7] if side == "left" else env._arm_qpos_adr[7:]
                    actual_q = env.data.qpos[arm_qpos_adr]
                    joint_gap = float(np.linalg.norm(arm_target_q - actual_q))
                    ctrl_gap = float(np.linalg.norm(env._arm_target[(0 if side=='left' else 7):(7 if side=='left' else 14)] - arm_target_q))
                    print(f"  tick={expert._state_step} side={side} cartesian_target={target.round(4)} "
                          f"actual_palm={actual.round(4)} pos_err_mm={pos_err*1000:.2f} "
                          f"IKtarget_vs_actualQ_norm_rad={joint_gap:.4f} ctrl_vs_IKtarget_norm_rad={ctrl_gap:.4f}")
        if expert.state == BimanualGraspState.CONTACT_ACQUIRE and prev_state != BimanualGraspState.CONTACT_ACQUIRE:
            print("ENTERED CONTACT_ACQUIRE")


if __name__ == "__main__":
    main()
