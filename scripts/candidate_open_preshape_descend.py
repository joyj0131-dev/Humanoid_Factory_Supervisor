"""Real-physics candidate comparison for FOREARM_SIDE_DESCEND's target
(height, curl) pair -- replacing the curl=0.95 table-avoidance workaround
with a genuinely open/low-curl preshape at a higher clearance height.

Runs the ACTUAL controller end-to-end (real env.step() physics, no
monkeypatched physics, only BimanualGraspConfig field overrides) through
FINGERTIP_PRECONTACT/CONTACT_ACQUIRE and reports the metrics needed to pick
a safe candidate. Read-only w.r.t. the tracked source file.

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/candidate_open_preshape_descend.py
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

NONTHUMB = ("index", "middle", "ring", "pinky")


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=4000, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def fingertip_object_separation(env: SharpaGraspEnv, side: str, obj_pos: np.ndarray) -> float:
    tips = [env.fingertip_pos(side, f) for f in NONTHUMB]
    centroid = np.mean(tips, axis=0)
    half = env.config.object_half_size
    return float(abs(centroid[1] - obj_pos[1]) - half)


def run_candidate(name: str, height: float, curl: float, seed: int = 0, max_steps: int = 3000) -> dict:
    env = make_env()
    cfg = replace(BimanualGraspConfig(), side_descend_height_m=height, side_descend_curl_target=curl)
    expert = SharpaBimanualGraspExpert(env, cfg)
    env.reset(seed=seed)
    max_table_force = 0.0
    max_torso_force = 0.0
    descend_entered = False
    precontact_entered = False
    contact_entered = False
    for _ in range(max_steps):
        if expert.state == BimanualGraspState.FOREARM_SIDE_DESCEND:
            descend_entered = True
        if expert.state == BimanualGraspState.FINGERTIP_PRECONTACT:
            precontact_entered = True
        if expert.state == BimanualGraspState.CONTACT_ACQUIRE:
            contact_entered = True
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        action = expert.step()
        env.step(action)
        max_table_force = max(max_table_force, env._hand_table_contact_force())
        max_torso_force = max(max_torso_force, env._torso_arm_collision_force())

    obj_pos = expert._object_pos()
    sep = {s: fingertip_object_separation(env, s, obj_pos) * 1000 for s in SIDES}

    result = {
        "name": name, "height": height, "curl": curl,
        "final_state": expert.state.name, "failure_reason": str(expert.failure_reason),
        "max_table_force_n": max_table_force, "max_torso_force_n": max_torso_force,
        "descend_entered": descend_entered, "precontact_entered": precontact_entered,
        "contact_entered": contact_entered,
        "fingertip_object_separation_mm": sep,
        "functional_orientation_gate": expert._functional_orientation.get("gate") if expert._functional_orientation else None,
        "side_grasp_gate": expert._side_grasp_gate,
        "precontact_stable_streak": expert._max_precontact_stable_streak,
        "step_count": expert._total_step,
    }
    return result


def main() -> None:
    candidates = [
        ("A_baseline_curl095_h003", 0.03, 0.95),
        ("B_h007_curl05", 0.07, 0.50),
        ("C_h007_curl035", 0.07, 0.35),
        ("D_h005_curl05", 0.05, 0.50),
    ]
    results = []
    for name, h, c in candidates:
        print(f"\n{'='*20} candidate {name} (height={h}, curl={c}) {'='*20}")
        r = run_candidate(name, h, c)
        results.append(r)
        for k, v in r.items():
            print(f"  {k}: {v}")

    print(f"\n{'='*20} SUMMARY {'='*20}")
    for r in results:
        print(f"{r['name']}: final={r['final_state']}/{r['failure_reason']} "
              f"table_force_max={r['max_table_force_n']:.2f}N torso_force_max={r['max_torso_force_n']:.2f}N "
              f"descend={r['descend_entered']} precontact={r['precontact_entered']} contact={r['contact_entered']} "
              f"sep_mm={r['fingertip_object_separation_mm']}")


if __name__ == "__main__":
    main()
