"""Tripod Closure Planner session, Stage 1: runs
humanoid_learning.expert.tripod_closure_planner.audit_hand_geometry at
two points -- a fresh reset pose, and the canonical SIZE_12 rollout's
own CONTACT_ACQUIRE-end / THUMB_OPPOSE-entry pose (the real starting
point Stage 2/3 must plan a closure from) -- and prints a human-readable
report. Run with:

    python scripts/audit_fingertip_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.grasp_env import FixedBaseGraspEnv
from humanoid_learning.expert.grasp_expert import BimanualSidePinchExpert, GraspExpertConfig, GraspState
from humanoid_learning.expert.tripod_closure_planner import audit_hand_geometry


def make_env():
    return FixedBaseGraspEnv(GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF))


def print_report(label, report):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"object pos={np.round(report['object']['pos'], 4)} half_size={report['object']['half_size']}")
    print(f"thumb-thumb distance = {report['thumb_thumb_distance_m']:.4f} m")
    for side in ("left", "right"):
        s = report["sides"][side]
        print(f"\n--- {side.upper()} ---")
        print(f"  palm_pos={np.round(s['palm_pos'],4)}")
        print(f"  wrist_pos={np.round(s['wrist_pos'],4)}")
        print(f"  legacy centroid (body-origin, world) = {np.round(s['legacy_centroid_world'],4)}")
        print(f"  true centroid   (corrected site, world) = {np.round(s['true_centroid_world'],4)}")
        print(f"  |true - legacy| = {s['legacy_vs_true_centroid_delta_m']:.4f} m")
        for finger in ("thumb", "index", "middle"):
            wp = np.round(s['current_true_tip_world'][finger], 4)
            lp = np.round(s['current_true_tip_object_local'][finger], 4)
            print(f"  {finger:6s} true tip world={wp}  object-local={lp}")
        print("  joint ranges & tip sensitivity:")
        for jn, rng in s["joint_ranges"].items():
            trav = s["joint_tip_sensitivity"][jn]["tip_travel_m"]
            print(f"    {jn:28s} range=[{rng[0]:+.4f},{rng[1]:+.4f}]  full-range tip travel={trav:.4f}m")


# --- 1) fresh reset pose ---
env = make_env()
env.reset(seed=0)
report_reset = audit_hand_geometry(env)
print_report("STATE A: fresh reset pose", report_reset)

# --- 2) canonical rollout's CONTACT_ACQUIRE-end / THUMB_OPPOSE-entry pose ---
env2 = make_env()
env2.reset(seed=0)
expert = BimanualSidePinchExpert(env2, GraspExpertConfig())
outcome = None
for i in range(1200):
    outcome = expert.step()
    if outcome.state == GraspState.THUMB_OPPOSE:
        break
    if outcome.state in (GraspState.FAILURE, GraspState.SUCCESS):
        break
print(f"\nreached state={outcome.state.name} at step {i}")
report_contact = audit_hand_geometry(env2, expert)
print_report("STATE B: canonical rollout, THUMB_OPPOSE entry (step %d)" % i, report_contact)

import json
Path(PROJECT_ROOT / "results" / "grasp_feasibility_map").mkdir(parents=True, exist_ok=True)
out_path = PROJECT_ROOT / "results" / "tripod_closure_geometry_audit.json"
with open(out_path, "w") as f:
    json.dump({"state_a_reset": report_reset, "state_b_thumb_oppose_entry": report_contact}, f, indent=2)
print(f"\nwrote {out_path}")
