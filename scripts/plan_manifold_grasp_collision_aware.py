"""32nd Phase 4 Grasp Track session: collision-aware manifold planner
A/B comparison.

A. LEGACY_FORBIDDEN_BUCKETS -- the 31st session's own (narrower)
   forbidden-contact set (thumb_thumb, wrist_palm_object, table_hand
   only). Reproduces the 31st session's own multi-start behavior using
   this session's more general classifier underneath, so this is an
   apples-to-apples causal baseline, not a re-run of literally the old
   code.
B. ALL_FORBIDDEN_BUCKETS -- adds proximal_object, hand_hand, and
   self_collision, all found by this session's own audit of the 31st
   session's best candidate (index/middle proximal links penetrating
   the object by up to 3.3cm; torso-vs-shoulder self-collision, 7.6mm).

Run with:
    python scripts/plan_manifold_grasp_collision_aware.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.grasp_env import FixedBaseGraspEnv
from humanoid_learning.expert.grasp_expert import BimanualSidePinchExpert, GraspExpertConfig, GraspState
from humanoid_learning.expert.manifold_grasp_planner import (
    run_multistart, is_statically_feasible, ALL_FORBIDDEN_BUCKETS, LEGACY_FORBIDDEN_BUCKETS,
)


def make_env():
    return FixedBaseGraspEnv(GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF))


def reach_thumb_oppose():
    env = make_env()
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    outcome = None
    for i in range(1200):
        outcome = expert.step()
        if outcome.state in (GraspState.THUMB_OPPOSE, GraspState.FAILURE, GraspState.SUCCESS):
            break
    assert outcome.state == GraspState.THUMB_OPPOSE
    return env, expert


def summarize(label, results):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    rows = []
    for r in results:
        feasible = is_statically_feasible(r)
        max_face = max(abs(v) for v in r.per_finger_face_residual.values())
        min_margin = min(r.joint_margins_rad.values())
        rows.append(dict(
            name=r.name, topology=r.topology_name, cost=r.cost, max_face=max_face,
            n_forbidden=r.collisions["n_forbidden_contacts"], max_pen=r.collisions["max_penetration"],
            thumb_thumb=r.collisions["thumb_thumb"], hand_hand=r.collisions.get("hand_hand", 0),
            proximal_object=r.collisions.get("proximal_object", 0), wrist_palm_object=r.collisions.get("wrist_palm_object", 0),
            self_collision=r.collisions.get("self_collision", 0), table_hand=r.collisions.get("table_hand", 0),
            tt_dist=r.thumb_thumb_distance, min_margin=min_margin, travel=r.joint_travel_from_canonical,
            feasible=feasible,
        ))
        print(f"[{r.topology_name:16s}/{r.name:26s}] cost={r.cost:.4f} max_face={max_face:.4f} "
              f"n_forbidden={r.collisions['n_forbidden_contacts']} max_pen={r.collisions['max_penetration']:.4f} "
              f"tt_dist={r.thumb_thumb_distance:.4f} min_margin={min_margin:.4f}rad feasible={feasible}")
    n_feasible = sum(1 for r in rows if r["feasible"])
    print(f"\n{label}: {n_feasible}/{len(rows)} statically feasible")
    return rows, n_feasible


env_a, expert_a = reach_thumb_oppose()
results_a, ctx_a = run_multistart(env_a, expert_a, forbidden_buckets=LEGACY_FORBIDDEN_BUCKETS)
rows_a, feas_a = summarize("CONDITION A: legacy forbidden set (thumb_thumb, wrist_palm_object, table_hand)", results_a)

env_b, expert_b = reach_thumb_oppose()
results_b, ctx_b = run_multistart(env_b, expert_b, forbidden_buckets=ALL_FORBIDDEN_BUCKETS)
rows_b, feas_b = summarize("CONDITION B: full forbidden set (+proximal_object, hand_hand, self_collision)", results_b)

print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
print(f"A (legacy):  {feas_a}/{len(rows_a)} feasible, best cost={min(r['cost'] for r in rows_a):.4f}, "
      f"best max_face={min(r['max_face'] for r in rows_a):.4f}")
print(f"B (full):    {feas_b}/{len(rows_b)} feasible, best cost={min(r['cost'] for r in rows_b):.4f}, "
      f"best max_face={min(r['max_face'] for r in rows_b):.4f}")

best_a = min(results_a, key=lambda r: r.cost)
best_b = min(results_b, key=lambda r: r.cost)
print(f"\nBest-by-cost A: {best_a.topology_name}/{best_a.name}")
print(f"  collisions={best_a.collisions}")
print(f"  contact_pairs (forbidden only):")
for p in best_a.contact_pairs:
    if p["bucket"] != "allowed_tip_contact":
        print(f"    {p['bucket']:18s} {p['bodies']:60s} geoms={p['geom1']}<->{p['geom2']} pen={p['penetration']*1000:.2f}mm pos={np.round(p['pos'],4)}")

print(f"\nBest-by-cost B: {best_b.topology_name}/{best_b.name}")
print(f"  collisions={best_b.collisions}")
print(f"  contact_pairs (forbidden only):")
for p in best_b.contact_pairs:
    if p["bucket"] != "allowed_tip_contact":
        print(f"    {p['bucket']:18s} {p['bodies']:60s} geoms={p['geom1']}<->{p['geom2']} pen={p['penetration']*1000:.2f}mm pos={np.round(p['pos'],4)}")

out_path = PROJECT_ROOT / "results" / "manifold_grasp_collision_aware_ab.json"
with open(out_path, "w") as f:
    json.dump({"A_legacy": rows_a, "B_full": rows_b}, f, indent=2)
print(f"\nwrote {out_path}")
