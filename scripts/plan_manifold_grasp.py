"""Reachable Contact Manifold Grasp Planner session, multi-start driver.
Runs a deterministic multi-start (posture families x topology
candidates) six-fingertip joint optimization from the canonical SIZE_12
THUMB_OPPOSE-entry pose, reports every start's diagnostics, and selects
the best statically-feasible candidate (or the best-scoring infeasible
one if none is fully feasible). Run with:

    python scripts/plan_manifold_grasp.py
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
from humanoid_learning.expert.manifold_grasp_planner import run_multistart, is_statically_feasible


def make_env():
    return FixedBaseGraspEnv(GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF))


env = make_env()
env.reset(seed=0)
expert = BimanualSidePinchExpert(env, GraspExpertConfig())
outcome = None
for i in range(1200):
    outcome = expert.step()
    if outcome.state in (GraspState.THUMB_OPPOSE, GraspState.FAILURE, GraspState.SUCCESS):
        break
print(f"reached state={outcome.state.name} at step {i}")
assert outcome.state == GraspState.THUMB_OPPOSE

results, ctx = run_multistart(env, expert)

rows = []
for r in results:
    feasible = is_statically_feasible(r)
    max_face = max(abs(v) for v in r.per_finger_face_residual.values())
    min_margin = min(r.joint_margins_rad.values())
    rows.append(dict(
        name=r.name, topology=r.topology_name, cost=r.cost, converged=r.converged,
        max_face_residual=max_face, thumb_thumb_dist=r.thumb_thumb_distance,
        min_joint_margin_rad=min_margin, collisions=r.collisions, joint_travel=r.joint_travel_from_canonical,
        feasible=feasible,
    ))
    print(f"[{r.topology_name:16s}/{r.name:26s}] cost={r.cost:.4f} max_face_res={max_face:.4f} "
          f"tt_dist={r.thumb_thumb_distance:.4f} min_margin={min_margin:.4f}rad "
          f"collisions={r.collisions} travel={r.joint_travel_from_canonical:.3f} feasible={feasible}")

feasible_rows = [r for r in results if is_statically_feasible(r)]
if feasible_rows:
    best = min(feasible_rows, key=lambda r: r.cost)
    print(f"\nBEST FEASIBLE: {best.topology_name}/{best.name} cost={best.cost:.4f}")
else:
    best = min(results, key=lambda r: r.cost)
    print(f"\nNO FULLY FEASIBLE CANDIDATE. Best by cost (NOT feasible): {best.topology_name}/{best.name} cost={best.cost:.4f}")
    print(f"  max_face_residual={max(abs(v) for v in best.per_finger_face_residual.values()):.4f}")
    print(f"  collisions={best.collisions}")
    print(f"  min_joint_margin_rad={min(best.joint_margins_rad.values()):.4f}")

out_path = PROJECT_ROOT / "results" / "manifold_grasp_multistart.json"
with open(out_path, "w") as f:
    json.dump(rows, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
print(f"\nwrote {out_path}")
