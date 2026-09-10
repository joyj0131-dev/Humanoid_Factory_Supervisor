#!/usr/bin/env python3
"""How far can the part be from its canonical spot before the grasp fails?

This decides whether "walk to the station, then grasp" can work: the walking
policy arrives within 35-47 mm, so the grasp must tolerate at least that much.

    OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python3 scripts/audit_grasp_workspace_window.py
    OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python3 scripts/audit_grasp_workspace_window.py --free-base

Each rollout is a full grasp attempt (~4000 steps), so the default sweep takes a
while. Nothing here is a pass/fail gate; it measures a limit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs.grasp_config import GraspEnvConfig
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    BimanualGraspState,
    SharpaBimanualGraspExpert,
)
from humanoid_learning.expert.stance_stabilizer import StanceGains, StanceStabilizer

TUNED = StanceGains(pitch_kp=1.0, pitch_kd=0.1, roll_kp=0.7, roll_kd=0.07)
CANONICAL_XY = (0.27, 0.0)


def attempt(dx: float, dy: float, fixed_base: bool = True, seed: int = 0) -> dict:
    config = GraspEnvConfig(
        fixed_base=fixed_base,
        arm_gravity_compensation=True,
        max_episode_steps=5010,
        object_pos=(CANONICAL_XY[0] + dx, CANONICAL_XY[1] + dy, 0.0),
    )
    env = SharpaGraspEnv(config)
    try:
        env.reset(seed=seed)
        expert = SharpaBimanualGraspExpert(env)
        stabilizer = None if fixed_base else StanceStabilizer(env, TUNED)
        model, data = env.model, env.data
        object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, tc.OBJECT_GEOM)
        table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, tc.TABLE_GEOM)
        steps = 0
        for steps in range(1, 5001):
            action = expert.step()
            if stabilizer is not None:
                stabilizer.apply()
            env.step(action)
            if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
                break
        clearance = float(mujoco.mj_geomDistance(model, data, object_geom, table_geom, 1.0, None))
        return {
            "dx_mm": round(dx * 1000),
            "dy_mm": round(dy * 1000),
            "state": expert.state.name,
            "reason": str(expert.failure_reason).split(".")[-1],
            "steps": steps,
            "clearance_mm": round(clearance * 1000, 1),
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--free-base", action="store_true",
                        help="stand on both legs with the ankle stabiliser instead of a welded pelvis")
    parser.add_argument("--dx-mm", type=int, nargs="+", default=[-50, -30, -10, 0, 10, 35])
    parser.add_argument("--dy-mm", type=int, nargs="+", default=[0, 10, 20, 35])
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    base = "free" if args.free_base else "fixed"
    print(f"Grasp workspace window, {base} base. Canonical part spot is "
          f"({CANONICAL_XY[0]}, {CANONICAL_XY[1]}); offsets are from there.")
    print(f"{'dx_mm':>6} {'dy_mm':>6} {'result':>8} {'clear_mm':>9}  reason")
    rows = []
    for dx in args.dx_mm:
        for dy in args.dy_mm:
            row = attempt(dx / 1000.0, dy / 1000.0, fixed_base=not args.free_base)
            rows.append(row)
            print(f"{row['dx_mm']:6d} {row['dy_mm']:6d} {row['state']:>8} "
                  f"{row['clearance_mm']:9.1f}  {row['reason']}", flush=True)

    good = [r for r in rows if r["state"] == "SUCCESS"]
    print()
    if good:
        print(f"succeeded at {len(good)}/{len(rows)} offsets; "
              f"dx {min(r['dx_mm'] for r in good)}..{max(r['dx_mm'] for r in good)} mm, "
              f"dy {min(r['dy_mm'] for r in good)}..{max(r['dy_mm'] for r in good)} mm")
    else:
        print("no offset succeeded")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
