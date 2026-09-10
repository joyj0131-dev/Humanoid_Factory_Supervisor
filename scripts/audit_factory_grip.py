#!/usr/bin/env python3
"""Measure BOTH factory grippers over complete cycles; this is not a G1 grasp test.

Example: OPENBLAS_NUM_THREADS=1 python3 scripts/audit_factory_grip.py --cycles 2
--step-jaws provides the previous abrupt jaw controller for a causal A/B.
No contact parameters, object poses, or G1 controls are altered during a run.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from humanoid_learning.envs import factory_config as cfg
from humanoid_learning.envs.factory_env import FactoryEnv


def audit(cycles=2, step_jaws=False, seed=0):
    env = FactoryEnv()
    try:
        if step_jaws:
            for arm in env.arms:
                arm.jaw_target = lambda a=arm: (cfg.GRIPPER_OPEN if a.faulted
                                               else float(a.jaw_targets[a.waypoint_index]))
        env.reset(seed=seed, options={"fault_step": 10**9})
        jaws = [{env.model.geom(cfg.arm_body_name(k, f"{s}_geom")).id
                 for s in cfg.GRIPPER_JOINT_SUFFIXES} for k in range(2)]
        parts = [env.model.geom(cfg.part_geom_name(k)).id for k in range(2)]
        metrics = [dict(station=k, peak_z_m=0., peak_penetration_mm=0., peak_force_n=0.,
                        reached_outfeed=False, penetration_peak_step=0) for k in range(2)]
        steps = cycles * env.arms[0].cycle_length
        env.config.max_episode_steps = max(env.config.max_episode_steps, steps)
        for step in range(steps):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            for k, m in enumerate(metrics):
                part = env.part_position(k)
                local = env.poses[k].to_local_xy(part[:2])
                m["peak_z_m"] = max(m["peak_z_m"], float(part[2]))
                m["reached_outfeed"] |= bool(abs(local[1] - cfg.LOCAL_OUTFEED_XY[1]) < .06
                                              and part[2] > cfg.TABLE_TOP_Z)
                force = 0.
                for i in range(env.data.ncon):
                    contact = env.data.contact[i]
                    geoms = {contact.geom1, contact.geom2}
                    if parts[k] not in geoms or not geoms & jaws[k]:
                        continue
                    penetration = max(0., -float(contact.dist) * 1000)
                    if penetration > m["peak_penetration_mm"]:
                        m["peak_penetration_mm"] = penetration
                        m["penetration_peak_step"] = step + 1
                    value = np.zeros(6)
                    mujoco.mj_contactForce(env.model, env.data, i, value)
                    force += float(np.linalg.norm(value[:3]))
                m["peak_force_n"] = max(m["peak_force_n"], force)
        return dict(seed=seed, steps=steps, step_jaws=step_jaws,
                    sampling="post-env-step, all phases; not a substep maximum",
                    stations=metrics)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-jaws", action="store_true")
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    print(json.dumps(audit(args.cycles, args.step_jaws, args.seed), indent=2))
