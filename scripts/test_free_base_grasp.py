#!/usr/bin/env python3
"""Phase 5 step 1: can the G1 grasp while standing on its own two feet?

Run directly (the system pytest CLI is broken in this project):
    OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python3 scripts/test_free_base_grasp.py

These tests are slow (a full grasp is ~4300 steps); the toppling case is cut
short as soon as the fall is unambiguous.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from humanoid_learning.envs import frames, task_config as tc
from humanoid_learning.envs.grasp_config import GraspEnvConfig
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    BimanualGraspState,
    SharpaBimanualGraspExpert,
)
from humanoid_learning.expert.stance_stabilizer import StanceGains, StanceStabilizer

# Tuned by sweep, recorded in docs/FACTORY_ENVIRONMENT.md. Higher gains
# oscillate the robot over: kp 4 and 5 both toppled it.
TUNED = StanceGains(pitch_kp=1.0, pitch_kd=0.1, roll_kp=0.7, roll_kd=0.07)


def _rollout(fixed_base, stabilize, seed=0, max_steps=5000, stop_on_fall=False):
    env = SharpaGraspEnv(
        GraspEnvConfig(fixed_base=fixed_base, arm_gravity_compensation=True, max_episode_steps=max_steps + 10)
    )
    try:
        env.reset(seed=seed)
        expert = SharpaBimanualGraspExpert(env)
        stabilizer = StanceStabilizer(env, TUNED) if stabilize else None
        model, data = env.model, env.data
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, tc.OBJECT_GEOM)
        table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, tc.TABLE_GEOM)
        start_z = float(data.xpos[pelvis][2])
        worst_drop = worst_pitch = 0.0
        steps = 0
        for steps in range(1, max_steps + 1):
            action = expert.step()
            if stabilizer is not None:
                stabilizer.apply()
            env.step(action)
            _, pitch = frames.roll_pitch_from_quat(data.xquat[pelvis])
            worst_pitch = max(worst_pitch, abs(float(np.degrees(pitch))))
            worst_drop = max(worst_drop, start_z - float(data.xpos[pelvis][2]))
            if stop_on_fall and worst_drop > 0.20:
                break
            if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
                break
        return {
            "state": expert.state.name,
            "steps": steps,
            "clearance_mm": float(mujoco.mj_geomDistance(model, data, object_geom, table_geom, 1.0, None)) * 1000.0,
            "max_pitch_deg": worst_pitch,
            "pelvis_drop_mm": worst_drop * 1000.0,
        }
    finally:
        env.close()


def test_free_base_model_stands_with_no_action():
    env = SharpaGraspEnv(GraspEnvConfig(fixed_base=False, arm_gravity_compensation=True))
    try:
        env.reset(seed=0)
        pelvis = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        start_z = float(env.data.xpos[pelvis][2])
        for _ in range(1200):
            env.step(np.zeros(25))
        drop = (start_z - float(env.data.xpos[pelvis][2])) * 1000.0
        assert abs(drop) < 10.0, f"free base sagged {drop:.1f} mm just standing"
        print(f"    1200 steps of zero action: pelvis moved {drop:+.2f} mm")
    finally:
        env.close()


def test_fixed_base_grasp_is_unchanged():
    """Regression guard: adding the fixed_base option must not touch the
    canonical result every existing grasp number was measured on."""
    result = _rollout(fixed_base=True, stabilize=False)
    assert result["state"] == "SUCCESS", result
    assert result["steps"] == 4194, f"canonical rollout length changed: {result['steps']}"
    assert abs(result["clearance_mm"] - 81.0) < 0.5, result
    assert result["max_pitch_deg"] == 0.0, "a fixed base must not pitch at all"
    print(f"    fixed base: {result['state']} in {result['steps']} steps, "
          f"clearance {result['clearance_mm']:.2f} mm")


def test_unstabilized_free_base_really_does_topple():
    """Locks the measured failure, so nobody 'fixes' the stabiliser away.

    The robot's own CoM never leaves the support polygon while this happens
    (margin stays 61-179 mm against a 122 mm nominal), so this is a dynamic
    rocking failure, not a static balance one.
    """
    result = _rollout(fixed_base=False, stabilize=False, max_steps=1200, stop_on_fall=True)
    assert result["pelvis_drop_mm"] > 200.0, f"expected a fall, got {result}"
    assert result["max_pitch_deg"] > 30.0, result
    print(f"    without the stabiliser the pelvis dropped {result['pelvis_drop_mm']:.0f} mm "
          f"and pitched {result['max_pitch_deg']:.0f} deg")


def test_stabilized_free_base_grasps_while_standing():
    for seed in (0, 1, 2):
        result = _rollout(fixed_base=False, stabilize=True, seed=seed)
        assert result["state"] == "SUCCESS", (seed, result)
        assert result["clearance_mm"] > 50.0, (seed, result)
        assert result["max_pitch_deg"] < 10.0, (seed, result)
        assert result["pelvis_drop_mm"] < 10.0, (seed, result)
        print(f"    seed {seed}: {result['state']} in {result['steps']} steps, "
              f"clearance {result['clearance_mm']:.2f} mm, peak pitch {result['max_pitch_deg']:.2f} deg, "
              f"pelvis drop {result['pelvis_drop_mm']:.2f} mm")


def main() -> int:
    tests = [
        test_free_base_model_stands_with_no_action,
        test_fixed_base_grasp_is_unchanged,
        test_unstabilized_free_base_really_does_topple,
        test_stabilized_free_base_grasps_while_standing,
    ]
    passed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report, do not abort the suite
            print(f"FAIL  {test.__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS  {test.__name__}", flush=True)
    print(f"\n{passed} passed, {len(tests) - passed} failed out of {len(tests)}")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
