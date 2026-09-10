#!/usr/bin/env python3
"""Phase 5: does the G1 walk to each conveyor station and pass the Navigation Gate?

Run directly (the system pytest CLI is broken in this project):
    OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl python3 scripts/test_factory_walk.py

Slow: each run is ~1500 env steps of policy inference.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from humanoid_learning.envs import factory_config as fcfg
from humanoid_learning.envs.factory_env import ACTION_DIM, FactoryEnv
from humanoid_learning.expert.g1_walk_policy import POLICY_PATH, G1WalkPolicy, WalkToPose
from humanoid_learning.expert.stance_stabilizer import StanceGains, StanceStabilizer

TUNED = StanceGains(pitch_kp=1.0, pitch_kd=0.1, roll_kp=0.7, roll_kd=0.07)


def _walk(station: int, seed: int = 0, steps: int = 1500):
    env = FactoryEnv()
    try:
        env.reset(seed=seed, options={"fault_workcell": station, "fault_step": 0})
        walker = G1WalkPolicy(env)
        pose = env.poses[station]
        navigator = WalkToPose(walker, pose.manipulation_xy, pose.heading_rad)
        stabilizer = StanceStabilizer(env, TUNED)
        zero = np.zeros(ACTION_DIM)
        goal = np.asarray(pose.manipulation_xy)
        arrived_at = None
        drift = 0.0
        for i in range(steps):
            navigator.step()
            if walker.holding:
                stabilizer.apply()
            env.step(zero)
            if navigator.arrived and arrived_at is None:
                arrived_at = i
            if arrived_at is not None:
                drift = max(drift, float(np.linalg.norm(walker.base_pose()[0] - goal)))
        result = env.navigation_result()
        result["arrived_at"] = arrived_at
        result["drift_after_arrival_m"] = drift
        return result
    finally:
        env.close()


def test_policy_asset_is_installed():
    assert POLICY_PATH.exists(), (
        f"{POLICY_PATH} missing. Run: python3 scripts/install_g1_walk_policy.py"
    )
    notice = POLICY_PATH.parent / "NOTICE"
    assert notice.exists() and "BSD 3-Clause" in notice.read_text()
    print(f"    policy present ({POLICY_PATH.stat().st_size} bytes) with BSD-3 provenance recorded")


def test_walks_to_both_stations_and_passes_the_navigation_gate():
    for station in range(fcfg.N_WORKCELLS):
        result = _walk(station)
        assert result["passed"], (station, result)
        assert not result["fell"], (station, result)
        assert not result["teleported"], "base moved non-physically"
        assert not result["forbidden_contact"], (station, result)
        assert not result["entered_wrong_workcell_first"], (station, result)
        print(f"    station {station}: PASSED, final error "
              f"{result['final_position_error_m'] * 1000:.1f} mm, heading "
              f"{np.degrees(result['final_heading_error_rad']):.1f} deg, held "
              f"{result['max_hold_steps']} steps, max base step "
              f"{result['max_base_step_m'] * 1000:.1f} mm")


def test_it_is_really_walking_not_sliding():
    """The gate rejects teleportation, but check the motion looks like gait."""
    result = _walk(0)
    # A physical step at ~0.1 m/s over a 10 ms tick is millimetres, not
    # centimetres; a kinematic slide to the goal would show far larger jumps.
    assert result["max_base_step_m"] < 0.02, result
    assert result["max_base_step_m"] > 0.0005, "the base barely moved -- did it walk at all?"
    print(f"    max base displacement per step = {result['max_base_step_m'] * 1000:.2f} mm "
          "(teleport threshold is 50 mm)")


def test_arrival_hands_off_to_a_standing_hold_and_stays_put():
    """A velocity-command walking policy cannot stand still: commanding zero
    still steps, and it drifts ~0.2 m off a pose it just reached. Arrival must
    hand the legs to a standing hold, and only while both feet are loaded."""
    for station in range(fcfg.N_WORKCELLS):
        result = _walk(station)
        assert result["arrived_at"] is not None, f"station {station} never arrived"
        assert result["drift_after_arrival_m"] < 0.10, (station, result)
        print(f"    station {station}: arrived at step {result['arrived_at']}, then stayed within "
              f"{result['drift_after_arrival_m'] * 1000:.1f} mm")


def main() -> int:
    tests = [
        test_policy_asset_is_installed,
        test_walks_to_both_stations_and_passes_the_navigation_gate,
        test_it_is_really_walking_not_sliding,
        test_arrival_hands_off_to_a_standing_hold_and_stays_put,
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
