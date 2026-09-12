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


def _walk(station: int, seed: int = 0, steps: int = 1500):
    env = FactoryEnv()
    try:
        env.reset(seed=seed, options={"fault_workcell": station, "fault_step": 0})
        walker = G1WalkPolicy(env)
        pose = env.poses[station]
        navigator = WalkToPose(walker, pose.manipulation_xy, pose.heading_rad)
        zero = np.zeros(ACTION_DIM)
        goal = np.asarray(pose.manipulation_xy)
        arrived_at = None
        drift = 0.0
        for i in range(steps):
            navigator.step()
            # Deliberately NOT running the stance stabiliser here. After the
            # handoff the legs are held with stiff position gains, which is
            # already enough to stand; the ankle regulator is tuned for the
            # compliant grasp plant and destabilises this one (measured: the
            # robot fell at every station once the stabiliser stopped being a
            # silent no-op). It belongs to the grasp phase, not the walk.
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


def test_the_interactive_viewer_actually_drives_the_walker():
    """Runs scripts/view_factory.py's real interactive loop behind a stub viewer.

    This exists because --walk-to shipped broken: the walker was wired into the
    offscreen render path but two str.replace edits into the interactive loop
    silently matched nothing, so the window opened and the robot never moved.
    Rendering tests passed the whole time. Exercising the actual loop is the
    only thing that catches that.
    """
    import contextlib
    import importlib.util
    import types

    import mujoco
    import mujoco.viewer

    viewer_path = Path(__file__).resolve().parent / "view_factory.py"
    spec = importlib.util.spec_from_file_location("view_factory_under_test", viewer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class StubViewer:
        rtf_seen = False
        def __init__(self, ticks):
            self.ticks = ticks
            self.seen = 0
            self.cam = types.SimpleNamespace(lookat=np.zeros(3), distance=0.0,
                                             azimuth=0.0, elevation=0.0)

        def is_running(self):
            self.seen += 1
            return self.seen <= self.ticks

        @contextlib.contextmanager
        def lock(self):
            yield

        def set_texts(self, *a, **k):
            labels, values = a[0][-2:]
            if 'RTF (last 100 steps)' in labels:
                row = labels.splitlines().index('RTF (last 100 steps)')
                value = float(values.splitlines()[row].split('x')[0])
                assert np.isfinite(value) and value > 0
                StubViewer.rtf_seen = True

        def sync(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    original_launch, original_sleep = mujoco.viewer.launch_passive, module.time.sleep
    mujoco.viewer.launch_passive = lambda m, d, key_callback=None: StubViewer(1400)
    module.time.sleep = lambda _s: None
    try:
        for station in range(fcfg.N_WORKCELLS):
            StubViewer.rtf_seen = False
            env = FactoryEnv()
            try:
                pelvis = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                start = env.data.xpos[pelvis][:2].copy()
                args = types.SimpleNamespace(seed=0, walk_to=station, scenario="dropped_part",
                                             fault_workcell=None, fault_step=None, no_fault=False)
                module.run_interactive(env, args)
                assert StubViewer.rtf_seen, 'interactive viewer did not display measured RTF'
                travelled = float(np.linalg.norm(env.data.xpos[pelvis][:2] - start))
                result = env.navigation_result()
                assert travelled > 1.0, (
                    f"station {station}: the viewer loop moved the robot only {travelled * 1000:.0f} mm "
                    "-- the walker is not wired into the interactive path"
                )
                assert result["passed"], (station, result)
                print(f"    interactive loop, station {station}: robot travelled "
                      f"{travelled:.2f} m, gate passed, error "
                      f"{result['final_position_error_m'] * 1000:.1f} mm")
            finally:
                env.close()
    finally:
        mujoco.viewer.launch_passive = original_launch
        module.time.sleep = original_sleep


def main() -> int:
    tests = [
        test_policy_asset_is_installed,
        test_walks_to_both_stations_and_passes_the_navigation_gate,
        test_it_is_really_walking_not_sliding,
        test_arrival_hands_off_to_a_standing_hold_and_stays_put,
        test_the_interactive_viewer_actually_drives_the_walker,
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
