"""SharpaGraspEnv + SharpaGraspExpert tests (Phase 4, 35th session, Stage 4).

Run AFTER test_sharpa_wave_model.py and test_sharpa_g1_integration.py.
Run with:
    python3 scripts/test_sharpa_grasp.py

REAL FINDING (this session): the first working end-to-end rollout makes
genuine multi-finger contact (thumb/middle/wrap groups, 1.7-4.9N peak
force) but currently loses that contact (CONTACT_LOST) during
FORCE_SETTLE -- SIZE_12 Gate A is NOT yet achieved. This is reported
honestly (not forced to pass) -- see docs/history/PHASE4_GRASP_SESSION_35.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv, ACTION_DIM
from humanoid_learning.expert.sharpa_grasp_expert import SharpaGraspExpert, SharpaGraspState, SharpaFailureReason


def make_env(max_episode_steps: int = 8000) -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=max_episode_steps)
    return SharpaGraspEnv(config)


def test_env_constructs_resets_and_steps_without_nan():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    for _ in range(20):
        obs, r, term, trunc, info = env.step(np.zeros(ACTION_DIM))
    assert not info["unstable"]
    assert np.all(np.isfinite(obs))
    print(f"    env resets/steps cleanly, obs shape {obs.shape}, action dim {ACTION_DIM}")


def test_set_preshape_moves_only_preshape_joints_of_the_named_side():
    env = make_env()
    env.reset(seed=0)
    left_before = env.data.ctrl[env._preshape_act_ids["left"]].copy()
    right_before = env.data.ctrl[env._preshape_act_ids["right"]].copy()
    env.set_preshape("right", 1.0)
    left_after = env.data.ctrl[env._preshape_act_ids["left"]].copy()
    right_after = env.data.ctrl[env._preshape_act_ids["right"]].copy()
    assert np.allclose(left_before, left_after), "set_preshape('right', ...) must not touch the left hand"
    assert not np.allclose(right_before, right_after), "set_preshape('right', 1.0) must actually move the right hand's preshape joints"
    print("    set_preshape moves only the named side's preshape joints")


def test_deterministic_rollout_reproducible():
    env1 = make_env()
    expert1 = SharpaGraspExpert(env1)
    outcome1 = expert1.run(max_total_steps=2000)
    env2 = make_env()
    expert2 = SharpaGraspExpert(env2)
    outcome2 = expert2.run(max_total_steps=2000)
    assert outcome1.state == outcome2.state
    assert outcome1.step_count == outcome2.step_count
    assert outcome1.group_contact == outcome2.group_contact
    print(f"    deterministic: both runs reach {outcome1.state.name} at step {outcome1.step_count}")


def test_collision_census_reports_contype_1_pairs_only():
    """Stage 8 requirement: a basic collision census sanity check --
    every real contact reported by the env is between two contype=1
    (real collision, never the visual duplicate) geoms."""
    env = make_env()
    env.reset(seed=0)
    for _ in range(50):
        env.step(np.zeros(ACTION_DIM))
    import mujoco
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        assert env.model.geom_contype[c.geom1] == 1 and env.model.geom_contype[c.geom2] == 1
    print(f"    {env.data.ncon} real contacts at step 50, all contype=1 pairs")


def test_full_rollout_runs_to_a_terminal_state_without_crashing():
    """Headless rollout smoke test -- does NOT assert Gate A success
    (see this file's module docstring: Gate A is not yet achieved)."""
    env = make_env()
    expert = SharpaGraspExpert(env)
    outcome = expert.run(max_total_steps=8000)
    assert outcome.state in (SharpaGraspState.SUCCESS, SharpaGraspState.FAILURE)
    print(f"    rollout reached terminal state {outcome.state.name} "
          f"(failure_reason={outcome.failure_reason}) after {outcome.step_count} steps")


def test_real_multi_finger_contact_force_is_achieved_even_though_gate_a_is_not():
    """REAL FINDING this session: thumb/middle/wrap groups DO make real,
    measurable contact-force-based contact (not zero, not a sensor
    simulation) during a full rollout -- this is genuine progress even
    though the object is currently displaced too far / contact is lost
    before Gate A's stability requirement is met. This test locks in the
    CURRENT honest state: some contact, not yet a passing Gate A."""
    env = make_env()
    expert = SharpaGraspExpert(env)
    outcome = expert.run(max_total_steps=8000)
    n_contacted = sum(outcome.group_contact.values())
    print(f"    groups ever contacted: {outcome.group_contact}, "
          f"peak forces: {outcome.group_peak_force}, gate_a={outcome.gate_a}")
    assert n_contacted >= 1, "at least one finger group should make real contact-force-based contact"


def test_gate_a_is_not_falsely_reported_true_after_a_failure():
    """REAL BUG found and fixed this session: an earlier version of
    SharpaGraspExpert.run() computed gate_a as `state NOT IN <early
    states>`, which is True for FAILURE too (FAILURE is not one of the
    excluded early states) -- so a rollout that reached THUMB_OPPOSE and
    then failed with CONTACT_LOST incorrectly reported gate_a=True. This
    locks in the fix: gate_a must be False whenever the terminal state is
    FAILURE."""
    env = make_env()
    expert = SharpaGraspExpert(env)
    outcome = expert.run(max_total_steps=8000)
    if outcome.state == SharpaGraspState.FAILURE:
        assert outcome.gate_a is False, "gate_a must never be True when the rollout ended in FAILURE"
    print(f"    gate_a={outcome.gate_a} consistent with terminal state {outcome.state.name}")


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
