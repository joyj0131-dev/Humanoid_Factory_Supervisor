"""Tests for the SIZE_12 Thumb Contact-Loss Causality session: the new
FixedBaseGraspEnv.substep_hook (grasp_env.py) and
humanoid_learning.expert.substep_safety_trace's physics-substep-level
tracer built on it. Run with:
    python scripts/test_substep_safety_trace.py

The corrected tracer shows repeated force rollback/re-crossing inside one
continuous right-thumb contact episode.  Tests below lock in that actual
substep-level result and explicitly reject the former false assumptions
that each hand had exactly one episode and that terminal loss coincided
with a safety event in the exact same substep.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.grasp_env import FixedBaseGraspEnv
from humanoid_learning.expert.grasp_expert import BimanualSidePinchExpert, GraspExpertConfig, GraspState
from humanoid_learning.expert.substep_safety_trace import (
    collect_substep_trace, find_contact_episodes, check_safety_rollback_limit_cycle_criteria,
)


def make_env(**kwargs):
    return FixedBaseGraspEnv(GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF, **kwargs))


def _side_ids(env):
    thumb_body_id = {
        "left": mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_2_link"),
        "right": mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_2_link"),
    }
    thumb_qpos_adr = {"left": env._left_finger_qpos_adr[0:3], "right": env._right_finger_qpos_adr[0:3]}
    thumb_group_act_ids = {"left": env._left_group_act_ids[0], "right": env._right_group_act_ids[0]}
    return thumb_body_id, thumb_qpos_adr, thumb_group_act_ids


def test_substep_hook_none_by_default():
    env = make_env()
    env.reset(seed=0)
    assert env.substep_hook is None


def test_substep_hook_inert_when_set_to_a_no_op():
    """A no-op hook must not change the env's own control-path behavior
    (streak, final state) -- the hook only ever READS env/data, never
    writes to them."""
    env_a = make_env()
    env_a.reset(seed=0)
    expert_a = BimanualSidePinchExpert(env_a, GraspExpertConfig())
    outcome_a = None
    for _ in range(750):
        outcome_a = expert_a.step()
        if outcome_a.state in (GraspState.FAILURE, GraspState.SUCCESS):
            break

    env_b = make_env()
    env_b.reset(seed=0)
    calls = []
    env_b.substep_hook = lambda e, idx, prev: calls.append(idx)
    expert_b = BimanualSidePinchExpert(env_b, GraspExpertConfig())
    outcome_b = None
    for _ in range(750):
        outcome_b = expert_b.step()
        if outcome_b.state in (GraspState.FAILURE, GraspState.SUCCESS):
            break

    assert outcome_a.state == outcome_b.state
    assert outcome_a.max_bilateral_tripod_streak == outcome_b.max_bilateral_tripod_streak
    assert len(calls) > 0


def test_substep_hook_fires_exactly_frame_skip_times_per_step():
    env = make_env()
    env.reset(seed=0)
    calls = []
    env.substep_hook = lambda e, idx, prev: calls.append(idx)
    env.step(np.zeros(env.action_space.shape[0]))
    assert calls == list(range(env.config.frame_skip))


def test_size12_canonical_baseline_reproduced():
    """Sanity check (Section 0-2): the unchanged SIZE_12 baseline still
    reproduces the previously-documented streak=14 / CONTACT_LOST result
    before any further instrumentation is trusted."""
    env = make_env()
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    obj_body_id = env._object_body_id
    thumb_body_id, thumb_qpos_adr, thumb_group_act_ids = _side_ids(env)
    outcome, trace, synergy_log = collect_substep_trace(
        env, expert, obj_body_id, thumb_body_id, thumb_qpos_adr, thumb_group_act_ids, SIZE_12_HALF, max_steps=1200
    )
    assert outcome.state == GraspState.FAILURE
    assert outcome.max_bilateral_tripod_streak == 14
    assert len(trace["left"]) > 0 and len(trace["right"]) > 0


def test_size12_thumb_contact_episode_counts_match_corrected_trace():
    """The left thumb has two episodes; the right has one continuous
    episode.  This catches the old event-accumulation interpretation."""
    env = make_env()
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    obj_body_id = env._object_body_id
    thumb_body_id, thumb_qpos_adr, thumb_group_act_ids = _side_ids(env)
    outcome, trace, synergy_log = collect_substep_trace(
        env, expert, obj_body_id, thumb_body_id, thumb_qpos_adr, thumb_group_act_ids, SIZE_12_HALF, max_steps=1200
    )
    left_episodes = find_contact_episodes("left", trace["left"])
    right_episodes = find_contact_episodes("right", trace["right"])
    print(f"    left episodes={len(left_episodes)} right episodes={len(right_episodes)}")
    assert len(left_episodes) == 2
    assert len(right_episodes) == 1


def test_safety_rollback_limit_cycle_confirmed_within_episode():
    """Repeated rollback/re-crossing inside one episode is a cycle; a
    second touched=False gap is not required."""
    env = make_env()
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    obj_body_id = env._object_body_id
    thumb_body_id, thumb_qpos_adr, thumb_group_act_ids = _side_ids(env)
    outcome, trace, synergy_log = collect_substep_trace(
        env, expert, obj_body_id, thumb_body_id, thumb_qpos_adr, thumb_group_act_ids, SIZE_12_HALF, max_steps=1200
    )
    all_episodes = find_contact_episodes("left", trace["left"]) + find_contact_episodes("right", trace["right"])
    result = check_safety_rollback_limit_cycle_criteria(all_episodes, force_limit=8.0)
    print(f"    n_episodes={result['n_episodes']} n_qualifying={result['n_qualifying_episodes']} "
          f"reclose_pairs={result['n_reclose_spike_pairs']} confirmed={result['confirmed']}")
    assert result["n_reclose_spike_pairs"] >= 2
    assert result["confirmed"]


def test_substep_safety_flags_are_local_not_tick_accumulated():
    """A safety event marks only the substep that appended it.  The
    terminal left-thumb separation itself has no newly appended event,
    so the former exact-coincidence causal claim is not retained."""
    env = make_env()
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    obj_body_id = env._object_body_id
    thumb_body_id, thumb_qpos_adr, thumb_group_act_ids = _side_ids(env)
    outcome, trace, synergy_log = collect_substep_trace(
        env, expert, obj_body_id, thumb_body_id, thumb_qpos_adr, thumb_group_act_ids, SIZE_12_HALF, max_steps=1200
    )
    left_episodes = find_contact_episodes("left", trace["left"])
    assert len(left_episodes) == 2
    ep = left_episodes[-1]
    full = trace["left"]
    end_idx = next(k for k, r in enumerate(full) if r.global_substep == ep.end_gsubstep)
    assert not full[end_idx].safety_triggered_this_substep
    by_tick = {}
    for sample in trace["right"]:
        by_tick.setdefault(sample.control_step, []).append(sample.safety_triggered_this_substep)
    assert any(any(flags) and not all(flags) for flags in by_tick.values()), (
        "at least one control tick must contain both an event substep and later non-event substeps; "
        "otherwise tick-accumulated events are being mislabeled again"
    )


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
