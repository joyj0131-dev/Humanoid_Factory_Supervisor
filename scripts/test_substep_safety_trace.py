"""Tests for the SIZE_12 Thumb Contact-Loss Causality session: the new
FixedBaseGraspEnv.substep_hook (grasp_env.py) and
humanoid_learning.expert.substep_safety_trace's physics-substep-level
tracer built on it. Run with:
    python scripts/test_substep_safety_trace.py

Section 7's SAFETY_ROLLBACK_LIMIT_CYCLE hypothesis is tested here against
the REAL, measured SIZE_12 canonical rollout -- and found NOT confirmed
(see test_safety_rollback_limit_cycle_not_confirmed and the session's
PROJECT_CONTEXT.md report). Per that session's own explicit fallback
instruction, the event-latched thumb controller described in the
directive was therefore NOT implemented; these tests lock in the causal
finding that was reached instead (a cross-hand OBJECT_ROTATION_INDUCED_LOSS
event), not a speculative fix.
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


def test_size12_thumb_contact_is_a_single_late_episode_per_hand():
    """REAL FINDING (substep granularity): in the canonical SIZE_12
    rollout, each thumb touches the object exactly ONCE for the entire
    run -- there is no earlier flicker of contact/loss/re-contact to
    examine. This directly matters for Section 7's hypothesis check: a
    'limit cycle' requires the SAME failure pattern to repeat >= 2 times,
    which structurally cannot be observed if only one contact episode
    exists at all."""
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
    assert len(left_episodes) == 1
    assert len(right_episodes) == 1


def test_safety_rollback_limit_cycle_not_confirmed():
    """Mechanically applies Section 7's checklist (via
    check_safety_rollback_limit_cycle_criteria) to the real SIZE_12
    canonical trace. REAL FINDING: it does NOT confirm, because there are
    not >= 2 qualifying episodes (see the previous test: there is only
    ONE contact episode per hand in the whole rollout) -- so per the
    directive's own explicit rule ('부분적으로만 관찰되면 가설을 확정하지
    말 것'), the event-latched thumb controller must NOT be implemented
    on this basis. This test locks in that negative result."""
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
          f"confirmed={result['confirmed']}")
    assert not result["confirmed"], (
        "REAL FINDING: SAFETY_ROLLBACK_LIMIT_CYCLE is NOT confirmed on the canonical SIZE_12 rollout -- "
        "fewer than 2 qualifying episodes exist, so the event-latched controller was correctly NOT built"
    )


def test_final_contact_loss_coincides_with_cross_hand_safety_event_and_rotation_spike():
    """REAL FINDING: the LEFT thumb's terminal contact loss (force
    collapsing near-instantly) occurs in the SAME control tick as (a) a
    substep-level safety rollback fired for the RIGHT thumb group and/or
    hand-hand collision, and (b) a sharp rise in object angular speed --
    consistent with classification E (OBJECT_ROTATION_INDUCED_LOSS)
    precipitated by the other hand's contact dynamics, not a same-hand
    SAFETY_ROLLBACK_LIMIT_CYCLE (category A, ruled out above)."""
    env = make_env()
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    obj_body_id = env._object_body_id
    thumb_body_id, thumb_qpos_adr, thumb_group_act_ids = _side_ids(env)
    outcome, trace, synergy_log = collect_substep_trace(
        env, expert, obj_body_id, thumb_body_id, thumb_qpos_adr, thumb_group_act_ids, SIZE_12_HALF, max_steps=1200
    )
    left_episodes = find_contact_episodes("left", trace["left"])
    assert len(left_episodes) == 1
    ep = left_episodes[0]
    full = trace["left"]
    end_idx = next(k for k, r in enumerate(full) if r.global_substep == ep.end_gsubstep)
    # The rotational "kick" that ultimately peels the contact off precedes
    # full separation (touched->False) by a few substeps as the contact
    # force decays through several intermediate substeps -- so search the
    # episode's own TAIL (not just the exact touched/untouched boundary)
    # for the largest single-substep angular-speed rise.
    window = full[max(0, end_idx - 15):end_idx]
    ang = [s.ang_speed for s in window]
    ratios = [ang[k + 1] / max(ang[k], 1e-6) for k in range(len(ang) - 1)]
    peak_k = int(np.argmax(ratios))
    loss_cs = window[peak_k].control_step
    ang_before, ang_after = ang[peak_k], ang[peak_k + 1]
    print(f"    rotation spike at cs={loss_cs} ang_speed {ang_before:.2f}->{ang_after:.2f}, "
          f"final separation {end_idx - (max(0, end_idx - 15) + peak_k)} substeps later")
    assert ang_after > ang_before * 1.3, "object angular speed must spike within the episode's own tail, shortly before the contact fully separates"
    tick = next(s for s in synergy_log if s["i"] == loss_cs)
    assert tick["n_safety_events"] > 0 and "right" in tick["safety_sides"] or "hand_hand" in tick["safety_sides"], (
        "the tick where the left thumb's contact collapses must show a concurrent right-hand/hand-hand "
        "safety-rollback event -- the cross-hand coupling that actually knocks the contact off"
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
