"""Carry-by-walking contract tests. These are not completed-repair claims.

What the carry is verified to do, measured on seed 0 / station 0:
  * hold the block for the whole walk (grip never below 3.3 N over 2501 ticks,
    against 0.0 N at tick 437 before the arms were stiffened), and
  * arrive at the placing stance without falling.
What it does NOT yet do: place. The block still finishes ~99 mm short fore/aft
and closing that by arm reach alone topples the robot, so PLACE stays opt-in.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_learning.envs.factory_env import FactoryEnv
from humanoid_learning.expert.factory_recovery import (
    FactoryRecovery, PrecisionApproach, RecoveryConfig)
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SharpaBimanualGraspExpert


def test_carry_is_opt_in_and_only_reachable_through_place():
    config = RecoveryConfig()
    assert config.place_after_lift is False, 'placing must stay opt-in'
    assert config.carry_by_walking is True, 'when placing is on, carrying is the default route'
    assert 'CARRY' not in FactoryRecovery.TERMINAL
    print('    place opt-in; carry is the route it takes, not a separate switch')


def test_the_pick_keeps_its_own_lateral_throttle_and_command_bias():
    """The carry's looser settings must not leak into the approach that grasps."""
    config = RecoveryConfig()
    assert config.lateral_clamp_radius_m == 0.2, 'the pick approach must keep its throttle'
    assert config.forward_command_bias == -0.08, 'the pick approach must keep its bias'
    assert config.carry_lateral_clamp_radius_m < config.lateral_clamp_radius_m
    assert config.carry_arrival_radius_m > config.arrival_radius_m
    print(f"    pick: throttle {config.lateral_clamp_radius_m} m, arrive "
          f"{config.arrival_radius_m} m; carry: {config.carry_lateral_clamp_radius_m} m, "
          f"{config.carry_arrival_radius_m} m")


def test_precision_approach_accepts_an_explicit_pelvis_goal():
    assert 'pelvis_goal' in PrecisionApproach.step.__code__.co_varnames
    print('    navigation can be aimed at a pelvis pose, not only at a part')


def test_carry_grip_changes_are_actuator_only_and_are_undone_on_close():
    """Stiffening the arms and deepening the squeeze must not teleport anything,
    and must not leave the model retuned for whatever runs next."""
    env = FactoryEnv()
    try:
        env.reset(seed=0)
        recovery = FactoryRecovery(env)
        recovery.station = 0
        recovery.expert = SharpaBimanualGraspExpert(recovery.grasp)
        before_qpos = env.data.qpos.copy()
        before_gain = env.model.actuator_gainprm.copy()

        recovery._tighten_grip_for_carry()
        np.testing.assert_array_equal(env.data.qpos, before_qpos)
        assert not np.array_equal(env.model.actuator_gainprm, before_gain), (
            'the carry is supposed to stiffen the arms')
        assert recovery.grasp.config.arm_kp == recovery.config.carry_arm_kp, (
            'gravity feedforward divides by arm_kp and must track the retune')
        assert set(recovery.pick_palm_local) == {'left', 'right'}

        recovery._restore_pick_extension()
        np.testing.assert_array_equal(env.data.qpos, before_qpos)

        recovery.close()
        np.testing.assert_array_equal(env.model.actuator_gainprm, recovery._original_gain)
        print('    squeeze and stiffening write ctrl only; close() restores the model tuning')
    finally:
        env.close()


def main() -> int:
    tests = [
        test_carry_is_opt_in_and_only_reachable_through_place,
        test_the_pick_keeps_its_own_lateral_throttle_and_command_bias,
        test_precision_approach_accepts_an_explicit_pelvis_goal,
        test_carry_grip_changes_are_actuator_only_and_are_undone_on_close,
    ]
    passed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report, do not abort the suite
            print(f'FAIL  {test.__name__}: {exc}', flush=True)
        else:
            passed += 1
            print(f'PASS  {test.__name__}', flush=True)
    print(f'\n{passed} passed, {len(tests) - passed} failed out of {len(tests)}')
    return 0 if passed == len(tests) else 1


if __name__ == '__main__':
    raise SystemExit(main())
