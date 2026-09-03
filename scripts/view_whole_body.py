"""Interactive viewer for the canonical G1 + Sharpa stack.

Examples:
    DISPLAY=:0 python3 scripts/view_whole_body.py --stand
    DISPLAY=:0 python3 scripts/view_whole_body.py --posture
    DISPLAY=:0 python3 scripts/view_whole_body.py --planar
    DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --no-restart
    DISPLAY=:0 python3 scripts/view_whole_body.py --sharpa-hand-demo
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco.viewer
import numpy as np

from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.envs.planar_debug_env import PlanarDebugEnv
from humanoid_learning.envs.whole_body_env import ACTION_DIM, N_LEGS, WholeBodyEnv


def _pace(model, started: float) -> None:
    remaining = model.opt.timestep * 5 - (time.time() - started)
    if remaining > 0:
        time.sleep(remaining)


def _run_env(env, action_fn, sequence_steps: int | None = None) -> None:
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        tick = 0
        while viewer.is_running():
            started = time.time()
            action_fn(tick if sequence_steps is None else tick % sequence_steps)
            viewer.sync()
            tick += 1
            _pace(env.model, started)


def mode_stand() -> None:
    env = WholeBodyEnv(wbc.WholeBodyConfig())
    env.reset(seed=0)
    zero = np.zeros(ACTION_DIM, dtype=np.float32)
    print("G1+Sharpa standing hold. Close the viewer to exit.")
    _run_env(env, lambda _tick: env.step(zero))


def mode_posture() -> None:
    env = WholeBodyEnv(wbc.WholeBodyConfig())
    env.reset(seed=0)
    zero = np.zeros(ACTION_DIM, dtype=np.float32)
    phases = (
        [("hold", 30), ("knee+", 15), ("hold", 150), ("knee-", 15),
         ("hold", 60), ("waist_roll+", 30), ("waist_roll-", 30),
         ("waist_yaw+", 30), ("waist_yaw-", 30), ("hold", 60)]
    )
    schedule = [name for name, count in phases for _ in range(count)]
    print("G1+Sharpa posture sequence. Close the viewer to exit.")

    def step(tick: int) -> None:
        action = zero.copy()
        phase = schedule[tick]
        if phase == "knee+":
            action[[3, 9]] = 0.08
        elif phase == "knee-":
            action[[3, 9]] = -0.08
        elif phase == "waist_roll+":
            action[N_LEGS + 1] = 0.5
        elif phase == "waist_roll-":
            action[N_LEGS + 1] = -0.5
        elif phase == "waist_yaw+":
            action[N_LEGS] = 0.5
        elif phase == "waist_yaw-":
            action[N_LEGS] = -0.5
        env.step(action)

    _run_env(env, step, len(schedule))


def mode_planar() -> None:
    env = PlanarDebugEnv(wbc.PlanarDebugConfig())
    env.reset(seed=0)
    path = (
        [np.array([1.0, 0.0, 0.0], np.float32)] * 80
        + [np.array([0.0, 1.0, 0.0], np.float32)] * 80
        + [np.array([-1.0, 0.0, 0.0], np.float32)] * 80
        + [np.array([0.0, -1.0, 0.0], np.float32)] * 80
        + [np.array([0.0, 0.0, 1.0], np.float32)] * 80
    )
    print("Planar debug path (not walking). Close the viewer to exit.")
    _run_env(env, lambda tick: env.step(path[tick]), len(path))


def mode_grasp(
    object_pos_x: float,
    object_half_size: float | None,
    no_restart: bool,
    mount: str,
    visual_style: str,
) -> None:
    from humanoid_learning.envs.grasp_config import GraspEnvConfig
    from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
    from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
        BimanualGraspState,
        SharpaBimanualGraspExpert,
    )

    kwargs = {
        "object_pos": (object_pos_x, 0.0, 0.0),
        "arm_kp": 120.0,
        "arm_gravity_compensation": True,
        "sharpa_mount": mount,
        "sharpa_visual_style": visual_style,
    }
    if object_half_size is not None:
        kwargs["object_half_size"] = object_half_size
    env = SharpaGraspEnv(GraspEnvConfig(**kwargs))
    state = {"expert": SharpaBimanualGraspExpert(env), "terminal": False, "last": None}
    env.reset(seed=0)
    print("Official Sharpa bimanual grasp; Gate A is still incomplete.")

    def restart() -> None:
        env.reset(seed=0)
        state["expert"] = SharpaBimanualGraspExpert(env)
        state["terminal"] = False
        state["last"] = None

    def step(_tick: int) -> None:
        if state["terminal"]:
            return
        expert = state["expert"]
        env.step(expert.step())
        if expert.state != state["last"]:
            print(f"  -> {expert.state.name}, reason={expert.failure_reason}")
            state["last"] = expert.state
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            print(f"  terminal={expert.state.name}, reason={expert.failure_reason}, "
                  f"streak={expert._max_bilateral_streak}/{expert.config.bilateral_streak_required}")
            if no_restart:
                state["terminal"] = True
                print("  holding final pose (--no-restart)")
            else:
                restart()

    _run_env(env, step)


def mode_hand_demo(no_restart: bool, visual_style: str) -> None:
    from humanoid_learning.envs.grasp_config import GraspEnvConfig
    from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
    from humanoid_learning.expert.sharpa_hand_demo import HandDemoState, SharpaHandDemo

    env = SharpaGraspEnv(GraspEnvConfig(arm_kp=120.0, arm_gravity_compensation=True,
                                        sharpa_visual_style=visual_style))
    env.reset(seed=0)
    state = {"demo": SharpaHandDemo(env), "done": False, "last": None}
    print("Free-space Sharpa open/close diagnostic; no grasp Gate is evaluated.")

    def step(_tick: int) -> None:
        if state["done"]:
            return
        demo = state["demo"]
        env.step(demo.step())
        if demo.state != state["last"]:
            print(f"  -> {demo.state.name}")
            state["last"] = demo.state
        if demo.state == HandDemoState.DONE and demo._state_step >= 10:
            if no_restart:
                state["done"] = True
            else:
                demo.loop_if_done()

    _run_env(env, step)


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--stand", action="store_true")
    modes.add_argument("--posture", action="store_true")
    modes.add_argument("--planar", action="store_true")
    modes.add_argument("--grasp", action="store_true")
    modes.add_argument("--sharpa-hand-demo", action="store_true")
    parser.add_argument("--object-pos-x", type=float, default=0.27)
    parser.add_argument("--object-half-size", type=float, default=None)
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--sharpa-mount", choices=["wrist", "flange"], default="wrist")
    parser.add_argument("--sharpa-visual-style", choices=["upstream", "g1"], default="g1")
    args = parser.parse_args()

    if args.stand:
        mode_stand()
    elif args.posture:
        mode_posture()
    elif args.planar:
        mode_planar()
    elif args.grasp:
        mode_grasp(args.object_pos_x, args.object_half_size, args.no_restart,
                   args.sharpa_mount, args.sharpa_visual_style)
    else:
        mode_hand_demo(args.no_restart, args.sharpa_visual_style)


if __name__ == "__main__":
    main()
