"""Phase 4 whole-body interactive viewer.

Separate from scripts/view_scene.py (Foundation fixed-base viewer, has its
own uncommitted user changes -- left untouched) since this drives a
completely different (floating-base) model.

Modes:
    --stand    hold the stand pose (verifies passive standing balance)
    --posture  small, verified-stable knee bend + waist lean/rotate, then
               recover (see PROJECT_CONTEXT.md Phase 4 report for the
               magnitude sweep behind these numbers)
    --planar   debug/fallback-only planar-base mode (NOT real locomotion --
               see PlanarDebugEnv docstring): commands a square path
    --grasp    the bimanual reach + pinch + close + lift attempt exactly as
               run in scripts/test_whole_body.py -- shown as-is, including
               its known failure mode (the object gets knocked away during
               approach), because PROJECT_CONTEXT.md Phase 4 reports this
               honestly rather than hiding it.

Run locally (needs a real display):
    python scripts/view_whole_body.py --stand
    python scripts/view_whole_body.py --posture
    python scripts/view_whole_body.py --planar
    python scripts/view_whole_body.py --grasp
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# This machine is an AMD iGPU + NVIDIA RTX 3060 hybrid (PRIME) laptop; the
# GLFW viewer defaults to the AMD iGPU otherwise (confirmed via
# GL_RENDERER/GL_VENDOR, 2026-08-28). Must be set before glfw/mujoco.viewer
# create their GL context -- future training work (Phase 4 locomotion) will
# need this same NVIDIA GPU, so the viewer is pinned to it here too.
os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np

from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.envs.planar_debug_env import PlanarDebugEnv
from humanoid_learning.envs.whole_body_env import ACTION_DIM, N_LEGS, WholeBodyEnv


def run_viewer(model, data, action_fn, steps: int | None = None, loop: bool = True):
    """Keeps the window open until the user closes it. If ``loop`` is True
    (default), the sequence repeats via ``action_fn(i % steps)`` once it
    reaches ``steps`` -- a fixed-length demo must not silently close the
    window (that made an earlier run finish and tear down the GL context in
    ~4 seconds, likely before it ever became visible)."""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        i = 0
        while viewer.is_running():
            step_start = time.time()
            idx = (i % steps) if (loop and steps) else i
            action_fn(idx)
            viewer.sync()
            i += 1
            if steps is not None and not loop and i >= steps:
                print("  demo finished -- close the viewer window to exit.")
                while viewer.is_running():
                    time.sleep(0.1)
                break
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep * 5:
                time.sleep(model.opt.timestep * 5 - elapsed)


def mode_stand():
    env = WholeBodyEnv(wbc.WholeBodyConfig())
    env.reset(seed=0)
    zero = np.zeros(ACTION_DIM, dtype=np.float32)
    print("Standing hold (zero action). See PROJECT_CONTEXT.md Phase 4: stable over 1000 steps.")
    print("Close the viewer window to exit.")

    def step(i):
        env.step(zero)

    run_viewer(env.model, env.data, step, loop=False)


def mode_posture():
    env = WholeBodyEnv(wbc.WholeBodyConfig())
    env.reset(seed=0)
    zero = np.zeros(ACTION_DIM, dtype=np.float32)
    print("Sequence: stand -> small knee bend (mag=0.08) -> hold -> return -> waist lean -> waist rotate -> stand.")

    schedule = []
    schedule += [("hold", 30)]
    schedule += [("knee+", 15)]
    schedule += [("hold", 150)]
    schedule += [("knee-", 15)]
    schedule += [("hold", 60)]
    schedule += [("waist_roll+", 30)]
    schedule += [("waist_roll-", 30)]
    schedule += [("waist_yaw+", 30)]
    schedule += [("waist_yaw-", 30)]
    schedule += [("hold", 60)]

    flat = []
    for name, n in schedule:
        flat += [name] * n

    print("Close the viewer window to exit; the sequence repeats.")

    def step(i):
        if i == 0:
            env.reset(seed=0)
        a = zero.copy()
        if i < len(flat):
            name = flat[i]
            if name == "knee+":
                a[3] = 0.08
                a[9] = 0.08
            elif name == "knee-":
                a[3] = -0.08
                a[9] = -0.08
            elif name == "waist_roll+":
                a[N_LEGS + 1] = 0.5
            elif name == "waist_roll-":
                a[N_LEGS + 1] = -0.5
            elif name == "waist_yaw+":
                a[N_LEGS + 0] = 0.5
            elif name == "waist_yaw-":
                a[N_LEGS + 0] = -0.5
        obs, r, term, trunc, info = env.step(a)
        if i % 60 == 0:
            print(f"  step {i:4d} height={info['pelvis_height']:.4f} fallen={info['fallen']}")

    run_viewer(env.model, env.data, step, steps=len(flat) + 30)


def mode_planar():
    env = PlanarDebugEnv(wbc.PlanarDebugConfig())
    env.reset(seed=0)
    print("DEBUG/FALLBACK planar-base mode (NOT real locomotion). Commanding a square path.")
    legs_arms_note = "legs/waist/arms never move in this mode -- only the pelvis translates/rotates."
    print(f"  {legs_arms_note}")

    path = (
        [np.array([1.0, 0.0, 0.0], dtype=np.float32)] * 80
        + [np.array([0.0, 1.0, 0.0], dtype=np.float32)] * 80
        + [np.array([-1.0, 0.0, 0.0], dtype=np.float32)] * 80
        + [np.array([0.0, -1.0, 0.0], dtype=np.float32)] * 80
        + [np.array([0.0, 0.0, 1.0], dtype=np.float32)] * 80
    )

    def step(i):
        a = path[i] if i < len(path) else np.zeros(3, dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)

    run_viewer(env.model, env.data, step, steps=len(path) + 20)


def mode_grasp(object_pos_x: float = 0.27, object_half_size: float | None = None):
    """Bimanual side-pinch attempt with the redesigned 9-state controller
    (STABLE_START..HOLD, rate-limited targets, grip_center/grip_half_width
    force regulation) on the 2x-scaled object -- real controller, shown
    as-is. See PROJECT_CONTEXT.md Phase 4 Grasp Track for the actual
    measured outcome; this viewer does not hide failures.

    object_pos_x / object_half_size default to this script's ORIGINAL
    hardcoded values (0.27, GraspEnvConfig's own default 0.06) -- exposed
    as parameters (Thumb Opposition + Claw-Style Bimanual Grasp session
    follow-up) so a specific tested configuration can actually be viewed
    live instead of always showing the default regardless of what was
    tested headlessly."""
    from humanoid_learning.envs.grasp_config import GraspEnvConfig
    from humanoid_learning.envs.grasp_env import FixedBaseGraspEnv
    from humanoid_learning.expert.grasp_expert import BimanualSidePinchExpert, GraspExpertConfig, GraspState

    print("Bimanual side-pinch grasp attempt (Phase 4, redesigned 9-state controller).")
    print(f"object_pos_x={object_pos_x}  object_half_size={object_half_size}")
    print("Close the viewer window to exit; a new attempt restarts automatically.")

    # mujoco.viewer.launch_passive is bound to one model/data pair for its
    # whole lifetime -- reset the SAME env for each new attempt rather than
    # constructing a new one (which would build a different model/data the
    # already-open viewer could not switch to).
    config_kwargs = dict(object_pos=(object_pos_x, 0.0, 0.0), arm_kp=120.0)
    if object_half_size is not None:
        config_kwargs["object_half_size"] = object_half_size
    env = FixedBaseGraspEnv(GraspEnvConfig(**config_kwargs))
    state = {"env": env, "expert": None, "last_state": None, "frame": 0}

    def new_attempt():
        env.reset(seed=0)
        state["expert"] = BimanualSidePinchExpert(env, GraspExpertConfig())
        state["last_state"] = None
        state["frame"] = 0

    new_attempt()

    def step(_i):
        env, expert = state["env"], state["expert"]
        outcome = expert.step()
        if outcome.state != state["last_state"]:
            print(
                f"  [{state['frame']:4d}] -> {outcome.state.name}  reason={outcome.failure_reason}  "
                f"Lforce={outcome.left_force_filtered:.1f}N Rforce={outcome.right_force_filtered:.1f}N  "
                f"synergy(L,R)=({outcome.left_actual_synergy:.2f},{outcome.right_actual_synergy:.2f})"
            )
            state["last_state"] = outcome.state
        state["frame"] += 1
        if outcome.state in (GraspState.SUCCESS, GraspState.FAILURE):
            print(
                f"  attempt finished: {outcome.state.name}  "
                f"pre_lift_pop={outcome.pre_lift_pop_height:.4f}m  "
                f"controlled_lift_gain={outcome.controlled_lift_gain:.4f}m  "
                f"final_height_above_initial={outcome.final_height_above_initial:.4f}m\n  restarting...\n"
            )
            new_attempt()

    with mujoco.viewer.launch_passive(state["env"].model, state["env"].data) as viewer:
        i = 0
        while viewer.is_running():
            step_start = time.time()
            step(i)
            viewer.sync()
            i += 1
            elapsed = time.time() - step_start
            if elapsed < state["env"].model.opt.timestep * 5:
                time.sleep(state["env"].model.opt.timestep * 5 - elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stand", action="store_true")
    parser.add_argument("--posture", action="store_true")
    parser.add_argument("--planar", action="store_true")
    parser.add_argument("--grasp", action="store_true")
    parser.add_argument("--object-pos-x", type=float, default=0.27, help="object x position, meters from robot origin (--grasp only)")
    parser.add_argument("--object-half-size", type=float, default=None, help="object half-size, meters (--grasp only; default: GraspEnvConfig's own default, 0.06)")
    args = parser.parse_args()

    modes = [args.stand, args.posture, args.planar, args.grasp]
    if sum(bool(m) for m in modes) != 1:
        parser.error("pass exactly one of --stand / --posture / --planar / --grasp")

    if args.stand:
        mode_stand()
    elif args.posture:
        mode_posture()
    elif args.planar:
        mode_planar()
    elif args.grasp:
        mode_grasp(object_pos_x=args.object_pos_x, object_half_size=args.object_half_size)


if __name__ == "__main__":
    main()
