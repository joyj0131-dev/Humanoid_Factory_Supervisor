#!/usr/bin/env python3
"""Interactive viewer for the two-workcell factory.

A separate entry point rather than another branch inside view_whole_body.py:
that script is already 5 modes and a dozen grasp-specific flags, and the factory
shares none of them.

    DISPLAY=:0 python3 scripts/view_factory.py
    DISPLAY=:0 python3 scripts/view_factory.py --seed 1 --fault-workcell 0
    python3 scripts/view_factory.py --offscreen --out results/factory/scene.png

Headless machines should use --offscreen (it sets MUJOCO_GL=egl itself).

The G1 only stands: there is no walking controller in this repository, so the
supervisor does not travel between the cells here. What the viewer shows is the
factory running, one cell faulting, and where the robot would have to stand.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from humanoid_learning.envs import factory_config as fcfg


def _describe(env, info) -> str:
    faulted = info["arm_faulted"]
    cells = " | ".join(
        f"wc{k}: {'FAULT' if faulted[k] else 'running'} wp{info['arm_waypoint'][k]}"
        for k in range(fcfg.N_WORKCELLS)
    )
    target = info["target_workcell"]
    return f"{cells} || target={'none yet' if target < 0 else f'wc{target}'}"


CAMERA_DISTANCE = 6.2
CAMERA_AZIMUTH = 180.0
CAMERA_ELEVATION = -20.0


def _camera_lookat(config) -> list[float]:
    """Centre on the midpoint between the G1 home pose and the two cells, so
    the robot and both workcells all fit in frame."""
    cells = config.workcells
    centre_x = float(np.mean([p.manipulation_xy[0] for p in cells]))
    return [centre_x * 0.55, 0.0, 0.85]


def _camera(config):
    import mujoco

    camera = mujoco.MjvCamera()
    camera.lookat[:] = _camera_lookat(config)
    camera.distance = CAMERA_DISTANCE
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION
    return camera


def _update_beacons(env) -> None:
    """Presentation only: recolour each cell's beacon from its arm state. Kept
    in the viewer rather than the env so the env never mutates the model for
    display purposes."""
    import mujoco

    for k, arm in enumerate(env.arms):
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, fcfg.beacon_geom_name(k))
        if gid >= 0:
            env.model.geom_rgba[gid] = (
                fcfg.BEACON_FAULT_RGBA if arm.faulted else fcfg.BEACON_RUNNING_RGBA
            )


def run_offscreen(env, args) -> None:
    import mujoco
    from PIL import Image

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _, info = env.reset(seed=args.seed, options=_reset_options(args))
    camera = _camera(env.factory)
    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
    shots = {}
    for step in range(args.steps):
        _, _, _, _, info = env.step(zero)
        _update_beacons(env)
        if step + 1 in args.capture:
            shots[step + 1] = info
            with mujoco.Renderer(env.model, height=720, width=1280) as renderer:
                renderer.update_scene(env.data, camera)
                path = out if len(args.capture) == 1 else out.with_name(f"{out.stem}_{step + 1:04d}{out.suffix}")
                Image.fromarray(renderer.render()).save(path)
                print(f"step {step + 1:5d}  {_describe(env, info)}  -> {path}")
    if not shots:
        print("no frames captured; check --capture against --steps")


def run_interactive(env, args) -> None:
    import mujoco.viewer

    _, info = env.reset(seed=args.seed, options=_reset_options(args))
    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
    print(f"Factory viewer: layout={env.factory.layout}, seed={args.seed}, "
          f"fault scheduled for wc{env.fault_workcell} at step {env.fault_step}.")
    print("The G1 stands (no walking controller exists). Close the viewer to exit.")
    target_dt = env.model.opt.timestep * env.config.frame_skip
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = _camera_lookat(env.factory)
        viewer.cam.distance = CAMERA_DISTANCE
        viewer.cam.azimuth = CAMERA_AZIMUTH
        viewer.cam.elevation = CAMERA_ELEVATION
        announced = False
        while viewer.is_running():
            started = time.time()
            _, _, _, truncated, info = env.step(zero)
            _update_beacons(env)
            if info["fault_active"] and not announced:
                announced = True
                print(f"  step {info['step_count']:5d}  FAULT injected -> {_describe(env, info)}")
            if truncated:
                env.reset(seed=args.seed, options=_reset_options(args))
                announced = False
            viewer.sync()
            remaining = target_dt - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)


def _reset_options(args) -> dict:
    options = {}
    if args.fault_workcell is not None:
        options["fault_workcell"] = args.fault_workcell
    if args.fault_step is not None:
        options["fault_step"] = args.fault_step
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", choices=sorted(fcfg.LAYOUT_PRESETS), default=fcfg.DEFAULT_LAYOUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fault-workcell", type=int, choices=range(fcfg.N_WORKCELLS), default=None,
                        help="override the seed's choice of which cell fails")
    parser.add_argument("--fault-step", type=int, default=None)
    parser.add_argument("--offscreen", action="store_true", help="render PNGs instead of opening a window")
    parser.add_argument("--out", default="results/factory/factory.png")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--capture", type=int, nargs="+", default=[150, 400],
                        help="steps at which to save a frame in --offscreen mode")
    args = parser.parse_args()

    if args.offscreen:
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        os.environ.setdefault("MUJOCO_GL", "glx")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

    from humanoid_learning.envs.factory_env import FactoryEnv

    env = FactoryEnv(fcfg.FactoryConfig(layout=args.layout))
    try:
        (run_offscreen if args.offscreen else run_interactive)(env, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
