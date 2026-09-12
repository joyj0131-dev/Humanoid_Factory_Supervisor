#!/usr/bin/env python3
"""Interactive viewer for the conveyor-line factory.

A separate entry point rather than another branch inside view_whole_body.py:
that script is already 5 modes and a dozen grasp-specific flags, and the factory
shares none of them.

    DISPLAY=:0 python3 scripts/view_factory.py
    DISPLAY=:0 python3 scripts/view_factory.py --seed 1 --fault-workcell 0
    python3 scripts/view_factory.py --offscreen --out results/factory/scene.png

Headless machines should use --offscreen (it sets MUJOCO_GL=egl itself).

The G1 only stands: there is no walking controller in this repository, so the
supervisor does not travel between the stations here. What the viewer shows is
the line running, the arms picking parts off the belt, one station faulting, the
line stopping, and where the robot would have to stand to recover it.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from queue import SimpleQueue
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# envs package imports MuJoCo while loading factory_config. Select the backend
# before that import, not later in main(), when the GL backend is already cached.
if "--offscreen" in sys.argv:
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import numpy as np

from humanoid_learning.envs import factory_config as fcfg


def _describe(env, info) -> str:
    faulted = info["arm_faulted"]
    cells = " | ".join(
        f"st{k}: {'FAULT' if faulted[k] else 'running'} wp{info['arm_waypoint'][k]}"
        for k in range(fcfg.N_WORKCELLS)
    )
    target = info["target_workcell"]
    lines = info.get("lines_running", [info["belt_running"]] * fcfg.N_LINES)
    belt = " ".join(f"L{k}:{'RUN' if r else 'STOP'}" for k, r in enumerate(lines))
    return (f"{belt} [{info['line_state']}] || {cells} || "
            f"call={'none' if target < 0 else f'st{target}'}")


CAMERA_DISTANCE = 6.4
CAMERA_AZIMUTH = 143.0
CAMERA_ELEVATION = -27.0


def _camera_lookat(config) -> list[float]:
    """Centre on the work area between the two lines, so the corridor, both
    belts and both tables fit in frame."""
    lines = config.workcells
    centre_x = float(np.mean([p.manipulation_xy[0] for p in lines]))
    centre_y = float(np.mean([p.manipulation_xy[1] for p in lines])) * 0.6
    return [centre_x, centre_y, 0.85]


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
                (1.0, 0.65, 0.05, 1.0) if arm.faulted and env.line_state.name != "FAULT_RAISED"
                else fcfg.BEACON_FAULT_RGBA if arm.faulted else fcfg.BEACON_RUNNING_RGBA
            )


def _g1_status(env, walk) -> str:
    """What the G1 is actually doing, so the panel cannot go stale."""
    if walk is None:
        return "standing (no walk requested)"
    walker, navigator, _ = walk
    result = env.navigation_result()
    if walker.holding:
        return (f"arrived at station {navigator_station(navigator, env)} "
                f"({result['final_position_error_m'] * 1000:.0f} mm) -- standing hold")
    return f"walking to station {navigator_station(navigator, env)}"


def navigator_station(navigator, env) -> int:
    goal = np.asarray(navigator.goal_xy)
    for pose in env.poses:
        if float(np.linalg.norm(goal - np.asarray(pose.manipulation_xy))) < 1e-6:
            return pose.index
    return -1


def _panel(env, info, paused=False, walk=None) -> tuple[str, str]:
    target = info["target_workcell"]
    lines = info.get("lines_running", [info["belt_running"]] * fcfg.N_LINES)
    labels = ["FACTORY SUPERVISOR", "Simulation", "Line", "Belts", "Request",
              "Verification", "G1", "Line 0 (belt faces corridor)",
              "Line 1 (table faces corridor)", "Controls", "Manual signals"]
    values = [f"t={env.data.time:.1f}s", "PAUSED" if paused else "RUNNING",
              info["line_state"],
              "  ".join(f"L{k} {'ON' if r else 'STOPPED'}" for k, r in enumerate(lines)),
              "none" if target < 0 else f"Station {target}",
              f"{info['recovery_progress']:.0%}", _g1_status(env, walk),
              *[f"{'FAULT' if info['arm_faulted'][k] else 'CYCLE'} / step {info['arm_waypoint'][k]}"
                for k in range(fcfg.N_LINES)], "SPACE pause | R reset | 0/1/2 cameras",
              "A accept | C request verification (no physical recovery)"]
    if info["mission_events"]:
        labels.append("Last event")
        values.append(info["mission_events"][-1]["event"])
    labels.extend(["Scenario", "Detection"])
    values.extend([info["scenario"], info["scenario_status"]])
    if 'recovery_state' in info:
        values[labels.index('G1')] = info['recovery_state'] + (
            f" / {info['grasp_state']}" if info.get('grasp_state') and info['recovery_state'] == 'GRASP' else '')
        labels.extend(['Best lift hold', 'Failure'])
        values.extend([f"{info['lift_hold_steps'] * env.model.opt.timestep * env.config.frame_skip:.2f}s",
                       str(info.get('recovery_failure') or '-')])
    if 'viewer_rtf' in info:
        labels.append('RTF (last 100 steps)')
        values.append(f"{info['viewer_rtf']:.2f}x (includes rendering)")
    if info.get('carry_error_m') is not None:
        labels.append('Carry error')
        values.append(f"{info['carry_error_m'] * 1000:.0f} mm to the placing stance")
    if 'place_stage' in info:
        labels.extend(['Place', 'Place error'])
        values.extend([info['place_stage'], f"{info['place_error_m'] * 1000:.1f} mm"])
    return "\n".join(labels), "\n".join(values)


def run_offscreen(env, args) -> None:
    import mujoco
    from PIL import Image, ImageDraw

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _, info = env.reset(seed=args.seed, options=_reset_options(args))
    camera = _camera(env.factory)
    if getattr(args, "camera", "overview") != "overview":
        # Same framing the interactive 1/2 keys use, so before/after close-ups
        # are comparable with what the window shows.
        pose = env.poses[int(args.camera[-1])]
        camera.lookat[:] = [*pose.canonical_part_xy, 0.95]
        camera.distance = 2.3
    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
    bundle = _make_walker(env, args.walk_to) if args.walk_to is not None else None
    recovery = _make_recovery(env, args)
    announced: dict = {}
    shots = {}
    for step in range(args.steps):
        if bundle is not None:
            _drive_walker(env, bundle, announced)
        info = recovery.step() if recovery is not None else env.step(zero)[4]
        _update_beacons(env)
        if step + 1 in args.capture:
            shots[step + 1] = info
            with mujoco.Renderer(env.model, height=720, width=1280) as renderer:
                renderer.update_scene(env.data, camera)
                path = out if len(args.capture) == 1 else out.with_name(f"{out.stem}_{step + 1:04d}{out.suffix}")
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                labels, values = _panel(env, info, walk=bundle)
                draw.rectangle((8, 8, 700, 240), fill=(20, 24, 30))
                draw.multiline_text((16, 16), labels, fill="white", spacing=3)
                draw.multiline_text((160, 16), values, fill="white", spacing=3)
                frame.save(path)
                print(f"step {step + 1:5d}  {_describe(env, info)}  -> {path}")
    if not shots:
        print("no frames captured; check --capture against --steps")
    if recovery is not None:
        print(f"Recovery result: {recovery.state}; failure={recovery.failure}")
        recovery.close()


def run_interactive(env, args) -> None:
    import mujoco.viewer

    _, info = env.reset(seed=args.seed, options=_reset_options(args))
    zero = np.zeros(env.action_space.shape[0], dtype=np.float32)
    print(f"Factory viewer: layout={env.factory.layout}, seed={args.seed}, "
          f"scenario={env.scenario}, station={env.fault_workcell}, eligible after step {env.fault_step}.")
    print("Green = producing, red = requested, amber = recovery/verification.")
    print("SPACE pause | R reset | 0 overview | 1/2 line camera | A accept | C verify.")
    print("A/C are manual task-manager messages only: they do not move G1 or restore the part.")
    if getattr(args, 'recover', False):
        print('Experimental live recovery: fault -> prepare hands -> walk -> grasp/lift.')
        print(f"Approach motion: {args.recovery_motion}")
        if getattr(args, 'place', False):
            print('PLACE PROTOTYPE: fault -> walk -> lift -> '
                  + ('carry (walking) -> ' if getattr(args, 'carry', True) else '')
                  + 'lower -> release -> retract -> restart.')
    elif args.walk_to is None:
        print("The G1 stands still; pass --walk-to 0 or --walk-to 1 to make it walk there.")
    print("Close the viewer to exit.")
    target_dt = env.model.opt.timestep * env.config.frame_skip
    keys = SimpleQueue()
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=keys.put) as viewer:
        viewer.cam.lookat[:] = _camera_lookat(env.factory)
        viewer.cam.distance = CAMERA_DISTANCE
        viewer.cam.azimuth = CAMERA_AZIMUTH
        viewer.cam.elevation = CAMERA_ELEVATION
        paused = False
        finished = False
        event_count = 0
        bundle = _make_walker(env, args.walk_to) if args.walk_to is not None else None
        recovery = _make_recovery(env, args)
        walk_announced: dict = {}
        rtf_samples = deque(maxlen=100)
        while viewer.is_running():
            started = time.time()
            wall_started = time.perf_counter()
            sim_started = env.data.time
            signal = None
            while not keys.empty():
                key = keys.get()
                if key == 32 and not finished:
                    paused = not paused
                elif key == ord("R"):
                    if recovery is not None:
                        recovery.close()
                    _, info = env.reset(seed=args.seed, options=_reset_options(args))
                    rtf_samples.clear()
                    sim_started = env.data.time
                    paused = finished = False
                    event_count = 0
                    signal = None
                    # Rebuild the walker: reset restored the stand pose, and the
                    # old one still holds the previous run's leg targets.
                    bundle = _make_walker(env, args.walk_to) if args.walk_to is not None else None
                    walk_announced = {}
                    recovery = _make_recovery(env, args)
                elif key in (ord("A"), ord("C")) and recovery is None:
                    signal = ("accept" if key == ord("A") else "complete", info["target_workcell"])
                elif key in (ord("0"), ord("1"), ord("2")):
                    with viewer.lock():
                        if key == ord("0"):
                            viewer.cam.lookat[:] = _camera_lookat(env.factory)
                            viewer.cam.distance = CAMERA_DISTANCE
                        else:
                            pose = env.poses[key - ord("1")]
                            viewer.cam.lookat[:] = [*pose.canonical_part_xy, 0.95]
                            viewer.cam.distance = 2.3
            if not paused:
                if bundle is not None:
                    _drive_walker(env, bundle, walk_announced)
                if recovery is not None:
                    info = recovery.step()
                    terminated = recovery.state in recovery.TERMINAL
                    truncated = False
                else:
                    _, _, terminated, truncated, info = env.step(zero, supervisor_signal=signal)
                if signal is not None:
                    print(f"Manual signal {signal}: accepted={info['supervisor_signal_accepted']}")
                if terminated or truncated:
                    paused = finished = True
                    print("Episode finished; scene retained. Press R to reset.")
                    if recovery is not None:
                        print(f"Recovery result: {recovery.state}; failure={recovery.failure}")
            for event in info["mission_events"][event_count:]:
                print(f"  step {event['step']:5d} station {event['station']}: {event['event']}")
            event_count = len(info["mission_events"])
            with viewer.lock():
                _update_beacons(env)
            if rtf_samples:
                info['viewer_rtf'] = sum(s for s, _ in rtf_samples) / sum(w for _, w in rtf_samples)
            viewer.set_texts((None, None, *_panel(env, info, paused, walk=bundle)))
            viewer.sync()
            remaining = target_dt - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)
            sim_delta = env.data.time - sim_started
            if sim_delta > 0:
                rtf_samples.append((sim_delta, time.perf_counter() - wall_started))
        if recovery is not None:
            recovery.close()


def _make_recovery(env, args):
    if not getattr(args, 'recover', False):
        return None
    from humanoid_learning.expert.factory_recovery import FactoryRecovery, RecoveryConfig
    return FactoryRecovery(env, RecoveryConfig(
        stand_off_m=getattr(args, 'recovery_stand_off', 0.27),
        kinematic_ik=getattr(args, 'kinematic_ik', True),
        place_after_lift=getattr(args, 'place', False),
        carry_by_walking=getattr(args, 'carry', True),
        motion_profile=getattr(args, 'recovery_motion', 'baseline')))


def _make_walker(env, station: int):
    """Walk the G1 to a station and hand off to a standing hold on arrival."""
    from humanoid_learning.expert.g1_walk_policy import G1WalkPolicy, WalkToPose
    from humanoid_learning.expert.stance_stabilizer import StanceGains, StanceStabilizer

    walker = G1WalkPolicy(env)
    pose = env.poses[station]
    navigator = WalkToPose(walker, pose.manipulation_xy, pose.heading_rad)
    stabilizer = StanceStabilizer(env, StanceGains(pitch_kp=1.0, pitch_kd=0.1,
                                                   roll_kp=0.7, roll_kd=0.07))
    print(f"Walking to station {station} at {np.round(pose.manipulation_xy, 3)}. "
          "Locomotion is Unitree's pre-trained G1 policy (BSD-3); see "
          "assets/policies/g1_walk/NOTICE.")
    return walker, navigator, stabilizer


def _drive_walker(env, bundle, announced: dict) -> None:
    walker, navigator, _stabilizer = bundle
    navigator.step()
    if walker.holding:
        # No ankle regulator here: the post-handoff stance is stiff-held and the
        # regulator is tuned for the compliant grasp plant. It is engaged when
        # manipulation starts, not while merely standing.
        if not announced.get("arrived"):
            announced["arrived"] = True
            result = env.navigation_result()
            print(f"  ARRIVED: position error {result['final_position_error_m'] * 1000:.1f} mm, "
                  f"heading error {np.degrees(result['final_heading_error_rad']):.1f} deg "
                  "-- handed off to a standing hold.")


def _reset_options(args) -> dict:
    options = {} if args.scenario is None else {"scenario": args.scenario}
    if args.fault_workcell is not None:
        options["fault_workcell"] = args.fault_workcell
    elif getattr(args, "walk_to", None) is not None:
        # "--walk-to 0" means "go deal with station 0", so fault that station.
        # Otherwise the seed may fault the other one and the Navigation Gate
        # would score the walk against a station the robot was never sent to.
        options["fault_workcell"] = args.walk_to
    if args.fault_step is not None:
        options["fault_step"] = args.fault_step
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", choices=sorted(fcfg.LAYOUT_PRESETS), default=fcfg.DEFAULT_LAYOUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenario", choices=list(fcfg.SCENARIOS), default=None,
                        help="default: jam on line 0, arm_drop on line 1")
    parser.add_argument("--no-fault", action="store_true", help="normal production only")
    parser.add_argument("--fault-workcell", type=int, choices=range(fcfg.N_WORKCELLS), default=None,
                        help="override the seed's choice of which cell fails")
    parser.add_argument("--fault-step", type=int, default=None,
                        help="earliest eligible step; fault requires the physical event")
    parser.add_argument("--walk-to", type=int, choices=range(fcfg.N_WORKCELLS), default=None,
                        help="walk the G1 from home to this station using Unitree's pre-trained "
                             "G1 locomotion policy, then hold a standing pose there")
    parser.add_argument('--recover', action='store_true',
                        help='experimental live fault-to-lift controller; does not yet place/restart')
    parser.add_argument('--recovery-stand-off', type=float, default=0.27)
    parser.add_argument('--place', action='store_true', help='experimental place/restart after --recover lift')
    parser.add_argument('--carry', action=argparse.BooleanOptionalAction, default=True,
                        help='with --place, walk the held part to the canonical spot before lowering')
    parser.add_argument('--kinematic-ik', action=argparse.BooleanOptionalAction, default=True,
                        help='skip unused dynamics in IK scratch data only; live physics is unchanged')
    parser.add_argument('--recovery-motion', choices=('baseline', 'compact', 'direct', 'smooth'), default='smooth',
                        help='smooth uses continuous approach targets; direct preserves the previous waypoint approach')
    parser.add_argument("--offscreen", action="store_true", help="render PNGs instead of opening a window")
    parser.add_argument("--out", default="results/factory/factory.png")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--camera", choices=("overview", "line0", "line1"), default="overview",
                        help="offscreen viewpoint; the line cameras match the interactive 1/2 keys")
    parser.add_argument("--capture", type=int, nargs="+", default=[150, 400],
                        help="steps at which to save a frame in --offscreen mode")
    args = parser.parse_args()
    if args.recover and (args.walk_to is not None or args.no_fault):
        parser.error('--recover cannot be combined with --walk-to or --no-fault')
    if args.place and not args.recover:
        parser.error('--place requires --recover')

    if args.offscreen:
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        # Match scripts/view_whole_body.py exactly. On this project's PRIME /
        # Optimus laptop BOTH variables are required: setting the NVIDIA GLX
        # vendor without also enabling render offload makes GLX context
        # creation fail with "BadValue (integer parameter out of range)".
        # MUJOCO_GL is deliberately left unset here so MuJoCo picks its own
        # windowed backend (glfw) -- "glx" is not a valid MUJOCO_GL value.
        os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

    from humanoid_learning.envs.factory_env import FactoryEnv

    config = fcfg.FactoryConfig(layout=args.layout)
    config.fault.enabled = not args.no_fault
    env = FactoryEnv(config)
    try:
        (run_offscreen if args.offscreen else run_interactive)(env, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
