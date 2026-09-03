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


def _hand_axis_segments(env, expert) -> list[tuple[np.ndarray, np.ndarray, tuple, str]]:
    """[Audit session, new] Debug-visualization geometry for
    --show-hand-axes: (start, end, rgba, label) segments per side, in
    world coordinates. Colors match the spec: red=local X, green=local Y,
    blue=local Z, yellow=new inside normal (approach axis, local X --
    same convention, kept distinct from red for the "inside normal" role
    per the spec's color list), magenta=empirical closing direction,
    white=to-object direction. Uses the SAME real functions the
    controller/Gates use (env.palm_pose, sharpa_bimanual_grasp_expert's
    LOCAL_CLOSING_VEC) -- not a separate, re-derived visualization-only
    computation."""
    import humanoid_learning.expert.sharpa_bimanual_grasp_expert as sbe
    from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SIDES

    segs: list[tuple[np.ndarray, np.ndarray, tuple, str]] = []
    obj_pos = expert._object_pos() if hasattr(expert, "_object_pos") else None
    axis_len = 0.05
    long_len = 0.08
    for side in SIDES:
        palm_pos, palm_R = env.palm_pose(side)
        colors = {
            "X": (0.85, 0.1, 0.1, 1.0), "Y": (0.1, 0.75, 0.1, 1.0), "Z": (0.1, 0.1, 0.85, 1.0),
        }
        for i, axis_name in enumerate(("X", "Y", "Z")):
            end = palm_pos + palm_R[:, i] * axis_len
            segs.append((palm_pos, end, colors[axis_name], f"{side}_local_{axis_name}"))
        inside_normal = palm_R @ sbe.LOCAL_APPROACH_VEC
        segs.append((palm_pos, palm_pos + inside_normal * long_len, (0.9, 0.9, 0.0, 1.0), f"{side}_inside_normal"))
        closing_dir = palm_R @ sbe.LOCAL_CLOSING_VEC
        segs.append((palm_pos, palm_pos + closing_dir * long_len, (0.85, 0.1, 0.85, 1.0), f"{side}_empirical_closing"))
        if obj_pos is not None:
            to_obj = obj_pos - palm_pos
            n = np.linalg.norm(to_obj)
            if n > 1e-6:
                to_obj = to_obj / n
                segs.append((palm_pos, palm_pos + to_obj * long_len, (0.95, 0.95, 0.95, 1.0), f"{side}_to_object"))
    return segs


def _draw_hand_axis_markers(viewer, segments) -> None:
    """Populates viewer.user_scn with one capsule geom per segment (the
    standard mujoco.viewer passive-viewer pattern for transient debug
    geometry -- cleared and rebuilt every call since user_scn.ngeom is
    reset to 0 at the top)."""
    scn = viewer.user_scn
    scn.ngeom = 0
    for start, end, rgba, _label in segments:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.004, np.asarray(start), np.asarray(end))
        g.rgba[:] = rgba
        scn.ngeom += 1


def mode_grasp(
    object_pos_x: float,
    object_half_size: float | None,
    no_restart: bool,
    mount: str,
    visual_style: str,
    show_hand_axes: bool = False,
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
    if show_hand_axes:
        print("  --show-hand-axes ON: red/green/blue=palm local XYZ, yellow=inside normal, "
              "magenta=empirical closing direction, white=to-object direction")

    def restart() -> None:
        env.reset(seed=0)
        state["expert"] = SharpaBimanualGraspExpert(env)
        state["terminal"] = False
        state["last"] = None

    def step(tick: int, viewer=None) -> None:
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
        if show_hand_axes and viewer is not None:
            _draw_hand_axis_markers(viewer, _hand_axis_segments(env, expert))
        # [Audit session] Text HUD substitute: mujoco.viewer's passive-mode
        # API has no simple on-screen text overlay, so state/force/qvel/
        # Gate status (spec Section 12) are printed to stdout periodically
        # instead of drawn in-scene.
        if show_hand_axes and tick % 60 == 0:
            import humanoid_learning.envs.sharpa_config as sc
            torso = env._torso_arm_collision_force()
            table = env._hand_table_contact_force()
            wrist_dof = np.concatenate([env._arm_dof_adr[4:7], env._arm_dof_adr[11:14]])
            wrist_qvel = float(np.max(np.abs(env.data.qvel[wrist_dof])))
            fo = getattr(expert, "_functional_orientation", {}) or {}
            obj_pos = expert._object_pos() if hasattr(expert, "_object_pos") else None
            half = env.config.object_half_size
            fingertip_table_clearance = {
                s: float(min(env.fingertip_pos(s, f)[2] for f in sc.FINGERS)) for s in ("left", "right")
            }
            fingertip_obj_sep_mm = {}
            if obj_pos is not None:
                for s in ("left", "right"):
                    tips = [env.fingertip_pos(s, f) for f in ("index", "middle", "ring", "pinky")]
                    centroid = np.mean(tips, axis=0)
                    fingertip_obj_sep_mm[s] = float(abs(centroid[1] - obj_pos[1]) - half) * 1000.0
            contact_groups = {
                s: [g for g, touched in expert._group_ever_contacted[s].items() if touched]
                for s in ("left", "right")
            }
            gate_a_now = expert._bilateral_streak >= expert.config.bilateral_streak_required
            # [Open-preshape session] Extended HUD per Section 12/17: added
            # fingertip-table Z clearance, fingertip-to-object-face
            # separation, contacted finger groups, Precontact tracking
            # error (from the controller's own live measurement, not a
            # separate re-derived one), FailureReason, and a live Gate A
            # streak/pass readout, on top of the existing torso/table/
            # qvel/Gate summary. Remaining inward closure travel is NOT
            # recomputed every tick here (that requires a state-preserving
            # extra env.step() probe, too expensive for a live viewer loop
            # at 60-tick cadence) -- see scripts/audit_sharpa_curl_table_
            # feasibility.py / candidate_open_preshape_descend.py for the
            # authoritative offline measurement of that quantity.
            print(f"  [hud] state={expert.state.name} reason={expert.failure_reason} "
                  f"torso_arm={torso:.2f}N hand_table={table:.2f}N wrist_qvel={wrist_qvel:.2f}rad/s "
                  f"fingertip_min_z={fingertip_table_clearance} fingertip_obj_sep_mm={fingertip_obj_sep_mm} "
                  f"contact_groups={contact_groups} precontact_pos_err={expert._precontact_final_pos_error} "
                  f"precontact_ori_err_deg={expert._precontact_final_ori_error_deg} "
                  f"precontact_streak={expert._precontact_stable_streak}/{expert.config.precontact_stable_streak_required} "
                  f"bilateral_streak={expert._bilateral_streak}/{expert.config.bilateral_streak_required} gate_a_now={gate_a_now} "
                  f"side_grasp_gate={expert._side_grasp_gate} functional_orientation_gate={fo.get('gate')}")

    if show_hand_axes:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            tick = 0
            while viewer.is_running():
                started = time.time()
                step(tick, viewer=viewer)
                viewer.sync()
                tick += 1
                _pace(env.model, started)
    else:
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
    parser.add_argument("--show-hand-axes", action="store_true",
                         help="--grasp only: draw palm local XYZ/inside-normal/empirical-closing/to-object "
                              "debug markers and print a periodic state/force/Gate HUD to stdout. Default OFF.")
    args = parser.parse_args()

    if args.stand:
        mode_stand()
    elif args.posture:
        mode_posture()
    elif args.planar:
        mode_planar()
    elif args.grasp:
        mode_grasp(args.object_pos_x, args.object_half_size, args.no_restart,
                   args.sharpa_mount, args.sharpa_visual_style, args.show_hand_axes)
    else:
        mode_hand_demo(args.no_restart, args.sharpa_visual_style)


if __name__ == "__main__":
    main()
