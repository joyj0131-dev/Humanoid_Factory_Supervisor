"""Offscreen visual evidence for the Functional Orientation fix (Section
11/16 of the audit-session spec). Same camera pattern as
scripts/capture_side_grasp_screenshots.py (mujoco.Renderer, offscreen,
verified headless -- DISPLAY was available this session but this script
still uses offscreen rendering for a reproducible, disclosed record; a
human has not visually reviewed these images, only this script's own
automated run).

[Audit session] Extended past the prior session's "after_descend_blocked"
(FOREARM_SIDE_DESCEND's torso-arm collision is fixed this session -- see
sharpa_bimanual_grasp_expert.py's module docstring) to actually reach
FINGERTIP_PRECONTACT, and adds a genuine BEFORE/AFTER pair: "before_*"
renders use the AUDITED, improper weight pair (0.95/0.05, the one a prior
commit shipped and this session reverted) via a monkeypatched
_object_facing_R, so the images show the actual regression this session
fixed, not just the final state. Small-curl frames mark each nonthumb
fingertip's open vs curled position with a colored sphere + connecting
line (yellow=open, cyan=curled), giving a real before/after displacement
visual, not just a numeric claim.

Captures:
    results/functional_orientation/before_open_{front,top}.png
    results/functional_orientation/before_small_curl_{front,top}.png
    results/functional_orientation/after_orientation_open_{front,top}.png
    results/functional_orientation/after_orientation_small_curl_{front,top}.png
    results/functional_orientation/after_preshape_{front,top}.png
    results/functional_orientation/after_precontact_{front,top}.png

Images are NOT committed (see .gitignore).

Usage:
    MUJOCO_GL=egl PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/capture_functional_orientation_evidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
from PIL import Image

from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv, ACTION_DIM
import humanoid_learning.expert.sharpa_bimanual_grasp_expert as sbe_module
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    SIDES,
    BimanualGraspState,
    SharpaBimanualGraspExpert,
    _wahba_R,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "functional_orientation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NONTHUMB = ("index", "middle", "ring", "pinky")


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=3000, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def _camera(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def _add_marker(scene, pos, rgba, size=0.008) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]), np.asarray(pos), np.eye(3).flatten(), np.array(rgba))
    scene.ngeom += 1


def _add_line(scene, start, end, rgba, width=0.003) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(start), np.asarray(end))
    g.rgba[:] = rgba
    scene.ngeom += 1


def snapshot(renderer, data, cam, path: Path, markers=None) -> None:
    renderer.update_scene(data, camera=cam)
    if markers:
        for kind, args in markers:
            if kind == "marker":
                _add_marker(renderer.scene, *args)
            else:
                _add_line(renderer.scene, *args)
    img = renderer.render()
    Image.fromarray(img).save(path)
    print(f"  wrote {path}")


def old_improper_object_facing_R(side, palm_pos, obj_pos):
    """[Audit session] The AUDITED, improper weight pair (0.95/0.05) a
    prior commit shipped -- used only for the "before" render pair, to
    show the actual regression this session fixed."""
    to_obj = obj_pos - palm_pos
    n = np.linalg.norm(to_obj)
    to_obj = to_obj / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    horiz = to_obj.copy()
    horiz[2] = 0.0
    hn = np.linalg.norm(horiz)
    horiz = horiz / hn if hn > 1e-9 else np.zeros(3)
    down = 0.82 * np.array([0.0, 0.0, -1.0]) + 0.18 * horiz
    down = down / np.linalg.norm(down)
    return _wahba_R([sbe_module.LOCAL_CLOSING_VEC, sbe_module.LOCAL_APPROACH_VEC], [to_obj, down], weights=[0.95, 0.05])


def apply_small_curl(env: SharpaGraspEnv, curl_amount: float = 0.15) -> None:
    for side_idx in (0, 1):
        steps = max(1, int(np.ceil(curl_amount / env.config.hand_synergy_action_scale)))
        for _ in range(steps):
            a = np.zeros(ACTION_DIM)
            for g in (1, 2, 3):
                a[17 + side_idx * 4 + g] = 1.0
            env.step(a)


def _tip_markers(env: SharpaGraspEnv, tips_open: dict) -> list:
    markers = []
    for side in SIDES:
        for f in NONTHUMB:
            open_pos = tips_open[side][f]
            curl_pos = env.fingertip_pos(side, f)
            markers.append(("marker", (open_pos, (0.95, 0.9, 0.1, 1.0))))
            markers.append(("marker", (curl_pos, (0.1, 0.85, 0.95, 1.0))))
            markers.append(("line", (open_pos, curl_pos, (0.9, 0.6, 0.1, 1.0))))
    return markers


def render_pair(tag_prefix: str, object_facing_R_fn) -> None:
    saved = sbe_module._object_facing_R
    sbe_module._object_facing_R = object_facing_R_fn
    try:
        env = make_env()
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        obj_lookat = np.array([0.27, 0.0, 0.85])
        cam_front = _camera(obj_lookat, distance=1.1, azimuth=0, elevation=-12)
        cam_top = _camera(obj_lookat, distance=1.0, azimuth=0, elevation=-65)
        expert = SharpaBimanualGraspExpert(env)
        for _ in range(1500):
            if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE, BimanualGraspState.FIVE_FINGER_PRESHAPE):
                break
            action = expert.step()
            env.step(action)
        if expert.state != BimanualGraspState.FIVE_FINGER_PRESHAPE:
            print(f"  {tag_prefix}: did not reach WRIST_SIDE_GRASP_ALIGN convergence ({expert.state.name}, {expert.failure_reason}) -- skipping")
            return
        snapshot(renderer, env.data, cam_front, RESULTS_DIR / f"{tag_prefix}_open_front.png")
        snapshot(renderer, env.data, cam_top, RESULTS_DIR / f"{tag_prefix}_open_top.png")
        tips_open = {side: {f: env.fingertip_pos(side, f).copy() for f in NONTHUMB} for side in SIDES}
        apply_small_curl(env, 0.15)
        markers = _tip_markers(env, tips_open)
        snapshot(renderer, env.data, cam_front, RESULTS_DIR / f"{tag_prefix}_small_curl_front.png", markers)
        snapshot(renderer, env.data, cam_top, RESULTS_DIR / f"{tag_prefix}_small_curl_top.png", markers)
    finally:
        sbe_module._object_facing_R = saved


def main() -> None:
    print("Rendering BEFORE pair (audited, improper 0.95/0.05 weight)...")
    render_pair("before", old_improper_object_facing_R)

    print("Rendering AFTER pair (this session's fixed 0.68/0.32 weight)...")
    render_pair("after_orientation", sbe_module._object_facing_R)

    print("Rendering full-rollout checkpoints (after_preshape, after_precontact) under the fixed default...")
    env = make_env()
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    obj_lookat = np.array([0.27, 0.0, 0.85])
    cam_front = _camera(obj_lookat, distance=1.1, azimuth=0, elevation=-12)
    cam_top = _camera(obj_lookat, distance=1.0, azimuth=0, elevation=-65)
    expert = SharpaBimanualGraspExpert(env)
    captured: set[str] = set()

    def maybe_capture(tag: str) -> None:
        if tag not in captured:
            snapshot(renderer, env.data, cam_front, RESULTS_DIR / f"{tag}_front.png")
            snapshot(renderer, env.data, cam_top, RESULTS_DIR / f"{tag}_top.png")
            captured.add(tag)

    for _ in range(2000):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        prev_state = expert.state
        action = expert.step()
        env.step(action)
        if prev_state == BimanualGraspState.FIVE_FINGER_PRESHAPE and expert.state == BimanualGraspState.FOREARM_SIDE_DESCEND:
            maybe_capture("after_preshape")
        if expert.state == BimanualGraspState.FINGERTIP_PRECONTACT and "after_precontact" not in captured:
            maybe_capture("after_precontact")

    if "after_precontact" not in captured:
        maybe_capture("after_precontact")  # honest final pose even if FINGERTIP_PRECONTACT itself later fails
    print(f"\nfinal state={expert.state.name} reason={expert.failure_reason}")
    print(f"captured: before, after_orientation, {sorted(captured)}")


if __name__ == "__main__":
    main()
