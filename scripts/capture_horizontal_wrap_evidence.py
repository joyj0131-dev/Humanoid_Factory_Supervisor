"""Offscreen before/after render evidence for the Horizontal-Wrap
posture session: BEFORE = old, retracted finger-down orientation
construction (fingers forced toward world -Z); AFTER = new
_object_facing_R (palm-inward + finger-horizontal + ulnar-edge-down),
same camera, same object/seed.

Offscreen (mujoco.Renderer, MUJOCO_GL=egl) -- this is a headless
verification, not a substitute for a human directly watching
scripts/view_whole_body.py --grasp --show-hand-axes --no-restart.

Usage:
    MUJOCO_GL=egl PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/capture_horizontal_wrap_evidence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
from PIL import Image

import humanoid_learning.expert.sharpa_bimanual_grasp_expert as sbe_module
from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    BimanualGraspState, SharpaBimanualGraspExpert,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "horizontal_wrap_evidence"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=4000, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def _camera(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def snapshot(renderer, data, cam, path: Path) -> None:
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    Image.fromarray(img).save(path)
    print(f"  wrote {path}")


def old_object_facing_R(side, palm_pos, obj_pos, current_approach_world=None, weights=None):
    to_obj = obj_pos - palm_pos
    n = np.linalg.norm(to_obj)
    to_obj = to_obj / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    horiz = to_obj.copy()
    horiz[2] = 0.0
    hn = np.linalg.norm(horiz)
    horiz = horiz / hn if hn > 1e-9 else np.zeros(3)
    inward_weight = 0.60
    down = (1 - inward_weight) * np.array([0.0, 0.0, -1.0]) + inward_weight * horiz
    down = down / np.linalg.norm(down)
    return sbe_module._wahba_R(
        [sbe_module.LOCAL_CLOSING_VEC, sbe_module.LOCAL_APPROACH_VEC], [to_obj, down], weights=[0.68, 0.32]
    )


def run_to_state(target_states, use_old: bool, max_steps=3000, extra_settle_ticks: int = 0):
    saved = sbe_module._object_facing_R
    if use_old:
        sbe_module._object_facing_R = old_object_facing_R
    try:
        env = make_env()
        expert = SharpaBimanualGraspExpert(env)
        env.reset(seed=0)
        for _ in range(max_steps):
            if expert.state in target_states or expert.state == BimanualGraspState.FAILURE:
                break
            action = expert.step()
            env.step(action)
        for _ in range(extra_settle_ticks):
            if expert.state == BimanualGraspState.FAILURE:
                break
            action = expert.step()
            env.step(action)
        return env, expert
    finally:
        sbe_module._object_facing_R = saved


def main() -> None:
    obj_lookat = np.array([0.27, 0.0, 0.85])
    cam_front = _camera(obj_lookat, distance=1.0, azimuth=90, elevation=-10)
    cam_top = _camera(obj_lookat, distance=1.0, azimuth=90, elevation=-75)
    cam_diag = _camera(obj_lookat, distance=1.0, azimuth=45, elevation=-30)

    runs = [
        ("before_finger_down_align", True, {BimanualGraspState.FIVE_FINGER_PRESHAPE}, 0),
        ("after_horizontal_wrap_align", False, {BimanualGraspState.FIVE_FINGER_PRESHAPE}, 0),
        ("after_preshape", False, {BimanualGraspState.FOREARM_SIDE_DESCEND}, 0),
        ("after_side_descend", False, {BimanualGraspState.FINGERTIP_PRECONTACT}, 0),
        ("after_precontact_settled", False, {BimanualGraspState.FINGERTIP_PRECONTACT}, 150),
    ]
    for label, use_old, targets, extra in runs:
        env, expert = run_to_state(targets, use_old, extra_settle_ticks=extra)
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        print(f"{label}: final_state={expert.state.name} reason={expert.failure_reason}")
        snapshot(renderer, env.data, cam_front, RESULTS_DIR / f"{label}_front.png")
        snapshot(renderer, env.data, cam_top, RESULTS_DIR / f"{label}_top.png")
        snapshot(renderer, env.data, cam_diag, RESULTS_DIR / f"{label}_diag.png")
        renderer.close()


if __name__ == "__main__":
    main()
