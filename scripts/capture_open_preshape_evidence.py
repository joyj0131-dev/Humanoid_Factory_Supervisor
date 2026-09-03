"""Offscreen before/after render evidence for the open-preshape session:
BEFORE = old baseline (curl=0.95, height=0.03); AFTER = new default
(curl=0.7, height=0.05), same camera. Not committed (see .gitignore).

Offscreen (mujoco.Renderer, MUJOCO_GL=egl) -- this is a headless
verification, not a substitute for a human directly watching
scripts/view_whole_body.py --grasp --show-hand-axes --no-restart.

Usage:
    MUJOCO_GL=egl PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/capture_open_preshape_evidence.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
from PIL import Image

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    BimanualGraspConfig, BimanualGraspState, SharpaBimanualGraspExpert,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "open_preshape_evidence"
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


def run_to_state(cfg: BimanualGraspConfig, target_states, max_steps=3000):
    env = make_env()
    expert = SharpaBimanualGraspExpert(env, cfg)
    env.reset(seed=0)
    for _ in range(max_steps):
        if expert.state in target_states or expert.state == BimanualGraspState.FAILURE:
            break
        action = expert.step()
        env.step(action)
    return env, expert


def main() -> None:
    obj_lookat = np.array([0.27, 0.0, 0.85])
    cam_front = _camera(obj_lookat, distance=1.0, azimuth=90, elevation=-10)
    cam_top = _camera(obj_lookat, distance=1.0, azimuth=90, elevation=-60)

    for label, cfg in [
        ("before_curl095", replace(BimanualGraspConfig(), side_descend_height_m=0.03, side_descend_curl_target=0.95,
                                    precontact_height_m=0.03)),
        ("after_open_preshape", BimanualGraspConfig()),
    ]:
        env, expert = run_to_state(cfg, {BimanualGraspState.FINGERTIP_PRECONTACT})
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        print(f"{label}: final_state={expert.state.name} reason={expert.failure_reason}")
        snapshot(renderer, env.data, cam_front, RESULTS_DIR / f"{label}_front.png")
        snapshot(renderer, env.data, cam_top, RESULTS_DIR / f"{label}_top.png")
        renderer.close()


if __name__ == "__main__":
    main()
