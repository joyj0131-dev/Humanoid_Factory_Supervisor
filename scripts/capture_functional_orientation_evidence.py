"""[This session] Offscreen visual evidence for the Functional Orientation
fix (Section 11/16 of this session's spec). Same camera pattern as
scripts/capture_side_grasp_screenshots.py (mujoco.Renderer, offscreen,
verified headless -- DISPLAY was available this session but this script
still uses offscreen rendering for a reproducible, disclosed record).

Captures, at WRIST_SIDE_GRASP_ALIGN's converged pose (the Functional
Orientation Gate's own evaluation point):
    results/functional_orientation/after_orientation_open_front.png
    results/functional_orientation/after_orientation_open_top.png
    results/functional_orientation/after_orientation_small_curl_front.png
    results/functional_orientation/after_orientation_small_curl_top.png
    results/functional_orientation/after_preshape_front.png
    results/functional_orientation/after_preshape_top.png
    results/functional_orientation/after_descend_blocked_front.png
    results/functional_orientation/after_descend_blocked_top.png

The "small_curl" pair is captured AFTER applying the same curl_probe
(0.15) _measure_functional_orientation itself uses, so before/after
fingertip motion toward the object is visible (curl is left applied in
this render only -- a fresh env is used, not the measurement's own
state-preserving call).

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
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    SIDES,
    BimanualGraspState,
    SharpaBimanualGraspExpert,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "functional_orientation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def snapshot(renderer, data, cam, path: Path) -> None:
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    Image.fromarray(img).save(path)
    print(f"  wrote {path}")


def apply_small_curl(env: SharpaGraspEnv, curl_amount: float = 0.15) -> None:
    for side_idx in (0, 1):
        steps = max(1, int(np.ceil(curl_amount / env.config.hand_synergy_action_scale)))
        for _ in range(steps):
            a = np.zeros(ACTION_DIM)
            for g in (1, 2, 3):
                a[17 + side_idx * 4 + g] = 1.0
            env.step(a)


def main() -> None:
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

    for _ in range(1500):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        prev_state = expert.state
        action = expert.step()
        env.step(action)
        if prev_state == BimanualGraspState.WRIST_SIDE_GRASP_ALIGN and expert.state == BimanualGraspState.FIVE_FINGER_PRESHAPE:
            maybe_capture("after_orientation_open")
            for side in SIDES:
                pos, R = env.palm_pose(side)
                print(f"    {side} palm pos={pos.round(3)} approach(col0)={R[:,0].round(3)}")
            apply_small_curl(env, 0.15)
            maybe_capture("after_orientation_small_curl")
        if prev_state == BimanualGraspState.FIVE_FINGER_PRESHAPE and expert.state == BimanualGraspState.FOREARM_SIDE_DESCEND:
            maybe_capture("after_preshape")
        if expert.state == BimanualGraspState.FAILURE:
            break

    maybe_capture("after_descend_blocked")
    print(f"\nfinal state={expert.state.name} reason={expert.failure_reason}")
    print(f"captured: {sorted(captured)}")


if __name__ == "__main__":
    main()
