"""Offscreen-render evidence for the Sharpa side-grasp posture change
(this session, Section 11 of the mandate): same camera, same object,
captured at matched rollout points, before/after the geometry fix.

NOT a live GUI capture -- this uses mujoco.Renderer (offscreen), verified
headless. If a real GUI session is available, run
    DISPLAY=:0 python3 scripts/view_whole_body.py --grasp --no-restart
separately to confirm visually; this script's images are the disclosed,
reproducible record.

"Before" = the shared FOREARM_FORWARD_REACH end pose (position-only,
natural clearance-derived orientation) -- the SAME pose the pre-session
FOREARM_DESCEND used to carry forward unchanged into a palm-down reach;
this session's WRIST_SIDE_GRASP_ALIGN starts from exactly this point, so
it is not a reconstruction of deleted code, it is the actual shared
hand-off state.

Images (not committed -- see .gitignore):
    results/side_grasp_pose/before_front.png
    results/side_grasp_pose/before_top.png
    results/side_grasp_pose/after_align_front.png
    results/side_grasp_pose/after_align_top.png
    results/side_grasp_pose/after_preshape_front.png
    results/side_grasp_pose/after_preshape_top.png
    results/side_grasp_pose/after_descend_blocked_front.png
    results/side_grasp_pose/after_descend_blocked_top.png

The last pair is honestly labeled "blocked", not "precontact" --
FINGERTIP_PRECONTACT is not yet reached this session (see
test_forward_reach_gate_now_passes_and_advances_to_next_blocker's
HAND_TABLE_COLLISION assertion).

Usage:
    MUJOCO_GL=egl PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/capture_side_grasp_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np
from PIL import Image

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    BimanualGraspState,
    SharpaBimanualGraspExpert,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "side_grasp_pose"
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


def snapshot(renderer: mujoco.Renderer, data: mujoco.MjData, cam: mujoco.MjvCamera, path: Path) -> None:
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    Image.fromarray(img).save(path)
    print(f"  wrote {path}")


def main() -> None:
    env = make_env()
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    obj_lookat = np.array([0.27, 0.0, 0.85])
    cam_front = _camera(obj_lookat, distance=1.1, azimuth=90, elevation=-10)
    cam_top = _camera(obj_lookat, distance=1.1, azimuth=90, elevation=-55)

    expert = SharpaBimanualGraspExpert(env)
    captured: set[str] = set()

    def maybe_capture(tag: str, cond: bool) -> None:
        if cond and tag not in captured:
            snapshot(renderer, env.data, cam_front, RESULTS_DIR / f"{tag}_front.png")
            snapshot(renderer, env.data, cam_top, RESULTS_DIR / f"{tag}_top.png")
            captured.add(tag)

    for _ in range(1500):
        if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
            break
        prev_state = expert.state
        action = expert.step()
        env.step(action)
        if prev_state == BimanualGraspState.FOREARM_FORWARD_REACH and expert.state == BimanualGraspState.WRIST_SIDE_GRASP_ALIGN:
            maybe_capture("before", True)
        if prev_state == BimanualGraspState.WRIST_SIDE_GRASP_ALIGN and expert.state == BimanualGraspState.FIVE_FINGER_PRESHAPE:
            maybe_capture("after_align", True)
        if prev_state == BimanualGraspState.FIVE_FINGER_PRESHAPE and expert.state == BimanualGraspState.FOREARM_SIDE_DESCEND:
            maybe_capture("after_preshape", True)
        if expert.state == BimanualGraspState.FAILURE:
            break

    # Final pose, whatever it is (honestly labeled "blocked" -- see module docstring).
    maybe_capture("after_descend_blocked", True)

    print(f"\nfinal state={expert.state.name} reason={expert.failure_reason}")
    print(f"captured: {sorted(captured)}")


if __name__ == "__main__":
    main()
