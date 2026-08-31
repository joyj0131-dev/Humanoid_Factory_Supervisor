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
    --grasp-safety-latch  the same SIZE_12 attempt with the 28th-session
               safety-event thumb latch A/B condition enabled.  This is
               a known negative experiment (tripod 14 -> 4), exposed only
               so its lower rotation but earlier contact loss can be
               inspected visually; it is not the canonical controller.
    --diagonal-feasibility  SIZE_12 diagonal four-face grasp: shows the
               current BEST STATIC candidate from humanoid_learning.expert.
               diagonal_feasibility's Stage U search as a frozen pose (not
               a live/dynamic attempt -- see PROJECT_CONTEXT.md, this
               candidate does not yet satisfy the intended fingertip-face
               assignment).

Run locally (needs a real display):
    python scripts/view_whole_body.py --stand
    python scripts/view_whole_body.py --posture
    python scripts/view_whole_body.py --planar
    python scripts/view_whole_body.py --grasp
    python scripts/view_whole_body.py --grasp-safety-latch
    python scripts/view_whole_body.py --diagonal-feasibility
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


def mode_diagonal_feasibility():
    """SIZE_12 Contact-Driven Four-Face Whole-Body Feasibility session:
    shows the CURRENT best static candidate found by humanoid_learning.
    expert.diagonal_feasibility's Stage U search (NOT a live/dynamic
    grasp attempt -- this is a frozen kinematic pose, mj_step is never
    called, so the viewer just displays the exact qpos the offline IK
    solve produced). Reruns the same coarse-to-fine search the test
    suite runs (scripts/test_diagonal_feasibility.py) so what's shown is
    always reproducible from the checked-in search code, not a
    hardcoded snapshot -- as of this session, the best candidate is
    collision-free with healthy wrist_yaw/elbow joint-limit margin (see
    PROJECT_CONTEXT.md for the waist_weight finding behind that), but
    does NOT yet satisfy the intended fingertip-face assignment -- the
    printed diagnostics below say so explicitly; this is a known-
    incomplete research candidate, not a claimed success."""
    from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
    from humanoid_learning.envs.grasp_env import FixedBaseGraspEnv
    from humanoid_learning.expert.diagonal_feasibility import (
        CANDIDATE_C1, CANDIDATE_C2, build_candidate_specific_seeds, build_posture_seeds,
        evaluate_static_pose, hand_corner_target, measure_per_finger_local_offsets, score_result,
    )
    from humanoid_learning.expert.coupled_ik import CoupledBilateralIK
    from humanoid_learning.expert.grasp_expert import GraspExpertConfig
    from humanoid_learning.envs import hand_synergy

    cfg = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF)
    env = FixedBaseGraspEnv(cfg)
    env.reset(seed=0)
    model = env.model

    left_palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_PALM_SITE)
    right_palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.RIGHT_PALM_SITE)
    waist_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in wbc.WAIST_JOINTS]
    waist_qpos_adr = np.array([model.jnt_qposadr[j] for j in waist_jids])
    waist_dof_adr = np.array([model.jnt_dofadr[j] for j in waist_jids])
    waist_low = model.jnt_range[waist_jids, 0].copy()
    waist_high = model.jnt_range[waist_jids, 1].copy()
    coupled = CoupledBilateralIK(
        model, left_palm_id, right_palm_id,
        waist_dof_adr, env._arm_dof_adr[:7], env._arm_dof_adr[7:],
        waist_qpos_adr, env._arm_qpos_adr[:7], env._arm_qpos_adr[7:],
        np.concatenate([waist_low, env._arm_ctrl_low[:7], env._arm_ctrl_low[7:]]),
        np.concatenate([waist_high, env._arm_ctrl_high[:7], env._arm_ctrl_high[7:]]),
    )
    stand_q17 = np.concatenate([
        env.data.qpos[waist_qpos_adr].copy(), env.data.qpos[env._arm_qpos_adr[:7]].copy(), env.data.qpos[env._arm_qpos_adr[7:]].copy(),
    ])
    obj_pos = env.data.qpos[env._object_qpos_adr:env._object_qpos_adr + 3].copy()
    obj_quat = env.data.qpos[env._object_qpos_adr + 3:env._object_qpos_adr + 7].copy()
    gcfg = GraspExpertConfig()
    left_offsets = measure_per_finger_local_offsets(env, "left", gcfg.thumb1_abduct_pose_left)
    right_offsets = measure_per_finger_local_offsets(env, "right", gcfg.thumb1_abduct_pose_right)
    left_open = np.array(hand_synergy.left_hand_targets(0.0), dtype=float); left_open[1] = gcfg.thumb1_abduct_pose_left
    right_open = np.array(hand_synergy.right_hand_targets(0.0), dtype=float); right_open[1] = gcfg.thumb1_abduct_pose_right
    finger_adr = (env._left_finger_qpos_adr, env._right_finger_qpos_adr)
    finger_targets = (left_open, right_open)
    seeds = build_posture_seeds(stand_q17)
    scratch = mujoco.MjData(model)

    def solve_one(candidate, seed, yaw, standoff, waist_weight):
        left_t = hand_corner_target(obj_pos, obj_quat, SIZE_12_HALF, candidate.left, palm_standoff=standoff, side="left", corner_yaw_deg=yaw)
        right_t = hand_corner_target(obj_pos, obj_quat, SIZE_12_HALF, candidate.right, palm_standoff=standoff, side="right", corner_yaw_deg=yaw)
        scratch.qpos[:] = env.data.qpos
        scratch.qvel[:] = 0.0
        return evaluate_static_pose(
            coupled, scratch, model, env._object_body_id, obj_pos, obj_quat, candidate, seed, left_t, right_t,
            left_offsets, right_offsets, finger_qpos_adr=finger_adr, finger_open_targets=finger_targets,
            corner_yaw_deg=(yaw, yaw), waist_weight=waist_weight,
        )

    print("Searching the same coarse grid scripts/test_diagonal_feasibility.py uses (this may take ~1-2 minutes)...")
    results = []
    for candidate in (CANDIDATE_C1, CANDIDATE_C2):
        all_seeds = seeds + build_candidate_specific_seeds(candidate.name, stand_q17)
        for yaw in (15.0, 30.0, 45.0):
            for standoff in (0.20, 0.24):
                for waist_weight in (None, 1.0):
                    for seed in all_seeds:
                        results.append(solve_one(candidate, seed, yaw, standoff, waist_weight))

    # Default to the WAIST-TWISTED result (waist_weight=1.0, waist let
    # loose instead of the solver's own 6x anti-waist default) -- that is
    # the specific comparison the user asked to see, not just whichever
    # candidate score_result ranks first overall (score_result's face-
    # assignment tie only breaks on raw error right now, which happened
    # to prefer an UN-twisted, joint-limit-saturated candidate last time).
    twisted = [r for r in results if r.waist_weight_used != 6.0]
    results.sort(key=score_result)
    twisted.sort(key=score_result)
    best = twisted[0] if twisted else results[0]
    best_default = results[0]

    print(
        f"\nShowing the WAIST-TWISTED candidate (waist_weight={best.waist_weight_used:.1f}, waist actually let loose)."
    )
    print(
        f"For comparison, the untwisted default (waist_weight={best_default.waist_weight_used:.1f}) has "
        f"wrist_yaw_margin={best_default.wrist_yaw_margin:.4f}/elbow_margin={best_default.elbow_margin:.4f} "
        f"vs this candidate's wrist_yaw_margin={best.wrist_yaw_margin:.4f}/elbow_margin={best.elbow_margin:.4f}.\n"
    )
    print(
        f"Best static candidate found: chirality={best.candidate} seed={best.seed_name} "
        f"corner_yaw={best.corner_yaw_deg} waist_weight={best.waist_weight_used:.1f}\n"
        f"  collision_free={best.collision_free}  face_assignment_satisfied={best.face_assignment_satisfied}\n"
        f"  wrist_yaw_margin={best.wrist_yaw_margin:.4f}  elbow_margin={best.elbow_margin:.4f}\n"
        f"  left_pos_error={best.left_pos_error:.4f}  right_pos_error={best.right_pos_error:.4f}\n"
        f"  left_ori_error={best.left_ori_error:.4f}  right_ori_error={best.right_ori_error:.4f}\n"
        f"  intended faces: L thumb={best.left_target_face_thumb} finger={best.left_target_face_finger}  "
        f"R thumb={best.right_target_face_thumb} finger={best.right_target_face_finger}\n"
        f"  ACTUAL faces:   L thumb={best.left_thumb_face_actual} finger={best.left_finger_face_actual}  "
        f"R thumb={best.right_thumb_face_actual} finger={best.right_finger_face_actual}\n"
    )
    if not best.face_assignment_satisfied:
        print("NOTE: fingertip face assignment is NOT yet correct (see PROJECT_CONTEXT.md 22nd-session report) --")
        print("      this is a known-incomplete research candidate, shown as-is, not a claimed success.\n")

    env.data.qpos[waist_qpos_adr] = best.q17[0:3]
    env.data.qpos[env._arm_qpos_adr[:7]] = best.q17[3:10]
    env.data.qpos[env._arm_qpos_adr[7:]] = best.q17[10:17]
    env.data.qpos[finger_adr[0]] = finger_targets[0]
    env.data.qpos[finger_adr[1]] = finger_targets[1]
    env.data.ctrl[env._arm_dof_adr[:7]] = best.q17[3:10]
    env.data.ctrl[env._arm_dof_adr[7:]] = best.q17[10:17]
    mujoco.mj_forward(model, env.data)

    print("Showing the frozen static pose (physics is NOT stepped -- this is the raw IK solution, not a held/settled")
    print("pose). Close the viewer window to exit.")
    with mujoco.viewer.launch_passive(model, env.data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def mode_whole_body_diagonal():
    """SIZE_12 Full-Body Diagonal Reach session: runs the Stage W
    (floating pelvis + legs + waist + arms) static feasibility search
    from humanoid_learning.expert.whole_body_diagonal_feasibility and
    reports the result.

    IMPORTANT: this session's search found NO candidate that is
    simultaneously (a) kinematically reachable (hand+foot task
    converged, joint limits safe), (b) collision-free, (c) COM-support-
    margin-safe, AND (d) fingertip-face-correct -- Static Full-Body
    Feasibility is FAIL for now, per PROJECT_CONTEXT.md's session
    report. This is NOT a claim that the resulting motion below is a
    successful grasp reach -- it is the robot ACTUALLY ATTEMPTING the
    best-found (known-imperfect) candidate under real physics (mj_step,
    driven by the leg/waist/arm position actuators, min-jerk from the
    stand pose), so whatever genuinely happens -- reaching cleanly,
    wobbling, or the pelvis/knee losing balance given the already-
    measured negative COM support margin -- is shown as-is. The full
    static breakdown is printed to the console FIRST."""
    from humanoid_learning.envs.whole_body_env import WholeBodyEnv
    from humanoid_learning.envs.grasp_config import SIZE_12_HALF
    from humanoid_learning.expert.diagonal_feasibility import CANDIDATE_C1, CANDIDATE_C2
    from humanoid_learning.expert.whole_body_diagonal_feasibility import (
        build_whole_body_setup, build_whole_body_posture_seeds, run_stage_w_search, stage_w_score,
    )

    print("Building the SIZE_12 whole-body Stage W search (floating pelvis + legs + waist + arms)...")
    env = WholeBodyEnv(wbc.WholeBodyConfig(), include_object=True)
    env.reset(seed=0)
    setup = build_whole_body_setup(env, SIZE_12_HALF)
    seeds = build_whole_body_posture_seeds(setup.stand_joints_q)
    print(f"Searching {len(seeds)} posture families x 2 chiralities x 2 corner angles (this may take a few minutes)...")
    results = run_stage_w_search(setup, (CANDIDATE_C1, CANDIDATE_C2), seeds, (30.0, 45.0), 0.20, env.data.qpos.copy())
    results.sort(key=stage_w_score)
    best = results[0]
    n_full = sum(1 for r in results if r.fully_passed)

    print(
        f"\nBest Stage W candidate: chirality={best.candidate} seed={best.seed_name} corner_yaw={best.corner_yaw_deg}\n"
        f"  kinematic reach converged (hands+feet, joint-limit safe): {best.ik.success}\n"
        f"    left_hand_pos_error={best.ik.left_hand_pos_error:.4f}  right_hand_pos_error={best.ik.right_hand_pos_error:.4f}\n"
        f"    left_foot_pos_error={best.ik.left_foot_pos_error:.4f}  right_foot_pos_error={best.ik.right_foot_pos_error:.4f}\n"
        f"    joint_limit_margin={best.ik.joint_limit_margin:.4f}\n"
        f"  collision_free={best.collision_free}  (pairs: {best.collision_pairs[:3]})\n"
        f"  COM support margin={best.com_support_margin:.4f}  (negative = COM projects outside the support polygon)\n"
        f"  fingertip face assignment satisfied={best.face_assignment_satisfied}\n"
        f"    intended (L thumb, L finger, R thumb, R finger)=({best.candidate == 'C1' and ('FACE_POS_X','FACE_POS_Y','FACE_NEG_X','FACE_NEG_Y') or ('FACE_NEG_X','FACE_POS_Y','FACE_POS_X','FACE_NEG_Y')})\n"
        f"    actual  ={best.actual_faces}\n"
        f"  fully_passed (ALL of the above)={best.fully_passed}\n"
    )
    print(f"{n_full} / {len(results)} candidates fully passed Static Full-Body Feasibility.")

    if n_full == 0:
        print(
            "\nSTATIC FULL-BODY FEASIBILITY: FAIL.\n"
            "Per PROJECT_CONTEXT.md's session report: kinematic reach (hand/foot position+orientation, joint-limit\n"
            "margin) IS achievable for several candidates, but every one of them fails at least one of the two NEW\n"
            "Stage W criteria -- collision (the hip region grazes the table when hinging forward to reach) and COM\n"
            "support margin (reaching the diagonal corner shifts the whole-body COM outside this session's simplified\n"
            "bounding-box support-polygon estimate by several cm) -- and fingertip face precision is also not yet\n"
            "exact. The viewer below will actually DRIVE the robot toward this known-imperfect candidate under real\n"
            "physics -- watch for wobbling, falling, or the knee/hip losing contact, which is the physical meaning\n"
            "of the negative COM margin measured above, not a bug in the playback.\n"
        )
    else:
        print("A genuine Static Full-Body PASS candidate was found -- driving the robot toward it below.")

    # Real dynamic playback (Section 13's structure, simplified to one
    # combined min-jerk phase from stand -> best-found target, since no
    # candidate reached the Static PASS bar that would justify staging
    # the full lower-body/pelvis/shoulder/elbow/wrist sub-phases
    # separately): actuator ctrl targets are min-jerk-interpolated over
    # ~4 seconds and mj_step is called every physics tick -- the pelvis
    # is NEVER written directly (it is unactuated; wherever it ends up
    # is a real consequence of leg motion + gravity + foot contact).
    target_leg = best.ik.joints_qpos[0:12]
    target_waist = best.ik.joints_qpos[12:15]
    target_arm = best.ik.joints_qpos[15:29]
    start_leg = env.data.ctrl[env._leg_act_ids].copy()
    start_waist = env.data.ctrl[env._waist_act_ids].copy()
    start_arm = env.data.ctrl[env._arm_act_ids].copy()
    env.data.ctrl[env._left_hand_act_ids] = setup.left_finger_open
    env.data.ctrl[env._right_hand_act_ids] = setup.right_finger_open

    duration_steps = int(4.0 / (setup.model.opt.timestep * env.config.frame_skip))
    print(f"Driving toward the candidate over {duration_steps} control steps (~4.0s simulated time)...")
    print("Close the viewer window to exit; the attempt repeats once it settles or falls.")

    def reset_to_stand():
        env.reset(seed=0)
        build_whole_body_setup(env, SIZE_12_HALF)  # re-applies the SIZE_12 object resize/placement
        env.data.ctrl[env._leg_act_ids] = start_leg
        env.data.ctrl[env._waist_act_ids] = start_waist
        env.data.ctrl[env._arm_act_ids] = start_arm
        mujoco.mj_forward(setup.model, env.data)

    reset_to_stand()
    state = {"i": 0, "reported_outcome": False}

    def step(_i):
        i = state["i"]
        t = min(1.0, i / duration_steps)
        s = 10 * t**3 - 15 * t**4 + 6 * t**5  # minimum-jerk profile
        env.data.ctrl[env._leg_act_ids] = start_leg + s * (target_leg - start_leg)
        env.data.ctrl[env._waist_act_ids] = start_waist + s * (target_waist - start_waist)
        env.data.ctrl[env._arm_act_ids] = start_arm + s * (target_arm - start_arm)
        for _ in range(env.config.frame_skip):
            mujoco.mj_step(setup.model, env.data)
        pelvis_z = env.data.xpos[mujoco.mj_name2id(setup.model, mujoco.mjtObj.mjOBJ_BODY, wbc.PELVIS_BODY)][2]
        fallen = pelvis_z < 0.4
        if i % 60 == 0:
            print(f"  step {i:4d}/{duration_steps} t={t:.2f} pelvis_z={pelvis_z:.3f} fallen={fallen}")
        if fallen and not state["reported_outcome"]:
            print(f"  >>> robot fell (pelvis_z={pelvis_z:.3f}) while attempting this candidate -- resetting shortly.")
            state["reported_outcome"] = True
        state["i"] += 1
        if state["i"] >= duration_steps + 120:  # ~2s hold at the end, then restart
            if not state["reported_outcome"]:
                print(f"  >>> attempt finished without falling (pelvis_z={pelvis_z:.3f}); resetting to show it again.")
            reset_to_stand()
            state["i"] = 0
            state["reported_outcome"] = False

    with mujoco.viewer.launch_passive(setup.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            step(state["i"])
            viewer.sync()
            elapsed = time.time() - step_start
            target_dt = setup.model.opt.timestep * env.config.frame_skip
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)


def mode_grasp(
    object_pos_x: float = 0.27,
    object_half_size: float | None = None,
    safety_latch: bool = False,
):
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

    condition = "SAFETY-EVENT LATCH (experimental negative result)" if safety_latch else "CANONICAL"
    print(f"Bimanual side-pinch grasp attempt — {condition}.")
    print(f"object_pos_x={object_pos_x}  object_half_size={object_half_size}")
    print("Close the viewer window to exit; a new attempt restarts automatically.")

    # mujoco.viewer.launch_passive is bound to one model/data pair for its
    # whole lifetime -- reset the SAME env for each new attempt rather than
    # constructing a new one (which would build a different model/data the
    # already-open viewer could not switch to).
    config_kwargs = dict(
        object_pos=(object_pos_x, 0.0, 0.0),
        arm_kp=120.0,
        persist_safety_synergy_rollback=safety_latch,
    )
    if object_half_size is not None:
        config_kwargs["object_half_size"] = object_half_size
    env = FixedBaseGraspEnv(GraspEnvConfig(**config_kwargs))
    state = {"env": env, "expert": None, "last_state": None, "frame": 0}

    def new_attempt():
        env.reset(seed=0)
        state["expert"] = BimanualSidePinchExpert(
            env,
            GraspExpertConfig(latch_thumb_after_first_contact=safety_latch),
        )
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
                f"reason={outcome.failure_reason}  "
                f"tripod={outcome.max_bilateral_tripod_streak}/30  "
                f"multifinger={outcome.max_bilateral_multifinger_streak}  "
                f"obj_xy={outcome.object_xy_displacement:.4f}m  "
                f"hand_hand={outcome.max_hand_hand_force_raw:.1f}N/"
                f"{outcome.max_hand_hand_contact_streak}step  "
                f"safety={outcome.substep_safety_event_count}  "
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
    parser.add_argument("--grasp-safety-latch", action="store_true", help="show the 28th-session failed safety-event latch A/B condition (not canonical)")
    parser.add_argument("--diagonal-feasibility", action="store_true", help="show the current best SIZE_12 diagonal four-face STATIC candidate (frozen pose, not a live grasp attempt)")
    parser.add_argument("--whole-body-diagonal", action="store_true", help="run the SIZE_12 full-body (pelvis/legs/waist/arms) Stage W diagonal-reach static feasibility search and report the result")
    parser.add_argument("--object-pos-x", type=float, default=0.27, help="object x position, meters from robot origin (--grasp only)")
    parser.add_argument("--object-half-size", type=float, default=None, help="object half-size, meters (--grasp only; default: GraspEnvConfig's own default, 0.06)")
    args = parser.parse_args()

    modes = [args.stand, args.posture, args.planar, args.grasp, args.grasp_safety_latch, args.diagonal_feasibility, args.whole_body_diagonal]
    if sum(bool(m) for m in modes) != 1:
        parser.error("pass exactly one of --stand / --posture / --planar / --grasp / --grasp-safety-latch / --diagonal-feasibility / --whole-body-diagonal")

    if args.stand:
        mode_stand()
    elif args.posture:
        mode_posture()
    elif args.planar:
        mode_planar()
    elif args.grasp:
        mode_grasp(object_pos_x=args.object_pos_x, object_half_size=args.object_half_size)
    elif args.grasp_safety_latch:
        mode_grasp(
            object_pos_x=args.object_pos_x,
            object_half_size=args.object_half_size,
            safety_latch=True,
        )
    elif args.diagonal_feasibility:
        mode_diagonal_feasibility()
    elif args.whole_body_diagonal:
        mode_whole_body_diagonal()


if __name__ == "__main__":
    main()
