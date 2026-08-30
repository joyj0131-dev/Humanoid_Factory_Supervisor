"""Tests for humanoid_learning.expert.whole_body_diagonal_feasibility
(SIZE_12 Full-Body Diagonal Reach session, Stage W).

These are STATIC (pure-kinematic, no physics rollout) tests, same
convention as scripts/test_diagonal_feasibility.py's Stage U tests --
they never call env.step(). Run with:
    python scripts/test_whole_body_diagonal_feasibility.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.envs.grasp_config import SIZE_12_HALF
from humanoid_learning.envs.whole_body_env import WholeBodyEnv
from humanoid_learning.expert.diagonal_feasibility import CANDIDATE_C1, CANDIDATE_C2
from humanoid_learning.expert.whole_body_diagonal_feasibility import (
    N_ARM,
    N_LEG,
    N_PELVIS,
    N_WAIST,
    build_whole_body_posture_seeds,
    build_whole_body_setup,
    evaluate_whole_body_pose,
    run_stage_w_search,
    stage_w_score,
    support_polygon_margin,
)


def _make_env_and_setup():
    env = WholeBodyEnv(wbc.WholeBodyConfig(), include_object=True)
    env.reset(seed=0)
    setup = build_whole_body_setup(env, SIZE_12_HALF)
    return env, setup


# ---------------------------------------------------------------------
# Section 18, item 4: full-body solver actually includes legs/pelvis as
# real DoF (not silently ignoring them).
# ---------------------------------------------------------------------
def test_whole_body_dof_layout_includes_pelvis_and_legs():
    assert N_PELVIS == 6
    assert N_LEG == 6  # per leg
    assert N_WAIST == 3
    assert N_ARM == 7  # per arm
    total = N_PELVIS + 2 * N_LEG + N_WAIST + 2 * N_ARM
    assert total == 35
    env, setup = _make_env_and_setup()
    assert len(setup.solver.dof_adr) == 35
    assert len(setup.joints_qpos_adr) == 29  # legs12 + waist3 + arms14


# ---------------------------------------------------------------------
# Section 18, item 5: both feet stay within the required position/
# orientation tolerance for a REAL solve (not just by construction).
#
# NOTE on why this samples across the whole seed set instead of one
# named seed: this session discovered (see whole_body_diagonal_
# feasibility.py's module docstring, "KNOWN UNRESOLVED ISSUE") that
# WholeBodyDiagonalIK.solve()'s convergence for a SPECIFIC (candidate,
# seed) pair is not reproducible in isolation -- the same call can
# converge cleanly or fail depending on unrelated prior solver calls on
# the same process, root cause not yet identified. Asserting on one
# named seed would make this test flaky in a way that reflects that
# unresolved bug, not a regression in the foot-task mechanism itself.
# Sampling across the full posture-family set and requiring at least
# ONE genuinely converged candidate to hold its feet is the honest,
# reproducible claim: the mechanism works when the solve converges at
# all -- it is the CONVERGENCE ITSELF that is not yet reliable.
# ---------------------------------------------------------------------
def test_foot_constraint_held_when_solve_converges():
    env, setup = _make_env_and_setup()
    seeds = build_whole_body_posture_seeds(setup.stand_joints_q)
    results = run_stage_w_search(setup, (CANDIDATE_C1, CANDIDATE_C2), seeds, (30.0, 45.0), 0.20, env.data.qpos.copy())
    converged = [r for r in results if r.ik.success]
    assert converged, "at least one candidate should reach kinematic convergence in a full sweep (see module docstring)"
    for r in converged:
        assert r.ik.left_foot_pos_error < 0.01, f"{r.candidate}/{r.seed_name}: left foot drifted {r.ik.left_foot_pos_error:.4f}m"
        assert r.ik.right_foot_pos_error < 0.01, f"{r.candidate}/{r.seed_name}: right foot drifted {r.ik.right_foot_pos_error:.4f}m"
        assert r.ik.left_foot_ori_error < np.radians(5)
        assert r.ik.right_foot_ori_error < np.radians(5)


# ---------------------------------------------------------------------
# Section 18, item 8: knee bend actually changes pelvis height. Checked
# via DIRECT forward kinematics on the seed's own joint values (mj_
# kinematics only, no IK solve) -- this is deliberately independent of
# WholeBodyDiagonalIK.solve()'s unresolved convergence-instability issue
# (see module docstring), since the claim being tested ("this seed's
# leg geometry implies a lower pelvis") is about the SEED itself, not
# about whether the solver later converges from it.
# ---------------------------------------------------------------------
def test_knee_bend_seed_actually_lowers_pelvis_height():
    env, setup = _make_env_and_setup()
    stand_pelvis_z = setup.stand_pelvis_qpos[2]
    seeds = build_whole_body_posture_seeds(setup.stand_joints_q)
    seed = next(s for s in seeds if s.name == "symmetric_knee_bend_medium")
    scratch = mujoco.MjData(setup.model)
    scratch.qpos[:] = env.data.qpos
    scratch.qpos[setup.joints_qpos_adr] = seed.joints_q
    # Re-plant both feet at their original world height/orientation by
    # solving ONLY for the pelvis pose given these fixed leg angles --
    # simplest robust proxy: since both feet must stay on the ground and
    # leg length from hip to foot effectively shortens as the knee
    # bends, the pelvis (hip origin) must drop for the feet to stay
    # planted. Verify via direct kinematics: keep pelvis qpos at stand,
    # forward-kinematics the bent legs, and check the FOOT sites now sit
    # ABOVE their original height (feet would rise into the air) --
    # the logical inverse/equivalent of "pelvis must drop to keep feet
    # planted", without invoking the IK solver at all.
    mujoco.mj_kinematics(setup.model, scratch)
    left_foot_site = mujoco.mj_name2id(setup.model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_FOOT_SITE)
    bent_foot_z_at_stand_pelvis = scratch.site_xpos[left_foot_site][2]
    stand_foot_z = setup.left_foot_target_pos[2]
    assert bent_foot_z_at_stand_pelvis > stand_foot_z + 0.01, (
        "bending the knee while holding the pelvis at its stand height should lift the foot off the ground "
        f"(stand foot z={stand_foot_z:.4f}, bent-knee foot z at stand pelvis={bent_foot_z_at_stand_pelvis:.4f}) -- "
        "i.e. keeping this foot planted requires the pelvis to drop, confirming the seed's knee bend is real"
    )


# ---------------------------------------------------------------------
# Section 18, item 6: COM support-polygon margin computation is a real
# geometric check, not a constant -- a robot leaned so far its COM is
# clearly beyond the foot rectangle must show a negative margin, and the
# unperturbed stand pose must show a margin near its own foot-rectangle
# center (small, since standing COM sits close to geometric center for
# G1's stand keyframe -- confirmed directly, not assumed).
# ---------------------------------------------------------------------
def test_support_polygon_margin_reflects_real_com_geometry():
    env, setup = _make_env_and_setup()
    margin_stand, com_xy_stand = support_polygon_margin(setup.model, env.data)
    assert abs(margin_stand) < 0.05, f"unperturbed stand pose should be near the support-polygon center, got margin={margin_stand:.4f}"
    # Force pelvis far sideways (way beyond the foot stance width) and
    # recompute -- margin must go clearly negative.
    scratch = mujoco.MjData(setup.model)
    scratch.qpos[:] = env.data.qpos
    scratch.qpos[1] += 0.5  # 0.5m sideways, way beyond the ~0.24m foot stance
    mujoco.mj_forward(setup.model, scratch)
    margin_shifted, _ = support_polygon_margin(setup.model, scratch)
    assert margin_shifted < margin_stand, "a large sideways pelvis shift must worsen (not improve) the support margin"
    assert margin_shifted < 0.0, "a 0.5m sideways shift, far beyond the foot stance width, must leave COM outside the support polygon"


# ---------------------------------------------------------------------
# Section 18, item 13: collision rejection. Checked at the raw STAND
# pose (no IK solve involved, so unaffected by the unresolved
# convergence-instability issue) -- the stand pose itself must be
# collision-free (a real sanity check on _scan_unwanted_collisions'
# reuse in this module), establishing the negative case the sweep's
# collision numbers are measured against.
# ---------------------------------------------------------------------
def test_whole_body_stand_pose_is_collision_free():
    from humanoid_learning.expert.diagonal_feasibility import _scan_unwanted_collisions
    env, setup = _make_env_and_setup()
    collisions = _scan_unwanted_collisions(setup.model, env.data, setup.obj_body_id)
    assert not collisions, f"the unperturbed stand pose must not collide with anything, got {collisions}"


# ---------------------------------------------------------------------
# Section 18, item 14 / Section 20: the actual Static Full-Body
# Feasibility verdict. THIS TEST DOCUMENTS A REAL, MEASURED RESULT (see
# PROJECT_CONTEXT.md's session report for the full breakdown) -- it is
# intentionally left FAILING per project convention, not adjusted to
# pass: kinematic reach (hand/foot position+orientation, joint-limit
# margin >= 0.02) IS achievable for several candidates (best found:
# left/right hand position error < 3mm, foot position error < 4mm), but
# EVERY tested candidate fails at least one of: collision (hip region
# grazes the table when hinging/kneeling forward to reach), COM support
# margin (reaching the corner shifts COM ~5-7cm outside this session's
# bounding-box support-polygon estimate), or exact fingertip-face
# assignment. Verdict: A (Full-body Static FAIL), per Section 20 --
# not because reach is impossible, but because reach+balance+collision+
# face-precision have not yet been achieved SIMULTANEOUSLY.
# ---------------------------------------------------------------------
def test_static_full_body_feasibility_size12():
    env, setup = _make_env_and_setup()
    seeds = build_whole_body_posture_seeds(setup.stand_joints_q)
    results = run_stage_w_search(setup, (CANDIDATE_C1, CANDIDATE_C2), seeds, (30.0, 45.0), 0.20, env.data.qpos.copy())
    results.sort(key=stage_w_score)
    best = results[0]
    n_full = sum(1 for r in results if r.fully_passed)
    print(
        f"    best candidate={best.candidate} seed={best.seed_name} yaw={best.corner_yaw_deg}\n"
        f"    kinematic_success={best.ik.success} collision_free={best.collision_free} "
        f"com_margin={best.com_support_margin:.4f} face_ok={best.face_assignment_satisfied}\n"
        f"    Lh_pos={best.ik.left_hand_pos_error:.4f} Rh_pos={best.ik.right_hand_pos_error:.4f} "
        f"Lf_pos={best.ik.left_foot_pos_error:.4f} Rf_pos={best.ik.right_foot_pos_error:.4f}\n"
        f"    {n_full} / {len(results)} candidates fully passed"
    )
    assert best.ik.success, "Gate: kinematic reach should be achievable for at least the best candidate"
    assert best.fully_passed, (
        "Static Full-Body Feasibility FAILED: no candidate is simultaneously kinematically reachable, "
        "collision-free, COM-balanced, AND fingertip-face-correct -- see PROJECT_CONTEXT.md for the full breakdown"
    )


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
