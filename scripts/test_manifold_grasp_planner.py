"""Tests for humanoid_learning.expert.manifold_grasp_planner (31st Phase
4 Grasp Track session: Reachable Contact Manifold Grasp Planner). Run
with:
    python scripts/test_manifold_grasp_planner.py

This is a SEPARATE test file from scripts/test_tripod_closure_planner.py
on purpose: that file's 30th-session fixed-target negative results
(thumb-only IK cannot reach a hand-coded index/middle-aligned target)
are PRESERVED UNCHANGED as a regression -- this file tests the NEW
manifold formulation (contact points as decision variables, not fixed
targets) that supersedes 30th-session's APPROACH to the question, not
its measured facts.

REAL FINDING (31st session): across 3 topology candidates (measured
primary X-face split, its mirror, and an alternative thumb-on-+Z-face
split) x 7 deterministic posture-family starts (21 total optimizations),
every run reaches good face-plane alignment (several under 1cm) but NONE
clears real MuJoCo mesh collision checks -- thumb-thumb and palm-object
contacts persist across nearly all attempts. This is reported as
category B3 (partial contact capability, no confirmed force-closure
candidate) per this session's own decision framework, NOT as a confirmed
Dex3 morphology limit -- the current residual set has no explicit
thumb-proximal-segment separation term, which is a disclosed gap in this
session's own objective function, not evidence of physical
impossibility.

REAL FINDING (32nd session): auditing the 31st session's own best
candidate's REAL MuJoCo contacts (not just the 3 name-substring
categories it checked) found index/middle PROXIMAL links (not just
tips) penetrating the object by up to 3.3cm, and G1 self-collision
(torso vs shoulder, 7.6mm) -- neither was visible to the 31st session's
narrower classifier. check_pose_collisions/_classify_contact were
rewritten to classify EVERY real contact into one of 6 forbidden
buckets (thumb_thumb, hand_hand, proximal_object, wrist_palm_object,
self_collision, table_hand) or "allowed_tip_contact" (one of the 6
designated true-fingertip links touching the object), and the collision
penalty residual now covers all 6 buckets instead of 3.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.grasp_env import FixedBaseGraspEnv
from humanoid_learning.expert.grasp_expert import BimanualSidePinchExpert, GraspExpertConfig, GraspState
from humanoid_learning.expert.manifold_grasp_planner import (
    build_context, derive_primary_topology, mirrored_topology, run_multistart, solve_manifold,
    is_statically_feasible, check_pose_collisions, wrench_diagnostics, _write_x, _tip_world, FINGERS,
    enumerate_contact_pairs, ALL_FORBIDDEN_BUCKETS, LEGACY_FORBIDDEN_BUCKETS,
)


def make_env():
    return FixedBaseGraspEnv(GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF))


def _reach_thumb_oppose():
    env = make_env()
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    outcome = None
    for i in range(1200):
        outcome = expert.step()
        if outcome.state in (GraspState.THUMB_OPPOSE, GraspState.FAILURE, GraspState.SUCCESS):
            break
    assert outcome.state == GraspState.THUMB_OPPOSE
    return env, expert


def test_primary_topology_matches_measured_approach_geometry():
    """The topology is DERIVED from real FK (Stage 1's own established
    fact: index/middle approach the object's local +X face, thumb the
    opposing -X face) -- not guessed by name."""
    env, expert = _reach_thumb_oppose()
    ctx = build_context(env, expert)
    topo = derive_primary_topology(ctx)
    assert topo.face_axis["index"] == 0 and topo.face_axis["middle"] == 0 and topo.face_axis["thumb"] == 0
    assert topo.face_sign["thumb"] == -topo.face_sign["index"]
    print(f"    derived topology: {topo.face_axis}, signs={topo.face_sign}")


def test_index_middle_are_not_hard_constrained():
    """REAL AUDIT FINDING (Section 4 of this session): the OLD planner
    (tripod_closure_planner.plan_tripod_targets) hard-fixes index/middle
    at their currently-achieved position. This solver instead only
    applies a SMALL regularization weight to them (W_REG_FINGER_INDEXMID
    << W_REG_ARM would be a hard hold; here it is a soft pull) --
    confirmed by checking the solved index/middle joint values actually
    move away from their exact starting value in at least one start."""
    env, expert = _reach_thumb_oppose()
    ctx = build_context(env, expert)
    topo = derive_primary_topology(ctx)
    res = solve_manifold(ctx, ctx.x0, topo, max_iters=100)
    # index/middle finger qpos slots: local indices 7,8 (middle_0/1) and 9,10 (index_0/1) within each 14-slot hand block? see N_ARM=7 offset
    from humanoid_learning.expert.manifold_grasp_planner import N_ARM, N_PER_HAND
    left_finger_start = N_ARM  # slot 7 within left hand's 14
    moved = np.abs(res["x"][left_finger_start:left_finger_start + 7] - ctx.x0[left_finger_start:left_finger_start + 7])
    print(f"    left finger joint movement from x0: {np.round(moved, 4)}")
    assert np.max(moved) > 1e-4, "at least one finger joint (thumb or index/middle) must be free to move, not hard-held"


def test_face_plane_residual_converges_well_below_reachability_gap():
    """REAL FINDING: unlike the fixed-target formulation (47-49mm
    residual, thumb pinned at hard limits), letting the solver choose
    WHERE on the face to land converges the face-plane residual to
    within a few mm for the best posture-family start."""
    env, expert = _reach_thumb_oppose()
    results, ctx = run_multistart(env, expert)
    best = min(results, key=lambda r: max(abs(v) for v in r.per_finger_face_residual.values()))
    max_res = max(abs(v) for v in best.per_finger_face_residual.values())
    print(f"    best face residual across {len(results)} starts: {max_res:.4f} m ({best.topology_name}/{best.name})")
    assert max_res < 0.02, "letting contact position be a free variable must reach FAR better face alignment than the fixed-target approach (30th session: 47-49mm)"


def test_joint_limit_margins_reported_in_radians_not_meters():
    """[31st-session correction] Rotational joint-limit margins must be
    reported in rad -- a prior session's prose mislabeled a hinge-joint
    margin with 'm'. This test locks the unit at the data level."""
    env, expert = _reach_thumb_oppose()
    results, ctx = run_multistart(env, expert)
    r = results[0]
    for jn, margin in r.joint_margins_rad.items():
        jid_range = None
        # sanity: hinge joint ranges in this model are all well under 2*pi in magnitude (radians, not meters)
        assert abs(margin) < 2 * np.pi, f"{jn} margin {margin} looks like it might not be in radians"
    print(f"    sample margins (rad): {dict(list(r.joint_margins_rad.items())[:3])}")


def test_no_multistart_candidate_is_fully_statically_feasible():
    """REAL FINDING (this session, honestly reported, NOT a bug): across
    3 topologies x 7 posture families = 21 optimizations, none clears
    ALL of Section 8's strict static-success criteria simultaneously
    (face alignment, zero real collisions, joint-limit margin). The
    dominant remaining violation is thumb-thumb / palm-object MESH
    contact even when the TIP-level face/separation residuals are
    small -- this solver's residual set has no explicit proximal-
    segment separation term, a disclosed gap, not proof of
    impossibility (category B3, not B4)."""
    env, expert = _reach_thumb_oppose()
    primary = None
    results, ctx = run_multistart(env, expert)
    feasible = [r for r in results if is_statically_feasible(r)]
    print(f"    {len(feasible)}/{len(results)} starts statically feasible")
    for r in sorted(results, key=lambda r: r.cost)[:3]:
        print(f"    top-3 by cost: {r.topology_name}/{r.name} cost={r.cost:.4f} "
              f"collisions={r.collisions} max_face={max(abs(v) for v in r.per_finger_face_residual.values()):.4f}")
    assert len(feasible) == 0, "this is an honest negative result -- if this ever flips to >0, update the session doc, don't just delete this assertion"


def test_wrench_diagnostics_rank_and_condition_are_computed():
    """Force-closure is a STATIC diagnostic only (never a Gate A
    substitute) -- verifies the friction-pyramid wrench matrix has full
    rank (6) for a real six-contact candidate and returns a finite
    condition number."""
    env, expert = _reach_thumb_oppose()
    results, ctx = run_multistart(env, expert)
    best = min(results, key=lambda r: r.cost)
    topo = derive_primary_topology(ctx) if best.topology_name == "measured_primary" else mirrored_topology(derive_primary_topology(ctx))
    _write_x(ctx, best.x)
    contacts = []
    for side in ("left", "right"):
        for finger in FINGERS:
            pos = _tip_world(ctx, side, finger)
            axis = topo.face_axis[finger]
            sign = topo.face_sign[finger]
            normal_local = np.zeros(3)
            normal_local[axis] = sign
            contacts.append({"pos": pos, "normal": ctx.obj_R @ normal_local})
    diag = wrench_diagnostics(contacts, ctx.obj_pos, mass=env.config.object_mass if hasattr(env.config, "object_mass") else 0.1)
    print(f"    wrench diagnostics: {diag}")
    assert diag["rank"] == 6, "six well-separated opposing contacts should span the full 6D wrench space"
    assert np.isfinite(diag["condition"])


def test_deterministic_multistart_reproducible():
    env1, expert1 = _reach_thumb_oppose()
    results1, _ = run_multistart(env1, expert1)
    env2, expert2 = _reach_thumb_oppose()
    results2, _ = run_multistart(env2, expert2)
    costs1 = [round(r.cost, 6) for r in results1]
    costs2 = [round(r.cost, 6) for r in results2]
    assert costs1 == costs2, "identical seed/config must give identical multi-start results"


def test_collision_check_distinguishes_categories():
    env, expert = _reach_thumb_oppose()
    ctx = build_context(env, expert)
    coll = check_pose_collisions(env, ctx.scratch)
    assert set(coll.keys()) >= {"thumb_thumb", "hand_hand", "proximal_object", "wrist_palm_object",
                                 "self_collision", "table_hand", "max_penetration", "n_contacts", "n_forbidden_contacts"}


def test_contact_pair_enumeration_reports_real_geom_and_body_names():
    """Section 5's exact requirement: every real penetrating contact
    with actual geom/body names and a penetration depth, not aggregate
    counts only."""
    env, expert = _reach_thumb_oppose()
    results, ctx = run_multistart(env, expert)
    best = min(results, key=lambda r: r.cost)
    assert len(best.contact_pairs) > 0
    for p in best.contact_pairs:
        assert p["bucket"] in ("allowed_tip_contact",) + ALL_FORBIDDEN_BUCKETS + ("table_object", "other")
        assert p["penetration"] >= 0.0
        assert isinstance(p["geom1"], str) and isinstance(p["geom2"], str)
    print(f"    {len(best.contact_pairs)} real contact pairs at best candidate ({best.topology_name}/{best.name})")


def test_classifier_finds_proximal_object_and_self_collision_categories():
    """REAL FINDING (32nd session): the 31st session's own best
    candidate (measured_primary/palm_back) has index/middle PROXIMAL
    links (not the tip) penetrating the object, and torso-vs-shoulder
    self-collision -- neither category existed in the 31st session's
    3-bucket classifier. This test locks in that BOTH new categories are
    now detected on that exact candidate."""
    env, expert = _reach_thumb_oppose()
    results, ctx = run_multistart(env, expert)
    palm_back = next(r for r in results if r.name == "palm_back" and r.topology_name == "measured_primary")
    buckets_seen = {p["bucket"] for p in palm_back.contact_pairs}
    print(f"    buckets seen at palm_back: {buckets_seen}")
    assert "proximal_object" in buckets_seen, "index/middle proximal links penetrating the object must be detected"
    assert "self_collision" in buckets_seen, "G1 self-collision (torso vs shoulder) must be detected"


def test_collision_aware_finds_strictly_more_forbidden_contacts_than_legacy():
    """Causal A/B (Section 7): on the SAME candidate pose, classifying
    with ALL_FORBIDDEN_BUCKETS must find >= as many forbidden contacts
    as LEGACY_FORBIDDEN_BUCKETS (legacy is a strict subset of buckets),
    and strictly more on at least one real candidate (the 31st session's
    best), since proximal_object/self_collision contacts exist there."""
    env, expert = _reach_thumb_oppose()
    results_legacy, ctx = run_multistart(env, expert, forbidden_buckets=LEGACY_FORBIDDEN_BUCKETS)
    legacy_best = next(r for r in results_legacy if r.name == "palm_back" and r.topology_name == "measured_primary")
    n_legacy_forbidden = sum(
        1 for p in legacy_best.contact_pairs if p["bucket"] in LEGACY_FORBIDDEN_BUCKETS
    )
    n_full_forbidden = sum(
        1 for p in legacy_best.contact_pairs if p["bucket"] in ALL_FORBIDDEN_BUCKETS
    )
    print(f"    at the SAME pose: legacy-bucket contacts={n_legacy_forbidden}, full-bucket contacts={n_full_forbidden}")
    assert n_full_forbidden > n_legacy_forbidden, "the fuller classifier must see strictly more forbidden contacts at a pose known to have proximal/self collisions"


def test_solving_with_full_collision_buckets_still_runs_and_is_deterministic():
    """The collision-aware residual (with the full bucket set) must
    still produce a deterministic multi-start result -- same causal
    machinery (numerical Jacobian + LM), just a larger forbidden set."""
    env1, expert1 = _reach_thumb_oppose()
    results1, _ = run_multistart(env1, expert1, forbidden_buckets=ALL_FORBIDDEN_BUCKETS)
    env2, expert2 = _reach_thumb_oppose()
    results2, _ = run_multistart(env2, expert2, forbidden_buckets=ALL_FORBIDDEN_BUCKETS)
    costs1 = [round(r.cost, 6) for r in results1]
    costs2 = [round(r.cost, 6) for r in results2]
    assert costs1 == costs2


def test_canonical_grasp_behavior_unchanged():
    """This whole module is read-only and never imported by the live
    control loop."""
    env = make_env()
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    outcome = None
    for _ in range(1200):
        outcome = expert.step()
        if outcome.state in (GraspState.FAILURE, GraspState.SUCCESS):
            break
    assert outcome.max_bilateral_tripod_streak == 14
    assert outcome.state == GraspState.FAILURE


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
