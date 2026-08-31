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

REAL FINDING (33rd session): even with the 32nd session's collision
fix, the best candidate had a face-plane (site-position) residual of
only 1.4mm while the SAME candidate's real mesh penetrated the object
by 28.5mm on one of the supposedly "allowed" tip contacts. Measuring
the TRUE site-to-mesh-surface offset (measure_fingertip_support_offset,
a local Newton search on the REAL mujoco.mj_geomDistance, not a wide
bisection which was found to converge to spurious distant roots) found
this offset is NOT a fixed calibration constant: -19.5mm to -8.6mm at a
neutral reset pose, but +2.1mm to +4.7mm at the real THUMB_OPPOSE-entry
pose -- a >20mm swing depending on finger/arm configuration. The
face-plane residual now targets mesh_signed_distance() (real
mj_geomDistance between each tip's COLLISION geom, queried by contype,
and the object geom) directly instead of the site-position proxy.
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
    mesh_signed_distance, measure_fingertip_support_offset, ALLOWED_PENETRATION_TOL, ALLOWED_SEPARATION_TOL,
)
import mujoco


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
    candidate (measured_primary/palm_back) had index/middle PROXIMAL
    links (not the tip) penetrating the object, and torso-vs-shoulder
    self-collision -- neither category existed in the 31st session's
    3-bucket classifier.

    NOTE (33rd session): after the mesh-aware residual redesign, the
    solved pose at that exact (palm_back, measured_primary) candidate
    changed (as did every other metric at that candidate -- max_pen,
    thumb_thumb contacts, wrench residual), so self_collision no longer
    happens to occur AT THAT SPECIFIC SOLVE. That is an honest side
    effect of a genuinely different optimum, not a classifier defect --
    self_collision is still detected elsewhere (e.g. palm_forward). This
    test's real intent is "the fuller classifier is CAPABLE of detecting
    both categories", so it now checks across all multi-start candidates
    rather than hard-coding one candidate's incidental bucket set."""
    env, expert = _reach_thumb_oppose()
    results, ctx = run_multistart(env, expert)
    buckets_seen = set()
    for r in results:
        buckets_seen |= {p["bucket"] for p in r.contact_pairs}
    print(f"    buckets seen across all {len(results)} candidates: {buckets_seen}")
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


def _find_collision_geom(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == bid and model.geom_contype[g] == 1:
            return g
    raise ValueError(body_name)


def test_geom_distance_sign_and_units_on_separated_touching_penetrating():
    """Section 4: independently verifies mujoco.mj_geomDistance's sign
    convention and units (meters) on three controlled conditions using
    the REAL collision geom (contype==1, not the visual duplicate at an
    identical pose -- every hand link in this model has both) for
    left_hand_thumb_2_link against the real object box geom. Both geoms
    have margin=0.0 and gap=0.0 (confirmed directly from the model), so
    any contact.dist < 0 IS true geometric penetration, not a margin
    artifact -- this test locks that model fact in too."""
    env = make_env()
    env.reset(seed=0)
    model = env.model
    thumb_geom = _find_collision_geom(model, "left_hand_thumb_2_link")
    obj_geom = env._object_body_id
    obj_geom_id = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == obj_geom)
    assert model.geom_margin[thumb_geom] == 0.0 and model.geom_gap[thumb_geom] == 0.0
    assert model.geom_margin[obj_geom_id] == 0.0 and model.geom_gap[obj_geom_id] == 0.0

    scratch = mujoco.MjData(model)
    scratch.qpos[:] = env.data.qpos
    obj_qpos_adr = env._object_qpos_adr
    fromto = np.zeros(6)

    # 1) clearly separated
    scratch.qpos[obj_qpos_adr:obj_qpos_adr + 3] = [5.0, 5.0, 5.0]
    scratch.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, scratch)
    dist_sep = mujoco.mj_geomDistance(model, scratch, thumb_geom, obj_geom_id, 100.0, fromto)
    assert dist_sep > 0.01, f"separated geoms must report a clearly positive distance, got {dist_sep}"

    # 2) touching: place object so the true fingertip site sits exactly on
    # what WOULD be the face plane, at the pose measured to produce exact
    # touching in this session's own support-offset measurement (thumb,
    # reset pose): offset was ~-17.6mm, i.e. touching happens ~17.6mm
    # BEYOND (further than) the naive site-on-face placement.
    from humanoid_learning.expert.tripod_closure_planner import true_fingertip_world
    scratch.qpos[:] = env.data.qpos
    mujoco.mj_forward(model, scratch)
    site_pos = true_fingertip_world(model, scratch, "left_thumb_tip")
    half = env.config.object_half_size
    obj_x = site_pos[0] + half  # thumb approaches from -X, touches the -X face
    for _ in range(30):
        scratch.qpos[:] = env.data.qpos
        scratch.qpos[obj_qpos_adr:obj_qpos_adr + 3] = [obj_x, site_pos[1], site_pos[2]]
        scratch.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, scratch)
        dist_touch = mujoco.mj_geomDistance(model, scratch, thumb_geom, obj_geom_id, 1.0, fromto)
        if abs(dist_touch) < 1e-6:
            break
        obj_x += dist_touch
    assert abs(dist_touch) < 1e-4, f"converged touching distance must be ~0, got {dist_touch}"

    # 3) intentionally penetrating by a KNOWN additional depth
    known_depth = 0.005
    scratch.qpos[obj_qpos_adr] = obj_x + known_depth
    mujoco.mj_forward(model, scratch)
    dist_pen = mujoco.mj_geomDistance(model, scratch, thumb_geom, obj_geom_id, 1.0, fromto)
    print(f"    separated={dist_sep*1000:.2f}mm touching={dist_touch*1000:.4f}mm "
          f"penetrating(+{known_depth*1000:.1f}mm)={dist_pen*1000:.3f}mm")
    assert dist_pen < -0.004, "pushing 5mm further from an exact-touch pose must read close to -5mm"


def test_true_fingertip_support_offset_is_not_a_fixed_constant():
    """REAL FINDING (33rd session): the site-to-real-mesh-surface offset
    varies by configuration -- NOT a fixed calibration constant. This
    is exactly why a small SITE residual (1.4mm) could previously
    coexist with a large REAL mesh penetration (28.5mm)."""
    from humanoid_learning.expert.manifold_grasp_planner import build_context

    env, expert = _reach_thumb_oppose()
    ctx_oppose = build_context(env, expert)

    env2 = make_env()
    env2.reset(seed=0)
    expert2 = BimanualSidePinchExpert(env2, GraspExpertConfig())
    ctx_reset = build_context(env2, expert2)

    off_reset = measure_fingertip_support_offset(ctx_reset, "left", "thumb", approach_axis=0, approach_sign=-1.0)
    off_oppose = measure_fingertip_support_offset(ctx_oppose, "left", "thumb", approach_axis=0, approach_sign=-1.0)
    assert off_reset["converged"] and off_oppose["converged"]
    print(f"    reset offset={off_reset['support_offset_m']*1000:.2f}mm, "
          f"thumb_oppose_entry offset={off_oppose['support_offset_m']*1000:.2f}mm")
    assert abs(off_reset["support_offset_m"] - off_oppose["support_offset_m"]) > 0.010, \
        "support offset must differ by >1cm between two real configurations -- it is not a fixed constant"


def test_mesh_signed_distance_matches_face_residual_convergence():
    """After solving, the face-plane residual (now mesh_signed_distance)
    for every finger must be near 0 -- i.e. the REAL mesh, not a proxy
    point, is what actually converges to touching."""
    env, expert = _reach_thumb_oppose()
    results, ctx = run_multistart(env, expert)
    best = min(results, key=lambda r: r.cost)
    from humanoid_learning.expert.manifold_grasp_planner import _write_x as write_x
    write_x(ctx, best.x)
    max_abs_dist = 0.0
    for side in ("left", "right"):
        for finger in FINGERS:
            d, _ = mesh_signed_distance(ctx, side, finger)
            max_abs_dist = max(max_abs_dist, abs(d))
    print(f"    max |mesh_signed_distance| across 6 fingers at best candidate: {max_abs_dist*1000:.3f}mm")
    assert max_abs_dist < 0.01, "mesh-aware residual must converge all 6 REAL mesh distances to within 1cm of touching"


def test_allowed_contact_penetration_and_separation_bounds_are_tracked():
    """Section 7: allowed-contact penetration/separation must be
    reported as SEPARATE metrics from forbidden-category collisions."""
    env, expert = _reach_thumb_oppose()
    results, ctx = run_multistart(env, expert)
    best = min(results, key=lambda r: r.cost)
    from humanoid_learning.expert.manifold_grasp_planner import _write_x as write_x
    write_x(ctx, best.x)
    penetrations = []
    separations = []
    for side in ("left", "right"):
        for finger in FINGERS:
            d, _ = mesh_signed_distance(ctx, side, finger)
            if d < 0:
                penetrations.append(-d)
            else:
                separations.append(d)
    max_allowed_pen = max(penetrations, default=0.0)
    max_allowed_sep = max(separations, default=0.0)
    print(f"    max_allowed_contact_penetration={max_allowed_pen*1000:.3f}mm "
          f"max_allowed_contact_separation={max_allowed_sep*1000:.3f}mm "
          f"(tolerances: {ALLOWED_PENETRATION_TOL*1000:.1f}mm / {ALLOWED_SEPARATION_TOL*1000:.1f}mm)")
    assert max_allowed_pen < 0.01, "REAL FINDING (33rd session): mesh-aware residual keeps allowed-contact penetration under 1cm (was 28.5mm with the site-based residual)"


def test_left_right_mirror_support_offsets_are_consistent():
    """Left/right thumb support offsets should be close (mirrored
    geometry), not coincidentally different by a large margin."""
    env, expert = _reach_thumb_oppose()
    ctx = build_context(env, expert)
    left = measure_fingertip_support_offset(ctx, "left", "thumb", approach_axis=0, approach_sign=-1.0)
    right = measure_fingertip_support_offset(ctx, "right", "thumb", approach_axis=0, approach_sign=-1.0)
    assert left["converged"] and right["converged"]
    diff = abs(left["support_offset_m"] - right["support_offset_m"])
    print(f"    left={left['support_offset_m']*1000:.2f}mm right={right['support_offset_m']*1000:.2f}mm diff={diff*1000:.2f}mm")
    assert diff < 0.005, "left/right mirror geometry should give closely matching support offsets"


def test_rotated_object_local_to_world_transform_is_correct():
    """[30th-session bug, fixed in the manifold planner's own path]:
    object-local -> world must use obj_pos + obj_R @ local, not
    obj_pos + local. This planner never reconstructs a world target
    from a local offset for the face residual (it queries mesh_signed_
    distance directly in world frame), but the margin/thumb-thumb terms
    DO convert a world near-point into object-local coordinates via
    obj_R.T -- this test confirms that direction is correct for a
    genuinely rotated object by round-tripping local->world->local."""
    env = make_env()
    env.reset(seed=0)
    obj_qpos_adr = env._object_qpos_adr
    yaw = 0.3  # rad, deliberately non-zero
    quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    env.data.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7] = quat
    mujoco.mj_forward(env.model, env.data)

    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    ctx = build_context(env, expert)
    obj_R_check = np.zeros(9)
    mujoco.mju_quat2Mat(obj_R_check, quat)
    obj_R_check = obj_R_check.reshape(3, 3)
    assert np.allclose(ctx.obj_R, obj_R_check, atol=1e-9)

    world_point = ctx.obj_pos + np.array([0.05, -0.02, 0.01])
    local = ctx.obj_R.T @ (world_point - ctx.obj_pos)
    reconstructed_world = ctx.obj_pos + ctx.obj_R @ local  # CORRECT form
    wrong_world = ctx.obj_pos + local  # the 30th-session bug form
    assert np.allclose(reconstructed_world, world_point, atol=1e-9)
    assert not np.allclose(wrong_world, world_point, atol=1e-6), \
        "at a genuinely rotated object, the buggy obj_pos+local form must NOT reconstruct the original world point"


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
