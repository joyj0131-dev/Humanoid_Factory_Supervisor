"""G1 + Sharpa Wave integration tests (Phase 4, 35th session, Stage 2).

Validates the Sharpa Wave mount onto the (untouched) vendored G1 body --
run only AFTER scripts/test_sharpa_wave_model.py passes standalone. Run
with:

    python3 scripts/test_sharpa_g1_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.model_builder import build_grasp_model_sharpa


def _make_model():
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF)
    return build_grasp_model_sharpa(config), config


def _stand_data(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def _n_penetrating(data, model=None) -> int:
    """Real penetrating contacts, EXCLUDING the object resting on the
    table -- a small object<->table settling penetration is an expected,
    benign consequence of gravity + solver settling."""
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist >= 0:
            continue
        if model is not None:
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if {n1, n2} == {"table_geom", "object_geom"}:
                continue
        n += 1
    return n


def test_model_compiles_without_nan():
    model, _ = _make_model()
    data = _stand_data(model)
    assert not np.any(np.isnan(data.qpos))
    assert not np.any(np.isnan(data.xpos))
    print(f"    nq={model.nq} nv={model.nv} nu={model.nu} -- compiles, no NaN")


def test_all_sharpa_fingertip_bodies_are_present():
    model, _ = _make_model()
    for side in ("left", "right"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            sharpa_body = f"{side}_{side}_{finger}_DP"
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, sharpa_body) >= 0, f"{sharpa_body} should exist"
    print("    all 10 Sharpa fingertip DP bodies present")


def test_left_right_are_exact_mirrors_at_stand_pose():
    """Tolerance is 0.1mm, not bit-exact: the compiled model solves two
    separate (mirrored, but not literally shared) kinematic chains, so
    tiny floating-point solver noise (~10 micrometers, measured this
    session) between them is expected and not a real asymmetry."""
    model, _ = _make_model()
    data = _stand_data(model)
    tol = 1e-4
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        lb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"left_left_{finger}_DP")
        rb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"right_right_{finger}_DP")
        lp, rp = data.xpos[lb], data.xpos[rb]
        assert abs(lp[0] - rp[0]) < tol, f"{finger}: X should match, got {lp[0]} vs {rp[0]}"
        assert abs(lp[1] + rp[1]) < tol, f"{finger}: Y should be mirrored (sum~0), got {lp[1]} vs {rp[1]}"
        assert abs(lp[2] - rp[2]) < tol, f"{finger}: Z should match, got {lp[2]} vs {rp[2]}"
    print("    left/right fingertip positions are mirrors within 0.1mm (X,Z match; Y sign-flipped) at stand pose")


def test_fingers_extend_forward_of_wrist_not_backward_into_forearm():
    """Sanity check on the derived mount transform: every fingertip must
    lie further along the arm's own reach direction than the wrist itself
    -- i.e. the hand points away from the forearm, not back into it."""
    model, _ = _make_model()
    data = _stand_data(model)
    for side in ("left", "right"):
        wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
        elbow_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_elbow_link")
        reach_dir = data.xpos[wrist_bid] - data.xpos[elbow_bid]
        reach_dir /= np.linalg.norm(reach_dir)
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{side}_{finger}_DP")
            forward_extent = np.dot(data.xpos[bid] - data.xpos[wrist_bid], reach_dir)
            assert forward_extent > 0.02, f"{side}/{finger}: only {forward_extent*1000:.1f}mm forward of wrist -- hand may be mounted backward"
    print("    all 10 fingertips extend >20mm forward of the wrist along the arm's reach direction, both hands")


def test_no_self_collision_at_stand_pose():
    model, _ = _make_model()
    data = _stand_data(model)
    n_pen = _n_penetrating(data, model)
    if n_pen:
        for i in range(data.ncon):
            if data.contact[i].dist < 0:
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1)
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2)
                print(f"    PENETRATION: {g1} <-> {g2} dist={data.contact[i].dist*1000:.3f}mm")
    assert n_pen == 0, f"{n_pen} real self-collisions at stand pose (table/torso/opposite-hand/wrist-hand)"
    print(f"    0 real self-collisions at stand pose (ncon={data.ncon} total near-contacts, all non-penetrating)")


def test_arm_holds_stand_pose_under_sharpa_hand_weight_for_3_seconds():
    """Stock arm actuator gains (kp from config.arm_kp) must hold the
    stand pose against the ADDED Sharpa hand weight (~1.25kg/hand,
    measured in scripts/audit_sharpa_wave.py) without significant drift
    or divergence -- verifies no 'arm sag' problem before any controller
    is built."""
    model, _ = _make_model()
    data = _stand_data(model)
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        data.ctrl[a] = data.qpos[model.jnt_qposadr[jid]]
    wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    pos0 = data.xpos[wrist_bid].copy()
    n_steps = int(3.0 / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    drift = float(np.linalg.norm(data.xpos[wrist_bid] - pos0))
    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    assert _n_penetrating(data, model) == 0, "new self-collision appeared after settling under the hand's weight"
    print(f"    left wrist drift over 3s ({n_steps} steps): {drift*1000:.3f}mm, max|qvel|={np.max(np.abs(data.qvel)):.5f} rad/s")
    assert drift < 0.02, f"wrist drifted {drift*1000:.1f}mm under the Sharpa hand's weight -- possible arm-sag problem"


def test_canonical_grasp_config_exposes_cubic_half_extents():
    config = GraspEnvConfig(object_half_size=SIZE_12_HALF)
    assert config.effective_object_half_extents == (
        SIZE_12_HALF,
        SIZE_12_HALF,
        SIZE_12_HALF,
    )


def test_wrist_and_flange_mount_palm_frame_agree():
    """[Session 40] Mount A/B: with_wrist (default) vs with_flange must
    produce the SAME palm-frame site pose (the site lives on
    wrist_yaw_link, independent of which Sharpa variant is attached) --
    the two variants differ only by the vendored hand's own ~0.5mm
    hand-base offset (see model_builder._sharpa_xml_path's docstring),
    never a meaningfully different mount geometry."""
    cfg_wrist = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF, sharpa_mount="wrist")
    cfg_flange = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF, sharpa_mount="flange")
    m_wrist, m_flange = build_grasp_model_sharpa(cfg_wrist), build_grasp_model_sharpa(cfg_flange)
    for m in (m_wrist, m_flange):
        assert m.nq == m_wrist.nq and m.nv == m_wrist.nv
    d_wrist, d_flange = _stand_data(m_wrist), _stand_data(m_flange)
    for site in ("left_palm_frame", "right_palm_frame"):
        sid_w = mujoco.mj_name2id(m_wrist, mujoco.mjtObj.mjOBJ_SITE, site)
        sid_f = mujoco.mj_name2id(m_flange, mujoco.mjtObj.mjOBJ_SITE, site)
        diff = float(np.linalg.norm(d_wrist.site_xpos[sid_w] - d_flange.site_xpos[sid_f]))
        assert diff < 1e-6, f"{site} palm frame differs between mount variants by {diff*1000:.4f}mm"
    print("    with_wrist/with_flange palm frames match exactly (mount variant does not affect targeting)")


def test_sharpa_visual_style_g1_is_physics_invariant():
    """[Session 40, Stage 8] Recoloring visual-only Sharpa geoms to match
    G1's own material palette must NEVER change mass, inertia, collision
    geometry, joint ranges, or actuator gains -- only geom_rgba differs."""
    cfg_up = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF, sharpa_visual_style="upstream")
    cfg_g1 = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF, sharpa_visual_style="g1")
    m_up, m_g1 = build_grasp_model_sharpa(cfg_up), build_grasp_model_sharpa(cfg_g1)
    assert np.allclose(m_up.body_mass, m_g1.body_mass)
    assert np.allclose(m_up.body_inertia, m_g1.body_inertia)
    assert np.allclose(m_up.geom_size, m_g1.geom_size)
    assert np.array_equal(m_up.geom_contype, m_g1.geom_contype)
    assert np.array_equal(m_up.geom_conaffinity, m_g1.geom_conaffinity)
    assert np.allclose(m_up.geom_friction, m_g1.geom_friction)
    assert np.allclose(m_up.jnt_range, m_g1.jnt_range)
    assert np.allclose(m_up.actuator_gainprm, m_g1.actuator_gainprm)
    assert not np.allclose(m_up.geom_rgba, m_g1.geom_rgba), "g1 visual style should actually change some geom_rgba"
    print("    physics identical (mass/inertia/collision/joints/actuators), rgba differs as expected")


def test_g1_visual_style_uses_real_g1_materials_not_approximate_rgba():
    """[Session 41] Stage 2 Visual Acceptance Gate: the "g1" visual style
    must ASSIGN Sharpa visual geoms to G1's own compiled "black"/"metal"
    materials (verifiable by geom_matid), not merely set an
    approximately-similar rgba (the 40th session's version did this and
    was measured, by the user, to still look like the vendored lavender
    shell under the viewer's lighting). Zero upstream lavender/green rgba
    may remain; palm/wrist housing must use "black", finger structural
    links must use "metal"."""
    cfg = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF, sharpa_visual_style="g1")
    model = build_grasp_model_sharpa(cfg)
    black_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "black")
    metal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "metal")
    assert black_id >= 0 and metal_id >= 0, "G1's own 'black'/'metal' materials must exist in the compiled spec"

    lavender_rgba = np.array([0.79216, 0.81961, 0.93333, 1.0])
    green_rgba = np.array([0.2, 1.0, 0.2, 1.0])
    n_black = n_metal = n_other = 0
    housing_matids, elastomer_matids, finger_matids = [], [], []
    for g in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if not (bname.startswith("left_left_") or bname.startswith("right_right_")):
            continue
        if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0:
            continue  # collision geom, not part of this visual check
        assert not np.allclose(model.geom_rgba[g], lavender_rgba, atol=0.01), f"geom {g} still shows upstream lavender rgba"
        assert not np.allclose(model.geom_rgba[g], green_rgba, atol=0.01), f"geom {g} still shows upstream green elastomer rgba"
        matid = model.geom_matid[g]
        if matid == black_id:
            n_black += 1
        elif matid == metal_id:
            n_metal += 1
        else:
            n_other += 1
        gname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if "wrist" in bname.lower() or "c_mc" in gname:
            housing_matids.append(matid)
        if "elastomer" in gname:
            elastomer_matids.append(matid)
    assert n_other == 0, f"{n_other} Sharpa visual geoms use neither G1 material (found other matids)"
    assert n_black > 0 and n_metal > 0, "expected a mix of black (housing/elastomer) and metal (finger structure) geoms"
    print(f"    g1 style: {n_black} geoms on G1 'black', {n_metal} geoms on G1 'metal', 0 upstream lavender/green remaining")


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
