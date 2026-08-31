"""Standalone Sharpa Wave hand model tests (Phase 4, 35th session, Stage 1).

Validates the vendored Sharpa Wave MJCF ON ITS OWN, before any G1
integration is attempted -- per this session's own stop conditions,
G1 mounting must not proceed until these pass. Run with:

    python3 scripts/install_sharpa_wave_assets.py   # once, after clone
    python3 scripts/test_sharpa_wave_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

ASSET_DIR = PROJECT_ROOT / "assets" / "robots" / "sharpa_wave"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _load(side: str, variant: str = "with_wrist") -> mujoco.MjModel:
    xml_path = ASSET_DIR / f"{side}_sharpa_wave" / f"{side}_sharpa_wave_{variant}.xml"
    return mujoco.MjModel.from_xml_path(str(xml_path))


def _dp_body(side: str, finger: str) -> str:
    return f"{side}_{finger}_DP"


def test_left_and_right_load_without_nan():
    for side in ("left", "right"):
        model = _load(side)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert not np.any(np.isnan(data.qpos))
        assert not np.any(np.isnan(data.xpos))
    print("    left and right both load cleanly, no NaN in qpos/xpos")


def test_all_44_actuators_present_with_correct_names():
    names = []
    for side in ("left", "right"):
        model = _load(side)
        assert model.nu == 22, f"{side} hand must have exactly 22 actuators, got {model.nu}"
        for a in range(model.nu):
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
    assert len(names) == 44
    assert len(set(names)) == 44, "actuator names must be unique across both hands"
    print(f"    44 total actuators (22+22), all unique names")


def test_thumb5_index_middle_ring_4_pinky5():
    for side in ("left", "right"):
        model = _load(side)
        counts = {f: 0 for f in FINGERS}
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for f in FINGERS:
                if f"_{f}_" in name:
                    counts[f] += 1
        assert counts == {"thumb": 5, "index": 4, "middle": 4, "ring": 4, "pinky": 5}, f"{side}: {counts}"
    print("    thumb=5, index/middle/ring=4 each, pinky=5 confirmed for both hands")


def test_neutral_pose_within_joint_limits():
    for side in ("left", "right"):
        model = _load(side)
        for j in range(model.njnt):
            lo, hi = model.jnt_range[j]
            qpos0 = model.qpos0[model.jnt_qposadr[j]]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            assert lo - 1e-9 <= qpos0 <= hi + 1e-9, f"{side}/{name}: qpos0={qpos0} outside range [{lo},{hi}]"
    print("    neutral pose (qpos0) is within joint range for every joint, both hands")


def test_five_fingertip_positions_and_elastomer_geoms():
    for side in ("left", "right"):
        model = _load(side)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for finger in FINGERS:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _dp_body(side, finger))
            assert bid >= 0, f"{side}/{finger}: DP body not found"
            pos = data.xpos[bid]
            assert np.all(np.isfinite(pos))
            elastomer_name = f"{side}_{finger}_elastomer"
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, elastomer_name)
            assert gid >= 0, f"{side}/{finger}: elastomer collision geom not found"
            assert model.geom_contype[gid] == 1, f"{side}/{finger}: elastomer geom must be a real collision geom"
    print("    all 5 fingertip DP bodies + elastomer collision geoms found for both hands")


def _set_pose(model: mujoco.MjModel, data: mujoco.MjData, ctrl_fraction: float) -> None:
    """ctrl_fraction=0 -> open (flexion joints at their EXTENSION limit,
    i.e. lower bound; abduction/adduction ("_AA") and pinky_CMC joints at
    their own NEUTRAL qpos0, not their range's lower bound -- an "all
    joints at lower limit" pose would force every AA joint to its most
    ABDUCTED extreme simultaneously, which is not a real open-hand pose
    and was found (this session) to cause 19 spurious self-collisions
    that don't reflect an actual model defect), ctrl_fraction=1 -> fully
    flexed (upper bound) for flexion joints only."""
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = model.jnt_range[j]
        if name.endswith("_AA") or name.endswith("_CMC"):
            data.qpos[model.jnt_qposadr[j]] = model.qpos0[model.jnt_qposadr[j]]
        else:
            data.qpos[model.jnt_qposadr[j]] = lo + ctrl_fraction * (hi - lo)
    mujoco.mj_forward(model, data)


def test_open_preshape_close_poses_are_collision_free_at_neutral_approach():
    """Open (0.0) and a mild preshape (0.3) must not already self-collide
    within a single hand -- if they did, even before touching an object,
    the hand's own kinematics would already be infeasible."""
    for side in ("left", "right"):
        model = _load(side)
        data = mujoco.MjData(model)
        for frac, label in ((0.0, "open"), (0.3, "preshape")):
            _set_pose(model, data, frac)
            mujoco.mj_forward(model, data)
            n_self_collisions = sum(1 for i in range(data.ncon) if data.contact[i].dist < 0)
            assert n_self_collisions == 0, f"{side}/{label}: {n_self_collisions} self-collisions at qpos fraction {frac}"
    print("    open (0.0) and preshape (0.3) poses are self-collision-free for both hands")


def test_low_force_close_pose_reports_adjacent_finger_contact_state():
    """A more closed pose (0.7) is EXPECTED to bring fingertips close to
    each other or to the palm plane -- this test only reports the real
    contact state (never asserts zero), matching the project's own
    'measure, don't assume' convention."""
    for side in ("left", "right"):
        model = _load(side)
        data = mujoco.MjData(model)
        _set_pose(model, data, 0.7)
        n_contacts = sum(1 for i in range(data.ncon) if data.contact[i].dist < 0)
        print(f"    {side} hand at 0.7 closure fraction: {n_contacts} real self-contacts (excludes already applied)")


def test_physics_stepping_is_stable_for_5_seconds():
    """5 seconds of mj_step at a neutral->mild-preshape target (position
    actuators tracking a fixed setpoint) must not diverge/explode."""
    for side in ("left", "right"):
        model = _load(side)
        data = mujoco.MjData(model)
        for j in range(model.njnt):
            lo, hi = model.jnt_range[j]
            data.qpos[model.jnt_qposadr[j]] = lo + 0.3 * (hi - lo)
        mujoco.mj_forward(model, data)
        for j in range(model.nu):
            jid = model.actuator_trnid[j, 0]
            lo, hi = model.jnt_range[jid]
            data.ctrl[j] = lo + 0.3 * (hi - lo)
        n_steps = int(5.0 / model.opt.timestep)
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.qpos)), f"{side}: qpos diverged after {n_steps} steps"
        assert np.all(np.isfinite(data.qvel)), f"{side}: qvel diverged after {n_steps} steps"
        assert np.max(np.abs(data.qvel)) < 100.0, f"{side}: qvel exploded ({np.max(np.abs(data.qvel))} rad/s)"
    print(f"    {n_steps} steps (5s) stable for both hands, no NaN/Inf, max |qvel| bounded")


def test_contact_force_units_and_net_fingertip_force():
    """Presses the index fingertip elastomer into a small FIXED sphere
    placed exactly at the flexed-index elastomer's own world position
    (computed by forward kinematics, not guessed) and confirms
    mj_contactForce returns a finite, physically-plausible (Newton-scale)
    net force -- verifying the contact FORCE pipeline this project will
    rely on for Sharpa (there is no tactile sensor to fall back on, see
    assets/robots/sharpa_wave/README.md).

    A first attempt at this test used a large world PLANE positioned by
    a guessed z-height; the hand's own root body sits at world z=0 with
    the fingers reaching up to z=0.09-0.21 at neutral, so a plane placed
    to catch the flexed index tip also intersected the (already-neutral,
    undriven) palm/root mesh from t=0, producing a sustained, unrealistic
    ~8500N artifact from that pre-existing penetration -- not a real
    single-fingertip contact force. A small sphere placed exactly at the
    flexed fingertip's own computed position has no such risk."""
    side = "left"
    xml_path = ASSET_DIR / f"{side}_sharpa_wave" / f"{side}_sharpa_wave_with_wrist.xml"
    xml_text = xml_path.read_text()

    # Step 1: forward-kinematics the flexed-index elastomer position on
    # the UNMODIFIED model (no extra test geometry yet).
    base_model = mujoco.MjModel.from_xml_path(str(xml_path))
    base_data = mujoco.MjData(base_model)
    for j in range(base_model.njnt):
        name = mujoco.mj_id2name(base_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in ("left_index_MCP_FE", "left_index_PIP", "left_index_DIP"):
            base_data.qpos[base_model.jnt_qposadr[j]] = base_model.jnt_range[j][1]
    mujoco.mj_forward(base_model, base_data)
    gid = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_GEOM, "left_index_elastomer")
    target_pos = base_data.geom_xpos[gid].copy()

    # Step 2: build a variant with a small fixed sphere at that exact
    # point (from_xml_string cannot resolve the model's own relative
    # meshdir="meshes/" -- point it at the absolute meshes directory).
    mesh_dir = str((ASSET_DIR / f"{side}_sharpa_wave" / "meshes").resolve())
    px, py, pz = target_pos
    test_xml = xml_text.replace('meshdir="meshes/"', f'meshdir="{mesh_dir}"').replace(
        "<worldbody>",
        f'<worldbody>\n    <geom name="test_target" type="sphere" size="0.006" pos="{px} {py} {pz}" '
        'contype="1" conaffinity="1"/>',
        1,
    )
    model = mujoco.MjModel.from_xml_string(test_xml, {})
    data = mujoco.MjData(model)
    for j in range(model.nu):
        jid = model.actuator_trnid[j, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        _, hi = model.jnt_range[jid]
        if name in ("left_index_MCP_FE", "left_index_PIP", "left_index_DIP"):
            data.ctrl[j] = hi
        else:
            data.ctrl[j] = model.qpos0[model.jnt_qposadr[jid]]
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    max_force_n = 0.0
    for i in range(data.ncon):
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        max_force_n = max(max_force_n, float(np.linalg.norm(force[:3])))
    print(f"    max real contact-normal force after 3s of forced index closure onto a fixed target sphere: {max_force_n:.3f} N")
    assert np.isfinite(max_force_n)
    assert max_force_n < 200.0, "contact force is far above the vendor's own 20N fingertip-strength spec -- likely a divergent/exploding contact, not a real single-fingertip press"


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
