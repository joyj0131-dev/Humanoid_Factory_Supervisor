"""Sharpa Wave hand model audit (Phase 4, 35th session, Stage 0/1).

Read-only introspection of the compiled MjModel for both hands -- every
number printed here is read directly from the compiled model, never
assumed from the vendor's documentation or product page. Run with:

    python3 scripts/audit_sharpa_wave.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

ASSET_DIR = PROJECT_ROOT / "assets" / "robots" / "sharpa_wave"
EXPECTED_FINGER_JOINT_COUNTS = {"thumb": 5, "index": 4, "middle": 4, "ring": 4, "pinky": 5}


def audit_hand(side: str, variant: str) -> None:
    xml_path = ASSET_DIR / f"{side}_sharpa_wave" / f"{side}_sharpa_wave_{variant}.xml"
    print(f"\n{'=' * 70}\n{side} hand, variant={variant}: {xml_path}\n{'=' * 70}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom} "
          f"nsensor={model.nsensor} nsite={model.nsite}")
    print(f"qpos has NaN: {bool(np.any(np.isnan(data.qpos)))}")

    print(f"\n--- {model.njnt} joints ---")
    finger_counts = {f: 0 for f in EXPECTED_FINGER_JOINT_COUNTS}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = model.jnt_range[j]
        qpos0 = model.qpos0[model.jnt_qposadr[j]]
        in_range = lo <= qpos0 <= hi
        print(f"  {name:28s} range=[{lo:+.4f},{hi:+.4f}] qpos0={qpos0:+.4f} in_range={in_range}")
        for finger in finger_counts:
            if f"_{finger}_" in name:
                finger_counts[finger] += 1
    print(f"  per-finger joint counts: {finger_counts} (expected {EXPECTED_FINGER_JOINT_COUNTS})")
    assert finger_counts == EXPECTED_FINGER_JOINT_COUNTS, "finger joint count mismatch"

    print(f"\n--- {model.nu} actuators ---")
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        lo, hi = model.actuator_ctrlrange[a]
        print(f"  {name:32s} ctrlrange=[{lo:+.4f},{hi:+.4f}]")

    print(f"\n--- {model.nbody} bodies ---")
    total_mass = 0.0
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        mass = model.body_mass[b]
        total_mass += mass
        print(f"  {name or '(unnamed)':28s} mass={mass:.5f}kg")
    print(f"  TOTAL hand mass: {total_mass:.4f}kg")

    print(f"\n--- {model.nsensor} sensors ---")
    if model.nsensor == 0:
        print("  NONE. The real-hardware dynamic tactile array (240x240 px/fingertip claimed by the vendor)"
              " is NOT present in this MJCF. Any per-finger force/contact observation in this project must"
              " come from mj_contactForce/data.contact, and must be labeled as contact-force-based, never"
              " as a simulated hardware tactile array.")
    else:
        for s in range(model.nsensor):
            print(f"  {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s)}")

    print(f"\n--- elastomer collision geoms ---")
    elastomer_geoms = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)
                       if "elastomer" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                       and model.geom_contype[g] == 1]
    for name in elastomer_geoms:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        print(f"  {name:28s} solref={model.geom_solref[gid]}")
    print(f"  {len(elastomer_geoms)} real (contype=1) elastomer collision geoms found (one per fingertip expected: 5)")

    print(f"\n--- self-collision <contact><exclude> pairs ---")
    print(f"  {model.nexclude} exclude pairs defined")

    print(f"\n--- neutral-pose (qpos0) fingertip world positions ---")
    dp_bodies = {f: f"{side}_{f}_DP" for f in ("index", "middle", "ring", "pinky")}
    dp_bodies["thumb"] = f"{side}_thumb_DP"
    for finger, body_name in dp_bodies.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            print(f"  {finger}: body {body_name!r} NOT FOUND")
            continue
        print(f"  {finger:8s} ({body_name}): world pos = {np.round(data.xpos[bid], 4)}")


def main() -> None:
    for side in ("left", "right"):
        for variant in ("with_wrist",):
            audit_hand(side, variant)
    print("\nAudit complete -- all numbers above read directly from the compiled MjModel.")


if __name__ == "__main__":
    main()
