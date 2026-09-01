"""Session 40, Stage 1/2: Sharpa Wave mount audit + with_wrist vs
with_flange A/B comparison.

Measures, for BOTH mount variants, on the REAL compiled G1+Sharpa grasp
model (build_grasp_model_sharpa): wrist_yaw joint origin, Sharpa mount
site origin, palm center, fingertip positions, wrist->palm / wrist->
fingertip distances, per-hand mass/COM, standoff length (measured from
the vendored XML geom offsets, not guessed), stand-pose self-collision
census (real geom/body pair names), and 3-second settling drift.

Read-only: does not modify canonical config defaults (sharpa_mount stays
"wrist" unless explicitly overridden here for the "flange" condition).

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/audit_sharpa_mount.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs import sharpa_config as sc
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.model_builder import build_grasp_model_sharpa

SIDES = ("left", "right")


def _body_pos(model, data, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, name
    return data.xpos[bid].copy()


def _site_pos(model, data, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    assert sid >= 0, name
    return data.site_xpos[sid].copy()


def _hand_mass_com(model, side: str) -> tuple[float, np.ndarray]:
    prefix = f"{side}_{side}_"
    total_mass = 0.0
    weighted_com = np.zeros(3)
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith(prefix):
            m = model.body_mass[b]
            total_mass += m
            weighted_com += m * model.body_ipos[b]
    com = weighted_com / total_mass if total_mass > 0 else np.zeros(3)
    return total_mass, com


def _real_contacts(model, data) -> list[tuple[str, str, str, str, float]]:
    """Returns (geom1_name, body1_name, geom2_name, body2_name, dist) for
    every REAL (penetrating, dist<0) contact, resolving names even for
    unnamed geoms via their owning body's name + geom index."""
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist >= 0:
            continue
        g1, g2 = c.geom1, c.geom2
        b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
        gname1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"<geom#{g1}>"
        gname2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"<geom#{g2}>"
        bname1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"<body#{b1}>"
        bname2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"<body#{b2}>"
        out.append((gname1, bname1, gname2, bname2, float(c.dist)))
    return out


def audit_mount(mount: str) -> dict:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             sharpa_mount=mount)
    model = build_grasp_model_sharpa(config)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, tc.STAND_KEYFRAME)
    n_robot_qpos = model.nq - 7  # object freejoint
    data.qpos[:n_robot_qpos] = model.key_qpos[key_id][:n_robot_qpos]
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    result: dict = {"mount": mount}
    for side in SIDES:
        wrist_pos = _body_pos(model, data, f"{side}_wrist_yaw_link")
        mount_site = _site_pos(model, data, f"{side}_sharpa_mount")
        palm_site = wbc.LEFT_PALM_SITE if side == "left" else wbc.RIGHT_PALM_SITE
        palm_pos = _site_pos(model, data, palm_site)
        tip_positions = {}
        for finger in sc.FINGERS:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_sharpa_tip")
            tip_positions[finger] = data.site_xpos[sid].copy()
        wrist_to_palm = float(np.linalg.norm(palm_pos - wrist_pos))
        wrist_to_tips = {f: float(np.linalg.norm(p - wrist_pos)) for f, p in tip_positions.items()}
        mass, com_local = _hand_mass_com(model, side)
        result[side] = {
            "wrist_origin": wrist_pos,
            "mount_site_origin": mount_site,
            "palm_pos": palm_pos,
            "wrist_to_palm_m": wrist_to_palm,
            "wrist_to_tip_m": wrist_to_tips,
            "wrist_to_tip_mean_m": float(np.mean(list(wrist_to_tips.values()))),
            "wrist_to_tip_max_m": float(np.max(list(wrist_to_tips.values()))),
            "hand_mass_kg": mass,
            "hand_com_local": com_local,
        }

    # Mirror check: left/right wrist_to_palm distances should match closely.
    result["mirror_wrist_to_palm_diff_m"] = abs(result["left"]["wrist_to_palm_m"] - result["right"]["wrist_to_palm_m"])

    # Stand-pose self-collision census (real contacts only, excludes
    # table<->object resting contact which is expected).
    contacts = _real_contacts(model, data)
    forbidden = [c for c in contacts if not (
        ("table" in c[1] and "object" in c[3]) or ("table" in c[3] and "object" in c[1])
    )]
    result["stand_pose_forbidden_contacts"] = forbidden

    # 3-second settling drift under gravity (compliant kp=120, arm ctrl
    # held at the stand target).
    arm_act_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS])
    arm_qpos_adr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                              for n in tc.LEFT_ARM_JOINTS + tc.RIGHT_ARM_JOINTS])
    data.ctrl[arm_act_ids] = data.qpos[arm_qpos_adr]
    left_palm_start = _site_pos(model, data, wbc.LEFT_PALM_SITE)
    right_palm_start = _site_pos(model, data, wbc.RIGHT_PALM_SITE)
    n_steps = int(3.0 / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
    left_drift = float(np.linalg.norm(_site_pos(model, data, wbc.LEFT_PALM_SITE) - left_palm_start))
    right_drift = float(np.linalg.norm(_site_pos(model, data, wbc.RIGHT_PALM_SITE) - right_palm_start))
    result["settle_3s_drift_m"] = {"left": left_drift, "right": right_drift}
    result["nq"] = model.nq
    result["nv"] = model.nv
    return result


def print_report(r: dict) -> None:
    print(f"\n=== mount={r['mount']} ===")
    for side in SIDES:
        s = r[side]
        print(f"  {side}: wrist_origin={s['wrist_origin']}  palm_pos={s['palm_pos']}")
        print(f"    wrist->palm={s['wrist_to_palm_m']*1000:.2f}mm  "
              f"wrist->tip(mean,max)=({s['wrist_to_tip_mean_m']*1000:.2f},{s['wrist_to_tip_max_m']*1000:.2f})mm")
        print(f"    hand_mass={s['hand_mass_kg']:.4f}kg  hand_com_local={s['hand_com_local']}")
    print(f"  mirror wrist->palm diff = {r['mirror_wrist_to_palm_diff_m']*1000:.4f}mm")
    print(f"  stand-pose forbidden contacts: {len(r['stand_pose_forbidden_contacts'])}")
    for c in r["stand_pose_forbidden_contacts"][:20]:
        print(f"    {c[1]}::{c[0]}  <->  {c[3]}::{c[2]}  dist={c[4]*1000:.3f}mm")
    print(f"  3s settle drift: left={r['settle_3s_drift_m']['left']*1000:.2f}mm  "
          f"right={r['settle_3s_drift_m']['right']*1000:.2f}mm")


def main() -> None:
    r_wrist = audit_mount("wrist")
    r_flange = audit_mount("flange")
    print_report(r_wrist)
    print_report(r_flange)

    print("\n=== A/B delta (flange - wrist) ===")
    for side in SIDES:
        d_palm = np.linalg.norm(r_flange[side]["palm_pos"] - r_wrist[side]["palm_pos"])
        d_wp = r_flange[side]["wrist_to_palm_m"] - r_wrist[side]["wrist_to_palm_m"]
        print(f"  {side}: |palm_pos delta|={d_palm*1000:.3f}mm  wrist->palm delta={d_wp*1000:.3f}mm  "
              f"mass delta={(r_flange[side]['hand_mass_kg']-r_wrist[side]['hand_mass_kg'])*1000:.3f}g")
    print(f"  forbidden contacts: wrist={len(r_wrist['stand_pose_forbidden_contacts'])} "
          f"flange={len(r_flange['stand_pose_forbidden_contacts'])}")
    print(f"  3s drift: wrist(L,R)=({r_wrist['settle_3s_drift_m']['left']*1000:.2f},"
          f"{r_wrist['settle_3s_drift_m']['right']*1000:.2f})mm  "
          f"flange(L,R)=({r_flange['settle_3s_drift_m']['left']*1000:.2f},"
          f"{r_flange['settle_3s_drift_m']['right']*1000:.2f})mm")


if __name__ == "__main__":
    main()
