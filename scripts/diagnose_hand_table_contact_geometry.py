"""Substep-level contact-pair audit for the two hand-table forces flagged
this session: ARM_LATERAL_CLEARANCE's ~27.13N peak and FOREARM_SIDE_
DESCEND's ~18.21N peak. Section 6/7 of this session's mandate requires
identifying the EXACT geom pair, whether it is fingertip/thumb/palm-
housing, whether the contact is a single-tick transient or a sustained
near-steady-state contact, and the causal order (target/waypoint change
-> clearance shrinks -> contact appears -> qfrc_constraint jumps ->
wrist qvel rises) BEFORE any controller change is made.

Read-only: never modifies SharpaBimanualGraspExpert/SharpaGraspEnv.
Monkeypatches mujoco.mj_step to record SUBSTEP-level state (env.step()
itself loops mj_step per physics substep internally; wrapping mj_step is
the only way to see inside that loop without editing source).

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/diagnose_hand_table_contact_geometry.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import (
    SIDES,
    BimanualGraspState,
    SharpaBimanualGraspExpert,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "hand_table_contact_geometry"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WATCH_STATES = (BimanualGraspState.ARM_LATERAL_CLEARANCE, BimanualGraspState.FOREARM_SIDE_DESCEND)


def make_env() -> SharpaGraspEnv:
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             max_episode_steps=3000, arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def hand_table_contacts(model, data) -> list[dict]:
    """All hand<->table contacts THIS substep, with geom names, dist, force, point, normal."""
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or ""
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or ""
        is_hand_table = (("table" in b1 and (b2.startswith("left_left_") or b2.startswith("right_right_"))) or
                          ("table" in b2 and (b1.startswith("left_left_") or b1.startswith("right_right_"))))
        if not is_hand_table:
            continue
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
        force6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force6)
        out.append({
            "body1": b1, "body2": b2, "geom1": g1, "geom2": g2,
            "dist": float(c.dist), "force_n": float(np.linalg.norm(force6[:3])),
            "pos": c.pos.copy(), "normal": np.asarray(c.frame[:3], dtype=np.float64).copy(),
        })
    return out


def main() -> None:
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)

    substep_log: list[dict] = []
    wrist_dof = np.concatenate([env._arm_dof_adr[4:7], env._arm_dof_adr[11:14]])

    orig_mj_step = mujoco.mj_step
    state_holder = {"state": expert.state, "state_step": 0, "waypoint": -1}

    def wrapped_mj_step(model, data):
        orig_mj_step(model, data)
        cur_state = state_holder["state"]
        if cur_state not in WATCH_STATES:
            return
        contacts = hand_table_contacts(model, data)
        raw_wrist_qvel = float(np.max(np.abs(data.qvel[wrist_dof])))
        qfrc_constraint_wrist = float(np.max(np.abs(data.qfrc_constraint[wrist_dof])))
        row = {
            "state": cur_state.name, "state_step": state_holder["state_step"],
            "waypoint": state_holder["waypoint"],
            "n_hand_table_contacts": len(contacts),
            "max_hand_table_force": max((c["force_n"] for c in contacts), default=0.0),
            "min_hand_table_dist": min((c["dist"] for c in contacts), default=float("nan")),
            "contact_bodies": ";".join(sorted({f"{c['body1']}<->{c['body2']}" for c in contacts})),
            "contact_geoms": ";".join(sorted({f"{c['geom1']}<->{c['geom2']}" for c in contacts})),
            "raw_wrist_qvel": raw_wrist_qvel,
            "qfrc_constraint_wrist_max": qfrc_constraint_wrist,
        }
        substep_log.append(row)

    mujoco.mj_step = wrapped_mj_step

    obs, info = env.reset(seed=0)
    try:
        for _ in range(2000):
            if expert.state in (BimanualGraspState.SUCCESS, BimanualGraspState.FAILURE):
                break
            state_holder["state"] = expert.state
            state_holder["state_step"] = expert._state_step
            state_holder["waypoint"] = getattr(expert, "_side_descend_waypoint", -1)
            action = expert.step()
            obs, r, term, trunc, info = env.step(action)
            if info.get("unstable") or trunc:
                break
    finally:
        mujoco.mj_step = orig_mj_step

    print(f"Final state={expert.state.name} reason={expert.failure_reason}")

    for s in WATCH_STATES:
        rows = [r for r in substep_log if r["state"] == s.name]
        print(f"\n=== {s.name}: {len(rows)} substeps recorded ===")
        if not rows:
            continue
        contact_rows = [r for r in rows if r["n_hand_table_contacts"] > 0]
        print(f"  substeps WITH hand-table contact: {len(contact_rows)} / {len(rows)}")
        if not contact_rows:
            print("  (no hand-table contact this state)")
            continue
        peak_row = max(contact_rows, key=lambda r: r["max_hand_table_force"])
        print(f"  PEAK force={peak_row['max_hand_table_force']:.2f}N at state_step={peak_row['state_step']} "
              f"wp={peak_row['waypoint']}: bodies={peak_row['contact_bodies']}")
        print(f"    geoms={peak_row['contact_geoms']}")
        print(f"    wrist_qvel_at_peak={peak_row['raw_wrist_qvel']:.3f}rad/s "
              f"qfrc_constraint_wrist={peak_row['qfrc_constraint_wrist_max']:.2f}")

        # unique body pairs seen overall + their max force
        pair_max: dict[str, float] = {}
        for r in contact_rows:
            for pair in r["contact_bodies"].split(";"):
                pair_max[pair] = max(pair_max.get(pair, 0.0), r["max_hand_table_force"])
        print("  all body pairs seen (with their OWN max force across all substeps, not necessarily simultaneous):")
        for pair, f in sorted(pair_max.items(), key=lambda kv: -kv[1]):
            print(f"    {pair}: max {f:.2f}N")

        # causal check: does wrist_qvel rise happen in the SAME substep as first
        # contact, or lag by some substeps? Find first contact substep and look
        # at qvel trend around it.
        first_contact_idx = next(i for i, r in enumerate(rows) if r["n_hand_table_contacts"] > 0)
        window = rows[max(0, first_contact_idx - 3): first_contact_idx + 6]
        print("  qvel/contact trace around FIRST contact this state (substep-indexed):")
        for r in window:
            print(f"    ss_state_step={r['state_step']} wp={r['waypoint']} "
                  f"contacts={r['n_hand_table_contacts']} force={r['max_hand_table_force']:.2f}N "
                  f"wrist_qvel={r['raw_wrist_qvel']:.3f} qfrc_constraint={r['qfrc_constraint_wrist_max']:.2f}")

        # steady-state vs transient: what fraction of LATE-state substeps still show contact?
        if len(rows) > 20:
            tail = rows[-20:]
            tail_contact_frac = sum(1 for r in tail if r["n_hand_table_contacts"] > 0) / len(tail)
            print(f"  contact fraction in LAST 20 substeps of this state: {tail_contact_frac*100:.0f}% "
                  f"({'sustained/steady-state' if tail_contact_frac > 0.5 else 'transient'})")

    csv_path = RESULTS_DIR / "substep_contact_trace.csv"
    if substep_log:
        keys = sorted({k for row in substep_log for k in row})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            for row in substep_log:
                w.writerow(row)
    print(f"\nRaw substep trace written to {csv_path}")


if __name__ == "__main__":
    main()
