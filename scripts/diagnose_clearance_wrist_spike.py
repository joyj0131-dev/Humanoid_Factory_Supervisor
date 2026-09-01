"""Session 42, Stage 1: substep-level decomposition of the ~8rad/s wrist
qvel spike measured at ARM_LATERAL_CLEARANCE entry (Session 41).

For each tick across STABLE_START's last ticks and ARM_LATERAL_CLEARANCE's
first ~60 ticks, records per-joint (waist+both arms): actual qpos/qvel,
ctrl register (ctrl-implied ~ "target"), qfrc_bias, qfrc_constraint,
qfrc_actuator, joint-limit margin, and real contact pairs -- to
distinguish target-discontinuity vs actuator-underdamping vs collision-
impulse vs inertial-coupling-from-upstream-motion as the cause.

Usage:
    PYTHONPYCACHEPREFIX=/tmp/phase45_pycache python3 scripts/diagnose_clearance_wrist_spike.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig, SIZE_12_HALF
from humanoid_learning.envs.sharpa_grasp_env import SharpaGraspEnv
from humanoid_learning.envs import task_config as tc, whole_body_config as wbc
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import SharpaBimanualGraspExpert, BimanualGraspState

JOINT_NAMES = list(wbc.WAIST_JOINTS) + list(tc.LEFT_ARM_JOINTS) + list(tc.RIGHT_ARM_JOINTS)


def make_env():
    config = GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=SIZE_12_HALF,
                             arm_gravity_compensation=True)
    return SharpaGraspEnv(config)


def joint_dof_qpos_adr(model):
    dof_adr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
    qpos_adr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
    act_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES])
    lo = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][0] for n in JOINT_NAMES])
    hi = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][1] for n in JOINT_NAMES])
    return dof_adr, qpos_adr, act_ids, lo, hi


def real_contacts(model, data):
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist >= 0:
            continue
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1]) or "?"
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2]) or "?"
        out.append((b1, b2, float(c.dist)))
    return out


def main():
    env = make_env()
    expert = SharpaBimanualGraspExpert(env)
    env.reset(seed=0)
    dof_adr, qpos_adr, act_ids, lo, hi = joint_dof_qpos_adr(env.model)

    rows = []
    entered_clearance_at = None
    prev_arm_target = None
    for i in range(150):
        state_before = expert.state
        if state_before == BimanualGraspState.ARM_LATERAL_CLEARANCE and entered_clearance_at is None:
            entered_clearance_at = expert._total_step
        action = expert.step()
        env.step(action)

        ctrl_reg = np.concatenate([env._waist_target, env._arm_target])
        target_jump = None if prev_arm_target is None else float(np.max(np.abs(ctrl_reg - prev_arm_target)))
        prev_arm_target = ctrl_reg.copy()

        qpos = env.data.qpos[qpos_adr]
        qvel = env.data.qvel[dof_adr]
        qfrc_bias = env.data.qfrc_bias[dof_adr]
        qfrc_constraint = env.data.qfrc_constraint[dof_adr]
        qfrc_actuator = env.data.qfrc_actuator[dof_adr]
        margin = np.minimum(qpos - lo, hi - qpos)

        rows.append(dict(
            tick=i, state=state_before.name, ctrl=ctrl_reg.copy(), target_jump=target_jump,
            qpos=qpos.copy(), qvel=qvel.copy(), qfrc_bias=qfrc_bias.copy(),
            qfrc_constraint=qfrc_constraint.copy(), qfrc_actuator=qfrc_actuator.copy(),
            margin=margin.copy(), contacts=real_contacts(env.model, env.data),
        ))
        if state_before == BimanualGraspState.ARM_LATERAL_CLEARANCE and expert._state_step > 90 and expert.state != BimanualGraspState.ARM_LATERAL_CLEARANCE:
            break

    print(f"Entered ARM_LATERAL_CLEARANCE at total_step={entered_clearance_at}")

    # Find the tick with peak |qvel| across all joints
    peak_tick, peak_val, peak_joint = None, 0.0, None
    for r in rows:
        mx = np.max(np.abs(r["qvel"]))
        if mx > peak_val:
            peak_val = mx
            peak_tick = r["tick"]
            peak_joint = JOINT_NAMES[int(np.argmax(np.abs(r["qvel"])))]
    print(f"Peak |qvel|={peak_val:.3f} rad/s at tick={peak_tick}, joint={peak_joint}")

    def print_row(r, joints=None):
        joints = joints or JOINT_NAMES
        print(f"--- tick {r['tick']} state={r['state']} target_jump(max over all joints)={r['target_jump']} ---")
        for jn in joints:
            idx = JOINT_NAMES.index(jn)
            print(f"  {jn:28s} qpos={r['qpos'][idx]:+.4f} qvel={r['qvel'][idx]:+.4f} "
                  f"ctrl={r['ctrl'][idx]:+.4f} qfrc_bias={r['qfrc_bias'][idx]:+.3f} "
                  f"qfrc_constraint={r['qfrc_constraint'][idx]:+.3f} qfrc_actuator={r['qfrc_actuator'][idx]:+.3f} "
                  f"margin={r['margin'][idx]:.4f}")
        if r["contacts"]:
            print("  real contacts:", r["contacts"])

    focus_joints = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint",
                     "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]

    print("\n=== STABLE_START -> ARM_LATERAL_CLEARANCE transition (last 3 STABLE_START ticks + first 10 CLEARANCE ticks) ===")
    for r in rows:
        if (r["state"] == "STABLE_START" and r["tick"] >= 7) or (r["state"] == "ARM_LATERAL_CLEARANCE" and r["tick"] < entered_clearance_at + 10):
            print_row(r, focus_joints)

    print(f"\n=== window around peak qvel tick={peak_tick} (+-5 ticks) ===")
    for r in rows:
        if peak_tick is not None and abs(r["tick"] - peak_tick) <= 5:
            print_row(r, focus_joints)


if __name__ == "__main__":
    main()
