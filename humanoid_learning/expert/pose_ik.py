"""Full position+orientation (SO(3)) damped-least-squares IK for a palm
frame site -- see PROJECT_CONTEXT.md Phase 4 Grasp Track.

This replaces the single-axis cross-product orientation approximation used
in earlier Phase 4 sessions (which assumed the wrong local axis and never
gave a real SO(3) error). Orientation error here is the axis-angle vector
of R_desired @ R_current^T (the standard SO(3) log map), not a cross
product of one axis pair.

No MuJoCo dependency beyond Jacobian/site data already computed by the
caller -- mirrors ik_solver.py's style (pure linear algebra core), reused
for either arm via PalmController below.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


def so3_log(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector (world frame) of a rotation matrix R -- the
    standard SO(3) log map. Returns 0 for (near-)identity R."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    w_hat = (R - R.T) / (2.0 * np.sin(theta))
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
    return w * theta


def orientation_error(current_R: np.ndarray, desired_R: np.ndarray) -> np.ndarray:
    """World-frame axis-angle error that rotates current_R toward
    desired_R (small steps of this direction reduce the misalignment)."""
    return so3_log(desired_R @ current_R.T)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: the rotation matrix for a world-frame axis-angle
    vector w (inverse of so3_log). Used to build an INFLATED orientation
    target (desired_R overshot by a measured residual) the same way a
    Cartesian position target can be inflated by adding a vector."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-8:
        return np.eye(3)
    axis = w / theta
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def dls_pose_step(
    jacp: np.ndarray,
    jacr: np.ndarray,
    pos_err: np.ndarray,
    ori_err: np.ndarray,
    damping: float = 0.05,
    pos_weight: float = 1.0,
    ori_weight: float = 1.0,
) -> np.ndarray:
    """One damped-least-squares joint-space step toward a 6D pose target.
    Position and orientation errors are weighted independently (a large
    ori_weight relative to pos_weight prioritizes fixing orientation
    first, and vice versa) before being stacked into a single 6D DLS
    solve -- see PROJECT_CONTEXT.md Phase 4 for why single-axis alignment
    was insufficient."""
    err6 = np.concatenate([pos_weight * pos_err, ori_weight * ori_err])
    J6 = np.vstack([jacp, jacr])
    jjt = J6 @ J6.T
    damped = jjt + (damping**2) * np.eye(jjt.shape[0])
    return J6.T @ np.linalg.solve(damped, err6)


@dataclass
class PoseConvergence:
    pos_error_norm: float
    ori_error_norm: float


class PalmController:
    """Position+orientation controller for one arm's palm-frame site.
    Holds only the site/joint indices needed for Jacobian computation --
    not a general IK framework, just enough to drive one palm to a 6D
    target with rate-limited, joint-limit-safe steps."""

    def __init__(self, model: mujoco.MjModel, palm_site_id: int, arm_dof_adr: np.ndarray):
        self.model = model
        self.palm_site_id = palm_site_id
        self.arm_dof_adr = arm_dof_adr

    def current_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        pos = data.site_xpos[self.palm_site_id].copy()
        R = data.site_xmat[self.palm_site_id].reshape(3, 3).copy()
        return pos, R

    def step_toward(
        self,
        data: mujoco.MjData,
        target_pos: np.ndarray,
        target_R: np.ndarray,
        gain: float = 0.3,
        damping: float = 0.05,
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
        max_dq: float = 0.5,
        arm_qpos: np.ndarray | None = None,
        rest_qpos: np.ndarray | None = None,
        null_gain: float = 0.0,
    ) -> tuple[np.ndarray, PoseConvergence]:
        """Returns (7-dim joint-space delta for this arm, convergence info).
        The delta is NOT yet scaled into an env action -- callers divide by
        their own action_scale, matching the rest of this project's Joint
        Position Delta convention.

        ``rest_qpos``/``null_gain`` add a secondary elbow-down/rest-posture
        objective in the task Jacobian's null space -- found necessary
        during the redesign (Phase 4 Grasp Track) when the primary 6D
        solve alone drove ``left_shoulder_roll_joint`` to its hard limit
        (2.25 rad) while other joints thrashed trying to compensate,
        producing the large unnecessary swings the previous state machine
        exhibited. With no joint-limit-avoidance term the DLS solve is free
        to spend the whole 7-DOF budget approaching a limit even after
        that stops helping the primary task; the null-space pull toward a
        neutral rest pose gives it something better to do with the
        redundant DOF instead."""
        cur_pos, cur_R = self.current_pose(data)
        pos_err = target_pos - cur_pos
        ori_err = orientation_error(cur_R, target_R)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.palm_site_id)
        jacp = jacp[:, self.arm_dof_adr]
        jacr = jacr[:, self.arm_dof_adr]

        dq = dls_pose_step(jacp, jacr, pos_err, ori_err, damping, pos_weight, ori_weight) * gain

        if rest_qpos is not None and null_gain > 0 and arm_qpos is not None:
            J6 = np.vstack([jacp, jacr])
            jjt = J6 @ J6.T
            damped_inv = np.linalg.inv(jjt + (damping**2) * np.eye(jjt.shape[0]))
            jpinv = J6.T @ damped_inv  # (n_joints, 6)
            null_proj = np.eye(J6.shape[1]) - jpinv @ J6
            dq_null = null_gain * (rest_qpos - arm_qpos)
            dq = dq + null_proj @ dq_null

        dq = np.clip(dq, -max_dq, max_dq)
        return dq, PoseConvergence(pos_error_norm=float(np.linalg.norm(pos_err)), ori_error_norm=float(np.linalg.norm(ori_err)))


def multi_start_feasibility(
    controller: PalmController,
    data: mujoco.MjData,
    target_pos: np.ndarray,
    target_R: np.ndarray,
    initial_qpos_candidates: list[np.ndarray],
    arm_qpos_adr: np.ndarray,
    steps: int = 150,
    gain: float = 0.3,
    damping: float = 0.05,
) -> list[PoseConvergence]:
    """Runs IK convergence from several different initial arm joint
    configurations toward the SAME target, so a single unlucky start pose
    is never mistaken for kinematic infeasibility (PROJECT_CONTEXT.md
    Phase 4, Section G/H instruction). Caller is responsible for actually
    resetting/stepping the sim between candidates; this function just runs
    the joint-space iteration on the already-positioned `data` and reports
    the final convergence for each candidate start.
    """
    results = []
    for q0 in initial_qpos_candidates:
        data.qpos[arm_qpos_adr] = q0
        mujoco.mj_forward(controller.model, data)
        conv = PoseConvergence(pos_error_norm=float("inf"), ori_error_norm=float("inf"))
        for _ in range(steps):
            dq, conv = controller.step_toward(data, target_pos, target_R, gain=gain, damping=damping)
            data.qpos[arm_qpos_adr] = np.clip(data.qpos[arm_qpos_adr] + dq, -np.pi, np.pi)
            mujoco.mj_forward(controller.model, data)
        results.append(conv)
    return results
