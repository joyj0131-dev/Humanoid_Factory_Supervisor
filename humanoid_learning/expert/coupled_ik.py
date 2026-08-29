"""Coupled bilateral waist-aware IK (Phase 4 Grasp Track, Coupled Bilateral
Waist-Aware IK Validation session).

Root-cause finding this solver exists to address: the previous session's
per-arm-only IK (each arm's Jacobian built from ONLY its own 7 joints)
showed a 6cm-object APPROACH target converging cleanly in isolation
(orientation error ~0.1 degree with the other arm frozen) but getting
stuck at a persistent ~30-150 degree orientation error when both arms run
together. Direct measurement found the waist (3 DoF, kp=500) drifting up
to ~2.5 degrees from BOTH arms' reaction torque while each arm's IK
computes a correction that assumes the waist is perfectly still --
because neither arm's Jacobian includes the waist's own columns, the
correction is spatially stale the instant it's applied if the shared base
moved in the meantime.

This module solves ALL of {waist yaw/roll/pitch, left arm x7,
right arm x7} = 17 DoF as ONE coupled task (12-row stacked: 3 pos + 3 ori
per hand) via ``solve()``: a bounded multi-iteration solve (backtracking
line search + adaptive damping bump on rejection) on a caller-provided
SCRATCH MjData (never the live sim) -- purely kinematic (mj_forward only).
This is what the A/B/D causal experiment used to establish that a coupled
solve (D) out-performs both the uncoupled baseline (A) and a locked-waist
diagnostic (B), and it is what the live grasp_expert.py controller calls
once per state entry (and again on each bounded "resolve", see below) to
get a joint target, tracked into the env via a separate smooth joint-space
trajectory (grasp_expert.py's ``_MinJerkJointTrajectory``).

The waist columns are weighted DOWN in the DLS solve (``joint_weight``,
default 6x for the waist) rather than left uniform: because the waist's
Jacobian columns are shared across BOTH 6-row hand blocks in the stacked
12-row task, a plain (unweighted) pseudo-inverse found it numerically
"cheaper" to lean on the waist for error reduction than either 7-DOF arm,
driving waist_pitch (real range only +-0.52 rad) straight into its hard
limit for marginal benefit and getting stuck there -- a local minimum an
arms-mostly solve does not hit for the same target. This is a SOFT
preference, not a hard lock: the weighted solve still uses the waist for
the genuine coupling correction the causal experiment validated, just
prefers the arms' much larger range first.

A one-shot kinematic ``solve()`` is only as good as the assumption that
the physical arm actually REACHES its joint target exactly. It does not:
MuJoCo's compliant kp=120 position actuators have a real steady-state
droop under the arm's own gravity load (measured up to ~0.04m/7 degrees
at this controller's typical reach). Naively re-running ``solve()`` again
from the drooped pose does NOT fix this -- ``solve()`` only ever answers
"what joint config satisfies this Cartesian target kinematically", which
a repeat solve answers with essentially the SAME joint target every time,
never one that deliberately overshoots to compensate for droop. The fix
grasp_expert.py's resolve logic uses instead is a JOINT-SPACE overshoot:
once a trajectory settles short of the target, it measures the actual
joint-space shortfall (target_q - actual_q) and commands a NEW trajectory
to (target_q + shortfall) -- if droop is roughly constant for a small
joint-space shift near the same posture (a reasonable local assumption),
the actuator's own droop on this inflated command cancels the shortfall
and the physical arm lands close to the original target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from humanoid_learning.expert import pose_ik


@dataclass
class CoupledIKResult:
    success: bool
    waist_q: np.ndarray
    left_q: np.ndarray
    right_q: np.ndarray
    left_pos_error: float
    left_ori_error: float
    right_pos_error: float
    right_ori_error: float
    iterations: int
    joint_limit_margin: float  # minimum margin across all 17 DoF at the solution
    collision_free: bool
    failure_reason: str | None = None
    error_history: list[float] = field(default_factory=list)


class CoupledBilateralIK:
    """Solves waist(3) + left_arm(7) + right_arm(7) = 17 DoF for a
    bimanual full-pose (position + SO(3) orientation) target, via bounded
    damped-least-squares with backtracking line search, joint-limit
    clipping, and null-space rest-posture regularization. Operates on a
    caller-provided scratch MjData (never the live env's data)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        left_palm_site: int,
        right_palm_site: int,
        waist_dof_adr: np.ndarray,
        left_arm_dof_adr: np.ndarray,
        right_arm_dof_adr: np.ndarray,
        waist_qpos_adr: np.ndarray,
        left_arm_qpos_adr: np.ndarray,
        right_arm_qpos_adr: np.ndarray,
        joint_low: np.ndarray,
        joint_high: np.ndarray,
        waist_weight: float = 6.0,
    ):
        self.model = model
        self.left_site = left_palm_site
        self.right_site = right_palm_site
        # Coupled DOF ordering, fixed for the lifetime of this solver:
        # [waist(3), left_arm(7), right_arm(7)] = 17.
        self.dof_adr = np.concatenate([waist_dof_adr, left_arm_dof_adr, right_arm_dof_adr])
        self.qpos_adr = np.concatenate([waist_qpos_adr, left_arm_qpos_adr, right_arm_qpos_adr])
        self.n = len(self.dof_adr)
        self.waist_slice = slice(0, 3)
        self.left_slice = slice(3, 10)
        self.right_slice = slice(10, 17)
        self.joint_low = joint_low  # length 17, in the SAME coupled order
        self.joint_high = joint_high
        # Weighted-DLS joint weight (Section 9 "waist stand-pose deviation
        # penalty ... can be high but must NOT be a hard lock"): the
        # waist's Jacobian columns are shared across BOTH 6-row hand
        # blocks in the stacked 12-row task, so a plain (unweighted)
        # pseudo-inverse solve found it numerically "cheaper" to lean on
        # the waist for error reduction than either 7-DOF arm -- direct
        # measurement showed this drove waist_pitch (real physical range
        # only +-0.52 rad) straight to its hard limit while still leaving
        # 0.04m/4.5deg residual error, a local minimum an arms-only
        # (waist-fixed) solve does not hit for the SAME target. Weighting
        # the waist columns down in the DLS solve (equivalent to a soft
        # cost on using them) makes the solver prefer the arms' much
        # larger, less-constrained range first, and only actually move
        # the waist when the arms alone cannot reduce the task error
        # further -- this is a SOFT preference (the waist columns are
        # still present and used, e.g. for the genuine coupling case the
        # causal A/B/D experiment identified), never a hard lock (which
        # would just be locked_waist condition B again).
        self.joint_weight = np.concatenate([np.full(3, waist_weight), np.ones(14)])

    @staticmethod
    def group_vector(waist: float, shoulder_elbow: float, wrist: float) -> np.ndarray:
        """Builds a 17-dim per-DOF vector (for joint_weight or rest_gain)
        from 3 group values in the solver's fixed [waist(3), left_arm(7),
        right_arm(7)] ordering, with each arm's 7 DoF split into
        shoulder+elbow (the first 4: pitch/roll/yaw/elbow, task_config.py
        LEFT/RIGHT_ARM_JOINTS order) and wrist (the last 3: roll/pitch/
        yaw) -- mirrored identically for both arms, matching this
        project's existing left/right-symmetric convention."""
        arm_group = np.array([shoulder_elbow] * 4 + [wrist] * 3)
        return np.concatenate([np.full(3, waist), arm_group, arm_group])

    def _get_q(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_adr].copy()

    def _set_q(self, data: mujoco.MjData, q: np.ndarray) -> None:
        data.qpos[self.qpos_adr] = q
        mujoco.mj_forward(self.model, data)

    def _task_error_and_jacobian(
        self, data: mujoco.MjData, left_target_pos, left_target_R, right_target_pos, right_target_R
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        """Builds the stacked 12-row task error and 12x17 Jacobian. Each
        palm's Jacobian columns for the OTHER arm are exactly zero because
        mj_jacSite computes true kinematic-tree dependence per DOF --
        selecting [waist, left_arm, right_arm] columns for the LEFT site
        naturally yields zero in the right_arm columns (a real kinematic
        fact, not a manually-imposed zero block), and symmetrically for
        the right site. The waist columns are shared and nonzero in BOTH
        blocks, which is exactly the coupling this solver is meant to
        resolve -- a shared-waist delta computed ONCE, consistently, for
        both hands at once, rather than two independent solves that could
        each move the waist their own way and overwrite each other."""
        left_pos, left_R = data.site_xpos[self.left_site].copy(), data.site_xmat[self.left_site].reshape(3, 3).copy()
        right_pos, right_R = data.site_xpos[self.right_site].copy(), data.site_xmat[self.right_site].reshape(3, 3).copy()

        left_pos_err = left_target_pos - left_pos
        left_ori_err = pose_ik.orientation_error(left_R, left_target_R)
        right_pos_err = right_target_pos - right_pos
        right_ori_err = pose_ik.orientation_error(right_R, right_target_R)

        jacp_l = np.zeros((3, self.model.nv))
        jacr_l = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp_l, jacr_l, self.left_site)
        jacp_r = np.zeros((3, self.model.nv))
        jacr_r = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp_r, jacr_r, self.right_site)

        J_left = np.vstack([jacp_l[:, self.dof_adr], jacr_l[:, self.dof_adr]])  # 6x17
        J_right = np.vstack([jacp_r[:, self.dof_adr], jacr_r[:, self.dof_adr]])  # 6x17
        J = np.vstack([J_left, J_right])  # 12x17

        err = np.concatenate([left_pos_err, left_ori_err, right_pos_err, right_ori_err])
        return err, J, float(np.linalg.norm(left_pos_err)), float(np.linalg.norm(left_ori_err)), float(np.linalg.norm(right_pos_err)), float(np.linalg.norm(right_ori_err))

    def solve(
        self,
        data: mujoco.MjData,
        left_target_pos: np.ndarray,
        left_target_R: np.ndarray,
        right_target_pos: np.ndarray,
        right_target_R: np.ndarray,
        rest_q: np.ndarray,
        max_iterations: int = 200,
        # Lowered from an initial 0.05: direct sweep against a PRE_GRASP
        # target that plateaued at 0.011m/0.05deg (just outside the
        # 0.01m tolerance) under weighted DLS showed damping=0.05 was
        # itself the blocker -- too much damping for this smaller,
        # already-well-conditioned weighted system, preventing the last
        # few millimeters of correction. damping<=0.02 converged cleanly
        # (0.006-0.009m, 12-34 iterations) on the SAME target.
        damping: float = 0.01,
        gain: float = 0.5,
        max_step: float = 0.3,
        rest_gain: float | np.ndarray = 0.3,
        pos_tol: float = 0.01,
        ori_tol_rad: float = np.radians(5.0),
        limit_margin_min: float = 0.02,
        joint_weight: np.ndarray | None = None,
        ori_task_weight: float = 1.0,
        require_orientation: bool = True,
    ) -> CoupledIKResult:
        """Bounded solve: iterates on the GIVEN scratch `data` (caller's
        responsibility to have copied it from the live sim first -- this
        method mutates `data.qpos`/derived fields but the caller decides
        whether/when to copy that back into the live env). Backtracking
        line search: if a step's error is worse than before it, halve the
        step and retry (up to a few times) rather than taking it anyway.

        ``joint_weight``/``rest_gain`` may be a scalar or a length-17
        array (Natural Arm Reach + Wrist Alignment session, Section 5 --
        hierarchical per-state joint roles): a caller can make shoulder/
        elbow "cheap" (low weight) and wrist "expensive" (high weight) for
        a gross-reach state, then flip that for a wrist-alignment state,
        without changing this solver's core iteration at all. Defaults to
        this instance's constructor-time ``joint_weight`` (waist-only
        penalty) when not given, matching prior behavior exactly.

        ``ori_task_weight``/``require_orientation`` (same session): a
        gross-reach state (NATURAL_ARM_LIFT/FOREARM_APPROACH) should not
        be FORCED to fight for the exact final grasp orientation this
        early -- that is what per-DOF joint_weight alone cannot express,
        since the primary task still targets the full 6D pose regardless
        of how expensive wrist is. Setting ``ori_task_weight`` small (a
        soft de-prioritization of the orientation ROWS of the task, not
        the DOF columns) and ``require_orientation=False`` (the
        convergence/success check ignores orientation entirely) lets
        orientation drift passively toward correct as a low-priority
        side-effect of the position solve, without spending wrist effort
        chasing it -- WRIST_ALIGN then re-solves the SAME target with
        orientation as the primary task."""
        weight = self.joint_weight if joint_weight is None else joint_weight
        task_scale = np.array([1, 1, 1, ori_task_weight, ori_task_weight, ori_task_weight] * 2)
        q = self._get_q(data)
        self._set_q(data, q)
        err, J, lp, lo, rp, ro = self._task_error_and_jacobian(
            data, left_target_pos, left_target_R, right_target_pos, right_target_R
        )

        def _converged(lp, lo, rp, ro) -> bool:
            return lp < pos_tol and rp < pos_tol and (not require_orientation or (lo < ori_tol_rad and ro < ori_tol_rad))

        err_eff = err * task_scale
        best_err_norm = float(np.linalg.norm(err_eff))
        error_history = [best_err_norm]
        it = 0
        for it in range(1, max_iterations + 1):
            if _converged(lp, lo, rp, ro):
                break

            # Weighted damped pseudo-inverse: J W^-1 J^T instead of J J^T --
            # equivalent to a soft per-DOF cost on using that joint (see
            # __init__'s joint_weight comment). Reduces exactly to the
            # ordinary DLS solve when joint_weight is all-ones. task_scale
            # additionally de-weights the orientation ROWS (not columns)
            # when ori_task_weight<1 -- see this method's docstring.
            J_eff = task_scale[:, None] * J
            Winv = 1.0 / weight
            JWJt = (J_eff * Winv) @ J_eff.T
            damped_inv = np.linalg.inv(JWJt + (damping**2) * np.eye(JWJt.shape[0]))
            J_pinv = (Winv[:, None] * J_eff.T) @ damped_inv  # 17x12
            dq_task = J_pinv @ err_eff

            # Null-space rest-posture regularization (elbow-down/rest,
            # waist-neutral) -- 17 DoF solving a 12-row task leaves a
            # 5-dim null space free to use for this without fighting the
            # primary objective.
            null_proj = np.eye(self.n) - J_pinv @ J_eff
            dq_rest = rest_gain * (rest_q - q)
            dq = gain * dq_task + null_proj @ dq_rest

            step_norm = float(np.linalg.norm(dq))
            if step_norm > max_step:
                dq = dq * (max_step / step_norm)

            # Backtracking line search: reject a step that makes the
            # stacked error WORSE, halving it up to 4 times before giving
            # up on this iteration (Section 6 "error 증가 시 step 축소 또는 거부").
            accepted = False
            trial_dq = dq.copy()
            for _ in range(5):
                q_trial = np.clip(q + trial_dq, self.joint_low, self.joint_high)
                self._set_q(data, q_trial)
                err_trial, J_trial, lp_t, lo_t, rp_t, ro_t = self._task_error_and_jacobian(
                    data, left_target_pos, left_target_R, right_target_pos, right_target_R
                )
                trial_norm = float(np.linalg.norm(err_trial * task_scale))
                if trial_norm <= best_err_norm * 1.001:  # small slack for numerical noise
                    q, err, J = q_trial, err_trial, J_trial
                    lp, lo, rp, ro = lp_t, lo_t, rp_t, ro_t
                    err_eff = err * task_scale
                    best_err_norm = trial_norm
                    accepted = True
                    break
                trial_dq = trial_dq * 0.5
            error_history.append(best_err_norm)
            if not accepted:
                # Damping bump: a step small enough to always be accepted
                # (near-zero) would trivially "succeed" at rejecting
                # forever without progress -- instead increase damping
                # once and keep iterating, matching a classic
                # Levenberg-Marquardt-style adaptive-damping fallback.
                damping = min(damping * 1.5, 1.0)

        margins = np.minimum(q - self.joint_low, self.joint_high - q)
        min_margin = float(margins.min())
        converged = _converged(lp, lo, rp, ro)
        failure_reason = None
        if not converged:
            failure_reason = "max_iterations_without_convergence"
        elif min_margin < limit_margin_min:
            failure_reason = "joint_limit_margin_too_small"

        return CoupledIKResult(
            success=converged and min_margin >= limit_margin_min,
            waist_q=q[self.waist_slice].copy(),
            left_q=q[self.left_slice].copy(),
            right_q=q[self.right_slice].copy(),
            left_pos_error=lp,
            left_ori_error=lo,
            right_pos_error=rp,
            right_ori_error=ro,
            iterations=it,
            joint_limit_margin=min_margin,
            collision_free=True,  # caller checks data.ncon separately (Section 12)
            failure_reason=failure_reason,
            error_history=error_history,
        )

