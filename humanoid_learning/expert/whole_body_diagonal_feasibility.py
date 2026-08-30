"""SIZE_12 Full-Body Diagonal Reach session: Stage W (full-body: floating
pelvis + legs + waist + arms) static feasibility, built on top of Stage
U's already-fixed diagonal_feasibility.py (corner_direction_local_at_yaw/
hand_corner_target -- see that module for the yaw=0-matches-baseline fix
this session made).

This is a SEPARATE solver from coupled_ik.CoupledBilateralIK (which only
ever handles the FIXED-base 17-DoF waist+arms case): the floating pelvis
is a MuJoCo free joint (7 qpos: xyz + wxyz quat; 6 dof: xyz + so3 tangent
angular velocity), so a naive "qpos[i] += dq[i]" per-DoF update (what
CoupledBilateralIK does for its all-hinge-joint DoF) is wrong for the
quaternion component -- this solver uses mujoco.mj_integratePos, which
handles the free joint's tangent-space integration correctly (and is a
no-op difference for the hinge joints, which stay purely additive).

Scope note (explicit, per this session's Section 4 priority list): only
a REDUCED posture-family set is implemented here (not the full 20 named
families in the spec) -- see build_whole_body_posture_seeds's docstring
for exactly which ones and why, given a 35-DoF solve is materially more
expensive per attempt than Stage U's 17-DoF one.

KNOWN UNRESOLVED ISSUE (discovered late in this session, NOT fixed --
see PROJECT_CONTEXT.md's session report): WholeBodyDiagonalIK.solve()'s
convergence outcome for at least some candidates is NOT reproducible
across otherwise-identical calls. Direct measurement: calling
evaluate_whole_body_pose() for (candidate=C2, seed="pelvis_yaw_right",
corner_yaw_deg=45.0) on a FRESH scratch MjData, as the very first solve
attempted on it, deterministically fails to converge (250/250 iterations,
left_hand_pos_error~0.17-0.43m depending on which "cold" variant is
tried); the SAME call, reached after a handful of OTHER (candidate,
seed) combinations were solved first on the SAME scratch object (fully
overwriting qpos/qvel via ``scratch.qpos[:] = base_qpos`` before each
one, so this is not simply forgetting to reset state), converges
cleanly in ~22 iterations to <1mm hand-position error. This was ruled
out as: joint_weight/foot_weight conditioning (tried foot_weight=8.0
instead of 30.0, no change), stale derived MjData state (tried an
explicit mj_forward warm-up call on the virgin scratch before ever
touching qpos, no change; tried mujoco.mj_resetData, no change), and a
qpos-copy/aliasing bug (verified obj_pos/obj_quat/seed values are
byte-identical between the two call contexts). The mechanism remains
UNIDENTIFIED. Practical consequence: any single evaluate_whole_body_pose
result, and by extension any run_stage_w_search sweep, may report worse
convergence than the same candidate can actually achieve, and a "good"
result found via a sweep has NOT been verified to reproduce from a cold
start -- so no specific candidate's numbers here should be treated as a
settled, reliable measurement without independently re-verifying it in
isolation across multiple fresh processes. Root-causing this is this
session's single most important unresolved next step, ahead of any
further posture-family or search-range expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from humanoid_learning.envs import hand_synergy
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc
from humanoid_learning.expert import pose_ik
from humanoid_learning.expert.diagonal_feasibility import (
    CandidateAssignment,
    HandCornerTarget,
    _classify_point_face,
    _scan_unwanted_collisions,
    hand_corner_target,
    measure_per_finger_local_offsets,
)
from humanoid_learning.expert.grasp_expert import GraspExpertConfig

# Coupled DoF ordering for this solver, fixed for its lifetime:
# [pelvis(6, free-joint tangent space), left_leg(6), right_leg(6),
#  waist(3), left_arm(7), right_arm(7)] = 35.
N_PELVIS = 6
N_LEG = 6
N_WAIST = 3
N_ARM = 7
N_TOTAL = N_PELVIS + 2 * N_LEG + N_WAIST + 2 * N_ARM  # 35


@dataclass
class WholeBodyIKResult:
    success: bool
    pelvis_qpos: np.ndarray  # 7 (xyz + wxyz quat)
    joints_qpos: np.ndarray  # 29 (legs12, waist3, arms14), same order as joints_qpos_adr
    left_hand_pos_error: float
    left_hand_ori_error: float
    right_hand_pos_error: float
    right_hand_ori_error: float
    left_foot_pos_error: float
    left_foot_ori_error: float
    right_foot_pos_error: float
    right_foot_ori_error: float
    joint_limit_margin: float  # legs/waist/arms only -- pelvis has no joint limit
    iterations: int
    failure_reason: str | None = None
    error_history: list[float] = field(default_factory=list)


class WholeBodyDiagonalIK:
    """Bounded weighted-DLS solve for [pelvis(6) + legs(12) + waist(3) +
    arms(14)] = 35 DoF against a stacked 24-row task: both feet (6 rows
    each, held near their CURRENT pose -- Section 7's foot-support
    invariant) and both hands (6 rows each, the diagonal corner target).
    Operates on a caller-provided SCRATCH MjData, exactly like
    CoupledBilateralIK -- never touches the live env."""

    def __init__(
        self, model: mujoco.MjModel,
        left_hand_site: int, right_hand_site: int, left_foot_site: int, right_foot_site: int,
        pelvis_dof_adr: np.ndarray, joints_dof_adr: np.ndarray, joints_qpos_adr: np.ndarray,
        joint_low: np.ndarray, joint_high: np.ndarray,
        pelvis_weight: float = 4.0, foot_weight: float = 30.0,
    ):
        self.model = model
        self.left_hand_site = left_hand_site
        self.right_hand_site = right_hand_site
        self.left_foot_site = left_foot_site
        self.right_foot_site = right_foot_site
        self.pelvis_dof_adr = pelvis_dof_adr  # length 6
        self.joints_dof_adr = joints_dof_adr  # length 29
        self.joints_qpos_adr = joints_qpos_adr  # length 29, 1:1 with joints_dof_adr (all hinges)
        self.dof_adr = np.concatenate([pelvis_dof_adr, joints_dof_adr])  # 35, columns into the full nv Jacobian
        self.n = len(self.dof_adr)
        self.joint_low = joint_low  # length 29 (legs+waist+arms only)
        self.joint_high = joint_high
        # Pelvis columns weighted down like coupled_ik's waist (soft
        # preference for the arms/legs to solve the task before pelvis
        # translates/rotates) -- feet get a MUCH higher task-row weight
        # (not a DoF weight) below, in solve(), to act as a near-hard
        # support constraint.
        self.joint_weight = np.concatenate([np.full(N_PELVIS, pelvis_weight), np.ones(29)])
        self.foot_weight = foot_weight

    def _site_jac(self, data: mujoco.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, site_id)
        return jacp[:, self.dof_adr], jacr[:, self.dof_adr]

    def _task_error_and_jacobian(
        self, data: mujoco.MjData,
        left_hand_pos, left_hand_R, right_hand_pos, right_hand_R,
        left_foot_pos, left_foot_R, right_foot_pos, right_foot_R,
    ):
        rows_err = []
        rows_J = []
        errs = {}
        for name, site, tp, tR in (
            ("left_hand", self.left_hand_site, left_hand_pos, left_hand_R),
            ("right_hand", self.right_hand_site, right_hand_pos, right_hand_R),
            ("left_foot", self.left_foot_site, left_foot_pos, left_foot_R),
            ("right_foot", self.right_foot_site, right_foot_pos, right_foot_R),
        ):
            cur_pos = data.site_xpos[site].copy()
            cur_R = data.site_xmat[site].reshape(3, 3).copy()
            pos_err = tp - cur_pos
            ori_err = pose_ik.orientation_error(cur_R, tR)
            jacp, jacr = self._site_jac(data, site)
            rows_err.append(np.concatenate([pos_err, ori_err]))
            rows_J.append(np.vstack([jacp, jacr]))
            errs[name] = (float(np.linalg.norm(pos_err)), float(np.linalg.norm(ori_err)))
        return np.concatenate(rows_err), np.vstack(rows_J), errs

    def _get_full_qpos_joints(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.joints_qpos_adr].copy()

    def solve(
        self, data: mujoco.MjData,
        left_hand_pos, left_hand_R, right_hand_pos, right_hand_R,
        left_foot_pos, left_foot_R, right_foot_pos, right_foot_R,
        rest_q_joints: np.ndarray,  # length 29, null-space bias for legs/waist/arms only
        max_iterations: int = 250, damping: float = 0.02, gain: float = 0.4, max_step: float = 0.2,
        rest_gain: float | np.ndarray = 0.15,
        pos_tol: float = 0.015, ori_tol_rad: float = np.radians(8.0),
        foot_pos_tol: float = 0.004, foot_ori_tol_rad: float = np.radians(1.5),
        limit_margin_min: float = 0.02, ori_task_weight: float = 0.6,
    ) -> WholeBodyIKResult:
        task_scale = np.array(
            [1, 1, 1, ori_task_weight, ori_task_weight, ori_task_weight] * 2  # hands
            + [self.foot_weight] * 3 + [self.foot_weight] * 3  # left foot pos+ori
            + [self.foot_weight] * 3 + [self.foot_weight] * 3  # right foot pos+ori
        )
        weight = self.joint_weight
        rest_gain_vec = rest_gain if isinstance(rest_gain, np.ndarray) else np.full(self.n, rest_gain)

        def _converged(errs):
            lh, lho = errs["left_hand"]
            rh, rho = errs["right_hand"]
            lf, lfo = errs["left_foot"]
            rf, rfo = errs["right_foot"]
            return (
                lh < pos_tol and rh < pos_tol and lho < ori_tol_rad and rho < ori_tol_rad
                and lf < foot_pos_tol and rf < foot_pos_tol and lfo < foot_ori_tol_rad and rfo < foot_ori_tol_rad
            )

        err, J, errs = self._task_error_and_jacobian(
            data, left_hand_pos, left_hand_R, right_hand_pos, right_hand_R,
            left_foot_pos, left_foot_R, right_foot_pos, right_foot_R,
        )
        err_eff = err * task_scale
        best_err_norm = float(np.linalg.norm(err_eff))
        error_history = [best_err_norm]
        it = 0
        for it in range(1, max_iterations + 1):
            if _converged(errs):
                break
            J_eff = task_scale[:, None] * J
            Winv = 1.0 / weight
            JWJt = (J_eff * Winv) @ J_eff.T
            damped_inv = np.linalg.inv(JWJt + (damping**2) * np.eye(JWJt.shape[0]))
            J_pinv = (Winv[:, None] * J_eff.T) @ damped_inv
            dq_task = J_pinv @ err_eff

            null_proj = np.eye(self.n) - J_pinv @ J_eff
            rest_q_full = np.concatenate([np.zeros(N_PELVIS), rest_q_joints])  # pelvis has no target qpos-space rest
            q_joints_now = self._get_full_qpos_joints(data)
            q_full_now = np.concatenate([np.zeros(N_PELVIS), q_joints_now])  # pelvis "q" is not meaningful in this space
            dq_rest = np.concatenate([np.zeros(N_PELVIS), rest_gain_vec[N_PELVIS:] * (rest_q_joints - q_joints_now)])
            dq = gain * dq_task + null_proj @ dq_rest

            step_norm = float(np.linalg.norm(dq))
            if step_norm > max_step:
                dq = dq * (max_step / step_norm)

            accepted = False
            trial_dq = dq.copy()
            qpos_backup = data.qpos.copy()
            for _ in range(5):
                # Clip the JOINT (non-pelvis) part to stay within limits
                # before integrating -- pelvis has no limit to clip.
                joints_now = self._get_full_qpos_joints(data)
                joints_trial = np.clip(joints_now + trial_dq[N_PELVIS:], self.joint_low, self.joint_high)
                clipped_dq = np.concatenate([trial_dq[:N_PELVIS], joints_trial - joints_now])
                qvel_full = np.zeros(self.model.nv)
                qvel_full[self.dof_adr] = clipped_dq
                data.qpos[:] = qpos_backup
                mujoco.mj_integratePos(self.model, data.qpos, qvel_full, 1.0)
                mujoco.mj_kinematics(self.model, data)
                err_trial, J_trial, errs_trial = self._task_error_and_jacobian(
                    data, left_hand_pos, left_hand_R, right_hand_pos, right_hand_R,
                    left_foot_pos, left_foot_R, right_foot_pos, right_foot_R,
                )
                trial_norm = float(np.linalg.norm(err_trial * task_scale))
                if trial_norm <= best_err_norm * 1.001:
                    err, J, errs = err_trial, J_trial, errs_trial
                    err_eff = err * task_scale
                    best_err_norm = trial_norm
                    accepted = True
                    break
                trial_dq = trial_dq * 0.5
            error_history.append(best_err_norm)
            if not accepted:
                data.qpos[:] = qpos_backup
                mujoco.mj_kinematics(self.model, data)
                damping = min(damping * 1.5, 1.0)

        joints_final = self._get_full_qpos_joints(data)
        margins = np.minimum(joints_final - self.joint_low, self.joint_high - joints_final)
        min_margin = float(margins.min())
        converged = _converged(errs)
        failure_reason = None
        if not converged:
            failure_reason = "max_iterations_without_convergence"
        elif min_margin < limit_margin_min:
            failure_reason = "joint_limit_margin_too_small"

        return WholeBodyIKResult(
            success=converged and min_margin >= limit_margin_min,
            pelvis_qpos=data.qpos[0:7].copy(),
            joints_qpos=joints_final,
            left_hand_pos_error=errs["left_hand"][0], left_hand_ori_error=errs["left_hand"][1],
            right_hand_pos_error=errs["right_hand"][0], right_hand_ori_error=errs["right_hand"][1],
            left_foot_pos_error=errs["left_foot"][0], left_foot_ori_error=errs["left_foot"][1],
            right_foot_pos_error=errs["right_foot"][0], right_foot_ori_error=errs["right_foot"][1],
            joint_limit_margin=min_margin, iterations=it, failure_reason=failure_reason,
            error_history=error_history,
        )


def support_polygon_margin(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, np.ndarray]:
    """Section 7/11: approximate support-polygon check using the ACTUAL 4
    foot-contact sphere geoms per foot (measured directly from the MJCF,
    not assumed) transformed to world XY, forming a rectangle per foot;
    the combined support region is their axis-aligned bounding box (a
    reasonable approximation of the true convex hull for two parallel
    rectangular feet, not an exact hull computation). Returns (margin,
    com_xy) -- margin is the signed distance from the whole-body COM's
    XY projection to the NEAREST edge of that bounding box (negative if
    outside -- a real balance failure)."""
    corners = []
    for body_name in (wbc.LEFT_FOOT_CONTACT_BODY, wbc.RIGHT_FOOT_CONTACT_BODY):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        body_pos = data.xpos[bid]
        body_R = data.xmat[bid].reshape(3, 3)
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == bid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                corners.append(body_pos + body_R @ model.geom_pos[g])
    corners = np.array(corners)  # (8, 3)
    com_xy = data.subtree_com[0][:2].copy()
    x_lo, x_hi = corners[:, 0].min(), corners[:, 0].max()
    y_lo, y_hi = corners[:, 1].min(), corners[:, 1].max()
    margin_x = min(com_xy[0] - x_lo, x_hi - com_xy[0])
    margin_y = min(com_xy[1] - y_lo, y_hi - com_xy[1])
    return float(min(margin_x, margin_y)), com_xy


@dataclass
class WholeBodyPostureSeed:
    name: str
    joints_q: np.ndarray  # length 29 (legs12, waist3, arms14) -- rest_q AND starting q for this seed
    pelvis_yaw_delta: float = 0.0  # radians, applied to the STARTING pelvis quat only (no null-space pull exists for pelvis -- see WholeBodyDiagonalIK.solve)
    pelvis_xy_delta: tuple[float, float] = (0.0, 0.0)  # meters, applied to the STARTING pelvis xy only (Section 8 item 6: "small pelvis translation toward object")


# Index offsets within the 29-length joints vector (verified against
# wbc.LEFT_LEG_JOINTS/RIGHT_LEG_JOINTS/WAIST_JOINTS and tc.LEFT_ARM_
# JOINTS/RIGHT_ARM_JOINTS ordering).
_L_LEG, _R_LEG = slice(0, 6), slice(6, 12)
_WAIST = slice(12, 15)  # yaw, roll, pitch
_L_ARM, _R_ARM = slice(15, 22), slice(22, 29)  # shoulder_pitch,roll,yaw, elbow, wrist_roll,pitch,yaw
_HIP_PITCH, _KNEE, _ANKLE_PITCH = 0, 3, 4  # offsets within a 6-length leg block
_WAIST_YAW, _WAIST_PITCH = 0, 2  # offsets within the 3-length waist block
_SHOULDER_PITCH, _SHOULDER_ROLL = 0, 1  # offsets within a 7-length arm block


def build_whole_body_posture_seeds(stand_joints_q: np.ndarray) -> list[WholeBodyPostureSeed]:
    """REDUCED posture-family set (Section 4/8 priority: "Stage U를
    반복 실행하는 데 세션을 모두 쓰지 마라... Stage W를 실제로 수행"). The
    full spec lists 20 named families; given a 35-DoF solve is
    materially more expensive per attempt than Stage U's 17-DoF one,
    this implements 9 representative families spanning the SAME
    physical roles (lower-body height/balance, pelvis yaw, torso lean,
    shoulder-forward, chirality-specific) rather than all 20 -- this
    reduction is disclosed here and in the session report, not silently
    assumed equivalent. Knee-bend deltas are a starting-point/null-space
    BIAS only, not a hand-solved squat IK -- the solver's own hard foot-
    position/orientation task rows correct the legs to whatever actually
    keeps both feet planted, regardless of how precise this seed is."""
    seeds = []

    def q():
        return stand_joints_q.copy()

    seeds.append(WholeBodyPostureSeed("stand_baseline", q()))

    v = q()
    v[_L_LEG][_KNEE] += 0.2; v[_R_LEG][_KNEE] += 0.2
    v[_L_LEG][_HIP_PITCH] -= 0.1; v[_R_LEG][_HIP_PITCH] -= 0.1
    v[_L_LEG][_ANKLE_PITCH] -= 0.1; v[_R_LEG][_ANKLE_PITCH] -= 0.1
    seeds.append(WholeBodyPostureSeed("symmetric_knee_bend_small", v))

    v = q()
    v[_L_LEG][_KNEE] += 0.4; v[_R_LEG][_KNEE] += 0.4
    v[_L_LEG][_HIP_PITCH] -= 0.2; v[_R_LEG][_HIP_PITCH] -= 0.2
    v[_L_LEG][_ANKLE_PITCH] -= 0.2; v[_R_LEG][_ANKLE_PITCH] -= 0.2
    seeds.append(WholeBodyPostureSeed("symmetric_knee_bend_medium", v))

    v = q()
    v[_L_LEG][_HIP_PITCH] += 0.3; v[_R_LEG][_HIP_PITCH] += 0.3
    v[_WAIST][_WAIST_PITCH] += 0.1
    seeds.append(WholeBodyPostureSeed("hip_hinge_forward", v))

    v = q()
    v[_WAIST][_WAIST_PITCH] += 0.2
    seeds.append(WholeBodyPostureSeed("torso_lean_forward", v))

    seeds.append(WholeBodyPostureSeed("pelvis_yaw_left", q(), pelvis_yaw_delta=0.2))
    seeds.append(WholeBodyPostureSeed("pelvis_yaw_right", q(), pelvis_yaw_delta=-0.2))

    v = q()
    v[_L_ARM][_SHOULDER_PITCH] += 0.6; v[_R_ARM][_SHOULDER_PITCH] += 0.6
    seeds.append(WholeBodyPostureSeed("shoulder_forward_bilateral", v))

    # C1's hard hand is LEFT (reaches the +X+Y corner, away from this
    # pipeline's natural +X-frontal approach -- see diagonal_
    # feasibility.build_candidate_specific_seeds' identical reasoning for
    # the fixed-base case): bias left shoulder-forward + a small pelvis
    # yaw + hip hinge toward that corner.
    v = q()
    v[_L_ARM][_SHOULDER_PITCH] += 0.6; v[_L_ARM][_SHOULDER_ROLL] += 0.3
    v[_L_LEG][_HIP_PITCH] += 0.15; v[_R_LEG][_HIP_PITCH] += 0.15
    v[_WAIST][_WAIST_YAW] += 0.15
    seeds.append(WholeBodyPostureSeed("C1_specific_left_hard_hand", v, pelvis_yaw_delta=0.15))

    v = q()
    v[_R_ARM][_SHOULDER_PITCH] += 0.6; v[_R_ARM][_SHOULDER_ROLL] -= 0.3
    v[_L_LEG][_HIP_PITCH] += 0.15; v[_R_LEG][_HIP_PITCH] += 0.15
    v[_WAIST][_WAIST_YAW] -= 0.15
    seeds.append(WholeBodyPostureSeed("C2_specific_right_hard_hand", v, pelvis_yaw_delta=-0.15))

    # Section 8 item 6 / Section 10 (COM/support margin): explicit
    # pelvis TRANSLATION toward the object -- none of the joint-only
    # families above give the solver any incentive to actually shift the
    # pelvis (it has no null-space rest pull, see WholeBodyDiagonalIK.
    # solve), so if the reach target pulls COM outside the support
    # polygon, only a genuinely different STARTING pelvis position can
    # plausibly find a different local minimum that keeps COM balanced
    # while still satisfying the hard foot constraints.
    v = q(); v[_L_LEG][_KNEE] += 0.15; v[_R_LEG][_KNEE] += 0.15
    seeds.append(WholeBodyPostureSeed("pelvis_translate_forward", v, pelvis_xy_delta=(0.05, 0.0)))
    v = q(); v[_L_LEG][_KNEE] += 0.15; v[_R_LEG][_KNEE] += 0.15
    seeds.append(WholeBodyPostureSeed("pelvis_translate_forward_left", v, pelvis_xy_delta=(0.05, 0.04)))
    v = q(); v[_L_LEG][_KNEE] += 0.15; v[_R_LEG][_KNEE] += 0.15
    seeds.append(WholeBodyPostureSeed("pelvis_translate_forward_right", v, pelvis_xy_delta=(0.05, -0.04)))

    return seeds


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


@dataclass
class WholeBodySetup:
    """Everything needed to run the Stage W search on a given
    WholeBodyEnv instance -- built once per env (joint/site/dof
    addressing is expensive to look up, the search itself is not)."""
    model: mujoco.MjModel
    solver: WholeBodyDiagonalIK
    joints_qpos_adr: np.ndarray
    stand_joints_q: np.ndarray
    stand_pelvis_qpos: np.ndarray
    left_foot_target_pos: np.ndarray
    left_foot_target_R: np.ndarray
    right_foot_target_pos: np.ndarray
    right_foot_target_R: np.ndarray
    left_hand_site: int
    right_hand_site: int
    obj_body_id: int
    obj_qpos_adr: int
    left_finger_qpos_adr: np.ndarray
    right_finger_qpos_adr: np.ndarray
    left_finger_offsets: dict
    right_finger_offsets: dict
    left_finger_open: np.ndarray
    right_finger_open: np.ndarray


def build_whole_body_setup(env, object_half_size: float) -> WholeBodySetup:
    """Resizes/repositions env's object to ``object_half_size`` (SIZE_12
    for this session) and builds a WholeBodySetup for the Stage W search.
    ``env`` must be a WholeBodyEnv(include_object=True) already reset."""
    model, data = env.model, env.data

    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tc.OBJECT_BODY)
    obj_geom_id = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == obj_body_id)
    model.geom_size[obj_geom_id] = [object_half_size] * 3
    obj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tc.OBJECT_JOINT)
    obj_qpos_adr = model.jnt_qposadr[obj_jid]
    obj_pos = np.array([
        env.config.table_pos[0], env.config.table_pos[1],
        env.config.table_pos[2] + env.config.table_half_size[2] + object_half_size + tc.OBJECT_TABLE_GAP,
    ])
    data.qpos[obj_qpos_adr:obj_qpos_adr + 3] = obj_pos
    data.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    floating_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tc.FLOATING_BASE_JOINT)
    pelvis_dof_adr = np.arange(model.jnt_dofadr[floating_jid], model.jnt_dofadr[floating_jid] + 6)

    def qpos_dof(names):
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        return np.array([model.jnt_qposadr[j] for j in jids]), np.array([model.jnt_dofadr[j] for j in jids])

    left_leg_qpos, left_leg_dof = qpos_dof(wbc.LEFT_LEG_JOINTS)
    right_leg_qpos, right_leg_dof = qpos_dof(wbc.RIGHT_LEG_JOINTS)
    waist_qpos, waist_dof = qpos_dof(wbc.WAIST_JOINTS)
    left_arm_qpos, left_arm_dof = qpos_dof(tc.LEFT_ARM_JOINTS)
    right_arm_qpos, right_arm_dof = qpos_dof(tc.RIGHT_ARM_JOINTS)
    joints_qpos_adr = np.concatenate([left_leg_qpos, right_leg_qpos, waist_qpos, left_arm_qpos, right_arm_qpos])
    joints_dof_adr = np.concatenate([left_leg_dof, right_leg_dof, waist_dof, left_arm_dof, right_arm_dof])

    def jrange(names):
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        return model.jnt_range[jids, 0].copy(), model.jnt_range[jids, 1].copy()

    low_parts, high_parts = [], []
    for names in (wbc.LEFT_LEG_JOINTS, wbc.RIGHT_LEG_JOINTS, wbc.WAIST_JOINTS, tc.LEFT_ARM_JOINTS, tc.RIGHT_ARM_JOINTS):
        lo, hi = jrange(names)
        low_parts.append(lo)
        high_parts.append(hi)
    joint_low = np.concatenate(low_parts)
    joint_high = np.concatenate(high_parts)

    left_hand_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_PALM_SITE)
    right_hand_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.RIGHT_PALM_SITE)
    left_foot_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.LEFT_FOOT_SITE)
    right_foot_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wbc.RIGHT_FOOT_SITE)

    solver = WholeBodyDiagonalIK(
        model, left_hand_site, right_hand_site, left_foot_site, right_foot_site,
        pelvis_dof_adr, joints_dof_adr, joints_qpos_adr, joint_low, joint_high,
    )

    left_finger_qpos_adr = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n, _, _ in wbc.LEFT_HAND_SYNERGY_TARGETS
    ])
    right_finger_qpos_adr = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n, _, _ in wbc.RIGHT_HAND_SYNERGY_TARGETS
    ])

    class _EnvShim:
        pass

    shim = _EnvShim()
    shim.model, shim.data = model, data
    shim._left_finger_qpos_adr = left_finger_qpos_adr
    shim._right_finger_qpos_adr = right_finger_qpos_adr
    gcfg = GraspExpertConfig()
    left_finger_offsets = measure_per_finger_local_offsets(shim, "left", gcfg.thumb1_abduct_pose_left)
    right_finger_offsets = measure_per_finger_local_offsets(shim, "right", gcfg.thumb1_abduct_pose_right)
    left_finger_open = np.array(hand_synergy.left_hand_targets(0.0), dtype=float)
    left_finger_open[1] = gcfg.thumb1_abduct_pose_left
    right_finger_open = np.array(hand_synergy.right_hand_targets(0.0), dtype=float)
    right_finger_open[1] = gcfg.thumb1_abduct_pose_right

    return WholeBodySetup(
        model=model, solver=solver, joints_qpos_adr=joints_qpos_adr,
        stand_joints_q=data.qpos[joints_qpos_adr].copy(), stand_pelvis_qpos=data.qpos[0:7].copy(),
        left_foot_target_pos=data.site_xpos[left_foot_site].copy(), left_foot_target_R=data.site_xmat[left_foot_site].reshape(3, 3).copy(),
        right_foot_target_pos=data.site_xpos[right_foot_site].copy(), right_foot_target_R=data.site_xmat[right_foot_site].reshape(3, 3).copy(),
        left_hand_site=left_hand_site, right_hand_site=right_hand_site,
        obj_body_id=obj_body_id, obj_qpos_adr=obj_qpos_adr,
        left_finger_qpos_adr=left_finger_qpos_adr, right_finger_qpos_adr=right_finger_qpos_adr,
        left_finger_offsets=left_finger_offsets, right_finger_offsets=right_finger_offsets,
        left_finger_open=left_finger_open, right_finger_open=right_finger_open,
    )


@dataclass
class StageWResult:
    candidate: str
    seed_name: str
    corner_yaw_deg: float
    ik: WholeBodyIKResult
    com_support_margin: float
    collision_free: bool
    collision_pairs: list
    face_assignment_satisfied: bool
    actual_faces: tuple  # (left_thumb, left_finger, right_thumb, right_finger)

    @property
    def fully_passed(self) -> bool:
        """Section 11's full Static PASS criterion: kinematic success
        (reach + foot-hold + joint-limit margin, from WholeBodyIKResult.
        success) AND collision-free AND COM support margin >= 0 AND the
        intended fingertip-face assignment."""
        return self.ik.success and self.collision_free and self.com_support_margin >= 0.0 and self.face_assignment_satisfied


def evaluate_whole_body_pose(
    setup: WholeBodySetup, candidate: CandidateAssignment, seed: WholeBodyPostureSeed,
    corner_yaw_deg: float, palm_standoff: float, scratch: mujoco.MjData,
) -> StageWResult:
    """Runs ONE Stage W static solve + collision scan + support-margin
    check + fingertip-face verification -- the full-body analogue of
    diagonal_feasibility.evaluate_static_pose."""
    model = setup.model
    obj_pos = scratch.qpos[setup.obj_qpos_adr:setup.obj_qpos_adr + 3].copy()
    obj_quat = scratch.qpos[setup.obj_qpos_adr + 3:setup.obj_qpos_adr + 7].copy()
    left_t = hand_corner_target(obj_pos, obj_quat, np.linalg.norm(model.geom_size[
        next(g for g in range(model.ngeom) if model.geom_bodyid[g] == setup.obj_body_id)
    ]) / np.sqrt(3), candidate.left, palm_standoff=palm_standoff, side="left", corner_yaw_deg=corner_yaw_deg)
    right_t = hand_corner_target(obj_pos, obj_quat, np.linalg.norm(model.geom_size[
        next(g for g in range(model.ngeom) if model.geom_bodyid[g] == setup.obj_body_id)
    ]) / np.sqrt(3), candidate.right, palm_standoff=palm_standoff, side="right", corner_yaw_deg=corner_yaw_deg)

    if seed.pelvis_yaw_delta != 0.0:
        scratch.qpos[3:7] = _quat_mul(_quat_from_yaw(seed.pelvis_yaw_delta), setup.stand_pelvis_qpos[3:7])
    scratch.qpos[0] += seed.pelvis_xy_delta[0]
    scratch.qpos[1] += seed.pelvis_xy_delta[1]
    scratch.qpos[setup.joints_qpos_adr] = seed.joints_q
    mujoco.mj_kinematics(model, scratch)

    ik = setup.solver.solve(
        scratch, left_t.palm_pos, left_t.palm_R, right_t.palm_pos, right_t.palm_R,
        setup.left_foot_target_pos, setup.left_foot_target_R, setup.right_foot_target_pos, setup.right_foot_target_R,
        rest_q_joints=seed.joints_q,
    )
    mujoco.mj_forward(model, scratch)
    com_margin, _ = support_polygon_margin(model, scratch)

    scratch.qpos[setup.left_finger_qpos_adr] = setup.left_finger_open
    scratch.qpos[setup.right_finger_qpos_adr] = setup.right_finger_open
    mujoco.mj_forward(model, scratch)
    collisions = _scan_unwanted_collisions(model, scratch, setup.obj_body_id)

    left_palm_pos = scratch.site_xpos[setup.left_hand_site].copy()
    left_palm_R = scratch.site_xmat[setup.left_hand_site].reshape(3, 3).copy()
    right_palm_pos = scratch.site_xpos[setup.right_hand_site].copy()
    right_palm_R = scratch.site_xmat[setup.right_hand_site].reshape(3, 3).copy()
    lt = left_palm_pos + left_palm_R @ setup.left_finger_offsets["thumb"]
    lf = left_palm_pos + left_palm_R @ (0.5 * (setup.left_finger_offsets["index"] + setup.left_finger_offsets["middle"]))
    rt = right_palm_pos + right_palm_R @ setup.right_finger_offsets["thumb"]
    rf = right_palm_pos + right_palm_R @ (0.5 * (setup.right_finger_offsets["index"] + setup.right_finger_offsets["middle"]))
    obj_quat_now = scratch.qpos[setup.obj_qpos_adr + 3:setup.obj_qpos_adr + 7]
    actual = (
        _classify_point_face(lt, obj_pos, obj_quat_now), _classify_point_face(lf, obj_pos, obj_quat_now),
        _classify_point_face(rt, obj_pos, obj_quat_now), _classify_point_face(rf, obj_pos, obj_quat_now),
    )
    face_ok = actual == (candidate.left.thumb_face, candidate.left.finger_face, candidate.right.thumb_face, candidate.right.finger_face)

    return StageWResult(
        candidate=candidate.name, seed_name=seed.name, corner_yaw_deg=corner_yaw_deg, ik=ik,
        com_support_margin=com_margin, collision_free=not collisions, collision_pairs=collisions,
        face_assignment_satisfied=face_ok, actual_faces=actual,
    )


def stage_w_score(r: StageWResult) -> tuple:
    """Same priority philosophy as diagonal_feasibility.score_result,
    extended with Stage W's two NEW required criteria (collision, COM
    support margin) ranked ahead of raw reach error."""
    kinematic_penalty = 0 if r.ik.success else 1
    collision_penalty = 0 if r.collision_free else 1
    support_penalty = 0 if r.com_support_margin >= 0.0 else 1
    face_penalty = 0 if r.face_assignment_satisfied else 1
    reach_error = r.ik.left_hand_pos_error + r.ik.right_hand_pos_error
    return (kinematic_penalty, collision_penalty, support_penalty, face_penalty, reach_error, -r.com_support_margin)


def run_stage_w_search(
    setup: WholeBodySetup, candidates, seeds: list[WholeBodyPostureSeed],
    corner_yaws: tuple, palm_standoff: float, base_qpos: np.ndarray,
) -> list[StageWResult]:
    """Runs the full coarse grid (candidates x corner_yaws x seeds) and
    returns every result, unsorted -- caller sorts with stage_w_score."""
    scratch = mujoco.MjData(setup.model)
    results = []
    for candidate in candidates:
        for yaw in corner_yaws:
            for seed in seeds:
                scratch.qpos[:] = base_qpos
                scratch.qvel[:] = 0.0
                results.append(evaluate_whole_body_pose(setup, candidate, seed, yaw, palm_standoff, scratch))
    return results
