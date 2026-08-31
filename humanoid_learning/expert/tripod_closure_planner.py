"""SIZE_12 Collision-Free Tripod Closure Planner session.

Stage 1 (this file's audit functions): a fresh, name-independent
forward-kinematics audit of the Dex3 hand geometry actually used by the
canonical controller -- palm frame, TRUE fingertip sites (as opposed to
the LEGACY body-origin/centroid convention grasp_expert.py's untouched
natural-approach pipeline still deliberately uses -- see
BimanualSidePinchExpert._measure_fingertip_grasp_offset's own docstring
for why that legacy convention must stay exactly as it is), per-joint
tip-position/normal sensitivity, joint ranges, and current
CONTACT_ACQUIRE-end fingertip/object geometry.

Stage 2 (TripodClosurePlanner): a static six-fingertip coordinated
closure optimizer that plans wrist/palm + thumb/index/middle targets
SIMULTANEOUSLY for a fixed object pose, instead of the legacy "one
3-tip centroid -> one palm target" approach THUMB_OPPOSE currently uses.

This module is READ-ONLY with respect to the live env/expert: it never
writes to env.data except on its own private scratch mjData copies, and
is not imported by BimanualSidePinchExpert's control loop. Nothing here
changes canonical --grasp behavior by itself; connecting a validated
plan to the live controller (Stage 4) is a separate, explicit,
default-off step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from humanoid_learning.envs import whole_body_config as wbc

FINGER_JOINT_SUFFIXES = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")


def _side_joint_names(side: str) -> list[str]:
    return [f"{side}_hand_{suf}_joint" for suf in FINGER_JOINT_SUFFIXES]


def true_fingertip_world(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    """The corrected fingertip position (17th-session fix): each
    *_TIP_SITE name maps (FINGERTIP_SITE_BODIES) to the distal link body
    whose ORIGIN sits at that finger's last joint axis (confirmed by
    direct sweep: rotating that joint alone leaves the body origin at
    (0,0,0) local displacement) -- the true fingertip is
    FINGERTIP_SITE_LOCAL_POS further out along that body's own frame,
    not the body origin itself and not a raw MuJoCo <site> element
    (there is none placed at the true tip)."""
    body_name = wbc.FINGERTIP_SITE_BODIES[site_name]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    local = np.array(wbc.FINGERTIP_SITE_LOCAL_POS[site_name], dtype=np.float64)
    R = data.xmat[body_id].reshape(3, 3)
    return data.xpos[body_id] + R @ local


def legacy_centroid_world(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    """The OLD convention BimanualSidePinchExpert._measure_fingertip_
    grasp_offset still deliberately uses for the whole untouched
    natural-approach pipeline: the body ORIGIN (not the corrected true
    tip) of the three distal links, averaged. Kept here only so callers
    can directly compare/diff against Stage 2's true-fingertip targets
    -- never used to plan new contacts."""
    names = (
        ("left_hand_thumb_2_link", "left_hand_index_1_link", "left_hand_middle_1_link")
        if side == "left" else
        ("right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link")
    )
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    return np.mean([data.xpos[i] for i in ids], axis=0)


@dataclass
class JointSweepResult:
    joint_name: str
    tip_site: str
    jnt_range: tuple[float, float]
    tip_travel_m: float  # |tip(hi) - tip(lo)| holding all other joints at baseline
    tip_positions: list = field(default_factory=list)  # (q, world_pos) samples


def sweep_joint_tip_sensitivity(
    model: mujoco.MjModel, scratch: mujoco.MjData, side: str, joint_name: str, tip_site: str,
    baseline_qpos: np.ndarray, finger_qpos_adr: np.ndarray, n_samples: int = 9,
) -> JointSweepResult:
    """Sweeps ONE finger joint across its FULL model.jnt_range (not the
    open/close synergy sub-range) holding every other joint at
    ``baseline_qpos``, measuring the TRUE fingertip site's world
    displacement -- the same independent-sweep method the 16th/26th
    sessions used, generalized to all 7 joints per hand and to the
    corrected tip site instead of body origin."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    lo, hi = model.jnt_range[jid]
    joint_names = _side_joint_names(side)
    slot = joint_names.index(joint_name)

    samples = []
    for frac in np.linspace(0.0, 1.0, n_samples):
        q = baseline_qpos.copy()
        q[slot] = lo + frac * (hi - lo)
        scratch.qpos[finger_qpos_adr] = q
        mujoco.mj_forward(model, scratch)
        pos = true_fingertip_world(model, scratch, tip_site)
        samples.append((float(q[slot]), pos.copy()))

    travel = float(np.linalg.norm(samples[-1][1] - samples[0][1]))
    return JointSweepResult(joint_name=joint_name, tip_site=tip_site, jnt_range=(float(lo), float(hi)),
                             tip_travel_m=travel, tip_positions=samples)


def audit_hand_geometry(env, expert=None, baseline_synergy: float = 0.15) -> dict:
    """Stage 1 master audit. Builds its own scratch MjData (never
    touches env.data) seeded from the env's CURRENT qpos (so if the
    caller has already driven a real rollout to some interesting pose --
    e.g. CONTACT_ACQUIRE end -- the per-joint sweeps and object-relative
    readings below reflect that actual arm configuration, not a cold
    reset). Returns a plain dict (JSON/print friendly), not a dataclass,
    since this is a diagnostic report, not a controller input."""
    model = env.model
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = env.data.qpos
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)

    report: dict = {"sides": {}}

    obj_pos = env.data.qpos[env._object_qpos_adr:env._object_qpos_adr + 3].copy()
    obj_quat = env.data.qpos[env._object_qpos_adr + 3:env._object_qpos_adr + 7].copy()
    obj_R = np.zeros(9)
    mujoco.mju_quat2Mat(obj_R, obj_quat)
    obj_R = obj_R.reshape(3, 3)
    half = env.config.object_half_size
    report["object"] = {"pos": obj_pos.tolist(), "half_size": half}

    for side in ("left", "right"):
        finger_qpos_adr = env._left_finger_qpos_adr if side == "left" else env._right_finger_qpos_adr
        joint_names = _side_joint_names(side)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        ranges = [tuple(float(x) for x in model.jnt_range[j]) for j in jids]

        palm_site_name = wbc.LEFT_PALM_SITE if side == "left" else wbc.RIGHT_PALM_SITE
        palm_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, palm_site_name)
        wrist_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                        f"{side}_wrist_yaw_link")

        tip_sites = {
            "thumb": wbc.LEFT_THUMB_TIP_SITE if side == "left" else wbc.RIGHT_THUMB_TIP_SITE,
            "index": wbc.LEFT_INDEX_TIP_SITE if side == "left" else wbc.RIGHT_INDEX_TIP_SITE,
            "middle": wbc.LEFT_MIDDLE_TIP_SITE if side == "left" else wbc.RIGHT_MIDDLE_TIP_SITE,
        }

        # current (as-of-scratch-qpos) true tip positions, object-local coords
        current_qpos = scratch.qpos[finger_qpos_adr].copy()
        current_tips_world = {name: true_fingertip_world(model, scratch, site) for name, site in tip_sites.items()}
        current_tips_object_local = {
            name: (obj_R.T @ (pos - obj_pos)).tolist() for name, pos in current_tips_world.items()
        }
        legacy_centroid = legacy_centroid_world(model, scratch, side)
        true_centroid = np.mean(list(current_tips_world.values()), axis=0)

        # per-joint sensitivity sweeps (all 7 joints -> whichever tip site they belong to)
        joint_to_tip = {
            f"{side}_hand_thumb_0_joint": "thumb", f"{side}_hand_thumb_1_joint": "thumb",
            f"{side}_hand_thumb_2_joint": "thumb", f"{side}_hand_middle_0_joint": "middle",
            f"{side}_hand_middle_1_joint": "middle", f"{side}_hand_index_0_joint": "index",
            f"{side}_hand_index_1_joint": "index",
        }
        sweeps = {}
        for jn in joint_names:
            tip = joint_to_tip[jn]
            res = sweep_joint_tip_sensitivity(model, scratch, side, jn, tip_sites[tip], current_qpos, finger_qpos_adr)
            sweeps[jn] = {"jnt_range": res.jnt_range, "tip_travel_m": res.tip_travel_m}
        scratch.qpos[finger_qpos_adr] = current_qpos  # restore after sweeps
        mujoco.mj_forward(model, scratch)

        report["sides"][side] = {
            "palm_pos": scratch.site_xpos[palm_site].tolist(),
            "palm_R": scratch.site_xmat[palm_site].reshape(3, 3).tolist(),
            "wrist_pos": scratch.xpos[wrist_body].tolist(),
            "joint_ranges": dict(zip(joint_names, ranges)),
            "current_finger_qpos": current_qpos.tolist(),
            "current_true_tip_world": {k: v.tolist() for k, v in current_tips_world.items()},
            "current_true_tip_object_local": current_tips_object_local,
            "legacy_centroid_world": legacy_centroid.tolist(),
            "true_centroid_world": true_centroid.tolist(),
            "legacy_vs_true_centroid_delta_m": float(np.linalg.norm(true_centroid - legacy_centroid)),
            "joint_tip_sensitivity": sweeps,
        }

    # cross-hand distances at the CURRENT (audited) pose
    lt = report["sides"]["left"]["current_true_tip_world"]
    rt = report["sides"]["right"]["current_true_tip_world"]
    thumb_thumb_dist = float(np.linalg.norm(np.array(lt["thumb"]) - np.array(rt["thumb"])))
    report["thumb_thumb_distance_m"] = thumb_thumb_dist

    return report


# ---------------------------------------------------------------------
# Stage 2: individual fingertip contact-target definition + small-DOF
# finger-only IK (arm/wrist untouched -- see module docstring: index and
# middle already reliably reach the +X face under the existing natural-
# approach pipeline, confirmed by Stage 1's audit above, so Stage 2 only
# needs to verify/re-plan what THUMB_OPPOSE currently gets wrong: the
# thumb's own individual target).
# ---------------------------------------------------------------------

_THUMB_JOINTS = {"thumb_0": 0, "thumb_1": 1, "thumb_2": 2}


def solve_finger_ik(
    model: mujoco.MjModel, scratch: mujoco.MjData, side: str, finger: str, tip_site: str,
    target_world: np.ndarray, finger_qpos_adr: np.ndarray, base_qpos: np.ndarray,
    slots: list[int], max_iters: int = 200, damping: float = 0.02, gain: float = 0.5, max_dq: float = 0.2,
) -> tuple[np.ndarray, float, bool]:
    """Small damped-least-squares IK solving ONLY the given finger's own
    joint slots (indices into the 7-slot per-hand finger qpos vector,
    matching FINGER_JOINT_SUFFIXES order) for a TRUE fingertip world
    target, holding every other finger joint at ``base_qpos``. Position-
    only (a fingertip is a point, not a frame) -- returns (solved 7-slot
    qpos, final position error norm, converged_within_joint_limits)."""
    q = base_qpos.copy()
    body_name = wbc.FINGERTIP_SITE_BODIES[tip_site]
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    local = np.array(wbc.FINGERTIP_SITE_LOCAL_POS[tip_site], dtype=np.float64)
    joint_names = _side_joint_names(side)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[s]) for s in slots]
    dof_adr = [model.jnt_dofadr[j] for j in jids]
    ranges = [model.jnt_range[j] for j in jids]

    err_norm = float("inf")
    for _ in range(max_iters):
        scratch.qpos[finger_qpos_adr] = q
        mujoco.mj_forward(model, scratch)
        R = scratch.xmat[body_id].reshape(3, 3)
        tip_world = scratch.xpos[body_id] + R @ local
        err = target_world - tip_world
        err_norm = float(np.linalg.norm(err))
        if err_norm < 1e-4:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, scratch, jacp, jacr, tip_world, body_id)
        J = jacp[:, dof_adr]  # (3, n_slots)
        JJt = J @ J.T + (damping ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err) * gain
        dq = np.clip(dq, -max_dq, max_dq)
        for k, s in enumerate(slots):
            q[s] = float(np.clip(q[s] + dq[k], ranges[k][0], ranges[k][1]))

    within_limits = all(ranges[k][0] - 1e-6 <= q[s] <= ranges[k][1] + 1e-6 for k, s in enumerate(slots))
    return q, err_norm, within_limits


def solve_hand_closure_ik(
    model: mujoco.MjModel, scratch: mujoco.MjData, side: str,
    thumb_target_world: np.ndarray, index_hold_world: np.ndarray, middle_hold_world: np.ndarray,
    wrist_qpos_adr: np.ndarray, wrist_dof_adr: np.ndarray, wrist_jids: list[int],
    finger_qpos_adr: np.ndarray, base_finger_qpos: np.ndarray, base_wrist_qpos: np.ndarray,
    max_iters: int = 300, damping: float = 0.03, gain: float = 0.4, max_dq: float = 0.15,
) -> dict:
    """Extends solve_finger_ik with 3 additional decision variables (this
    hand's own wrist roll/pitch/yaw -- explicitly permitted as a 'small
    soft deviation' decision variable, NOT the whole arm/shoulder/elbow)
    when thumb-only 3-DOF reach is insufficient (Stage 2's first result:
    thumb_1/thumb_2 pinned at their hard limits, ~45-49mm short in Y).
    Task: thumb tip -> NEW target (3 rows), index/middle tips -> STAY at
    their CURRENTLY-ACHIEVED position (3+3 rows) so wrist motion cannot
    silently break the already-working index/middle contact this
    planner is not trying to re-solve. 9 task rows, 6 decision columns
    (wrist 3 + thumb 3) -- index/middle joints themselves are NOT moved
    (if the wrist alone can hold their tip position while thumb reaches,
    that is a stronger, more honest result than also curling them)."""
    thumb_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_thumb_{i}_joint") for i in range(3)]
    thumb_dof_adr = [model.jnt_dofadr[j] for j in thumb_jids]
    thumb_ranges = [model.jnt_range[j] for j in thumb_jids]
    wrist_ranges = [model.jnt_range[j] for j in wrist_jids]

    thumb_site = f"{side}_thumb_tip"
    index_site = f"{side}_index_tip"
    middle_site = f"{side}_middle_tip"

    q_finger = base_finger_qpos.copy()
    q_wrist = base_wrist_qpos.copy()
    err_norm = float("inf")
    for _ in range(max_iters):
        scratch.qpos[finger_qpos_adr] = q_finger
        scratch.qpos[wrist_qpos_adr] = q_wrist
        mujoco.mj_forward(model, scratch)

        thumb_pos = true_fingertip_world(model, scratch, thumb_site)
        index_pos = true_fingertip_world(model, scratch, index_site)
        middle_pos = true_fingertip_world(model, scratch, middle_site)
        err = np.concatenate([
            thumb_target_world - thumb_pos,
            index_hold_world - index_pos,
            middle_hold_world - middle_pos,
        ])
        err_norm = float(np.linalg.norm(err))
        if err_norm < 1.5e-4:
            break

        cols = list(wrist_dof_adr) + list(thumb_dof_adr)
        rows = []
        for tip_body_name, local in (
            (wbc.FINGERTIP_SITE_BODIES[thumb_site], wbc.FINGERTIP_SITE_LOCAL_POS[thumb_site]),
            (wbc.FINGERTIP_SITE_BODIES[index_site], wbc.FINGERTIP_SITE_LOCAL_POS[index_site]),
            (wbc.FINGERTIP_SITE_BODIES[middle_site], wbc.FINGERTIP_SITE_LOCAL_POS[middle_site]),
        ):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tip_body_name)
            R = scratch.xmat[body_id].reshape(3, 3)
            world_pt = scratch.xpos[body_id] + R @ np.array(local)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, scratch, jacp, jacr, world_pt, body_id)
            rows.append(jacp[:, cols])
        J = np.vstack(rows)  # (9, 6)

        JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(JJt, err) * gain
        dq = np.clip(dq, -max_dq, max_dq)

        n_wrist = len(wrist_dof_adr)
        for k in range(n_wrist):
            lo, hi = wrist_ranges[k]
            q_wrist[k] = float(np.clip(q_wrist[k] + dq[k], lo, hi))
        for k in range(3):
            lo, hi = thumb_ranges[k]
            q_finger[k] = float(np.clip(q_finger[k] + dq[n_wrist + k], lo, hi))

    wrist_within = all(wrist_ranges[k][0] - 1e-6 <= q_wrist[k] <= wrist_ranges[k][1] + 1e-6 for k in range(len(wrist_ranges)))
    thumb_within = all(thumb_ranges[k][0] - 1e-6 <= q_finger[k] <= thumb_ranges[k][1] + 1e-6 for k in range(3))
    return dict(
        solved_finger_qpos=q_finger, solved_wrist_qpos=q_wrist, err_norm=err_norm,
        wrist_delta=q_wrist - base_wrist_qpos,
        within_limits=(wrist_within and thumb_within),
    )


@dataclass
class FingertipTarget:
    finger: str
    world_pos: np.ndarray
    object_local: np.ndarray
    face: str


def plan_tripod_targets(env, audit: dict, side: str, inset_margin: float = 0.008) -> dict[str, FingertipTarget]:
    """Stage 2's core: for one hand, KEEPS index/middle at their
    CURRENTLY-ACHIEVED true-tip positions (Stage 1 confirmed the existing
    natural-approach pipeline already reliably places them near the +X
    face -- this planner does not re-solve a problem that isn't broken),
    and defines a NEW thumb target that (a) sits on the opposing -X face
    at THIS HAND's OWN index/middle Y (not the global object centerline)
    and vertically between them -- directly targeting the root cause
    Stage 1's audit exposed: the current single shared centroid target
    pulls both hands' thumbs toward Y=0, so at THUMB_OPPOSE entry the two
    true thumb tips measured only 0.0088m apart (nearly colliding) even
    though each hand's OWN index/middle sit ~0.06m off the centerline."""
    obj_pos = np.array(audit["object"]["pos"])
    half = audit["object"]["half_size"]
    s = audit["sides"][side]
    index_local = np.array(s["current_true_tip_object_local"]["index"])
    middle_local = np.array(s["current_true_tip_object_local"]["middle"])
    thumb_local_now = np.array(s["current_true_tip_object_local"]["thumb"])

    face_sign = 1.0 if np.mean([index_local[0], middle_local[0]]) > 0 else -1.0
    thumb_face_sign = -face_sign
    target_local = np.array([
        thumb_face_sign * (half - inset_margin),
        0.5 * (index_local[1] + middle_local[1]),
        0.5 * (index_local[2] + middle_local[2]),
    ])
    # object orientation is identity at canonical yaw=0 (SIZE_12 default);
    # this planner does not (yet) handle non-zero object yaw.
    target_world = obj_pos + target_local

    return {
        "thumb": FingertipTarget("thumb", target_world, target_local,
                                  face=("FACE_NEG_X" if thumb_face_sign < 0 else "FACE_POS_X")),
        "index": FingertipTarget("index", obj_pos + index_local, index_local,
                                  face=("FACE_POS_X" if face_sign > 0 else "FACE_NEG_X")),
        "middle": FingertipTarget("middle", obj_pos + middle_local, middle_local,
                                   face=("FACE_POS_X" if face_sign > 0 else "FACE_NEG_X")),
    }
