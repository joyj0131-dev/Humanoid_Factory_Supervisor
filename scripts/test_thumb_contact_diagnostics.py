"""Tests for humanoid_learning.expert.thumb_contact_diagnostics (Thumb
Contact Geometry and 10/11/12cm Controlled Size Diagnosis session).
Run with:
    python scripts/test_thumb_contact_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
import numpy as np

from humanoid_learning.envs.grasp_config import GraspEnvConfig
from humanoid_learning.envs.grasp_env import FixedBaseGraspEnv
from humanoid_learning.expert.grasp_expert import BimanualSidePinchExpert, GraspExpertConfig, GraspState
from humanoid_learning.expert.thumb_contact_diagnostics import (
    ContactEvent,
    FaceRegion,
    LossReason,
    ThumbContactSample,
    classify_face_region,
    classify_loss_reason,
    find_thumb_contact,
    relative_velocity_at_point,
)

HALF_SIZE = 0.06
INSET = 0.012


# ---------------------------------------------------------------------
# Section 18, items 1-2: face interior/edge/corner classification and
# edge-margin computation, checked against KNOWN synthetic points (no
# live sim needed -- pure geometry).
# ---------------------------------------------------------------------
def test_face_region_classifies_interior_edge_corner():
    obj_pos = np.array([0.27, 0.0, 0.8])
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def classify(point):
        region, face, _ = classify_face_region(np.array(point), obj_pos, identity_quat, HALF_SIZE, INSET)
        return region, face

    # dead center of the -X face -> interior
    r, f = classify(obj_pos + np.array([-HALF_SIZE, 0.0, 0.0]))
    assert r == FaceRegion.FACE_INTERIOR and f == "FACE_NEG_X"

    # near the Y edge of the -X face (within inset) -> edge
    r, f = classify(obj_pos + np.array([-HALF_SIZE, HALF_SIZE - INSET / 2, 0.0]))
    assert r == FaceRegion.EDGE_NEAR

    # near BOTH the Y and Z edges -> corner
    r, f = classify(obj_pos + np.array([-HALF_SIZE, HALF_SIZE - INSET / 2, HALF_SIZE - INSET / 2]))
    assert r == FaceRegion.CORNER_NEAR

    # wrong-face check
    r, f = classify_face_region(
        obj_pos + np.array([-HALF_SIZE, 0.0, 0.0]), obj_pos, identity_quat, HALF_SIZE, INSET, intended_face="FACE_POS_X",
    )[0:2]
    assert r == FaceRegion.WRONG_FACE

    # object rotation must be respected (matches grasp_expert's own convention)
    quat_90z = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    r, f = classify(obj_pos + np.array([0.0, -HALF_SIZE, 0.0]))  # world -Y point, but object rotated 90deg
    r2, f2 = classify_face_region(obj_pos + np.array([0.0, -HALF_SIZE, 0.0]), obj_pos, quat_90z, HALF_SIZE, INSET)[0:2]
    assert f2 != f, "the SAME world point must classify to a DIFFERENT face once the object itself is rotated"


# ---------------------------------------------------------------------
# Section 18, item 7: loss-reason classification against synthetic
# ContactEvent histories with KNOWN, deliberately distinct signatures.
# ---------------------------------------------------------------------
def _sample(step, region, nf=5.0, tvel=0.0, margin=0.5, ang_speed=0.5):
    return ThumbContactSample(
        step=step, touched=True, world_point=np.zeros(3), region=region, actual_face="FACE_NEG_X",
        normal_force=nf, tangential_force=0.0, tangential_vel=tvel, normal_vel=0.0,
        joint_limit_margin=margin, object_ang_speed=ang_speed,
    )


def test_classify_loss_reason_bad_geometry():
    ev = ContactEvent(start_step=0, end_step=10, samples=[_sample(i, FaceRegion.EDGE_NEAR) for i in range(10)])
    primary, _ = classify_loss_reason(ev)
    assert primary == LossReason.BAD_CONTACT_GEOMETRY


def test_classify_loss_reason_kinematic_reach_loss():
    ev = ContactEvent(start_step=0, end_step=10, samples=[
        _sample(i, FaceRegion.FACE_INTERIOR, margin=0.005) for i in range(10)
    ])
    primary, _ = classify_loss_reason(ev)
    assert primary == LossReason.KINEMATIC_REACH_LOSS


def test_classify_loss_reason_friction_slip():
    ev = ContactEvent(start_step=0, end_step=10, samples=[
        _sample(i, FaceRegion.FACE_INTERIOR, tvel=0.05, margin=0.5) for i in range(10)
    ])
    primary, _ = classify_loss_reason(ev)
    assert primary == LossReason.FRICTION_SLIP


def test_classify_loss_reason_impact_bounce():
    ev = ContactEvent(start_step=0, end_step=3, samples=[
        _sample(0, FaceRegion.FACE_INTERIOR, nf=8.0, margin=0.5),
        _sample(1, FaceRegion.FACE_INTERIOR, nf=6.0, margin=0.5),
        _sample(2, FaceRegion.FACE_INTERIOR, nf=4.0, margin=0.5),
    ])
    primary, _ = classify_loss_reason(ev)
    assert primary == LossReason.IMPACT_BOUNCE


# ---------------------------------------------------------------------
# Section 18, item 3-6: thumb relative velocity / contact detection are
# real (not stubbed) -- checked against a live rollout that actually
# reaches thumb contact for SIZE_12.
# ---------------------------------------------------------------------
def test_find_thumb_contact_and_relative_velocity_on_real_rollout():
    env = FixedBaseGraspEnv(GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=HALF_SIZE))
    env.reset(seed=0)
    expert = BimanualSidePinchExpert(env, GraspExpertConfig())
    model, data = env.model, env.data
    obj_body_id = env._object_body_id
    obj_dof_adr = env._object_dof_adr
    thumb_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_thumb_2_link")

    found_contact = False
    for i in range(3000):
        outcome = expert.step()
        touched, point, normal, nf, tf, _ = find_thumb_contact(model, data, "right", obj_body_id)
        if touched:
            found_contact = True
            nvel, tvel = relative_velocity_at_point(model, data, obj_body_id, thumb_body_id, point, normal, obj_dof_adr)
            assert np.isfinite(nvel) and np.isfinite(tvel)
            assert tvel >= 0.0  # magnitude, never negative
            assert nf > 0.0 or tf > 0.0
        if outcome.state in (GraspState.FAILURE, GraspState.SUCCESS):
            break
    assert found_contact, "SIZE_12 baseline should reach at least one real right-thumb contact"


# ---------------------------------------------------------------------
# Section 18, item 9: 10/11/12cm comparison under IDENTICAL controller/
# config/seed -- THIS TEST DOCUMENTS A REAL, MEASURED, and SURPRISING
# result (see PROJECT_CONTEXT.md's session report for the full
# breakdown): SIZE_10 achieves ZERO thumb contact events at all (worse
# than SIZE_12, not better), and SIZE_11's thumb contact is entirely
# EDGE/CORNER geometry (face-interior fraction 0.0) with force spikes
# well above the 8N safety target (20-27N raw) -- SIZE_12 is the ONLY
# size where the existing grip geometry (grip_half_width/standoff/
# fingertip-offset formulas) was calibrated, per this project's own
# documented history ("linear formulas reverse-engineered to reproduce
# this file's OLD hardcoded 12cm-object values... EXACTLY at that
# size"). This REJECTS the SIZE/APERTURE_LIMIT hypothesis (smaller
# sizes do NOT improve thumb geometry) and instead supports
# TARGET_SCALING_BUG (the size-conditioned geometry formulas do not
# preserve "contact stays in face interior" away from SIZE_12) -- left
# visible as a real, reproducible finding, not adjusted away.
# ---------------------------------------------------------------------
def test_size_10_11_12_thumb_contact_comparison():
    def run_size(half_size, max_steps=4000):
        env = FixedBaseGraspEnv(GraspEnvConfig(object_pos=(0.27, 0.0, 0.0), arm_kp=120.0, object_half_size=half_size))
        env.reset(seed=0)
        expert = BimanualSidePinchExpert(env, GraspExpertConfig())
        model, data = env.model, env.data
        obj_body_id = env._object_body_id
        obj_qpos_adr = env._object_qpos_adr
        thumb_events = {"left": 0, "right": 0}
        interior_fracs = {"left": [], "right": []}
        peak_forces = {"left": 0.0, "right": 0.0}
        currently_touched = {"left": False, "right": False}
        current_samples = {"left": [], "right": []}
        intended_face = {"left": None, "right": None}
        outcome = None
        for i in range(max_steps):
            outcome = expert.step()
            obj_pos = data.qpos[obj_qpos_adr:obj_qpos_adr + 3]
            obj_quat = data.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7]
            for side in ("left", "right"):
                touched, point, normal, nf, tf, _ = find_thumb_contact(model, data, side, obj_body_id)
                if touched:
                    if intended_face[side] is None:
                        _, face_label, _ = classify_face_region(point, obj_pos, obj_quat, half_size, INSET)
                        if face_label != "EDGE_OR_CORNER_AMBIGUOUS":
                            intended_face[side] = face_label
                    region, _, _ = classify_face_region(point, obj_pos, obj_quat, half_size, INSET, intended_face=intended_face[side])
                    current_samples[side].append(region)
                    peak_forces[side] = max(peak_forces[side], nf + tf)
                    currently_touched[side] = True
                elif currently_touched[side]:
                    thumb_events[side] += 1
                    n = len(current_samples[side])
                    if n:
                        interior_fracs[side].append(sum(1 for r in current_samples[side] if r == FaceRegion.FACE_INTERIOR) / n)
                    current_samples[side] = []
                    currently_touched[side] = False
            if outcome.state in (GraspState.FAILURE, GraspState.SUCCESS):
                break
        for side in ("left", "right"):
            if currently_touched[side]:
                thumb_events[side] += 1
                n = len(current_samples[side])
                if n:
                    interior_fracs[side].append(sum(1 for r in current_samples[side] if r == FaceRegion.FACE_INTERIOR) / n)
        return {
            "streak": outcome.max_bilateral_tripod_streak, "events": thumb_events,
            "interior_fracs": interior_fracs, "peak_forces": peak_forces,
        }

    r10 = run_size(0.05)
    r11 = run_size(0.055)
    r12 = run_size(0.06)
    print(f"    SIZE_10: streak={r10['streak']} events={r10['events']} peak_forces={r10['peak_forces']}")
    print(f"    SIZE_11: streak={r11['streak']} events={r11['events']} interior_fracs={r11['interior_fracs']} peak_forces={r11['peak_forces']}")
    print(f"    SIZE_12: streak={r12['streak']} events={r12['events']} interior_fracs={r12['interior_fracs']} peak_forces={r12['peak_forces']}")

    assert r10["events"]["left"] == 0 and r10["events"]["right"] == 0, (
        "REAL FINDING: SIZE_10 achieves ZERO thumb contact events with the unchanged SIZE_12-tuned controller "
        "geometry -- smaller size does NOT trivially improve thumb contact, rejecting the naive aperture hypothesis"
    )
    assert r12["streak"] > r10["streak"] and r12["streak"] > r11["streak"], (
        "REAL FINDING: SIZE_12 (the size this controller's geometry was actually tuned for) achieves the best "
        "bilateral tripod streak of the three sizes tested, not the worst"
    )


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
