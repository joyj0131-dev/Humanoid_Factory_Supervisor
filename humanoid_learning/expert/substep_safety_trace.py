"""SIZE_12 Thumb Contact-Loss Causality session: PHYSICS-SUBSTEP-level
trace of both thumbs' contact/force/safety-rollback history, built on top
of FixedBaseGraspEnv.substep_hook (a read-only callback invoked once per
physics substep inside step()'s frame_skip loop -- see grasp_env.py's
comment on the hook for why control-tick-granularity sampling, which is
all thumb_contact_diagnostics.py and grasp_wrench_diagnostics.py could see
before this session, cannot observe a within-tick rollback-then-reclose
pattern even if one exists.

Read-only diagnostic module (same isolation posture as the other
*_diagnostics.py modules): never imported by BimanualSidePinchExpert's
control loop or by any production env/expert default path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from humanoid_learning.expert.thumb_contact_diagnostics import (
    FaceRegion,
    classify_face_region,
    find_thumb_contact,
    relative_velocity_at_point,
)

INSET_MARGIN = 0.012


@dataclass
class SubstepRecord:
    control_step: int
    substep_idx: int
    global_substep: int
    state: str
    touched: bool
    region: FaceRegion
    face: str
    normal_force: float
    tangential_force: float
    resultant_force: float
    tangential_vel: float
    normal_vel: float
    ang_speed: float
    thumb_qpos: np.ndarray
    thumb_ctrl: np.ndarray
    prev_group_ctrl: np.ndarray
    safety_triggered_this_substep: bool


@dataclass
class ContactEpisode:
    side: str
    start_gsubstep: int
    end_gsubstep: int | None
    start_cs: int
    end_cs: int | None
    samples: list[SubstepRecord] = field(default_factory=list)

    @property
    def peak_force(self) -> float:
        return max((s.resultant_force for s in self.samples), default=0.0)


def collect_substep_trace(env, expert, obj_body_id: int, thumb_body_id: dict, thumb_qpos_adr: dict,
                           thumb_group_act_ids: dict, half_size: float, max_steps: int = 1200):
    """Runs the given (already-reset) env/expert pair to completion (or
    max_steps), attaching env.substep_hook to record a SubstepRecord per
    physics substep per hand. Returns (outcome, trace, synergy_log) where
    trace = {"left": [...], "right": [...]} and synergy_log is a list of
    per-CONTROL-STEP dicts (group_synergy/raw force/safety-event summary),
    matching what grasp_expert.py's own per-tick force regulation actually
    sees (i.e. the state AFTER all that tick's substeps, including any
    rollback, have already happened)."""
    model, data = env.model, env.data
    obj_qpos_adr = env._object_qpos_adr
    obj_dof_adr = env._object_dof_adr
    intended_face = {"left": None, "right": None}
    trace = {"left": [], "right": []}
    box = {"i": -1}

    def hook(e, substep_idx, prev_group_ctrl):
        i = box["i"]
        gsub = i * e.config.frame_skip + substep_idx
        obj_pos = e.data.qpos[obj_qpos_adr:obj_qpos_adr + 3].copy()
        obj_quat = e.data.qpos[obj_qpos_adr + 3:obj_qpos_adr + 7].copy()
        ang_speed = float(np.linalg.norm(e.data.qvel[obj_dof_adr + 3:obj_dof_adr + 6]))
        for side_idx, side in enumerate(("left", "right")):
            touched, point, normal, nf, tf, _ = find_thumb_contact(e.model, e.data, side, obj_body_id)
            region, face_label, nvel, tvel = FaceRegion.NO_CONTACT, "NONE", 0.0, 0.0
            if touched:
                if intended_face[side] is None:
                    _, fl, _ = classify_face_region(point, obj_pos, obj_quat, half_size, INSET_MARGIN)
                    if fl != "EDGE_OR_CORNER_AMBIGUOUS":
                        intended_face[side] = fl
                region, face_label, _ = classify_face_region(
                    point, obj_pos, obj_quat, half_size, INSET_MARGIN, intended_face=intended_face[side],
                )
                nvel, tvel = relative_velocity_at_point(
                    e.model, e.data, obj_body_id, thumb_body_id[side], point, normal, obj_dof_adr
                )
            rollback_now = bool(np.allclose(e.data.ctrl[thumb_group_act_ids[side]], prev_group_ctrl[side_idx][0]))
            safety_now = any(ev[0] == side and ev[1] == 0 for ev in e.last_safety_events) or \
                any(ev[0] == "hand_hand" for ev in e.last_safety_events)
            trace[side].append(SubstepRecord(
                control_step=i, substep_idx=substep_idx, global_substep=gsub,
                state=expert.state.name, touched=touched, region=region, face=face_label,
                normal_force=nf, tangential_force=tf, resultant_force=float(np.hypot(nf, tf)),
                tangential_vel=tvel, normal_vel=nvel, ang_speed=ang_speed,
                thumb_qpos=e.data.qpos[thumb_qpos_adr[side]].copy(),
                thumb_ctrl=e.data.ctrl[thumb_group_act_ids[side]].copy(),
                prev_group_ctrl=np.array(prev_group_ctrl[side_idx][0]),
                safety_triggered_this_substep=safety_now,
            ))

    env.substep_hook = hook
    synergy_log = []
    outcome = None
    from humanoid_learning.expert.grasp_expert import GraspState
    for i in range(max_steps):
        box["i"] = i
        outcome = expert.step()
        events_this_tick = list(env.last_safety_events)
        synergy_log.append(dict(
            i=i, state=outcome.state.name,
            left_thumb_syn=float(expert.left_group_synergy[0]), right_thumb_syn=float(expert.right_group_synergy[0]),
            left_force_raw0=float(expert.left_group_force_raw[0]), right_force_raw0=float(expert.right_group_force_raw[0]),
            n_safety_events=len(events_this_tick),
            max_safety_force=max((e[2] for e in events_this_tick), default=0.0),
            safety_sides=[e[0] for e in events_this_tick],
        ))
        if outcome.state in (GraspState.FAILURE, GraspState.SUCCESS):
            break
    env.substep_hook = None
    return outcome, trace, synergy_log


def find_contact_episodes(side: str, records: list[SubstepRecord]) -> list[ContactEpisode]:
    """Splits one hand's substep record list into contiguous
    touched=True episodes (substep granularity -- finer than
    thumb_contact_diagnostics.py's control-step-granularity ContactEvent,
    needed because a rollback-reclose limit cycle could in principle
    start and resolve entirely within fewer physics substeps than one
    control tick)."""
    episodes: list[ContactEpisode] = []
    cur: ContactEpisode | None = None
    for r in records:
        if r.touched and cur is None:
            cur = ContactEpisode(side=side, start_gsubstep=r.global_substep, end_gsubstep=None,
                                  start_cs=r.control_step, end_cs=None, samples=[r])
        elif r.touched and cur is not None:
            cur.samples.append(r)
        elif not r.touched and cur is not None:
            cur.end_gsubstep = r.global_substep
            cur.end_cs = r.control_step
            episodes.append(cur)
            cur = None
    if cur is not None:
        episodes.append(cur)
    return episodes


def check_safety_rollback_limit_cycle_criteria(episodes: list[ContactEpisode], force_limit: float = 8.0) -> dict:
    """Section 7's strict multi-criteria checklist, applied mechanically:
    (1) force-threshold-crossing, (2) actual rollback executed, (3)
    rollback-then-force-drop-or-loss, (4) resumed closing after rollback
    (thumb_ctrl approaches or exceeds the pre-rollback commanded value
    again within the same episode), (5) re-contact force spike after a
    drop-then-recover within one episode, evaluated per episode, and (6)
    the pattern (an episode independently satisfying 1-5) occurring at
    least twice across ALL episodes of ALL hands. Returns a dict with the
    per-criterion tallies and the final confirmed: bool -- this function
    does not editorialize about OTHER possible causes (e.g. rotation-
    induced loss), it only mechanically tests THIS one hypothesis."""
    qualifying = 0
    details = []
    for ep in episodes:
        samples = ep.samples
        force_crossed = any(s.resultant_force > force_limit for s in samples)
        rollback_seen = any(s.safety_triggered_this_substep for s in samples)
        if not (force_crossed and rollback_seen):
            details.append(dict(episode=(ep.side, ep.start_cs, ep.end_cs), qualifies=False, reason="no_force_cross_or_no_rollback"))
            continue
        # crude drop-then-reclose-then-spike-again detector within the episode
        forces = [s.resultant_force for s in samples]
        n = len(forces)
        reclose_spike_pairs = 0
        i = 0
        while i < n - 1:
            if forces[i] > force_limit:
                # find next local minimum then next re-crossing
                j = i + 1
                while j < n and forces[j] >= forces[i]:
                    j += 1
                if j < n:
                    k = j
                    while k < n and forces[k] <= force_limit:
                        k += 1
                    if k < n:
                        reclose_spike_pairs += 1
                        i = k
                        continue
            i += 1
        qualifies = reclose_spike_pairs >= 1
        if qualifies:
            qualifying += 1
        details.append(dict(episode=(ep.side, ep.start_cs, ep.end_cs), qualifies=qualifies,
                             reclose_spike_pairs=reclose_spike_pairs))
    return dict(n_episodes=len(episodes), n_qualifying_episodes=qualifying,
                confirmed=qualifying >= 2, details=details)
