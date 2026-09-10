"""Minimal standing stabiliser for free-base manipulation (Phase 5, step 1).

Why this exists, measured rather than assumed:

The bimanual grasp Expert was developed against a FIXED base, where the welded
pelvis silently absorbed every reaction torque the arms produced. Replaying the
identical commands with the pelvis free makes the robot rock with growing
amplitude -- pitch swinging -0.4 -> -3.2 -> +5.6 -> -3.9 degrees -- until a foot
leaves the ground (measured: left foot normal force 0.00 N at step 300) and it
topples. The robot's own centre of mass never leaves the support polygon while
this happens (margin stays 61-179 mm against a 122 mm nominal), so this is a
DYNAMIC problem, not a static balance one, and a wider stance would not fix it.

What this is: a textbook ankle-strategy regulator. It reads pelvis roll/pitch
and their rates and commands the four ankle actuators to oppose them, on top of
the stand pose. That is all.

What this is NOT: a walking controller, a ZMP/capture-point planner, or a
whole-body balance solver. It cannot take a step, so it cannot recover from a
disturbance that needs one. PROJECT_CONTEXT.md permits controller-assisted
architecture as an implementation means; this does not make the G1 a walker and
must never be reported as locomotion.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from humanoid_learning.envs import frames
from humanoid_learning.envs import whole_body_config as wbc

ANKLE_PITCH_JOINTS = ("left_ankle_pitch_joint", "right_ankle_pitch_joint")
ANKLE_ROLL_JOINTS = ("left_ankle_roll_joint", "right_ankle_roll_joint")


@dataclass
class StanceGains:
    """Ankle-strategy gains, in radians of ankle command per radian of tilt."""

    pitch_kp: float = 6.0
    pitch_kd: float = 0.6
    roll_kp: float = 4.0
    roll_kd: float = 0.4
    # Ankle commands are clamped so the stabiliser cannot drive the joint to a
    # limit and lever the robot over on its own.
    max_command_rad: float = 0.30


class StanceStabilizer:
    """Holds a free-base G1 upright while its arms work.

    Writes only the four ankle actuators. It never touches the arms, hands,
    waist or the environment's action, so a grasp controller running alongside
    it is unchanged.
    """

    def __init__(self, env, gains: StanceGains | None = None):
        self.env = env
        self.gains = gains or StanceGains()
        model = env.model
        self.enabled = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint") >= 0

        def act(name: str) -> int:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert aid >= 0, f"actuator not found: {name}"
            return aid

        self.pitch_ids = np.array([act(n) for n in ANKLE_PITCH_JOINTS])
        self.roll_ids = np.array([act(n) for n in ANKLE_ROLL_JOINTS])
        self.pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wbc.PELVIS_BODY)
        self._neutral_pitch = model.actuator_ctrlrange[self.pitch_ids].mean(axis=1) * 0.0
        self._neutral_roll = model.actuator_ctrlrange[self.roll_ids].mean(axis=1) * 0.0

    # ------------------------------------------------------------------
    def tilt(self) -> tuple[float, float, float, float]:
        """Pelvis roll/pitch and their rates, measured from the model."""
        data = self.env.data
        roll, pitch = frames.roll_pitch_from_quat(data.xquat[self.pelvis_body])
        # Free-joint angular velocity lives in qvel[3:6], in the world frame.
        angular = data.qvel[3:6]
        return float(roll), float(pitch), float(angular[0]), float(angular[1])

    def apply(self) -> dict[str, float]:
        """Command the ankles to oppose the current tilt. Call once per env step,
        before the environment integrates."""
        if not self.enabled:
            return {"roll": 0.0, "pitch": 0.0, "pitch_cmd": 0.0, "roll_cmd": 0.0}
        gains = self.gains
        roll, pitch, roll_rate, pitch_rate = self.tilt()
        # Positive pitch tips the pelvis backward; the ankle must push the same
        # way the tilt is going to bring the body back over the feet.
        pitch_cmd = float(np.clip(gains.pitch_kp * pitch + gains.pitch_kd * pitch_rate,
                                  -gains.max_command_rad, gains.max_command_rad))
        roll_cmd = float(np.clip(gains.roll_kp * roll + gains.roll_kd * roll_rate,
                                 -gains.max_command_rad, gains.max_command_rad))
        data = self.env.data
        model = self.env.model
        data.ctrl[self.pitch_ids] = np.clip(
            self._neutral_pitch + pitch_cmd,
            model.actuator_ctrlrange[self.pitch_ids, 0],
            model.actuator_ctrlrange[self.pitch_ids, 1],
        )
        data.ctrl[self.roll_ids] = np.clip(
            self._neutral_roll + roll_cmd,
            model.actuator_ctrlrange[self.roll_ids, 0],
            model.actuator_ctrlrange[self.roll_ids, 1],
        )
        return {"roll": roll, "pitch": pitch, "pitch_cmd": pitch_cmd, "roll_cmd": roll_cmd}
