"""Walking for the factory G1, using Unitree's pre-trained G1 locomotion policy.

Locomotion here is an EXTERNAL TOOL, not part of this project's research. The
BC vs BC+PPO comparison this project studies applies to the recovery
manipulation skill; walking is treated as given. Provenance, license and
checksum are recorded in assets/policies/g1_walk/NOTICE.

Why an external policy: a from-scratch attempt got the robot to shift 90% of its
weight onto one foot but never to full single support -- it toppled every time
the second foot's normal force reached 0.00 N, because the foot sole is only
37.8 mm in half-width. Stepping needs full single support, so that route needed
a ZMP/capture-point controller of its own.

The policy contract, read from unitree_rl_gym's deploy_mujoco.py and g1.yaml:

    obs (47) = [ base angular velocity (3) * 0.25,
                 gravity direction in base frame (3),
                 velocity command (3) * [2.0, 2.0, 0.25],
                 leg joint pos - default (12),
                 leg joint vel (12) * 0.05,
                 previous action (12),
                 sin/cos of a 0.8 s gait phase (2) ]
    action (12) -> leg targets = action * 0.25 + default angles
    run at 50 Hz, with leg PD gains kp=[100,100,100,150,40,40], kd=[2,2,2,4,2,2]

Two adaptations are needed here and are applied explicitly:
  * Unitree's scene is legs-only and indexes ``qpos[7:]``; this robot has a
    waist, arms and two Sharpa hands, so leg indices are resolved by joint name.
  * This env's leg actuators are position actuators at kp=500. A MuJoCo position
    actuator with gainprm=[kp] and biasprm=[0,-kp,-kv] computes exactly
    ``kp*(ctrl - q) - kv*qvel``, i.e. the policy's own PD law, so the gains are
    retuned to the policy's values rather than a torque path being bolted on.

Known sim-to-sim gap, stated rather than hidden: the policy was trained on a
stock G1, and this robot carries 2.49 kg of Sharpa hands (7% of its mass) that
the policy never saw. It walks anyway, but it under-tracks velocity commands,
which is why the navigator below uses integral action.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from humanoid_learning.envs import frames
from humanoid_learning.envs import task_config as tc
from humanoid_learning.envs import whole_body_config as wbc

POLICY_PATH = Path(tc.PROJECT_ROOT) / "assets" / "policies" / "g1_walk" / "motion.pt"

# Verbatim from unitree_rl_gym deploy/deploy_mujoco/configs/g1.yaml.
LEG_KP = np.array([100, 100, 100, 150, 40, 40] * 2, dtype=np.float64)
LEG_KD = np.array([2, 2, 2, 4, 2, 2] * 2, dtype=np.float64)
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2, dtype=np.float64)
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25])
GAIT_PERIOD = 0.8
POLICY_DECIMATION = 2  # this env steps at 100 Hz; the policy runs at 50 Hz
NUM_LEG_ACTIONS = 12
NUM_OBS = 47


def gravity_orientation(quaternion) -> np.ndarray:
    """Gravity direction in the base frame (unitree_rl_gym's formulation)."""
    qw, qx, qy, qz = quaternion
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ])


@dataclass
class NavigationGains:
    """Closed-loop pose tracking on top of the velocity-command policy."""

    kp_linear: float = 0.6
    kp_yaw: float = 1.0
    ki_linear: float = 0.9
    max_linear: float = 0.35
    max_yaw: float = 0.3
    # Integral only near the goal, so it cannot wind up during the approach.
    integral_radius_m: float = 0.6
    integral_clamp: float = 0.5
    # Scale the speed limit down close in, so the robot settles instead of
    # oscillating around the target.
    slowdown_radius_m: float = 0.5
    min_speed_fraction: float = 0.25
    # Once inside this band, stop commanding motion and let the policy simply
    # stand. Without it the robot reaches the pose, holds, then walks itself
    # back out -- it met the gate's 1 s hold but did not settle, which is no use
    # for standing still and manipulating.
    # Measured: the closest this controller actually gets is 49.9 mm (station 0)
    # and 53.9 mm (station 1), after which it swings back out to 230-340 mm. A
    # 0.05 band was therefore just barely unreachable. 0.06 latches both, 40 mm
    # inside the gate's 100 mm limit, and the wide release band stops standing
    # wobble from restarting the walk.
    arrive_radius_m: float = 0.06
    arrive_yaw_rad: float = 0.10
    release_radius_m: float = 0.20


class G1WalkPolicy:
    """Drives the 12 leg joints from a velocity command."""

    def __init__(self, env, policy_path: Path | str = POLICY_PATH):
        import torch  # imported lazily: only walking needs it

        path = Path(policy_path)
        if not path.exists():
            raise FileNotFoundError(
                f"G1 locomotion policy not found at {path}. "
                "Run: python3 scripts/install_g1_walk_policy.py"
            )
        self._torch = torch
        self.policy = torch.jit.load(str(path))
        self.policy.eval()
        self.env = env
        model = env.model

        def joint(name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"joint not found: {name}"
            return jid

        self.act_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in wbc.LEG_JOINTS]
        )
        self.qpos_adr = np.array([model.jnt_qposadr[joint(n)] for n in wbc.LEG_JOINTS])
        self.dof_adr = np.array([model.jnt_dofadr[joint(n)] for n in wbc.LEG_JOINTS])
        base = joint(tc.FLOATING_BASE_JOINT)
        self.base_qpos_adr = model.jnt_qposadr[base]
        self.base_dof_adr = model.jnt_dofadr[base]
        self.pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wbc.PELVIS_BODY)
        self.foot_bodies = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
            for side in ("left", "right")
        ]

        self.holding = False
        self._apply_policy_gains()
        self.reset()

    def _apply_policy_gains(self) -> None:
        """Retune the leg position actuators to the policy's PD law."""
        model = self.env.model
        for k, aid in enumerate(self.act_ids):
            model.actuator_gainprm[aid, 0] = LEG_KP[k]
            model.actuator_biasprm[aid, 1] = -LEG_KP[k]
            model.actuator_biasprm[aid, 2] = -LEG_KD[k]

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Put the legs in the policy's default stance and clear its history."""
        self.action = np.zeros(NUM_LEG_ACTIONS)
        self.leg_target = DEFAULT_ANGLES.copy()
        self._tick = 0
        self.env.data.qpos[self.qpos_adr] = DEFAULT_ANGLES
        self.env._leg_target[:] = DEFAULT_ANGLES
        mujoco.mj_forward(self.env.model, self.env.data)

    def base_pose(self) -> tuple[np.ndarray, float]:
        data = self.env.data
        return data.xpos[self.pelvis_body][:2].copy(), float(
            frames.yaw_from_quat(data.xquat[self.pelvis_body])
        )

    def observe(self, command: np.ndarray) -> np.ndarray:
        data = self.env.data
        base = self.base_qpos_adr
        angular = data.qvel[self.base_dof_adr + 3 : self.base_dof_adr + 6] * ANG_VEL_SCALE
        gravity = gravity_orientation(data.qpos[base + 3 : base + 7])
        joint_pos = (data.qpos[self.qpos_adr] - DEFAULT_ANGLES) * DOF_POS_SCALE
        joint_vel = data.qvel[self.dof_adr] * DOF_VEL_SCALE
        phase = (self._tick * POLICY_DECIMATION * 0.005) % GAIT_PERIOD / GAIT_PERIOD
        return np.concatenate([
            angular, gravity, np.asarray(command) * CMD_SCALE,
            joint_pos, joint_vel, self.action,
            [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)],
        ]).astype(np.float32)

    def in_double_support(self, min_force: float = 40.0) -> bool:
        """Both feet carrying real load.

        Freezing the legs mid-gait -- possibly with a foot in the air -- leaves
        the robot leaning and it settles ~90 mm away from where it arrived. The
        handoff waits for a moment when both feet are down.
        """
        model, data = self.env.model, self.env.data
        loads = [0.0, 0.0]
        for c in range(data.ncon):
            contact = data.contact[c]
            geoms = {contact.geom1, contact.geom2}
            for k, body in enumerate(self.foot_bodies):
                if any(model.geom_bodyid[g] == body for g in geoms):
                    value = np.zeros(6)
                    mujoco.mj_contactForce(model, data, c, value)
                    loads[k] += float(np.linalg.norm(value[:3]))
        return min(loads) >= min_force

    def hold_stance(self) -> None:
        """Stop walking and hand the legs back to a standing hold.

        A velocity-command walking policy cannot stand still: commanding zero
        still produces gait steps, and this one drifts ~0.2 m away from a pose
        it has just reached (0.6 m over 15 s from a standing start). Arriving is
        therefore a HANDOFF, not a smaller velocity command -- the leg actuators
        go back to the stiff position gains and hold wherever the feet are.
        """
        model = self.env.model
        stiff = 500.0
        damping = 2.0 * np.sqrt(stiff)
        for aid in self.act_ids:
            model.actuator_gainprm[aid, 0] = stiff
            model.actuator_biasprm[aid, 1] = -stiff
            model.actuator_biasprm[aid, 2] = -damping
        self.leg_target = self.env.data.qpos[self.qpos_adr].copy()
        self.env._leg_target[:] = self.leg_target
        self.holding = True

    def step(self, command) -> np.ndarray:
        """Advance one env step. Returns the leg joint targets it commanded."""
        if self.holding:
            self.env._leg_target[:] = self.leg_target
            return self.leg_target
        if self._tick % POLICY_DECIMATION == 0:
            observation = self.observe(np.asarray(command, dtype=np.float64))
            with self._torch.no_grad():
                tensor = self._torch.from_numpy(observation).unsqueeze(0)
                self.action = self.policy(tensor).numpy().squeeze()
            self.leg_target = self.action * ACTION_SCALE + DEFAULT_ANGLES
        self._tick += 1
        # The env applies ctrl[legs] = _leg_target each step, so writing the
        # target here and passing a zero leg action leaves it exactly in place.
        self.env._leg_target[:] = self.leg_target
        return self.leg_target


class WalkToPose:
    """Closed-loop navigation to a station's manipulation pose.

    Converts a pose error into the velocity command the policy expects. The
    integral term exists because the policy under-tracks small commands on this
    robot (the 2.49 kg of hands it never trained with), which otherwise leaves a
    steady ~70 mm shortfall -- uncomfortably close to the gate's 100 mm limit.
    """

    def __init__(self, walker: G1WalkPolicy, goal_xy, goal_yaw: float,
                 gains: NavigationGains | None = None):
        self.walker = walker
        self.goal_xy = np.asarray(goal_xy, dtype=np.float64)
        self.goal_yaw = float(goal_yaw)
        self.gains = gains or NavigationGains()
        self.integral = np.zeros(2)
        self.arrived = False

    def command(self) -> np.ndarray:
        g = self.gains
        position, yaw = self.walker.base_pose()
        error_world = self.goal_xy - position
        cos, sin = np.cos(-yaw), np.sin(-yaw)
        error = np.array([
            cos * error_world[0] - sin * error_world[1],
            sin * error_world[0] + cos * error_world[1],
        ])
        yaw_error = float((self.goal_yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi)

        distance = float(np.linalg.norm(error))
        # Latch arrival with hysteresis so small wobble does not restart walking.
        if self.arrived:
            # Once handed off to a standing hold, stay handed off: restarting the
            # walk from a stiff-legged stance is a separate problem.
            pass
        elif (distance < g.arrive_radius_m and abs(yaw_error) < g.arrive_yaw_rad
              and self.walker.in_double_support()):
            self.arrived = True
        if self.arrived:
            if not self.walker.holding:
                self.walker.hold_stance()
            return np.zeros(3)

        if distance < g.integral_radius_m:
            self.integral = np.clip(self.integral + error * 0.02, -g.integral_clamp, g.integral_clamp)
        speed = g.max_linear * min(1.0, max(g.min_speed_fraction, distance / g.slowdown_radius_m))
        return np.array([
            float(np.clip(g.kp_linear * error[0] + g.ki_linear * self.integral[0], -speed, speed)),
            float(np.clip(g.kp_linear * error[1] + g.ki_linear * self.integral[1], -speed, speed)),
            float(np.clip(g.kp_yaw * yaw_error, -g.max_yaw, g.max_yaw)),
        ])

    def step(self) -> np.ndarray:
        return self.walker.step(self.command())
