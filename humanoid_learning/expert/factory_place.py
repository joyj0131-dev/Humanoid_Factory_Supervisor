"""Actuator-only place/retract prototype; never moves the free object directly.

Reached only after FactoryRecovery has WALKED the held block to the placing
stance, which is what made a loaded transfer survivable at all: reaching for the
canonical spot from the picking stance is a 233 mm pull that closed no distance
whatsoever (object error 221 -> 217 mm) and toppled the robot in 380 ticks.

The stage order is deliberate. DESCEND puts the block straight down on the belt
at whatever XY the carry reached, so the belt takes the weight BEFORE the arms
are asked to move it anywhere; only then does SLIDE push it to the canonical
spot. Translating first, at height, is what the earlier version did, and holding
the load out at extension is what toppled the robot.

Known limitation, measured on seed 0 / station 0: DESCEND now lands the block on
the belt within 2 mm of its resting height, and the robot falls during SLIDE
with the part still 118 mm from the canonical spot. RELEASE/RETRACT/restart are
wired but have never run, so this stays opt-in. Note the carry leaves ~99 mm of
that gap fore/aft and the loaded gait will not close it (fore/aft error pinned
at +110 mm for 2500 ticks, with no forbidden contact recorded, so the conveyor
rail is not what stops it).
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from humanoid_learning.envs import factory_config as fc
from humanoid_learning.expert.sharpa_bimanual_grasp_expert import _quintic_scale
from humanoid_learning.expert.sharpa_contact_lift import SharpaContactLift


@dataclass
class PlaceConfig:
    descend_ticks: int = 500
    slide_ticks: int = 600
    settle_ticks: int = 200
    release_ticks: int = 150
    retract_ticks: int = 400
    # The part only has to land inside the recovery gate's 60 mm, and every
    # millimetre demanded past that is bought with loaded arm extension.
    position_tolerance_m: float = 0.03
    # With the belt carrying the weight, turn the held block square before
    # letting go: the slide turned it 4.7 -> 17.4 deg in the hands (one hand
    # pushing 22 N against the other's 9 N), past the 0.15 rad yaw gate with
    # everything else (2.6 cm, flat, still, hands clear) already passing.
    square_yaw: bool = True
    square_ticks: int = 150
    square_settle_ticks: int = 50
    square_tolerance_rad: float = 0.05
    square_tries: int = 5
    # The block slips in the hands: each turn removed only 35-40 % of the
    # error (17 -> 11 -> 7 -> 4.5 deg), so turn the hands further than the error.
    square_gain: float = 1.6
    # Later turns at gain 1 and stop once a turn stops helping by 1 deg: more
    # tries walked the error back out (measured -4.1 -> -5.2 -> -7.5 -> -7.8).
    square_later_gain: float = 1.0
    # Re-seat before squaring: the block arrives tilted 9-14 deg in the hands
    # and squaring it wedged (40-45 N) and dragged it 15 -> 40 mm off target.
    # With the belt under it, open the palms briefly so it settles flat, then
    # close back onto the same targets.
    reseat: bool = True
    reseat_open_m: float = 0.02
    reseat_ticks: int = 60
    reseat_settle_ticks: int = 40
    square_min_gain_deg: float = 1.0
    # Optional: once the belt carries the weight the hands only have to steer the block.
    # Unregulated, the squeeze climbed 21 -> 88 N through slide and square on
    # the longest carry, wedged the 13 deg tilted block between the palms
    # (tilt 21 deg) and it toppled 7 cm away on release. Hold the mean pad
    # force in a band by opening/closing the palms along the grasp axis.
    # Off by default: bands of 5-15 N and 6-30 N both scored 9/12 against 11/12
    # without it (one block lost at 5 N, others not verified) -- kept to retry.
    squeeze_band_n: tuple | None = None
    squeeze_rate_m: float = 0.0001
    squeeze_open_max_m: float = 0.02
    squeeze_close_max_m: float = 0.004
    # Retract in two moves: up off the block, then back and down in front of
    # the torso. The old 8 cm back / 8 cm up left the palms over the belt edge,
    # where the restarted arm's jaw swung into the left thumb and knocked G1
    # over (measured: roll 0.05 -> 0.61 rad within 1 s of the restart).
    retract_up_m: float = 0.06
    retract_back_m: float = 0.22
    retract_down_m: float = 0.10
    retract_lift_fraction: float = 0.4
    # At the slide's end the hands can hold the block 3-5 mm off the belt
    # (measured: 6 and 17 mm from the target, no belt contact, so no release).
    # Press the commanded height down until the belt touches it.
    slide_press_rate_m: float = 0.0001
    # Open along the measured palm-to-palm axis and only retract once neither
    # hand pushes on the block: one palm still loading it at 12 N after the
    # fixed 5 cm opening dragged it 3 -> 7 cm off the target on the retract.
    release_open_m: float = 0.05
    release_open_max_m: float = 0.09
    release_open_rate_m: float = 0.0002
    release_free_n: float = 0.3
    release_free_wait_ticks: int = 200
    # Opening the fingers while the palms spread spun a loosely held block
    # 2.6 -> 18-19.5 deg (0.9 rad/s) on release. 'fingers_first' holds the
    # palms still while the fingers open; 'spread_first' keeps the fingers
    # closed until the palms are apart; 'together' is the old behaviour.
    release_order: str = 'together'
    release_finger_ticks: int = 80
    slide_press_max_m: float = 0.02


class FactoryPlace:
    def __init__(self, recovery, config=None):
        self.recovery = recovery
        self.env, self.expert = recovery.env, recovery.expert
        self.view = recovery.grasp
        self.config = config or PlaceConfig()
        self.stage, self.tick, self.failure = 'DESCEND', 0, None
        self.target = self.env._part_rest_pos(recovery.station, fc.LOCAL_CANONICAL_PART_XY)
        self.start = self.env.part_position(recovery.station).copy()
        self.start_R = self.env.data.xmat[self.view._object_body_id].reshape(3, 3).copy()
        # No reorientation at all. The block is lifted straight up off a flat
        # rest, so it is already flat; tilt_deg records that rather than assuming
        # it. Commanding a "correct" orientation instead is what broke the two
        # previous attempts: driving to identity asked for 105.8 deg of wrist
        # turn, the IK saturated, grip force climbed 4 -> 36 N as the arms fought
        # each other, the descent stalled 82 mm above the belt and the robot went
        # over. Extracting a yaw to level it was worse still (169.5 deg): a box
        # is 90 deg rotationally symmetric, so "angle from identity" is not a
        # thing the arms should be asked to minimise.
        self.place_R = self.start_R
        self.commanded_rotation_deg = 0.0
        self.tilt_deg = float(np.degrees(np.arccos(
            np.clip(np.max(np.abs(self.start_R[2, :])), -1., 1.))))
        self.offset = {}
        self.hand_R = {}
        for side in ('left', 'right'):
            p, R = self.view.palm_pose(side)
            self.offset[side] = self.start_R.T @ (p - self.start)
            self.hand_R[side] = self.start_R.T @ R
        self.unsupported_ticks = 0
        self.final_error_m = float('inf')
        self.hands_clear = False
        self.square_tries = 0
        self.square_errors_deg = []
        self.slide_press_m = 0.
        self.squeeze_open_m = 0.
        self.peak_force_n = 0.

    def _yaw_error(self):
        """Signed block yaw off the nearest quarter turn (a cube is 90 deg symmetric)."""
        R = self.env.data.xmat[self.view._object_body_id].reshape(3, 3)
        up = int(np.argmax(np.abs(R[2]))); side = (up + 1) % 3
        yaw = np.arctan2(R[1, side], R[0, side])
        return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)

    def _begin_release(self):
        self.release_start = {s: self.view.palm_pose(s)[0].copy() for s in self.offset}
        self.release_R = {s: self.view.palm_pose(s)[1].copy() for s in self.offset}
        axis = self.release_start['right'] - self.release_start['left']
        axis[2] = 0.
        self.release_axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        self.release_extra_m, self.release_free = 0., False
        self._advance('RELEASE')

    def _hands_free(self):
        forces = SharpaContactLift.measure_support_forces(self.view)
        return all(np.linalg.norm(f) < self.config.release_free_n for f in forces.values())

    def _begin_square(self):
        error = self._yaw_error()
        stalled = (len(self.square_errors_deg) >= 2 and abs(self.square_errors_deg[-1])
                   > abs(self.square_errors_deg[-2]) - self.config.square_min_gain_deg)
        if (not self.config.square_yaw or abs(error) < self.config.square_tolerance_rad
                or self.square_tries >= self.config.square_tries or stalled):
            self._begin_release()
            return
        gain = self.config.square_gain if self.square_tries == 0 else self.config.square_later_gain
        self.square_tries += 1
        self.square_from, self.square_error = self.place_R.copy(), gain * error
        # Turn about the block's MEASURED vertical axis. Turning about the
        # commanded centre, with the block hanging centimetres off it in the
        # hands, made the arms fight (45-60 N), tilt it 15-20 deg and push it
        # 14 -> 37 mm off the target before the release.
        actual = self.env.part_position(self.recovery.station)
        self.square_centre = np.r_[actual[:2], 0.]
        self.square_targets = {s: t.copy() for s, t in self.last_targets.items()}
        self.square_rotations = {s: R.copy() for s, R in self.last_rotations.items()}
        self._advance('SQUARE')

    def _advance(self, stage):
        self.stage, self.tick = stage, 0

    def _move(self, pos, R):
        targets = {s: pos + R @ self.offset[s] for s in self.offset}
        rotations = {s: R @ self.hand_R[s] for s in self.hand_R}
        self._solve(targets, rotations)

    def _solve(self, targets, rotations):
        self.last_targets, self.last_rotations = targets, rotations
        if (self.config.squeeze_band_n and self.stage in ('SLIDE', 'SQUARE')
                and self._object_supported_by_belt()):
            forces = SharpaContactLift.measure_support_forces(self.view)
            mean = float(np.mean([np.linalg.norm(f) for f in forces.values()]))
            self.peak_force_n = max(self.peak_force_n, mean)
            low, high = self.config.squeeze_band_n
            rate = self.config.squeeze_rate_m
            if mean > high:
                self.squeeze_open_m = min(self.squeeze_open_m + rate, self.config.squeeze_open_max_m)
            elif mean < low:
                self.squeeze_open_m = max(self.squeeze_open_m - rate, -self.config.squeeze_close_max_m)
        if self.squeeze_open_m:
            axis = targets['right'] - targets['left']
            axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
            targets = {'left': targets['left'] - axis * self.squeeze_open_m,
                       'right': targets['right'] + axis * self.squeeze_open_m}
        result = self.expert._solve_both(
            targets, rotations, require_orientation=True, ori_task_weight=1.,
            rest_q=self.expert._current_rest_q(), rest_gain=0.05, pos_tol=0.001,
            joint_weight=self.expert.DESCEND_JOINT_WEIGHT)
        self.expert._apply_ik_result(result)

    def _object_supported_by_belt(self):
        return self.view._object_body_id in self.env.belt.parts_on_belt(self.env)

    def _hands_clear(self):
        # Actual geometry distances, not commanded hand positions.
        hands = SharpaContactLift.hand_wrist_body_ids(self.view)
        hand_ids = hands['left'] | hands['right']
        object_geom = self.env.model.geom(fc.part_geom_name(self.recovery.station)).id
        arm_prefix = fc.arm_body_name(self.recovery.station, '')
        obstacles = [object_geom] + [i for i in range(self.env.model.ngeom)
                    if (self.env.model.geom(i).name or '').startswith(arm_prefix)]
        min_distance = 1.
        for geom in range(self.env.model.ngeom):
            if int(self.env.model.geom_bodyid[geom]) in hand_ids:
                for obstacle in obstacles:
                    min_distance = min(min_distance, float(mujoco.mj_geomDistance(
                        self.env.model, self.env.data, geom, obstacle, 1., None)))
        return min_distance > 0.04

    def step(self):
        cfg = self.config
        action = np.zeros(25)
        self.tick += 1
        actual = self.env.part_position(self.recovery.station)
        if self.stage == 'DESCEND':
            # Straight down at the current XY: get the belt under the block
            # before asking the arms to move it anywhere.
            f = _quintic_scale(self.tick / cfg.descend_ticks)
            floor = np.r_[self.start[:2], self.target[2]]
            self._move((1-f)*self.start + f*floor, self.place_R)
            self.final_error_m = float(np.linalg.norm(actual - self.target))
            if self.tick >= cfg.descend_ticks:
                if self._object_supported_by_belt():
                    self.slide_start = actual.copy()
                    self._advance('SLIDE')
                elif self.tick >= cfg.descend_ticks + cfg.settle_ticks:
                    self.failure = 'PLACE_SUPPORT_NOT_ESTABLISHED'
        elif self.stage == 'SLIDE':
            f = _quintic_scale(min(1., self.tick / cfg.slide_ticks))
            self.final_error_m = float(np.linalg.norm(actual - self.target))
            supported = self._object_supported_by_belt()
            if self.tick >= cfg.slide_ticks and self.final_error_m < cfg.position_tolerance_m and not supported:
                self.slide_press_m = min(self.slide_press_m + cfg.slide_press_rate_m, cfg.slide_press_max_m)
            self._move((1-f)*self.slide_start + f*self.target - np.r_[0., 0., self.slide_press_m], self.place_R)
            if self.tick >= cfg.slide_ticks:
                if self.final_error_m < cfg.position_tolerance_m and supported:
                    if cfg.reseat:
                        self.reseat_targets = {s: t.copy() for s, t in self.last_targets.items()}
                        self.reseat_rotations = {s: R.copy() for s, R in self.last_rotations.items()}
                        axis = self.reseat_targets['right'] - self.reseat_targets['left']; axis[2] = 0.
                        self.reseat_axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
                        self._advance('RESEAT')
                    else:
                        self.square_errors_deg.append(round(float(np.degrees(self._yaw_error())), 2))
                        self._begin_square()
                elif self.tick >= cfg.slide_ticks + cfg.settle_ticks:
                    self.failure = 'PLACE_SLIDE_NOT_REACHED'
        elif self.stage == 'RESEAT':
            # open (reseat_ticks), hold open (settle), close back (reseat_ticks)
            t = self.tick
            if t <= cfg.reseat_ticks:
                w = _quintic_scale(t / cfg.reseat_ticks)
            elif t <= cfg.reseat_ticks + cfg.reseat_settle_ticks:
                w = 1.
            else:
                w = 1. - _quintic_scale(min(1., (t - cfg.reseat_ticks - cfg.reseat_settle_ticks) / cfg.reseat_ticks))
            o = cfg.reseat_open_m * w
            self._solve({'left': self.reseat_targets['left'] - self.reseat_axis * o,
                         'right': self.reseat_targets['right'] + self.reseat_axis * o}, self.reseat_rotations)
            self.final_error_m = float(np.linalg.norm(actual - self.target))
            if t >= 2 * cfg.reseat_ticks + cfg.reseat_settle_ticks + 20:
                self.last_targets, self.last_rotations = self.reseat_targets, self.reseat_rotations
                self.square_errors_deg.append(round(float(np.degrees(self._yaw_error())), 2))
                self._begin_square()
        elif self.stage == 'SQUARE':
            # Turn the commanded block frame about vertical through the block
            # centre; the hands follow it and so does the belt-supported block.
            f = _quintic_scale(min(1., self.tick / cfg.square_ticks))
            c, s_ = np.cos(-self.square_error * f), np.sin(-self.square_error * f)
            Rz = np.array([[c, -s_, 0.], [s_, c, 0.], [0., 0., 1.]])
            self.place_R = Rz @ self.square_from
            k = self.square_centre
            self._solve({s: k + Rz @ (t - k) for s, t in self.square_targets.items()},
                        {s: Rz @ R for s, R in self.square_rotations.items()})
            self.final_error_m = float(np.linalg.norm(actual - self.target))
            self.commanded_rotation_deg = float(np.degrees(np.arccos(np.clip(
                (np.trace(self.start_R.T @ self.place_R) - 1) / 2, -1., 1.))))
            if self.tick >= cfg.square_ticks + cfg.square_settle_ticks:
                self.square_errors_deg.append(round(float(np.degrees(self._yaw_error())), 2))
                self._begin_square()
        elif self.stage == 'READY_TO_VERIFY':
            # Keep holding the retracted pose through verification and the
            # restart: left uncommanded, the palms drifted 5 cm back toward the
            # block in 22 ticks and failed the hands-clear check.
            action[17:25] = -1.
            self._solve(self.last_targets, self.last_rotations)
        elif self.stage in ('RELEASE', 'RETRACT'):
            action[17:25] = -1.
            delay = cfg.release_finger_ticks if cfg.release_order == 'fingers_first' else 0
            if cfg.release_order == 'spread_first' and self.stage == 'RELEASE':
                action[17:25] = 0.
            opening = cfg.release_open_m * _quintic_scale(min(1., max(0, self.tick - delay) / cfg.release_ticks))
            if self.stage == 'RELEASE' and self.tick >= cfg.release_ticks + delay and not self._hands_free():
                self.release_extra_m = min(self.release_extra_m + cfg.release_open_rate_m,
                                           cfg.release_open_max_m - cfg.release_open_m)
            if self.stage == 'RETRACT':
                opening = cfg.release_open_m
            spread = {'left': -self.release_axis * (opening + self.release_extra_m),
                      'right': self.release_axis * (opening + self.release_extra_m)}
            up = back = 0.
            if self.stage == 'RETRACT':
                t, a = min(1., self.tick / cfg.retract_ticks), cfg.retract_lift_fraction
                up = _quintic_scale(min(1., t / a)); back = _quintic_scale(max(0., (t - a) / (1. - a)))
            lift = self.expert.task_rotation @ np.array([-cfg.retract_back_m*back, 0.,
                                                         cfg.retract_up_m*up - cfg.retract_down_m*back])
            self._solve({s: self.release_start[s] + spread[s] + lift for s in self.offset}, self.release_R)
            if self.stage == 'RELEASE' and self.tick >= cfg.release_ticks + delay:
                self.release_free = self._hands_free()
                if self.release_free or self.tick >= cfg.release_ticks + delay + cfg.release_free_wait_ticks:
                    self._advance('RETRACT')
            elif self.stage == 'RETRACT' and self.tick >= cfg.retract_ticks:
                self.hands_clear = self._hands_clear()
                if self.hands_clear and self.env.task_manager._part_is_back(self.env):
                    self._advance('READY_TO_VERIFY')
                elif self.tick >= cfg.retract_ticks + cfg.settle_ticks:
                    self.failure = 'PLACE_RETRACT_OR_SETTLE_NOT_ACHIEVED'
        if self.stage in ('DESCEND', 'SLIDE', 'SQUARE', 'RESEAT'):
            forces = SharpaContactLift.measure_support_forces(self.view)
            supported = all(np.linalg.norm(f) > 0.5 for f in forces.values())
            self.unsupported_ticks = 0 if supported or self._object_supported_by_belt() else self.unsupported_ticks + 1
            if self.unsupported_ticks * self.env.config.frame_skip * self.env.model.opt.timestep > 0.5:
                self.failure = 'PLACE_CONTACT_LOST'
        action[:3] = self.expert._waist_action_toward_target()
        action[3:17] = self.expert._arm_action_toward_target()
        return action
