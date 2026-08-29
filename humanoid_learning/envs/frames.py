"""Base(pelvis)-relative coordinate transforms for the whole-body env.

Convention (documented, not left implicit -- see PROJECT_CONTEXT.md Phase 4,
Section G): transforms are **yaw-only**. A world-frame vector is expressed
relative to the base by translating to the pelvis origin and rotating by
-yaw about the world z-axis. Pelvis roll/pitch are NOT baked into this
rotation -- they are exposed separately as raw observation features
instead.

Why yaw-only and not full 3D orientation: this project's frame is meant to
normalize the robot's heading (so left/right target positions read the same
regardless of which way the robot is facing), not its tilt. A full
roll+pitch+yaw-relative transform would rotate the world's "down" direction
out of the frame whenever the robot leans, making gravity/tilt information
implicit and harder for a policy to use directly; yaw-only keeps "up" fixed
while still removing the arbitrary heading dependence. This matches the
common base-yaw-frame convention used in legged-robot observation design.
"""

from __future__ import annotations

import numpy as np


def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    """Extracts the yaw (rotation about world z) from a wxyz quaternion,
    ignoring roll/pitch -- i.e. the heading of the body's own x-axis
    projected onto the world xy-plane."""
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def roll_pitch_from_quat(quat_wxyz: np.ndarray) -> tuple[float, float]:
    """Roll (rotation about x) and pitch (about y), independent of yaw --
    used as raw tilt features alongside the yaw-only position/velocity
    transforms (see module docstring)."""
    w, x, y, z = quat_wxyz
    roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def rotate_z(vec_xy_or_xyz: np.ndarray, angle: float) -> np.ndarray:
    """Rotates the xy components of a 2D or 3D vector by ``angle`` about z;
    a z component (if present) passes through unchanged."""
    v = np.asarray(vec_xy_or_xyz, dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    x, y = v[0], v[1]
    out = v.copy()
    out[0] = c * x - s * y
    out[1] = s * x + c * y
    return out


def world_to_base(world_pos: np.ndarray, base_pos: np.ndarray, base_yaw: float) -> np.ndarray:
    """Translates by -base_pos then rotates by -base_yaw. Works for 2D (xy)
    or 3D (xyz) position vectors; z (if present) is translated but not
    rotated (yaw-only convention, see module docstring)."""
    world_pos = np.asarray(world_pos, dtype=np.float64)
    base_pos = np.asarray(base_pos, dtype=np.float64)
    delta = world_pos - base_pos[: world_pos.shape[0]]
    return rotate_z(delta, -base_yaw)


def base_to_world(base_pos_local: np.ndarray, base_pos: np.ndarray, base_yaw: float) -> np.ndarray:
    """Inverse of world_to_base: rotates by +base_yaw then translates by
    +base_pos."""
    base_pos_local = np.asarray(base_pos_local, dtype=np.float64)
    base_pos = np.asarray(base_pos, dtype=np.float64)
    rotated = rotate_z(base_pos_local, base_yaw)
    return rotated + base_pos[: rotated.shape[0]]


def world_vel_to_base(world_vel: np.ndarray, base_yaw: float) -> np.ndarray:
    """Rotates a world-frame linear/angular velocity vector into the base
    yaw-frame (no translation -- velocities are frame-relative, not
    position-relative)."""
    return rotate_z(np.asarray(world_vel, dtype=np.float64), -base_yaw)
