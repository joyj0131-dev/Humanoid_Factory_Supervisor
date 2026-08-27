"""Damped least squares (DLS) inverse kinematics -- pure linear algebra.

No MuJoCo dependency here on purpose: this module only knows about
Jacobians and position errors, so it is trivial to unit test and reuse for
either arm. Jacobian computation (a MuJoCo-specific operation) lives on
BimanualReachEnv.arm_jacobians(), not here.
"""

from __future__ import annotations

import numpy as np


def dls_step(jacobian: np.ndarray, position_error: np.ndarray, damping: float) -> np.ndarray:
    """One damped-least-squares joint-space step toward a position target.

    delta_q = J^T (J J^T + damping^2 I)^-1 * position_error

    This is the standard Wampler/Nakamura DLS formula. Compared to the
    plain Jacobian pseudo-inverse, the damping term keeps the step bounded
    near kinematic singularities instead of producing huge joint deltas.

    Args:
        jacobian: (3, n_joints) position Jacobian.
        position_error: (3,) target - current end-effector position.
        damping: damping factor (lambda). Larger = more robust near
            singularities but slower convergence.

    Returns:
        (n_joints,) joint-space delta.
    """
    jjt = jacobian @ jacobian.T
    damped = jjt + (damping**2) * np.eye(jjt.shape[0])
    return jacobian.T @ np.linalg.solve(damped, position_error)
