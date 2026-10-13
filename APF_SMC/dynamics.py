"""
Rigid-body attitude dynamics and RK4 propagation (port of RK4_rotation.m).

State x = [q0, q1, q2, q3, wx, wy, wz]  (scalar-first quaternion, rad/s)
"""

import numpy as np

from apf_smc_controller import qmult, pure, skew


def diff_rotation(x: np.ndarray, I: np.ndarray, invI: np.ndarray,
                  u: np.ndarray) -> np.ndarray:
    """State derivative: q_dot = 0.5 q ⊗ [0, w],  I w_dot = -w x I w + u."""
    q = x[:4]
    w = x[4:]
    f1 = 0.5 * qmult(q, pure(w))
    f2 = invI @ (-skew(w) @ I @ w + u)
    return np.concatenate((f1, f2))


def RK4_rotation(x0: np.ndarray, I: np.ndarray, invI: np.ndarray,
                 u: np.ndarray, dt: float) -> np.ndarray:
    """One RK4 step with the torque u held constant; quaternion renormalised."""
    k1 = dt * diff_rotation(x0, I, invI, u)
    k2 = dt * diff_rotation(x0 + 0.5 * k1, I, invI, u)
    k3 = dt * diff_rotation(x0 + 0.5 * k2, I, invI, u)
    k4 = dt * diff_rotation(x0 + k3, I, invI, u)
    x_dt = x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    x_dt[:4] /= np.linalg.norm(x_dt[:4])
    return x_dt
