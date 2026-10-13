"""
APF-based sliding mode controller for constrained attitude reorientation.

Standalone module: depends only on NumPy. It contains
  * quaternion utilities (scalar-first convention, q = [q0, q1, q2, q3])
  * constraint-matrix construction for forbidden / mandatory cones
  * the log-barrier potential function and its gradient / time derivative
  * the sliding mode control law
  * an ``APFSMCController`` class wrapping everything, which stores the
    anti-unwinding sign (formerly the MATLAB global ``sign_qe0``) as state

Python port of: APF_SMC_controller.m, cal_V_related.m, calM.m, Lf.m, Lm.m,
dLdq.m, ddLdq2.m, qmult.m, qconj.m, qvec.m, sat.m, sgn.m, sigmoid.m
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Quaternion utilities (scalar first)
# ----------------------------------------------------------------------------


def qmult(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Quaternion product q ⊗ r.

    Accepts shape (4,) or (4, N); broadcasting works as in the MATLAB version
    (a single quaternion times an array of quaternions, or column-wise).
    """
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    return np.stack([
        q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3],
        q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2],
        q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1],
        q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0],
    ])


def qconj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate. Shape (4,) or (4, N)."""
    p = np.array(q, dtype=float, copy=True)
    p[1:] = -p[1:]
    return p


def qvec(q: np.ndarray) -> np.ndarray:
    """Vector part of a quaternion."""
    return np.asarray(q)[1:4]


def pure(w: np.ndarray) -> np.ndarray:
    """Embed a 3-vector as a pure quaternion [0, w]."""
    return np.concatenate(([0.0], np.asarray(w, dtype=float)))


def skew(w: np.ndarray) -> np.ndarray:
    """Cross-product matrix: skew(w) @ v == np.cross(w, v)."""
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


# ----------------------------------------------------------------------------
# Switching functions
# ----------------------------------------------------------------------------


def sgn(x: float) -> float:
    """Sign function with sgn(0) = 1."""
    return 1.0 if x == 0 else float(np.sign(x))


def sat(x: np.ndarray, epsilon: float) -> np.ndarray:
    """Saturation with boundary layer epsilon (element-wise)."""
    return np.clip(np.asarray(x, dtype=float) / epsilon, -1.0, 1.0)


def sigmoid(x: np.ndarray, delta: float) -> np.ndarray:
    """Smooth sign approximation x / (|x| + delta) (element-wise)."""
    x = np.asarray(x, dtype=float)
    return x / (np.abs(x) + delta)


# ----------------------------------------------------------------------------
# Attitude constraints
# ----------------------------------------------------------------------------


def calM(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Constraint matrix for a cone.

    x     : unit vector to avoid / observe, inertial frame (3,)
    y     : instrument boresight unit vector, body frame (3,)
    alpha : cone half angle [rad]

    q.T @ M @ q = x·(C(q) y) - cos(alpha), with C(q) body-to-inertial.
    Forbidden zone satisfied if q.T M q < 0; mandatory zone if > 0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.cross(y, x)
    xy = x @ y
    M = np.empty((4, 4))
    M[0, 0] = xy - np.cos(alpha)
    M[0, 1:] = c
    M[1:, 0] = c
    M[1:, 1:] = np.outer(x, y) + np.outer(y, x) - (xy + np.cos(alpha)) * np.eye(3)
    return M


def constraint_index(M: np.ndarray, q: np.ndarray) -> float:
    """Return q.T @ M @ q."""
    return float(q @ M @ q)


# ----------------------------------------------------------------------------
# Log-barrier terms and derivatives
# ----------------------------------------------------------------------------


def Lf(Mf: np.ndarray, q: np.ndarray, kf: float = 1.0) -> float:
    """Barrier term for a forbidden zone."""
    result = -(q @ Mf @ q) / 2.0

    # Avoid nan due to negative or zero log argument.
    if result <= 0:
        result = 1e-9

    return -kf * np.log(result)


def Lm(Mm: np.ndarray, q: np.ndarray, km: float = 1.0) -> float:
    """Barrier term for a mandatory zone."""
    return -km * np.log((q @ Mm @ q) / 2.0)


def dLdq(M: np.ndarray, q: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Gradient of Lf / Lm w.r.t. q, shape (4,). Identical for both."""
    Mq = M @ q
    return -2.0 * k / (q @ Mq) * Mq


def ddLdq2(M: np.ndarray, q: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Hessian of Lf / Lm w.r.t. q, shape (4, 4). Identical for both."""
    Mq = M @ q
    qMq = q @ Mq
    return 4.0 * k / qMq**2 * np.outer(Mq, Mq) - 2.0 * k / qMq * M


# ----------------------------------------------------------------------------
# Anti-unwinding sign (replaces the MATLAB global variable sign_qe0)
# ----------------------------------------------------------------------------


def compute_sign_qe0(q0: np.ndarray, qd: np.ndarray) -> float:
    """Sign of the scalar part of qe = qd* ⊗ q0 (with sgn(0) = 1).

    Selects which of the two antipodal targets (+qd or -qd) is closer, so the
    spacecraft does not rotate the long way round (unwinding).
    """
    qe = qmult(qconj(qd), q0)
    return sgn(qe[0])


# ----------------------------------------------------------------------------
# Potential function
# ----------------------------------------------------------------------------


def _as_stack(M: Optional[np.ndarray]) -> np.ndarray:
    """Normalise a constraint set to shape (n, 4, 4) (n may be 0)."""
    if M is None:
        return np.zeros((0, 4, 4))
    M = np.asarray(M, dtype=float)
    if M.ndim == 2:
        M = M[None, :, :]
    return M


def cal_V_related(q: np.ndarray, qd: np.ndarray, w: np.ndarray,
                  Mf: Optional[np.ndarray], Mm: Optional[np.ndarray],
                  beta1: float, beta2: float, sign_qe0: float,
                  kf: float = 1.0, km: float = 1.0
                  ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Potential V, its gradient and the time derivative of the gradient.

    V = beta1*(2 - 2 s qd·q) + beta2*(2 - 2 s qd·q)*(sum Lf + sum Lm)

    q, qd    : current / desired attitude quaternions (4,)
    w        : angular rate [rad/s] (3,)
    Mf, Mm   : forbidden / mandatory constraint matrices, shape (n, 4, 4),
               (4, 4) for a single cone, or None / empty for no cones
    sign_qe0 : anti-unwinding sign, s in the formula (see compute_sign_qe0)

    Returns (V, grad_V (4,), dot_grad_V (4,)).
    """
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    Mf = _as_stack(Mf)
    Mm = _as_stack(Mm)

    L_sum = 0.0
    dL_sum = np.zeros(4)
    ddL_sum = np.zeros((4, 4))
    for M in Mf:
        L_sum += Lf(M, q, kf)
        dL_sum += dLdq(M, q, kf)
        ddL_sum += ddLdq2(M, q, kf)
    for M in Mm:
        L_sum += Lm(M, q, km)
        dL_sum += dLdq(M, q, km)
        ddL_sum += ddLdq2(M, q, km)


    s = sign_qe0
    att = 2.0 - 2.0 * s * (qd @ q)      # attractive term, = ||q - s*qd||^2

    V = beta1 * att + beta2 * att * L_sum
    grad_V = (-2.0 * s * beta1 * qd
              - 2.0 * s * beta2 * qd * L_sum
              + att * beta2 * dL_sum)
    hess_V = (-2.0 * s * beta2 * (np.outer(qd, dL_sum) + np.outer(dL_sum, qd))
              + att * beta2 * ddL_sum)
    q_dot = 0.5 * qmult(q, pure(w))
    dot_grad_V = hess_V @ q_dot

    #print(f"L_sum={L_sum}, dL_sum={dL_sum}, ddL_sum={ddL_sum}, V={V}, grad_V={grad_V}, hess_V={hess_V}, dot_grad_V={dot_grad_V}")
    return float(V), grad_V, dot_grad_V


# ----------------------------------------------------------------------------
# Sliding mode control law
# ----------------------------------------------------------------------------


def APF_SMC_controller(q: np.ndarray, w: np.ndarray, I: np.ndarray,
                       grad_V: np.ndarray, dot_grad_V: np.ndarray,
                       alpha: float, sigma: float, k: float,
                       epsilon: float = 1e-4, switching: str = "sat",
                       delta: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """APF-based sliding mode control torque.

    s = w + alpha * vec(q* ⊗ grad_V)
    u = w x (I w) - alpha I vec(d/dt(q* ⊗ grad_V)) - sigma*sat(s) - k*s

    switching : "sat" (boundary layer epsilon) or "sigmoid" (parameter delta)

    Returns (u (3,), s (3,)).
    """
    q = np.asarray(q, dtype=float)
    w = np.asarray(w, dtype=float)
    s = w + alpha * qvec(qmult(qconj(q), grad_V))
    dot_term = (-0.5 * qmult(qmult(pure(w), qconj(q)), grad_V)
                + qmult(qconj(q), dot_grad_V))

    if switching == "sat":
        sw = sat(s, epsilon)
    elif switching == "sigmoid":
        sw = sigmoid(s, delta)
    else:
        raise ValueError(f"unknown switching function '{switching}'")

    u = skew(w) @ I @ w - alpha * I @ qvec(dot_term) - sigma * sw - k * s
    return u, s


# ----------------------------------------------------------------------------
# Convenience class
# ----------------------------------------------------------------------------


@dataclass
class APFSMCParams:
    """Tuning parameters. Defaults are those of scenario F4."""
    beta2: float = 0.01          # repulsive potential weight
    beta1: float = 0.05          # attractive potential weight
    kf: float = 1.0              # forbidden-zone barrier coefficient
    km: float = 1.0              # mandatory-zone barrier coefficient
    alpha: float = 2.0           # sliding surface gain
    sigma: float = 0.35          # switching gain (> disturbance bound)
    k: float = 4.0               # linear reaching gain
    epsilon: float = 1e-4        # boundary layer for sat()
    switching: str = "sat"       # "sat" or "sigmoid"
    delta: float = 1e-4          # parameter for sigmoid()


@dataclass
class APFSMCController:
    """APF-SMC attitude controller with stored anti-unwinding sign.

    Example
    -------
    >>> ctrl = APFSMCController(I, qd, Mf=Mf, params=APFSMCParams())
    >>> ctrl.reset(q0)                 # fixes sign_qe0 from the initial state
    >>> u, info = ctrl.compute(q, w)   # call every control step
    """
    I: np.ndarray
    qd: np.ndarray
    Mf: Optional[np.ndarray] = None
    Mm: Optional[np.ndarray] = None
    params: APFSMCParams = field(default_factory=APFSMCParams)
    sign_qe0: Optional[float] = None

    def __post_init__(self):
        self.I = np.asarray(self.I, dtype=float)
        self.qd = np.asarray(self.qd, dtype=float)
        self.qd = self.qd / np.linalg.norm(self.qd)
        self.Mf = _as_stack(self.Mf)
        self.Mm = _as_stack(self.Mm)

    @classmethod
    def from_cones(cls, I, qd, forbidden: Sequence = (), mandatory: Sequence = (),
                   params: Optional[APFSMCParams] = None) -> "APFSMCController":
        """Build from cone definitions, each a tuple (x_inertial, y_body, half_angle_rad)."""
        Mf = np.array([calM(x, y, a) for x, y, a in forbidden]).reshape(-1, 4, 4)
        Mm = np.array([calM(x, y, a) for x, y, a in mandatory]).reshape(-1, 4, 4)
        return cls(I=I, qd=qd, Mf=Mf, Mm=Mm, params=params or APFSMCParams())

    def reset(self, q0: np.ndarray, sign_qe0: Optional[float] = None) -> float:
        """Fix the anti-unwinding sign for a new manoeuvre.

        By default it is computed from q0; pass sign_qe0 = ±1 to force it
        (e.g. to test the long-way rotation).
        """
        self.sign_qe0 = compute_sign_qe0(q0, self.qd) if sign_qe0 is None else float(sign_qe0)
        return self.sign_qe0

    def set_target(self, qd: np.ndarray, q_current: np.ndarray) -> None:
        """Change the desired attitude and recompute the sign."""
        self.qd = np.asarray(qd, dtype=float) / np.linalg.norm(qd)
        self.reset(q_current)

    def constraint_indices(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """q.T M q for every forbidden / mandatory cone."""
        f = np.array([q @ M @ q for M in self.Mf])
        m = np.array([q @ M @ q for M in self.Mm])
        return f, m

    def compute(self, q: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Control torque for the current state.

        Returns (u, info) where info holds s, V, grad_V and dot_grad_V.
        """
        if self.sign_qe0 is None:
            raise RuntimeError("call reset(q0) before compute() to set sign_qe0")
        p = self.params
        V, grad_V, dot_grad_V = cal_V_related(
            q, self.qd, w, self.Mf, self.Mm, p.beta1, p.beta2,
            self.sign_qe0, p.kf, p.km)
        u, s = APF_SMC_controller(
            q, w, self.I, grad_V, dot_grad_V, p.alpha, p.sigma, p.k,
            p.epsilon, p.switching, p.delta)
        return u, {"s": s, "V": V, "grad_V": grad_V, "dot_grad_V": dot_grad_V}
