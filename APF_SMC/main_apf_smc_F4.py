"""
Simulation of "APF-based sliding mode control for constrained reorientation",
new potential function, scenario F4 (four forbidden zones).

Port of Main_APF_based_SMC_reorientation_new_F4.m.
Results are saved to data_APF_SMC_new_paper_F4.npz (and .mat if SciPy exists).
"""

import numpy as np

from apf_smc_controller import (APFSMCController, APFSMCParams, calM,
                                qconj, qmult)
from dynamics import RK4_rotation


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def run_simulation(tend=150.0, dt=0.01, params=None, sign_qe0=None,
                   disturbance=True, verbose=True):
    d2r = np.pi / 180.0
    n = int(round(tend / dt))
    t = np.arange(n + 1) * dt

    I = np.diag([60.0, 50.0, 70.0])           # inertia [kg m^2]
    invI = np.linalg.inv(I)

    q0 = unit([0.6026, -0.73, 0.2, 0.253])     # initial attitude
    w0 = np.array([0.0, 0.0, 0.0]) * d2r       # initial rate [rad/s]
    qd = unit([1.0, 0.05, 0.05, 0.0])          # desired attitude

    # ---------------- forbidden zones ----------------
    y_body = np.array([0.0, 0.0, 1.0])          # instrument boresight, body frame
    forbidden = [
        (unit([0.253, -0.37, -0.894]), y_body, 25 * d2r),   # F1
        (unit([0.0, 0.707, 0.707]),    y_body, 40 * d2r),   # F2
        (unit([-0.853, 0.436, -0.286]), y_body, 30 * d2r),  # F3
        (unit([0.957, -0.256, 0.131]), y_body, 20 * d2r),   # F4
    ]
    mandatory = []                              # no mandatory zones

    Mf = np.array([calM(x, y, a) for x, y, a in forbidden]).reshape(-1, 4, 4)
    Mm = np.array([calM(x, y, a) for x, y, a in mandatory]).reshape(-1, 4, 4)
    n_cone_f, n_cone_m = len(Mf), len(Mm)

    # ---------------- controller ----------------
    if params is None:
        params = APFSMCParams(beta2=0.005 * 2, beta1=5 * 0.005 * 2,
                              alpha=2.0, sigma=0.15 + 0.2, k=4.0,
                              epsilon=1e-4)
    ctrl = APFSMCController(I=I, qd=qd, Mf=Mf, Mm=Mm, params=params)
    sgn0 = ctrl.reset(q0, sign_qe0=sign_qe0)   # replaces the global sign_qe0

    # ---------------- disturbance ----------------
    omega_taud = 0.01
    d_inb = np.vstack([-0.02 + 0.03 * np.sin(omega_taud * t),
                       0.05 + 0.04 * np.sin(omega_taud * t),
                       0.04 - 0.07 * np.sin(omega_taud * t)])   # [N m]
    if not disturbance:
        d_inb[:] = 0.0

    # ---------------- storage ----------------
    x_array = np.zeros((7, n + 1))
    x_array[:, 0] = np.concatenate((q0, w0))
    tau_array = np.zeros((3, n + 1))
    s_array = np.zeros((3, n + 1))
    Vp_array = np.zeros(n + 1)

    # ---------------- main loop ----------------
    for i in range(n + 1):
        q = x_array[:4, i]
        w = x_array[4:, i]
        u, info = ctrl.compute(q, w)
        tau_array[:, i] = u
        s_array[:, i] = info["s"]
        Vp_array[i] = info["V"]
        if i < n:
            x_array[:, i + 1] = RK4_rotation(x_array[:, i], I, invI,
                                             u + d_inb[:, i], dt)

    # ---------------- post-processing ----------------
    q_array = x_array[:4, :]
    w_array = x_array[4:, :]
    qe_array = qmult(qconj(qd), q_array)
    index_F_array = np.einsum("in,kij,jn->kn", q_array, Mf, q_array)
    index_M_array = np.einsum("in,kij,jn->kn", q_array, Mm, q_array)
    energy_cost = float(np.sum(tau_array**2) * dt)

    results = dict(t=t, dt=dt, x_array=x_array, q_array=q_array,
                   w_array=w_array, qe_array=qe_array, tau_array=tau_array,
                   s_array=s_array, Vp_array=Vp_array,
                   index_F_array=index_F_array, index_M_array=index_M_array,
                   energy_cost=energy_cost, sign_qe0=sgn0,
                   n_cone_f=n_cone_f, n_cone_m=n_cone_m, Mf=Mf, Mm=Mm,
                   q0=q0, qd=qd, I=I, d_inb_array=d_inb)

    if verbose:
        err = np.degrees(2 * np.arccos(np.clip(np.abs(qe_array[0]), 0, 1)))
        print(f"sign_qe0            : {sgn0:+.0f}")
        print(f"final attitude error: {err[-1]:.4f} deg")
        print(f"energy cost         : {energy_cost:.3f}")
        if n_cone_f:
            print("max q'Mf q per cone :", np.round(index_F_array.max(axis=1), 4),
                  "(all must be < 0)")
    return results


def save_results(results, name="data_APF_SMC_new_paper_F4"):
    np.savez(name + ".npz", **results)
    try:
        from scipy.io import savemat
        savemat(name + ".mat", results)       # loadable in MATLAB for comparison
    except ImportError:
        pass


if __name__ == "__main__":
    res = run_simulation()
    save_results(res)
    # To plot:  python plot_result_apf_smc_F4.py
