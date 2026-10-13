"""
Plot results of scenario F4 (port of plot_result_APF_SMC_F4.m).

Usage:  python plot_result_apf_smc_F4.py   (after running main_apf_smc_F4.py)
Figures are written to ./figures (created if missing) in EPS and PNG format.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 15
PLOT_SAVE = True
T_SHOW_END = 120
FIG_DIR = Path("figures")
LINESTYLES = ["-", "--", "-.", ":"]

matplotlib.rcParams.update({"font.size": FONTSIZE, "lines.linewidth": 1.0})


def _save(fig, name):
    if PLOT_SAVE:
        FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(FIG_DIR / f"{name}.eps", format="eps", bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def _stack(t, data, labels, ylims, name, xlabel="Time (sec)", scale=1.0):
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, 2.2 * len(labels)),
                             sharex=True)
    for ax, row, lab, yl in zip(axes, data, labels, ylims):
        ax.plot(t, row * scale, "k-")
        ax.set_ylabel(lab)
        ax.grid(True)
        ax.set_xlim(0, T_SHOW_END)
        if yl is not None:
            ax.set_ylim(*yl)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_results(r):
    t = r["t"]

    _stack(t, r["qe_array"],
           [r"$q_{e0}$", r"$q_{e1}$", r"$q_{e2}$", r"$q_{e3}$"],
           [(0.5, 1.2), None, (-0.2, 0.5), (-0.2, 0.5)],
           "F4_plot_attitude")

    _stack(t, r["w_array"],
           [r"$\omega_x$ (deg/s)", r"$\omega_y$ (deg/s)", r"$\omega_z$ (deg/s)"],
           [(-5, 10), None, (-5, 5)],
           "F4_plot_angular_rate", scale=180 / np.pi)

    _stack(t, r["tau_array"],
           [r"$u_x$ (Nm)", r"$u_y$ (Nm)", r"$u_z$ (Nm)"],
           [None, None, None],
           "F4_plot_control_torque")

    fig, ax = plt.subplots(figsize=(8, 5))
    for j in range(3):
        ax.plot(t, r["s_array"][j], "k" + LINESTYLES[j], label=f"$s_{j + 1}$")
    ax.set_ylabel(r"$\bar{s}$")
    ax.set_xlabel("Time (sec)")
    ax.set_xlim(0, T_SHOW_END)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    _save(fig, "F4_plot_s")

    if int(r["n_cone_f"]) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        for j, row in enumerate(r["index_F_array"]):
            ax.plot(t, row, "k" + LINESTYLES[j % 4], linewidth=1.2,
                    label=f"$F_{j + 1}$")
        ax.set_ylabel(r"$\mathbf{q}^T M_F \mathbf{q}$")
        ax.set_xlabel("Time (sec)")
        ax.set_xlim(0, T_SHOW_END)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        _save(fig, "F4_plot_index_Mf")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, r["Vp_array"], "k-")
    ax.set_ylabel(r"$V_p$")
    ax.set_xlabel("Time (sec)")
    ax.set_xlim(0, T_SHOW_END)
    ax.grid(True)
    fig.tight_layout()
    _save(fig, "F4_plot_Vp")


if __name__ == "__main__":
    data = dict(np.load("data_APF_SMC_new_paper_F4.npz"))
    plot_results(data)
    plt.show()
