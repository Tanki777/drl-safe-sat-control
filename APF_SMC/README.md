# APF-Based Sliding Mode Control for Constrained Attitude Reorientation

Python implementation of an attitude controller that rotates a spacecraft from an initial orientation to a desired one while keeping an onboard instrument (for example, a camera or star tracker) pointed away from **forbidden zones**, such as the Sun, and optionally inside **mandatory zones**, such as a ground-station direction. The controller remains robust to bounded external disturbance torques.

The method combines two ideas:

- An **artificial potential function (APF)** that is low near the target attitude and becomes infinitely high at the edge of every forbidden or mandatory cone. Following the downhill direction of this potential produces a path that reaches the target and stays clear of the cones.
- A **sliding mode controller (SMC)** that forces the spacecraft onto this downhill motion and rejects disturbances.

This code is a port of the original MATLAB implementation and reproduces its results.

---

## Contents

| File                      | Purpose                                                                                                                                                                                                                          |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `apf_smc_controller.py`     | **The controller.** Standalone and depends only on NumPy. Contains the quaternion utilities, cone constraint matrices, potential function and control law. This is the only file you need to use the controller in your own project. |
| `dynamics.py`               | Rigid-body attitude dynamics and a fourth-order Runge–Kutta (RK4) integrator. Used by the example simulation.                                                                                                                    |
| `main_apf_smc_F4.py`        | Example simulation for "scenario F4": a large reorientation with four forbidden zones and a disturbance torque.                                                                                                                  |
| `plot_result_apf_smc_F4.py` | Plots the results of the example simulation.                                                                                                                                                                                     |

---

## Requirements

- Python 3.8 or newer
- NumPy (required)
- Matplotlib (for plotting only)
- SciPy (optional; used only to also save results as a MATLAB `.mat` file)

Install everything with:

```bash
pip install numpy matplotlib scipy
```

---

## Quick start

Run the example simulation, then plot the results:

```bash
python main_apf_smc_F4.py
python plot_result_apf_smc_F4.py
```

The simulation prints a short summary:

```
sign_qe0            : +1
final attitude error: 0.0083 deg
energy cost         : 30.234
max q'Mf q per cone : [-1.1714 -0.0514 -0.287  -0.3058] (all must be < 0)
```

All values in the last line are negative, which confirms that no forbidden zone was entered during the manoeuvre.

The simulation saves its data to `data_APF_SMC_new_paper_F4.npz` (and `.mat` if SciPy is installed). The plotting script writes figures in EPS and PNG format to a `figures/` folder, which it creates automatically. It produces six figures:

| Figure                 | Shows                                                                             |
|------------------------|-----------------------------------------------------------------------------------|
| `F4_plot_attitude`       | Error quaternion between current and desired attitude. Converges to [1, 0, 0, 0]. |
| `F4_plot_angular_rate`   | Angular velocity in deg/s.                                                        |
| `F4_plot_control_torque` | Control torque in N·m.                                                            |
| `F4_plot_s`              | Sliding variable. Converges to zero.                                              |
| `F4_plot_index_Mf`       | Constraint index qᵀM_F q for each forbidden zone. Must stay below zero.           |
| `F4_plot_Vp`             | Potential function value. Decreases to zero.                                      |

---

## Using the controller in your own code

The recommended interface is the `APFSMCController` class.

```python
import numpy as np
from apf_smc_controller import APFSMCController, APFSMCParams

deg = np.pi / 180

I  = np.diag([60.0, 50.0, 70.0])              # inertia matrix [kg m^2]
qd = np.array([1.0, 0.05, 0.05, 0.0])         # desired attitude (normalised automatically)

# Each cone: (direction in inertial frame, boresight in body frame, half-angle [rad])
boresight = np.array([0.0, 0.0, 1.0])
forbidden = [
    (np.array([0.0, 0.707, 0.707]), boresight, 40 * deg),
    (np.array([0.957, -0.256, 0.131]), boresight, 20 * deg),
]
mandatory = []                                 # optional, same format

ctrl = APFSMCController.from_cones(I, qd, forbidden=forbidden,
                                   mandatory=mandatory,
                                   params=APFSMCParams())

ctrl.reset(q0)                  # call once at the start of each manoeuvre

# inside your control loop:
u, info = ctrl.compute(q, w)    # u: control torque [N m], shape (3,)
```

`info` is a dictionary with extra quantities that are useful for logging:

| Key        | Meaning                              |
|------------|--------------------------------------|
| `s`          | Sliding variable (3,)                |
| `V`          | Potential function value (scalar)    |
| `grad_V`     | Gradient of V with respect to q (4,) |
| `dot_grad_V` | Time derivative of the gradient (4,) |

`ctrl.constraint_indices(q)` returns qᵀMq for every forbidden and mandatory cone, which lets you monitor the constraint margins online.

### Lower-level functions

If you prefer to manage the state yourself, the individual building blocks are available as plain functions with the same names as the MATLAB files:

```python
from apf_smc_controller import calM, cal_V_related, APF_SMC_controller, compute_sign_qe0

Mf = np.array([calM(x, y, a) for x, y, a in forbidden])   # shape (n, 4, 4)
sign_qe0 = compute_sign_qe0(q0, qd)

V, grad_V, dot_grad_V = cal_V_related(q, qd, w, Mf, None,
                                      beta1=0.05, beta2=0.01,
                                      sign_qe0=sign_qe0)
u, s = APF_SMC_controller(q, w, I, grad_V, dot_grad_V,
                          alpha=2.0, sigma=0.35, k=4.0, epsilon=1e-4)
```

Pass `None` (or an empty array) for `Mf` or `Mm` when there are no cones of that type. Unlike the class, these functions do not normalise `qd`, so pass a unit quaternion.

---

## Conventions

Getting these wrong is the most common source of errors, so please check them before using the code.

- **Quaternions are scalar-first:** q = [q0, q1, q2, q3], where q0 is the scalar part. They must have unit norm.
- **q describes the rotation from the body frame to the inertial frame.** The kinematics are q̇ = ½ q ⊗ [0, ω].
- **Angular velocity ω** is expressed in the body frame, in rad/s.
- **Cone directions** (the Sun, a target, and so on) are unit vectors in the **inertial** frame. **Instrument boresights** are unit vectors in the **body** frame. `calM` does not normalise them, so make sure both are unit vectors before passing them in.
- **Cone half-angles** are in radians.
- **Torques** are in N·m, inertia in kg·m².
- **Arrays** are 1-D NumPy arrays: quaternions have shape (4,), vectors shape (3,). Constraint matrices are stacked as shape (n, 4, 4).

### Constraint sign convention

For each cone, the matrix M from `calM` satisfies

qᵀ M q = x · (C(q) y) − cos α

where x is the cone direction, y the boresight, α the half-angle and C(q) the rotation matrix. The constraints are met when:

- **forbidden zone:** qᵀ M_F q **< 0** (boresight is more than α away from x)
- **mandatory zone:** qᵀ M_M q **> 0** (boresight is within α of x)

**The initial attitude must satisfy all constraints.** The potential is undefined (logarithm of a negative number) outside the allowed region, and the controller will return `NaN`. Check with `ctrl.constraint_indices(q0)` before starting.

---

## The anti-unwinding sign (`sign_qe0`)

A quaternion q and its negative −q represent the same physical orientation. Without care, a controller can drive the spacecraft toward −q_d when q_d was closer, making it rotate almost 360° instead of a short path ("unwinding").

The code avoids this by computing, once at the start of the manoeuvre,

sign_qe0 = sgn(q_e0),  where q_e = q_d* ⊗ q0

and steering toward `sign_qe0 · q_d`, whichever of the two targets is nearer. By convention sgn(0) = +1.

- `ctrl.reset(q0)` computes and stores the sign. **Call it before the first** `compute()` **call** and at the start of each new manoeuvre; otherwise `compute()` raises an error.
- `ctrl.reset(q0, sign_qe0=-1)` forces a particular sign, for example to study the long-way rotation.
- `ctrl.set_target(qd_new, q_current)` changes the target and recomputes the sign from the current attitude.

The sign stays fixed during the manoeuvre by design.

---

## Parameters

All tuning parameters live in `APFSMCParams`. The defaults are those used in scenario F4.

| Parameter | Default | Meaning                                                                            |
|-----------|---------|------------------------------------------------------------------------------------|
| `beta1`     | 0.05    | Weight of the attractive part of the potential (pulls toward the target).          |
| `beta2`     | 0.01    | Weight of the repulsive part (pushes away from cone boundaries).                   |
| `kf`, `km`    | 1.0     | Barrier coefficients for forbidden and mandatory zones.                            |
| `alpha`     | 2.0     | Sliding surface gain. Larger values give faster motion along the potential.        |
| `sigma`     | 0.35    | Switching gain [N·m]. Must exceed the largest disturbance torque on each axis.     |
| `k`         | 4.0     | Linear reaching gain. Larger values bring the state to the sliding surface faster. |
| `switching` | `"sat"`   | Switching function: `"sat"` (saturation) or `"sigmoid"` (smooth).                      |
| `epsilon`   | 1e-4    | Boundary-layer width for `"sat"`.                                                    |
| `delta`     | 1e-4    | Smoothing parameter for `"sigmoid"`.                                                 |

Example of changing parameters:

```python
params = APFSMCParams(alpha=1.5, sigma=0.5, epsilon=1e-3)
```

### Tuning notes

- **Disturbance rejection:** choose `sigma` larger than the maximum disturbance torque magnitude on each axis. In scenario F4 the disturbance peaks at about 0.11 N·m, so 0.35 gives a comfortable margin.
- **Boundary layer and chattering:** a smaller `epsilon` gives higher accuracy, but if it becomes too small relative to the control update interval, the torque starts to chatter. A useful guideline is

  `epsilon > dt · sigma / (2 · I_min)`

  where `dt` is the control update interval and `I_min` the smallest principal inertia. For scenario F4 this gives about 3.5 × 10⁻⁵, so the default 10⁻⁴ is safe. If you run the controller at a slower rate, increase `epsilon` accordingly.
- **Larger boundary layer:** a larger `epsilon` smooths the torque but leaves a small steady-state error, because disturbances are no longer fully rejected inside the layer. For example, `epsilon = 0.01` in scenario F4 leaves an error of about 0.7°.
- `beta1` **versus** `beta2`**:** if the attractive weight is too small relative to the repulsive weight, the attitude may get trapped in a local minimum of the potential near a cone. Increasing `beta1 / beta2` helps.
- **No constraints:** setting `beta2 = 0` (or passing no cones) turns the controller into an unconstrained sliding mode attitude controller.

---

## Modifying the example scenario

Everything scenario-specific is at the top of `run_simulation()` in `main_apf_smc_F4.py`: inertia, initial and desired attitudes, the list of forbidden and mandatory cones, controller parameters and the disturbance model. `run_simulation()` also accepts arguments for common changes:

```python
from main_apf_smc_F4 import run_simulation, save_results
from apf_smc_controller import APFSMCParams

res = run_simulation(tend=200.0,             # simulation length [s]
                     dt=0.01,                # time step [s]
                     params=APFSMCParams(epsilon=1e-3),
                     sign_qe0=None,          # None = automatic, or +1 / -1
                     disturbance=True)       # False = no disturbance torque
save_results(res, name="my_run")
```

The returned dictionary contains the full time histories (`t`, `q_array`, `w_array`, `qe_array`, `tau_array`, `s_array`, `Vp_array`, `index_F_array`, `index_M_array`) and summary values (`energy_cost`, `sign_qe0`).

To plot a run saved under a different name, change the file name at the bottom of `plot_result_apf_smc_F4.py`, or import `plot_results` and pass it the dictionary. The time range shown is set by `T_SHOW_END`, and `PLOT_SAVE = False` disables writing files.

---

## Troubleshooting

| Symptom                                       | Likely cause                                                                                                                                                                                        |
|-----------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `RuntimeError: call reset(q0) before compute()` | `reset()` was not called before the control loop.                                                                                                                                                     |
| `NaN` in the torque or potential                | The current attitude violates a constraint: the initial attitude is inside a forbidden zone or outside a mandatory zone, or the time step is too large and the state jumped across a cone boundary. |
| Torque chatters at high frequency             | `epsilon` is too small for the control rate (see Tuning notes).                                                                                                                                       |
| Attitude stops short of the target            | Local minimum of the potential; try increasing `beta1 / beta2`. A small remaining error can also come from a large `epsilon`.                                                                           |
| Spacecraft takes the long way round           | `sign_qe0` was forced to the wrong value, or `reset()` was called with a different initial attitude than the actual one.                                                                                |
| Figures are empty or clipped                  | Several subplots use fixed `ylim` ranges tuned for scenario F4; adjust them in `plot_result_apf_smc_F4.py`.                                                                                             |
