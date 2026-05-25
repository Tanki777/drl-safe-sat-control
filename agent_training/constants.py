"""
Contains all constants.

Author: Cemal Yilmaz - 2026
"""

import numpy as np

class Constants:
    J_b = np.diag([0.1672, 0.1259, 0.06121])  # Body's moment of inertia (no wheels) [kg·m²] (diagonal 3x3)

    INERTIA_WHEEL = 0.00001722  # Moment of inertia of one wheel (kgm^2)
    TORQUE_WHEEL_MAX = 0.001 # Maximum wheel torque (Nm)
    SPEED_WHEEL_MAX = 6000.0 # Maximum wheel speed (RPM)
    MOMENTUM_WHEEL_MAX = 0.01 # Maximum wheel momentum (Nms^-1)
    # TODO: clarify whether 10 mNms^-1 or 1/3 of that should be used. Paper says 10 total, basilisk example uses 10 per wheel.
    TIME_DELTA = 0.1 # Simulation timestep (s)