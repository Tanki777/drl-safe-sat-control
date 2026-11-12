"""
Contains all constants.

Author: Cemal Yilmaz - 2026
"""

import numpy as np

class Constants:
    J_b = [0.02 / 3,  0.,         0.,
            0.,        0.1256 / 3, 0.,
            0.,        0.,         0.1256 / 3]  # Body's moment of inertia (no wheels) [kg m²]

    INERTIA_WHEEL = 0.00001592  # Moment of inertia of one wheel (kgm^2)
    TORQUE_WHEEL_MAX = 0.001 # Maximum wheel torque (Nm)
    SPEED_WHEEL_MAX = 6000.0 # Maximum wheel speed (RPM)
    MOMENTUM_WHEEL_MAX = 0.01 # Maximum wheel momentum (Nms^-1)
    TIME_DELTA = 0.1 # Simulation timestep (s)