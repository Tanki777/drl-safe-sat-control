"""
Contains all constants.

Author: Cemal Yilmaz - 2026
"""

class Constants:
    INERTIA_BODY = [0.006667,  0.,         0.,
            0.,        0.04187, 0.,
            0.,        0.,         0.04187]  # Body's moment of inertia (no wheels) [kg m^2]

    INERTIA_WHEEL = 0.00001592  # Moment of inertia of one wheel (kgm^2). Not directly used, since Basilisk calculates it from max momentum.
    TORQUE_WHEEL_MAX = 0.001 # Maximum wheel torque (Nm)
    SPEED_WHEEL_MAX = 6000.0 # Maximum wheel speed (RPM)
    MOMENTUM_WHEEL_MAX = 0.01 # Maximum wheel momentum (Nms)
    TIME_DELTA = 0.1 # Simulation timestep (s)