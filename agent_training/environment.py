"""
The training environment for the satellite reorientation task.
Includes the reward function for the agent.

Author: Cemal Yilmaz - 2026
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch as th
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for frame rendering

import math
import sys
import os
import warnings
import copy

from numba import njit
from scipy.spatial.transform import Rotation
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW, unitTestSupport
from Basilisk.simulation import spacecraft, reactionWheelStateEffector
from Basilisk.architecture import messaging
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agent_training.constants import Constants

# Add parent directory to path for imports (must be before local imports)
_drl_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _drl_repo_dir not in sys.path:
    sys.path.insert(0, _drl_repo_dir)

# Scaling factors for normalization in observations
scale_torque = Constants.TORQUE_WHEEL_MAX
scale_torque_norm = np.sqrt(scale_torque**2 + scale_torque**2 + scale_torque**2)  # Only 3 wheels now

scale_angular_velocity_sat = 30.0
scale_angular_velocity_wheels = 630.0
scale_margin_koz = np.pi  # radians
max_abs_state_value = 1e6


@njit
def normalize_quaternion(q):
    """
    Normalize a quaternion to have unit norm.
    Args:
        q: Input quaternion as a numpy array [w, x, y, z].
    Returns:
        q_normalized: Normalized quaternion with unit norm.
    """
    #norm = np.linalg.norm(q)
    norm = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)   # using custom calculation of norm in order to use numba
    if norm > 0:  # Avoid division by zero
        return q / norm
    return q  # Return unchanged if norm is zero

@njit
def normalize_vector(v):
    """
    Normalize a 3D vector to have unit norm.
    Args:
        v: Input vector as a numpy array [x, y, z].
    Returns:
        v_normalized: Normalized vector with unit norm.
    """
    #norm = np.linalg.norm(v)
    norm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)   # using custom calculation of norm in order to use numba
    if norm > 0:  # Avoid division by zero
        return v / norm
    return v  # Return unchanged if norm is zero


@njit
def calc_vector_norm(v):
    """
    Calculate the norm of a 3D vector.
    Args:
        v: Input vector as a numpy array [x, y, z].
    Returns:
        norm: The norm of the vector.
    """
    #norm = np.linalg.norm(v)
    norm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)   # using custom calculation of norm in order to use numba
    
    return norm


@njit
def rotate_vector_by_quaternion(v, q):
    """
    Rotate a vector v by a quaternion q.
    Args:
        v: Input vector as a numpy array [x, y, z].
        q: Quaternion representing the rotation as a numpy array [w, x, y, z].
    Returns:
        v_rotated: The rotated vector as a numpy array [x, y, z].
    """
    v = v.astype(np.float64)
    w, x, y, z = q

    # Convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=np.float64)

    return R @ v

@njit
def rotate_vector_by_quaternion_to_body_frame(v, q):
    """
    Rotate a vector v by a quaternion q from world frame to body frame.
    Args:
        v: Input vector as a numpy array [x, y, z] in world frame.
        q: Quaternion representing the rotation as a numpy array [w, x, y, z].
    Returns:
        v_rotated: The rotated vector as a numpy array [x, y, z] in body frame.
    """
    v = v.astype(np.float64)
    w, x, y, z = q

    # Convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=np.float64)

    return R.T @ v

@njit
def calc_margin_koz(q, normal_vector_koz, half_angle_koz):
    """
    Calculate the margin angle to the keep out zone defined by normal_vector_koz and half_angle_koz.
    Args:
        q: The current attitude quaternion of the satellite as a numpy array [w, x, y, z].
        normal_vector_koz: The normal vector of the keep out zone in inertial frame as a numpy array [x, y, z].
        half_angle_koz: The half angle of the keep out zone in radians.
    Returns:
        margin_angle: The margin angle to the keep out zone in radians.
    """
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    body_axis_arr = rotate_vector_by_quaternion(x_axis, q)

    norm_body = np.sqrt(body_axis_arr[0]**2 + body_axis_arr[1]**2 + body_axis_arr[2]**2)
    norm_koz = np.sqrt(normal_vector_koz[0]**2 + normal_vector_koz[1]**2 + normal_vector_koz[2]**2)
    
    # Calculate the angle between the satellite's body axis and the normal vector of the keep out zone using the dot product
    cos_theta = (body_axis_arr[0] * normal_vector_koz[0] + 
                 body_axis_arr[1] * normal_vector_koz[1] + 
                 body_axis_arr[2] * normal_vector_koz[2]) / (norm_body * norm_koz)
    
    # Manual clip for numba compatibility
    cos_theta = min(max(cos_theta, -1.0), 1.0)
    
    theta = np.arccos(cos_theta)
    margin_angle = theta - half_angle_koz
    
    return margin_angle


def quatToMRP(q):
    R = Rotation.from_quat(q, scalar_first=True)
    mrp = R.as_mrp()
    return mrp

def MRPToQuat(sigma):
    R = Rotation.from_mrp(sigma)
    quat = R.as_quat(scalar_first=True)
    return quat

def build_spacecraft(q_init, omega_init):
    satellite = spacecraft.Spacecraft()
    satellite.ModelTag = "satellite"

    # Hub inertia [kg m^2]
    inertia = [0.02 / 3,  0.,         0.,
                0.,        0.1256 / 3, 0.,
                0.,        0.,         0.1256 / 3]
    satellite.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(inertia)

    satellite.hub.mHub = 4.0 # TODO clarify if needed
    satellite.hub.r_BcB_B = [[0.0], [0.0], [0.0]] # position vector of body-fixed point B relative to center of mass

    sigma_init = quatToMRP(q_init)

    # Basilisk attitude state uses MRPs, not quaternions
    satellite.hub.sigma_BNInit = [[sigma_init[0]], [sigma_init[1]], [sigma_init[2]]]
    satellite.hub.omega_BN_BInit = [[omega_init[0]], [omega_init[1]], [omega_init[2]]]

    return satellite


def build_basilisk_sim(omega_wheel_init, satellite, dt) -> tuple[SimulationBaseClass.SimBaseClass, messaging.ArrayMotorTorqueMsg, reactionWheelStateEffector.ReactionWheelStateEffector]:
    sim = SimulationBaseClass.SimBaseClass()

    process = sim.CreateNewProcess("dynProcess")
    task_name = "dynTask"
    process.addTask(
        sim.CreateNewTask(task_name, macros.sec2nano(dt))
    )

    rw_effector = reactionWheelStateEffector.ReactionWheelStateEffector()
    rw_effector.ModelTag = "reactionWheels"

    rw_factory = simIncludeRW.rwFactory()
    varRWModel = messaging.BalancedWheels

    # Three orthogonal wheels, as configured in Bevo-2. Reaction wheel model: 3x Rocketlab 10 mNms-1
    RW1 = rw_factory.create(
        "custom",
        [1.0, 0.0, 0.0],
        Omega=float(omega_wheel_init[0]),
        u_max=Constants.TORQUE_WHEEL_MAX,
        Omega_max=Constants.SPEED_WHEEL_MAX,
        maxMomentum=Constants.MOMENTUM_WHEEL_MAX,
        RWModel=varRWModel
    )
    RW2 = rw_factory.create(
        "custom",
        [0.0, 1.0, 0.0],
        Omega=float(omega_wheel_init[1]),
        u_max=Constants.TORQUE_WHEEL_MAX,
        Omega_max=Constants.SPEED_WHEEL_MAX,
        maxMomentum=Constants.MOMENTUM_WHEEL_MAX,
        RWModel=varRWModel
    )
    RW3 = rw_factory.create(
        "custom",
        [0.0, 0.0, 1.0],
        Omega=float(omega_wheel_init[2]),
        u_max=Constants.TORQUE_WHEEL_MAX,
        Omega_max=Constants.SPEED_WHEEL_MAX,
        maxMomentum=Constants.MOMENTUM_WHEEL_MAX,
        RWModel=varRWModel
    )

    rw_factory.addToSpacecraft(
        satellite.ModelTag,
        rw_effector,
        satellite,
    )

    # Stand-alone RW motor torque command message
    cmd_payload = messaging.ArrayMotorTorqueMsgPayload()
    cmd_payload.motorTorque = [0.0, 0.0, 0.0]

    rw_cmd_msg = messaging.ArrayMotorTorqueMsg().write(cmd_payload)
    rw_effector.rwMotorCmdInMsg.subscribeTo(rw_cmd_msg)

    # Add modules to task.
    # RW effector before spacecraft dynamics.
    sim.AddModelToTask(task_name, rw_effector, 2)
    sim.AddModelToTask(task_name, satellite, 1)

    return sim, rw_cmd_msg, rw_effector


@njit
def reward_function(state, _q0_prev, torque, torque_prev, phase, state_koz, koz_violation_cnt, time_elapsed):
    q0_current = state[0]
    ang_vel_sat_x = state[4]
    ang_vel_sat_y = state[5]
    ang_vel_sat_z = state[6]
    q0_prev = _q0_prev
    torque_1 = torque[0]
    torque_2 = torque[1]
    torque_3 = torque[2]
    torque_1_prev = torque_prev[0]
    torque_2_prev = torque_prev[1]
    torque_3_prev = torque_prev[2]
    margin_koz = state_koz[0][0] # TODO: support multiple KOZs
    
    # Clamp q0 values to [-1, 1] to prevent acos() domain errors (NaN) with large torques
    # Using min/max instead of np.clip for numba compatibility with scalars
    q0_current = min(max(q0_current, -1.0), 1.0)
    q0_prev = min(max(q0_prev, -1.0), 1.0)
    
    err_phi_current = 2 * math.acos(q0_current)   # in [rad]
    err_phi_prev = 2 * math.acos(q0_prev)   # in [rad]

    err_phi_current = err_phi_current * 180.0 / np.pi
    err_phi_prev = err_phi_prev * 180.0 / np.pi

    err_phi_delta = err_phi_prev - err_phi_current

    ang_vel_norm = calc_vector_norm(np.array([ang_vel_sat_x, ang_vel_sat_y, ang_vel_sat_z]))

    r_total = 0
    USE_REWARD = "mod37"
    
    if USE_REWARD == "paper1":
        # Reward for reducing attitude error
        #r1 = (err_phi_prev - err_phi_current)  # positive if error decreased

        # Penalty for high angular velocity (more than 0.1 rad/s)
        r2 = 0.0
        if ang_vel_sat_x > 0.1 or ang_vel_sat_y > 0.1 or ang_vel_sat_z > 0.1:
            r2 = -1

        # Reward for reducing attitude error and pointing accuracy
        r3 = 0.0
        if err_phi_current < 0.25:
            r3 = 1
        else:
            r3 = 0.5 * (1 - ((err_phi_current-0.25)/180.0)**0.6)

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        r_total = r2 + r3 + r4

    if USE_REWARD == "yang":

        r_err = np.exp(-err_phi_current/(0.14*360))
        r_torque = -0.05 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)/scale_torque_norm - 0.005 * (np.sqrt((torque_1-torque_1_prev)**2 + (torque_2-torque_2_prev)**2 + (torque_3-torque_3_prev)**2))
        r_acc = 0
        if err_phi_current < 0.25:
            r_acc = 9

        r_direction = 0
        if err_phi_current > err_phi_prev:
            r_direction = -1

        r_total = r_err + r_torque + r_acc + r_direction

    if USE_REWARD == "yangMod1":
        r_err = np.exp(-err_phi_current/(0.14*360))
        #r_torque = -0.05 * np.sqrt(torque_1**2 + torque_2**2 + torque_3**2)/scale_torque_norm_yang - 0.005 * (np.sqrt((torque_1-torque_1_prev)**2 + (torque_2-torque_2_prev)**2 + (torque_3-torque_3_prev)**2))
        r_acc = 0
        if err_phi_current < 0.25:
            r_acc = 9

        r_direction = 0
        if err_phi_current > err_phi_prev:
            r_direction = -1

        r_total = r_err + r_acc + r_direction

    if USE_REWARD == "prak":
        # Reward for reducing attitude error
        r1 = (err_phi_prev - err_phi_current)  # positive if error decreased

        # Bonus for high accuracy
        r3 = 0.0
        if err_phi_current < 0.25:
            r3 = 0.01  # bonus for reaching the goal
        else:
            r3 = -0.01

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            if margin_koz <= 0.0:
                r5 = -1.0
            else:
                r5 = -1.0*math.exp(-66.0*margin_koz)

        r_total = r1 + r3 + r4 + r5

    if USE_REWARD == "prakModAcc":
        # Reward for reducing attitude error
        r1 = (err_phi_prev - err_phi_current)  # positive if error decreased

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.25:
            if phase == 2:
                r2 = 0.5
            else:
                r2 = 0.1
        # Gradial bonus for getting closer to target when already close to target
        elif err_phi_current < 5.0:
            r2 = 0.1*math.exp(-0.35*err_phi_current)
        # Penalty if farther away
        else:
            r2 = -0.01

        # Penalty for high angular velocity (more than 0.1 rad/s or 5.7 deg/s)
        r3 = 0.0
        if (ang_vel_sat_x > 0.1) or (ang_vel_sat_y > 0.1) or (ang_vel_sat_z > 0.1):
            r3 = -1.0

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0*math.exp(-10.0*margin_koz)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod2":
        """
        Goal: reduce violation rate.
        Result: very good.
        Note: pointing accuracy still does not recover.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            r1 = err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.2:
            r2 = 0.02
        elif err_phi_current < 0.25:
            r2 = 0.02 * ((0.25-err_phi_current)/0.05)
        # Penalty if farther away
        else:
            r2 = -0.01


        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod22":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: Settling rate improved from 10% to 50%
        Note: bad reward design: if koz violation, no bonus for accuracy but violation penalty not hard enough.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod23":
        """
        Goal: recover pointing accuracy in phase 2 (improve mod22).
        Result: Adding the -0.01 penalty makes both accuracy and violation worse.
        Note: Added -0.01 penalty when not within desired accuracy.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            else:
                r2 = -0.01
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = -0.1
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            else:
                r2 = -0.01
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod24":
        """
        Goal: recover pointing accuracy in phase 2 (improve mod22).
        Result: worse than before
        Note: Increased bonus for desired pointing accuracy
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.1
            elif err_phi_current < 0.25:
                r2 = 0.1 * (((0.25-err_phi_current)/(10.0/9.0 * 0.05)) + 0.1)
            
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.1
            elif err_phi_current < 0.25:
                r2 = 0.1 * (((0.25-err_phi_current)/(10.0/9.0 * 0.05)) + 0.1)
            
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod25":
        """
        Goal: recover pointing accuracy in phase 2 (improve mod22).
        Result: Worse
        Note: Removed constant terms for reducing attitude error. Removed torque penalty. Adjusted pointing bonus.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.0 * err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.0 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 0.02 + 0.1*(0.25-err_phi_current) 
    
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.25:
                r2 = 0.02 + 0.1*(0.25-err_phi_current) 
            
        # Penalty for using large torques
        r4 = 0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod26":
        """
        Goal: recover pointing accuracy in phase 2 (improve mod22).
        Result: 
        Note: Removed constant terms for reducing attitude error. Removed torque penalty. Adjusted pointing bonus.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.0 * err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.0 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 0.04 + 0.1*(0.25-err_phi_current)
            # Bonus for being close to desired accuracy
            elif err_phi_current < 2.0:
                r2 = 0.04 * (1.0/(2.0-0.25) * (2.0-err_phi_current))**2
    
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.25:
                r2 = 0.04 + 0.1*(0.25-err_phi_current)
            # Bonus for being close to desired accuracy
            elif err_phi_current < 2.0:
                r2 = 0.04 * (1.0/(2.0-0.25) * (2.0-err_phi_current))**2
            
        # Penalty for using large torques
        r4 = 0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        #print(f"DEBUG err {err_phi_current:.2f}, vel {ang_vel_norm:.2f}, margin {(margin_koz*180/np.pi):.2f}, r1 {r1:.3f}, r2 {r2:.3f}, , r3 {r3:.3f}, , r4 {r4:.3f}, , r5 {r5:.3f}")

        r_total = r1 + r2 + r4 + r5
        #print(f"err_delta {err_phi_delta:.3f}, err_cur {err_phi_current:.3f}, margin {margin_koz:.3f}")
        #print(f"r1 {r1:.3f}, r2 {r2:.3f}, r4 {r4:.3f}, r5 {r5:.3f}")

    if USE_REWARD == "mod27":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: higher accuracy bonus, s-shaped function.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.25:
                r2 = 0.01 + 0.2*(np.cbrt(0.125) - np.cbrt(err_phi_current - 0.125)) # [0.01, 0.21]
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.25:
                r2 = 0.01 + 0.2*(np.cbrt(0.125) - np.cbrt(err_phi_current - 0.125))
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod28":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: rmeove constant term in r1. remove different weighting in r1.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            r1 = err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod29":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: rmeove constant term in r1. remove different weighting in r1. remove torque penalty.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            r1 = err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            

        # Penalty for using large torques
        r4 = 0.0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod30":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: rmeove constant term in r1. remove different weighting in r1. remove torque penalty. add back bad accuracy penalty
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            r1 = err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            
        elif phase == 2:
            # Penalty if violated KOZ or not within desired accuracy
            if koz_violation_cnt > 0 or err_phi_current >= 0.25:
                r2 = -0.04
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            

        # Penalty for using large torques
        r4 = 0.0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod31":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: rmeove constant term in r1. remove different weighting in r1. remove torque penalty.
                change bad accuracy penalty to based on timestep
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            r1 = err_phi_delta
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            
        elif phase == 2:
            # Penalty if violated KOZ or not within desired accuracy
            if koz_violation_cnt > 0 or err_phi_current >= 0.25:
                r2 = -0.000005 * time_elapsed**2 * err_phi_current
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * ((0.25-err_phi_current)/0.05)
            

        # Penalty for using large torques
        r4 = 0.0

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod32":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: increased accuracy bonus
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            
        elif phase == 2:
            # No accuracy bonus if violated KOZ
            if koz_violation_cnt > 0:
                r2 = 0.0
            # Bonus for desired accuracy
            elif err_phi_current < 0.2:
                r2 = 0.1
            elif err_phi_current < 0.25:
                r2 = 0.1 * (((0.25-err_phi_current)/0.05)) + 0.01
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -1.0 * (1.0 - margin_koz/0.17)
            # No penalty if farther away
            else:
                r5 = 0.0

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod33":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: removed pointing accuracy bonus not given if KOZ violation. added pointing accuracy penalty
                when too far away. increased KOZ penalty. added persistent penalty once inside KOZ.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
                if err_phi_current > 0.25:
                    r1 += 0.01
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                    if err_phi_current > 0.25:
                        r1 += 0.01
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta
                if err_phi_current > 0.25:
                    r1 -= 0.012

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            
        elif phase == 2:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.03
            elif err_phi_current < 0.25:
                r2 = 0.03 * (0.25-err_phi_current) / 0.05 # [0, 0.03]
            else:
                r2 = -0.01 * min((err_phi_current-0.25)/4.75, 1.0) # [-0.01, 0]
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -2.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -2.0 * (1.0 - margin_koz/0.17)

            # Add persistent penalty for the remaining episode once inside KOZ
            if koz_violation_cnt > 0:
                r5 = r5 - 0.05

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod34":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: diff mod33: removed constant reward in r1. slightly increased accuracy penalty.
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            
        elif phase == 2:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.03
            elif err_phi_current < 0.25:
                r2 = 0.03 * (0.25-err_phi_current) / 0.05 # [0, 0.03]
            else:
                r2 = -0.01 - 0.01 * min((err_phi_current-0.25)/4.75, 1.0) # [-0.02, -0.01]
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -2.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -2.0 * (1.0 - margin_koz/0.17)

            # Add persistent penalty for the remaining episode once inside KOZ
            if koz_violation_cnt > 0:
                r5 = r5 - 0.05

        
        r_total = r1 + r2 + r4 + r5

    if USE_REWARD == "mod35":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: diff mod34: added settling reward
        """

        # Reward for reducing attitude error
        r1 = 0 
        # Phase 1
        if phase == 1:
            if err_phi_delta >= 0:
                r1 = err_phi_delta
            # Increasing error is punished more than decreasing error is rewarded
            else:
                r1 = 1.2 * err_phi_delta
                
            
        # Phase 2
        else:      
            if err_phi_delta >= 0:
                if margin_koz > 0.17:
                    r1 = err_phi_delta
                elif margin_koz > 0:
                    r1 = err_phi_delta * (margin_koz/0.17)
                else:
                    r1 = 0
            
            else:
                r1 = 1.2 * err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        if phase == 1:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.02
            elif err_phi_current < 0.25:
                r2 = 0.02 * (((0.25-err_phi_current)/0.1) + 0.5)
            
        elif phase == 2:
            # Bonus for desired accuracy
            if err_phi_current < 0.2:
                r2 = 0.03
            elif err_phi_current < 0.25:
                r2 = 0.03 * (0.25-err_phi_current) / 0.05 # [0, 0.03]
            else:
                r2 = -0.01 - 0.01 * min((err_phi_current-0.25)/4.75, 1.0) # [-0.02, -0.01]
            

        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -2.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif margin_koz < 0.17:
                r5 = -2.0 * (1.0 - margin_koz/0.17)

            # Add persistent penalty for the remaining episode once inside KOZ
            if koz_violation_cnt > 0:
                r5 = r5 - 0.05

        # Reward if settled
        r6 = 0.0
        if phase == 2:
            if err_phi_current < 0.25 and ang_vel_norm < 0.001:
                r6 = 0.02
        
        r_total = r1 + r2 + r4 + r5 + r6

    if USE_REWARD == "mod37":
        """
        Goal: recover pointing accuracy in phase 2.
        Result: 
        Note: take a step back, see what causes pointing accuracy to become worse.
        """

        # Reward for reducing attitude error
        r1 = err_phi_delta

        # Bonus for high accuracy
        r2 = 0.0
        # Bonus for desired accuracy
        if err_phi_current < 0.2:
            r2 = 0.02
        elif err_phi_current < 0.25:
            r2 = 0.02 * ((0.25-err_phi_current)/0.05)
        # Penalty if farther away
        else:
            r2 = -0.01


        # Penalty for using large torques
        r4 = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r5 = 0.0
        if phase == 2:
            # Maximum penalty inside of KOZ
            if margin_koz <= 0.0:
                r5 = -1.0
            # No penalty if farther away
            else:
                r5 = 0.0

        r_total = r1 + r2 + r4 + r5

    return r_total


class BasiliskRWEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None, initial_state=None):
        super(BasiliskRWEnv).__init__()

        self.episode_count = 0

        self.dt = Constants.TIME_DELTA
        self.steps = 0
        self.sim_time = 0.0

        self.rw_effector = None
        self.rw_cmd_msg = None

        self.MAX_ZONES = 4
        self.KOZ_FEATURE_DIM = 4
        self.SAT_OBS_DIM = 10

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(3,),
            dtype=np.float64,
        )

        self.observation_space = spaces.Dict({
            "satellite": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.SAT_OBS_DIM,),
                dtype=np.float64
            ),
            "zones": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.MAX_ZONES, self.KOZ_FEATURE_DIM),
                dtype=np.float64
            ),
            "zones_mask": spaces.Box(
                low=0,
                high=1,
                shape=(self.MAX_ZONES,),
                dtype=np.float64
            )
        })

        self.sim = None
        self.satellite = None
        self.render_mode = render_mode

        # If no initial state is provided, use default randomization parameters
        if initial_state is None:
            self.min_initial_angle = 0.0  # degrees - minimum initial attitude error
            self.max_initial_angle = 90.0  # degrees - maximum initial attitude error
            self.min_initial_angular_velocity = 0.0  # deg/s - minimum initial tumbling rate
            self.max_initial_angular_velocity = 0.1  # deg/s - maximum initial tumbling rate
            self.max_steps = 3000
            self.min_half_angle_koz = 0.0  # degrees
            self.max_half_angle_koz = 0.0  # degrees
            self.min_nr_koz = 0
            self.max_nr_koz = 0
        else:
            self.min_initial_angle = initial_state[0]
            self.max_initial_angle = initial_state[1]
            self.min_initial_angular_velocity = initial_state[2]
            self.max_initial_angular_velocity = initial_state[3]
            self.max_steps = initial_state[4]
            self.min_half_angle_koz = initial_state[5]
            self.max_half_angle_koz = initial_state[6]
            self.min_nr_koz = initial_state[7]
            self.max_nr_koz = initial_state[8]

        if self.max_half_angle_koz > 0.0:
            self.PHASE = 2
        else:
            self.PHASE = 1

        self.current_nr_koz = np.random.randint(self.min_nr_koz, self.max_nr_koz+1) # Excludes upper bound

        # Custom metrics tracking for TensorBoard
        self.initial_error_angle = 0.0
        self.initial_angular_velocity_mag = 0.0
        self.episode_torques = []
        self.episode_torques_prev = []
        self.settled = False
        self.settling_time = None  # means not settled
        self.settling_threshold_deg = 0.25  # degrees for considering "settled"
        self.min_margin_koz = 0.0
        self.entered_koz_count = 0

        self.x_axis = np.array([1, 0, 0], dtype=np.float64) # Boresight vector (body frame)

        # Set initial state (will be randomized in reset())
        self.reset()

    def _generate_quaternion_with_vector_angle(self, reference_vector, min_angle_deg, max_angle_deg):
        """
        Generate a quaternion that rotates the reference_vector by an angle between 
        min_angle_deg and max_angle_deg in a random direction.
        
        Args:
            reference_vector: The vector to rotate (e.g., [1, 0, 0])
            min_angle_deg: Minimum angle (degrees) between original and rotated vector
            max_angle_deg: Maximum angle (degrees) between original and rotated vector
            
        Returns:
            quaternion: A quaternion [w, x, y, z] that rotates reference_vector by the desired angle
        """
        if max_angle_deg == 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        # Normalize the reference vector
        ref_vec = np.array(reference_vector, dtype=np.float64)
        ref_vec = ref_vec / np.linalg.norm(ref_vec)
        
        # If min and max are equal, use that angle directly
        if min_angle_deg == max_angle_deg:
            angle_deg = max_angle_deg

        # If min and max are not equal, sample randomly
        else:
            # Random angle between min and max, following an uniform distribution
            angle_deg = np.random.uniform(min_angle_deg, max_angle_deg)

        angle_rad = angle_deg * np.pi / 180  # convert to radians
        
        # Generate a random axis perpendicular to the reference vector
        # Method: Generate random vector, then project out the parallel component
        random_vec = np.random.randn(3)
        # Remove component parallel to reference vector
        parallel_component = np.dot(random_vec, ref_vec) * ref_vec
        perpendicular_vec = random_vec - parallel_component
        
        # Normalize to get the rotation axis
        axis = perpendicular_vec / np.linalg.norm(perpendicular_vec)
        
        # Convert axis-angle to quaternion
        q0 = np.cos(angle_rad / 2)
        q_vec = np.sin(angle_rad / 2) * axis

        # Ensure positive scalar part of quaternion
        if q0 < 0:
            q0 = -q0
            q_vec = -q_vec
        
        quaternion = np.array([q0, q_vec[0], q_vec[1], q_vec[2]], dtype=np.float64)
        return normalize_quaternion(quaternion)
    
    def _generate_keep_out_zone(self, initial_quaternion, min_half_angle_deg, max_half_angle_deg):
        """
        Generates a keep out zone defined by a normal vector and half-angle.
        Args:
            initial_quaternion: The initial attitude quaternion of the satellite.
            min_half_angle_deg: Minimum half-angle of the keep out zone in degrees.
            max_half_angle_deg: Maximum half-angle of the keep out zone in degrees.
        Returns:
            res: A tuple containing:
            normal_vector_koz: The normal vector of the keep out zone in inertial frame.
            half_angle_koz: The half-angle of the keep out zone in radians.
        """
        # Convert initial boresight quaternion to vector in inertial frame
        initial_vector_boresight_inertial = rotate_vector_by_quaternion(self.x_axis, initial_quaternion) #r_F inertial frame

        # Calculate normal vector of keep out zone to be the bisector (middle between initial boresight and target boresight, same plane)
        normal_vector_koz = normalize_vector(initial_vector_boresight_inertial + self.x_axis)

        # Random half-angle between min and max
        half_angle_koz = np.random.uniform(min_half_angle_deg, max_half_angle_deg) * np.pi / 180  # in radians

        # Test: set half angle to 0 if (effective) initial error angle is not large enough
        # if self.initial_error_angle - (half_angle_koz * 180 / np.pi) < 30:
        #     half_angle_koz = 0

        return normal_vector_koz, half_angle_koz

    

    def _get_sat_state(self):
        state = self.satellite.scStateOutMsg.read()
        state_rw = self.rw_effector.rwSpeedOutMsg.read()

        sigma = np.array(state.sigma_BN, dtype=np.float64)
        omega = np.array(state.omega_BN_B, dtype=np.float64)
        omega_rw = np.array(state_rw.wheelSpeeds[0:3], dtype=np.float64)

        quat = MRPToQuat(sigma)
        quat = np.array(quat, dtype=np.float64)

        

        return np.concatenate([quat, omega, omega_rw]).astype(np.float64)
    
    def _get_koz_state(self, quat):
        
        koz_state = np.zeros((self.MAX_ZONES, self.KOZ_FEATURE_DIM), dtype=np.float64)

        # TODO: support multiple KOZs
        if self.current_nr_koz > 0:
            margin_koz = calc_margin_koz(quat, self.normal_vector_koz, self.half_angle_koz)
            normal_vector_bf = rotate_vector_by_quaternion_to_body_frame(self.normal_vector_koz, quat)
            direction_vector_bf = normal_vector_bf - self.x_axis

            koz_state[0][0] = np.array(margin_koz, dtype=np.float64) 
            koz_state[0][1:4] = np.array(direction_vector_bf, dtype=np.float64)

        # TODO: sort

        return koz_state
    
    def _get_koz_mask_state(self):
        mask = np.zeros((self.MAX_ZONES,), dtype=np.float64)
        mask[:self.current_nr_koz] = 1

        return mask
    
    def _get_state(self):
        sat_state = self._get_sat_state()
        koz_state = self._get_koz_state(sat_state[:4])
        koz_mask_state = self._get_koz_mask_state()

        state = {
            "satellite": sat_state.astype(np.float64),
            "zones": koz_state.astype(np.float64),
            "zones_mask": koz_mask_state.astype(np.float64)
        }

        return state
        

    def _apply_action(self, action):
        wheel_motor_torque = (
            np.clip(action, -1.0, 1.0) * Constants.TORQUE_WHEEL_MAX
        )

        cmd_payload = messaging.ArrayMotorTorqueMsgPayload()
        cmd_payload.motorTorque = wheel_motor_torque.tolist()

        self.rw_cmd_msg.write(cmd_payload, self.sim.TotalSim.CurrentNanos)

    

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.episode_count += 1

        self.steps = 0
        self.sim_time = 0.0

        # Generate random initial attitude error (0° to max_initial_angle)
        q_array_initial = self._generate_quaternion_with_vector_angle(self.x_axis, self.min_initial_angle, self.max_initial_angle)
        
        # Generate random initial angular velocities
        omega_min_rad = self.min_initial_angular_velocity * np.pi / 180  # Convert to rad/s
        omega_max_rad = self.max_initial_angular_velocity * np.pi / 180  # Convert to rad/s
        
        # Generate random magnitudes between min and max
        omega_magnitude = np.random.uniform(omega_min_rad, omega_max_rad)
        
        # Generate random direction (uniformly distributed on unit sphere)
        omega_direction = np.random.randn(3)
        omega_direction_norm = np.linalg.norm(omega_direction)
        if omega_direction_norm < 1e-12:
            omega_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            omega_direction = omega_direction / omega_direction_norm
        
        # Scale direction by magnitude
        omega_initial = (omega_magnitude * omega_direction).astype(np.float64)

        wheel_velocities_initial = np.zeros(3, dtype=np.float64)

        # Generate keep out zone, vector in inertial frame (--> constant per episode), half angle in radians
        self.normal_vector_koz, self.half_angle_koz = self._generate_keep_out_zone(q_array_initial, self.min_half_angle_koz, self.max_half_angle_koz)

        self.current_nr_koz = np.random.randint(self.min_nr_koz, self.max_nr_koz+1) # Excludes upper bound
        
        # Calculate margin angle to keep out zone
        margin_koz = calc_margin_koz(q_array_initial, self.normal_vector_koz, self.half_angle_koz)

        sat_state = np.concatenate((q_array_initial, omega_initial, wheel_velocities_initial))
        koz_state = self._get_koz_state(q_array_initial)
        koz_mask_state = self._get_koz_mask_state()
        self.state = {
            "satellite": sat_state.astype(np.float64),
            "zones": koz_state.astype(np.float64),
            "zones_mask": koz_mask_state.astype(np.float64)
        }

        self.torque_prev = np.zeros(3, dtype=np.float64)

        # Initialize custom metrics for this episode
        self.initial_error_angle = 2 * math.acos(min(max(abs(q_array_initial[0]), 0.0), 1.0)) * 180 / np.pi  # degrees
        self.initial_angular_velocity_mag = np.linalg.norm(omega_initial) * 180 / np.pi  # deg/s
        self.episode_torques = []
        self.episode_torques_prev = []
        self.settled = False
        self.settling_time = None
        self.min_margin_koz = np.pi
        self.entered_koz_count = 0

        # Update min margin koz angle
        if margin_koz < self.min_margin_koz:
            self.min_margin_koz = margin_koz

        # Update entered koz count
        if margin_koz < 0.0:
            self.entered_koz_count += 1

        # Copy state into observation
        obs = copy.deepcopy(self.state)

        self.satellite = build_spacecraft(q_array_initial, omega_initial)
        self.sim, self.rw_cmd_msg, self.rw_effector = build_basilisk_sim(wheel_velocities_initial, self.satellite, Constants.TIME_DELTA)
        self.sim.InitializeSimulation()

        return obs, {}

    def step(self, action):
        q0_prev = self.state["satellite"][0]


        self._apply_action(action)

        self.sim_time += self.dt
        self.sim.ConfigureStopTime(macros.sec2nano(self.sim_time))
        self.sim.ExecuteSimulation()

        self.state = self._get_state()
        
        reward = reward_function(self.state["satellite"], q0_prev, action * Constants.TORQUE_WHEEL_MAX, self.torque_prev, self.PHASE, self.state["zones"], self.entered_koz_count, self.steps*self.dt)

        # Update KOZ metrics
        # Update min margin koz angle
        margin_koz = self.state["zones"][0][0] # TODO: support multiple KOZs
       
        if margin_koz < self.min_margin_koz:
            self.min_margin_koz = margin_koz

        # Update entered koz count
        if margin_koz < 0.0:
            self.entered_koz_count += 1

        # Copy state into observation
        obs = copy.deepcopy(self.state)
        

        self.episode_torques.append(np.linalg.norm(action * Constants.TORQUE_WHEEL_MAX))
        self.episode_torques_prev.append(np.linalg.norm(self.torque_prev))

        # Check settling condition
        current_error_deg = 2 * math.acos(min(max(abs(self.state["satellite"][0]), 0.0), 1.0)) * 180 / np.pi
        is_within_accuracy = True if current_error_deg <= self.settling_threshold_deg else False

        # From unsettled to settled
        if not self.settled and is_within_accuracy:
            self.settled = True
            self.settling_time = self.steps * self.dt

        # From settled to unsettled
        elif self.settled and not is_within_accuracy:
            self.settled = False
            self.settling_time = None

        self.torque_prev = action * Constants.TORQUE_WHEEL_MAX  # Update previous torque for the next step

        self.steps += 1
        truncated = False
        terminated = self.steps >= self.max_steps

        info = {}

        if terminated:
            final_error_angle = current_error_deg
            avg_torque = np.mean(self.episode_torques) if self.episode_torques else 0.0
            max_torque = np.max(self.episode_torques) if self.episode_torques else 0.0
            max_torque_prev = np.max(self.episode_torques_prev) if self.episode_torques else 0.0
            min_margin_koz = self.min_margin_koz * 180 / np.pi  # convert to degrees
            
            info.update({
                "custom_metrics/initial_error_angle": self.initial_error_angle,
                "custom_metrics/initial_angular_velocity": self.initial_angular_velocity_mag,
                "custom_metrics/final_error_angle": final_error_angle,
                "custom_metrics/settling_time": self.settling_time,
                "custom_metrics/avg_torque": avg_torque,
                "custom_metrics/max_torque": max_torque,
                "custom_metrics/max_torque_prev": max_torque_prev,
                "custom_metrics/settled": float(self.settled),
                "custom_metrics/min_margin_koz": min_margin_koz,
                "custom_metrics/entered_koz_count": float(self.entered_koz_count)
            })

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the current state of the environment.
        Depending on the render mode, it either prints the state information or returns an RGB array representing the satellite's attitude.
        """

        if self.render_mode == "human":
            return

        if self.render_mode == "rgb_array":
            q = self.state["satellite"][:4]

            # Rotate the satellite body axis (x-axis) by the quaternion
            body_axis = rotate_vector_by_quaternion(self.x_axis, q)

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=30, azim=135)

            # Draw world x-axis (target axis)
            ax.quiver(0, 0, 0, 1, 0, 0, color="red")

            # Draw the satellite body axis
            ax.quiver(0, 0, 0, body_axis[0], body_axis[1], body_axis[2], color="black", linewidth=3)

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_box_aspect([1, 1, 1])

            # Convert the figure to an RGB array
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            plt.close(fig)

            return frame

    def close(self):
        pass


class LSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, sat_obs_dim: int, lstm_out_dim: int):
        self.sat_obs_dim = sat_obs_dim
        self.lstm_out_dim = lstm_out_dim
        self.total_obs_dim = sat_obs_dim + lstm_out_dim

        # Number of features for the extractor corresponds to the combined observation, as the extractor also combines it.
        super().__init__(observation_space=observation_space, features_dim=self.total_obs_dim)

        # Get the maximum number of KOZs and the LSTM input dimension (= feature dimension of a KOZ)
        zones_max, lstm_in_dim = observation_space["zones"].shape
        self.zones_max = zones_max
        self.lstm_in_dim = lstm_in_dim


        # The LSTM 
        self.lstm = th.nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=lstm_out_dim,
            num_layers=1, # TODO: clarify
            batch_first=True # Because TODO
        )

    def forward(self, observations):
        sat_obs: th.Tensor = observations["satellite"]
        zones_obs: th.Tensor = observations["zones"]
        zones_mask: th.Tensor = observations["zones_mask"] # Defines which vectors in the zones obs are actual zones at this forward pass
        batch_size = sat_obs.shape[0]
        device = sat_obs.device

        # How many KOZs each batch observation contains
        zones_count = zones_mask.sum(dim=1).long()

        # A bit mask for the batch: True --> has at least one KOZ; False has no KOZ.
        non_zero_zones_mask = zones_count > 0

        # From the mask, get the indices of the batch for the observations with at least one KOZ
        non_zero_indices = non_zero_zones_mask.nonzero(as_tuple=True)[0]

        # Container for LSTM output of the entire batch
        self.lstm_out = th.zeros(batch_size, self.lstm_out_dim, device=device)

        # Number of batch obsverations with at least one KOZ
        non_zero_obs_count = non_zero_indices.numel()

        # If there is at least one obs with KOZs, process the LSTM
        if non_zero_obs_count > 0:

            # Get all observations (and their KOZ count) with at least one KOZ from the batch
            non_zero_zones = zones_obs[non_zero_indices]
            non_zero_zones_count = zones_count[non_zero_indices]

            # Create an LSTM sequence (consisting of as many LSTM cells as there are KOZs) for each batch obs (which has at least 1 KOZ)
            lstm_sequences = th.nn.utils.rnn.pack_padded_sequence(
                input=non_zero_zones,
                lengths=non_zero_zones_count.cpu(), # As we provide it as a tensor (and not a list), must be on the CPU
                batch_first=True,
                enforce_sorted=False
            )

            # Process the LSTM
            output, (hidden_state_out, cell_state_out) = self.lstm(input=lstm_sequences)

            # Store the final hidden state of each batch observation
            self.lstm_out[non_zero_indices] = hidden_state_out[-1] # of the last layer
            #self.lstm_out[non_zero_indices] = zones_obs[non_zero_indices, 0]

        # Combine satellite obs and LSTM output
        combined = th.cat([sat_obs, self.lstm_out], dim=1)

        ####################
        # for param_name, param in self.lstm.named_parameters():
        #     # Get the parameter tensor
        #     param_value = param.detach().cpu()

        #     # Separate weights and biases per gate and convert from tensor --> list.
        #     input_values = param_value[0:self.lstm_out_dim].tolist()
        #     print(f"{param_name}: {input_values}")

        return combined