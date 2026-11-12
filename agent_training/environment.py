"""
The training environment for the satellite reorientation task.
Includes the reward function for the agent.

Author: Cemal Yilmaz - 2026
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch as th

import math
import sys
import os
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
scale_torque_norm = np.sqrt(scale_torque**2 + scale_torque**2 + scale_torque**2)


@njit
def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion to have unit norm.

    Args:
        q (ndarray): Input quaternion as a numpy array [w, x, y, z].
    Returns:
        q_normalized (ndarray): Normalized quaternion with unit norm.
    """

    norm = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)   # using custom calculation of norm in order to use numba
    
    if norm > 0:  # Avoid division by zero
        return q / norm
    
    return q  # Return unchanged if norm is zero

@njit
def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a 3D vector to have unit norm.

    Args:
        v (ndarray): Input vector as a numpy array [x, y, z].
    Returns:
        v_normalized (ndarray): Normalized vector with unit norm.
    """

    norm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)   # using custom calculation of norm in order to use numba
    
    if norm > 0:  # Avoid division by zero
        return v / norm
    
    return v  # Return unchanged if norm is zero


@njit
def calc_vector_norm(v: np.ndarray) -> float:
    """
    Calculate the norm of a 3D vector.

    Args:
        v (ndarray): Input vector as a numpy array [x, y, z].
    Returns:
        norm (float): The norm of the vector.
    """
    
    norm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)   # using custom calculation of norm in order to use numba
    
    return norm


@njit
def rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate a vector v by a quaternion q.
    
    Args:
        v (ndarray): Input vector as a numpy array [x, y, z].
        q (ndarray): Quaternion representing the rotation as a numpy array [w, x, y, z].
    Returns:
        v_rotated (ndarray): The rotated vector as a numpy array [x, y, z].
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
def rotate_vector_by_quaternion_to_body_frame(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate a vector v by a quaternion q from world frame to body frame.

    Args:
        v (ndarray): Input vector as a numpy array [x, y, z] in world frame.
        q (ndarray): Quaternion representing the rotation as a numpy array [w, x, y, z].
    Returns:
        v_rotated (ndarray): The rotated vector as a numpy array [x, y, z] in body frame.
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
def calc_margin_koz(q: np.ndarray, normal_vector_koz: np.ndarray, half_angle_koz: float) -> float:
    """
    Calculate the margin angle to the keep out zone defined by normal_vector_koz and half_angle_koz.

    Args:
        q (ndarray): The current attitude quaternion of the satellite as a numpy array [w, x, y, z].
        normal_vector_koz (ndarray): The normal vector of the keep out zone in inertial frame as a numpy array [x, y, z].
        half_angle_koz (float): The half angle of the keep out zone in radians.
    Returns:
        margin_angle (float): The margin angle to the keep out zone in radians.
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


def quatToMRP(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to Modified Rodrigues Parameters (MRP).

    Args:
        q (ndarray): Input quaternion.
    Returns:
        mrp (ndarray): Output MRP.
    """

    R = Rotation.from_quat(q, scalar_first=True)
    mrp = R.as_mrp()

    return mrp

    
def MRPToQuat(sigma: np.ndarray) -> np.ndarray:
    """
    Convert Modified Rodrigues Parameters (MRP) to a quaternion.

    Args:
        sigma (ndarray): Input MRP.
    Returns:
        quat (ndarray): Output quaternion.
    """

    R = Rotation.from_mrp(sigma)
    quat = R.as_quat(scalar_first=True)

    return quat


def build_spacecraft(q_init: np.ndarray, omega_init: np.ndarray) -> spacecraft.Spacecraft:
    """
    Builds a Basilisk spacecraft object with the specified initial attitude and angular velocity.

    Args:
        q_init (ndarray): Initial attitude quaternion.
        omega_init (ndarray): Initial angular velocity.
    Returns:
        satellite (Spacecraft): The built spacecraft object.
    """

    satellite = spacecraft.Spacecraft()
    satellite.ModelTag = "satellite"

    # body inertia
    satellite.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(Constants.INERTIA_BODY)

    satellite.hub.r_BcB_B = [[0.0], [0.0], [0.0]] # position vector of body-fixed point B relative to center of mass

    sigma_init = quatToMRP(q_init)

    # Basilisk attitude state uses MRPs, not quaternions
    satellite.hub.sigma_BNInit = [[sigma_init[0]], [sigma_init[1]], [sigma_init[2]]]
    satellite.hub.omega_BN_BInit = [[omega_init[0]], [omega_init[1]], [omega_init[2]]]

    return satellite


def build_basilisk_sim(omega_wheel_init: np.ndarray, satellite: spacecraft.Spacecraft, dt: float) -> tuple[SimulationBaseClass.SimBaseClass, messaging.ArrayMotorTorqueMsg, reactionWheelStateEffector.ReactionWheelStateEffector]:
    """
    Builds a Basilisk simulation with the specified initial wheel speeds, spacecraft, and time step.

    Args:
        omega_wheel_init (ndarray): Initial wheel speeds.
        satellite (Spacecraft): The spacecraft to include in the simulation.
        dt (float): The time step for the simulation.

    Returns:
        sim, rw_cmd_msg, rw_effector (tuple[SimulationBaseClass, ArrayMotorTorqueMsg, ReactionWheelStateEffector]): The built simulation, the reaction wheel command message, and the reaction wheel effector.
    """
    
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

    # Standalone RW motor torque command message
    cmd_payload = messaging.ArrayMotorTorqueMsgPayload()
    cmd_payload.motorTorque = [0.0, 0.0, 0.0]

    rw_cmd_msg = messaging.ArrayMotorTorqueMsg().write(cmd_payload)
    rw_effector.rwMotorCmdInMsg.subscribeTo(rw_cmd_msg)

    # Add modules to task.
    # RW effector before spacecraft dynamics.
    sim.AddModelToTask(task_name, rw_effector, 2)
    sim.AddModelToTask(task_name, satellite, 1)

    return sim, rw_cmd_msg, rw_effector


def reward_function(state: np.ndarray, _q0_prev: np.ndarray, torque: np.ndarray, torque_prev: np.ndarray, phase: str, state_koz: np.ndarray, koz_violation_cnt: int, time_elapsed: float, termination: bool, discount_factor: float, koz_margin_min_prev: float) -> float:
    """
    Calculates the reward based on the current state and additional parameters.
    Some parameters are only used for archived reward functions and are kept for compatibility.

    Args:
        state (ndarray): The current state of the satellite.
        _q0_prev (float): The previous quaternion scalar component.
        torque (ndarray): The current torque applied to the satellite.
        torque_prev (ndarray): The previous torque applied to the satellite.
        phase (str): The current phase ("phase 1" or "phase 2").
        state_koz (ndarray): The current state of the keep out zones.
        koz_violation_cnt (int): The count of keep out zone violations.
        time_elapsed (float): The elapsed time in the episode.
        termination (bool): Whether the episode has terminated.
        discount_factor (float): The discount factor for future rewards.
        koz_margin_min_prev (float): The previous minimum margin angle to the keep out zones.
    Returns:
        float: The calculated reward.
    """
    
    q0_current = state[0]
    q0_prev = _q0_prev
    torque_1 = torque[0]
    torque_2 = torque[1]
    torque_3 = torque[2]
    koz_margin_min = np.min(state_koz[:, 0]) # For KOZ related reward calculation, use the minimum margin angle of all KOZs.
    
    # Clamp q0 values to [-1, 1] to prevent acos() domain errors (NaN) with large torques
    # Using min/max instead of np.clip for numba compatibility with scalars
    q0_current = min(max(abs(q0_current), -1.0), 1.0)
    q0_prev = min(max(abs(q0_prev), -1.0), 1.0)
    
    err_phi_current = 2 * math.acos(q0_current)   # in [rad]
    err_phi_prev = 2 * math.acos(q0_prev)   # in [rad]

    err_phi_current = err_phi_current * 180.0 / np.pi
    err_phi_prev = err_phi_prev * 180.0 / np.pi

    err_phi_delta = err_phi_current - err_phi_prev

    """ The lines below are kept, as they are used for archived reward functions. """
    # ang_vel_sat_x = state[4]
    # ang_vel_sat_y = state[5]
    # ang_vel_sat_z = state[6]
    # closest_koz_idx = np.argmin(state_koz[:, 0])
    # torque_1_prev = torque_prev[0]
    # torque_2_prev = torque_prev[1]
    # torque_3_prev = torque_prev[2]
    # koz_direction_closest = state_koz[closest_koz_idx, 1:4] # The direction of the closest KOZ in inertial frame
    # koz_margins_violated = [margin for margin in state_koz[:, 0] if margin <= 0]
    # ang_vel_norm = calc_vector_norm(np.array([ang_vel_sat_x, ang_vel_sat_y, ang_vel_sat_z]))
    # ang_vel = np.array([ang_vel_sat_x, ang_vel_sat_y, ang_vel_sat_z])
    # boresight_unit_vector = np.array([1.0, 0.0, 0.0]) # The boresight vector in body frame is along the x-axis
    # koz_margin_delta = koz_margin_min - koz_margin_min_prev

    r_total = 0
    USE_REWARD = "mod224ph2c2"

    if USE_REWARD == "mod224ph2c2":

        # Reward for reducing attitude error
        r_err = 0 
        # Phase 1
        if phase == "phase 1":
            r_err = -err_phi_delta
            
        # Phase 2
        else:      
            if koz_margin_min > 0.17:
                r_err = -err_phi_delta
            elif koz_margin_min > 0:
                r_err = -err_phi_delta * (koz_margin_min/0.17)
            else:
                r_err = 0
            
        # Bonus for high accuracy
        r_acc = 0.0
        if phase == "phase 1":
            # Bonus for desired accuracy
            r_acc = 0.06 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            
        elif phase == "phase 2":
            # Bonus for desired accuracy
            r_acc = 0.06 * np.exp(-((err_phi_current * 2.0) ** 2.0))
            

        # Penalty for using large torques
        r_torque = - 1.0*(abs(torque_1)+abs(torque_2)+abs(torque_3))

        # Penalty for entering / being close to keep out zone
        r_koz = 0.0
        if phase == "phase 2":
            # Maximum penalty inside of KOZ
            if koz_margin_min <= 0.0:
                r_koz = -1.0
            # Gradial penalty starting at 0.17 rad or 9.7 deg margin
            elif koz_margin_min <= 0.17:
                r_koz = -1.0 * (1.0 - koz_margin_min/0.17)
            # No penalty if farther away
            else:
                r_koz = 0.0

        r_total = r_err + r_acc + r_torque + r_koz

    return r_total


class BasiliskRWEnv(gym.Env):
    """
    A custom environment for the Basilisk RW simulation.
    """

    def __init__(self, initial_state=None, phase_type=None, discount_factor=0.99):
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

        self.PHASE = phase_type
        self.discount_factor = discount_factor # gamma

        # Custom metrics tracking for TensorBoard
        self.initial_error_angle = 0.0
        self.initial_angular_velocity_mag = 0.0
        self.episode_torques = []
        self.episode_torques_prev = []
        self.settled = False
        self.settling_time = None  # means not settled
        self.settling_threshold_deg = 1  # degrees for considering "settled"
        self.consecutive_settled_steps = 0

        self.x_axis = np.array([1, 0, 0], dtype=np.float64) # Boresight vector (body frame)

        # Set initial state (will be randomized in reset())
        self.reset()

    def _generate_quaternion_with_vector_angle(self, reference_vector: list, min_angle_deg: float, max_angle_deg: float) -> np.ndarray:
        """
        Generate a quaternion that rotates the reference_vector by an angle between 
        min_angle_deg and max_angle_deg in a random direction.
        
        Args:
            reference_vector (list): The vector to rotate (e.g., [1, 0, 0])
            min_angle_deg (float): Minimum angle (degrees) between original and rotated vector
            max_angle_deg (float): Maximum angle (degrees) between original and rotated vector
            
        Returns:
            quaternion (ndarray): A quaternion [w, x, y, z] that rotates reference_vector by the desired angle
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
    
    def _generate_keep_out_zones(self, active_koz_config: dict, initial_quaternion: np.ndarray, min_half_angle_deg: float, max_half_angle_deg: float) -> None:
        """
        Generates keep out zones defined by a normal vector and half-angle.

        Args:
            active_koz_config (dict): The dictionary containing the active KOZ normal vectors and half angles
                                of this episode.
            initial_quaternion (ndarray): The initial attitude quaternion of the satellite.
            min_half_angle_deg (float): Minimum half-angle of the keep out zone in degrees.
            max_half_angle_deg (float): Maximum half-angle of the keep out zone in degrees.
        """

        # Convert initial boresight quaternion to vector in inertial frame
        initial_vector_boresight_inertial = rotate_vector_by_quaternion(self.x_axis, initial_quaternion) #r_F inertial frame

        # First KOZ is always along shortest trajectory around center with some deviation
        # Calculate initial normal vector of keep out zone to be the bisector (middle between initial boresight and target boresight, same plane)
        bisector = normalize_vector(initial_vector_boresight_inertial + self.x_axis)

        # Calculate rotation axis such that rotated bisector keeps same angle to both boresight and initial attitude
        rotation_axis = normalize_vector(self.x_axis - initial_vector_boresight_inertial) 

        # Random half-angle between min and max
        half_angle_koz_1 = np.random.uniform(min_half_angle_deg, max_half_angle_deg) * np.pi / 180  # in radians

        # Rotate the bisector by a deviation angle using rodrigues formula
        # Deviation angle is randomly sampled between [-half_angle, half_angle]
        deviation_angle = np.random.uniform(-half_angle_koz_1, half_angle_koz_1)
        normal_vector_koz_1 = (bisector * np.cos(deviation_angle) + np.cross(rotation_axis, bisector) 
            * np.sin(deviation_angle) + rotation_axis*np.dot(rotation_axis,bisector) * (1-np.cos(deviation_angle)))

        # Add first KOZ
        active_koz_config["normal_vector"].append(normal_vector_koz_1)
        active_koz_config["half_angle_rad"].append(half_angle_koz_1)

        # If only 1 KOZ, return.
        if self.current_nr_koz == 1:
            return
        else:
            # Minimum margin angle between two KOZs
            #MARGIN = 20.0 * np.pi / 180.0
            MARGIN = 0 * np.pi / 180.0

            # For each additional KOZ, generate another KOZ
            for koz_nr in range(2, self.current_nr_koz+1):

                # Limit: max 3 KOZs
                if koz_nr > 3:
                    break

                # Random half angle
                half_angle_koz = np.random.uniform(min_half_angle_deg, max_half_angle_deg) * np.pi / 180  # in radians

                # Second KOZ is arranged left, third KOZ right to first KOZ.
                direction = 1 if koz_nr == 2 else -1

                # Rotate the bisector by a deviation angle using rodrigues formula
                # Deviation angle from first KOZ is chosen to guarantee min margin between KOZs and some deviation for new KOZ
                deviation_angle = half_angle_koz_1 + MARGIN + half_angle_koz + np.random.uniform(0, half_angle_koz)
                
                deviation_angle = deviation_angle * direction


                normal_vector_koz = (normal_vector_koz_1 * np.cos(deviation_angle) + np.cross(rotation_axis, normal_vector_koz_1) 
                    * np.sin(deviation_angle) + rotation_axis*np.dot(rotation_axis,normal_vector_koz_1) * (1-np.cos(deviation_angle)))
        
                # Add new KOZ
                active_koz_config["normal_vector"].append(normal_vector_koz)
                active_koz_config["half_angle_rad"].append(half_angle_koz)

    def _get_sat_state(self) -> np.ndarray:
        """
        Returns the current satellite state from Basilisk.

        Returns:
            sat_state (ndarray): The satellite state as a numpy array [quat, omega, omega_rw].
        """

        state = self.satellite.scStateOutMsg.read()
        state_rw = self.rw_effector.rwSpeedOutMsg.read()

        sigma = np.array(state.sigma_BN, dtype=np.float64)
        omega = np.array(state.omega_BN_B, dtype=np.float64)
        omega_rw = np.array(state_rw.wheelSpeeds[0:3], dtype=np.float64)

        quat = MRPToQuat(sigma)
        quat = np.array(quat, dtype=np.float64)

        return np.concatenate([quat, omega, omega_rw]).astype(np.float64)
    
    def _get_koz_state(self, quat: np.ndarray) -> np.ndarray:
        """
        Returns the current keep out zone state based on the quaternion.

        Args:
            quat (ndarray): The current attitude quaternion of the satellite
        Returns:
            koz_state (ndarray): The keep out zone state as a numpy array of shape (MAX_ZONES, KOZ_FEATURE_DIM).
        """

        inactive_features_state = np.array([np.pi, 1, 0, 0], dtype=np.float64)
        koz_state = np.tile(inactive_features_state, (self.MAX_ZONES,1)) # Initialize all KOZs as inactive

        if self.current_nr_koz > 0:

            for koz_idx in range(self.current_nr_koz):
                margin_koz = calc_margin_koz(quat, self.active_koz_config["normal_vector"][koz_idx], self.active_koz_config["half_angle_rad"][koz_idx])
                normal_vector_bf = rotate_vector_by_quaternion_to_body_frame(self.active_koz_config["normal_vector"][koz_idx], quat)
                direction_vector_bf = normal_vector_bf - self.x_axis
                direction_vector_bf = normalize_vector(direction_vector_bf)

                koz_state[koz_idx][0] = np.array(margin_koz, dtype=np.float64) 
                koz_state[koz_idx][1:4] = np.array(direction_vector_bf, dtype=np.float64)

        return koz_state
    
    def _get_koz_mask_state(self) -> np.ndarray:
        """
        Returns a mask indicating which keep out zones are active.

        Returns:
            mask (ndarray): A binary mask of shape (MAX_ZONES,) where 1 indicates an active KOZ and 0 indicates an inactive KOZ.
        """

        mask = np.zeros((self.MAX_ZONES,), dtype=np.float64) # Initialize all KOZs as inactive
        mask[:self.current_nr_koz] = 1 # Set active KOZs to 1

        return mask
    
    def _get_state(self) -> dict:
        """
        Returns the current combined state (satellite, KOZs, KOZ mask).

        Returns:
            state (dict): A dictionary containing the satellite state, KOZ state, and KOZ mask.
        """

        sat_state = self._get_sat_state()
        koz_state = self._get_koz_state(sat_state[:4])
        koz_mask_state = self._get_koz_mask_state()

        state = {
            "satellite": sat_state.astype(np.float64),
            "zones": koz_state.astype(np.float64),
            "zones_mask": koz_mask_state.astype(np.float64)
        }

        return state

    def _apply_action(self, action: np.ndarray) -> None:
        """
        Applies the action to the reaction wheels through Basilisk.

        Args:
            action (ndarray): The action to apply
        """

        wheel_motor_torque = (np.clip(action, -1.0, 1.0) * Constants.TORQUE_WHEEL_MAX)

        cmd_payload = messaging.ArrayMotorTorqueMsgPayload()
        cmd_payload.motorTorque = wheel_motor_torque.tolist()

        self.rw_cmd_msg.write(cmd_payload, self.sim.TotalSim.CurrentNanos)

    def _determine_termination(self) -> str:
        """
        Determines whether the episode should terminate based on the current state.

        Returns:
            termination (str): "failure" if the satellite entered a KOZ, "success" if the satellite has been settled for 10 seconds, or "" if the episode should continue.
        """

        # Failure on entering KOZ.
        if self.entered_koz_count > 0:
            return "failure"

        # Success on being settled for 10 seconds.
        elif self.consecutive_settled_steps >= (10.0/Constants.TIME_DELTA):
            return "success"

        else:
            return ""

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.episode_count += 1

        self.steps = 0
        self.sim_time = 0.0

        # Generate random initial attitude error
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

        # Sampled initial attitude error angle in degree
        initial_error_angle = 2 * math.acos(min(max(abs(q_array_initial[0]), 0.0), 1.0)) * 180 / np.pi  # degrees

        # Temporary hard coding: only generate KOZ when initial attitude error is at least 90 deg
        if initial_error_angle >= 90.0:
            self.current_nr_koz = np.random.randint(self.min_nr_koz, self.max_nr_koz+1) # Excludes upper bound
        else:
            self.current_nr_koz = 0

        self.active_koz_config = {
            "normal_vector": [],
            "half_angle_rad": []
        }

        # Initialize custom metrics for this episode
        self.initial_error_angle = initial_error_angle
        self.initial_angular_velocity_mag = np.linalg.norm(omega_initial) * 180 / np.pi  # deg/s
        self.episode_torques = []
        self.episode_torques_prev = []
        self.settled = False
        self.settling_time = None
        self.consecutive_settled_steps = 0
        self.min_margin_koz = np.pi
        self.entered_koz_count = 0

        if self.current_nr_koz > 0:
            # Generate keep out zones, vector in inertial frame (--> constant per episode), half angle in radians
            self._generate_keep_out_zones(self.active_koz_config, q_array_initial, self.min_half_angle_koz, self.max_half_angle_koz)

            for koz_idx in range(self.current_nr_koz):
                # Calculate margin angle to keep out zone
                margin_koz = calc_margin_koz(q_array_initial, self.active_koz_config["normal_vector"][koz_idx], self.active_koz_config["half_angle_rad"][koz_idx])

                # Update min margin koz angle
                if margin_koz < self.min_margin_koz:
                    self.min_margin_koz = margin_koz

                # Update entered koz count
                if margin_koz < 0.0:
                    self.entered_koz_count += 1

        sat_state = np.concatenate((q_array_initial, omega_initial, wheel_velocities_initial))
        koz_state = self._get_koz_state(q_array_initial)
        koz_mask_state = self._get_koz_mask_state()
        self.state = {
            "satellite": sat_state.astype(np.float64),
            "zones": koz_state.astype(np.float64),
            "zones_mask": koz_mask_state.astype(np.float64)
        }

        self.torque_prev = np.zeros(3, dtype=np.float64)

        # Copy state into observation
        obs = copy.deepcopy(self.state)

        self.satellite = build_spacecraft(q_array_initial, omega_initial)
        self.sim, self.rw_cmd_msg, self.rw_effector = build_basilisk_sim(wheel_velocities_initial, self.satellite, Constants.TIME_DELTA)
        self.sim.InitializeSimulation()

        return obs, {}

    def step(self, action):
        q0_prev = self.state["satellite"][0]
        koz_margin_min_prev = np.min(self.state["zones"][:,0])

        self._apply_action(action)

        self.sim_time += self.dt
        self.sim.ConfigureStopTime(macros.sec2nano(self.sim_time))
        self.sim.ExecuteSimulation()

        self.state = self._get_state()

        # Update KOZ metrics
        if self.current_nr_koz > 0:
            for koz_idx in range(self.current_nr_koz):
                # Calculate margin angle to keep out zone
                margin_koz = self.state["zones"][koz_idx][0]

                # Update min margin koz angle
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

        # Save previous settled state.
        settled_prev = self.settled

        # From unsettled to settled
        if not self.settled and is_within_accuracy:
            self.settled = True
            self.settling_time = self.steps * self.dt

        # From settled to unsettled
        elif self.settled and not is_within_accuracy:
            self.settled = False
            self.settling_time = None
            self.consecutive_settled_steps = 0

        # If it remained settled, increment counter.
        if settled_prev and self.settled:
            self.consecutive_settled_steps += 1

        """ Line below is kept here for archived reward functions, which may use early termination. """
        #termination = self._determine_termination()
        termination = "" # Never terminate early for this reward function.

        reward = reward_function(self.state["satellite"], q0_prev, action * Constants.TORQUE_WHEEL_MAX, self.torque_prev, self.PHASE, self.state["zones"], self.entered_koz_count, self.steps*self.dt, termination, self.discount_factor, koz_margin_min_prev)

        # Update previous torque for the next step
        self.torque_prev = action * Constants.TORQUE_WHEEL_MAX  

        self.steps += 1

        terminated = termination == "failure" or termination == "success"
        truncated = not terminated and self.steps >= self.max_steps

        info = {}

        if terminated or truncated:
            final_error_angle = current_error_deg
            avg_torque = np.mean(self.episode_torques) if self.episode_torques else 0.0
            max_torque = np.max(self.episode_torques) if self.episode_torques else 0.0
            max_torque_prev = np.max(self.episode_torques_prev) if self.episode_torques else 0.0
            min_margin_koz = self.min_margin_koz * 180 / np.pi  # convert to degrees
            
            info.update({
                "termination": termination,
                "truncated": truncated,
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

    def close(self):
        pass


class LSTM(BaseFeaturesExtractor):
    """
    Top level LSTM class, which inherits BaseFeaturesExtractor from SB3.
    Processes the KOZ observations, processes the LSTM sequence and returns the concatenated state to SAC network.
    """

    def __init__(self, observation_space: spaces.Dict, sat_obs_dim: int, lstm_out_dim: int):
        self.sat_obs_dim = sat_obs_dim
        self.lstm_out_dim = lstm_out_dim
        self.USE_LSTM = True
        self.KOZ_FEATURE_SCALING = (np.pi, 1.0, 1.0, 1.0)

        # Get the maximum number of KOZs and the LSTM input dimension (= feature dimension of a KOZ)
        zones_max, lstm_in_dim = observation_space["zones"].shape
        self.zones_max = zones_max
        self.lstm_in_dim = lstm_in_dim

        # The output size must match the representation returned by forward().
        self.total_obs_dim = (
            sat_obs_dim + lstm_out_dim
            if self.USE_LSTM
            else sat_obs_dim + zones_max * lstm_in_dim
        )
        super().__init__(observation_space=observation_space, features_dim=self.total_obs_dim)

        # The LSTM 
        self.lstm = th.nn.LSTM(
            input_size=lstm_in_dim,
            hidden_size=lstm_out_dim,
            num_layers=1,
            batch_first=True # Expects batch dimension first
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

        if self.USE_LSTM:

            # Container for LSTM output of the entire batch
            self.lstm_out = th.zeros(batch_size, self.lstm_out_dim, device=device)

            # Number of batch obsverations with at least one KOZ
            non_zero_obs_count = non_zero_indices.numel()

            # If there is at least one obs with KOZs, process the LSTM
            if non_zero_obs_count > 0:

                # Get all observations (and their KOZ count) with at least one KOZ from the batch
                non_zero_zones = zones_obs[non_zero_indices]
                non_zero_zones_count = zones_count[non_zero_indices]
                zones_mask_filtered = zones_mask[non_zero_indices] # Removes observations from zones mask with no KOZ

                # Manually normalize KOZ observation.
                non_zero_zones = non_zero_zones / non_zero_zones.new_tensor(self.KOZ_FEATURE_SCALING)

                # Sort KOZs by margin, such that the zone with smallest margin is processed last by the LSTM.
                koz_margins = non_zero_zones[:,:,0] # Margin is index 0
                koz_margins_mod = koz_margins.masked_fill(~zones_mask_filtered.bool(), float("-inf")) # Replace inactive KOZ margins values with -inf for sorting, so that they end up last and cut off by pack_padded_sequence.
                sort_indices = th.argsort(koz_margins_mod, dim=1, descending=True)
                gather_index = sort_indices.unsqueeze(-1).expand(-1, -1, non_zero_zones.shape[-1]) # Transform index dimension from (batch,4) to (batch,4,4) to also sort other KOZ features besides margin.
                non_zero_zones = th.gather(non_zero_zones, dim=1, index=gather_index) # Apply sorting to unmodified obs.

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

        # Dont use LSTM, just use (padded) KOZ observation as is, and flatten it to a vector.
        else:
            # Manually normalize KOZ observation.
            zones_obs = zones_obs / zones_obs.new_tensor(self.KOZ_FEATURE_SCALING)

            # Flatten the KOZ observation to a vector, and store it as the LSTM output.
            self.lstm_out = zones_obs.view(batch_size, -1)


        # Combine satellite obs and LSTM output
        combined = th.cat([sat_obs, self.lstm_out], dim=1)

        """ Uncomment lines below for debugging LSTM output, weights and biases. """
        # for param_name, param in self.lstm.named_parameters():
        #     # Get the parameter tensor
        #     param_value = param.detach().cpu()

        #     # Separate weights and biases per gate and convert from tensor --> list.
        #     input_values = param_value[0:self.lstm_out_dim].tolist()
        #     print(f"{param_name}: {input_values}")

        # if self.timestep < 500:
        #     print(f"{self.timestep}: {self.lstm_out}")

        #self.timestep += 1

        return combined