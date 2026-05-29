"""
The training environment for the satellite reorientation task.
Includes the reward function for the agent.

Author: Cemal Yilmaz - 2026
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for frame rendering

import math
import sys
import os
import warnings

from numba import njit
from scipy.spatial.transform import Rotation
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeRW, unitTestSupport
from Basilisk.simulation import spacecraft, reactionWheelStateEffector
from Basilisk.architecture import messaging

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
def rotate_vector_by_quaternion(v, q):
    """
    Rotate a vector v by a quaternion q.
    Args:
        v: Input vector as a numpy array [x, y, z].
        q: Quaternion representing the rotation as a numpy array [w, x, y, z].
    Returns:
        v_rotated: The rotated vector as a numpy array [x, y, z].
    """
    v = v.astype(np.float32)
    w, x, y, z = q

    # Convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

    return R @ v

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
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
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

def reward_function(state, _q0_prev, torque, torque_prev, phase):
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
    margin_koz = state[10]
    
    # Clamp q0 values to [-1, 1] to prevent acos() domain errors (NaN) with large torques
    # Using min/max instead of np.clip for numba compatibility with scalars
    q0_current = min(max(q0_current, -1.0), 1.0)
    q0_prev = min(max(q0_prev, -1.0), 1.0)
    
    err_phi_current = 2 * math.acos(q0_current)   # in [rad]
    err_phi_prev = 2 * math.acos(q0_prev)   # in [rad]

    err_phi_current = err_phi_current * 180.0 / np.pi
    err_phi_prev = err_phi_prev * 180.0 / np.pi

    r_total = 0
    USE_REWARD = "prak"
    
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

        r_total = r1 + r3 + r4

    # Penalty for entering / being close to keep out zone
    r5 = 0.0
    if phase == 2:
        if margin_koz <= 0.0:
            r5 = -1.0
        else:
            r5 = -1.0*math.exp(-66.0*margin_koz)

    return r_total + r5


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

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(3,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),   # [quat(4), omega_BN_B(3), wheel_speed(3), margin_koz(1)]
            dtype=np.float32,
        )

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
        else:
            self.min_initial_angle = initial_state[0]
            self.max_initial_angle = initial_state[1]
            self.min_initial_angular_velocity = initial_state[2]
            self.max_initial_angular_velocity = initial_state[3]
            self.max_steps = initial_state[4]
            self.min_half_angle_koz = initial_state[5]
            self.max_half_angle_koz = initial_state[6]

        if self.max_half_angle_koz > 0.0:
            self.PHASE = 2
        else:
            self.PHASE = 1

        # Custom metrics tracking for TensorBoard
        self.initial_error_angle = 0.0
        self.initial_angular_velocity_mag = 0.0
        self.episode_torques = []
        self.episode_torques_prev = []
        self.settled = False
        self.settling_time = None  # means not settled
        self.settling_threshold_deg = 0.25  # degrees for considering "settled"
        self.settling_velocity_threshold = 0.01  # rad/s for angular velocity
        self.min_margin_koz = 0.0
        self.entered_koz_count = 0

        self.x_axis = np.array([1, 0, 0]) # For frame rendering

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
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Normalize the reference vector
        ref_vec = np.array(reference_vector, dtype=np.float32)
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
        
        quaternion = np.array([q0, q_vec[0], q_vec[1], q_vec[2]], dtype=np.float32)
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

        return normal_vector_koz, half_angle_koz

    def _build_spacecraft(self, q_init, omega_init):
        self.satellite = spacecraft.Spacecraft()
        self.satellite.ModelTag = "satellite"

        # Hub inertia [kg m^2]
        inertia = [0.02 / 3,  0.,         0.,
                    0.,        0.1256 / 3, 0.,
                    0.,        0.,         0.1256 / 3]
        self.satellite.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(inertia)

        self.satellite.hub.mHub = 4.0 # TODO clarify if needed
        self.satellite.hub.r_BcB_B = [[0.0], [0.0], [0.0]] # position vector of body-fixed point B relative to center of mass

        sigma_init = quatToMRP(q_init)

        # Basilisk attitude state uses MRPs, not quaternions
        self.satellite.hub.sigma_BNInit = [[sigma_init[0]], [sigma_init[1]], [sigma_init[2]]]
        self.satellite.hub.omega_BN_BInit = [[omega_init[0]], [omega_init[1]], [omega_init[2]]]

    def _get_state(self):
        state = self.satellite.scStateOutMsg.read()
        state_rw = self.rw_effector.rwSpeedOutMsg.read()

        sigma = np.array(state.sigma_BN, dtype=np.float32)
        omega = np.array(state.omega_BN_B, dtype=np.float32)
        omega_rw = np.array(state_rw.wheelSpeeds[0:3], dtype=np.float32)

        quat = MRPToQuat(sigma)
        quat = np.array(quat, dtype=np.float32)

        margin_koz = calc_margin_koz(quat, self.normal_vector_koz, self.half_angle_koz)

        return np.concatenate([quat, omega, omega_rw, np.array([margin_koz])]).astype(np.float32)

    def _apply_action(self, action):
        wheel_motor_torque = (
            np.clip(action, -1.0, 1.0) * Constants.TORQUE_WHEEL_MAX
        )

        cmd_payload = messaging.ArrayMotorTorqueMsgPayload()
        cmd_payload.motorTorque = wheel_motor_torque.tolist()

        self.rw_cmd_msg.write(cmd_payload, self.sim.TotalSim.CurrentNanos)

    def _build_basilisk_sim(self, q_init, omega_init, omega_wheel_init):
        self.sim = SimulationBaseClass.SimBaseClass()

        process = self.sim.CreateNewProcess("dynProcess")
        task_name = "dynTask"
        process.addTask(
            self.sim.CreateNewTask(task_name, macros.sec2nano(self.dt))
        )

        self._build_spacecraft(q_init, omega_init)

        self.rw_effector = reactionWheelStateEffector.ReactionWheelStateEffector()
        self.rw_effector.ModelTag = "reactionWheels"

        rw_factory = simIncludeRW.rwFactory()
        varRWModel = messaging.BalancedWheels
        #maxMomentum = 0.00001722*scale_angular_velocity_wheels*2*np.pi / 60.0
        #maxMomentum = 0.01 * 1e9
        inertia_wheel = Constants.INERTIA_WHEEL

        # Three orthogonal wheels
        RW1 = rw_factory.create(
            "custom",
            [1.0, 0.0, 0.0],
            Omega=float(omega_wheel_init[0]),
            u_max=Constants.TORQUE_WHEEL_MAX,
            Omega_max=Constants.SPEED_WHEEL_MAX,
            maxMomentum=Constants.MOMENTUM_WHEEL_MAX,
            #Js= inertia_wheel,
            RWModel=varRWModel
        )
        RW2 = rw_factory.create(
            "custom",
            [0.0, 1.0, 0.0],
            Omega=float(omega_wheel_init[1]),
            u_max=Constants.TORQUE_WHEEL_MAX,
            Omega_max=Constants.SPEED_WHEEL_MAX,
            maxMomentum=Constants.MOMENTUM_WHEEL_MAX,
            #Js= inertia_wheel,
            RWModel=varRWModel
        )
        RW3 = rw_factory.create(
            "custom",
            [0.0, 0.0, 1.0],
            Omega=float(omega_wheel_init[2]),
            u_max=Constants.TORQUE_WHEEL_MAX,
            Omega_max=Constants.SPEED_WHEEL_MAX,
            maxMomentum=Constants.MOMENTUM_WHEEL_MAX,
            #Js= inertia_wheel,
            RWModel=varRWModel
        )

        rw_factory.addToSpacecraft(
            self.satellite.ModelTag,
            self.rw_effector,
            self.satellite,
        )

        # Stand-alone RW motor torque command message
        cmd_payload = messaging.ArrayMotorTorqueMsgPayload()
        cmd_payload.motorTorque = [0.0, 0.0, 0.0]

        self.rw_cmd_msg = messaging.ArrayMotorTorqueMsg().write(cmd_payload)
        self.rw_effector.rwMotorCmdInMsg.subscribeTo(self.rw_cmd_msg)

        # Add modules to task.
        # RW effector before spacecraft dynamics.
        self.sim.AddModelToTask(task_name, self.rw_effector, 2)
        self.sim.AddModelToTask(task_name, self.satellite, 1)

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
            omega_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            omega_direction = omega_direction / omega_direction_norm
        
        # Scale direction by magnitude
        omega_initial = (omega_magnitude * omega_direction).astype(np.float32)

        wheel_velocities_initial = np.zeros(3, dtype=np.float32)

        # Generate keep out zone, vector in inertial frame (--> constant per episode), half angle in radians
        self.normal_vector_koz, self.half_angle_koz = self._generate_keep_out_zone(q_array_initial, self.min_half_angle_koz, self.max_half_angle_koz)
        
        # Calculate margin angle to keep out zone
        margin_koz = calc_margin_koz(q_array_initial, self.normal_vector_koz, self.half_angle_koz)

        self.state = np.concatenate((q_array_initial, omega_initial, wheel_velocities_initial, np.array([margin_koz])))

        self.torque_prev = np.zeros(3, dtype=np.float32)

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

        # Normalize observation
        obs = self.state.copy()
        #obs[4:7] = obs[4:7] / scale_angular_velocity_sat  # Normalize satellite angular velocity
        #obs[7:10] = obs[7:10] / scale_angular_velocity_wheels  # Normalize RW speeds
        #obs[10] = obs[10] / scale_margin_koz  # Normalize margin to keep out zone
        obs = obs.astype(np.float32)

        self._build_basilisk_sim(q_array_initial, omega_initial, wheel_velocities_initial)
        self.sim.InitializeSimulation()

        return obs, {}

    def step(self, action):
        q0_prev = self.state[0]


        self._apply_action(action)

        self.sim_time += self.dt
        self.sim.ConfigureStopTime(macros.sec2nano(self.sim_time))
        self.sim.ExecuteSimulation()

        self.state = self._get_state()
        
        reward = reward_function(self.state, q0_prev, action * Constants.TORQUE_WHEEL_MAX, self.torque_prev, self.PHASE)

        # Normalize observation
        obs = self.state.copy()
        #obs[4:7] = obs[4:7] / scale_angular_velocity_sat  # Normalize satellite angular velocity
        #obs[7:10] = obs[7:10] / scale_angular_velocity_wheels  # Normalize RW speeds
        #obs[10] = obs[10] / scale_margin_koz  # Normalize margin to keep out zone
        obs = obs.astype(np.float32)

        self.episode_torques.append(np.linalg.norm(action * Constants.TORQUE_WHEEL_MAX))
        self.episode_torques_prev.append(np.linalg.norm(self.torque_prev))

        # Check settling condition
        current_error_deg = 2 * math.acos(min(max(abs(self.state[0]), 0.0), 1.0)) * 180 / np.pi
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
        attitude = self.state[:4]
        omega = self.state[4:7]*scale_angular_velocity_sat
        torque = self.state[14:17]*scale_torque

        if self.render_mode == "human":
            print(f"Step: {self.steps}, Attitude: {attitude}, Omega: {omega}, Torque: {torque}")
            return

        if self.render_mode == "rgb_array":
            q = self.state[:4]

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