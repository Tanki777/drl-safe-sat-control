"""
Visualization tools.

Author: Cemal Yilmaz - 2026
"""

import os
import sys
import subprocess
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datetime
import visualtorch
import torch as th
from pathlib import Path
from Basilisk.utilities import SimulationBaseClass, macros, vizSupport
from Basilisk.simulation import spacecraft, vizInterface, simpleInstrument
from Basilisk.architecture import sysModel, messaging

# Add parent directory to path for imports (must be before local imports)
_drl_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _drl_repo_dir not in sys.path:
    sys.path.insert(0, _drl_repo_dir)

parent_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(parent_dir)
repo_parent_dir = os.path.dirname(repo_dir)
video_dir = os.path.join(repo_parent_dir, "videos")
viz_dir = os.path.join(repo_parent_dir, "viz_files")

# Create video directory if it doesn't exist
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

# Create viz directory if it does not exist
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)


from agent_training.constants import Constants
from agent_training.environment import quatToMRP
from agent_simulation.evaluation import create_evaluation_env, load_agent, load_evaluation_data, simulate_episode
from config.config import Config


def _rotation_matrix_from_quaternion(quaternion):
    """Return the body-to-inertial rotation matrix for a [w, x, y, z] quaternion."""
    q0, q1, q2, q3 = quaternion

    return np.array([
        [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q3 * q0), 2 * (q1 * q3 + q2 * q0)],
        [2 * (q1 * q2 + q3 * q0), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q1 * q0)],
        [2 * (q1 * q3 - q2 * q0), 2 * (q2 * q3 + q1 * q0), 1 - 2 * (q1 * q1 + q2 * q2)],
    ])


def _normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        return vector
    
    return vector / norm




class EpisodeReplayPublisher(sysModel.SysModel):
    """
    Publish previously recorded spacecraft and reaction-wheel states as
    Basilisk messages.
    """

    def __init__(
        self,
        times: list,
        quaternions: list,
        angular_velocities: list,
        wheel_torques: list,
        wheel_speeds: list
    ):
        super().__init__()

        self.ModelTag = "episodeReplayPublisher"

        self.times = times
        self.quaternions = quaternions
        self.angular_velocities = angular_velocities
        self.wheel_torques = wheel_torques
        self.wheel_speeds = wheel_speeds

        self.num_samples = len(self.times)
        self.num_wheels = self.wheel_torques.shape[1]


        # Fixed location at origin as we only care about attitude.
        self.positions = np.tile(
            np.array([0.0, 0.0, 0.0]),
            (self.num_samples, 1),
        )

        # No translational velocity as we only care about attitude.
        self.translational_velocities = np.zeros(
            (self.num_samples, 3),
            dtype=np.float64,
        )

        # Satellite state message
        self.sc_state_out_msg = messaging.SCStatesMsg()
        self.body_x_axis_marker_out_msg = messaging.SCStatesMsg()

        # Reaction wheel state messages
        self.rw_state_out_msgs = [
            messaging.RWConfigLogMsg()
            for _ in range(self.num_wheels)
        ]

        self.sample_index = 0

    def Reset(self, current_sim_nanos):
        self.sample_index = 0

    def UpdateState(self, current_sim_nanos):
        index = min(self.sample_index, self.num_samples - 1)

        q = self.quaternions[index].copy()
        
        # Convert to MRP since that is what Basilisk uses for attitude.
        sigma_BN = quatToMRP(q)

        # The message payload which contains the next satellite state
        sc_payload_msg = messaging.SCStatesMsgPayload()
        sc_payload_msg.r_BN_N = self.positions[index].tolist()
        sc_payload_msg.v_BN_N = self.translational_velocities[index].tolist()
        sc_payload_msg.sigma_BN = np.asarray(sigma_BN).tolist()
        sc_payload_msg.omega_BN_B = self.angular_velocities[index].tolist()

        # Forward the next satellite state to Basilisk simulation.
        self.sc_state_out_msg.write(
            sc_payload_msg,
            current_sim_nanos
        )

        # For each reaction wheel, add the next state to the payload message and forward to Basilisk simulation.
        for wheel_index, rw_msg in enumerate(self.rw_state_out_msgs):
            rw_payload_msg = messaging.RWConfigLogMsgPayload()

            rw_payload_msg.Omega = float(
                self.wheel_speeds[index, wheel_index]
            )
            rw_payload_msg.u_current = float(
                self.wheel_torques[index, wheel_index]
            )

            rw_msg.write(
                rw_payload_msg,
                current_sim_nanos
            )

        # Increase the sample index if not end of episode.
        if self.sample_index < self.num_samples - 1:
            self.sample_index += 1


def create_koz(koz_normal_vector, koz_half_angle, koz_name, koz_idx, viz):

    # Color picker
    koz_colors = ["red", "green", "blue", "yellow"]

    koz_distance = 1*10**10 # Very far away to simulate fixed point in world frame
    koz_payload_msg = messaging.SCStatesMsgPayload()
    koz_payload_msg.r_BN_N = (koz_distance * koz_normal_vector).tolist()
    koz_payload_msg.sigma_BN = [0,0,0]
    koz_msg = messaging.SCStatesMsg()
    koz_msg.write(koz_payload_msg)

    koz_sc_data = vizInterface.VizSpacecraftData()
    koz_sc_data.spacecraftName = koz_name
    koz_sc_data.scStateInMsg.subscribeTo(koz_msg)

    # Use a simple sphere model for the KOZ object
    vizSupport.createCustomModel(
        viz,
        modelPath="SPHERE",
        simBodiesToModify=[koz_name],
        scale=[0.01, 0.01, 0.01]
    )

    # Create Cone attached to satellite
    vizSupport.createConeInOut(viz, toBodyName=koz_name, coneColor=vizSupport.toRGBA255(koz_colors[koz_idx], alpha=0.1),
                               isKeepIn=False, normalVector_B=[1,0,0], incidenceAngle=koz_half_angle,
                               coneHeight=0.25, fromBodyName="satellite")
    
    # Create line from satellite to KOZ normal vector
    vizSupport.createPointLine(viz, toBodyName=koz_name, lineColor=vizSupport.toRGBA255(koz_colors[koz_idx], alpha=0.1), fromBodyName="satellite")

    # Need to return koz_msg to prevent being deleted.
    return koz_sc_data, koz_msg


def create_axis(vector: np.ndarray, name, color, viz):
    distance = 1*10**10 # Very far away to simulate fixed point in world frame
    payload_msg = messaging.SCStatesMsgPayload()
    payload_msg.r_BN_N = (distance * vector).tolist()
    payload_msg.sigma_BN = [0,0,0]
    msg = messaging.SCStatesMsg()
    msg.write(payload_msg)

    sc_data = vizInterface.VizSpacecraftData()
    sc_data.spacecraftName = name
    sc_data.scStateInMsg.subscribeTo(msg)

    # Use a simple sphere model for the KOZ object
    vizSupport.createCustomModel(
        viz,
        modelPath="SPHERE",
        simBodiesToModify=[name],
        scale=[0.01, 0.01, 0.01]
    )
    
    # Create line from satellite to target body
    vizSupport.createPointLine(viz, toBodyName=name, lineColor=vizSupport.toRGBA255(color, alpha=0.1), fromBodyName="satellite")

    # Need to return msg to prevent being deleted.
    return sc_data, msg


def create_satellite_axis(name, vector: np.ndarray):
    sensor = vizInterface.GenericSensor()
    sensor.r_SB_B = [0,0,0]
    sensor.fieldOfView = vizInterface.DoubleVector([0.01])
    sensor.normalVector = vector
    sensor.color = vizInterface.IntVector(vizSupport.toRGBA255("blue"))
    sensor.label = name
    sensor.size = 0.25

    return sensor


def save_episode_as_viz(
    output_file: str,
    episode_data: dict,
    save_file: bool,
    vizard_exe: str
):
    
    if not save_file:
        return
    
    output_path = Path(output_file).resolve()
  
    # Create Basilisk simulation and process
    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess("replayProcess")
    task_name = "replayTask"

    # Add replay task
    process.addTask(
        sim.CreateNewTask(
            task_name,
            macros.sec2nano(Constants.TIME_DELTA) # task rate
        )
    )

    # Get the episode data as a model
    replay = EpisodeReplayPublisher(
        times=episode_data["times"],
        quaternions=episode_data["quaternion"],
        angular_velocities=episode_data["omega"],
        wheel_torques=episode_data["torques"],
        wheel_speeds=episode_data["omega_wheels"],
    )

    # Add replay data model to task
    sim.AddModelToTask(task_name, replay)

    # Dummy satellite object for properly creating viz interface
    sat_dummy = spacecraft.Spacecraft()
    sat_dummy.ModelTag = "satellite"

    # Create the vizard interface
    viz: vizInterface.VizInterface = vizSupport.enableUnityVisualization(sim, task_name, sat_dummy, saveFile=str(output_path))
    viz.scData.clear() # Clear dummy satellite state

    # Create dummy sensors for satellite axes
    sat_axis_x = create_satellite_axis("sat_axis_x", np.array([1,0,0]))
    sat_axis_y = create_satellite_axis("sat_axis_y", np.array([0,1,0]))
    sat_axis_z = create_satellite_axis("sat_axis_z", np.array([0,0,1]))

    # Configure and populate the vizard satellite data
    sc_data = vizInterface.VizSpacecraftData()
    sc_data.spacecraftName = "satellite"

    sc_data.scStateInMsg.subscribeTo( # satellite state
        replay.sc_state_out_msg
    )

    sc_data.rwInMsgs = messaging.RWConfigLogMsgInMsgsVector( # reaction wheel states
        [rw_msg.addSubscriber() for rw_msg in replay.rw_state_out_msgs]
    )

    sc_data.modelDictionaryKey = "3Usat" # use 3U CubeSat model
    sc_data.genericSensorList = vizInterface.GenericSensorVector([sat_axis_x, sat_axis_y, sat_axis_z])

    # Create KOZ. Need to return koz_msg to prevent being deleted.
    koz1_sc_data, koz1_msg = create_koz(episode_data["normal_vector_koz"], episode_data["half_angle_koz"], "koz1", 0, viz)
    #koz2_sc_data, koz2_msg = create_koz(np.array([1,0,0]), 0.1, "koz2", 1, viz)
    target_x_sc_data, target_x_msg = create_axis(np.array([1,0,0]), "target_x", "white", viz)
    target_y_sc_data, target_y_msg = create_axis(np.array([0,1,0]), "target_y", "teal", viz)
    target_z_sc_data, target_z_msg = create_axis(np.array([0,0,1]), "target_z", "teal", viz)

    # Add the satellite data to vizard
    viz.scData.push_back(sc_data)
    viz.scData.push_back(koz1_sc_data)
    #viz.scData.push_back(koz2_sc_data)
    viz.scData.push_back(target_x_sc_data)
    viz.scData.push_back(target_y_sc_data)
    viz.scData.push_back(target_z_sc_data)

    # Vizard settings
    settings: vizInterface.VizSettings = viz.settings

    settings.spacecraftCSon = -1
    settings.showCSLabels = -1
   
    sim.InitializeSimulation()

    stop_time = float(episode_data["times"][-1])

    # Add a small fraction of dt so that the final sample is processed.
    sim.ConfigureStopTime(
        macros.sec2nano(stop_time + 0.5 * Constants.TIME_DELTA)
    )

    sim.ExecuteSimulation()

    # Start Vizard and load the file
    subprocess.Popen([vizard_exe, "--args", "-loadFile", output_path])


def print_result(phi_final, omega_final, cumulative_reward_final):
    """
    Print the final results of the evaluation in a readable format.
    """
    print(f"Final rotation angle: {phi_final:.6f}°")
    print(f"Final angular velocity: {np.sqrt(omega_final[0]**2 + omega_final[1]**2 + omega_final[2]**2)*180/np.pi:.6f} deg/s")
    print(f"Total cumulative reward: {cumulative_reward_final:.2f}")
    print(f"Target accuracy: 0.25°")
    print(f"Attitude control: {"SUCCESS" if phi_final < 0.25 else "NOT CONVERGED"}")
    print(f"Velocity settling: {"SUCCESS" if np.sqrt(omega_final[0]**2 + omega_final[1]**2 + omega_final[2]**2)*180/np.pi < 0.5 else "NOT SETTLED"}")


def quat_to_axis_angle(q):
        """
        Convert quaternion to rotation axis and angle.
        Args:
            q: Quaternion as a list or array [q0, q1, q2, q3]
        Returns:
            res: A tuple containing:
            axis: Rotation axis as a numpy array [x, y, z]
            angle: Rotation angle in radians
        """
        q0, q1, q2, q3 = q
        
        # Angle phi = 2 * arccos(|q0|) (same as in reward function)
        angle = 2 * np.arccos(np.abs(q0))
        
        # Rotation axis = q_vec / |q_vec| (normalized quaternion vector part)
        q_vec_norm = np.sqrt(q1**2 + q2**2 + q3**2)
        if q_vec_norm > 1e-6:  # Avoid division by zero
            axis = np.array([q1, q2, q3]) / q_vec_norm
        else:
            axis = np.array([0, 0, 0])  # No rotation axis if angle is zero
            
        return axis, angle


def plot_actual_attitude(simulation_data: dict):
    """
    Plot the satellite attitude trajectory based on rotation axis and angle phi.
    Args:
        simulation_data: A dictionary containing the simulation data for plotting.
    """
    
    # Switch to interactive backend for 3D plots
    matplotlib.use("TkAgg")

    # Parse quaternion and angular velocity components
    q_0, q_1, q_2, q_3 = simulation_data["quaternion"][:, 0], simulation_data["quaternion"][:, 1], simulation_data["quaternion"][:, 2], simulation_data["quaternion"][:, 3]
    omega_x, omega_y, omega_z = simulation_data["omega"][:, 0], simulation_data["omega"][:, 1], simulation_data["omega"][:, 2]
    omega_w_x, omega_w_y, omega_w_z = simulation_data["omega_wheels"][:, 0], simulation_data["omega_wheels"][:, 1], simulation_data["omega_wheels"][:, 2]
    norm_q = simulation_data["quaternion_norm"]
    torques_array = simulation_data["torques"]
    rewards_array = simulation_data["rewards"]
    cumulative_rewards = simulation_data["cumulative_rewards"]
    times = simulation_data["times"]
    normal_vector_koz = simulation_data["normal_vector_koz"] # normal vector in world frame
    half_angle_koz = simulation_data["half_angle_koz"]
    margin_angles_koz = simulation_data["margin_angles_koz"]
    direction_koz = simulation_data["direction_koz"] # normal vector in body frame
    min_margin_koz = simulation_data["min_margin_koz"]
    cnt_Koz_violations = simulation_data["cnt_Koz_violations"]
    lstm_output = simulation_data["lstm_output"]


    print("Minimum margin KOZ:", min_margin_koz*180/np.pi, "degrees")
    print("Count KOZ violations:", cnt_Koz_violations)
    print("Half angle KOZ:", half_angle_koz*180/np.pi, "degrees")
    

    # Extract rotation axes and angles for all time points
    rotation_axes = []
    rotation_angles = []
    
    for i in range(len(q_0)):
        axis, angle = quat_to_axis_angle([q_0[i], q_1[i], q_2[i], q_3[i]])
        rotation_axes.append(axis)
        rotation_angles.append(angle)
    
    # Convert to numpy arrays
    rotation_axes = np.array(rotation_axes)  # Shape: (N, 3)
    rotation_angles = np.array(rotation_angles)  # Shape: (N,)
    
    # Convert to degrees
    rotation_angles_deg = rotation_angles * 180 / np.pi

    # Calculate the body X-axis direction (boresight) at each time point using the quaternion rotation
    body_axis_arr = []
    for i in range(len(q_0)):
        q = [q_0[i], q_1[i], q_2[i], q_3[i]]
        w, x, y, z = q
        R = np.array([
                    [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
                    [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
                    [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
                ])
        body_axis = R @ np.array([1, 0, 0])  # body X-axis
        body_axis_arr.append(body_axis)
    
    # Convert to numpy array for proper indexing
    body_axis_arr = np.array(body_axis_arr)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 3D Rotation Axis Trajectory (This is the key trajectory for phi angle!)
    ax1 = fig.add_subplot(341, projection="3d")
    
    # Plot trajectory on unit sphere (rotation axes are unit vectors)
    ax1.plot(body_axis_arr[:, 0], body_axis_arr[:, 1], body_axis_arr[:, 2], "b-", alpha=0.7, linewidth=3, label="Boresight Axis Trajectory")
    ax1.scatter(body_axis_arr[0, 0], body_axis_arr[0, 1], body_axis_arr[0, 2], color="green", s=100, label="Start")
    ax1.scatter(body_axis_arr[-1, 0], body_axis_arr[-1, 1], body_axis_arr[-1, 2], color="red", s=100, label="End")
    ax1.scatter(1, 0, 0, color="gold", s=150, marker="*", label="Target")

    def _generate_keep_out_zone_circle():
        # Create circle points for the keep out zone

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points = []

        for angle in theta:
            # Generate points on the circle in the plane perpendicular to the normal vector
            v = np.array([np.cos(angle), np.sin(angle), 0])

            # Rotate v to be perpendicular to koz_normal
            if np.allclose(normal_vector_koz, [0, 0, 1]):
                rot_axis = np.array([1, 0, 0])
            else:
                rot_axis = np.cross([0, 0, 1], normal_vector_koz)
                rot_axis /= np.linalg.norm(rot_axis)

            angle_to_rotate = np.arccos(np.dot(normal_vector_koz, [0, 0, 1]))

            # Rodrigues' rotation formula
            v_rotated = (v * np.cos(angle_to_rotate) +
                        np.cross(rot_axis, v) * np.sin(angle_to_rotate) +
                        rot_axis * np.dot(rot_axis, v) * (1 - np.cos(angle_to_rotate)))
            
            # Scale to the radius of the keep out zone circle
            radius = np.sin(half_angle_koz)
            circle_point = normal_vector_koz * np.cos(half_angle_koz) + v_rotated * radius
            circle_points.append(circle_point)

        return circle_points

    # Plot keep out zone as a ring on the unit sphere
    if normal_vector_koz is not None and half_angle_koz is not None:
        circle_points = _generate_keep_out_zone_circle()
        circle_points = np.array(circle_points)
        ax1.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], "orange", linewidth=2, label="Keep Out Zone")
    
    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color="gray")
    
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-1.1, 1.1])
    ax1.set_zlim([-1.1, 1.1])
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Boresight Trajectory on Unit Sphere")
    ax1.legend()
    
    # Rotation angle φ vs time (same as in reward function)
    ax2 = fig.add_subplot(342)
    ax2.plot(times[:len(rotation_angles_deg)], rotation_angles_deg, "purple", linewidth=3, label="Angle $\\phi$")
    ax2.axhline(y=0.25, color="r", linestyle="--", linewidth=2, label="Accuracy Threshold (0.25°)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Rotation Angle $\\phi$ (°)")
    ax2.set_title("Rotation Angle $\\phi$ vs Time\n(Used in reward function)")
    ax2.grid(True)
    ax2.legend()
    ax2.set_yscale("log")
    
    # Cumulative Reward vs Time
    ax3 = fig.add_subplot(343)  # New subplot for cumulative reward
    ax3.plot(times[:len(cumulative_rewards)], cumulative_rewards, "orange", linewidth=3, label="Cumulative Reward")
    ax3.plot(times[:len(rewards_array)], rewards_array, "lightcoral", alpha=0.6, linewidth=1, label="Step Reward")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Reward")
    ax3.set_title("Reward Evolution")
    ax3.grid(True)
    ax3.legend()
    
    # Plot quaternion
    ax4 = fig.add_subplot(344)
    ax4.plot(times, q_0, label="$q_0$")
    ax4.plot(times, q_1, label="$q_1$")
    ax4.plot(times, q_2, label="$q_2$")
    ax4.plot(times, q_3, label="$q_3$")
    ax4.plot(times, norm_q, label="norm")
    ax4.set_title("Attitude")
    ax4.set_ylabel("Quaternion")
    ax4.legend()
    ax4.grid()

    # Plot angular velocity
    ax5 = fig.add_subplot(345)
    ax5.plot(times, omega_x * (180 / np.pi), label="$\\omega_x$")
    ax5.plot(times, omega_y * (180 / np.pi), label="$\\omega_y$")
    ax5.plot(times, omega_z * (180 / np.pi), label="$\\omega_z$")
    ax5.set_title("Angular velocity")
    ax5.set_ylabel("$\\omega$ (deg/s)")
    ax5.legend()
    ax5.grid()

    # Plot wheel velocity
    ax5_2 = fig.add_subplot(346)
    ax5_2.plot(times, omega_w_x, label="$\\omega_{w,x}$")
    ax5_2.plot(times, omega_w_y, label="$\\omega_{w,y}$")
    ax5_2.plot(times, omega_w_z, label="$\\omega_{w,z}$")
    ax5_2.set_title("Wheel Velocity")
    ax5_2.set_ylabel("$\\omega$ (rad/s)")
    ax5_2.legend()
    ax5_2.grid()

    # Plot torque input
    ax6 = fig.add_subplot(347)
    ax6.plot(times, torques_array[:, 0], label="$\\tau_1$")
    ax6.plot(times, torques_array[:, 1], label="$\\tau_2$")
    ax6.plot(times, torques_array[:, 2], label="$\\tau_3$")
    ax6.set_title("Control torques")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("$\\tau$ (Nm)")
    ax6.legend()
    ax6.grid()

    # Plot keep out zone margin angle
    ax7 = fig.add_subplot(348)
    ax7.plot(times, margin_angles_koz[:, 0], label="Margin Angle KOZ") # TODO: support multiple KOZs
    ax7.set_title("Keep Out Zone Margin Angle")
    ax7.set_ylabel("Angle (degrees)")
    ax7.legend()
    ax7.grid()

    # Plot keep out zone direction vector (body frame)
    ax8 = fig.add_subplot(349)
    ax8.plot(times, direction_koz[:, 0, 0], label="x") # TODO: support multiple KOZs
    ax8.plot(times, direction_koz[:, 0, 1], label="y") # TODO: support multiple KOZs
    ax8.plot(times, direction_koz[:, 0, 2], label="z") # TODO: support multiple KOZs
    ax8.set_title("Keep Out Zone Direction Vector (Body Frame)")
    ax8.legend()
    ax8.grid()

    # Plot LSTM output
    ax9 = fig.add_subplot(3,4,10)
    ax9.plot(times, lstm_output[:, 0, 0], label="$h_0$") # TODO: support multiple KOZs
    ax9.plot(times, lstm_output[:, 0, 1], label="$h_1$") # TODO: support multiple KOZs
    ax9.plot(times, lstm_output[:, 0, 2], label="$h_2$") # TODO: support multiple KOZs
    ax9.plot(times, lstm_output[:, 0, 3], label="$h_3$") # TODO: support multiple KOZs
    ax9.set_title("LSTM Output (Hidden States)")
    ax9.legend()
    ax9.grid()
    
    plt.tight_layout()
    plt.show()

    print("Initial rotation angle:", rotation_angles_deg[0], "degrees")

    print_result(rotation_angles_deg[-1], simulation_data["omega"][-1], cumulative_rewards[-1])
    
    return 


def plot_for_report(simulation_data: dict, time_end=300):
    """
    Plot the data and arrange it for report format.

    Args:
        simulation_data: A dictionary containing the simulation data for plotting.
        time_end: The end time for the plots (default: 300 seconds).
    """
    
    # Switch to interactive backend for 3D plots
    matplotlib.use("TkAgg")

    # Parse quaternion and angular velocity components
    q_0, q_1, q_2, q_3 = simulation_data["quaternion"][:, 0], simulation_data["quaternion"][:, 1], simulation_data["quaternion"][:, 2], simulation_data["quaternion"][:, 3]
    omega_x, omega_y, omega_z = simulation_data["omega"][:, 0], simulation_data["omega"][:, 1], simulation_data["omega"][:, 2]
    omega_w_x, omega_w_y, omega_w_z = simulation_data["omega_wheels"][:, 0], simulation_data["omega_wheels"][:, 1], simulation_data["omega_wheels"][:, 2]
    torques_array = simulation_data["torques"]
    times = simulation_data["times"]
    normal_vector_koz = simulation_data["normal_vector_koz"]
    half_angle_koz = simulation_data["half_angle_koz"]
    margin_angles_koz = simulation_data["margin_angles_koz"]
    min_margin_koz = simulation_data["min_margin_koz"]
    cnt_Koz_violations = simulation_data["cnt_Koz_violations"]

    print("Minimum margin KOZ:", min_margin_koz*180/np.pi, "degrees")
    print("Count KOZ violations:", cnt_Koz_violations)
    print("Half angle KOZ:", half_angle_koz*180/np.pi, "degrees")
    

    # Extract rotation axes and angles for all time points
    rotation_axes = []
    rotation_angles = []
    
    for i in range(len(q_0)):
        axis, angle = quat_to_axis_angle([q_0[i], q_1[i], q_2[i], q_3[i]])
        rotation_axes.append(axis)
        rotation_angles.append(angle)
    
    # Convert to numpy arrays
    rotation_axes = np.array(rotation_axes)  # Shape: (N, 3)
    rotation_angles = np.array(rotation_angles)  # Shape: (N,)

    # Calculate the body X-axis direction (boresight) at each time point using the quaternion rotation
    body_axis_arr = []
    for i in range(len(q_0)):
        q = [q_0[i], q_1[i], q_2[i], q_3[i]]
        w, x, y, z = q
        R = np.array([
                    [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
                    [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
                    [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
                ])
        body_axis = R @ np.array([1, 0, 0])  # body X-axis
        body_axis_arr.append(body_axis)
    
    # Convert to numpy array for proper indexing
    body_axis_arr = np.array(body_axis_arr)
    
    # figure for the trajectory
    fig1 = plt.figure(figsize=(8, 8))
    
    # 3D Rotation Axis Trajectory (This is the key trajectory for phi angle!)
    ax1 = fig1.add_subplot(111, projection="3d")
    
    # Adjust subplot to fill more of the figure space
    fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Plot trajectory on unit sphere (rotation axes are unit vectors)
    ax1.plot(body_axis_arr[:, 0], body_axis_arr[:, 1], body_axis_arr[:, 2], "b-", alpha=0.7, linewidth=3, label="Boresight axis trajectory")
    ax1.scatter(body_axis_arr[0, 0], body_axis_arr[0, 1], body_axis_arr[0, 2], color="green", s=50, label="Start")
    ax1.scatter(body_axis_arr[-1, 0], body_axis_arr[-1, 1], body_axis_arr[-1, 2], color="red", s=50, label="End")

    # Plot target last with higher zorder to ensure it's always on top
    ax1.scatter(1, 0, 0, color="gold", s=200, marker="*", label="Target", zorder=1000, edgecolors='black', linewidths=1)

    def _generate_keep_out_zone_circle():
        # Create circle points for the keep out zone

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points = []

        for angle in theta:
            # Generate points on the circle in the plane perpendicular to the normal vector
            v = np.array([np.cos(angle), np.sin(angle), 0])

            # Rotate v to be perpendicular to koz_normal
            if np.allclose(normal_vector_koz, [0, 0, 1]):
                rot_axis = np.array([1, 0, 0])
            else:
                rot_axis = np.cross([0, 0, 1], normal_vector_koz)
                rot_axis /= np.linalg.norm(rot_axis)

            angle_to_rotate = np.arccos(np.dot(normal_vector_koz, [0, 0, 1]))

            # Rodrigues' rotation formula
            v_rotated = (v * np.cos(angle_to_rotate) +
                        np.cross(rot_axis, v) * np.sin(angle_to_rotate) +
                        rot_axis * np.dot(rot_axis, v) * (1 - np.cos(angle_to_rotate)))
            
            # Scale to the radius of the keep out zone circle
            radius = np.sin(half_angle_koz)
            circle_point = normal_vector_koz * np.cos(half_angle_koz) + v_rotated * radius
            circle_points.append(circle_point)

        return circle_points

    # Plot keep out zone as a ring on the unit sphere
    if normal_vector_koz is not None and half_angle_koz is not None:
        circle_points = _generate_keep_out_zone_circle()
        circle_points = np.array(circle_points)
        ax1.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], "orange", linewidth=2, label="Keep-out zone")
    
    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color="gray")
    
    # Remove cartesian grid (x, y, z panes and axes)
    ax1.grid(False)
    ax1.set_axis_off()
  
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 0.2), fontsize=6)
    
    # cut time and data
    times = times[:int(time_end/Constants.TIME_DELTA)]
    q_0 = q_0[:int(time_end/Constants.TIME_DELTA)]
    q_1 = q_1[:int(time_end/Constants.TIME_DELTA)]
    q_2 = q_2[:int(time_end/Constants.TIME_DELTA)]
    q_3 = q_3[:int(time_end/Constants.TIME_DELTA)]
    omega_x = omega_x[:int(time_end/Constants.TIME_DELTA)]
    omega_y = omega_y[:int(time_end/Constants.TIME_DELTA)]
    omega_z = omega_z[:int(time_end/Constants.TIME_DELTA)]
    torques_array = torques_array[:int(time_end/Constants.TIME_DELTA)]
    margin_angles_koz = margin_angles_koz[:int(time_end/Constants.TIME_DELTA)]

    # figure for attitude, angular velocity, torque and KOZ margin
    fig2 = plt.figure(figsize=(6, 6))
    
    # Plot quaternion
    ax4 = fig2.add_subplot(411)
    ax4.plot(times, q_0, label="$q_0$")
    ax4.plot(times, q_1, label="$q_1$")
    ax4.plot(times, q_2, label="$q_2$")
    ax4.plot(times, q_3, label="$q_3$")
    ax4.set_ylabel("q")
    ax4.legend(loc="upper right")
    ax4.grid()

    # Plot angular velocity
    ax5 = fig2.add_subplot(412)
    ax5.plot(times, omega_x * (180 / np.pi), label="$\\omega_x$")
    ax5.plot(times, omega_y * (180 / np.pi), label="$\\omega_y$")
    ax5.plot(times, omega_z * (180 / np.pi), label="$\\omega_z$")
    ax5.set_ylabel("$\\omega$ [deg/s]")
    ax5.legend(loc="upper right")
    ax5.grid()
    # Plot torque input
    ax6 = fig2.add_subplot(413)
    ax6.plot(times, torques_array[:, 0], label="$\\tau_1$")
    ax6.plot(times, torques_array[:, 1], label="$\\tau_2$")
    ax6.plot(times, torques_array[:, 2], label="$\\tau_3$")
    ax6.set_ylabel("$\\tau$ [Nm]")
    #ax6.set_xlabel("Time [s]")
    ax6.legend(loc="upper right")
    ax6.grid()

    # Plot keep out zone margin angle
    ax7 = fig2.add_subplot(414)
    ax7.plot(times, margin_angles_koz, label="$\\theta_{margin}$")
    ax7.set_xlabel("Time [s]")
    ax7.set_ylabel("$\\theta_{margin}$ [deg]")
    ax7.legend(loc="upper right")
    ax7.grid()
    
    plt.tight_layout()
    plt.show()
    
    return


def visualize_net_arch(model, env):
    policy = model.policy
    observation_space = getattr(env, "observation_space", None)

    if observation_space is None:
        raise ValueError("Could not infer observation space from env or model.")

    obs_keys = list(observation_space.spaces.keys())
    input_shape = tuple(
        (1, *observation_space.spaces[key].shape)
        for key in obs_keys
    )

    class PolicyObservationAdapter(th.nn.Module):
        def __init__(self, policy, obs_keys):
            super().__init__()
            self.actor = policy.actor
            self.obs_keys = obs_keys

        def forward(self, *obs_parts):
            observations = dict(zip(self.obs_keys, obs_parts))
            sat_obs = observations["satellite"]
            zones_obs = observations["zones"]

            # Ensure the LSTM path is included in the rendered graph.
            if "zones_mask" in observations:
                zones_mask = th.ones_like(observations["zones_mask"])
                zones_obs = zones_obs * zones_mask.unsqueeze(-1)

            features_extractor = self.actor.features_extractor
            _, (hidden_state_out, _) = features_extractor.lstm(zones_obs)
            features = th.cat([sat_obs, hidden_state_out[-1]], dim=1)

            latent_pi = self.actor.latent_pi(features)
            mean_actions = self.actor.mu(latent_pi)

            log_std = self.actor.log_std(latent_pi)
            return th.cat([mean_actions, log_std], dim=1)
        
    op = visualtorch.GraphStyleOptions()

    print(policy)
        
    img = visualtorch.render(
        PolicyObservationAdapter(policy, obs_keys),
        input_shape=input_shape,
        style="graph",
        show_dimension=True,
        show_arrows=True
    )
    img.save("net_arch_test.png")
    

### MAIN ###
if __name__ == "__main__":
   
    # Set initial state for evaluation environment
    INITIAL_STATE = [
        Config.Visualization.MIN_INITIAL_ERROR_ANGLE,
        Config.Visualization.MAX_INITIAL_ERROR_ANGLE,
        Config.Visualization.MIN_INITIAL_ANGULAR_VELOCITY,
        Config.Visualization.MAX_INITIAL_ANGULAR_VELOCITY,
        Config.Visualization.MAX_STEPS,
        Config.Visualization.MIN_HALF_ANGLE_KOZ,
        Config.Visualization.MAX_HALF_ANGLE_KOZ,
        Config.Visualization.MIN_NR_KOZ,
        Config.Visualization.MAX_NR_KOZ
    ]

    time_human = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    eval_env = create_evaluation_env(INITIAL_STATE, Config.Visualization.MODEL_NAME, Config.Visualization.TIMESTEP)
    model = load_agent(Config.Visualization.MODEL_NAME, Config.Visualization.TIMESTEP, seed_random = True)

    """ Uncomment the lines below to run 1 simulation and plot the results. """
    simulation_data = simulate_episode(model, eval_env, Config.Visualization.MAX_STEPS, Config.Visualization.MODEL_NAME, create_video=Config.Visualization.CREATE_VIDEO)
    #visualize_episode_basilisk(simulation_data, show_basilisk_viz=Config.Visualization.SHOW_BASILISK_VIZ)
    viz_file_path = os.path.join(viz_dir, f"{Config.Visualization.MODEL_NAME}_{INITIAL_STATE}_{time_human}.bin")
    save_episode_as_viz(viz_file_path, simulation_data, Config.Visualization.SHOW_BASILISK_VIZ, Config.Visualization.VIZARD_EXE_PATH)
    #plot_actual_attitude(simulation_data)
    visualize_net_arch(model, eval_env)
    #plot_for_report(simulation_data, time_end=300)

    """ Uncomment the lines below if you have saved evaluation data (from evaluate_agent()) to load all the episodes.
        loaded contains ALL episodes, therefore in loaded[] should be the index of the episode you want to plot.
    """
    #loaded = load_evaluation_data("rewMod22_phFull_3_ph2_4000000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 1, 1]_ep[1000]_2026-07-16-16-39-32.npz")
    #plot_actual_attitude(loaded[708])
    #plot_for_report(loaded[0],time_end=300)
