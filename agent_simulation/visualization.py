"""
Visualization tools.

Author: Cemal Yilmaz - 2026
"""

import os
import sys
import subprocess
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter, MultipleLocator
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
    rewards_discounted_array = simulation_data["rewards_discounted"]
    cumulative_rewards = simulation_data["cumulative_rewards"]
    cumulative_rewards_discounted = simulation_data["cumulative_rewards_discounted"]
    times = simulation_data["times"]
    normal_vector_koz_array = simulation_data["normal_vector_koz_array"] # normal vector in world frame
    half_angle_koz_array = simulation_data["half_angle_koz_array"]
    margin_angles_koz = simulation_data["margin_angles_koz"]
    direction_koz = simulation_data["direction_koz"] # normal vector in body frame
    min_margin_koz = simulation_data["min_margin_koz"]
    cnt_Koz_violations = simulation_data["cnt_Koz_violations"]
    lstm_output = simulation_data["lstm_output"]


    print("Minimum margin KOZ:", min_margin_koz, "degrees")
    print("Count KOZ violations:", cnt_Koz_violations)
    for idx, angle in enumerate(half_angle_koz_array):
        print(f"Half angle KOZ {idx+1}:", angle*180/np.pi, "degrees")

    # How many KOZs there are in this episode
    koz_cnt = len(normal_vector_koz_array)

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

    def _generate_keep_out_zone_circles():
        # Create circle points for the keep out zone

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points_array = []

        for idx in range(len(normal_vector_koz_array)):
            circle_points = []

            for angle in theta:
                # Generate points on the circle in the plane perpendicular to the normal vector
                v = np.array([np.cos(angle), np.sin(angle), 0])

                # Rotate v to be perpendicular to koz_normal
                if np.allclose(normal_vector_koz_array[idx], [0, 0, 1]):
                    rot_axis = np.array([1, 0, 0])
                else:
                    rot_axis = np.cross([0, 0, 1], normal_vector_koz_array[idx])
                    rot_axis /= np.linalg.norm(rot_axis)

                angle_to_rotate = np.arccos(np.dot(normal_vector_koz_array[idx], [0, 0, 1]))

                # Rodrigues' rotation formula
                v_rotated = (v * np.cos(angle_to_rotate) +
                            np.cross(rot_axis, v) * np.sin(angle_to_rotate) +
                            rot_axis * np.dot(rot_axis, v) * (1 - np.cos(angle_to_rotate)))
                
                # Scale to the radius of the keep out zone circle
                radius = np.sin(half_angle_koz_array[idx])
                circle_point = normal_vector_koz_array[idx] * np.cos(half_angle_koz_array[idx]) + v_rotated * radius
                circle_points.append(circle_point)

            circle_points_array.append(circle_points)

        return circle_points_array
    
    # Plot keep out zone as a ring on the unit sphere
    if normal_vector_koz_array is not None and half_angle_koz_array is not None:
        circle_points_array = _generate_keep_out_zone_circles()
        circle_points_array = np.array(circle_points_array)

        ax1.plot(circle_points_array[0][:, 0], circle_points_array[0][:, 1], circle_points_array[0][:, 2], "orange", linewidth=2, label="Keep Out Zone 1") if len(normal_vector_koz_array) > 0 else None
        ax1.plot(circle_points_array[1][:, 0], circle_points_array[1][:, 1], circle_points_array[1][:, 2], "red", linewidth=2, label="Keep Out Zone 2") if len(normal_vector_koz_array) > 1 else None
        ax1.plot(circle_points_array[2][:, 0], circle_points_array[2][:, 1], circle_points_array[2][:, 2], "purple", linewidth=2, label="Keep Out Zone 3") if len(normal_vector_koz_array) > 2 else None
    
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
    ax3.plot(times[:len(cumulative_rewards)], cumulative_rewards, "orange", linewidth=2, label="Cumulative Reward")
    ax3.plot(times[:len(cumulative_rewards_discounted)], cumulative_rewards_discounted, "green", linewidth=1, label="Cumulative Reward Discounted")
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
    for koz_idx in range(koz_cnt):
        ax7.plot(times, margin_angles_koz[:, koz_idx], label=f"KOZ {koz_idx+1}")
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

    # Plot LSTM output if available.
    ax9 = fig.add_subplot(3,4,10)
    if lstm_output.shape[-1] > 0:
        ax9.plot(times, lstm_output[:, 0, 0], label="$h_0$")
        ax9.plot(times, lstm_output[:, 0, 1], label="$h_1$")
        ax9.plot(times, lstm_output[:, 0, 2], label="$h_2$")
        ax9.plot(times, lstm_output[:, 0, 3], label="$h_3$")
        ax9.set_title("LSTM Output (Hidden States)")
        ax9.legend()
    else:
        ax9.set_title("APF Controller (No LSTM)")
        ax9.set_axis_off()
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
    norm_q = simulation_data["quaternion_norm"]
    torques_array = simulation_data["torques"]
    rewards_array = simulation_data["rewards"]
    rewards_discounted_array = simulation_data["rewards_discounted"]
    cumulative_rewards = simulation_data["cumulative_rewards"]
    cumulative_rewards_discounted = simulation_data["cumulative_rewards_discounted"]
    times = simulation_data["times"]
    normal_vector_koz_array = simulation_data["normal_vector_koz_array"] # normal vector in world frame
    half_angle_koz_array = simulation_data["half_angle_koz_array"]
    margin_angles_koz = simulation_data["margin_angles_koz"]
    direction_koz = simulation_data["direction_koz"] # normal vector in body frame
    min_margin_koz = simulation_data["min_margin_koz"]
    cnt_Koz_violations = simulation_data["cnt_Koz_violations"]
    lstm_output = simulation_data["lstm_output"]

    print("Minimum margin KOZ:", min_margin_koz, "degrees")
    print("Count KOZ violations:", cnt_Koz_violations)
    for idx, angle in enumerate(half_angle_koz_array):
        print(f"Half angle KOZ {idx+1}:", angle*180/np.pi, "degrees")

    print("Sum torque magnitude:", np.sum(np.linalg.norm(torques_array, axis=1)), "Nm")

    # How many KOZs there are in this episode
    koz_cnt = len(normal_vector_koz_array)

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
    
    # figure for the trajectory
    fig1 = plt.figure(figsize=(8, 8))
    
    # 3D Rotation Axis Trajectory (This is the key trajectory for phi angle!)
    ax1 = fig1.add_subplot(111, projection="3d")
    
    # Adjust subplot to fill more of the figure space
    #fig1.subplots_adjust(left=0, right=5, top=5, bottom=0)
    
    # Plot trajectory on unit sphere (rotation axes are unit vectors)
    ax1.plot(body_axis_arr[:, 0], body_axis_arr[:, 1], body_axis_arr[:, 2], "b-", alpha=0.7, linewidth=3, label="Boresight Trajectory")
    ax1.scatter(body_axis_arr[0, 0], body_axis_arr[0, 1], body_axis_arr[0, 2], color="green", s=50, label="Start")
    ax1.scatter(body_axis_arr[-1, 0], body_axis_arr[-1, 1], body_axis_arr[-1, 2], color="red", s=50, label="End")

    # Plot target last with higher zorder to ensure it's always on top
    ax1.scatter(1, 0, 0, color="gold", s=200, marker="*", label="Target", zorder=1000, edgecolors='black', linewidths=1)

    def _generate_keep_out_zone_circles():
        # Create circle points for the keep out zone

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_points_array = []

        for idx in range(len(normal_vector_koz_array)):
            circle_points = []

            for angle in theta:
                # Generate points on the circle in the plane perpendicular to the normal vector
                v = np.array([np.cos(angle), np.sin(angle), 0])

                # Rotate v to be perpendicular to koz_normal
                if np.allclose(normal_vector_koz_array[idx], [0, 0, 1]):
                    rot_axis = np.array([1, 0, 0])
                else:
                    rot_axis = np.cross([0, 0, 1], normal_vector_koz_array[idx])
                    rot_axis /= np.linalg.norm(rot_axis)

                angle_to_rotate = np.arccos(np.dot(normal_vector_koz_array[idx], [0, 0, 1]))

                # Rodrigues' rotation formula
                v_rotated = (v * np.cos(angle_to_rotate) +
                            np.cross(rot_axis, v) * np.sin(angle_to_rotate) +
                            rot_axis * np.dot(rot_axis, v) * (1 - np.cos(angle_to_rotate)))
                
                # Scale to the radius of the keep out zone circle
                radius = np.sin(half_angle_koz_array[idx])
                circle_point = normal_vector_koz_array[idx] * np.cos(half_angle_koz_array[idx]) + v_rotated * radius
                circle_points.append(circle_point)

            circle_points_array.append(circle_points)

        return circle_points_array
    
    # Plot keep out zone as a ring on the unit sphere
    if normal_vector_koz_array is not None and half_angle_koz_array is not None:
        circle_points_array = _generate_keep_out_zone_circles()
        circle_points_array = np.array(circle_points_array)

        ax1.plot(circle_points_array[0][:, 0], circle_points_array[0][:, 1], circle_points_array[0][:, 2], "orange", linewidth=2, label="Keep Out Zone 1") if len(normal_vector_koz_array) > 0 else None
        ax1.plot(circle_points_array[1][:, 0], circle_points_array[1][:, 1], circle_points_array[1][:, 2], "red", linewidth=2, label="Keep Out Zone 2") if len(normal_vector_koz_array) > 1 else None
        ax1.plot(circle_points_array[2][:, 0], circle_points_array[2][:, 1], circle_points_array[2][:, 2], "purple", linewidth=2, label="Keep Out Zone 3") if len(normal_vector_koz_array) > 2 else None
    
    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color="gray")
    ax1.set_box_aspect((1, 1, 1))
    
    # Remove cartesian grid (x, y, z panes and axes)
    ax1.grid(False)
    ax1.set_axis_off()
  
    ax1.legend(loc="lower center", bbox_to_anchor=(0.65, 0.2), fontsize=6)

    plt.tight_layout()
    plt.show()
    
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
    cumulative_rewards = cumulative_rewards[:int(time_end/Constants.TIME_DELTA)]
    rotation_angles_deg = rotation_angles_deg[:int(time_end/Constants.TIME_DELTA)]

    # figure for reward, attitude, angular velocity, torque and KOZ margin
    fig2 = plt.figure(figsize=(6, 8))

    # Cumulative Reward vs Time
    ax4_1 = fig2.add_subplot(511)  # New subplot for cumulative reward
    ax4_1.plot(times, cumulative_rewards, label="Cumulative Reward")
    ax4_1.set_ylabel("Reward")
    ax4_1.grid()
    ax4_1.legend(loc="lower right")
    
    # Plot attitude error
    ax4 = fig2.add_subplot(512)
    ax4.plot(times, rotation_angles_deg, label="$\\phi_{\\mathrm{e}}$")
    ax4.axhline(y=1.0, color="r", linestyle="--", label="Accuracy Threshold (1°)")
    ax4.set_ylabel("Attitude Error [deg]")
    ax4.grid(True)
    ax4.legend(loc="upper right")
    ax4.set_yscale("log")

    # Plot angular velocity
    ax5 = fig2.add_subplot(513)
    ax5.plot(times, omega_x * (180 / np.pi), label="$\\omega_x$")
    ax5.plot(times, omega_y * (180 / np.pi), label="$\\omega_y$")
    ax5.plot(times, omega_z * (180 / np.pi), label="$\\omega_z$")
    ax5.set_ylabel("Ang. Velocity [deg/s]")
    ax5.legend(loc="upper right")
    ax5.grid()
    # Plot torque input
    ax6 = fig2.add_subplot(514)
    ax6.plot(times, torques_array[:, 0], label="$\\tau_1$")
    ax6.plot(times, torques_array[:, 1], label="$\\tau_2$")
    ax6.plot(times, torques_array[:, 2], label="$\\tau_3$")
    ax6.set_ylabel("Torque [Nm]")
    #ax6.set_xlabel("Time [s]")
    ax6.legend(loc="upper right")
    ax6.grid()

    # Plot keep out zone margin angle
    ax7 = fig2.add_subplot(515)

    for koz_idx in range(koz_cnt):
            ax7.plot(times, margin_angles_koz[:, koz_idx], label="$\\theta_{\\mathrm{margin}}$"+f" (KOZ {koz_idx+1})")

    ax7.set_xlabel("Time [s]")
    ax7.set_ylabel("KOZ Margin [deg]")
    ax7.legend(loc="lower right")
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


def plot_tensorboard_data():
    # Switch to interactive backend for 3D plots
    matplotlib.use("TkAgg")

    timestep_formatter = FuncFormatter(lambda step, _: f"{step / 1_000_000:g}M")

    project_dir = os.path.dirname(repo_parent_dir)
    data_dir = os.path.join(project_dir, "figures", "tensorboard", "data_csv", "rewMod224ph2c2_fast_schedFullv1c3")
    data_dir_2 = os.path.join(project_dir, "figures", "tensorboard", "data_csv", "rewMod224ph2c2_fast_kozMinMargin0_schedFullv1c3")

    data = dict()
    data_2 = dict()

    # first data
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            tag = file.replace(".csv", "")
            data[tag] = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=(1, 2), unpack=True, ndmin=2)

    # second data
    for file in os.listdir(data_dir_2):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir_2, file)
            tag = file.replace(".csv", "")
            data_2[tag] = np.loadtxt(file_path, delimiter=",", skiprows=1, usecols=(1, 2), unpack=True, ndmin=2)

    # Figure 1
    fig1 = plt.figure(figsize=(6, 8))

    
    
    # Plot actor_loss
    ax1_1 = fig1.add_subplot(411)
    ax1_1.plot(*data["train_actor_loss"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[20,20+\theta]°$")
    ax1_1.plot(*data_2["train_actor_loss"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[0,\theta]°$")
    ax1_1.set_ylabel("Actor Loss")
    ax1_1.legend(loc="lower right")
    ax1_1.grid()

    # Plot critic_loss log scale
    ax1_2 = fig1.add_subplot(412)
    ax1_2.plot(*data["train_critic_loss"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[20,20+\theta]°$")
    ax1_2.plot(*data_2["train_critic_loss"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[0,\theta]°$")
    ax1_2.set_ylabel("Critic Loss")
    ax1_2.set_yscale("log")
    ax1_2.legend(loc="upper right")
    ax1_2.grid()

    # Plot ent_coef log scale
    ax1_3 = fig1.add_subplot(413)
    ax1_3.plot(*data["train_ent_coef"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[20,20+\theta]°$")
    ax1_3.plot(*data_2["train_ent_coef"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[0,\theta]°$")
    ax1_3.set_ylabel("Entropy Coefficient")
    ax1_3.set_yscale("log")
    ax1_3.legend(loc="upper right")
    ax1_3.grid()

    # Plot fps
    ax1_4 = fig1.add_subplot(414)
    ax1_4.plot(*data["time_fps"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[20,20+\theta]°$")
    ax1_4.plot(*data_2["time_fps"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[0,\theta]°$")
    ax1_4.set_xlabel("Timestep")
    ax1_4.set_ylabel("FPS")
    ax1_4.legend(loc="upper right")
    ax1_4.grid()

    for ax in fig1.axes:
        ax.xaxis.set_major_formatter(timestep_formatter)

    plt.tight_layout()
    plt.show()

    # Figure 2
    fig2 = plt.figure(figsize=(6, 8))

    # Plot ep_rew_mean
    ax2_1 = fig2.add_subplot(411)
    ax2_1.plot(*data["rollout_ep_rew_mean"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[20,20+\theta]°$")
    ax2_1.plot(*data_2["rollout_ep_rew_mean"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[0,\theta]°$")
    ax2_1.set_ylabel("Reward")
    ax2_1.legend(loc="lower right")
    ax2_1.grid()

    # Plot final_error_angle_max log scale
    ax2_2 = fig2.add_subplot(412)
    ax2_2.plot(*data["custom_final_error_angle_max"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[20,20+\theta]°$")
    ax2_2.plot(*data_2["custom_final_error_angle_max"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[0,\theta]°$")
    ax2_2.set_ylabel("Final Error Angle [deg]")
    ax2_2.set_yscale("log")
    ax2_2.legend(loc="upper right")
    ax2_2.grid()

    # Plot entered_koz_count_mean log scale
    ax2_3 = fig2.add_subplot(413)
    ax2_3.plot(*data["custom_entered_koz_count_mean"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[20,20+\theta]°$")
    ax2_3.plot(*data_2["custom_entered_koz_count_mean"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[0,\theta]°$")
    ax2_3.set_ylabel("Entered KOZ Count")
    #ax2_3.set_yscale("log")
    ax2_3.legend(loc="upper right")
    ax2_3.grid()

    # Plot min_margin_koz_mean
    ax2_4 = fig2.add_subplot(414)
    ax2_4.plot(*data["custom_min_margin_koz_mean"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[20,20+\theta]°$")
    ax2_4.plot(*data_2["custom_min_margin_koz_mean"], linewidth=1, label=r"$\Omega_\mathrm{margin}=[0,\theta]°$")
    ax2_4.set_xlabel("Timestep")
    ax2_4.set_ylabel("Min KOZ Margin [deg]")
    ax2_4.legend(loc="lower right")
    ax2_4.grid()

    for ax in fig2.axes:
        ax.xaxis.set_major_formatter(timestep_formatter)

    plt.tight_layout()
    plt.show()

    # Average the magnitude of each parameter's change between logged timesteps.
    def mean_absolute_change(tags):
        steps = data[tags[0]][0]

        changes = np.stack([
            np.abs(np.diff(data[tag][1]))
            for tag in tags
        ])

        return steps[1:], np.mean(changes, axis=0)

    gates = ("cell", "forget", "input", "output")
    
    connections = ("hh", "ih")

    # Figure 3: one mean across four biases for each gate and connection.
    bias_change_means = {
        (gate, connection): mean_absolute_change([
            f"lstm_actor_bias_b_{connection}_{gate}_h{row}"
            for row in range(4)
        ])

        for gate in gates
        for connection in connections
    }

    weight_change_means = {
        (gate, connection): mean_absolute_change([
            f"lstm_actor_weight_w_{connection}_{gate}_h{row}{"h" if connection == "hh" else "i"}{column}"
            for row in range(4)
            for column in range(4)
        ])

        for gate in gates
        for connection in connections
    }

    fig3 = plt.figure(figsize=(12, 10))

    ax3_1 = fig3.add_subplot(211)
    ax3_1.plot(*bias_change_means[("cell", "hh")], linewidth=1, label=r"$\overline{|\Delta b_{h,g}|}$")
    ax3_1.plot(*bias_change_means[("cell", "ih")], linewidth=1, label=r"$\overline{|\Delta b_{i,g}|}$")
    ax3_1.plot(*bias_change_means[("forget", "hh")], linewidth=1, label=r"$\overline{|\Delta b_{h,f}|}$")
    ax3_1.plot(*bias_change_means[("forget", "ih")], linewidth=1, label=r"$\overline{|\Delta b_{i,f}|}$")
    ax3_1.plot(*bias_change_means[("input", "hh")], linewidth=1, label=r"$\overline{|\Delta b_{h,i}|}$")
    ax3_1.plot(*bias_change_means[("input", "ih")], linewidth=1, label=r"$\overline{|\Delta b_{i,i}|}$")
    ax3_1.plot(*bias_change_means[("output", "hh")], linewidth=1, label=r"$\overline{|\Delta b_{h,o}|}$")
    ax3_1.plot(*bias_change_means[("output", "ih")], linewidth=1, label=r"$\overline{|\Delta b_{i,o}|}$")
    ax3_1.set_ylabel("Bias Change")
    ax3_1.legend(loc="upper right", ncols=4)

    for legend_line in ax3_1.get_legend().get_lines():
        legend_line.set_linewidth(2)

    ax3_1.grid()
    ax3_1.xaxis.set_major_formatter(timestep_formatter)

    ax3_2 = fig3.add_subplot(212)
    ax3_2.plot(*weight_change_means[("cell", "hh")], linewidth=1, label=r"$\overline{|\Delta W_{h,g}|}$")
    ax3_2.plot(*weight_change_means[("cell", "ih")], linewidth=1, label=r"$\overline{|\Delta W_{i,g}|}$")
    ax3_2.plot(*weight_change_means[("forget", "hh")], linewidth=1, label=r"$\overline{|\Delta W_{h,f}|}$")
    ax3_2.plot(*weight_change_means[("forget", "ih")], linewidth=1, label=r"$\overline{|\Delta W_{i,f}|}$")
    ax3_2.plot(*weight_change_means[("input", "hh")], linewidth=1, label=r"$\overline{|\Delta W_{h,i}|}$")
    ax3_2.plot(*weight_change_means[("input", "ih")], linewidth=1, label=r"$\overline{|\Delta W_{i,i}|}$")
    ax3_2.plot(*weight_change_means[("output", "hh")], linewidth=1, label=r"$\overline{|\Delta W_{h,o}|}$")
    ax3_2.plot(*weight_change_means[("output", "ih")], linewidth=1, label=r"$\overline{|\Delta W_{i,o}|}$")
    ax3_2.set_xlabel("Timestep")
    ax3_2.set_ylabel("Weight Change")
    ax3_2.legend(loc="upper right", ncols=4)

    for legend_line in ax3_2.get_legend().get_lines():
        legend_line.set_linewidth(2)

    ax3_2.grid()
    ax3_2.xaxis.set_major_formatter(timestep_formatter)

    plt.tight_layout()
    #plt.show()

    # Figure 4: average each bias over consecutive 5M-timestep windows.
    window_size = 5_000_000

    def smooth_bias(series):
        steps, values = series
        smoothed_steps = []
        smoothed_values = []

        for window_start in range(0, int(steps[-1]) + 1, window_size):
            in_window = (steps >= window_start) & (steps < window_start + window_size)
            if np.any(in_window):
                smoothed_steps.append(np.mean(steps[in_window]))
                smoothed_values.append(np.mean(values[in_window]))

        return np.array(smoothed_steps), np.array(smoothed_values)

    smoothed_biases = {
        tag: smooth_bias(series)
        for tag, series in data.items()
        if tag.startswith("lstm_actor_bias_")
    }

    # Use smoothed_biases to create the bias change.
    bias_changes_smoothed = {
        tag: (steps, np.cumsum(np.diff(values, prepend=values[0])))
        for tag, (steps, values) in smoothed_biases.items()
    }

    fig4 = plt.figure(figsize=(12, 10))

    # Plot smoothed biases
    ax4_1 = fig4.add_subplot(211)
    ax4_1.plot(*smoothed_biases["lstm_actor_bias_b_hh_cell_h0"], linewidth=1, label=r"$b_{h,g,(0)}$")
    ax4_1.plot(*smoothed_biases["lstm_actor_bias_b_hh_cell_h1"], linewidth=1, label=r"$b_{h,g,(1)}$")
    ax4_1.plot(*smoothed_biases["lstm_actor_bias_b_hh_cell_h2"], linewidth=1, label=r"$b_{h,g,(2)}$")
    ax4_1.plot(*smoothed_biases["lstm_actor_bias_b_hh_cell_h3"], linewidth=1, label=r"$b_{h,g,(3)}$")
    ax4_1.plot(*smoothed_biases["lstm_actor_bias_b_hh_forget_h0"], linewidth=1, label=r"$b_{h,f,(0)}$")
    ax4_1.plot(*smoothed_biases["lstm_actor_bias_b_hh_forget_h1"], linewidth=1, label=r"$b_{h,f,(1)}$")
    ax4_1.legend(loc="upper left", ncols=4)

    # Plot smoothed bias changes
    ax4_2 = fig4.add_subplot(212)
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_cell_h0"], linewidth=1, label=r"$\Delta b_{h,g,(0)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_cell_h1"], linewidth=1, label=r"$\Delta b_{h,g,(1)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_cell_h2"], linewidth=1, label=r"$\Delta b_{h,g,(2)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_cell_h3"], linewidth=1, label=r"$\Delta b_{h,g,(3)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_forget_h0"], linewidth=1, label=r"$\Delta b_{h,f,(0)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_forget_h1"], linewidth=1, label=r"$\Delta b_{h,f,(1)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_forget_h2"], linewidth=1, label=r"$\Delta b_{h,f,(2)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_forget_h3"], linewidth=1, label=r"$\Delta b_{h,f,(3)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_input_h0"], linewidth=1, label=r"$\Delta b_{h,i,(0)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_input_h1"], linewidth=1, label=r"$\Delta b_{h,i,(1)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_input_h2"], linewidth=1, label=r"$\Delta b_{h,i,(2)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_input_h3"], linewidth=1, label=r"$\Delta b_{h,i,(3)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_output_h0"], linewidth=1, label=r"$\Delta b_{h,o,(0)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_output_h1"], linewidth=1, label=r"$\Delta b_{h,o,(1)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_output_h2"], linewidth=1, label=r"$\Delta b_{h,o,(2)}$")
    ax4_2.plot(*bias_changes_smoothed["lstm_actor_bias_b_hh_output_h3"], linewidth=1, label=r"$\Delta b_{h,o,(3)}$")
    ax4_2.legend(loc="upper left", ncols=4)
    
    #plt.show()


def plot_pointing_accuracy_histogram(loaded_episodes_list: list, evaluation_labels: list):
    # Switch to interactive backend for 3D plots
    matplotlib.use("TkAgg")

    # Extract final error angles from loaded episodes per evaluation
    final_error_angles_list = []
    for loaded_episodes in loaded_episodes_list:
        final_error_angles_list.append([2*np.arccos(np.abs(episode["quaternion"][-1,0])) * 180/np.pi for episode in loaded_episodes])

    # Create histogram bin dimension and weight.
    bin_width_fine = 0.2
    bin_width_coarse = 1.0 # 1.0 / 4.0
    max_angle = max((max(angles) for angles in final_error_angles_list if angles), default=0)
    num_bins_fine = int(np.floor(max_angle / bin_width_fine)) + 1
    num_bins_coarse = int(np.floor(max_angle / bin_width_coarse)) + 1
    bin_edges_fine = np.arange(num_bins_fine) * bin_width_fine
    bin_edges_coarse = np.arange(num_bins_coarse) * bin_width_coarse

    weights_list = [
        np.full(len(angles), 100 / len(angles)) if angles else np.array([])
        for angles in final_error_angles_list
    ]

    fig1 = plt.figure(figsize=(12, 6))
    ax1_1 = fig1.add_subplot(211)

    # Plot evaluations side by side within each shared bin.
    ax1_1.hist(final_error_angles_list, bins=bin_edges_coarse, weights=weights_list,
               alpha=0.7, label=evaluation_labels, histtype="bar", stacked=False)
    ax1_1.set_title("Total Range")
    ax1_1.set_ylabel("Episodes [%]")
    ax1_1.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    ax1_1.xaxis.set_major_locator(MultipleLocator(2)) # 2 / 10
    ax1_1.grid(axis='y', alpha=0.75)
    ax1_1.legend(loc="upper right")

    ax1_2 = fig1.add_subplot(212)
    ax1_2.hist(final_error_angles_list, bins=bin_edges_fine, weights=weights_list,
                alpha=0.7, label=evaluation_labels, histtype="bar", stacked=False)
    ax1_2.set_title("Fine Close-Target Range")
    ax1_2.set_xlabel("Final Attitude Error [deg]")
    ax1_2.set_ylabel("Episodes [%]")
    ax1_2.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    ax1_2.xaxis.set_major_locator(MultipleLocator(0.2))
    ax1_2.grid(axis='y', alpha=0.75)
    ax1_2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


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

    PHASE_TYPE = Config.Visualization.PHASE_TYPE
    CONTROLLER = Config.Visualization.CONTROLLER.upper()

    time_human = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    eval_env = create_evaluation_env(PHASE_TYPE, INITIAL_STATE, Config.Visualization.MODEL_NAME, Config.Visualization.TIMESTEP, controller=CONTROLLER)
    model = load_agent(Config.Visualization.MODEL_NAME, Config.Visualization.TIMESTEP, seed_random = True) if CONTROLLER == "DRL" else None

    """ Uncomment the lines below to run 1 simulation and plot the results. """
    simulation_data = simulate_episode(model, eval_env, Config.Visualization.MAX_STEPS, Config.Visualization.MODEL_NAME, create_video=Config.Visualization.CREATE_VIDEO, controller=CONTROLLER)
    #visualize_episode_basilisk(simulation_data, show_basilisk_viz=Config.Visualization.SHOW_BASILISK_VIZ)
    viz_file_path = os.path.join(viz_dir, f"{Config.Visualization.MODEL_NAME}_{INITIAL_STATE}_{time_human}.bin")
    #save_episode_as_viz(viz_file_path, simulation_data, Config.Visualization.SHOW_BASILISK_VIZ, Config.Visualization.VIZARD_EXE_PATH)
    #plot_actual_attitude(simulation_data)
    #visualize_net_arch(model, eval_env)
    plot_for_report(simulation_data, time_end=300)
    #plot_tensorboard_data()

    """ Uncomment the lines below if you have saved evaluation data (from evaluate_agent()) to load all the episodes.
        loaded contains ALL episodes, therefore in loaded[] should be the index of the episode you want to plot.
    """

    #loaded_0 = load_evaluation_data("rewMod224ph2c2_fast_kozMinMargin0_schedfullv1c3_48400000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 0, 0]_ep[10000]_2026-10-05-17-28-17.npz")
    #loaded_1 = load_evaluation_data("rewMod224ph2c2_fast_kozMinMargin0_schedfullv1c3_48400000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 1, 1]_ep[10000]_2026-10-05-18-19-17.npz")
    #loaded_2 = load_evaluation_data("rewMod224ph2c2_fast_kozMinMargin0_schedfullv1c3_48400000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 2, 2]_ep[10000]_2026-10-05-19-15-21.npz")
    #loaded_3 = load_evaluation_data("rewMod224ph2c2_fast_kozMinMargin0_schedfullv1c3_48400000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 3, 3]_ep[10000]_2026-10-05-20-15-06.npz")
    #loaded_4 = load_evaluation_data("rewMod224ph2c2_fast_kozMinMargin0_schedfullv1c3_48400000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 3, 3]_ep[10000]_kozMinMargin0_2026-10-05-21-08-08.npz")

    #loaded_0 = load_evaluation_data("rewMod224ph2c2_fast_schedfullv1c3_52000000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 0, 0]_ep[10000]_2026-10-05-11-28-59.npz")
    #loaded_1 = load_evaluation_data("rewMod224ph2c2_fast_schedfullv1c3_52000000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 1, 1]_ep[10000]_2026-10-05-12-25-31.npz")
    #loaded_2 = load_evaluation_data("rewMod224ph2c2_fast_schedfullv1c3_52000000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 2, 2]_ep[10000]_2026-10-05-14-01-11.npz")
    #loaded_3 = load_evaluation_data("rewMod224ph2c2_fast_schedfullv1c3_52000000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 3, 3]_ep[10000]_2026-10-05-14-56-28.npz")
    #loaded_4 = load_evaluation_data("rewMod224ph2c2_fast_schedfullv1c3_52000000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 3, 3]_ep[10000]_kozMinMargin0_2026-10-05-15-55-56.npz")
    
    #loaded_4 = load_evaluation_data("rewMod224ph2c2b_fast_lstm8_kozMinMargin0_schedFullv1c3_53300000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 3, 3]_ep[10000]_2026-10-08-23-31-55.npz")
    
    #loaded_4 = load_evaluation_data("rewMod224ph2c2e_fast_lstm8_kozMinMargin0_schedFullv1c3_50000000_[90.0, 110.0, 0.0, 0.01, 3000, 25.0, 30.0, 3, 3]_ep[100]_2026-10-12-18-40-24.npz")

    #plot_actual_attitude(loaded_4[11])
    #plot_for_report(loaded_4[5],time_end=300)
    #multiple_loaded = [loaded_0, loaded_1, loaded_2, loaded_3, loaded_4]
    #evaluation_labels = ["0 KOZs", "1 KOZ"]
    evaluation_labels = ["0 KOZs", "1 KOZ", "2 KOZs", "3 KOZs", "3 KOZs ($\\Omega_{\\mathrm{margin}} = [0, \\theta]$°)"]
    #plot_pointing_accuracy_histogram([loaded_4], [evaluation_labels[4]])
