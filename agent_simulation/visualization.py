"""
Visualization tools.

Author: Cemal Yilmaz - 2026
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib
from stable_baselines3 import SAC
import numpy as np
import imageio.v2 as imageio
import time

# Add parent directory to path for imports (must be before local imports)
_drl_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _drl_repo_dir not in sys.path:
    sys.path.insert(0, _drl_repo_dir)

parent_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(parent_dir)
repo_parent_dir = os.path.dirname(repo_dir)
video_dir = os.path.join(repo_parent_dir, "videos")

# Create evaluation data directory if it doesn't exist
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

from agent_training.environment import SatDynEnv, SatThrusterEnv, BasiliskRWEnv, scale_torque, scale_angular_velocity_sat, scale_margin_koz, scale_angular_velocity_wheels
from agent_training.constants import dt
from agent_simulation.evaluation import create_evaluation_env, load_agent
from config.config import Config


def simulate_agent(model: SAC, eval_env: BasiliskRWEnv, max_steps: int, model_name: str, create_video: bool = False):
    """
    Simulate the agent in the evaluation environment.
    Args:
        model: The trained model.
        eval_env: The evaluation environment.
        max_steps: Maximum number of steps to simulate.
        model_name: The name of the model being simulated.
        create_video: Whether to create a video of the simulation.
    Returns:
        simulation_data: A dictionary containing the simulation data for plotting.
    """
    # Arrays for storing data
    times = np.linspace(0, max_steps/10, max_steps)  # Assuming dt=0.1s
    states = []
    torques = []
    rewards = []
    frames = []

    obs, _ = eval_env.reset()
    done = False
    normal_vector_koz = eval_env.normal_vector_koz
    half_angle_koz = eval_env.half_angle_koz
    min_margin_koz = 0
    cnt_Koz_violations = 0

    # Simulation loop
    while not done:
        action_agent, _states = model.predict(obs, deterministic=True)

        states.append(obs.copy())

        # Step the environment
        obs, reward, done, truncated, info = eval_env.step(action_agent) # action_filtered is the filter output if applied, else same as action_agent

        torques.append(action_agent)
        rewards.append(reward)
        
        # Render the environment and store the frame for video
        if create_video:
            frame = eval_env.render()
            frames.append(frame)

        min_margin_koz = eval_env.min_margin_koz
        cnt_Koz_violations = eval_env.entered_koz_count

    eval_env.close()

    # Save as MP4
    timestamp = time.time()
    output_path = os.path.join(video_dir, f"{model_name}_{timestamp}.mp4")

    if create_video:
        imageio.mimsave(output_path, frames, fps=30)
        print(f"Saved video to {output_path}")

    # Extract the solution for attitude (in terms of quaternion) and angular velocity
    states_array = np.array(states)
    torques_array = np.array(torques) * scale_torque
    rewards_array = np.array(rewards)

    # Calculate cumulative reward
    cumulative_rewards = np.cumsum(rewards_array)

    # store the norm of quaternions
    norm_q = np.linalg.norm(states_array[:, :4], axis=1)

    simulation_data = {
        "quaternion": states_array[:, :4],
        "quaternion_norm": norm_q,
        "torques": torques_array,
        "omega": states_array[:, 4:7]*scale_angular_velocity_sat,
        "omega_wheels": states_array[:, 7:10]*scale_angular_velocity_wheels,
        "rewards": rewards_array,
        "cumulative_rewards": cumulative_rewards,
        "times": times,
        "normal_vector_koz": normal_vector_koz,
        "half_angle_koz": half_angle_koz,
        # "margin_angles_koz": states_array[:, 20]*scale_margin_koz*180/np.pi,
        "margin_angles_koz": torques_array[:,0],
        "min_margin_koz": min_margin_koz,
        "cnt_Koz_violations": cnt_Koz_violations
        }
    
    return simulation_data


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
    ax1 = fig.add_subplot(241, projection="3d")
    
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
    ax2 = fig.add_subplot(242)
    ax2.plot(times[:len(rotation_angles_deg)], rotation_angles_deg, "purple", linewidth=3, label="Angle $\\phi$")
    ax2.axhline(y=0.25, color="r", linestyle="--", linewidth=2, label="Accuracy Threshold (0.25°)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Rotation Angle $\\phi$ (°)")
    ax2.set_title("Rotation Angle $\\phi$ vs Time\n(Used in reward function)")
    ax2.grid(True)
    ax2.legend()
    ax2.set_yscale("log")
    
    # Cumulative Reward vs Time
    ax3 = fig.add_subplot(243)  # New subplot for cumulative reward
    ax3.plot(times[:len(cumulative_rewards)], cumulative_rewards, "orange", linewidth=3, label="Cumulative Reward")
    ax3.plot(times[:len(rewards_array)], rewards_array, "lightcoral", alpha=0.6, linewidth=1, label="Step Reward")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Reward")
    ax3.set_title("Reward Evolution")
    ax3.grid(True)
    ax3.legend()
    
    # Plot quaternion
    ax4 = fig.add_subplot(244)
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
    ax5 = fig.add_subplot(245)
    ax5.plot(times, omega_x * (180 / np.pi), label="$\\omega_x$")
    ax5.plot(times, omega_y * (180 / np.pi), label="$\\omega_y$")
    ax5.plot(times, omega_z * (180 / np.pi), label="$\\omega_z$")
    ax5.set_title("Angular velocity")
    ax5.set_ylabel("$\\omega$ (deg/s)")
    ax5.legend()
    ax5.grid()

    # Plot wheel velocity
    ax5_2 = fig.add_subplot(246)
    ax5_2.plot(times, omega_w_x, label="$\\omega_{w,x}$")
    ax5_2.plot(times, omega_w_y, label="$\\omega_{w,y}$")
    ax5_2.plot(times, omega_w_z, label="$\\omega_{w,z}$")
    ax5_2.set_title("Wheel Velocity")
    ax5_2.set_ylabel("$\\omega$ (rad/s)")
    ax5_2.legend()
    ax5_2.grid()

    # Plot torque input
    ax6 = fig.add_subplot(247)
    ax6.plot(times, torques_array[:, 0], label="$\\tau_1$")
    ax6.plot(times, torques_array[:, 1], label="$\\tau_2$")
    ax6.plot(times, torques_array[:, 2], label="$\\tau_3$")
    ax6.set_title("Control torques")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("$\\tau$ (Nm)")
    ax6.legend()
    ax6.grid()

    # Plot keep out zone margin angle
    ax7 = fig.add_subplot(248)
    ax7.plot(times, margin_angles_koz, label="Margin Angle KOZ")
    ax7.set_title("Keep Out Zone Margin Angle")
    ax7.set_ylabel("Angle (degrees)")
    ax7.legend()
    ax7.grid()
    
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
    times = times[:int(time_end/dt)]
    q_0 = q_0[:int(time_end/dt)]
    q_1 = q_1[:int(time_end/dt)]
    q_2 = q_2[:int(time_end/dt)]
    q_3 = q_3[:int(time_end/dt)]
    omega_x = omega_x[:int(time_end/dt)]
    omega_y = omega_y[:int(time_end/dt)]
    omega_z = omega_z[:int(time_end/dt)]
    torques_array = torques_array[:int(time_end/dt)]
    margin_angles_koz = margin_angles_koz[:int(time_end/dt)]

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
        Config.Visualization.MAX_HALF_ANGLE_KOZ
    ]

    eval_env = create_evaluation_env(INITIAL_STATE, Config.Visualization.USE_SAFETY_FILTER)
    model = load_agent(Config.Visualization.MODEL_NAME, Config.Visualization.TIMESTEP, seed_random = True)

    """ Uncomment the lines below to run 1 simulation and plot the results. """
    simulation_data = simulate_agent(model, eval_env, Config.Visualization.MAX_STEPS, Config.Visualization.MODEL_NAME, create_video=Config.Visualization.CREATE_VIDEO)
    plot_actual_attitude(simulation_data)
    #plot_for_report(simulation_data, time_end=300)

    """ Uncomment the lines below if you have saved evaluation data (from evaluate_agent()) to load all the episodes.
        loaded contains ALL episodes, therefore in loaded[] should be the index of the episode you want to plot.
    """
    #loaded = load_evaluation_data("evaluation_test2.npz")
    #plot_actual_attitude(loaded[2])
    #plot_for_report(loaded[0],time_end=300)