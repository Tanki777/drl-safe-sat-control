"""
Evaluation tools for the agent.

Author: Cemal Yilmaz - 2026
"""
import numpy as np
import time
import os
import sys
import random
import imageio.v2 as imageio

# Add parent directory to path for imports (must be before local imports)
_drl_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _drl_repo_dir not in sys.path:
    sys.path.insert(0, _drl_repo_dir)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent_training import environment as sat_env
from agent_training.constants import Constants
from agent_training.environment import BasiliskRWEnv, scale_torque, rotate_vector_by_quaternion_to_body_frame, LSTM
from APF_SMC.apf_smc_controller import APFSMCController, APFSMCParams
from config.config import Config

# Backward compatibility for models saved when trainer.py imported LSTM via
# `from environment import LSTM`. Cloudpickle stores that module path.
sys.modules.setdefault("environment", sat_env)

parent_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(parent_dir)
repo_parent_dir = os.path.dirname(repo_dir)
eval_data_dir = os.path.join(repo_parent_dir, "evaluation_data")
video_dir = os.path.join(repo_parent_dir, "videos")

# Create video directory if it doesn't exist
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

# Create evaluation data directory if it doesn't exist
if not os.path.exists(eval_data_dir):
    os.makedirs(eval_data_dir)


def load_agent(model_name: str, timestep: int, seed_random: bool = False):
    """
    Load the agent to visualize.
    Args:
        model_name: Name of the model file (without .zip extension).
        timestep: Timestep for evaluation.
    Returns:
        model: The loaded SAC model.
    """
    model_path = f"models/{model_name}/{model_name}_{timestep}.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = SAC.load(model_path, device=Config.General.DEVICE)

    if seed_random:
        random.seed(time.time())
        np.random.seed(int(time.time()))
        seed_rng = random.randint(0, 1000000000)
        #model.set_random_seed(seed_rng)
        model.action_space.seed(seed_rng)
        print(f"Model loaded with random seed: {seed_rng}")

    return model


def create_evaluation_env(phase_type, initial_state, model_name, timestep, controller: str = "DRL"):
    """
    Create the evaluation environment.
    Args:
        phase_type: String phase type, e.g. phase 1, 2, transition etc.
        initial_state: Initial state configuration for environment.
        model_name: Name of the DRL model. Unused for the APF controller.
        timestep: Training timestep of the DRL model. Unused for the APF controller.
        controller: Controller to use, either "DRL" (default) or "APF".
    Returns:
        eval_env: The created evaluation environment.
    """
    controller = controller.upper()
    if controller not in ("DRL", "APF"):
        raise ValueError(f"Unknown controller: {controller}. Use \"DRL\" or \"APF\".")

    eval_env = DummyVecEnv([lambda: BasiliskRWEnv(render_mode="rgb_array", initial_state=initial_state, phase_type=phase_type)])

    if controller == "DRL":
        vec_normalize_path = f"models/{model_name}/{model_name}_{timestep}_vecnormalize.pkl"
        eval_env = VecNormalize.load(os.path.join(repo_parent_dir, vec_normalize_path), eval_env)
        eval_env.training = False  # Set to evaluation mode (no normalization updates)
        eval_env.norm_reward = False  # Do not normalize rewards during evaluation

    return eval_env


def simulate_episode(model: SAC, eval_env: BasiliskRWEnv, max_steps: int, model_name: str, create_video: bool = False, controller: str = "DRL"):
    """
    Simulate the DRL or APF controller in the evaluation environment for one episode.
    Both controllers use the environment's Basilisk reaction-wheel dynamics.
    Args:
        model: The trained model. Can be None for the APF controller.
        eval_env: The evaluation environment.
        max_steps: Maximum number of steps to simulate.
        model_name: The name of the model being simulated.
        create_video: Whether to create a video of the simulation.
        controller: Controller to use, either "DRL" (default) or "APF".
    Returns:
        simulation_data: A dictionary containing the simulation data for plotting.
            The APF controller has no LSTM output (shape: steps x 1 x 0).
    """
    controller = controller.upper()
    if controller not in ("DRL", "APF"):
        raise ValueError(f"Unknown controller: {controller}. Use \"DRL\" or \"APF\".")
    if controller == "DRL" and model is None:
        raise ValueError("A trained model is required for the DRL controller.")
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero.")

    # Arrays for storing data
    states = []
    states_koz = []
    lstm_output = []
    torques = []
    rewards_arr = []
    frames = []

    obs = eval_env.reset()
    done = False
    normal_vector_koz_array = eval_env.get_attr("active_koz_config")[0]["normal_vector"]
    half_angle_koz_array = eval_env.get_attr("active_koz_config")[0]["half_angle_rad"]
    min_margin_koz = 0
    cnt_Koz_violations = 0
    state = eval_env.get_attr("state")[0]
    zones_mask = state["zones_mask"].copy()

    if controller == "DRL":
        gamma = model.gamma

        # Get LSTM
        policy = getattr(model, "policy")
        actor = getattr(policy, "actor")
        features_extractor_actor: LSTM = getattr(actor, "features_extractor")

    # APF controller
    else:
        gamma = eval_env.get_attr("discount_factor")[0]

        # Tuned parameters for this satellite.
        apf_params = APFSMCParams(
            beta2=1.0 * 0.005 * 2,
            beta1=1.0 * 5 * 0.005 * 2,
            alpha=0.5, # 2.0
            sigma=2e-4, # 0.15 + 0.2
            k=5e-3, # 4.0
            epsilon=5e-3 # 1e-4
            )

        # Use the same spacecraft, target attitude and boresight as the DRL environment.
        I = np.array(eval_env.get_attr("satellite")[0].hub.IHubPntBc_B, dtype=np.float64).reshape(3, 3)
        qd = np.array([1.0, 0.0, 0.0, 0.0]) # Target attitude quaternion.
        boresight = eval_env.get_attr("x_axis")[0]
        forbidden = [(normal_vector, boresight, half_angle)
                     for normal_vector, half_angle in zip(normal_vector_koz_array, half_angle_koz_array)]
        apf_controller = APFSMCController.from_cones(I, qd, forbidden=forbidden, params=apf_params)
        apf_controller.reset(state["satellite"][:4])

    # Simulation loop
    contains_inf_torque = False
    while not done and len(rewards_arr) < max_steps:
        state = eval_env.get_attr("state")[0]

        if controller == "DRL":
            action, _states = model.predict(obs, deterministic=True)
            lstm_output.append(features_extractor_actor.lstm_out.tolist())

        else:
            # APF returns body torque. Reaction wheel motor torque has the opposite sign.
            torque, _info = apf_controller.compute(state["satellite"][:4], state["satellite"][4:7])
            #print("torque=",torque)
            if not np.all(np.isfinite(torque)):
                contains_inf_torque = True
                
            action = np.clip(-torque / scale_torque, -1.0, 1.0)[None, :]

            # Store an empty pseudo LSTM output for the APF controller to maintain consistent data structure.
            lstm_output.append(np.empty((1, 0)))

        states.append(state["satellite"].copy())
        states_koz.append(state["zones"].copy())

        # Step the environment
        obs, rewards, dones, infos = eval_env.step(action)
        done = bool(dones[0])
        info = infos[0]

        torques.append(action[0].copy())
        rewards_arr.append(rewards[0])

        if done:
            min_margin_koz = info["custom_metrics/min_margin_koz"]
            cnt_Koz_violations = info["custom_metrics/entered_koz_count"]
        
        # Render the environment and store the frame for video
        if create_video:
            frame = eval_env.render()
            frames.append(frame)

    if not done:
        min_margin_koz = eval_env.get_attr("min_margin_koz")[0] * 180 / np.pi
        cnt_Koz_violations = eval_env.get_attr("entered_koz_count")[0]

    if contains_inf_torque:
        print("WARNING: APF torque is not finite.")

    # Save as MP4
    timestamp = time.time()
    output_path = os.path.join(video_dir, f"{model_name}_{timestamp}.mp4")

    if create_video:
        imageio.mimsave(output_path, frames, fps=30)
        print(f"Saved video to {output_path}")

    # Extract the solution for attitude (in terms of quaternion) and angular velocity
    states_array = np.array(states)
    states_koz_array = np.array(states_koz)
    lstm_output_array = np.array(lstm_output)
    torques_array = np.array(torques) * scale_torque
    rewards_array = np.array(rewards_arr)
    rewards_discounted_array = np.zeros_like(rewards_array)

    for idx, r in enumerate(rewards_array):
        rewards_discounted_array[idx] = rewards_array[idx] * gamma**idx

    # Calculate cumulative reward
    cumulative_rewards = np.cumsum(rewards_array)
    cumulative_rewards_discounted = np.cumsum(rewards_discounted_array)

    # store the norm of quaternions
    norm_q = np.linalg.norm(states_array[:, :4], axis=1)

    # Store time array in seconds for plotting.
    times = np.arange(len(rewards_array)) * Constants.TIME_DELTA

    simulation_data = {
        "quaternion": states_array[:, :4],
        "quaternion_norm": norm_q,
        "torques": torques_array,
        "omega": states_array[:, 4:7],
        "omega_wheels": states_array[:, 7:10],
        "rewards": rewards_array,
        "rewards_discounted": rewards_discounted_array,
        "cumulative_rewards": cumulative_rewards,
        "cumulative_rewards_discounted": cumulative_rewards_discounted,
        "times": times,
        "normal_vector_koz_array": normal_vector_koz_array,
        "half_angle_koz_array": half_angle_koz_array,
        "margin_angles_koz": states_koz_array[:, :, 0] * 180 / np.pi,
        "direction_koz": states_koz_array[:, :, 1:4],
        "min_margin_koz": min_margin_koz,
        "cnt_Koz_violations": cnt_Koz_violations,
        "lstm_output": lstm_output_array,
        "zones_mask": zones_mask
        }
    
    return simulation_data


def evaluate_agent_worker(model_name: str, timestep: int, phase_type: str, initial_state: list, max_steps: int, episodes: int, worker_id: int, controller: str = "DRL"):
    """
    Worker function to evaluate the DRL or APF controller for a subset of episodes.
    This function will be run in parallel by multiple processes.
    Args:
        model_name: Name of the model file (without .zip extension) or APF evaluation label.
        timestep: Training timestep of the DRL model. Unused for the APF controller.
        phase_type: String phase type e.g. phase 1, 2, transition etc.
        initial_state: Initial state configuration for environment.
        max_steps: Maximum steps per episode.
        episodes: Number of episodes to run.
        worker_id: ID of this worker process.
        controller: Controller to use, either "DRL" (default) or "APF".
    Returns:
        Dictionary containing evaluation results.
    """
    controller = controller.upper()

    # Load model in worker process for the DRL controller.
    model = load_agent(model_name, timestep) if controller == "DRL" else None
    
    # Create environment in worker process
    eval_env = create_evaluation_env(phase_type, initial_state, model_name, timestep, controller=controller)
    
    koz_violation_episodes = 0
    ep_rewards = []
    err_angles_final = []
    ang_vels_final = []
    min_margins_koz = []
    cnts_koz_violations = []
    simulation_data = []

    for episode in range(episodes):
        print(f"Worker {worker_id}: Episode {episode+1}/{episodes}...", end="\r")
        
        # Simulate one episode
        episode_data = simulate_episode(model, eval_env, max_steps, model_name, create_video=False, controller=controller)

        # Append data of this episode to all episodes data for this worker
        simulation_data.append(episode_data)
        
        # Add episode results to evaluation lists
        min_margins_koz.append(episode_data["min_margin_koz"])  # in degrees
        cnts_koz_violations.append(episode_data["cnt_Koz_violations"])
        ep_rewards.append(episode_data["cumulative_rewards"][-1]) # episode reward

        q0_final = episode_data["quaternion"][-1, 0]  # final quaternion scalar part
        err_angle_final = 2 * np.arccos(np.abs(q0_final)) * 180/np.pi  # final rotation angle in degrees
        err_angles_final.append(err_angle_final)

        ang_vel_final = episode_data["omega"][-1, :] * 180/np.pi  # final angular velocity in deg/s
        ang_vel_final_mag = np.sqrt(ang_vel_final[0]**2 + ang_vel_final[1]**2 + ang_vel_final[2]**2) # magnitude
        ang_vels_final.append(ang_vel_final_mag)

        # Track number of episodes with KOZ violation
        if episode_data["cnt_Koz_violations"] > 0:
            koz_violation_episodes += 1

    eval_env.close()
    
    # Return results from this worker
    return {
        "koz_violation_episodes": koz_violation_episodes,
        "ep_rewards": ep_rewards,
        "err_angles_final": err_angles_final,
        "ang_vels_final": ang_vels_final,
        "min_margins_koz": min_margins_koz,
        "cnts_koz_violations": cnts_koz_violations,
        "simulation_data": simulation_data
    }


def evaluate_agent(model_name: str, timestep: int, phase_type: str, initial_state: list, max_steps: int, episodes: int, num_workers: int = 4, controller: str = "DRL", apf_params: APFSMCParams = None):
    """
    Simulate the DRL or APF controller in parallel using multiple processes and save the data at the end.
    Args:
        model_name: Name of the model file (without .zip extension) or APF evaluation label.
        timestep: Training timestep of the DRL model. Unused for the APF controller.
        phase_type: String phase type e.g. phase 1, 2, transition etc.
        initial_state: Initial state configuration for environment.
        max_steps: Maximum steps per episode.
        episodes: Total number of episodes to run.
        num_workers: Number of parallel worker processes (default: 4).
        controller: Controller to use, either "DRL" (default) or "APF".
    """
    import multiprocessing as mp

    controller = controller.upper()
    if controller not in ("DRL", "APF"):
        raise ValueError(f"Unknown controller: {controller}. Use \"DRL\" or \"APF\".")
    
    # Divide episodes among workers
    episodes_per_worker = episodes // num_workers
    remaining_episodes = episodes % num_workers
    
    # Create list of episode counts for each worker
    episode_counts = [episodes_per_worker] * num_workers
    for i in range(remaining_episodes):
        episode_counts[i] += 1
    
    print(f"Running {episodes} episodes with {controller} across {num_workers} workers...")
    print(f"Episodes per worker: {episode_counts}")
    print()
    
    # Create pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Start all workers in parallel
        results = []
        for worker_id in range(num_workers):
            result = pool.apply_async(
                evaluate_agent_worker,
                args=(model_name, timestep, phase_type, initial_state, max_steps, episode_counts[worker_id], worker_id, controller)
            )
            results.append(result)
        
        # Wait for all workers to complete and collect results
        worker_results = [r.get() for r in results]
    
    # Combine results from all workers
    koz_violation_episodes = sum(r["koz_violation_episodes"] for r in worker_results)
    ep_rewards = []
    err_angles_final = []
    ang_vels_final = []
    min_margins_koz = []
    cnts_koz_violations = []
    simulation_data = []
    
    for r in worker_results:
        ep_rewards.extend(r["ep_rewards"])
        err_angles_final.extend(r["err_angles_final"])
        ang_vels_final.extend(r["ang_vels_final"])
        min_margins_koz.extend(r["min_margins_koz"])
        cnts_koz_violations.extend(r["cnts_koz_violations"])
        simulation_data.extend(r["simulation_data"])

    # Convert to numpy arrays
    err_angles_final_array = np.array(err_angles_final)
    ang_vels_final_array = np.array(ang_vels_final)
    min_margins_koz_array = np.array(min_margins_koz)
    cnts_koz_violations_array = np.array(cnts_koz_violations)

    # Save episode data
    time_iso = time.strftime("%Y-%m-%d-%H-%M-%S")
    if controller == "DRL":
        save_path = os.path.join(eval_data_dir, f"{model_name}_{timestep}_{initial_state}_ep[{episodes}]_{time_iso}.npz")
    else:
        save_path = os.path.join(eval_data_dir, f"{model_name}_APF_{initial_state}_ep[{episodes}]_{time_iso}.npz")
    np.savez(save_path, data=np.array(simulation_data), dtype=object)

    calc_metrics(simulation_data)
    # Print results
    # print()
    # print(f"Violation rate: {koz_violation_episodes/episodes*100}%")
    # print(f"Agent was within KOZ for these amount of steps per episode:")
    # print(f"--- Mean: {np.mean(cnts_koz_violations_array):.2f}, Std: {np.std(cnts_koz_violations_array):.2f}, Min: {np.min(cnts_koz_violations_array)}, Max: {np.max(cnts_koz_violations_array)}")
    # print(f"Minimum margin to KOZ (degrees):")
    # print(f"--- Mean: {np.mean(min_margins_koz_array):.2f}, Std: {np.std(min_margins_koz_array):.2f}, Min: {np.min(min_margins_koz_array):.2f}, Max: {np.max(min_margins_koz_array):.2f}")
    # print(f"Final rotation angle (degrees):")
    # print(f"--- Mean: {np.mean(err_angles_final_array):.2f}, Std: {np.std(err_angles_final_array):.2f}, Min: {np.min(err_angles_final_array):.2f}, Max: {np.max(err_angles_final_array):.2f}")
    # print(f"Final angular velocity (deg/s):")
    # print(f"--- Mean: {np.mean(ang_vels_final_array):.4f}, Std: {np.std(ang_vels_final_array):.4f}, Min: {np.min(ang_vels_final_array):.4f}, Max: {np.max(ang_vels_final_array):.4f}")
    # print(f"Rewards:")
    # print(f"--- Mean: {np.mean(ep_rewards):.2f}, Std: {np.std(ep_rewards):.2f}, Min: {np.min(ep_rewards):.2f}, Max: {np.max(ep_rewards):.2f}")


def calc_metrics(data: list):
    """
    Calculate some metrics across the episodes.
    Inputs:
        data: List of episode data dictionaries.
    """
  
    episode_rewards = []
    koz_margin_violated = []
    koz_margin_not_violated = []

    settled_count = 0
    settled_final_err = []
    settling_time = []
    control_effort = []
    

    for i, episode_data in enumerate(data):
        episode_rewards.append(episode_data["cumulative_rewards"][-1])

        # settling definition: settled if attitude error stays below target accuracy (0.25 deg) after some time (settling time).
        tmp_settling_time = 0
        tmp_settled = False

        for idx, quat in enumerate(episode_data["quaternion"]):
            err_angle = 2 * np.arccos(np.abs(quat[0])) * 180/np.pi
            if err_angle <= 1.0:
                if not tmp_settled:
                    tmp_settling_time = idx * Constants.TIME_DELTA
                tmp_settled = True
            else:
                tmp_settled = False

        # If settled, include these metrics
        if tmp_settled:
            settled_count += 1
            settling_time.append(tmp_settling_time)
            settled_final_err.append(2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi)
            control_effort.append(np.sum(np.linalg.norm(episode_data["torques"], axis=1)**2))

        if episode_data["min_margin_koz"] < 0.0:
            koz_margin_violated.append(episode_data["min_margin_koz"])
        else:
            koz_margin_not_violated.append(episode_data["min_margin_koz"])
       
    print()
    print("Episodes: ", len(data))
    print()
    print(f"KOZ margin violated: {len(koz_margin_violated)} episodes")
    print(f"KOZ margin not violated: {len(koz_margin_not_violated)} episodes")
    if len(koz_margin_violated) > 0:
        print(f"Mean KOZ margin violated: {np.mean(koz_margin_violated)} +- {np.std(koz_margin_violated)} degrees")
    if len(koz_margin_not_violated) > 0:
        print(f"Mean KOZ margin not violated: {np.mean(koz_margin_not_violated)} +- {np.std(koz_margin_not_violated)} degrees")
    print()
    print(f"Settled episodes: {settled_count}")
    print(f"Mean final error of settled episodes: {np.mean(settled_final_err)} +- {np.std(settled_final_err)} degrees")
    print(f"Mean settling time: {np.mean(settling_time)} +- {np.std(settling_time)} seconds")
    print(f"Mean control effort: {np.mean(control_effort)} +- {np.std(control_effort)} N^2m^2*s")
    print(f"Rewards: {np.mean(episode_rewards)} +- {np.std(episode_rewards)}")
    print("--------------------------------------------------")


def load_evaluation_data(file_name: str):
    """
    Load the simulation data.
    The part inside of the for loop can be modified to find specific episodes you are interested in.
    For example, episodes where the mean torque was above some threshold, episodes where the safety constraint was violated, etc.
    You could then print the index (i) of these episodes to the terminal.

    Args:
        file_name: Name of the .npz file containing the evaluation data. Must be located in the evaluation_data directory.
    Returns:
        data: The loaded simulation data.
    """
    file_path = os.path.join(eval_data_dir, file_name)
    _data = np.load(file_path, allow_pickle=True)
    data = _data["data"].tolist()  # Convert back to list of dicts

    reward_min = None
    reward_min_idx = None
    settled_count = 0
    settled_final_err = []

    for i, episode_data in enumerate(data):
    #     if np.mean(np.linalg.norm(episode_data["torques"], axis=1)) > 0.0004 and episode_data["cnt_Koz_violations"] > 0:
    #         #print(i,end=",")
    #         pass

    #     if 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 1.0 and episode_data["cnt_Koz_violations"] > 0:
    #         #print(i,end=",")
    #         pass
        
    #     # Off-border: no violation, target reached, large KOZ
    #     if (2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 0.5 and episode_data["min_margin_koz"] > 10.0 
    #         and 2 * np.arccos(np.abs(episode_data["quaternion"][0,0])) * 180/np.pi < 100.0 and episode_data["cnt_Koz_violations"] == 0 and episode_data["half_angle_koz_array"]*180/np.pi > 25.0):
    #         #print(i,end=",")
    #         pass

    #     # Along-border: no violation, target reached, large KOZ
    #     if (2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 0.5 and episode_data["min_margin_koz"] < 0.5 
    #         and 2 * np.arccos(np.abs(episode_data["quaternion"][0,0])) * 180/np.pi < 100.0 and episode_data["cnt_Koz_violations"] == 0 and episode_data["half_angle_koz_array"]*180/np.pi > 25.0):
    #         #print(i,end=",")
    #         pass
        
    #     # Stuck: Low episode reward, low control effort, far away from target
    #     if (episode_data["cumulative_rewards"][-1] < -100 and np.sum(np.linalg.norm(episode_data["torques"], axis=1)**2) < 0.0005
    #         and 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi > 30.0):
    #         #print(i,end=",")
    #         pass

    #     # Oscillation: Low episode reward, high control effort, close to target
    #     if (episode_data["cumulative_rewards"][-1] < -100 and np.sum(np.linalg.norm(episode_data["torques"], axis=1)**2) > 0.0008
    #         and 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 10.0):
    #         #print(i,end=",")
    #         pass

    #     if 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 0.25:
    #         #print(i,end=",")
    #         pass
        
    #     # Violation
        if episode_data["cnt_Koz_violations"] > 0:
            print(i,end=",")
            pass

    #     if episode_data["cumulative_rewards"][-1] < -2800:
    #         pass

    #     if 2 * np.arccos(np.abs(episode_data["quaternion"][0,0])) * 180/np.pi < 120 and episode_data["cumulative_rewards"][-1] > 120:
    #         #print(i,end=",")
    #         pass

    #     if np.linalg.norm(episode_data["omega"][-1]) * 180 / np.pi > 2.0:
    #         #print(i,end=",")
    #         pass

    #     if np.mean(np.linalg.norm(episode_data["omega"], axis=1)) * 180 / np.pi > 2.0:
    #         #print(i,end=",")
    #         pass

    #     if np.max(np.linalg.norm(episode_data["omega"], axis=1)) * 180 / np.pi > 11.0:
    #         #print(i,end=",")
    #         pass

    #     if i == 238:
    #         pass
        
    #     # if the last attitude error is above 5 degree and the last 50 omega norms are below 0.1 deg/s each
    #     omega_settled = True
    #     for omega in episode_data["omega"][-200:]:
    #         if np.linalg.norm(omega)*180/np.pi > 2.0:
    #             omega_settled = False
    #     if 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 5.0 and omega_settled:
    #         #print(i,end=",")
    #         settled_count += 1
    #         settled_final_err.append(2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi)
    #         pass

    #     if not omega_settled:
    #         #print(i,end=",")
    #         pass
        
    #     if reward_min is None:
    #         reward_min = episode_data["cumulative_rewards"][-1]
    #         reward_min_idx = i
    #     if episode_data["cumulative_rewards"][-1] < reward_min:
    #         #print(i,end=",")
    #         reward_min = episode_data["cumulative_rewards"][-1]
    #         reward_min_idx = i
    #         pass
    # print()
    return data


### MAIN ###
if __name__ == "__main__":

    # Set initial state for evaluation environment
    INITIAL_STATE = [
        Config.Evaluation.MIN_INITIAL_ERROR_ANGLE,
        Config.Evaluation.MAX_INITIAL_ERROR_ANGLE,
        Config.Evaluation.MIN_INITIAL_ANGULAR_VELOCITY,
        Config.Evaluation.MAX_INITIAL_ANGULAR_VELOCITY,
        Config.Evaluation.MAX_STEPS,
        Config.Evaluation.MIN_HALF_ANGLE_KOZ,
        Config.Evaluation.MAX_HALF_ANGLE_KOZ,
        Config.Evaluation.MIN_NR_KOZ,
        Config.Evaluation.MAX_NR_KOZ,
    ]

    PHASE_TYPE = Config.Evaluation.PHASE_TYPE

    """ Uncomment the lines below to load saved evaluation data and calculate some metrics for multiple episodes.
    """
    #loaded = load_evaluation_data("rewMod224ph2c2_fast_kozMinMargin0_schedfullv1c3_48400000_[90.0, 180.0, 0.0, 0.01, 3000, 15.0, 30.0, 3, 3]_ep[10000]_2026-10-05-20-15-06.npz")
    #calc_metrics(loaded)
   
    """ Uncomment evaluate_agent() below to simulate the agent over multiple episodes and save the data at the end. """
    t_start = time.time()
    # Run evaluation with possibly parallel workers and a defined number of episodes
    evaluate_agent(Config.Evaluation.MODEL_NAME, Config.Evaluation.TIMESTEP, PHASE_TYPE, INITIAL_STATE, Config.Evaluation.MAX_STEPS, episodes=100, num_workers=8, controller=Config.Evaluation.CONTROLLER)
    t_end = time.time()

    print()
    print(f"Evaluation time: {t_end - t_start:.2f} seconds")
