"""
Evaluation tools for the agent.

Author: Cemal Yilmaz - 2026
"""
import numpy as np
import time
import os
import sys
import random

# Add parent directory to path for imports (must be before local imports)
_drl_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _drl_repo_dir not in sys.path:
    sys.path.insert(0, _drl_repo_dir)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent_training.constants import Constants
from agent_training.environment import BasiliskRWEnv, scale_torque, scale_angular_velocity_sat, scale_margin_koz
from config.config import Config

parent_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(parent_dir)
repo_parent_dir = os.path.dirname(repo_dir)
eval_data_dir = os.path.join(repo_parent_dir, "evaluation_data")

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


def create_evaluation_env(initial_state, model_name, timestep):
    """
    Create the evaluation environment.
    Args:
        initial_state: Initial state configuration for environment.
    Returns:
        eval_env: The created evaluation environment.
    """
    vec_normalize_path = f"models/{model_name}/{model_name}_{timestep}.pkl"

    eval_env = DummyVecEnv([lambda: BasiliskRWEnv(render_mode="rgb_array", initial_state=initial_state)])
    eval_env = VecNormalize.load(os.path.join(repo_parent_dir, vec_normalize_path), eval_env)
    eval_env.training = False  # Set to evaluation mode (no normalization updates)
    eval_env.norm_reward = False  # Do not normalize rewards during evaluation

    return eval_env


def evaluate_agent_worker(model_name: str, timestep: int, initial_state: list, max_steps: int, episodes: int, worker_id: int):
    """
    Worker function to evaluate agent for a subset of episodes.
    This function will be run in parallel by multiple processes.
    Args:
        model_name: Name of the model file (without .zip extension).
        timestep: Timestep for evaluation.
        initial_state: Initial state configuration for environment.
        max_steps: Maximum steps per episode.
        episodes: Number of episodes to run.
        worker_id: ID of this worker process.
    Returns:
        Dictionary containing evaluation results.
    """
    # Load model in worker process
    model = load_agent(model_name, timestep)
    
    # Create environment in worker process
    eval_env = create_evaluation_env(initial_state, model_name, timestep)
    
    koz_violation_episodes = 0
    ep_rewards = []
    err_angles_final = []
    ang_vels_final = []
    min_margins_koz = []
    cnts_koz_violations = []
    simulation_data = []

    for episode in range(episodes):
        print(f"Worker {worker_id}: Episode {episode+1}/{episodes}...", end="\r")
        times = np.linspace(0, max_steps/10, max_steps)  # Assuming dt=0.1s
        states = []
        torques = []
        rewards = []

        obs, _ = eval_env.reset()
        done = False
        reward_cum = 0 # for this episode

        # Simulation loop
        while not done:
            action, _states = model.predict(obs, deterministic=True)

            states.append(obs.copy())

            # Step the environment
            obs, reward, done, truncated, info = eval_env.step(action)
            torques.append(action.copy())
            rewards.append(reward)

            # Add up reward
            reward_cum += reward

        if eval_env.entered_koz_count > 0:
            koz_violation_episodes += 1

        # Convert to numpy arrays
        states_array = np.array(states)
        torques_array = np.array(torques) * scale_torque
        rewards_array = np.array(rewards)

        # Store episode data in a dictionary
        episode_data = {
            "quaternion": states_array[:, :4],
            "quaternion_norm": np.linalg.norm(states_array[:, :4], axis=1),
            "torques": torques_array,
            "omega": states_array[:, 4:7]*scale_angular_velocity_sat,
            "rewards": rewards_array,
            "cumulative_rewards": np.cumsum(rewards_array),
            "times": times,
            "normal_vector_koz": eval_env.normal_vector_koz,
            "half_angle_koz": eval_env.half_angle_koz,
            "margin_angles_koz": states_array[:, 10]*scale_margin_koz*180/np.pi,
            "min_margin_koz": eval_env.min_margin_koz,
            "cnt_Koz_violations": eval_env.entered_koz_count
        }

        simulation_data.append(episode_data)
        
        # Add episode results to evaluation lists
        min_margins_koz.append(eval_env.min_margin_koz*180/np.pi)  # in degrees
        cnts_koz_violations.append(eval_env.entered_koz_count)
        ep_rewards.append(reward_cum)

        q0_final = obs[0]
        err_angle_final = 2 * np.arccos(np.abs(q0_final)) * 180/np.pi  # final rotation angle in degrees
        err_angles_final.append(err_angle_final)

        ang_vel_final = obs[4:7] * scale_angular_velocity_sat * 180/np.pi  # final angular velocity in deg/s
        ang_vel_final_mag = np.sqrt(ang_vel_final[0]**2 + ang_vel_final[1]**2 + ang_vel_final[2]**2) # magnitude
        ang_vels_final.append(ang_vel_final_mag)

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


def evaluate_agent(model_name: str, timestep: int, initial_state: list, max_steps: int, episodes: int, num_workers: int = 4):
    """
    Simulate the agent in parallel using multiple processes and saves the data at the end.
    Args:
        model_name: Name of the model file (without .zip extension).
        timestep: Timestep for evaluation.
        initial_state: Initial state configuration for environment.
        max_steps: Maximum steps per episode.
        episodes: Total number of episodes to run.
        num_workers: Number of parallel worker processes (default: 4).
    """
    import multiprocessing as mp
    
    # Divide episodes among workers
    episodes_per_worker = episodes // num_workers
    remaining_episodes = episodes % num_workers
    
    # Create list of episode counts for each worker
    episode_counts = [episodes_per_worker] * num_workers
    for i in range(remaining_episodes):
        episode_counts[i] += 1
    
    print(f"Running {episodes} episodes across {num_workers} workers...")
    print(f"Episodes per worker: {episode_counts}")
    print()
    
    # Create pool of workers
    with mp.Pool(processes=num_workers) as pool:
        # Start all workers in parallel
        results = []
        for worker_id in range(num_workers):
            result = pool.apply_async(
                evaluate_agent_worker,
                args=(model_name, timestep, initial_state, max_steps, episode_counts[worker_id], worker_id)
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
    save_path = os.path.join(eval_data_dir, f"{model_name}_{timestep}_{initial_state}_ep[{episodes}]_{time_iso}.npz")
    np.savez(save_path, data=np.array(simulation_data), dtype=object)

    # Print results
    print()
    print(f"Violation rate: {koz_violation_episodes/episodes*100}%")
    print(f"Agent was within KOZ for these amount of steps per episode:")
    print(f"--- Mean: {np.mean(cnts_koz_violations_array):.2f}, Std: {np.std(cnts_koz_violations_array):.2f}, Min: {np.min(cnts_koz_violations_array)}, Max: {np.max(cnts_koz_violations_array)}")
    print(f"Minimum margin to KOZ (degrees):")
    print(f"--- Mean: {np.mean(min_margins_koz_array):.2f}, Std: {np.std(min_margins_koz_array):.2f}, Min: {np.min(min_margins_koz_array):.2f}, Max: {np.max(min_margins_koz_array):.2f}")
    print(f"Final rotation angle (degrees):")
    print(f"--- Mean: {np.mean(err_angles_final_array):.2f}, Std: {np.std(err_angles_final_array):.2f}, Min: {np.min(err_angles_final_array):.2f}, Max: {np.max(err_angles_final_array):.2f}")
    print(f"Final angular velocity (deg/s):")
    print(f"--- Mean: {np.mean(ang_vels_final_array):.4f}, Std: {np.std(ang_vels_final_array):.4f}, Min: {np.min(ang_vels_final_array):.4f}, Max: {np.max(ang_vels_final_array):.4f}")
    print(f"Rewards:")
    print(f"--- Mean: {np.mean(ep_rewards):.2f}, Std: {np.std(ep_rewards):.2f}, Min: {np.min(ep_rewards):.2f}, Max: {np.max(ep_rewards):.2f}")


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
            if err_angle <= 0.25:
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

        if episode_data["min_margin_koz"]*180/np.pi < 0.0:
            koz_margin_violated.append(episode_data["min_margin_koz"]*180/np.pi)
        else:
            koz_margin_not_violated.append(episode_data["min_margin_koz"]*180/np.pi)
       
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
        if np.mean(np.linalg.norm(episode_data["torques"], axis=1)) > 0.0004 and episode_data["cnt_Koz_violations"] > 0:
            #print(i,end=",")
            pass

        if 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 1.0 and episode_data["cnt_Koz_violations"] > 0:
            #print(i,end=",")
            pass
        
        # Off-border: no violation, target reached, large KOZ
        if (2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 0.5 and episode_data["min_margin_koz"]*180/np.pi > 10.0 
            and 2 * np.arccos(np.abs(episode_data["quaternion"][0,0])) * 180/np.pi < 100.0 and episode_data["cnt_Koz_violations"] == 0 and episode_data["half_angle_koz"]*180/np.pi > 25.0):
            #print(i,end=",")
            pass

        # Along-border: no violation, target reached, large KOZ
        if (2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 0.5 and episode_data["min_margin_koz"]*180/np.pi < 0.5 
            and 2 * np.arccos(np.abs(episode_data["quaternion"][0,0])) * 180/np.pi < 100.0 and episode_data["cnt_Koz_violations"] == 0 and episode_data["half_angle_koz"]*180/np.pi > 25.0):
            #print(i,end=",")
            pass
        
        # Stuck: Low episode reward, low control effort, far away from target
        if (episode_data["cumulative_rewards"][-1] < -100 and np.sum(np.linalg.norm(episode_data["torques"], axis=1)**2) < 0.0005
            and 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi > 30.0):
            #print(i,end=",")
            pass

        # Oscillation: Low episode reward, high control effort, close to target
        if (episode_data["cumulative_rewards"][-1] < -100 and np.sum(np.linalg.norm(episode_data["torques"], axis=1)**2) > 0.0008
            and 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 10.0):
            #print(i,end=",")
            pass

        if 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi > 30.0:
            #print(i,end=",")
            pass
        
        # Violation
        if episode_data["cnt_Koz_violations"] > 0:
            print(i,end=",")
            pass

        if episode_data["cumulative_rewards"][-1] < -2800:
            pass

        if 2 * np.arccos(np.abs(episode_data["quaternion"][0,0])) * 180/np.pi < 120 and episode_data["cumulative_rewards"][-1] > 120:
            #print(i,end=",")
            pass

        if np.linalg.norm(episode_data["omega"][-1]) * 180 / np.pi > 2.0:
            #print(i,end=",")
            pass

        if np.mean(np.linalg.norm(episode_data["omega"], axis=1)) * 180 / np.pi > 2.0:
            #print(i,end=",")
            pass

        if np.max(np.linalg.norm(episode_data["omega"], axis=1)) * 180 / np.pi > 11.0:
            #print(i,end=",")
            pass

        if i == 238:
            pass
        
        # if the last attitude error is above 5 degree and the last 50 omega norms are below 0.1 deg/s each
        omega_settled = True
        for omega in episode_data["omega"][-200:]:
            if np.linalg.norm(omega)*180/np.pi > 2.0:
                omega_settled = False
        if 2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi < 5.0 and omega_settled:
            #print(i,end=",")
            settled_count += 1
            settled_final_err.append(2 * np.arccos(np.abs(episode_data["quaternion"][-1,0])) * 180/np.pi)
            pass

        if not omega_settled:
            #print(i,end=",")
            pass
        
        if reward_min is None:
            reward_min = episode_data["cumulative_rewards"][-1]
            reward_min_idx = i
        if episode_data["cumulative_rewards"][-1] < reward_min:
            #print(i,end=",")
            reward_min = episode_data["cumulative_rewards"][-1]
            reward_min_idx = i
            pass
    print()
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
        Config.Evaluation.MAX_HALF_ANGLE_KOZ
    ]

    """ Uncomment the lines below to load saved evaluation data and calculate some metrics for multiple episodes.
    """
    #loaded = load_evaluation_data("rewPrak_sched_180_2_7400000_[0.0, 180.0, 0.0, 0.01, 3000, 0.0, 0.0]_filter[0]_ep[1000]_2026-04-24-06-35-37.npz")
    #calc_metrics(loaded)
   
    """ Uncomment evaluate_agent() below to simulate the agent over multiple episodes and save the data at the end. """
    t_start = time.time()
    # Run evaluation with possibly parallel workers and a defined number of episodes
    evaluate_agent(Config.Evaluation.MODEL_NAME, Config.Evaluation.TIMESTEP, INITIAL_STATE, Config.Evaluation.MAX_STEPS, episodes=10000, num_workers=8)
    t_end = time.time()

    print()
    print(f"Evaluation time: {t_end - t_start:.2f} seconds")