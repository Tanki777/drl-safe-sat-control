"""
Barebone training script. Includes environment creation, model loading/creation, training loop with custom callback for logging and saving, and TensorBoard server management.

Author: Cemal Yilmaz - 2026
"""

import os
import sys
import time
import datetime
import subprocess

# Add parent directory to path for imports (must be before local imports)
_drl_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _drl_repo_dir not in sys.path:
    sys.path.insert(0, _drl_repo_dir)

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

from agent_training import environment as sat_env
from agent_training.environment import LSTM
from config.config import Config

# Backward compatibility for checkpoints saved when this script imported LSTM
# via `from environment import LSTM`.
sys.modules.setdefault("environment", sat_env)


# Terminal colors
RED_START = "\033[91m"
GREEN_START = "\033[92m"
YELLOW_START = "\033[93m"
COLOR_END = "\033[0m"

# Get the log and models path
parent_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(parent_dir)
repo_parent_dir = os.path.dirname(repo_dir)
log_path = os.path.join(repo_parent_dir, "tensorboard")
if not os.path.exists(log_path):
    os.makedirs(log_path)
models_path = os.path.join(repo_parent_dir, "models")
replay_buffer_path = os.path.join(repo_parent_dir, "models_replay_buffers")

class CustomCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard and saving the model at regular intervals.
    Logs custom metrics from the environment info dict at the end of each rollout and saves the model every `save_interval` timesteps. Also logs hyperparameters at the start of training.
    """
    def __init__(self, check_freq, save_interval, model_name, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_interval = save_interval
        self.model_name = model_name
        
        # Custom metrics accumulators
        self.custom_metrics = {
            "initial_error_angle": [],
            "initial_angular_velocity": [],
            "final_error_angle": [],
            "settling_time": [],
            "avg_torque": [],
            "max_torque": [],
            "settled": [],
            "min_margin_koz": [],
            "entered_koz_count": []
        }

    def _log_network_lstm(self, lstm: LSTM, network_name: str):
        """
        Logs each weight and bias of the given LSTM.
        Important: the sizes used are only valid for an LSTM which is not bidirectional and where proj_size = 0.
        For more details, see PyTorch's LSTM class.

        Args:
            lstm: The LSTM features extractor.
            network_name: The name of the network the LSTM belongs to.
        """

        input_size = lstm.input_size
        hidden_size = lstm.hidden_size

        # Go through all parameters
        for param_name, param in lstm.named_parameters():
            # Get the parameter tensor
            param_value = param.detach().cpu()

            # Base metric name
            metric_name = f"custom_lstm/{network_name}/{param_name}"

            # Separate weights and biases per gate and convert from tensor --> list.
            input_values = param_value[0:hidden_size].tolist()
            forget_values = param_value[hidden_size:2*hidden_size].tolist()
            cell_values = param_value[2*hidden_size:3*hidden_size].tolist()
            output_values = param_value[3*hidden_size:4*hidden_size].tolist()

            # The input-hidden weights per gate are of shape (hidden_size, layer_input_size) 
            if param_name.startswith("lstm.weight_ih"):
                for h in range(hidden_size):

                    # The first layer (index 0) has layer_input_size = input_size
                    if param_name.startswith("lstm.weight_ih_l0"):
                        layer_input_size = input_size
                    # Higher layers have layer_input_size = hidden_size
                    else:
                        layer_input_size = hidden_size

                    for i in range(layer_input_size):
                        self.logger.record(f"{metric_name}_input_h{h}i{i}", input_values[h][i])
                        self.logger.record(f"{metric_name}_forget_h{h}i{i}", forget_values[h][i])
                        self.logger.record(f"{metric_name}_cell_h{h}i{i}", cell_values[h][i])
                        self.logger.record(f"{metric_name}_output_h{h}i{i}", output_values[h][i])

            # The hidden-hidden weights per gate are of shape (hidden_size, hidden_size)
            elif param_name.startswith("lstm.weight_hh"):
                for h1 in range(hidden_size):
                    for h2 in range(hidden_size):
                        self.logger.record(f"{metric_name}_input_h{h1}h{h2}", input_values[h1][h2])
                        self.logger.record(f"{metric_name}_forget_h{h1}h{h2}", forget_values[h1][h2])
                        self.logger.record(f"{metric_name}_cell_h{h1}h{h2}", cell_values[h1][h2])
                        self.logger.record(f"{metric_name}_output_h{h1}h{h2}", output_values[h1][h2])

            # Both input-hidden and hidden-hidden biases per gate are of shape (hidden_size)
            elif param_name.startswith("lstm.bias_ih") or param_name.startswith("lstm.bias_hh"):

                for h in range(hidden_size):
                    self.logger.record(f"{metric_name}_input_h{h}", input_values[h])
                    self.logger.record(f"{metric_name}_forget_h{h}", forget_values[h])
                    self.logger.record(f"{metric_name}_cell_h{h}", cell_values[h])
                    self.logger.record(f"{metric_name}_output_h{h}", output_values[h])

    def _log_lstm_parameters(self):
        """
        Log LSTM weights and biases in Tensorboard.
        """

        policy = getattr(self.model, "policy")
        
        actor = getattr(policy, "actor")
        critic = getattr(policy, "critic")

        features_extractor_actor = getattr(actor, "features_extractor")
        features_extractor_critic = getattr(critic, "features_extractor")

        self._log_network_lstm(features_extractor_actor, "actor")
        self._log_network_lstm(features_extractor_critic, "critic")

           

    def _on_training_start(self):
        # Define the metrics that will appear in the HPARAMS Tensorboard tab by referencing their key
        hparam_dict = {
                "algorithm": self.model.__class__.__name__,
                "learning rate": self.model.learning_rate,
                "tau": self.model.tau,
                "gamma": self.model.gamma
        }
        
        # Tensorbaord will find and display metrics from the SCALARS tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0
        }
        
        
        self.logger.record("hparams", HParam(hparam_dict, metric_dict), exclude=("stdout", "log", "json", "csv"))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        is_end_of_episode = False
            
        # Collect custom metrics from episode endings
        for info in infos:
            if isinstance(info, dict):
                # Check if this info contains custom metrics (episode ended)
                has_custom_metrics = any(key.startswith("custom_metrics/") for key in info.keys())

                if has_custom_metrics:
                    is_end_of_episode = True

                    for metric_name in self.custom_metrics.keys():
                        metric_key = f"custom_metrics/{metric_name}"

                        if metric_key in info:
                            self.custom_metrics[metric_name].append(info[metric_key])
                  
        if is_end_of_episode:
            self._log_custom_metrics()
            self._log_lstm_parameters()

        # Save model every save_interval total timesteps
        if self.num_timesteps % self.save_interval == 0:
            # Save the model
            save_model(self.model, self.model_name, save_latest=False)
            
        return True
    
    
    def _log_custom_metrics(self):
        """Log accumulated custom metrics to TensorBoard"""
        for metric_name, values in self.custom_metrics.items():
            if values:  # Only log if we have data
                if metric_name == "settling_time":
                    # For settling_time, ignore non settled case when calculating mean
                    non_none_values = [v for v in values if v]
                    if non_none_values:
                        mean_value = sum(non_none_values) / len(non_none_values)
                    else:
                        mean_value = None
                elif metric_name == "min_margin_koz":
                    # For min_margin_koz, take the minimum value
                    mean_value = min(values)
                else:
                    mean_value = sum(values) / len(values)
                # Log to TensorBoard using the logger
                if mean_value:
                    self.logger.record(f"custom/{metric_name}_mean", mean_value)
            
                # Log max values
                if metric_name in ["final_error_angle", "settling_time", "initial_error_angle"]:
                    non_none_values = [v for v in values if v]

                    # Only log max if there is at least one non-None value
                    if non_none_values:
                        max_value = max(non_none_values)
                        self.logger.record(f"custom/{metric_name}_max", max_value)

                # Clear the accumulated values
                self.custom_metrics[metric_name] = []

        


def start_tensorboard():
    """Start TensorBoard server in background, access with http://localhost:6006"""
    print("|")
    print(f"|---{YELLOW_START}Looking for TensorBoard logs in: {log_path}{COLOR_END}")

    # Check if the log directory exists
    if not os.path.exists(log_path):
        print(f"|-----{RED_START}Log directory does not exist: {log_path}{COLOR_END}")
        print(f"|-----{RED_START}Available directories:{COLOR_END}")
        for item in os.listdir(repo_dir):
            item_path = os.path.join(repo_dir, item)
            if os.path.isdir(item_path) and "tensorboard" in item.lower():
                print(f"|-------{RED_START}{item}{COLOR_END}")
    else:
        print(f"|-----{GREEN_START}Log directory found: {log_path}{COLOR_END}")

    try:
        print("|")
        print(f"|---{YELLOW_START}Starting TensorBoard server...{COLOR_END}")
        
        # Start TensorBoard process in background
        process = subprocess.Popen([
            "tensorboard",
            f"--logdir={log_path}",
            "--port=6006",
            "--host=localhost"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("|-----Access TensorBoard at: http://localhost:6006")
        print("|-----Press Ctrl+C to stop the server")
        
        # Wait a moment for TensorBoard to start
        time.sleep(3)

        return process
    
    except FileNotFoundError:
        print(f"|-----{RED_START}TensorBoard not found.{COLOR_END}")


def stop_tensorboard(process):
    """Stop TensorBoard server"""
    if process is not None:
        try:
            print(f"|---{YELLOW_START}Stopping TensorBoard server...{COLOR_END}")
            process.terminate()
            process.wait(timeout=5)
            print(f"|-----{GREEN_START}TensorBoard server stopped{COLOR_END}")
        except subprocess.TimeoutExpired:
            print(f"|-----{RED_START}Force killing TensorBoard server...{COLOR_END}")
            process.kill()
            process.wait()
            print(f"|-----{RED_START}TensorBoard server force stopped{COLOR_END}")
        except Exception as e:
            print(f"|-----{RED_START}Error stopping TensorBoard: {e}{COLOR_END}")


def create_environment(model_name, initial_state=None, phase_name=None):
    """
    Create the training environment
    Args:
        model_name: Name of the model for tensorboard logging
        initial_state: Optional initial state to reset the environment to
        phase_name: Optional phase name to include in monitor log filename
    Returns:
        env: The created and wrapped environment
    """
    print("--------------------------------------------")
    print(f"|---{YELLOW_START}Creating environment...{COLOR_END}")
    
    # Create monitor directory for episode logging
    monitor_dir = os.path.join(repo_parent_dir, "monitor_logs")
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)

    # Track custom metrics in monitor wrapper
    custom_info_keywords = (
        "custom_metrics/initial_error_angle",
        "custom_metrics/initial_angular_velocity", 
        "custom_metrics/final_error_angle",
        "custom_metrics/settling_time",
        "custom_metrics/avg_torque",
        "custom_metrics/max_torque",
        "custom_metrics/max_torque_prev",
        "custom_metrics/settled",
    )

    # If phase name is available, use it in the monitor log filename
    if phase_name:
        phase_name = phase_name.replace(" ", "_").replace(":", "")  # whitespace and colon can cause issues in filenames
        monitor_log_file = os.path.join(monitor_dir, f"{model_name}_{phase_name}")
    
    # If phase name not available, use timestamp
    else:
        timestamp = int(time.time())
        monitor_log_file = os.path.join(monitor_dir, f"{model_name}_{timestamp}")

    # Create vectorized environment
    monitor_wrapper_kwargs = dict(info_keywords=custom_info_keywords)

    if initial_state is not None:
        # Need to use a lambda to pass initial_state parameter
        env = make_vec_env(lambda: sat_env.BasiliskRWEnv(initial_state=initial_state), n_envs=8, vec_env_cls=DummyVecEnv, monitor_dir=monitor_log_file, monitor_kwargs=monitor_wrapper_kwargs)

    else:
        env = make_vec_env(sat_env.BasiliskRWEnv, n_envs=8, vec_env_cls=DummyVecEnv, monitor_dir=monitor_log_file, monitor_kwargs=monitor_wrapper_kwargs)
    
    #env = VecNormalize(env)

    return env


def create_or_load_model(env, continue_training, model_name, log_path):
    """
    Create a new SAC model or load an existing one depending on continue_training.
    Args:
        env: The training environment
        continue_training: Boolean indicating whether to continue training from an existing model
        model_name: Name of the model file
        log_path: Path for tensorboard logs
    Returns:
        res: A tuple containing:
        model: The created or loaded SAC model
        save_path: Path where the model will be saved
        latest_model_path: Path to the latest saved model
    """
    # Ensure the directories exist
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if not os.path.exists(os.path.join(models_path, model_name)):
        os.makedirs(os.path.join(models_path, model_name))

    if not os.path.exists(replay_buffer_path):
        os.makedirs(replay_buffer_path)

    if not os.path.exists(os.path.join(replay_buffer_path, model_name)):
        os.makedirs(os.path.join(replay_buffer_path, model_name))

    # Setting the path to save the model
    save_path = os.path.join(models_path, model_name)
    latest_model_path = os.path.join(models_path, model_name, f"{model_name}_latest.zip")

    # Setting the path to save the replay buffer
    latest_replay_buffer_path = os.path.join(replay_buffer_path, model_name, f"{model_name}_latest.pkl")

    # Setting the path to save the VecNormalize data
    latest_norm_path = os.path.join(models_path, model_name, f"{model_name}_latest_vecnormalize.pkl")

    print("|")
    print(f"|---{YELLOW_START}Creating/Loading the agent...{COLOR_END}")
    
    # Try to load existing model if CONTINUE_TRAINING is True
    if continue_training and not os.path.exists(latest_model_path):
        print(f"|-----{RED_START} Latest model path not found: {latest_model_path}")

    if continue_training and os.path.exists(latest_model_path):
        print(f"|-----{YELLOW_START}Loading existing model from: {latest_model_path}{COLOR_END}")

        try:
            # Load VecNormalize data into env
            env = VecNormalize.load(latest_norm_path, env)
            env.training = True
            env.norm_reward = False

            model = SAC.load(latest_model_path, device=Config.General.DEVICE)
            model.set_env(env) 
            print(f"|-----{GREEN_START}Successfully loaded existing model.{COLOR_END}")
            print(f"|-----Previous total timesteps: {model.num_timesteps}")
            
            # Update tensorboard log directory to continue logging
            model.tensorboard_log = log_path
            
        except Exception as e:
            print(f"|-----{RED_START}Failed to load model: {e}{COLOR_END}")
            print(f"|-----{YELLOW_START}Creating new model instead...{COLOR_END}")
            continue_training = False

        # Try to load existing replay buffer
        if os.path.exists(latest_replay_buffer_path):
            try:
                model.load_replay_buffer(latest_replay_buffer_path)
                print(f"|-----{GREEN_START}Successfully loaded existing replay buffer.{COLOR_END}")
                print(f"DEBUG: Loaded replay buffer with {model.replay_buffer.size()} transitions.")
            except Exception as e:
                print(f"|-----{RED_START}Failed to load replay buffer: {e}{COLOR_END}")
                print(f"|-----{YELLOW_START}Continuing without loading replay buffer...{COLOR_END}")
    
    # Create new model if not loading existing one
    if not continue_training or not os.path.exists(latest_model_path):
        print(f"|-----{YELLOW_START}Creating new model from scratch...{COLOR_END}")

        # Add normalization wrapper
        env = VecNormalize(env, norm_reward=False, norm_obs_keys=["satellite", "zones"]) # Do not normalize the zones mask

        # Create LSTM extractor
        policy_kwargs = dict(
            features_extractor_class=LSTM,
            features_extractor_kwargs=dict(
                sat_obs_dim=10,
                lstm_out_dim=1
            )
        )

        model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, learning_rate=1e-4, buffer_size=1_000_000, learning_starts=10_000, batch_size=256, gradient_steps=-1,verbose=1, device=Config.General.DEVICE,
                    tensorboard_log=log_path, seed=None, ent_coef='auto')  # Use absolute path for consistency
        
    return model, save_path, latest_model_path


def train_agent(model, total_timesteps, check_freq, save_interval, model_name):
    """
    Train the agent model with custom callback for logging and saving.
    Args:
        model: The SAC model to train
        total_timesteps: Number of timesteps to train
        check_freq: Frequency of callback checks
        save_interval: Interval of timesteps to save the model
        model_name: Name of the model for tensorboard logging
    Returns:
        model: The trained SAC model
    """
    custom_callback = CustomCallback(check_freq=check_freq, save_interval=save_interval, model_name=model_name)

    print("|")
    print(f"|---{YELLOW_START}Start training the agent...{COLOR_END}")
    start_time = time.time()
    
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=custom_callback, tb_log_name=model_name, reset_num_timesteps=False)
    end_time = time.time()
    
    # Print training duration in a formatted way
    training_duration = datetime.timedelta(seconds=end_time - start_time)
    
    # Convert timedelta to datetime for formatting
    duration_datetime = datetime.datetime(1900, 1, 1) + training_duration
    formatted_duration = duration_datetime.strftime("%H:%M:%S")

    print(f"|-----Training completed in: {formatted_duration}")
    print(f"|-----Current total timesteps: {model.num_timesteps}")

    return model


def save_model(model, model_name, save_latest=True):
    """
    Save the trained model.
    Args:
        model: The trained SAC model
        model_name: Name of the model
        save_latest: Whether to also save the model as the latest version for future loading
    """
    # Save the updated model
    print("|")
    print(f"|---{YELLOW_START}Saving improved model...{COLOR_END}")
    
    # Save model backup
    _model_path = os.path.join(models_path, model_name)
    backup_path = os.path.join(_model_path, f"{model_name}_{model.num_timesteps}")
    model.save(backup_path)

    # Save replay buffer
    _replay_path = os.path.join(replay_buffer_path, model_name)
    backup_path_replay = os.path.join(_replay_path, f"{model_name}_{model.num_timesteps}")
    model.save_replay_buffer(backup_path_replay)

    # Save normalization data for the VecNormalize wrapper
    if isinstance(model.get_env(), VecNormalize):
        norm_path = os.path.join(models_path, model_name, f"{model_name}_{model.num_timesteps}_vecnormalize.pkl")
        VecNormalize.save(model.get_env(), norm_path)
        print(f"|-----{GREEN_START}VecNormalize data saved to: {norm_path}{COLOR_END}")

        if save_latest:
            latest_norm_path = os.path.join(models_path, model_name, f"{model_name}_latest_vecnormalize.pkl")
            VecNormalize.save(model.get_env(), latest_norm_path)
            print(f"|-----{GREEN_START}Latest VecNormalize data saved to: {latest_norm_path}{COLOR_END}")
    
    if save_latest:
        # Save as latest model (for next session)
        latest_model_path = os.path.join(models_path, model_name, f"{model_name}_latest")
        model.save(latest_model_path)

        # Save replay buffer as latest
        latest_replay_path = os.path.join(replay_buffer_path, model_name, f"{model_name}_latest")
        model.save_replay_buffer(latest_replay_path)
    
    print(f"|-----{GREEN_START}Model saved to:{COLOR_END}")
    if save_latest:
        print(f"|-------Latest: {latest_model_path}")
    print(f"|-------Backup: {backup_path}")
    print(f"|-----{GREEN_START}Replay buffer saved to:{COLOR_END}")
    if save_latest:
        print(f"|-------Latest: {latest_replay_path}")
    print(f"|-------Backup: {backup_path_replay}")
