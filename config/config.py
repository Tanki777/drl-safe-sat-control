"""
Config file.

Author: Cemal Yilmaz - 2026
"""
import json
import os

# Load JSON config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

with open(CONFIG_PATH, "r") as file:
    config_data = json.load(file)

class Config():
    """
    Config class to hold all configuration parameters for training, evaluation, visualization, and environment simulation.
    """

    class General():
        DEVICE = config_data["general"]["DEVICE"]

    class Training():
        SCHEDULE_FILE_NAME = config_data["training"]["SCHEDULE_FILE_NAME"]
        CONTINUE_TRAINING = config_data["training"]["CONTINUE_TRAINING"]
        USE_SAFETY_FILTER = config_data["training"]["USE_SAFETY_FILTER"]
        MODEL_NAME = config_data["training"]["MODEL_NAME"]

    class Evaluation():
        MODEL_NAME = config_data["evaluation"]["MODEL_NAME"]
        TIMESTEP = config_data["evaluation"]["TIMESTEP"]
        USE_SAFETY_FILTER = config_data["evaluation"]["USE_SAFETY_FILTER"]
        MAX_STEPS = config_data["evaluation"]["MAX_STEPS"]
        MIN_INITIAL_ERROR_ANGLE = config_data["evaluation"]["MIN_INITIAL_ERROR_ANGLE"]
        MAX_INITIAL_ERROR_ANGLE = config_data["evaluation"]["MAX_INITIAL_ERROR_ANGLE"]
        MIN_INITIAL_ANGULAR_VELOCITY = config_data["evaluation"]["MIN_INITIAL_ANGULAR_VELOCITY"]
        MAX_INITIAL_ANGULAR_VELOCITY = config_data["evaluation"]["MAX_INITIAL_ANGULAR_VELOCITY"]
        MIN_HALF_ANGLE_KOZ = config_data["evaluation"]["MIN_HALF_ANGLE_KOZ"]
        MAX_HALF_ANGLE_KOZ = config_data["evaluation"]["MAX_HALF_ANGLE_KOZ"]

    class Visualization():
        MODEL_NAME = config_data["visualization"]["MODEL_NAME"]
        TIMESTEP = config_data["visualization"]["TIMESTEP"]
        USE_SAFETY_FILTER = config_data["visualization"]["USE_SAFETY_FILTER"]
        MAX_STEPS = config_data["visualization"]["MAX_STEPS"]
        MIN_INITIAL_ERROR_ANGLE = config_data["visualization"]["MIN_INITIAL_ERROR_ANGLE"]
        MAX_INITIAL_ERROR_ANGLE = config_data["visualization"]["MAX_INITIAL_ERROR_ANGLE"]
        MIN_INITIAL_ANGULAR_VELOCITY = config_data["visualization"]["MIN_INITIAL_ANGULAR_VELOCITY"]
        MAX_INITIAL_ANGULAR_VELOCITY = config_data["visualization"]["MAX_INITIAL_ANGULAR_VELOCITY"]
        MIN_HALF_ANGLE_KOZ = config_data["visualization"]["MIN_HALF_ANGLE_KOZ"]
        MAX_HALF_ANGLE_KOZ = config_data["visualization"]["MAX_HALF_ANGLE_KOZ"]
        CREATE_VIDEO = config_data["visualization"]["CREATE_VIDEO"]

    class EnvSimulator():
        MAX_STEPS = config_data["env_simulator"]["MAX_STEPS"]
        MIN_INITIAL_ERROR_ANGLE = config_data["env_simulator"]["MIN_INITIAL_ERROR_ANGLE"]
        MAX_INITIAL_ERROR_ANGLE = config_data["env_simulator"]["MAX_INITIAL_ERROR_ANGLE"]
        MIN_INITIAL_ANGULAR_VELOCITY = config_data["env_simulator"]["MIN_INITIAL_ANGULAR_VELOCITY"]
        MAX_INITIAL_ANGULAR_VELOCITY = config_data["env_simulator"]["MAX_INITIAL_ANGULAR_VELOCITY"]
        MIN_HALF_ANGLE_KOZ = config_data["env_simulator"]["MIN_HALF_ANGLE_KOZ"]
        MAX_HALF_ANGLE_KOZ = config_data["env_simulator"]["MAX_HALF_ANGLE_KOZ"]

    
config = Config()