from src.model.model_config.model_config_schema import ModelConfig
from src.model.model_config.mode_config_validator import _validate_config_json_file, _validate_loading_config_file
from pathlib import Path
import json


"""
What:
    Load the model trainer configuration from JSON.
"""

def load_config() -> ModelConfig:
    
    #Initiate the root folder.
    root_folder = Path(__file__).resolve().parents[2]

    #Initiate the config folder.
    config_file = root_folder / "model" / "model_config" / "model_config.json"

    #Validate the configuration file.
    _validate_config_json_file(config_file)
    
    "MATCH ANY ERROR WHILE LOADING THE JSON FILE."
    try:
        with open(config_file, "r") as file:
            config = json.load(file)

    except PermissionError as error:
        raise PermissionError(f"Cannot read configuration file: {config_file}") from error

    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON format: {config_file}") from error
    
    #Validate the loaded configuration.
    _validate_loading_config_file(config)

    #Return the configurations.
    return ModelConfig(
        dataset_file_path = config["training_config"]["dataset_path"],
        trained_models_path = config["training_config"]["trained_models_path"],
        test_size= float(config["training_config"]["test_size"]),
        random_state= int(config["training_config"]["random_state"])
    )
