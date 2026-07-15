from pathlib import Path

"""
What:
    Validator for the config loader.
"""

def _validate_config_json_file(config_file: Path) -> None:

    #Check if the path is a Path object.
    if not isinstance(config_file, Path):
        raise TypeError("config_file must be a pathlib.Path object.")

    #Check if the configuration file exists.
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_file}")

    #Check if the configuration path is a file.
    if not config_file.is_file():
        raise ValueError(f"Expected a file but received: {config_file}")

    #Check if the file is a JSON file.
    if config_file.suffix.lower() != ".json":
        raise ValueError("Configuration file must have a '.json' extension.")


"""
What:
    Validate the loaded configuration.
"""

def _validate_loading_config_file(config: dict) -> None:

    #Check if the configuration is a dictionary.
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a dictionary.")

    #Check if the training configuration exists.
    if "training_config" not in config:
        raise KeyError("'training_config' section is missing.")

    training_config = config["training_config"]

    #Required configuration keys.
    required_keys = [
        "dataset_path",
        "trained_models_path",
        "test_size",
        "random_state"
    ]

    #Check if all required keys exist.
    for key in required_keys:

        if key not in training_config:
            raise KeyError(f"Missing required configuration key: '{key}'.")

    #Validate test_size.
    test_size = float(training_config["test_size"])

    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    #Validate random_state.
    random_state = int(training_config["random_state"])

    if random_state < 0:
        raise ValueError("random_state cannot be negative.")