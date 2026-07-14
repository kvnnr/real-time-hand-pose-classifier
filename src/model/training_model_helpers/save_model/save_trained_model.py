from src.model.training_model_helpers.save_model.save_model_validator import _validate_input, _validate_output
from sklearn.base import ClassifierMixin
from pathlib import Path

import joblib


"""
What:
    Save the chosen trained model.

Responsibilities:
    Save the trained model in the "trained_models" folder.
    as "pose_classifier.joblib".

Input:
    trained_model: ClassifierMixin
        The trained model with the landmark and labels dataset.
    
    trained_models_path: path
        The directory path of the folder of "trained_models". (Using the config file)
        
Process:
    Validated the trained_models_path.
    Validated model if fitted.
    Handle exeption erros in locating the directory path of pose_classifier.joblib.
    Save the model to specific directory.
    Validated the pose_classifier.joblib file.

Output:
    None:
        Trained model pose_classifier.joblib in the folder.

Invariants:
    Model must be fitted before saving.
    config_trained_models_path must exist.
    pose_classifier.joblib must exist.
"""

#Save the train model. | Return: None
def save_trained_model(trained_model: ClassifierMixin, trained_models_path: Path) -> None:
    
     # Validate inputs.
    _validate_input(trained_model, trained_models_path)

    # Load the trained models path using config file.
    model_file = trained_models_path / "pose_classifier.joblib"

    # Save the model.
    try:
        joblib.dump(trained_model, model_file)

    except (OSError, PermissionError) as error:
        raise RuntimeError(f"Unable to save model to '{model_file}'.") from error

    # Validate output.
    _validate_output(model_file)