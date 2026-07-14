from pathlib import Path
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

"""
What:
    Handle all validation for the model_accuracy module.
"""

# Validate input.
def _validate_input(model: ClassifierMixin, trained_models_path: Path) -> None:

    # Check model type.
    if not isinstance(model, ClassifierMixin):
        raise ValueError("Model must inherit from ClassifierMixin.")

    # Check if model is fitted.
    try:
        check_is_fitted(model)

    except NotFittedError:
        raise ValueError("Model must be fitted before saving.")

    # Check config type.
    if not isinstance(trained_models_path, dict):
        raise ValueError("trained_models_path must be a dictionary.")

    # Check required config key.
    if "trained_models_path" not in trained_models_path:
        raise ValueError("'trained_models_path' is missing in config.")

    if "path" not in trained_models_path["trained_models_path"]:
        raise ValueError("'path' is missing in trained_models_path config.")

    # Check path type.
    if not isinstance(trained_models_path["trained_models_path"]["path"], Path):
        raise ValueError("Configured path must be a pathlib.Path.")

    # Check directory exists.
    if not trained_models_path["trained_models_path"]["path"].exists():
        raise ValueError("trained_models directory does not exist.")

    if not trained_models_path["trained_models_path"]["path"].is_dir():
        raise ValueError("Configured trained_models_path must be a directory.")

# Validate output.
def _validate_output(model_file: Path) -> None:

    # Check if model file exists.
    if not model_file.exists():
        raise ValueError("pose_classifier.joblib was not created.")

    # Check if output is a file.
    if not model_file.is_file():
        raise ValueError("pose_classifier.joblib is not a file.")
