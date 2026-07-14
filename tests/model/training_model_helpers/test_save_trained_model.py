from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.model.training_model_helpers.save_model.save_trained_model import save_trained_model

"""
What:
    Unit tests for save_trained_model().

Purpose:
    Verify that the trained model is properly validated
    before saving and correctly stored as
    pose_classifier.joblib.

Testing Strategy:
    Use a temporary directory to simulate the
    trained_models folder.

    Use a fitted RandomForestClassifier as the
    trained model.

    Verify all input validation and output
    requirements independently.
"""

# ------------------------------------------------------------------
# Helper method.
# ------------------------------------------------------------------

def create_fitted_model() -> RandomForestClassifier:

    """
    Create a small fitted model for testing.

    Return:
        Trained RandomForestClassifier.
    """

    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([0, 0, 1, 1])

    model = RandomForestClassifier(random_state=48)
    model.fit(X, y)

    return model

# ------------------------------------------------------------------
# Success Test
# ------------------------------------------------------------------

def test_save_trained_model_output_required_file_exists(tmp_path: Path):

    """
    What:
        Verify that pose_classifier.joblib
        is created after saving.

    Expected:
        File exists inside the configured folder.
    """

    # Create a fitted model.
    model = create_fitted_model()

    # Create configuration.
    config = {"trained_models_path": {"path": tmp_path}}

    # Save the model.
    save_trained_model(model, config)

    # Expected output file.
    saved_model = tmp_path / "pose_classifier.joblib"

    # Verify file exists.
    assert saved_model.exists()

    # Verify output is a file.
    assert saved_model.is_file()

# ------------------------------------------------------------------
# Input Validation
# ------------------------------------------------------------------

def test_save_trained_model_input_invalid_model_type():

    """
    Verify model must inherit from ClassifierMixin.
    """

    config = {"trained_models_path": {"path": Path(".")}}

    with pytest.raises(ValueError):
        save_trained_model("invalid", config)

def test_save_trained_model_input_model_not_fitted(tmp_path: Path):

    """
    Verify model must already be trained.
    """

    model = RandomForestClassifier()
    config = {"trained_models_path": {"path": tmp_path}}

    with pytest.raises(ValueError):
        save_trained_model(model, config)

def test_save_trained_model_input_config_not_dictionary():

    """
    Verify configuration must be dictionary.
    """

    model = create_fitted_model()
    with pytest.raises(ValueError):
        save_trained_model(model, "invalid")

def test_save_trained_model_input_missing_trained_models_path_key(tmp_path: Path):

    """
    Verify required configuration key exists.
    """

    model = create_fitted_model()
    config = {}

    with pytest.raises(ValueError):
        save_trained_model(model, config)

def test_save_trained_model_input_missing_path_key(tmp_path: Path):

    """
    Verify required path key exists.
    """

    model = create_fitted_model()
    config = {"trained_models_path": {}}

    with pytest.raises(ValueError):
        save_trained_model(model, config)

def test_save_trained_model_input_path_not_path_object():

    """
    Verify configured path must be pathlib.Path.
    """
    model = create_fitted_model()
    config = {"trained_models_path": {"path": "trained_models"}}

    with pytest.raises(ValueError):
        save_trained_model(model, config)

def test_save_trained_model_input_directory_not_exist():

    """
    Verify configured directory must exist.
    """

    model = create_fitted_model()
    config = {"trained_models_path": {"path": Path("directory_that_does_not_exist")}}

    with pytest.raises(ValueError):
        save_trained_model(model, config)

def test_save_trained_model_input_path_not_directory(tmp_path: Path):

    """
    Verify configured path must be a directory.
    """

    model = create_fitted_model()
    file = tmp_path / "dummy.txt"
    file.write_text("dummy")

    config = {"trained_models_path": {"path": file}}

    with pytest.raises(ValueError):
        save_trained_model(model, config)

# ------------------------------------------------------------------
# Exception Handling
# ------------------------------------------------------------------

def test_save_trained_model_raise_runtime_error_when_joblib_dump_fails(monkeypatch, tmp_path: Path):

    """
    Verify RuntimeError is raised when
    joblib fails while saving the model.
    """

    model = create_fitted_model()
    config = {"trained_models_path": {"path": tmp_path}}

    # Force joblib.dump() to fail.
    def fake_dump(*args, **kwargs):
        raise OSError("Cannot save.")

    monkeypatch.setattr("joblib.dump", fake_dump)
    
    with pytest.raises(RuntimeError):
        save_trained_model(model, config)