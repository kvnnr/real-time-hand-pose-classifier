import pytest
import numpy as np

from pathlib import Path

from src.model.training_model_helpers.dataset_loader.dataset_loader import (load_dataset, _validate_X_y, _validate_path)


# -------------------------
# FIXTURES
# -------------------------

@pytest.fixture
def valid_dataset(tmp_path: Path):
    
    """
    Create temporary valid NPZ dataset.
    """
    
    dataset_path = tmp_path / "hand_pose_dataset.npz"

    X = np.random.rand(100, 63)
    y = np.array(
        [
            "fist",
            "open_hand"
        ] * 50
    )

    np.savez(dataset_path, X=X, y=y)

    return dataset_path

@pytest.fixture
def valid_X_y():

    X = np.random.rand(50, 63)

    y = np.array(["fist"] * 50)

    return X, y

# =========================
# PATH VALIDATION TESTS
# =========================

def test_validate_path_accepts_valid_npz(valid_dataset):

    # Should not raise error
    _validate_path(valid_dataset)

def test_validate_path_wrong_type():

    with pytest.raises(TypeError):
        _validate_path("dataset.npz")

def test_validate_path_missing_file(tmp_path):

    fake_path = tmp_path / "missing.npz"

    with pytest.raises(FileNotFoundError):
        _validate_path(fake_path)

def test_validate_path_directory(tmp_path):

    folder = tmp_path / "dataset"

    folder.mkdir()

    with pytest.raises(ValueError):
        _validate_path(folder)

def test_validate_path_wrong_extension(tmp_path):

    file = tmp_path / "dataset.csv"
    file.write_text("test")
    with pytest.raises(ValueError):
        _validate_path(file)

def test_validate_path_empty_file(tmp_path):

    file = tmp_path / "empty.npz"
    file.touch()
    with pytest.raises(ValueError):
        _validate_path(file)

# =========================
# X AND Y VALIDATION TESTS
# =========================

def test_validate_X_y_valid(valid_X_y):

    X, y = valid_X_y
    _validate_X_y(X, y)

def test_validate_X_not_numpy_array():

    X = [[1,2,3]]
    y = np.array(["fist"])
    with pytest.raises(TypeError):
        _validate_X_y(X, y)

def test_validate_y_not_numpy_array():

    X = np.random.rand(1,63)
    y = ["fist"]
    with pytest.raises(TypeError):
        _validate_X_y(X, y)

def test_validate_empty_X():

    X = np.array([])
    y = np.array(["fist"])
    with pytest.raises(ValueError):
        _validate_X_y(X, y)

def test_validate_empty_y():

    X = np.random.rand(1,63)
    y = np.array([])
    with pytest.raises(ValueError):
        _validate_X_y(X, y)

def test_validate_sample_count_mismatch():

    X = np.random.rand(10,63)
    y = np.array(["fist"] * 5)
    with pytest.raises(ValueError):
        _validate_X_y(X,y)

def test_validate_X_wrong_dimension():

    X = np.random.rand(10,63,1)
    y = np.array(["fist"] * 10)
    with pytest.raises(ValueError):
        _validate_X_y(X,y)

def test_validate_y_wrong_dimension():

    X = np.random.rand(10,63)
    y = np.array(
        [
            ["fist"],
            ["open"]
        ]
    )
    with pytest.raises(ValueError):
        _validate_X_y(X,y)

def test_validate_X_non_numeric():

    X = np.array(
        [
            ["abc","def"]
        ]
    )
    y = np.array(["fist"])
    with pytest.raises(TypeError):
        _validate_X_y(X,y)

# =========================
# LOAD DATASET TESTS
# =========================

def test_load_dataset_returns_numpy_arrays(valid_dataset):

    X, y = load_dataset(valid_dataset)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

def test_load_dataset_correct_shape(valid_dataset):

    X, y = load_dataset(valid_dataset)
    assert X.shape == (100,63)
    assert y.shape == (100,)

def test_load_dataset_missing_X(tmp_path):
    
    dataset_path = tmp_path / "missing_X.npz"

    np.savez(dataset_path,y=np.array([0, 1, 2]))
    with pytest.raises(ValueError,match="Dataset must contain"):
        load_dataset(dataset_path)

def test_load_dataset_missing_y(tmp_path):

    dataset_path = tmp_path / "missing_y.npz"

    np.savez(dataset_path,X=np.random.rand(10, 63))
    with pytest.raises(ValueError, match="Dataset must contain"):
        load_dataset(dataset_path)