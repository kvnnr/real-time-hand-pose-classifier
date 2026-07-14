from pathlib import Path
from src.model.training_model_helpers.dataset_loader.loader_validator import _validate_path, _validate_X_y
from src.model.schemas.dataset_schema import Dataset
import numpy as np

"""
What:
    Helper method for training the model.

Responsibilities:
    Load the landmarks (X) and Labels (y) from the
    dataset file (hand_pose_dataset.npz)

Preconditions:
    dataset_schema.py:
        Store all output in the dataclass for easy access.

    Validator:
        Handle all validation of dataset loader module.
        →loader_validator.py
        
Input:
    dataset_file_path: Path
       Path to the .npz dataset file.

Process:
    Validate directory path.
    Load landmarks and labels from the dataset file.
    Validate X, and y.
    Return X, y | #NOTE: via dataset(dataclass)

Output:
    X:
        - np.ndarray
        - contains landmarks/features
        - shape: (samples, features)
        - numeric values
    
    y:
        - np.ndarray
        - contains labels
        - shape: (samples,)
        - same number of samples as X

Invariants:
    Directory path must exist.
    Must return np.ndarray
"""

#NOTE: MAIN FUNCTION ↓

#Load the dataset. | Return: tuple[np.ndarray, np.ndarray]
#Return X. y
def load_dataset(dataset_file_path: Path) -> Dataset:
    
    _validate_path(dataset_file_path)

    #Load the dataset from directory file.
    with np.load(dataset_file_path) as dataset:

        required_keys = {"X", "y"}
        
        #Check if the required keys are inside the dataset file.
        if not required_keys.issubset(dataset.files):
            raise ValueError(
                f"Dataset must contain {required_keys}, "
                f"found {dataset.files}"
                )

        #Load the landmarks(X) and labels(y)
        X = dataset['X']
        y = dataset['y']

    _validate_X_y(X, y)

    return Dataset(X, y)

