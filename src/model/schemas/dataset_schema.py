from dataclasses import dataclass
import numpy as np

"""
What:
    Stores the separated training and testing datasets.

Why:
    Avoid passing multiple arrays independently.
"""

@dataclass
class Dataset:

    X_landmarks: np.ndarray
    y_labels: np.ndarray


@dataclass
class DatasetSplit:

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray