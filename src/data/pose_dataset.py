import logging
from typing import List, Union
from pathlib import Path
import numpy as np


class StorePoseDataset:

    """
    Persists collected gesture data into dataset format.

    Input:
        Normalized landmarks & Label Name

    Process:
        Saves dataset as NPZ file. Landmark = X, Labels = Y

    Output:
        NPZ of Data set. landmark = X, Labels = Y
    """

    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_SET_FILE = ROOT_DIR / "data" / "hand_pose_dataset.npz"

    def __init__(self):

        self.x = [] #Will contain landmarks (63 Values)
        self.y = [] #Will contain the labels
        self.labels = set()

        # Set up logging configuration.
        self.logger = logging.getLogger(__name__)

    #Clear the temporary storage of Dataset Collection.
    def clear_temporary_storage(self) -> None:

        """Wipes the internal RAM buffers to prevent memory leaks."""

        self.x.clear()
        self.y.clear()
        self.labels.clear()

        self.logger.info("Temporary storage successfully cleared.")

    #Validation of inputs. | Return: landmark, labels
    def validate_input(self, landmark: List[float], labels: str) -> tuple[np.ndarray, str]:

        #-----------------------
        #Validation of landmarks
        #-----------------------

        if not isinstance(landmark, list):
            raise TypeError("Landmarks in Dataset should be a list!")

        if len(landmark) == 0:
            raise ValueError("Landmarks must not be empty")

        try:

            """
            Why:
                To check if all landmarks are float without looping.
            """

            #Temporary convert landmark into np.array.
            landmark_array = np.asarray(landmark, dtype=np.float32)

        except (TypeError, ValueError) as e:
            raise TypeError("Landmarks must be numeric floats") from e

        if landmark_array.ndim != 1:
            raise ValueError("Landmark must be flattened 1D vector")

        if landmark_array.size != 63:
            raise ValueError("Landmarks must have 63 values!")

        if not np.isfinite(landmark_array).all():
            raise ValueError("Landmarks contain NaN or Inf values")

        #--------------------
        #Validation of labels
        #--------------------

        if not isinstance(labels, str):
            raise TypeError("Labels must be string!")

        if not labels.strip():
            raise ValueError("Labels must not be empty!")

        return landmark_array, labels.strip()

    #Add pose dataset to temporary store. | Return: self.x, self.y (with landmarks and labels inside).
    def add_pose(self, landmark: List, labels: str) -> None:

        """Receives normalized landmarks and a label from the orchestration layer."""

        #Validate the Input.
        validated_landmark, validated_labels = self.validate_input(landmark, labels)

        #Add landmarks to temporary storage.
        self.x.append(validated_landmark)

        #Add labels to temporary storage.
        self.y.append(validated_labels)

        #Store unique labels.
        self.labels.add(validated_labels)

    #Save Landmarks and Labels into NPZ file
    def save_dataset(self, filepath: Union[str, Path] = DATA_SET_FILE) -> None:

        """Converts internal lists to NumPy arrays, saves them, and flushes RAM."""

        if not self.x:
            self.logger.warning("No data found in temporary storage to save.")
            return

        filepath = Path(filepath)

        #Make sure the destination folder actually exists before writing.
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Could not create directory for {filepath}") from e

        # Convert to standard NumPy arrays.
        X_array = np.asarray(self.x, dtype=np.float32)
        y_array = np.asarray(self.y, dtype=str)

        #Save compressed to save space and efficiency
        np.savez_compressed(filepath, X=X_array, y=y_array, labels=np.asarray(sorted(self.labels)))

        self.logger.info(
            "Dataset successfully saved to %s with %d samples and %d unique labels.",
            filepath, len(self.x), len(self.labels)
        )

        #Clear temporary storage after saving
        self.clear_temporary_storage()