#Pipeline:
from src.pipeline.extract_landmark import ExtractLandmarkPipeline

#Modules:
#save_pose.py 
from src.data.save_pose import SavePoseDataset
#Hand detector 
from src.detection.hand_detector import HandDetector
#Normalizer
from src.feature.landmark_normalizer.normalizer import Normalizer

import numpy as np

class SavePoseDatasetsPipeline:
    
    """
    Pipeline:
        Extract and Normalized landmarks from raw frames | via (extract_landmark MODULE)
                ↓
        Save Extracted landmarks to NPZ | via (save_pose MODULE)


    What:
        Coordinates saving a labeled hand pose sample, including its
        landmarks and reference image, into the /data folder.

    Preconditions:
        MODULE:
            →extract_landmark
            →save_pose
    
    Input:
        label: str
        RAW frames: np.array

    Process:
        Validate the raw frames (np.array).
        Extract and Normalized the landmarks from raw frames.
        Persist the landmarks to the NPZ dataset.
        Save the reference image for the corresponding pose.

    Output:
        A new labeled pose sample is successfully stored in the dataset,
        and its corresponding reference image is saved into data/raw folder.
    """
    
    #Initiate objects.
    def __init__(self) -> None:
        
        #Objects
        self.detect = HandDetector()
        self.normalizer = Normalizer()
        self.save_landmark = SavePoseDataset()
        self.extract_landmark = ExtractLandmarkPipeline(self.detect, self.normalizer)
    
    #------------
    #Validations.
    #------------

    def _validate_label(self, label: str) -> None:
        
        """Validate pose label."""

        if not isinstance(label, str):
            raise TypeError("landmark_labels must be a string.")

        if label.strip() == "":
            raise ValueError("landmark_labels cannot be empty.")

    def _validate_frame(self, frame: np.ndarray) -> None:
        
        """Validate RGB frame."""

        if not isinstance(frame, np.ndarray):
            raise TypeError("rgb_frame must be a numpy.ndarray.")

        if frame.size == 0:
            raise ValueError("rgb_frame cannot be empty.")

        if frame.ndim != 3:
            raise ValueError("rgb_frame must have shape (height, width, channels).")

    def _validate_landmarks(self, landmarks: list[float]) -> None:
        """Validate extracted landmarks."""

        if landmarks is None:
            raise ValueError("No landmarks were extracted.")

        if not isinstance(landmarks, list):
            raise TypeError("Extracted landmarks must be a list.")

        if len(landmarks) == 0:
            raise ValueError("Extracted landmark list cannot be empty.")

        for value in landmarks:
            if not isinstance(value, (int, float)):
                raise TypeError(
                    "Landmarks must contain only numeric values."
                )
    
    
    #Extract and Normalized landmark → Save as dataset (NPZ). | Return: landmarks. ← indication purposes.
    def extract_and_save_landmark(self, landmark_labels: str, rgb_frame:np.ndarray) -> list[float]:
        
        # Validate input.
        self._validate_label(landmark_labels)
        self._validate_frame(rgb_frame)

        #Get the Normalized 21 landmarks.
        result = self.extract_landmark.extract(rgb_frame)
        
        # Validate output.
        self._validate_landmarks(result)
        
        #Add pose dataset to temporary store.
        #self.x (landmarks), self.y (labels).
        self.save_landmark.add_pose(result, landmark_labels)

        #Save the Extracted landmarks and labels to NPZ file.
        self.save_landmark.save_dataset()

        return result