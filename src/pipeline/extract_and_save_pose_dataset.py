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
    SavePoseDatasetsPipeline Contract

    What:
        Coordinate the extraction and saving of a labeled hand pose dataset.

    Why:
        Convert an RGB frame into normalized hand landmarks and save
        the extracted pose into the dataset for future model training.

    Input:
        landmark_labels: str
            Pose label assigned by the user.

        rgb_frame: np.ndarray
            RGB frame captured from the camera.

    Process:
        Validate the input label.
        Validate the RGB frame.
        Extract normalized hand landmarks.
        Validate the extracted landmarks.
        Save the pose label and landmarks to the dataset.

    Output:
        List[float]
            Normalized landmark feature vector that was saved.

        None
            Returned when no hand is detected.
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
    def extract_and_save_landmark(self, landmark_labels: str, rgb_frame:np.ndarray) -> list[float] | None:

        """
        Extract and save a labeled hand pose.

        Pipeline:
            RGB Frame
                ↓
            ExtractLandmarkPipeline
                ↓
            Normalized Landmark Features
                ↓
            SavePoseDataset
                ↓
            NPZ Dataset

        Input:
            landmark_labels:
                User-defined pose label.

            rgb_frame:
                RGB Numpy Matrix.

        Process:
            Validate the pose label.
            Validate the RGB frame.
            Extract normalized landmarks.
            Check if a hand is detected.
            Validate the extracted landmarks.
            Save the landmarks and label.

        Output:
            List[float]
                Saved normalized landmark feature vector.

            None
                Returned when no hand is detected.
        """

        # Validate input.
        self._validate_label(landmark_labels)
        self._validate_frame(rgb_frame)

        #Extract the normalized landmark features.
        result = self.extract_landmark.extract(rgb_frame)
        
        #Check if no hand is detected.
        if result is None:
            return None

        # Validate output.
        self._validate_landmarks(result)
        
        #Store the extracted landmarks and pose label.
        #Temporary dataset:
        #self.x -> landmarks
        #self.y -> labels
        self.save_landmark.add_pose(result, landmark_labels)

        #Save the Extracted landmarks and labels to NPZ file.
        self.save_landmark.save_dataset()

        return result