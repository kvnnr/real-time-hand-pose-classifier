import numpy as np
import json

from typing import Protocol
from pathlib import Path

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarker
from mediapipe.tasks.python import BaseOptions

#Protocal for the landmarks
class LandmarkProto(Protocol):
    x: float
    y: float
    z: float

#Main Function
class HandDetector:

    """ 
    Detect the hands from Numpy matrix

    Input:
        Numpy Matrix of frames (RGB)

    Process:
        Detect Palm
        Plot 21 Hand landmarks
    
    Output:
        UnNormalized 21 Hand landmarks | X Y Z
    """

    #Variables: self.hand_detector
    def __init__(self, detector = None) -> None:
        
        self.config = {} #Variable for config file.
        
        #This ensures the project can find its configuration file safely
        config_path = Path(__file__).parent.parent.parent / "config.json"

        #This ensures the project can find the hand landmarker task.
        model_path = Path(__file__).parent / "hand_landmarker.task"

        #Error handling for the config file.
        try:

            #This load the config.json into self.config variable.
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Unable to load configuration file: {config_path}") from e

        #Store and initiate what task will be used.
        base_options = BaseOptions(model_asset_path = str(model_path))

        #Initialize the MediaPipe Hands inference configurations.
        options = vision.HandLandmarkerOptions ( 
            
            #Initialize the configurations of hand detector.
            base_options = base_options,
            num_hands = int(self.config.get("max_num_hands", 2)),
            min_hand_detection_confidence = float(self.config.get("min_detection_confidence", 0.5)),
            min_hand_presence_confidence = float(self.config.get("min_tracking_confidence", 0.5)),
            min_tracking_confidence = float(self.config.get("min_tracking_confidence", 0.5))
        )

        #Initiate the Hand detector.
        self.hand_detector = detector or vision.HandLandmarker.create_from_options(options)

    #Validation of inputs
    def validate_input(self, frame: np.ndarray) -> np.ndarray:

        if frame is None:
            raise ValueError("No frame detected!")
        
        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be Numpy array!")
        
        if frame.size == 0:
            raise ValueError("Frame is empty!")
        
        if frame.ndim != 3:
            raise ValueError("Frame must have 3 dimensions!")
        
        if frame.shape[2] != 3:
            raise ValueError("Frame must have 3 Color Channels!")
        
        if frame.dtype != np.uint8:
            raise TypeError("Invalid Image dtype. Must be uint8!")
        
        return frame

    #Validation of landmarks.
    def validate_hand_landmark(self, landmarks: list[list[LandmarkProto]]) -> list[list[LandmarkProto]]:
        
        if landmarks is None:
            raise ValueError("No landmarks detected!")
        
        if not isinstance(landmarks, list):
            raise TypeError("Invalid landmarks data type!")
        
        if len(landmarks) == 0:
            raise ValueError("No hands detected!")
        
        for hand in landmarks: #Check each landmark.
            
            if not isinstance(hand, list):
                raise TypeError("Each hand must be a list of landmarks!")

            if len(hand) != 21:
                raise ValueError("Each hand must contain exactly 21 landmarks.")

            for landmark in hand:
                if not hasattr(landmark, "x") or not hasattr(landmark, "y") or not hasattr(landmark, "z"):
                    raise TypeError("Invalid landmark object structure.")

                if not isinstance(landmark.x, (int, float)):
                    raise TypeError("landmark.x must be numeric")
                   
        return landmarks

    #Hand landmark detector | Return: UnNormalized 21 Hand landmarks
    def detect_hands(self, rgb_frames: np.ndarray) -> list[list[LandmarkProto]]:

        #Validate rgb_frames first
        validated_frames = self.validate_input(rgb_frames)

        #Catch any mediapipe process errors.
        try:        

            #Process the validated frames into MEDIAPIPE FORMAT.
            """
            Why:
                Mediapipe task can't handle raw Numpy matrix. Wrap it in a special
                container mediapipe task can understand and process.
            """
            formatted_frames = vision.Image(image_format=vision.ImageFormat.SRGB, data = validated_frames)
            
            #Process the 21 landmarks from formatted frames.
            result = self.hand_detector.detect(formatted_frames)

            #Extract the 21 landmarks.
            landmarks = result.hand_landmarks

            #Validate the landmarks.
            validated_unnormalized_landmarks = self.validate_hand_landmark(landmarks)

        except Exception as e:
            raise RuntimeError(f"MediaPipe processing failed!") from e
        
        #Return the 21 landmarks
        return validated_unnormalized_landmarks

    #Close all resources mediapipe uses
    def close_mediapipe(self):

        """Release MediaPipe resources."""
        self.hand_detector.close()