#Hand detector Module
from detection.hand_detector import HandDetector
#Normalizer Module
from src.feature.landmark_normalizer.normalizer import Normalizer

import numpy as np
from typing import List

class LandmarkPipeline:
    """
    LandmarkPipeline Contract

    What:
        A processing pipeline that extracts and normalizes 21 hand landmarks from live RGB frames.

    Why:
        To convert raw camera input into structured, ML-ready hand pose features for downstream tasks
        such as gesture recognition or classification.

    How:
        Uses a HandDetector module to extract raw hand landmarks,
        then applies a Normalizer module to convert landmarks into a consistent coordinate space.

    Inputs:
        detector (HandDetector)
            Module responsible for detecting hand landmarks from frames.

        normalizer (Normalizer)
            Module responsible for normalizing landmark coordinates.

    Preconditions:
        - detector must be initialized and functional.
        - normalizer must be initialized and functional.
        - detector must return valid 21 hand landmarks or None/empty if no hand is detected.
        - input frames must be RGB numpy arrays with valid image shape.

    Process:
        Receive RGB frames from camera pipeline
        Pass frames into HandDetector to extract raw 21 landmarks
        Pass extracted landmarks into Normalizer for scaling and alignment
        Return processed normalized landmark vector

    Output:
        List[float] or np.ndarray
            A normalized representation of 21 hand landmarks suitable for ML models
            OR empty output if no hand is detected.
    """
    
    #Initiates Modules.
    def __init__(self, detector: HandDetector, normalizer: Normalizer) -> None:
        
        #Objects Initializations. 
        self.detector = detector
        self.normalizer = normalizer

    #Get the 21 Normalized Landmarks from live frame. Return: 21 Normalized Landmarks.
    def extract(self, rgb_frames: np.ndarray) -> List[float]:
        
        """
        Pipeline:
            rgb_frames (from camera → grab grames → convert BGR to RGB)
                ↓
            hand_detector.py (Extract the 21 landmarks of hands.)
                ↓
            normalizer.py (LandmarkProcesser → Make the landmarks relative to wrist, and proper ratio.)
                ↓
            Output (Extracted normalized landmarks)
        """

        #Process RGB frames. Return: UnNormalized 21 Hand landmarks.
        unnormalized_landmarks = self.detector.detect_hands(rgb_frames)

        #Convert Normalized the landmarks. Return: Normalized landmarks.
        normalized_landmarks = self.normalizer.transform(unnormalized_landmarks)

        return normalized_landmarks