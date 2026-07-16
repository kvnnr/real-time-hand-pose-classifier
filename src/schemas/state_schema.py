from dataclasses import dataclass
import numpy as np

@dataclass
class VisionState:

    """
    VisionState Contract

    What:
        Stores the runtime vision state of the application.

    Why:
        Centralize all vision-related data into a single object
        that can be shared between pipelines, inference modules,
        and the user interface.

    Contains:

        Camera Frames

        Extracted Features

        Prediction Results

        Detection Status

        Performance Metrics

    Invariants:

        Each attribute represents the latest state produced
        during the current iteration of the main loop.
    """

    # --------------------------
    # Camera Frames
    # --------------------------

    frame: np.ndarray | None = None

    rgb_frame: np.ndarray | None = None

    bgr_frame: np.ndarray | None = None

    # --------------------------
    # Detection
    # --------------------------

    landmarks: list | None = None

    hand_detected: bool = False

    # --------------------------
    # Feature Extraction
    # --------------------------

    features: list | None = None

    # --------------------------
    # Prediction
    # --------------------------

    prediction: str = "Waiting..."

    confidence: float = 0.0

    # --------------------------
    # Model Information
    # --------------------------

    accuracy: float | None = None
