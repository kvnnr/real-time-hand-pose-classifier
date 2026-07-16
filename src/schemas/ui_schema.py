from dataclasses import dataclass
from typing import Any


@dataclass
class UiState:

    """
    UiState Contract

    What:
        Stores all information required
        by the user interface.

    Why:
        Separates UI data from the
        computer vision processing state.

        Allows UI components to evolve
        independently.

    Input:
        None

    Process:
        Stores runtime UI information.

    Output:
        UI state container.

    Failure Conditions:
        None.

    Invariants:
        UI state only contains
        presentation information.
    """
    
    # User entered pose name.
    pose_label: str = ""

    # Model prediction result.
    prediction: str = "Waiting..."

    # Prediction confidence score.
    confidence: float = 0.0

    # Current FPS.
    fps: float = 0.0

    # Hand detection status.
    hand_detected: bool = False

    # Machine learning model status.
    model_loaded: bool = False

    #Handle keyboard Input.
    key = Any