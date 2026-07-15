from dataclasses import dataclass
import numpy as np

@dataclass
class VisionState:
    
    """
    What:
        Contains all data Vision state of the program.
    Why:
        To make it all in one place and organize.
    """

    frame: np.ndarray | None = None
    rgb_frame: np.ndarray | None = None
    bgr_frame: np.ndarray | None = None
    accuracy: float | None = None