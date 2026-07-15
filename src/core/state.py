from dataclasses import dataclass
import numpy as np
from typing import Any

@dataclass
class AppState:
    
    """
    What:
        Contains all data state of the program.
    Why:
        To make it all in one place and organize.
    """
    pose_label: str = ""
    frame: np.ndarray | None = None
    rgb_frame: np.ndarray | None = None
    key: Any = None
    bgr_frame: np.ndarray | None = None
    accuracy: float | None = None