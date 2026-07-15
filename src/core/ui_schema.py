from dataclasses import dataclass
from typing import Any

@dataclass
class UiState:
    
    """
    What:
        Contains all data state of the program.
    Why:
        To make it all in one place and organize.
    """
    key: Any = None
    pose_label: str = ""