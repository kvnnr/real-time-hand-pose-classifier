from dataclasses import dataclass
from pathlib import Path

"""
What:
    Stores the model configuration values.
"""

@dataclass
class ModelConfig:

    dataset_file_path: Path
    test_size: float
    random_state: int
    trained_models_path: Path