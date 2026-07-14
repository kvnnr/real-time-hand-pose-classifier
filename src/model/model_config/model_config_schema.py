from dataclasses import dataclass
from pathlib import Path

"""
What:
    Configuration object for the model trainer.
"""

@dataclass(frozen=True)
class ModelConfig:
    dataset_file_path: Path
    trained_models_path: Path
    test_size: float
    random_state: int