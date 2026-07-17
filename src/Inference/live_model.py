
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import ClassifierMixin

class PredictionModel:
    """
    Predicts hand poses using a trained classifier.

    Input:
        Normalized hand landmarks.
        Shape: (63,)

    Process:
        1. Validate input.
        2. Reshape features for sklearn.
        3. Predict class label.
        4. Calculate prediction confidence.

    Output:
        Tuple:
            (
                predicted_pose,
                confidence
            )

        confidence:
            Float between 0.0 - 1.0

    Failure Conditions:
        FileNotFoundError:
            Trained model does not exist.

        ValueError:
            Invalid feature shape.

        AttributeError:
            Model does not support probability prediction.
    """

    ROOT_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = ROOT_DIR / "trained_models" / "pose_classifier.joblib"


    def __init__(self) -> None:
        
        """
        Loads trained classifier once.
        """

        self._model: ClassifierMixin = joblib.load(self.MODEL_PATH)
        self.loaded = True
        
    def predict(self, features: np.ndarray) -> tuple[str, float]:
        
        """
        Predict hand gesture with confidence.

        Input:
            features:
                Normalized landmark vector.

                Shape:
                    (63,)

        Output:
            prediction:
                Predicted class name.

            confidence:
                Probability of prediction.

                Range:
                    0.0 - 1.0
        """

        # Validate type.
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a NumPy ndarray.")

        # Validate shape.
        if features.ndim != 1:
            raise ValueError("features must have shape (63,).")

        # sklearn expects:
        # (samples, features)

        features = features.reshape(1, -1)

        # Prediction.
        prediction = self._model.predict(features)[0]

        # Confidence.
        probabilities = self._model.predict_proba(features)[0]

        confidence = float(np.max(probabilities))

        return (str(prediction),confidence)
    
    def reload(self) -> None:
        """
        Reload the trained classifier from disk.

        Why:
            Picks up a newly trained model file
            without restarting the application.

        Failure Conditions:
            FileNotFoundError:
                Trained model does not exist.
        """

        self._model = joblib.load(self.MODEL_PATH)
        self.loaded = True