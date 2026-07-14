from sklearn.base import ClassifierMixin
from src.model.training_model_helpers.predict_model.prediction_model_validator import _validate_input

import numpy as np

"""
What:
    Predict hand pose labels using the trained machine
    learning model.

Responsibilities:
    - Validate prediction inputs.
    - Predict pose labels.
    - Return prediction results.

Input:
    X: np.ndarray | Use X_train for traning | X_live_landmark for realtime.
        Normalized landmark features.

    model: ClassifierMixin
        Previously trained machine learning model.

Process:
    1. Validate X and model.
    2. Predict labels.
    3. Return predictions.

Output:
    predictions:
        np.ndarray containing predicted labels.

Failure Conditions:
    - X is not numpy array.
    - Model is not fitted.
    - Prediction fails.

Invariants:
    - Returned prediction must be numpy.ndarray.
"""


#Predict labels. | Return: predictions
def model_predictor(X: np.ndarray, trained_model: ClassifierMixin) -> np.ndarray:

    _validate_input(X, trained_model)

    #Predict labels.
    try:
        predictions = trained_model.predict(X)
    except Exception as error:
        raise RuntimeError("Model prediction failed.") from error

    return predictions