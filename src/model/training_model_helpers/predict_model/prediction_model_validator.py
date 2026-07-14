from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

import numpy as np

"""
What:
    Handle all validation for the model_predictor module.
"""

def _validate_input(X: np.ndarray,model: ClassifierMixin):

    #Check data type.
    if not isinstance(X, np.ndarray):
        raise TypeError("Prediction dataset must be Numpy array.")

    #Check model interface.
    if not hasattr(model, "predict"):
        raise TypeError("Model must implement predict().")

    #Check if it's empty.
    if X.size == 0:
        raise ValueError("Prediction dataset must not be empty.")

    #Check dimensions.
    if X.ndim != 2:
        raise ValueError("Prediction dataset must be 2D array.")

    #Check feature size.
    if X.shape[1] != 63:
        raise ValueError("Expected normalized hand landmarks with 63 features.")

    #Check if model has already been trained.
    try:
        check_is_fitted(model)
    except NotFittedError as error:
        raise ValueError("Model must be trained before prediction.") from error


def _validate_output(predictions: np.ndarray):

    #Check data type.
    if not isinstance(predictions, np.ndarray):
        raise TypeError("Prediction output must be Numpy array.")

    #Check empty output.
    if predictions.size == 0:
        raise ValueError("Prediction output must not be empty.")

    #Check dimensions.
    if predictions.ndim != 1:
        raise ValueError("Prediction output must be 1D array.")