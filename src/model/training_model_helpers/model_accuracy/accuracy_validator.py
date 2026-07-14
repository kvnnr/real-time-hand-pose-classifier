import numpy as np

"""
What:
    Handle all validation for the model_accuracy module.
"""

def _validate_input(predictions: np.ndarray,y_test: np.ndarray):

    #Check data type.
    if not isinstance(predictions, np.ndarray):
        raise TypeError("Predictions must be Numpy array.")

    if not isinstance(y_test, np.ndarray):
        raise TypeError("Test labels must be Numpy array.")

    #Check if it's empty.
    if predictions.size == 0:
        raise ValueError("Predictions must not be empty.")

    if y_test.size == 0:
        raise ValueError("Test labels must not be empty.")

    #Check dimensions.
    if predictions.ndim != 1:
        raise ValueError("Predictions must be 1D array.")

    if y_test.ndim != 1:
        raise ValueError("Test labels must be 1D array.")

    #Check sample count.
    if len(predictions) != len(y_test):
        raise ValueError("Predictions and test labels must have the same number of samples.")


def _validate_output(accuracy: float):

    #Check data type.
    if not isinstance(accuracy, (float, np.floating)):
        raise TypeError("Accuracy score must be float.")

    #Check range.
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("Accuracy score must be between 0.0 and 1.0.")