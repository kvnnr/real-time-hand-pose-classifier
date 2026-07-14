import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier

from src.model.training_model_helpers.predict_model.prediction_model import model_predictor
from src.model.training_model_helpers.predict_model.prediction_model_validator import (_validate_input,_validate_output)

"""
What:
    Test model predictor module.

Responsibilities:
    Test input validation.
    Test output validation.
    Test prediction behavior.

Input:
    Synthetic normalized landmark dataset.

Process:
    Validate invalid inputs.
    Train fake classifier.
    Predict labels.
    Validate prediction output.

Output:
    All tests should pass.

Invariants:
    Returned prediction must be numpy.ndarray.
"""

#NOTE: TEST INPUT VALIDATION ↓

def test_validate_input_accept_valid_dataset():

    #Create fake dataset.
    X_train = np.random.rand(10,63)
    y_train = np.array([
        0,0,0,0,0,
        1,1,1,1,1
    ])

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    #Create prediction dataset.
    X = np.random.rand(2,63)

    #Validate input.
    _validate_input(X,model)

def test_validate_input_reject_non_numpy_dataset():

    #Create fake dataset.
    X = [[1,2,3]]
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(TypeError):
        _validate_input(X,model)

def test_validate_input_reject_empty_dataset():

    #Create empty dataset.
    X = np.array([])
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X,model)

def test_validate_input_reject_invalid_dimension():

    #Create fake dataset.
    X = np.random.rand(63)
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X,model)

def test_validate_input_reject_invalid_feature_count():

    #Create fake dataset.
    X = np.random.rand(5,21)
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X,model)

def test_validate_input_reject_model_without_predict():

    #Create fake dataset.
    X = np.random.rand(5,63)
    model = object()

    #Check error.
    with pytest.raises(TypeError):
        _validate_input(X, model)

def test_validate_input_reject_untrained_model():

    #Create fake dataset.
    X = np.random.rand(5,63)
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X,model)

#NOTE: TEST OUTPUT VALIDATION ↓

def test_validate_output_accept_numpy_array():

    #Create fake prediction.
    predictions = np.array([0, 1, 0])

    #Validate output.
    _validate_output(predictions)

def test_validate_output_reject_empty_prediction():

    #Create empty prediction.
    predictions = np.array([])

    #Check error.
    with pytest.raises(ValueError):
        _validate_output(predictions)

def test_validate_output_reject_non_numpy_prediction():

    #Create fake prediction.
    predictions = [0,1]

    #Check error.
    with pytest.raises(TypeError):
        _validate_output(predictions)

def test_validate_output_reject_invalid_dimension():

    #Create fake prediction.
    predictions = np.array([[0], [1]])

    #Check error.
    with pytest.raises(ValueError):
        _validate_output(predictions)

#NOTE: TEST MAIN FUNCTION ↓

def test_model_predictor_return_numpy_array():

    #Create training dataset.
    X_train = np.random.rand(10,63)
    y_train = np.array([
        0,0,0,0,0,
        1,1,1,1,1
    ])

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    #Create prediction dataset.
    X = np.random.rand(3,63)

    #Predict labels.
    prediction = model_predictor(X, model)

    #Check output.
    assert isinstance(prediction, np.ndarray)

def test_model_predictor_prediction_count_match_input():

    #Create training dataset.
    X_train = np.random.rand(10,63)
    y_train = np.array([
        0,0,0,0,0,
        1,1,1,1,1
    ])

    model = RandomForestClassifier()
    model.fit(X_train,  y_train)

    #Create prediction dataset.
    X = np.random.rand(7,63)

    #Predict labels.
    prediction = model_predictor(X, model)

    #Check prediction count.
    assert len(prediction) == len(X)

def test_model_predictor_raise_runtime_error_when_prediction_fail():

    #Create fake training dataset.
    X_train = np.random.rand(10,63)

    y_train = np.array([
        0,0,0,0,0,
        1,1,1,1,1
    ])

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    #Force predict() to fail.
    def fake_predict(X):
        raise Exception("Prediction failed.")

    model.predict = fake_predict

    X = np.random.rand(2,63)

    #Check error.
    with pytest.raises(RuntimeError):
        model_predictor(X, model)