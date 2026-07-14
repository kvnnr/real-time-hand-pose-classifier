import numpy as np
import pytest

from sklearn.ensemble import RandomForestClassifier

from src.model.training_model_helpers.model_trainer.model_trainer import model_trainer
from src.model.training_model_helpers.model_trainer.trainer_validator import _validate_input

"""
What:
    Test model trainer module.

Responsibilities:
    Test input validation.
    Test model training behavior.
    Test returned trained model.

Input:
    Synthetic normalized landmark dataset.

Process:
    Validate invalid inputs.
    Train fake classifier.
    Check returned model.

Output:
    All tests should pass if module behaves correctly.

Invariants:
    Model trainer must return fitted model.
    Training dataset must contain 63 landmark features.
"""

#NOTE: TEST INPUT VALIDATION ↓
def test_validate_input_accept_valid_dataset():

    #Create fake landmark dataset.
    X_train = np.random.rand(4,63)
    y_train = np.array([0,0,1,1])
    model = RandomForestClassifier()

    #Validate input.
    _validate_input(X_train,y_train,model)

def test_validate_input_reject_non_numpy_X():

    #Create fake dataset.
    X_train = [[1,2,3]]
    y_train = np.array([0])
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(TypeError):
        _validate_input(X_train, y_train, model)

def test_validate_input_reject_non_numpy_y():

    #Create fake dataset.
    X_train = np.random.rand(2,63)
    y_train = [0,1]
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(TypeError):
        _validate_input(X_train, y_train, model)



def test_validate_input_reject_model_without_fit():

    #Create fake dataset.
    X_train = np.random.rand(2,63)
    y_train = np.array([0,1])

    #Create invalid model.
    model = object()

    #Check error.
    with pytest.raises(TypeError):
        _validate_input(X_train, y_train, model)

def test_validate_input_reject_empty_X():

    #Create empty dataset.
    X_train = np.array([])
    y_train = np.array([0,1])
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X_train, y_train, model)

def test_validate_input_reject_empty_y():

    #Create fake dataset.
    X_train = np.random.rand(2,63)
    y_train = np.array([])
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X_train, y_train, model)

def test_validate_input_reject_invalid_X_dimension():

    #Create fake dataset.
    X_train = np.random.rand(2,63,1)
    y_train = np.array([0,1])
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X_train, y_train, model)

def test_validate_input_reject_invalid_y_dimension():

    #Create fake dataset.
    X_train = np.random.rand(2,63)
    y_train = np.array([[0],[1]])
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X_train, y_train, model)

def test_validate_input_reject_different_sample_size():

    #Create fake dataset.
    X_train = np.random.rand(4,63)
    y_train = np.array([0,1])
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X_train, y_train, model)

def test_validate_input_reject_wrong_feature_size():

    #Create fake dataset.
    X_train = np.random.rand(4,21)
    y_train = np.array([0,0,1,1])
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X_train, y_train, model)

def test_validate_input_reject_single_class():

    #Create fake dataset.
    X_train = np.random.rand(4,63)
    y_train = np.array([0,0,0,0])
    model = RandomForestClassifier()

    #Check error.
    with pytest.raises(ValueError):
        _validate_input(X_train, y_train, model)

#NOTE: TEST MAIN FUNCTION ↓

def test_model_trainer_return_model():

    #Create fake dataset.
    X_train = np.random.rand(10,63)
    y_train = np.array([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1
    ])

    model = RandomForestClassifier(random_state=42)

    #Train model.
    trained_model = model_trainer(X_train,y_train, model)

    #Check output.
    assert trained_model is model

def test_model_trainer_return_fitted_model():

    #Create fake dataset.
    X_train = np.random.rand(10,63)
    y_train = np.array([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1
    ])

    model = RandomForestClassifier(random_state=42)

    #Train model.
    trained_model = model_trainer(X_train,y_train, model)

    #Check model is fitted.
    assert hasattr(trained_model, "estimators_")

def test_model_trainer_can_predict_after_training():

    #Create fake dataset.
    X_train = np.random.rand(10,63)

    y_train = np.array([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1
    ])

    model = RandomForestClassifier(random_state=42)

    #Train model.
    trained_model = model_trainer(X_train,y_train, model)

    #Predict sample.
    prediction = trained_model.predict( X_train)

    #Check prediction.
    assert isinstance(prediction, np.ndarray)
    assert len(prediction) == len(X_train)