from sklearn.base import ClassifierMixin
from src.model.training_model_helpers.model_trainer.trainer_validator import _validate_input
import numpy as np
from src.model.schemas.dataset_schema import DatasetSplit

"""
What:
    Train the selected machine learning model using
    prepared landmark datasets.

Preconditions:
    dataset_schema.py:
    Store all output in the dataclass for easy access.

Responsibilities:
    - Validate training inputs.
    - Fit the model using training data.
    - Return the trained model.

Input:
    #NOTE: splitted_dataset: DATACLASS (DatasetSplit)
        →X_train: np.ndarray
            Training features containing normalized landmarks.

        →y_train: np.ndarray
            Training labels corresponding to each sample.

Process:
    1. Unpack X_train and y_train from DatasetSplit
    2. Validate X_train and y_train.
    3. Train the selected model.
    4. Return the fitted model.

Output:
    model:
        A trained machine learning classifier.

Failure Conditions:
    - X_train or y_train is not a numpy array.
    - Training data is empty.
    - Feature and label sizes do not match.
    - Model training fails.

Invariants:
    - Returned model must be fitted.
"""

#Training the model. | Return: model
def model_trainer(splitted_dataset: DatasetSplit, model: ClassifierMixin) -> ClassifierMixin:
    
    #Unpack X and y train.
    X_train = splitted_dataset.X_train
    y_train = splitted_dataset.y_train

    _validate_input(X_train, y_train, model)

    #Train the model and try to catch some error while training.
    try:
        model.fit(X_train,y_train)
    except Exception as error:
        raise RuntimeError("Model training failed.") from error

    return model