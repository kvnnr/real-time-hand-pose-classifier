# tests/model/test_prediction_model.py

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.Inference.live_model import PredictionModel


@pytest.fixture
def mock_classifier() -> MagicMock:
    
    """
    Provides a fake sklearn classifier.

    Output:
        MagicMock:
            Mimics predict() and predict_proba().
    """

    classifier = MagicMock()

    classifier.predict.return_value = np.array(["open_palm"])
    classifier.predict_proba.return_value = np.array([[0.1, 0.9]])

    return classifier


@pytest.fixture
def model(mock_classifier: MagicMock) -> PredictionModel:
    
    """
    Provides a PredictionModel with a mocked classifier.

    Process:
        1. Patch joblib.load to skip disk access.
        2. Instantiate PredictionModel.

    Output:
        PredictionModel:
            Instance backed by mock_classifier.
    """

    # Skip real file loading.
    with patch("src.Inference.live_model.joblib.load", return_value=mock_classifier):
        instance = PredictionModel()

    return instance


class TestInit:
    """
    Tests model loading on construction.
    """

    def test_loads_model_from_model_path(self, mock_classifier: MagicMock) -> None:
        
        """
        Confirms joblib.load is called with MODEL_PATH.
        """

        # Patch and construct.
        with patch("src.Inference.live_model.joblib.load", return_value=mock_classifier) as mock_load:
            instance = PredictionModel()

        mock_load.assert_called_once_with(PredictionModel.MODEL_PATH)
        assert instance._model is mock_classifier

    def test_raises_file_not_found_error_when_model_missing(self) -> None:
        
        """
        Confirms missing model file propagates FileNotFoundError.
        """

        with patch("src.Inference.live_model.joblib.load", side_effect=FileNotFoundError):

            with pytest.raises(FileNotFoundError):
                PredictionModel()


class TestPredict:
    """
    Tests prediction behavior.
    """

    def test_returns_prediction_and_confidence(self, model: PredictionModel) -> None:
        
        """
        Confirms valid input returns (pose, confidence).
        """

        # Input features.
        features = np.zeros(63)

        prediction, confidence = model.predict(features)

        assert prediction == "open_palm"
        assert confidence == pytest.approx(0.9)

    def test_reshapes_features_before_predicting(self, model: PredictionModel, mock_classifier: MagicMock) -> None:
        
        """
        Confirms features are reshaped to (1, -1) for sklearn.
        """

        # Input features.
        features = np.zeros(63)

        model.predict(features)

        # Check reshaped array passed to predict.
        called_args, _ = mock_classifier.predict.call_args
        passed_features = called_args[0]

        assert passed_features.shape == (1, 63)

    def test_confidence_is_float_type(self, model: PredictionModel) -> None:
        
        """
        Confirms confidence is a plain Python float.
        """

        # Input features.
        features = np.zeros(63)

        _, confidence = model.predict(features)

        assert isinstance(confidence, float)

    def test_prediction_is_str_type(self, model: PredictionModel) -> None:
        
        """
        Confirms prediction is a plain Python string.
        """

        # Input features.
        features = np.zeros(63)

        prediction, _ = model.predict(features)

        assert isinstance(prediction, str)

    def test_raises_type_error_when_features_not_ndarray(self, model: PredictionModel) -> None:
        
        """
        Confirms non-ndarray input raises TypeError.
        """

        # Invalid type.
        features = [0.0] * 63

        with pytest.raises(TypeError):
            model.predict(features)

    def test_raises_value_error_when_features_not_1d(self, model: PredictionModel) -> None:
        
        """
        Confirms wrong-shaped input raises ValueError.
        """

        # Invalid shape.
        features = np.zeros((7, 9))

        with pytest.raises(ValueError):
            model.predict(features)

    def test_raises_attribute_error_when_model_lacks_predict_proba(self, mock_classifier: MagicMock) -> None:
        
        """
        Confirms model without predict_proba raises AttributeError.
        """

        # Remove probability support.
        del mock_classifier.predict_proba

        with patch("src.Inference.live_model.joblib.load", return_value=mock_classifier):
            instance = PredictionModel()

        features = np.zeros(63)

        with pytest.raises(AttributeError):
            instance.predict(features)