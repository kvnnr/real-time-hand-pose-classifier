# tests/model/test_train_model_pipeline.py

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sklearn.base import ClassifierMixin

from src.model.train_pipeline import TrainModelPipeline


@pytest.fixture
def mock_model() -> MagicMock:
    
    """
    Provides a fake classifier.

    Output:
        MagicMock:
            Mimics ClassifierMixin.
    """

    return MagicMock(spec=ClassifierMixin)


@pytest.fixture
def mock_config() -> MagicMock:
    
    """
    Provides a fake ModelConfig.

    Output:
        MagicMock:
            Carries dataset_file_path, test_size,
            random_state, trained_models_path.
    """

    config = MagicMock()

    config.dataset_file_path = "dataset/hand_pose_dataset.npz"
    config.test_size = 0.2
    config.random_state = 42
    config.trained_models_path = "trained_models/pose_classifier.joblib"

    return config


@pytest.fixture
def pipeline(mock_model: MagicMock, mock_config: MagicMock) -> TrainModelPipeline:
    
    """
    Provides a TrainModelPipeline instance.

    Input:
        mock_model: MagicMock
        mock_config: MagicMock

    Output:
        TrainModelPipeline:
            Instance ready for testing.
    """

    return TrainModelPipeline(mock_model, mock_config)


class TestInit:
    """
    Tests pipeline construction.
    """

    def test_sets_model_and_config(self, mock_model: MagicMock, mock_config: MagicMock) -> None:
        
        """
        Confirms model and config are stored as-is.
        """

        pipeline = TrainModelPipeline(mock_model, mock_config)

        assert pipeline.model is mock_model
        assert pipeline.config is mock_config

    def test_trained_model_starts_as_none(self, pipeline: TrainModelPipeline) -> None:
        
        """
        Confirms trained_model defaults to None.
        """

        assert pipeline.trained_model is None

    def test_accuracy_starts_as_none(self, pipeline: TrainModelPipeline) -> None:
        
        """
        Confirms accuracy defaults to None.
        """

        assert pipeline.accuracy is None


class TestTrainAndSaveModel:
    """
    Tests the full training pipeline.
    """

    def test_runs_pipeline_in_order_and_returns_accuracy(
        self,
        pipeline: TrainModelPipeline,
        mock_config: MagicMock,
        ) -> None:
        
        """
        Confirms each pipeline step is called with the
        correct arguments, in the correct order, and the
        final accuracy is returned.
        """

        # Fake return values for each step.
        X_y_dataset = MagicMock(name="X_y_dataset")
        splitted_dataset = MagicMock(name="splitted_dataset")
        trained_model = MagicMock(name="trained_model")
        predictions = MagicMock(name="predictions")

        with (
            patch("src.model.train_pipeline.load_dataset", return_value=X_y_dataset) as mock_load_dataset,
            patch("src.model.train_pipeline.split_dataset", return_value=splitted_dataset) as mock_split_dataset,
            patch("src.model.train_pipeline.model_trainer", return_value=trained_model) as mock_model_trainer,
            patch("src.model.train_pipeline.model_predictor", return_value=predictions) as mock_model_predictor,
            patch("src.model.train_pipeline.get_accuracy_score", return_value=0.95) as mock_get_accuracy_score,
            patch("src.model.train_pipeline.save_trained_model") as mock_save_trained_model,
            ):

            accuracy = pipeline.train_and_save_model()

        # Load dataset.
        mock_load_dataset.assert_called_once_with(Path(mock_config.dataset_file_path))

        # Split dataset.
        mock_split_dataset.assert_called_once_with(
            X_y_dataset,
            mock_config.test_size,
            mock_config.random_state,
            )

        # Train model.
        mock_model_trainer.assert_called_once_with(splitted_dataset, pipeline.model)

        # Predict.
        mock_model_predictor.assert_called_once_with(splitted_dataset.X_test, trained_model)

        # Evaluate accuracy.
        mock_get_accuracy_score.assert_called_once_with(predictions, splitted_dataset.y_test)

        # Save trained model.
        mock_save_trained_model.assert_called_once_with(trained_model, Path(mock_config.trained_models_path))

        assert accuracy == 0.95

    def test_sets_trained_model_attribute(self, pipeline: TrainModelPipeline) -> None:
        
        """
        Confirms self.trained_model is set after training.
        """

        trained_model = MagicMock(name="trained_model")

        with (
            patch("src.model.train_pipeline.load_dataset"),
            patch("src.model.train_pipeline.split_dataset"),
            patch("src.model.train_pipeline.model_trainer", return_value=trained_model),
            patch("src.model.train_pipeline.model_predictor"),
            patch("src.model.train_pipeline.get_accuracy_score", return_value=0.8),
            patch("src.model.train_pipeline.save_trained_model"),
            ):

            pipeline.train_and_save_model()

        assert pipeline.trained_model is trained_model

    def test_sets_accuracy_attribute(self, pipeline: TrainModelPipeline) -> None:
        
        """
        Confirms self.accuracy is set after training.
        """

        with (
            patch("src.model.train_pipeline.load_dataset"),
            patch("src.model.train_pipeline.split_dataset"),
            patch("src.model.train_pipeline.model_trainer"),
            patch("src.model.train_pipeline.model_predictor"),
            patch("src.model.train_pipeline.get_accuracy_score", return_value=0.73),
            patch("src.model.train_pipeline.save_trained_model"),
            ):

            pipeline.train_and_save_model()

        assert pipeline.accuracy == 0.73

    def test_returns_float_type(self, pipeline: TrainModelPipeline) -> None:
        
        """
        Confirms train_and_save_model returns a float.
        """

        with (
            patch("src.model.train_pipeline.load_dataset"),
            patch("src.model.train_pipeline.split_dataset"),
            patch("src.model.train_pipeline.model_trainer"),
            patch("src.model.train_pipeline.model_predictor"),
            patch("src.model.train_pipeline.get_accuracy_score", return_value=0.5),
            patch("src.model.train_pipeline.save_trained_model"),
            ):

            accuracy = pipeline.train_and_save_model()

        assert isinstance(accuracy, float)

    def test_propagates_error_when_dataset_loading_fails(self, pipeline: TrainModelPipeline) -> None:
        
        """
        Confirms failure in load_dataset stops the pipeline.
        """

        with patch("src.model.train_pipeline.load_dataset", side_effect=FileNotFoundError):

            with pytest.raises(FileNotFoundError):
                pipeline.train_and_save_model()

    def test_does_not_save_model_when_training_fails(self, pipeline: TrainModelPipeline) -> None:
        
        """
        Confirms save_trained_model is not called if
        model_trainer raises.
        """

        with (
            patch("src.model.train_pipeline.load_dataset"),
            patch("src.model.train_pipeline.split_dataset"),
            patch("src.model.train_pipeline.model_trainer", side_effect=ValueError),
            patch("src.model.train_pipeline.model_predictor"),
            patch("src.model.train_pipeline.get_accuracy_score"),
            patch("src.model.train_pipeline.save_trained_model") as mock_save_trained_model,
            ):

            with pytest.raises(ValueError):
                pipeline.train_and_save_model()

        mock_save_trained_model.assert_not_called()