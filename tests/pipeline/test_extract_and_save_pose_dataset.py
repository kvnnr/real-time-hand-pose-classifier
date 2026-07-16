
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.pipeline.extract_and_save_pose_dataset import SavePoseDatasetsPipeline


@pytest.fixture
def pipeline() -> SavePoseDatasetsPipeline:
    
    """
    Provides a SavePoseDatasetsPipeline with mocked
    collaborators.

    Process:
        1. Patch HandDetector, Normalizer, SavePoseDataset,
           and ExtractLandmarkPipeline.
        2. Instantiate SavePoseDatasetsPipeline.

    Output:
        SavePoseDatasetsPipeline:
            Instance backed by mocked objects.
    """

    with (
        patch("src.pipeline.extract_and_save_pose_dataset.HandDetector"),
        patch("src.pipeline.extract_and_save_pose_dataset.Normalizer"),
        patch("src.pipeline.extract_and_save_pose_dataset.SavePoseDataset"),
        patch("src.pipeline.extract_and_save_pose_dataset.ExtractLandmarkPipeline"),
        ):

        instance = SavePoseDatasetsPipeline()

    return instance


def make_rgb_frame() -> np.ndarray:
    
    """
    Builds a valid RGB frame for testing.

    Output:
        np.ndarray:
            Shape (height, width, channels).
    """

    return np.zeros((480, 640, 3))


class TestInit:
    """
    Tests pipeline construction.
    """

    def test_initializes_all_collaborators(self) -> None:
        
        """
        Confirms detect, normalizer, save_landmark, and
        extract_landmark are constructed on init.
        """

        with (
            patch("src.pipeline.extract_and_save_pose_dataset.HandDetector") as mock_detector_cls,
            patch("src.pipeline.extract_and_save_pose_dataset.Normalizer") as mock_normalizer_cls,
            patch("src.pipeline.extract_and_save_pose_dataset.SavePoseDataset") as mock_save_cls,
            patch("src.pipeline.extract_and_save_pose_dataset.ExtractLandmarkPipeline") as mock_extract_cls,
            ):

            instance = SavePoseDatasetsPipeline()

        assert instance.detect is mock_detector_cls.return_value
        assert instance.normalizer is mock_normalizer_cls.return_value
        assert instance.save_landmark is mock_save_cls.return_value
        assert instance.extract_landmark is mock_extract_cls.return_value

        # Confirm ExtractLandmarkPipeline receives detect and normalizer.
        mock_extract_cls.assert_called_once_with(instance.detect, instance.normalizer)


class TestValidateLabel:
    """
    Tests _validate_label().
    """

    def test_raises_type_error_when_label_not_str(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms non-string label raises TypeError.
        """

        with pytest.raises(TypeError):
            pipeline._validate_label(123)

    def test_raises_value_error_when_label_empty(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms empty/whitespace label raises ValueError.
        """

        with pytest.raises(ValueError):
            pipeline._validate_label("   ")

    def test_accepts_valid_label(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms a valid label does not raise.
        """

        pipeline._validate_label("open_palm")


class TestValidateFrame:
    """
    Tests _validate_frame().
    """

    def test_raises_type_error_when_frame_not_ndarray(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms non-ndarray frame raises TypeError.
        """

        with pytest.raises(TypeError):
            pipeline._validate_frame([[0, 0, 0]])

    def test_raises_value_error_when_frame_empty(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms empty frame raises ValueError.
        """

        # Empty frame.
        frame = np.array([])

        with pytest.raises(ValueError):
            pipeline._validate_frame(frame)

    def test_raises_value_error_when_frame_not_3d(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms non-3D frame raises ValueError.
        """

        # Missing channel dimension.
        frame = np.zeros((480, 640))

        with pytest.raises(ValueError):
            pipeline._validate_frame(frame)

    def test_accepts_valid_frame(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms a valid RGB frame does not raise.
        """

        pipeline._validate_frame(make_rgb_frame())


class TestValidateLandmarks:
    """
    Tests _validate_landmarks().
    """

    def test_raises_type_error_when_landmarks_not_list(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms non-list landmarks raises TypeError.
        """

        with pytest.raises(TypeError):
            pipeline._validate_landmarks((0.1, 0.2, 0.3))

    def test_raises_value_error_when_landmarks_empty(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms empty landmark list raises ValueError.
        """

        with pytest.raises(ValueError):
            pipeline._validate_landmarks([])

    def test_raises_type_error_when_landmarks_not_numeric(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms non-numeric landmark values raise TypeError.
        """

        with pytest.raises(TypeError):
            pipeline._validate_landmarks([0.1, "0.2", 0.3])

    def test_accepts_valid_landmarks(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms a valid numeric landmark list does not raise.
        """

        pipeline._validate_landmarks([0.1, 0.2, 0.3])


class TestExtractAndSaveLandmark:
    """
    Tests extract_and_save_landmark().
    """

    def test_returns_landmarks_and_saves_dataset(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms a detected hand returns the extracted
        landmarks and saves the dataset.
        """

        # Fake extracted landmarks.
        landmarks = [0.1, 0.2, 0.3]
        pipeline.extract_landmark.extract.return_value = landmarks

        result = pipeline.extract_and_save_landmark("open_palm", make_rgb_frame())

        assert result == landmarks

        # Confirm the pose was added and saved.
        pipeline.save_landmark.add_pose.assert_called_once_with(landmarks, "open_palm")
        pipeline.save_landmark.save_dataset.assert_called_once()

    def test_returns_none_when_no_hand_detected(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms None is returned when no hand is detected,
        and the dataset is not saved.
        """

        # No hand detected.
        pipeline.extract_landmark.extract.return_value = None

        result = pipeline.extract_and_save_landmark("open_palm", make_rgb_frame())

        assert result is None

        pipeline.save_landmark.add_pose.assert_not_called()
        pipeline.save_landmark.save_dataset.assert_not_called()

    def test_raises_type_error_when_label_invalid(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms invalid label short-circuits before
        extraction.
        """

        with pytest.raises(TypeError):
            pipeline.extract_and_save_landmark(123, make_rgb_frame())

        pipeline.extract_landmark.extract.assert_not_called()

    def test_raises_value_error_when_frame_invalid(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms invalid frame short-circuits before
        extraction.
        """

        # Empty frame.
        frame = np.array([])

        with pytest.raises(ValueError):
            pipeline.extract_and_save_landmark("open_palm", frame)

        pipeline.extract_landmark.extract.assert_not_called()

    def test_raises_value_error_when_extracted_landmarks_empty(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms an empty extracted landmark list raises
        ValueError and is not saved.
        """

        # Extraction returns an empty list.
        pipeline.extract_landmark.extract.return_value = []

        with pytest.raises(ValueError):
            pipeline.extract_and_save_landmark("open_palm", make_rgb_frame())

        pipeline.save_landmark.add_pose.assert_not_called()
        pipeline.save_landmark.save_dataset.assert_not_called()

    def test_calls_extract_with_rgb_frame(self, pipeline: SavePoseDatasetsPipeline) -> None:
        
        """
        Confirms extract() is called with the given RGB
        frame.
        """

        pipeline.extract_landmark.extract.return_value = [0.1, 0.2, 0.3]

        frame = make_rgb_frame()

        pipeline.extract_and_save_landmark("open_palm", frame)

        pipeline.extract_landmark.extract.assert_called_once_with(frame)