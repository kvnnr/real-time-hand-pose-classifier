
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.pipeline.extract_landmark import ExtractLandmarkPipeline


@pytest.fixture
def mock_detector() -> MagicMock:
    
    """
    Provides a fake HandDetector.

    Output:
        MagicMock:
            Mimics detect_hands().
    """

    return MagicMock()


@pytest.fixture
def mock_normalizer() -> MagicMock:
    
    """
    Provides a fake Normalizer.

    Output:
        MagicMock:
            Mimics transform().
    """

    return MagicMock()


@pytest.fixture
def pipeline(mock_detector: MagicMock, mock_normalizer: MagicMock) -> ExtractLandmarkPipeline:
    
    """
    Provides an ExtractLandmarkPipeline with mocked
    collaborators.

    Input:
        mock_detector: MagicMock
        mock_normalizer: MagicMock

    Output:
        ExtractLandmarkPipeline:
            Instance ready for testing.
    """

    return ExtractLandmarkPipeline(mock_detector, mock_normalizer)


def make_rgb_frame() -> np.ndarray:
    
    """
    Builds a fake RGB frame for testing.

    Output:
        np.ndarray:
            Shape (height, width, channels).
    """

    return np.zeros((480, 640, 3))


class TestInit:
    """
    Tests pipeline construction.
    """

    def test_sets_detector_and_normalizer(self, mock_detector: MagicMock, mock_normalizer: MagicMock) -> None:
        
        """
        Confirms detector and normalizer are stored as-is.
        """

        pipeline = ExtractLandmarkPipeline(mock_detector, mock_normalizer)

        assert pipeline.detector is mock_detector
        assert pipeline.normalizer is mock_normalizer


class TestExtract:
    """
    Tests extract().
    """

    def test_returns_normalized_landmarks_when_hand_detected(
        self,
        pipeline: ExtractLandmarkPipeline,
        mock_detector: MagicMock,
        mock_normalizer: MagicMock,
        ) -> None:
        
        """
        Confirms detected landmarks are passed to the
        normalizer, and the normalized result is returned.
        """

        # Fake raw and normalized landmarks.
        unnormalized_landmarks = MagicMock(name="unnormalized_landmarks")
        normalized_landmarks = [0.1, 0.2, 0.3]

        mock_detector.detect_hands.return_value = unnormalized_landmarks
        mock_normalizer.transform.return_value = normalized_landmarks

        frame = make_rgb_frame()

        result = pipeline.extract(frame)

        # Detect called with the RGB frame.
        mock_detector.detect_hands.assert_called_once_with(frame)

        # Normalize called with the raw landmarks.
        mock_normalizer.transform.assert_called_once_with(unnormalized_landmarks)

        assert result == normalized_landmarks

    def test_returns_none_when_no_hand_detected(
        self,
        pipeline: ExtractLandmarkPipeline,
        mock_detector: MagicMock,
        mock_normalizer: MagicMock,
        ) -> None:
        
        """
        Confirms None is returned when no hand is detected,
        and normalization is skipped.
        """

        # No hand detected.
        mock_detector.detect_hands.return_value = None

        result = pipeline.extract(make_rgb_frame())

        assert result is None

        mock_normalizer.transform.assert_not_called()

    def test_calls_detect_hands_with_given_frame(
        self,
        pipeline: ExtractLandmarkPipeline,
        mock_detector: MagicMock,
        ) -> None:
        
        """
        Confirms detect_hands() receives the exact frame
        passed to extract().
        """

        mock_detector.detect_hands.return_value = None

        frame = make_rgb_frame()

        pipeline.extract(frame)

        mock_detector.detect_hands.assert_called_once_with(frame)