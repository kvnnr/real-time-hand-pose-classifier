import numpy as np
import pytest

from types import SimpleNamespace
from unittest.mock import patch, mock_open, MagicMock

from src.detection.hand_detector import HandDetector


#-----------------------------------------------------------------------------
#Helper: builds a HandDetector instance without touching the real config.json
#or hand_landmarker.task files on disk.
#-----------------------------------------------------------------------------
def build_detector(mock_mp_detector: MagicMock = None) -> HandDetector:

    """
    Build a HandDetector with config loading and MediaPipe initialization mocked out.

    Input:
        mock_mp_detector: MagicMock | None
            Fake MediaPipe HandLandmarker to inject. If None, a fresh MagicMock is used.

    Process:
        Patch open() to return fake config JSON.
        Patch json.load() to return a fake config dict.
        Patch BaseOptions and vision.HandLandmarker.create_from_options.
        Instantiate HandDetector.

    Output:
        HandDetector
            Instance with a mocked internal hand_detector.
    """

    fake_config = {"max_num_hands": 2}
    fake_detector = mock_mp_detector or MagicMock()

    with patch("builtins.open", mock_open(read_data="{}")), \
         patch("src.detection.hand_detector.json.load", return_value=fake_config), \
         patch("src.detection.hand_detector.BaseOptions"), \
         patch("src.detection.hand_detector.vision.HandLandmarker.create_from_options", return_value=fake_detector):

        detector = HandDetector()

    return detector


#-----------------------------------------------------------------------------
#Helper: builds a fake landmark object with x, y, z attributes.
#-----------------------------------------------------------------------------
def make_landmark(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> SimpleNamespace:

    """
    Variables:
        x, y, z: coordinate values for the fake landmark.
    """

    return SimpleNamespace(x=x, y=y, z=z)


class TestHandDetectorInit:

    #Test successful config load and detector creation case.
    def test_init_loads_config_and_creates_detector(self):

        """
        Variables:
            detector: HandDetector built via build_detector()
        """

        detector = build_detector()

        assert detector.config == {"max_num_hands": 2}
        assert detector.hand_detector is not None

    #Test missing config file raises RuntimeError case.
    def test_init_missing_config_raises_runtime_error(self):

        """
        Variables:
            None
        """

        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError):
                HandDetector()

    #Test invalid JSON in config file raises RuntimeError case.
    def test_init_invalid_json_raises_runtime_error(self):

        """
        Variables:
            None
        """

        with patch("builtins.open", mock_open(read_data="not valid json")):
            with pytest.raises(RuntimeError):
                HandDetector()

    #Test injected detector is used instead of a new one case.
    def test_init_uses_provided_detector(self):

        """
        Variables:
            fake_detector: MagicMock standing in for a real MediaPipe HandLandmarker
        """

        fake_detector = MagicMock()
        fake_config = {"max_num_hands": 1}

        with patch("builtins.open", mock_open(read_data="{}")), \
             patch("src.detection.hand_detector.json.load", return_value=fake_config):

            detector = HandDetector(detector=fake_detector)

        assert detector.hand_detector is fake_detector


class TestValidateInput:

    def setup_method(self):

        """
        Variables:
            self.detector: HandDetector built via build_detector()
        """

        self.detector = build_detector()

    #Test valid RGB frame returns the same frame case.
    def test_validate_input_valid_frame_returns_frame(self):

        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        result = self.detector.validate_input(frame)

        assert result is frame

    #Test None frame raises ValueError case.
    def test_validate_input_none_frame_raises_value_error(self):

        with pytest.raises(ValueError):
            self.detector.validate_input(None)

    #Test non-array frame raises TypeError case.
    def test_validate_input_invalid_type_raises_type_error(self):

        with pytest.raises(TypeError):
            self.detector.validate_input([[1, 2, 3]])

    #Test empty frame raises ValueError case.
    def test_validate_input_empty_frame_raises_value_error(self):

        frame = np.array([], dtype=np.uint8)

        with pytest.raises(ValueError):
            self.detector.validate_input(frame)

    #Test wrong number of dimensions raises ValueError case.
    def test_validate_input_invalid_dimensions_raises_value_error(self):

        frame = np.zeros((10, 10), dtype=np.uint8)

        with pytest.raises(ValueError):
            self.detector.validate_input(frame)

    #Test wrong number of color channels raises ValueError case.
    def test_validate_input_invalid_channels_raises_value_error(self):

        frame = np.zeros((10, 10, 4), dtype=np.uint8)

        with pytest.raises(ValueError):
            self.detector.validate_input(frame)

    #Test wrong dtype raises TypeError case.
    def test_validate_input_invalid_dtype_raises_type_error(self):

        frame = np.zeros((10, 10, 3), dtype=np.float32)

        with pytest.raises(TypeError):
            self.detector.validate_input(frame)


class TestValidateHandLandmark:

    def setup_method(self):

        """
        Variables:
            self.detector: HandDetector built via build_detector()
        """

        self.detector = build_detector()

    #Test None landmarks returns None case.
    def test_validate_hand_landmark_none_returns_none(self):

        result = self.detector.validate_hand_landmark(None)

        assert result is None

    #Test empty list returns None case.
    def test_validate_hand_landmark_empty_list_returns_none(self):

        result = self.detector.validate_hand_landmark([])

        assert result is None

    #Test valid landmarks are returned unchanged case.
    def test_validate_hand_landmark_valid_landmarks_returns_landmarks(self):

        hand = [make_landmark() for _ in range(21)]
        landmarks = [hand]

        result = self.detector.validate_hand_landmark(landmarks)

        assert result == landmarks

    #Test non-list landmarks raises TypeError case.
    def test_validate_hand_landmark_invalid_type_raises_type_error(self):

        with pytest.raises(TypeError):
            self.detector.validate_hand_landmark("not a list")

    #Test hand entry that is not a list raises TypeError case.
    def test_validate_hand_landmark_hand_not_list_raises_type_error(self):

        with pytest.raises(TypeError):
            self.detector.validate_hand_landmark(["not a hand list"])

    #Test hand with wrong landmark count raises ValueError case.
    def test_validate_hand_landmark_wrong_landmark_count_raises_value_error(self):

        hand = [make_landmark() for _ in range(20)]

        with pytest.raises(ValueError):
            self.detector.validate_hand_landmark([hand])

    #Test landmark missing x/y/z attributes raises TypeError case.
    def test_validate_hand_landmark_missing_attributes_raises_type_error(self):

        hand = [make_landmark() for _ in range(20)] + [SimpleNamespace()]

        with pytest.raises(TypeError):
            self.detector.validate_hand_landmark([hand])

    #Test landmark with non-numeric x raises TypeError case.
    def test_validate_hand_landmark_non_numeric_x_raises_type_error(self):

        hand = [make_landmark() for _ in range(20)] + [make_landmark(x="bad")]

        with pytest.raises(TypeError):
            self.detector.validate_hand_landmark([hand])


class TestDetectHands:

    def setup_method(self):

        """
        Variables:
            self.mock_mp_detector: MagicMock standing in for the MediaPipe HandLandmarker
            self.detector: HandDetector built with self.mock_mp_detector injected
        """

        self.mock_mp_detector = MagicMock()
        self.detector = build_detector(mock_mp_detector=self.mock_mp_detector)

    #Test valid frame with a detected hand returns landmarks case.
    def test_detect_hands_valid_frame_returns_landmarks(self):

        hand = [make_landmark() for _ in range(21)]
        self.mock_mp_detector.detect.return_value = SimpleNamespace(hand_landmarks=[hand])

        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch("src.detection.hand_detector.mp.Image"):
            result = self.detector.detect_hands(frame)

        assert result == [hand]

    #Test frame with no detected hand returns None case.
    def test_detect_hands_no_hand_returns_none(self):

        self.mock_mp_detector.detect.return_value = SimpleNamespace(hand_landmarks=[])

        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch("src.detection.hand_detector.mp.Image"):
            result = self.detector.detect_hands(frame)

        assert result is None

    #Test invalid frame raises ValueError before MediaPipe runs case.
    def test_detect_hands_invalid_frame_raises_value_error(self):

        with pytest.raises(ValueError):
            self.detector.detect_hands(None)

        self.mock_mp_detector.detect.assert_not_called()

    #Test MediaPipe failure raises RuntimeError case.
    def test_detect_hands_mediapipe_failure_raises_runtime_error(self):

        self.mock_mp_detector.detect.side_effect = Exception("mediapipe blew up")

        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch("src.detection.hand_detector.mp.Image"):
            with pytest.raises(RuntimeError):
                self.detector.detect_hands(frame)


class TestCloseMediapipe:

    def setup_method(self):

        """
        Variables:
            self.mock_mp_detector: MagicMock standing in for the MediaPipe HandLandmarker
            self.detector: HandDetector built with self.mock_mp_detector injected
        """

        self.mock_mp_detector = MagicMock()
        self.detector = build_detector(mock_mp_detector=self.mock_mp_detector)

    #Test close_mediapipe calls close on the underlying detector case.
    def test_close_mediapipe_calls_close_on_detector(self):

        self.detector.close_mediapipe()

        self.mock_mp_detector.close.assert_called_once()