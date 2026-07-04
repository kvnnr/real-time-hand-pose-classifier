from src.camera.camera import Camera, FrameViewer
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestCameraFeatures:
    """
    Unit tests for Camera module.

    Design goals:
    - Fully isolate hardware dependency (cv.VideoCapture)
    - Ensure deterministic behavior using mocks
    - Validate both success and failure execution paths
    - Keep each test self-contained and state-independent
    """

    def setup_method(self):
        """
        Create fresh instances for each test.

        Why:
        - Prevents shared state between tests
        - Ensures reproducibility
        """

        self.camera = Camera()

        # Reusable synthetic frame (BGR format)
        self.mock_frame_bgr = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]]
            ],
            dtype=np.uint8
        )

        # Expected RGB conversion result
        self.expected_frame_rgb = np.array(
            [
                [[0, 0, 255], [0, 255, 0]],
                [[255, 0, 0], [255, 255, 255]]
            ],
            dtype=np.uint8
        )

    # -----------------------------
    # Helper: create mock camera
    # -----------------------------
    def _create_mock_camera(self, is_opened=True, read_return=(True, None)):
        """
        Centralized mock factory.

        Why:
        - Removes duplication (DRY principle)
        - Makes test intent explicit
        - Allows controlled camera behavior per test
        """

        mock_camera = MagicMock()
        mock_camera.isOpened.return_value = is_opened
        mock_camera.read.return_value = read_return
        mock_camera.release = MagicMock()

        return mock_camera

    # -----------------------------
    # Test: camera open success
    # -----------------------------
    def test_camera_open_success(self):
        """
        Validate that camera initializes correctly when device is available.
        """

        mock_camera = self._create_mock_camera(is_opened=True)

        with patch("src.camera.camera.cv.VideoCapture", return_value=mock_camera):
            self.camera.open_camera()

            assert self.camera.cap is mock_camera, \
                "Camera instance should be assigned when open succeeds"

    # -----------------------------
    # Test: camera open failure
    # -----------------------------
    def test_camera_open_failure_raises_error(self):
        """
        Ensure system fails explicitly when camera cannot be opened.
        """

        mock_camera = self._create_mock_camera(is_opened=False)

        with patch("src.camera.camera.cv.VideoCapture", return_value=mock_camera):
            with pytest.raises(RuntimeError, match="Could not open the camera stream"):
                self.camera.open_camera()

    # -----------------------------
    # Test: frame output type
    # -----------------------------
    def test_get_frame_returns_numpy_array(self):
        """
        Validate that get_frames returns a numpy ndarray.
        """

        mock_camera = self._create_mock_camera(
            is_opened=True,
            read_return=(True, self.mock_frame_bgr)
        )

        with patch("src.camera.camera.cv.VideoCapture", return_value=mock_camera):
            self.camera.open_camera()

            frame = self.camera.get_frames()

            assert isinstance(frame, np.ndarray), \
                "Frame output must be numpy array"

    # -----------------------------
    # Test: frame read failure
    # -----------------------------
    def test_get_frame_read_failure(self):
        """
        Ensure runtime error is raised when frame capture fails.
        """

        mock_camera = self._create_mock_camera(
            is_opened=True,
            read_return=(False, None)
        )

        with patch("src.camera.camera.cv.VideoCapture", return_value=mock_camera):
            self.camera.open_camera()

            with pytest.raises(RuntimeError, match="No frame detected"):
                self.camera.get_frames()

    # -----------------------------
    # Test: camera not initialized
    # -----------------------------
    def test_get_frame_without_initialization(self):
        """
        Ensure system rejects frame requests when camera is not initialized.
        """

        with pytest.raises(RuntimeError, match="Camera has not been initialized"):
            self.camera.get_frames()

    # -----------------------------
    # Test: frame shape contract
    # -----------------------------
    def test_frame_shape_contract(self):
        """
        Validate structural integrity of output frame.
        """

        mock_camera = self._create_mock_camera(
            is_opened=True,
            read_return=(True, self.mock_frame_bgr)
        )

        with patch("src.camera.camera.cv.VideoCapture", return_value=mock_camera):
            self.camera.open_camera()

            frame = self.camera.get_frames()

            assert frame.ndim == 3, "Frame must be 3-dimensional (H, W, C)"
            assert frame.shape[2] == 3, "Frame must have 3 color channels"
            assert frame.shape[0] > 0 and frame.shape[1] > 0, \
                "Frame height and width must be positive"

    # -----------------------------
    # Test: empty frame handling
    # -----------------------------
    def test_empty_frame_handling(self):
        """
        Validate behavior when camera returns empty frame.
        """

        mock_camera = self._create_mock_camera(
            is_opened=True,
            read_return=(True, np.array([]))
        )

        with patch("src.camera.camera.cv.VideoCapture", return_value=mock_camera):
            self.camera.open_camera()

            frame = self.camera.get_frames()

            assert frame.size == 0, "Frame should be empty array"

    # -----------------------------
    # Test: BGR to RGB conversion
    # -----------------------------
    def test_bgr_to_rgb_conversion(self):
        """
        Validate correct color space transformation.
        """

        mock_camera = self._create_mock_camera(
            is_opened=True,
            read_return=(True, self.mock_frame_bgr)
        )

        with patch("src.camera.camera.cv.VideoCapture", return_value=mock_camera):
            self.camera.open_camera()

            frame = self.camera.get_frames()
            rgb = self.camera.bgr_to_rgb(frame)

            assert np.array_equal(rgb, self.expected_frame_rgb), \
                "BGR to RGB conversion must be correct"

    # -----------------------------
    # Test: camera release
    # -----------------------------
    def test_camera_release(self):

        """
        Ensure camera resource is properly released.
        """

        mock_camera = self._create_mock_camera(is_opened=True)

        with patch("src.camera.camera.cv.VideoCapture", return_value=mock_camera):
            self.camera.open_camera()
            self.camera.release_camera()

            assert self.camera.cap is None, \
                "Camera reference must be cleared after release"

            mock_camera.release.assert_called_once()

class TestFrameViewer:
    
    """
    Unit tests for FrameViewer class.

    Design goals:
    - Isolate OpenCV dependencies (imshow, waitKey, destroyAllWindows)
    - Validate correct delegation behavior
    - Ensure stateless UI operations
    - Avoid testing OpenCV internals
    """

    def setup_method(self):
        """
        Create fresh instance per test.

        Why:
        - Prevent shared state
        - Ensure deterministic behavior
        """
        self.viewer = FrameViewer(window_name="TestWindow")

        self.mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # -----------------------------
    # Test: initialization contract
    # -----------------------------
    def test_window_name_initialization(self):
        """
        Validate window name is correctly assigned.
        """
        assert self.viewer.window_name == "TestWindow"

    # -----------------------------
    # Test: show delegates to cv.imshow
    # -----------------------------
    def test_show_calls_imshow(self):
        """
        Ensure show() correctly delegates to OpenCV imshow.
        """
        with patch("src.camera.camera.cv.imshow") as mock_imshow:
            self.viewer.show(self.mock_frame)

            mock_imshow.assert_called_once_with(
                "TestWindow",
                self.mock_frame
            )

    # -----------------------------
    # Test: waitKey bitmask logic
    # -----------------------------
    def test_operation_key_bitmask(self):
        """
        Ensure waitKey return is correctly masked with 0xFF.
        """
        with patch("src.camera.camera.cv.waitKey", return_value=0x123):
            key = self.viewer.operation_key()

            assert key == (0x123 & 0xFF)

    # -----------------------------
    # Test: waitKey delay contract
    # -----------------------------
    def test_operation_key_delay(self):
        """
        Ensure waitKey is called with correct delay parameter.
        """
        with patch("src.camera.camera.cv.waitKey") as mock_wait:
            mock_wait.return_value = 0

            self.viewer.operation_key()

            mock_wait.assert_called_once_with(1)

    # -----------------------------
    # Test: close delegates correctly
    # -----------------------------
    def test_close_calls_destroy_all_windows(self):
        """
        Ensure close() properly triggers OpenCV cleanup.
        """
        with patch("src.camera.camera.cv.destroyAllWindows") as mock_destroy:
            self.viewer.close()

            mock_destroy.assert_called_once()

    # -----------------------------
    # Test: show return behavior
    # -----------------------------
    def test_show_returns_none_behavior(self):
        """
        Ensure show does not alter OpenCV behavior contract.
        """
        with patch("src.camera.camera.cv.imshow", return_value=None):
            result = self.viewer.show(self.mock_frame)

            assert result is None

    # -----------------------------
    # Test: stateless behavior
    # -----------------------------
    def test_show_is_stateless(self):
        """
        Ensure repeated calls do not modify internal state.
        """
        with patch("src.camera.camera.cv.imshow") as mock_imshow:
            for _ in range(5):
                self.viewer.show(self.mock_frame)

            assert mock_imshow.call_count == 5

    # -----------------------------
    # Test: no side effect on state
    # -----------------------------
    def test_window_name_unchanged_after_operations(self):
        """
        Ensure UI operations do not mutate internal configuration.
        """
        original_name = self.viewer.window_name

        with patch("src.camera.camera.cv.imshow"):
            self.viewer.show(self.mock_frame)

        assert self.viewer.window_name == original_name