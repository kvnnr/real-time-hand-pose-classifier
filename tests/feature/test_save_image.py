import numpy as np
import pytest

from pathlib import Path
from unittest.mock import patch

from src.feature.save_pose_image.save_image import SavePoseImage


#-----------------------------------------------------------------------------
#Helper: builds a valid BGR frame.
#-----------------------------------------------------------------------------
def make_frame(height: int = 10, width: int = 10, channels: int = 3) -> np.ndarray:

    """
    Variables:
        height, width, channels: shape of the fake BGR frame.
    """

    return np.zeros((height, width, channels), dtype=np.uint8)


class TestValidate:

    def setup_method(self):

        """
        Variables:
            self.saver: SavePoseImage instance
        """

        self.saver = SavePoseImage()

    #Test valid path and frame returns the frame case.
    def test_validate_valid_returns_frame(self, tmp_path: Path):

        image_path = tmp_path / "fist_00001.jpeg"
        frame = make_frame()

        result = self.saver.validate(image_path, frame)

        assert result is frame

    #Test non-Path image_path returns an error string case.
    def test_validate_invalid_path_returns_string(self):

        frame = make_frame()

        result = self.saver.validate("not a path", frame)

        assert isinstance(result, str)

    #Test non-ndarray frame returns an error string case.
    def test_validate_invalid_frame_type_returns_string(self, tmp_path: Path):

        image_path = tmp_path / "fist_00001.jpeg"

        result = self.saver.validate(image_path, "not a frame")

        assert isinstance(result, str)

    #Test empty frame returns an error string case.
    def test_validate_empty_frame_returns_string(self, tmp_path: Path):

        image_path = tmp_path / "fist_00001.jpeg"
        frame = np.array([], dtype=np.uint8)

        result = self.saver.validate(image_path, frame)

        assert isinstance(result, str)

    #Test frame with wrong number of dimensions returns an error string case.
    def test_validate_invalid_dimensions_returns_string(self, tmp_path: Path):

        image_path = tmp_path / "fist_00001.jpeg"
        frame = np.zeros((10, 10), dtype=np.uint8) #Missing channel dimension.

        result = self.saver.validate(image_path, frame)

        assert isinstance(result, str)


class TestSaveImage:

    def setup_method(self):

        """
        Variables:
            self.saver: SavePoseImage instance
        """

        self.saver = SavePoseImage()

    #Test saving a valid frame builds the correct path and writes the image case.
    def test_save_image_valid_writes_image(self, tmp_path: Path):

        frame = make_frame()

        with patch("src.feature.save_pose_image.save_image.cv.imwrite", return_value=True) as mock_imwrite:
            self.saver.save_image(tmp_path, frame, "fist", 1)

        expected_path = tmp_path / "fist_00001.jpeg"

        assert self.saver.image_path == expected_path
        mock_imwrite.assert_called_once_with(str(expected_path), frame)

    #Test the counter is zero-padded to five digits in the filename case.
    def test_save_image_zero_pads_counter(self, tmp_path: Path):

        frame = make_frame()

        with patch("src.feature.save_pose_image.save_image.cv.imwrite", return_value=True):
            self.saver.save_image(tmp_path, frame, "open_palm", 7)

        assert self.saver.image_path.name == "open_palm_00007.jpeg"

    #Test invalid frame raises ValueError and skips writing case.
    def test_save_image_invalid_frame_raises_value_error(self, tmp_path: Path):

        with patch("src.feature.save_pose_image.save_image.cv.imwrite") as mock_imwrite:
            with pytest.raises(ValueError):
                self.saver.save_image(tmp_path, "not a frame", "fist", 1)

        mock_imwrite.assert_not_called()

    #Test OpenCV write failure raises IOError case.
    def test_save_image_write_failure_raises_io_error(self, tmp_path: Path):

        frame = make_frame()

        with patch("src.feature.save_pose_image.save_image.cv.imwrite", return_value=False):
            with pytest.raises(IOError):
                self.saver.save_image(tmp_path, frame, "fist", 1)