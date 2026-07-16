import numpy as np
import pytest

from pathlib import Path
from unittest.mock import patch

from src.data.save_pose import SavePoseDataset


#-----------------------------------------------------------------------------
#Helper: builds a valid 63-value landmark list.
#-----------------------------------------------------------------------------
def make_landmark(value: float = 0.5) -> list[float]:

    """
    Variables:
        value: float value repeated 63 times to form a flattened landmark vector.
    """

    return [value] * 63


class TestClearTemporaryStorage:

    def setup_method(self):

        """
        Variables:
            self.dataset: SavePoseDataset instance
        """

        self.dataset = SavePoseDataset()

    #Test clearing wipes x, y, and labels case.
    def test_clear_temporary_storage_wipes_buffers(self):

        self.dataset.x.append(np.zeros(63, dtype=np.float32))
        self.dataset.y.append("fist")
        self.dataset.labels.add("fist")

        self.dataset.clear_temporary_storage()

        assert self.dataset.x == []
        assert self.dataset.y == []
        assert self.dataset.labels == set()


class TestValidateInput:

    def setup_method(self):

        """
        Variables:
            self.dataset: SavePoseDataset instance
        """

        self.dataset = SavePoseDataset()

    #Test valid landmark and label return array and stripped string case.
    def test_validate_input_valid_returns_array_and_label(self):

        landmark = make_landmark()

        result_landmark, result_label = self.dataset.validate_input(landmark, " fist ")

        assert isinstance(result_landmark, np.ndarray)
        assert result_landmark.shape == (63,)
        assert result_label == "fist"

    #Test non-list landmark raises TypeError case.
    def test_validate_input_landmark_not_list_raises_type_error(self):

        with pytest.raises(TypeError):
            self.dataset.validate_input("not a list", "fist")

    #Test empty landmark list raises ValueError case.
    def test_validate_input_empty_landmark_raises_value_error(self):

        with pytest.raises(ValueError):
            self.dataset.validate_input([], "fist")

    #Test non-numeric landmark values raise TypeError case.
    def test_validate_input_non_numeric_landmark_raises_type_error(self):

        landmark = ["bad"] * 63

        with pytest.raises(TypeError):
            self.dataset.validate_input(landmark, "fist")

    #Test nested landmark list raises ValueError case.
    def test_validate_input_nested_landmark_raises_value_error(self):

        landmark = [[0.5] * 3] * 21

        with pytest.raises(ValueError):
            self.dataset.validate_input(landmark, "fist")

    #Test wrong landmark size raises ValueError case.
    def test_validate_input_wrong_size_raises_value_error(self):

        landmark = make_landmark()[:-1]

        with pytest.raises(ValueError):
            self.dataset.validate_input(landmark, "fist")

    #Test NaN in landmark raises ValueError case.
    def test_validate_input_nan_landmark_raises_value_error(self):

        landmark = make_landmark()
        landmark[0] = float("nan")

        with pytest.raises(ValueError):
            self.dataset.validate_input(landmark, "fist")

    #Test Inf in landmark raises ValueError case.
    def test_validate_input_inf_landmark_raises_value_error(self):

        landmark = make_landmark()
        landmark[0] = float("inf")

        with pytest.raises(ValueError):
            self.dataset.validate_input(landmark, "fist")

    #Test non-string label raises TypeError case.
    def test_validate_input_label_not_string_raises_type_error(self):

        with pytest.raises(TypeError):
            self.dataset.validate_input(make_landmark(), 123)

    #Test empty label raises ValueError case.
    def test_validate_input_empty_label_raises_value_error(self):

        with pytest.raises(ValueError):
            self.dataset.validate_input(make_landmark(), "   ")


class TestAddPose:

    def setup_method(self):

        """
        Variables:
            self.dataset: SavePoseDataset instance
        """

        self.dataset = SavePoseDataset()

    #Test adding a valid pose stores landmark, label, and unique label case.
    def test_add_pose_stores_landmark_and_label(self):

        self.dataset.add_pose(make_landmark(), "fist")

        assert len(self.dataset.x) == 1
        assert self.dataset.y == ["fist"]
        assert self.dataset.labels == {"fist"}

    #Test adding an invalid pose raises and does not store anything case.
    def test_add_pose_invalid_input_raises_and_skips_storage(self):

        with pytest.raises(ValueError):
            self.dataset.add_pose([], "fist")

        assert self.dataset.x == []
        assert self.dataset.y == []


class TestSaveDataset:

    def setup_method(self):

        """
        Variables:
            self.dataset: SavePoseDataset instance
        """

        self.dataset = SavePoseDataset()

    #Test saving with no data logs a warning and skips writing case.
    def test_save_dataset_no_data_warns_and_skips(self, tmp_path: Path):

        filepath = tmp_path / "dataset.npz"

        with patch.object(self.dataset.logger, "warning") as mock_warning:
            self.dataset.save_dataset(filepath)

        mock_warning.assert_called_once()
        assert not filepath.exists()

    #Test saving new data creates a fresh NPZ file case.
    def test_save_dataset_creates_new_file(self, tmp_path: Path):

        filepath = tmp_path / "nested" / "dataset.npz"

        self.dataset.add_pose(make_landmark(), "fist")
        self.dataset.save_dataset(filepath)

        assert filepath.exists()

        saved = np.load(filepath)

        assert saved["X"].shape == (1, 63)
        assert list(saved["y"]) == ["fist"]

    #Test saving appends to an existing dataset case.
    def test_save_dataset_appends_to_existing_file(self, tmp_path: Path):

        filepath = tmp_path / "dataset.npz"

        self.dataset.add_pose(make_landmark(0.1), "fist")
        self.dataset.save_dataset(filepath)

        self.dataset.add_pose(make_landmark(0.9), "open_palm")
        self.dataset.save_dataset(filepath)

        saved = np.load(filepath)

        assert saved["X"].shape == (2, 63)
        assert sorted(saved["y"]) == ["fist", "open_palm"]

    #Test saving clears temporary storage afterward case.
    def test_save_dataset_clears_storage_after_save(self, tmp_path: Path):

        filepath = tmp_path / "dataset.npz"

        self.dataset.add_pose(make_landmark(), "fist")
        self.dataset.save_dataset(filepath)

        assert self.dataset.x == []
        assert self.dataset.y == []
        assert self.dataset.labels == set()

    #Test directory creation failure raises RuntimeError case.
    def test_save_dataset_mkdir_failure_raises_runtime_error(self, tmp_path: Path):

        filepath = tmp_path / "dataset.npz"

        self.dataset.add_pose(make_landmark(), "fist")

        with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
            with pytest.raises(RuntimeError):
                self.dataset.save_dataset(filepath)