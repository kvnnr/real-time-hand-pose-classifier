from src.data.save_pose import SavePoseDataset
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch


class TestStorePoseDatasetValidation:

    """

    Unit tests for Store Pose Dataset (Validation) module.

    Variables:
    self.test_dataset ← Main object

    self.valid_landmark, self.not_list_landmark, self.empty_landmark
    self.non_numeric_landmark, self.wrong_size_landmark
    self.nan_landmark, self.inf_landmark
    self.valid_label, self.non_string_label, self.empty_label

    """

    def setup_method(self):

        """
        This runs automatically before every individual test.
        It ensures a fresh instance of StorePoseDataset is always available.
        """

        #Create a new object of Store Pose Dataset class.
        self.test_dataset = SavePoseDataset()

        #-----------------------------
        # Helper: Mock Landmark Lists
        #-----------------------------

        #Valid landmark (63 flattened values)
        self.valid_landmark = [0.1] * 63

        #Not a list variable
        self.not_list_landmark = "invalid_input"

        #Empty landmark list variable
        self.empty_landmark = []

        #Non numeric landmark variable
        self.non_numeric_landmark = ["a", "b", "c"]

        #Wrong size landmark variable (10 instead of 63)
        self.wrong_size_landmark = [0.1] * 10

        #NaN landmark variable
        self.nan_landmark = [0.1] * 63
        self.nan_landmark[0] = float("nan")

        #Inf landmark variable
        self.inf_landmark = [0.1] * 63
        self.inf_landmark[0] = float("inf")

        #-----------------------------
        # Helper: Mock Labels
        #-----------------------------

        #Valid label variable
        self.valid_label = "fist"

        #Non string label variable
        self.non_string_label = 123

        #Empty label variable
        self.empty_label = "   "

    #Test valid landmark and label case.
    def test_valid_input(self):

        #Store the return of validation method.
        landmark_array, label = self.test_dataset.validate_input(self.valid_landmark, self.valid_label)

        #Check if it returns a numpy array and stripped label after validation.
        assert isinstance(landmark_array, np.ndarray), \
            "Error: Something wrong with validation of landmarks."
        assert label == "fist", \
            "Error: Something wrong with validation of labels."

    #Test landmark not a list.
    def test_not_list_landmark(self):

        #Check if it raises error when landmark is not a list.
        with pytest.raises(TypeError, match="Landmarks in Dataset should be a list!"):
            self.test_dataset.validate_input(self.not_list_landmark, self.valid_label) #type: ignore

    #Test empty landmark.
    def test_empty_landmark(self):

        #Check if it raises error when landmark is empty.
        with pytest.raises(ValueError, match="Landmarks must not be empty"):
            self.test_dataset.validate_input(self.empty_landmark, self.valid_label)

    #Test non numeric landmark.
    def test_non_numeric_landmark(self):

        #Check if it raises error when landmark is not numeric.
        with pytest.raises(TypeError, match="Landmarks must be numeric floats"):
            self.test_dataset.validate_input(self.non_numeric_landmark, self.valid_label)

    #Test landmark not 63 values.
    def test_wrong_size_landmark(self):

        #Check if it raises error when landmark does not have 63 values.
        with pytest.raises(ValueError, match="Landmarks must have 63 values!"):
            self.test_dataset.validate_input(self.wrong_size_landmark, self.valid_label)

    #Test landmark with NaN value.
    def test_nan_landmark(self):

        #Check if it raises error when landmark contains NaN.
        with pytest.raises(ValueError, match="Landmarks contain NaN or Inf values"):
            self.test_dataset.validate_input(self.nan_landmark, self.valid_label)

    #Test landmark with Inf value.
    def test_inf_landmark(self):

        #Check if it raises error when landmark contains Inf.
        with pytest.raises(ValueError, match="Landmarks contain NaN or Inf values"):
            self.test_dataset.validate_input(self.inf_landmark, self.valid_label)

    #Test label not a string.
    def test_non_string_label(self):

        #Check if it raises error when label is not a string.
        with pytest.raises(TypeError, match="Labels must be string!"):
            self.test_dataset.validate_input(self.valid_landmark, self.non_string_label) #type: ignore

    #Test empty label.
    def test_empty_label(self):

        #Check if it raises error when label is empty.
        with pytest.raises(ValueError, match="Labels must not be empty!"):
            self.test_dataset.validate_input(self.valid_landmark, self.empty_label)


class TestStorePoseDatasetAddPose:

    """

    Unit tests for Store Pose Dataset (add_pose) module.

    Variables:
    self.test_dataset ← Main object

    self.valid_landmark, self.invalid_landmark
    self.label_one, self.label_two

    """

    def setup_method(self):

        """
        This runs automatically before every individual test.
        It ensures a fresh instance of StorePoseDataset is always available.
        """

        #Create a new object of Store Pose Dataset class.
        self.test_dataset = SavePoseDataset()

        #-----------------------------
        # Helper: Mock Landmark & Labels
        #-----------------------------

        #Valid landmark variable
        self.valid_landmark = [0.1] * 63

        #Invalid landmark variable (empty)
        self.invalid_landmark = []

        #Label variables
        self.label_one = "fist"
        self.label_two = "open_palm"

    #Test add_pose stores landmark and label correctly.
    def test_add_pose_appends_data(self):

        #Add a single pose to temporary storage.
        self.test_dataset.add_pose(self.valid_landmark, self.label_one)

        #Check if landmark and label were appended correctly.
        assert len(self.test_dataset.x) == 1, \
            "Error: Something wrong with storing landmarks."
        assert self.test_dataset.y[0] == "fist", \
            "Error: Something wrong with storing labels."
        assert "fist" in self.test_dataset.labels, \
            "Error: Something wrong with tracking unique labels."

    #Test add_pose keeps labels unique.
    def test_add_pose_unique_labels(self):

        #Add poses, one label repeated.
        self.test_dataset.add_pose(self.valid_landmark, self.label_one)
        self.test_dataset.add_pose(self.valid_landmark, self.label_one)
        self.test_dataset.add_pose(self.valid_landmark, self.label_two)

        #Check duplicate label did not create a duplicate entry.
        assert len(self.test_dataset.labels) == 2, \
            "Error: Something wrong with unique label tracking."
        assert len(self.test_dataset.x) == 3, \
            "Error: Something wrong with storing repeated landmarks."

    #Test add_pose raises error on invalid landmark.
    def test_add_pose_invalid_landmark(self):

        #Check if it raises error when landmark is invalid.
        with pytest.raises(ValueError, match="Landmarks must not be empty"):
            self.test_dataset.add_pose(self.invalid_landmark, self.label_one)


class TestStorePoseDatasetClearStorage:

    """

    Unit tests for Store Pose Dataset (clear_temporary_storage) module.

    Variables:
    self.test_dataset ← Main object

    self.valid_landmark, self.valid_label

    """

    def setup_method(self):

        """
        This runs automatically before every individual test.
        It ensures a fresh instance of StorePoseDataset is always available.
        """

        #Create a new object of Store Pose Dataset class.
        self.test_dataset = SavePoseDataset()

        #Valid landmark and label variables
        self.valid_landmark = [0.1] * 63
        self.valid_label = "fist"

    #Test clear_temporary_storage wipes all buffers.
    def test_clear_temporary_storage(self):

        #Add data then clear it.
        self.test_dataset.add_pose(self.valid_landmark, self.valid_label)
        self.test_dataset.clear_temporary_storage()

        #Check if all internal buffers were wiped.
        assert self.test_dataset.x == [], \
            "Error: Something wrong with clearing landmark storage."
        assert self.test_dataset.y == [], \
            "Error: Something wrong with clearing label storage."
        assert self.test_dataset.labels == set(), \
            "Error: Something wrong with clearing unique labels."


class TestStorePoseDatasetSave:

    """

    Unit tests for Store Pose Dataset (save_dataset) module.

    Variables:
    self.test_dataset ← Main object

    self.valid_landmark, self.valid_label

    Note:
    save_dataset touches the filesystem (mkdir + np.savez_compressed),
    so those are mocked out instead of writing real files during tests.

    """

    def setup_method(self):

        """
        This runs automatically before every individual test.
        It ensures a fresh instance of StorePoseDataset is always available.
        """

        #Create a new object of Store Pose Dataset class.
        self.test_dataset = SavePoseDataset()

        #Valid landmark and label variables
        self.valid_landmark = [0.1] * 63
        self.valid_label = "fist"

    #Test save_dataset skips writing when storage is empty.
    def test_save_with_no_data(self, caplog):

        #Mock the external dependency (numpy file write).
        with patch("numpy.savez_compressed") as mock_savez:
            self.test_dataset.save_dataset(filepath="dummy_path.npz")

        #Check if savez was never called and warning was logged.
        assert not mock_savez.called, \
            "Error: Something wrong, savez should not be called with no data."
        assert "No data found" in caplog.text, \
            "Error: Something wrong, warning log was not triggered."

    #Test save_dataset creates parent directory and writes data.
    def test_save_creates_dir_and_writes(self, tmp_path):

        #Add data then save to a fake nested path.
        self.test_dataset.add_pose(self.valid_landmark, self.valid_label)
        fake_filepath = tmp_path / "nested" / "hand_pose_dataset.npz"

        #Mock the external dependency (numpy file write).
        with patch("numpy.savez_compressed") as mock_savez:
            self.test_dataset.save_dataset(filepath=fake_filepath)

        #Check if parent directory was created and savez was called once.
        assert fake_filepath.parent.exists(), \
            "Error: Something wrong with creating parent directory."
        mock_savez.assert_called_once()

    #Test save_dataset clears storage after a successful save.
    def test_save_clears_storage_after_save(self, tmp_path):

        #Add data then save it.
        self.test_dataset.add_pose(self.valid_landmark, self.valid_label)
        fake_filepath = tmp_path / "hand_pose_dataset.npz"

        #Mock the external dependency (numpy file write).
        with patch("numpy.savez_compressed"):
            self.test_dataset.save_dataset(filepath=fake_filepath)

        #Check if temporary storage was cleared after saving.
        assert self.test_dataset.x == [], \
            "Error: Something wrong, storage was not cleared after save."

    #Test save_dataset raises error when directory creation fails.
    def test_save_raises_on_mkdir_failure(self, tmp_path):

        #Add data then attempt to save.
        self.test_dataset.add_pose(self.valid_landmark, self.valid_label)
        fake_filepath = tmp_path / "hand_pose_dataset.npz"

        #Mock the external dependency (filesystem mkdir) to simulate failure.
        with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
            with pytest.raises(RuntimeError, match="Could not create directory"):
                self.test_dataset.save_dataset(filepath=fake_filepath)