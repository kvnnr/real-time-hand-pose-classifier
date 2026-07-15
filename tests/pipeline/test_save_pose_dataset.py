import unittest
from unittest.mock import MagicMock
import numpy as np

from src.pipeline.extract_and_save_pose_dataset import SavePoseDatasetsPipeline


class TestSavePoseDatasetValidation(unittest.TestCase):

    #Initialize objects.
    def setUp(self):

        self.pipeline = SavePoseDatasetsPipeline()


    #-------------
    #Label Validation
    #-------------

    #Test: Valid label.
    def test_validate_label_valid(self):

        self.pipeline._validate_label("OpenPalm")


    #Test: Invalid label type.
    def test_validate_label_invalid_type(self):

        with self.assertRaises(TypeError):
            self.pipeline._validate_label(100)


    #Test: Empty label.
    def test_validate_label_empty(self):

        with self.assertRaises(ValueError):
            self.pipeline._validate_label("")


    #-------------
    #Frame Validation
    #-------------

    #Test: Valid frame.
    def test_validate_frame_valid(self):

        frame = np.zeros((480,640,3), dtype=np.uint8)

        self.pipeline._validate_frame(frame)


    #Test: Invalid frame type.
    def test_validate_frame_invalid_type(self):

        with self.assertRaises(TypeError):
            self.pipeline._validate_frame("frame")


    #Test: Empty frame.
    def test_validate_frame_empty(self):

        frame = np.array([])

        with self.assertRaises(ValueError):
            self.pipeline._validate_frame(frame)


    #Test: Invalid frame dimensions.
    def test_validate_frame_invalid_dimension(self):

        frame = np.zeros((480,640))

        with self.assertRaises(ValueError):
            self.pipeline._validate_frame(frame)


    #-------------
    #Landmark Validation
    #-------------

    #Test: Valid landmarks.
    def test_validate_landmarks_valid(self):

        landmarks = [0.1] * 63

        self.pipeline._validate_landmarks(landmarks)


    #Test: None landmarks.
    def test_validate_landmarks_none(self):

        with self.assertRaises(ValueError):
            self.pipeline._validate_landmarks(None)


    #Test: Invalid landmark type.
    def test_validate_landmarks_invalid_type(self):

        with self.assertRaises(TypeError):
            self.pipeline._validate_landmarks("landmarks")


    #Test: Empty landmark list.
    def test_validate_landmarks_empty(self):

        with self.assertRaises(ValueError):
            self.pipeline._validate_landmarks([])


    #Test: Landmark contains invalid value.
    def test_validate_landmarks_invalid_value(self):

        landmarks = [0.1, 0.5, "ABC"]

        with self.assertRaises(TypeError):
            self.pipeline._validate_landmarks(landmarks)



class TestSavePoseDatasetPipeline(unittest.TestCase):

    #Initialize objects.
    def setUp(self):

        self.pipeline = SavePoseDatasetsPipeline()

        #Mock modules.
        self.pipeline.extract_landmark = MagicMock()
        self.pipeline.save_landmark = MagicMock()

        self.frame = np.zeros((480,640,3), dtype=np.uint8)
        self.label = "OpenPalm"
        self.landmarks = [0.1] * 63

        self.pipeline.extract_landmark.extract.return_value = self.landmarks


    #Test:
    #Extract landmarks → Save dataset.
    def test_extract_and_save_landmark(self):

        result = self.pipeline.extract_and_save_landmark(
            self.label,
            self.frame
        )

        #Verify extraction.
        self.pipeline.extract_landmark.extract.assert_called_once_with(
            self.frame
        )

        #Verify saved landmarks.
        self.pipeline.save_landmark.add_pose.assert_called_once_with(
            self.landmarks,
            self.label
        )

        #Verify dataset saved.
        self.pipeline.save_landmark.save_dataset.assert_called_once()

        #Verify return value.
        self.assertEqual(result, self.landmarks)


if __name__ == "__main__":
    unittest.main()