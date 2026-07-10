from src.pipeline.extract_landmark import LandmarkPipeline
import numpy as np


class FakeDetector:

    #Fake Hand Detector Module
    def detect_hands(self, frame):

        #Return fake unnormalized landmarks
        return [0, 1, 2, 3]


class FakeNormalizer:

    #Fake Normalizer Module
    def transform(self, landmarks):

        #Return fake normalized landmarks
        return [0.1, 0.2, 0.3, 0.4]


class TestLandmarkPipeline:

    """
    Variables:
        pipeline -> object
        frame -> sample input
    """

    def setup_method(self):

        #Object creation
        self.pipeline = LandmarkPipeline(
            FakeDetector(),
            FakeNormalizer()
        )

        #Sample frame input
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    #Check if pipeline correctly extracts landmarks
    def test_extract_landmarks(self):

        #Run pipeline
        result = self.pipeline.extract(self.frame)

        #Expected output from fake normalizer
        expected = [0.1, 0.2, 0.3, 0.4]

        #Assert output is correct
        assert result == expected, \
            "Error: Pipeline did not return correct normalized landmarks"

    #Check if detector is called correctly
    def test_detector_output_is_used(self):

        #Run pipeline
        result = self.pipeline.extract(self.frame)

        #Assert output is list (basic safety check)
        assert isinstance(result, list), \
            "Error: Output should be a list of landmarks"

    #Check pipeline integration flow
    def test_pipeline_flow_integration(self):

        #Run pipeline
        result = self.pipeline.extract(self.frame)

        #Check length consistency
        assert len(result) == 4, \
            "Error: Pipeline output length mismatch"