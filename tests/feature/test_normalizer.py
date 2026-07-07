from src.feature.landmark_normalizer.normalizer import Normalizer
import pytest

#Test for validation feature
class TestNormalizerValidation():
    VALID_LANDMARKS = 21

    #normalizer | self.valid_input
    def setup_method(self):
        """
        This runs automatically before every individual test.
        It ensures a fresh instance of Normalizer is always available.
        """
        #Resuable Valid Mock data
        self.normalizer = Normalizer()
        self.valid_landmarks= [[float(i), float(i), float(i)] for i in range(self.VALID_LANDMARKS)]
        self.invalid_landmark_amount = [[float(i), float(i), float(i)] for i in range(10)]
        self.invalid_coord_data_type = [[1.2, 3.5, 'Keven']] * self.VALID_LANDMARKS
        self.invalid_coord_amount = [[float(i), float(i)] for i in range(self.VALID_LANDMARKS)]
        self.invalid_data_type = "Keven"

    #Test if Landmark is a valid Data stype
    def test_validate_not_list(self):
        with pytest.raises(TypeError, match= "Expected a list."):
            self.normalizer.validate(self.invalid_data_type)
    
    #Test if Landmark is empty
    def test_validate_empty_landmark(self):
        with pytest.raises(ValueError, match="Landmarks can't be empty."):
            self.normalizer.validate([])

    #Test if List have 21 Landmarks
    def test_validate_requires_21_landmarks(self):
        with pytest.raises(ValueError, match= ("Expected 21 landmarks.")):
            self.normalizer.validate(self.invalid_landmark_amount)
    
    #Test if landmark have 3 coordinates
    def test_validate_landmark_requires_3_coord(self):
        with pytest.raises(ValueError, match ='Expected each landmark to be a list of 3 coordinates.'):
            self.normalizer.validate(self.invalid_coord_amount)
    
    #Test if Coordiates are Valid Data type
    def test_validate_coordinate_data_type(self):
        with pytest.raises(TypeError, match = "Coordinates must be float"):
            self.normalizer.validate(self.invalid_coord_data_type)
    
    #Test if the Landmark is valid
    def test_validate_valid_input(self):
        self.normalizer.validate(self.valid_landmarks)

#Test translating landmarks relative to wrist
class TestTranslateToRelativeToWrist():

    WRIST_LANDMARK_START = 0
    WRIST_LANDMARK_END = 3
    TOTAL_COORDINATES = 63

    #self.translate | self.valid_landmark
    def setup_method(self):
        """
        This runs automatically before every individual test.
        It ensures a fresh instance of Normalizer is always available.
        """
        self.translate = Normalizer() #Initialize the self.translate_relative
        #Produces random valid float for valid landmark
        self.valid_landmark = [(float(i), float(i + 1), float(i + 2)) for i in range(21)]
    
    #Test if the relative landmarks data type is a list
    def test_relative_landmark_data_type(self):
        
        #Store if the raw landmark into sample_coordinates list
        sample_coordinates = self.translate.trans_relative_wrist(self.valid_landmark)
        #Check if the relative landmark is a list
        assert isinstance(sample_coordinates, list), \
        "Relative Coordinates is not a list."

    #Test if the relative landmarks is empty
    def test_relative_landmark_is_empty(self):
        
        sample_landmark = self.translate.trans_relative_wrist(self.valid_landmark)
        #Check if it's empty
        assert len(sample_landmark) > 0,\
        "The Relative landmarks must not be empty"

    #Test if the wrist is the NEW origin
    def test_wrist_new_origin(self):

        #Store in the relative_to_wrist the translated landmark from self.valid_landmark
        relative_to_wrist = self.translate.trans_relative_wrist(self.valid_landmark)

        #Check if the wrist Landmark (0-3) are all ZERO (new origin)
        assert all(i == 0 for i in relative_to_wrist[self.WRIST_LANDMARK_START:self.WRIST_LANDMARK_END]), \
        "Error: Wrist landmark base values are not properly zeroed out."

    #Test if all landmark are relative to wrist
    def test_landmark_relative_to_wrist(self):

        #Sample data !(ASSUME VALID RAW LANDMARKS)!
        sample_landmark = [ (5.0, 10.0, 20.0),   
                             (7.0, 15.0, 22.0),
                             (3.0, 12.0, 18.0),]
        sample_expected_landmark =  [0.0, 0.0, 0.0, 2.0, 5.0, 2.0, -2.0, 2.0, -2.0]
                                    
        #Store the sample relative coordinates
        sample_relative_coord = self.translate.trans_relative_wrist(sample_landmark)
        #Checks if the landmarks are properly subtract by WRIST (ORIGIN)
        assert sample_relative_coord == sample_expected_landmark, \
            "Relative coordinates do not match the expected values."

    #Test if the total coordinates are 63    
    def test_landmark_relative_have_63_coord(self):

        sample_landmark = self.translate.trans_relative_wrist(self.valid_landmark)
        #Check if the relative landmarks are 63 coordinates
        assert len(sample_landmark) == self.TOTAL_COORDINATES,\
        "The total coordinates must be 63"

class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

#Mock object missing required attributes (used to test validation)
class MockInvalidLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        #no z attribute

#Test for LandmarkProcessor method
class TestNormalizerLandmarkProcessor():

    #normalizer | self.valid_raw_landmarks
    def setup_method(self):

        """
        This runs automatically before every individual test.
        It ensures a fresh instance of Normalizer is always available.
        """

        self.normalizer = Normalizer()
        
        #Reusable valid mock data: List[List[NormalizedLandmark]]
        self.valid_raw_landmarks = [[MockLandmark(float(i), float(i + 1), float(i + 2)) for i in range(21)]]
        self.empty_outer_list = []
        self.empty_inner_list = [[]]
        self.invalid_missing_attr = [[MockInvalidLandmark(1.0, 2.0)]]

    #Test if raw_landmarks outer list is empty
    def test_landmark_processor_empty_outer_list(self):
        with pytest.raises(ValueError, match="raw_landmarks must not empty"):
            self.normalizer.LandmarkProcessor(self.empty_outer_list)

    #Test if raw_landmarks inner list is empty
    def test_landmark_processor_empty_inner_list(self):
        with pytest.raises(ValueError, match="raw_landmarks must not empty"):
            self.normalizer.LandmarkProcessor(self.empty_inner_list)

    #Test if landmark is missing x, y, or z attribute
    def test_landmark_processor_missing_attribute(self):
        with pytest.raises(ValueError, match="Each landmark must have x, y, and z attributes"):
            self.normalizer.LandmarkProcessor(self.invalid_missing_attr)

    #Test if the output is a list
    def test_landmark_processor_output_is_list(self):
        result = self.normalizer.LandmarkProcessor(self.valid_raw_landmarks)
        assert isinstance(result, list), \
        "LandmarkProcessor output is not a list."

    #Test if each converted landmark is a list of 3 coordinates
    def test_landmark_processor_output_structure(self):
        result = self.normalizer.LandmarkProcessor(self.valid_raw_landmarks)
        assert all(isinstance(point, list) and len(point) == 3 for point in result), \
        "Each converted landmark must be a list of 3 coordinates."

    #Test if the coordinate values are correctly extracted (x, y, z)
    def test_landmark_processor_correct_values(self):
        sample_raw = [[MockLandmark(1.0, 2.0, 3.0), MockLandmark(4.0, 5.0, 6.0)]]
        expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = self.normalizer.LandmarkProcessor(sample_raw)
        assert result == expected, \
        "Converted coordinates do not match expected values."

    #Test if the output length matches the input length
    def test_landmark_processor_output_length(self):
        result = self.normalizer.LandmarkProcessor(self.valid_raw_landmarks)
        assert len(result) == 21, \
        "Converted landmark list must contain 21 landmarks."