from src.feature.normalizer import Normalizer
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
        self.valid_landmarks= [(float(i), float(i), float(i)) for i in range(self.VALID_LANDMARKS)]
        self.invalid_landmark_amount = [(float(i), float(i), float(i)) for i in range(10)]
        self.invalid_coord_data_type = [(1.2, 3.5, 'Keven')] * self.VALID_LANDMARKS
        self.invalid_coord_amount = [(float(i), float(i)) for i in range(self.VALID_LANDMARKS)]
        self.invalid_data_type = "Keven"

    #Test if Landmark is a valid Data stype
    def test_validate_not_list(self):
        with pytest.raises(TypeError, match= f"Expected a list."):
            self.normalizer.validate(self.invalid_data_type)
    
    #Test if Landmark is empty
    def test_validate_empty_landmark(self):
        with pytest.raises(ValueError, match="Landmarks can't be empty."):
            self.normalizer.validate([])

    #Test if List have 21 Landmarks
    def test_validate_requires_21_landmarks(self):
        with pytest.raises(ValueError, match= (f"Expected 21 tuples of landmarks.")):
            self.normalizer.validate(self.invalid_landmark_amount)
    
    #Test if landmark have 3 coordinates
    def test_validate_landmark_requires_3_coord(self):
        with pytest.raises(ValueError, match ='Expected landmark must be a tuple of 3 coordinates.'):
            self.normalizer.validate(self.invalid_coord_amount)
    
    #Test if Coordiates are Valid Data type
    def test_validate_coordinate_data_type(self):
        with pytest.raises(TypeError, match = "Coordinates must be integer or float"):
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
