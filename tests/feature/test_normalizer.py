from src.feature.normalizer import Normalizer
import pytest


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
    