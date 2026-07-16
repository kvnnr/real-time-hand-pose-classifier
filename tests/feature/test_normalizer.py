import math
import pytest

from types import SimpleNamespace

from src.feature.landmark_normalizer.normalizer import Normalizer

#-----------------------------------------------------------------------------
#Helper: builds a fake NormalizedLandmark object with x, y, z attributes.
#-----------------------------------------------------------------------------
def make_landmark(x: float, y: float, z: float) -> SimpleNamespace:

    """
    Variables:
        x, y, z: coordinate values for the fake landmark.
    """

    return SimpleNamespace(x=x, y=y, z=z)


#-----------------------------------------------------------------------------
#Helper: builds 21 fake landmarks wrapped in the MediaPipe outer-list shape.
#-----------------------------------------------------------------------------
def make_raw_landmarks() -> list[list[SimpleNamespace]]:

    """
    Variables:
        hand: List[SimpleNamespace] of 21 landmarks, wrist at index 0
              and middle finger base at index 9.
    """

    hand = [make_landmark(0.0, 0.0, 0.0)] #Index 0: wrist.

    for i in range(1, 21):
        hand.append(make_landmark(float(i), float(i) + 1, float(i) + 2))

    return [hand]


class TestNormalizerLandmarkProcessor:

    def setup_method(self):

        """
        Variables:
            self.normalizer: Normalizer instance
        """

        self.normalizer = Normalizer()

    #Test None raw_landmarks returns None case.
    def test_landmark_processor_none_returns_none(self):

        result = self.normalizer.landmark_processor(None)

        assert result is None

    #Test empty outer list returns None case.
    def test_landmark_processor_empty_outer_list(self):

        result = self.normalizer.landmark_processor([])

        assert result is None

    #Test empty inner list returns None case.
    def test_landmark_processor_empty_inner_list(self):

        result = self.normalizer.landmark_processor([[]])

        assert result is None

    #Test valid raw landmarks are converted to float coordinates case.
    def test_landmark_processor_valid_converts_coordinates(self):

        raw_landmarks = [[make_landmark(1.0, 2.0, 3.0), make_landmark(4.0, 5.0, 6.0)]]

        result = self.normalizer.landmark_processor(raw_landmarks)

        assert result == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    #Test landmark missing x/y/z attributes raises ValueError case.
    def test_landmark_processor_missing_attributes_raises_value_error(self):

        raw_landmarks = [[SimpleNamespace(x=1.0, y=2.0)]] #Missing z.

        with pytest.raises(ValueError):
            self.normalizer.landmark_processor(raw_landmarks)


class TestNormalizerValidate:

    def setup_method(self):

        """
        Variables:
            self.normalizer: Normalizer instance
        """

        self.normalizer = Normalizer()

    #Test valid 21-landmark list returns the same landmarks case.
    def test_validate_valid_landmarks_returns_landmarks(self):

        landmark = [[float(i), float(i), float(i)] for i in range(21)]

        result = self.normalizer.validate(landmark)

        assert result == landmark

    #Test non-list landmark raises TypeError case.
    def test_validate_not_list_raises_type_error(self):

        with pytest.raises(TypeError):
            self.normalizer.validate("not a list")

    #Test empty landmark list raises ValueError case.
    def test_validate_empty_raises_value_error(self):

        with pytest.raises(ValueError):
            self.normalizer.validate([])

    #Test wrong number of landmarks raises ValueError case.
    def test_validate_wrong_count_raises_value_error(self):

        landmark = [[0.0, 0.0, 0.0] for _ in range(20)]

        with pytest.raises(ValueError):
            self.normalizer.validate(landmark)

    #Test landmark point with wrong coordinate count raises ValueError case.
    def test_validate_wrong_point_length_raises_value_error(self):

        landmark = [[0.0, 0.0, 0.0] for _ in range(20)] + [[0.0, 0.0]]

        with pytest.raises(ValueError):
            self.normalizer.validate(landmark)

    #Test non-float coordinate raises TypeError case.
    def test_validate_non_float_coordinate_raises_type_error(self):

        landmark = [[0.0, 0.0, 0.0] for _ in range(20)] + [[0, 0.0, 0.0]] #int, not float.

        with pytest.raises(TypeError):
            self.normalizer.validate(landmark)


class TestNormalizerTransRelativeWrist:

    def setup_method(self):

        """
        Variables:
            self.normalizer: Normalizer instance
        """

        self.normalizer = Normalizer()

    #Test translation subtracts wrist coordinates from every landmark case.
    def test_trans_relative_wrist_translates_around_wrist(self):

        processed_landmark = [
            [1.0, 2.0, 3.0], #Wrist.
            [4.0, 6.0, 9.0],
        ]

        result = self.normalizer.trans_relative_wrist(processed_landmark)

        assert result == [0.0, 0.0, 0.0, 3.0, 4.0, 6.0]

    #Test wrist landmark itself becomes the zero vector case.
    def test_trans_relative_wrist_wrist_becomes_origin(self):

        processed_landmark = [[5.0, 5.0, 5.0]]

        result = self.normalizer.trans_relative_wrist(processed_landmark)

        assert result == [0.0, 0.0, 0.0]


class TestNormalizerTransScaleInvariance:

    def setup_method(self):

        """
        Variables:
            self.normalizer: Normalizer instance
        """

        self.normalizer = Normalizer()

    #Test coordinates are divided by the wrist-to-middle-finger-base distance case.
    def test_trans_scale_invariance_scales_by_middle_finger_distance(self):

        relative_landmark = [0.0] * 27 + [3.0, 4.0, 0.0] + [0.0] * 33 #Index 9 (27:30) = (3, 4, 0) -> distance 5.

        result = self.normalizer.trans_scale_invariance(relative_landmark)

        assert result[27:30] == pytest.approx([0.6, 0.8, 0.0])

    #Test zero scaling factor raises ValueError case.
    def test_trans_scale_invariance_zero_scaling_factor_raises_value_error(self):

        relative_landmark = [0.0] * 63 #Middle finger base at wrist -> distance 0.

        with pytest.raises(ValueError):
            self.normalizer.trans_scale_invariance(relative_landmark)


class TestNormalizerTransform:

    def setup_method(self):

        """
        Variables:
            self.normalizer: Normalizer instance
        """

        self.normalizer = Normalizer()

    #Test None raw_landmark returns None case.
    def test_transform_none_returns_none(self):

        result = self.normalizer.transform(None)

        assert result is None

    #Test empty raw_landmark returns None case.
    def test_transform_empty_returns_none(self):

        result = self.normalizer.transform([])

        assert result is None

    #Test valid raw landmarks produce a 63-value normalized vector case.
    def test_transform_valid_returns_normalized_vector(self):

        raw_landmarks = make_raw_landmarks()

        result = self.normalizer.transform(raw_landmarks)

        assert isinstance(result, list)
        assert len(result) == 63
        assert result[0] == 0.0 and result[1] == 0.0 and result[2] == 0.0 #Wrist stays at origin.
        assert all(math.isfinite(value) for value in result)