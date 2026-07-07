from typing import List
import math as m

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

class Normalizer:
    
    """
    Convert landmarks relative to wrist

    Input:
        List[List[NormalizedLandmark]] | UnNormalized 21 Hand landmarks | X Y Z

    Process:
        Convert MediaPipe's NormalizedLandmark objects -> List[List[float].
        Translate landmarks relative to wrist  
        Normalize hand size for scale invariance

    Output:
        List of 63 floats (Hand landmarks) | List[List[float]]
    """
    
    WRIST_INDEX = 0 #Index of the wrist.
    MIDDLE_FINGER_BASE_START = 27
    MIDDLE_FINGER_BASE_END = 30

    #Convert MediaPipe's NormalizedLandmark objects -> List[List[float].| Return: converted_landmarks
    def LandmarkProcessor(self, raw_landmarks: List[List[NormalizedLandmark]]) -> List[List[float]]:

        #Validate if landmark empty
        if not raw_landmarks or not raw_landmarks[0]:
            raise ValueError("raw_landmarks must not empty")

        #Validate each landmark coordinates.
        for lm in raw_landmarks[0]:
            if not all(hasattr(lm, attr) for attr in ("x", "y", "z")):
                raise ValueError("Each landmark must have x, y, and z attributes")

        # Convertion
        converted_landmarks = [[lm.x, lm.y, lm.z] for lm in raw_landmarks[0]]
        return converted_landmarks
    
    #Validation landmark if True, have 21 list, 3 coordinate each List and if it's a list. | Return: validated_landmark
    def validate(self, landmark: List[List[float]]) -> List[List[float]]:

        if not isinstance(landmark, list):
            raise TypeError(f"Expected a list, but got {type(landmark).__name__}.")
        
        if not landmark:
            raise ValueError("Landmarks can't be empty.")

        if not len(landmark) == 21:
            raise ValueError(f"Expected 21 landmarks, but got {len(landmark)}.")
        
        if not all(isinstance(point, list) and len(point) == 3 for point in landmark):
            raise ValueError('Expected each landmark to be a list of 3 coordinates.')

        if not all(isinstance(coord, float) for point in landmark for coord in point):
            raise TypeError("Coordinates must be float")
  
        return landmark
    
    #Translate landmarks relative to wrist. | Return: relative_landmarks
    def trans_relative_wrist(self, processed_landmark: List[List[float]]) -> List[float]:
        """
        Why:
            To make the wrist as point of origin to avoid in accurate reading of data.
        """
        x_wrist, y_wrist, z_wrist = processed_landmark[self.WRIST_INDEX] #Get the coordinates of WRIST.
        relative_coordinates = [] #Container of Coordinate relative to wrist.

        for point in processed_landmark: #landmark: [[2.0,5.0,6.9],[8.4,7.5,6.6], ......] -> point: [2.0, 5.0, 6.9]

            x_ref, y_ref, z_ref = point #Get the 3 coordinate in each landmark (0-21).
            #Get the relative of each coordinate relative to wrist.
            #Store the relative coordinates.
            relative_coordinates.extend((x_ref - x_wrist, y_ref - y_wrist, z_ref - z_wrist)) 

        return relative_coordinates
    
    #Normalize hand size for scale invariance. | Return: scaled_coordinates
    def trans_scale_invariance(self, relative_landmark: List[float]) -> List[float]:

        """Formula used:
        D = √(x^2 + y^2 + z^2)

        Why:
            To get a scalling factor to fix the scale invariances of the data since hand can be far away and give small
            values which confuse the classifer or to big.
        """

        scaled_coordinates = []
        #Get the Landmark 9 (Base of middle finger) for scale factor
        x_mfinger, y_mfinger, z_mfinger = relative_landmark[self.MIDDLE_FINGER_BASE_START:self.MIDDLE_FINGER_BASE_END]

        #Get scaling factor from WRIST to BASE MIDDLE FINGER
        scaling_factor = m.sqrt((x_mfinger **2) + (y_mfinger **2) + (z_mfinger **2))

        #Edge Case if scaling_factor is ZERO (0)
        if scaling_factor == 0:
            raise ValueError('Scaling factor MUST NOT be zero.')
        
        #Iterate to landmark List[float] & Fix scale invariance of all coordinates.
        for coord in relative_landmark:
            new_coord = coord / scaling_factor
            scaled_coordinates.append(new_coord) # Store scaled factors.

        return scaled_coordinates

     #Translate the Raw landmarks to relative and scaled landmarks. | Return: final_landmarks  

    #Main Function. | Return: final_landmarks
    def transform(self, raw_landmark: List[List[NormalizedLandmark]]) -> List[float]:
        
        #Convert the Landmark.
        converted = self.LandmarkProcessor(raw_landmark)

        #Validation the landmark
        validated_landmarks = self.validate(converted)

        relative_landmarks = self.trans_relative_wrist(validated_landmarks) #Landmarks -> Relative Landmarks
        final_landmarks = self.trans_scale_invariance(relative_landmarks) #Relative Landmarks -> Scaled landmarks

        return final_landmarks
    
