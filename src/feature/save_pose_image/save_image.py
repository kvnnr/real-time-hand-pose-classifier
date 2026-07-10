from pathlib import Path
from typing import Union
import cv2 as cv
import numpy as np

class SavePoseImage:
    """
    What:
        Persists a captured camera frame as an image file
        inside the dataset directory for a specified pose label.

    Input:
        folder_path: Path
            Directory path of the pose folder.

        frame: np.ndarray
            Raw BGR image captured from OpenCV.

        label: str
            Name of the pose class used to determine the dataset folder.

        counter: int
            Unique image index provided by the counter manager.
            Used to prevent filename collisions.

    Process:
        - Construct the dataset image path:
            data/raw/<label>/<label>_<counter>.jpeg

        - Validate:
            - Image path type.
            - Frame type.
            - Frame is not empty.
            - Frame dimensions are valid.

        - Encode and save the image using OpenCV.

    Output:
        None

    Side Effects:
        - Creates a new image file on disk:
            data/raw/<label>/<label>_<counter>.jpeg

    Failure Conditions:
        - Invalid image path.
        - Invalid frame type.
        - Empty frame.
        - Unsupported image format.
        - Image writing failure.

    Invariants:
        - Existing images are not overwritten if a unique counter is provided.
        - Input frame is not modified.
        - Saved image belongs to the specified pose label.
    """

    def validate(self,image_path: Path, frame: np.ndarray) -> Union[np.ndarray, str]:

        # Validate path
        if not isinstance(image_path, Path):
            return "Image path must be a pathlib.Path object."

        # Validate frame type
        if not isinstance(frame, np.ndarray):
            return "Frame must be a numpy array."

        # Validate empty frame
        if frame.size == 0:
            return "Frame is empty."

        # Validate dimensions
        if len(frame.shape) != 3:
            return "Frame must have height, width, and channels."

        return frame
    
    
    #Find the folder base on label and save the frame (current pose).
    """
    Why return counter(int)?:
        To avoid images overwritten every new instances of the program/system.
    """
    def save_image(self, folder_path: Path, bgr_frame: np.ndarray, label: str, counter: int) -> None:
        
        #Find the dataset folder path.
        #Create image path. f"{label}_{counter}.jpeg" → Ex: fist_0001.jpeg
        self.image_path = folder_path / f"{label}_{counter:05d}.jpeg"

        #Validation of frame and path of frame.
        validation = self.validate(self.image_path, bgr_frame)

        # Validation failed
        if isinstance(validation, str):
            raise ValueError(validation)

        #Save the current frame to folder base on label.
        success = cv.imwrite(str(self.image_path), bgr_frame)

        if not success:
            raise IOError("Failed to save image.")
        print(f"{label} pose was successfully saved: {self.image_path}")
        
