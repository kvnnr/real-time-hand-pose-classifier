from pathlib import Path
import json
import numpy as np

from cv2.typing import MatLike
import cv2 as cv

#Main Camera Class
class Camera:

    """
    Captures frames from webcam
    
    Inputs:
        None (Reads directly from the hardware camera index)
        
    Process:
        Grabs video frame of live video
        Convert frames into Numpy matrix | Frames → Numpy
        Numpy (BGR) → Numpy RGB

    Return: 
        Numpy Matrix of frames (RGB)
    """

    #self.config{" "} | self.cap.
    def __init__(self) -> None:
        self.cap: cv.VideoCapture | None = None #Act as a bridge from webcam to code.
        self.config =  {} #Variable for config file.

        config_path = Path(__file__).parent.parent.parent / "config.json"

        #Check if the config.json is valid.
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        except FileNotFoundError: #Check if config.json is found.
            print('\nError: config.json was not found. Falling back to default camera index = 0.')
            self.config = {"camera_index": 0}

        except json.JSONDecodeError:#Check if config.json is corrupted.
            print("Error: 'config.json is corrupted. Failling back to default camera index = 0.")
            self.config = {"camera_index": 0} 
    
    #Return self.cap.
    def open_camera(self) -> cv.VideoCapture:

        """Opens the camera connection with error handling."""

        #Variable for camera index
        camera_indx = self.config["camera_index"]

        #Bridge the camera and code.
        self.cap = cv.VideoCapture(camera_indx)

        #Check if camera is opened 
        if not self.cap.isOpened():
            raise RuntimeError(f"Error: Could not open the camera stream at index {camera_indx}.")
        print("Camera successfully initialized.")
        
        return self.cap

    #Grab the frames from video | Return: NDarray (frame).
    def get_frames(self) -> MatLike:

        """Grab the frames from the video with error handling"""
        
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError('Error: Camera has not been initialized.')
        
        #Store the state of camera and frame.
        ret, frame = self.cap.read()

        #Check if it receive any frame.
        if not ret:
            raise RuntimeError("No frame detected. Exiting loop or handling stream end...")
        
        return frame

    #Convert frame from BGR -> RGB | Return: rgb_frames.
    def bgr_to_rgb(self, frames: MatLike) -> MatLike:
        """
        Convert frames from BGR -> RGB
        """
        rgb_frames = cv.cvtColor(frames, cv.COLOR_BGR2RGB)

        return rgb_frames
    
    #Release the camera and destry all windows
    def release_camera(self) -> None:
        
        #Check if camera is On and turn it off
        if self.cap is not None:
            self.cap.release()
            self.cap = None

#Frame control Operations
class FrameViewer:
   
    """
    All Frame operations
    """
    #Initialize the Name of Window.
    def __init__(self, window_name = "Live Stream") -> None:
        self.window_name = window_name

    #Show the Camera current frame/Window.
    def show(self, frame: np.ndarray) -> None:
        return cv.imshow(self.window_name, frame)
    
    #Initialazation of key operation.
    def operation_key(self) -> int:
        return cv.waitKey(1) & 0xFF

    #Closes all opened window.
    def close(self) -> None:
        cv.destroyAllWindows()



