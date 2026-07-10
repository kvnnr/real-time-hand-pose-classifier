import numpy as np
import cv2 as cv

#Frame control Operations
class FrameOps:
   
    """

    All Frame operations:

    →Update current frame.
    →Get key pressed by user.
    →Close OpenCV resources

    """

    #Initialize the Name of Window.
    def __init__(self, window_name = "Hand Pose Recognition") -> None:
        self.window_name = window_name

    #Show the Camera current frame/Window.
    def update_frame(self, frame: np.ndarray) -> None:
        cv.imshow(self.window_name, frame)
    
    #Initialazation of key operation.
    def get_key(self) -> int:
        return cv.waitKey(1) & 0xFF

    #Closes all opened window.
    def close_windows(self) -> None:
        cv.destroyAllWindows()