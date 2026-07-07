import numpy as np
import cv2 as cv

#Create and Update the text box.
class TextBox:

    """
    TextBox Contract

    What:
        A simple OpenCV-based utility that renders a text box overlay on a video frame.

    Inputs:
        frame (np.ndarray)
            Input image frame in BGR format from OpenCV video stream.

        text (str)
            String content to be displayed inside the text box.

    Process:
        Receive input frame from camera or video pipeline
        Draw a rectangular box on a fixed region of the frame
        Render the provided text inside the rectangle using OpenCV putText
        Overlay both shapes directly onto the original frame
        Return updated frame with visual annotation

    Output:
        np.ndarray
            Modified frame containing the rendered text box and text overlay.
    """
    
    #Initialized the OpenCameraVision object.
    def __init__(self) -> None:
        self.cv = cv
    
    #Draw Temporary Text box.
    def create_text_box(self, frame: np.ndarray, text: str) -> np.ndarray:
    
        # Draw textbox
        textbox = self.cv.rectangle(frame, (100, 410), (500, 450), (255,255,255), 2)
        
        #Draw current text.
        result = self.cv.putText(textbox, text, (110,440),cv.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)

        #Return the frame with updated text box.
        return result