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
        Draw the application information panel
        Draw the current pose label
        Draw the available keyboard controls
        Return the updated frame

    Output:
        np.ndarray
            Modified frame containing the rendered user interface.
    """

    """
    Visualization:
    +----------------------------------------------------------------------------------+
    | ┌──────────────────────────────────────────────────────────────────────────────┐ |
    | │ Real-Time Hand Pose Dataset Collector                                        │ |
    | │                                                                              │ |
    | │ Current Pose : Open Palm                                                     │ |
    | │                                                                              │ |
    | │ [ENTER] Create Pose Folder                                                   │ |
    | │ [TAB] Save Sample    [ESC] Exit    [=] Train Model                                             │ |
    | └──────────────────────────────────────────────────────────────────────────────┘ |
    |                                                                                  |
    |                                                                                  |
    |                            ( Webcam Live Feed )                                  |
    |                                                                                  |
    |                           ✋  User's Hand Here                                   |
    |                                                                                  |
    |                                                                                  |
    |                                                                                  |
    |                                                                                  |
    |                                                                                  |
    |                                                                                  |
    +----------------------------------------------------------------------------------+

    """

    #Initialized the OpenCameraVision object.
    def __init__(self) -> None:
        self.cv = cv

    #Draw Temporary Text box.
    def create_text_box(self, frame: np.ndarray, text: str) -> np.ndarray:

        #Main information box.
        self.cv.rectangle(
            frame,
            (15, 15),
            (450, 150),
            (255, 255, 255),
            2
        )

        #Application title.
        self.cv.putText(
            frame,
            "Real-Time Hand Pose Dataset Collector",
            (25, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        #Current pose label.
        self.cv.putText(
            frame,
            f"Current Pose : {text}",
            (25, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        #Instructions.
        self.cv.putText(
            frame,
            "[ENTER] Create Pose Folder",
            (25, 100),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        self.cv.putText(frame,
            "[TAB] Save Sample    [ESC] Exit    [=] Train Model",
            (25, 125),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        #Return the updated frame.
        return frame