import cv2 as cv
import numpy as np


class TransparentPanel:

    """
    TransparentPanel Contract

    What:

        Draws a semi-transparent panel
        on top of an OpenCV frame.

    Why:

        A transparent background makes the
        user interface easier to read while
        still allowing the camera feed to
        remain visible.

    Input:

        frame

            Type:
                np.ndarray

            Camera frame.

        top_left

            Type:
                tuple[int, int]

            Starting coordinate.

        bottom_right

            Type:
                tuple[int, int]

            Ending coordinate.

        opacity

            Type:
                float

            Range:
                0.0 - 1.0

    Process:

        1. Copy the frame.

        2. Draw a filled rectangle.

        3. Blend rectangle with
           original frame.

    Output:

        Updated frame.

    Failure Conditions:

        Invalid frame.

        Invalid opacity.

    Invariants:

        Original frame resolution
        is preserved.
    """

    PANEL_COLOR = (40,40,40)

    def draw(self,frame: np.ndarray, top_left: tuple[int,int], bottom_right: tuple[int,int], opacity: float = 0.60) -> np.ndarray:

        if frame is None:
            raise ValueError("Frame cannot be None.")

        if not isinstance(frame,np.ndarray):
            raise TypeError("Frame must be numpy ndarray.")

        if opacity < 0 or opacity > 1:
            raise ValueError("Opacity must be between 0 and 1.")

        overlay = frame.copy()

        cv.rectangle(
            overlay,
            top_left,
            bottom_right,
            self.PANEL_COLOR,
            -1
        )

        cv.addWeighted(
            overlay,
            opacity,
            frame,
            1-opacity,
            0,
            frame
        )

        return frame