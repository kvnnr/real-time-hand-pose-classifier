import cv2 as cv
import numpy as np


class ConfidenceBar:

    """
    Confidence Bar Contract

    What:
        Draw a visual confidence bar.

    Why:
        Understand progress bars much
        faster than numerical percentages.

    Input:
        frame: np.ndarray
        confidence: float

    Process:
        Draw outline.
        Fill rectangle according to confidence.

    Output:

        Updated frame.
    """

    BAR_WIDTH = 250

    BAR_HEIGHT = 18

    def draw(self, frame: np.ndarray, confidence: float) -> np.ndarray:

        x = 30
        y = 285

        filled = int(self.BAR_WIDTH * confidence)

        cv.rectangle(
            frame,
            (x,y),
            (x+self.BAR_WIDTH,y+self.BAR_HEIGHT),
            (255,255,255),
            1
        )

        cv.rectangle(
            frame,
            (x,y),
            (x+filled,y+self.BAR_HEIGHT),
            (0,255,0),
            -1
        )

        return frame