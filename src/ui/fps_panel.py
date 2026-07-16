import cv2 as cv
import numpy as np

from src.ui.text_styles import TextStyles


class FPSPanel:

    """
    FPSPanel Contract

    What:

        Display current camera FPS.

    Why:

        Allow monitoring of application
        performance in real time.

    Input:

        frame

        fps

    Output:

        Updated frame.
    """

    def draw(self, frame: np.ndarray, fps: float) -> np.ndarray:

        cv.putText(
            frame,
            f"FPS : {fps:.2f}",
            (530,35),
            TextStyles.TEXT_FONT,
            0.6,
            TextStyles.GREEN,
            2
        )

        return frame