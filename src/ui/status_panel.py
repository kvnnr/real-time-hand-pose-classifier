import cv2 as cv
import numpy as np

from src.ui.text_styles import TextStyles

class StatusPanel:

    """
    StatusPanel Contract

    What:
        Draw application status.

    Why:
        Displays system information:
            - Hand detection state
            - Model state

    Input:
        frame:
            Camera frame.

        hand_detected:
            Whether hand landmarks exist.

        model_loaded:
            Whether the ML model is ready.

    Output:
        Updated frame.
    """

    def draw(self, frame: np.ndarray, hand_detected: bool, model_loaded: bool) -> np.ndarray:

        hand = "Detected" if hand_detected else "Not Detected"
        model = "Loaded" if model_loaded else "Not Loaded"

        # Move status to left side.
        x_position = 30
        y_position = 350

        cv.putText(
            frame,
            f"Hand : {hand}",
            (x_position, y_position),
            TextStyles.TEXT_FONT,
            0.55,
            TextStyles.WHITE,
            1
        )

        cv.putText(
            frame,
            f"Model : {model}",
            (x_position, y_position + 25),
            TextStyles.TEXT_FONT,
            0.55,
            TextStyles.WHITE,
            1
        )

        return frame