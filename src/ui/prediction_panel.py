import cv2 as cv
import numpy as np

from src.ui.text_styles import TextStyles


class PredictionPanel:

    """
    PredictionPanel Contract

    What:
        Draws the predicted hand pose and model confidence
        on the user interface.

    Why:
        The prediction is the most important information
        presented to the user.

        This panel highlights the current model output
        using large readable text and color-coded feedback.

    Input:

        frame:
            OpenCV camera frame.

        prediction:
            Predicted pose label.

        confidence:
            Model confidence.

            Range:
                0.0 - 1.0

    Process:

        Draw prediction title.

        Draw prediction label.

        Draw confidence percentage.

    Output:

        Updated camera frame.

    Failure Conditions:

        Invalid frame.

    Invariants:

        Frame resolution is preserved.
    """

    def draw(self, frame: np.ndarray, prediction: str, confidence: float) -> np.ndarray:

        if confidence >= 0.90:
            color = TextStyles.GREEN

        elif confidence >= 0.70:
            color = TextStyles.YELLOW

        else:
            color = TextStyles.RED

        cv.putText(
            frame,
            "Prediction",
            (30,200),
            TextStyles.TEXT_FONT,
            0.65,
            TextStyles.WHITE,
            2
        )

        cv.putText(
            frame,
            prediction,
            (30,240),
            TextStyles.TITLE_FONT,
            1.1,
            color,
            3
        )

        cv.putText(
            frame,
            f"Confidence : {confidence*100:.2f}%",
            (30,270),
            TextStyles.TEXT_FONT,
            0.55,
            TextStyles.WHITE,
            1
        )

        return frame