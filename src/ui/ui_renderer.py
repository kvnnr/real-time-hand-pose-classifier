import numpy as np
from src.ui.prediction_panel import PredictionPanel
from src.ui.confidence_bar import ConfidenceBar
from src.ui.fps_panel import FPSPanel
from src.ui.status_panel import StatusPanel
from src.ui.transparent_panel import TransparentPanel
from src.ui.text_styles import TextStyles
from src.schemas.ui_schema import UiState

import cv2 as cv


class UIRenderer:

    """
    UIRenderer Contract

    What:

        Responsible for rendering the
        complete application user interface.

    Why:

        Centralizes every visual element
        into one rendering class.

        Individual UI components can later
        be delegated to their own modules.

    Input:

        frame

            Camera frame.

        pose_name

            Current pose label.

    Process:

        Draw background panel.

        Draw application title.

        Draw current pose.

        Draw keyboard controls.

    Output:

        Updated camera frame.

    Failure Conditions:

        Invalid frame.

    Invariants:

        Frame resolution remains unchanged.
    """

    def __init__(self):

        #Objects creations.
        self.panel = TransparentPanel()
        self.prediction = PredictionPanel()
        self.confidence = ConfidenceBar()
        self.fps = FPSPanel()
        self.status = StatusPanel()

    def _validation_of_frame(self, frame: np.ndarray) -> None:

        if frame is None:
            raise ValueError("Frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise TypeError("Frame must be numpy ndarray.")
        
    def draw(self, frame: np.ndarray, ui_state: UiState ) -> np.ndarray:
        
        self._validation_of_frame(frame)


        self.panel.draw(
            frame,
            (15,15),
            (520,170)
        )


        cv.putText(
            frame,
            "Hand Pose Recognition System",
            (30,40),
            TextStyles.TITLE_FONT,
            TextStyles.TITLE_SCALE,
            TextStyles.WHITE,
            TextStyles.TITLE_THICKNESS
        )


        cv.putText(
            frame,
            f"Current Pose : {ui_state.pose_label}",
            (30,75),
            TextStyles.TEXT_FONT,
            TextStyles.NORMAL_SCALE,
            TextStyles.GREEN,
            2
        )


        self.prediction.draw(
            frame,
            ui_state.prediction,
            ui_state.confidence
        )


        self.confidence.draw(
            frame,
            ui_state.confidence
        )


        self.fps.draw(
            frame,
            ui_state.fps
        )


        self.status.draw(
            frame,
            ui_state.hand_detected,
            ui_state.model_loaded
        )

        cv.putText(
            frame,
            "[ENTER] Create Pose Folder",
            (30,110),
            TextStyles.TEXT_FONT,
            TextStyles.NORMAL_SCALE,
            TextStyles.WHITE,
            1
        )

        cv.putText(
            frame,
            "[TAB] Save Sample",
            (30,140),
            TextStyles.TEXT_FONT,
            TextStyles.NORMAL_SCALE,
            TextStyles.WHITE,
            1
        )

        cv.putText(
            frame,
            "[=] Train Model",
            (220,140),
            TextStyles.TEXT_FONT,
            TextStyles.NORMAL_SCALE,
            TextStyles.WHITE,
            1
        )

        cv.putText(
            frame,
            "[ESC] Exit",
            (380,140),
            TextStyles.TEXT_FONT,
            TextStyles.NORMAL_SCALE,
            TextStyles.WHITE,
            1
        )

        return frame