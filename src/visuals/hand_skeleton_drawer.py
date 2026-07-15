import cv2
import numpy as np

from mediapipe.tasks.python.components.containers.landmark import (NormalizedLandmark)

class HandSkeletonDrawer:

    """
    HandSkeletonDrawer Contract

    What:
        Draws detected hand skeleton connections and landmarks
        on the camera frame.

    Why:
        Provide visual feedback of detected hand landmarks.
        Used for debugging, visualization, and verifying that
        MediaPipe hand detection is working correctly.

    Input:
        frame:
            OpenCV BGR camera frame.

            Type:
                np.ndarray

        landmarks:
            Raw MediaPipe hand landmarks.

            Type:
                list[list[NormalizedLandmark]]

            Description:
                Contains one or more detected hands.
                Each hand contains exactly 21 landmarks.

        connections:
            MediaPipe hand skeleton topology.

            Type:
                list[Connection]

            Description:
                Defines which landmark points should be
                connected together.

    Process:
        1. Validate if hand landmarks exist.
        2. Convert normalized landmark coordinates into
           pixel coordinates based on frame dimensions.
        3. Iterate through each detected hand.
        4. Draw skeleton connections between landmarks.
        5. Return the frame with skeleton overlay.

    Output:
        np.ndarray:
            Camera frame containing hand skeleton visualization.

    Failure Conditions:
        - No landmarks detected:
            Returns the original frame.

        - Invalid frame:
            Raises ValueError.

        - Invalid landmark structure:
            Raises TypeError.

    Invariants:
        - Original frame resolution is preserved.
        - Landmark indexing follows MediaPipe hand topology.
        - Visualization does not modify landmark data.
    """

    # Skeleton drawing configuration.
    LINE_COLOR = (0, 255, 0)
    LINE_THICKNESS = 2

    def __init__(self,connections: list) -> None:

        """
        Initialize HandSkeletonDrawer.

        Input:
            connections:
                MediaPipe hand landmark connections.

        Process:
            Store skeleton topology used for drawing lines.

        Output:
            None
        """

        self.connections = connections

    def draw(self,frame: np.ndarray,landmarks: list[list[NormalizedLandmark]] | None) -> np.ndarray:

        """
        Draw hand skeleton overlay on camera frame.

        Input:
            frame:
                BGR image from OpenCV.

            landmarks:
                Raw MediaPipe detected landmarks.

                Format:
                    [
                        [
                            landmark0,
                            landmark1,
                            ...
                            landmark20
                        ]
                    ]

        Process:
            Validate landmarks.
            Convert normalized coordinates into pixels.
            Draw connections between landmark points.

        Output:
            np.ndarray:
                Frame with hand skeleton drawn.

        Failure Conditions:
            Returns original frame when no hand is detected.
        """

        # No hand detected.
        if not landmarks:
            return frame

        # Validate frame.
        if frame is None:
            raise ValueError(
                "Frame cannot be None."
            )

        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "Frame must be numpy array."
            )

        height, width, _ = frame.shape

        # Process every detected hand.
        for hand_landmarks in landmarks:

            # Convert normalized coordinates to pixels.
            points = [
                (
                    int((landmark.x or 0) * width),
                    int((landmark.y or 0) * height)
                )
                for landmark in hand_landmarks
            ]

            # Draw skeleton connections.
            for connection in self.connections:

                start = connection.start
                end = connection.end

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    self.LINE_COLOR,
                    self.LINE_THICKNESS
                )

            # Draw landmark points.
            for point in points:

                cv2.circle(
                    frame,
                    point,
                    4,
                    (0, 0, 255),
                    -1
                )

        return frame