import cv2 as cv


class TextStyles:

    """
    TextStyles Contract

    What:
        Centralized collection of user interface
        styling constants.

    Why:
        Prevent hard-coded values from being scattered
        throughout the project.

        Every UI component should use these values to
        maintain a consistent appearance.

    Contains:

        Fonts

        Font Sizes

        Colors

        Line Thickness

        UI Margins

    Process:

        No processing is performed.

        This class simply stores reusable constants.

    Output:

        Shared styling configuration.

    Failure Conditions:

        None.

    Invariants:

        Styling values remain constant while the
        application is running.
    """

    # --------------------------
    # Fonts
    # --------------------------

    TITLE_FONT = cv.FONT_HERSHEY_DUPLEX

    TEXT_FONT = cv.FONT_HERSHEY_SIMPLEX

    SMALL_FONT = cv.FONT_HERSHEY_PLAIN

    # --------------------------
    # Font Scale
    # --------------------------

    TITLE_SCALE = 0.9

    NORMAL_SCALE = 0.60

    SMALL_SCALE = 1.0

    LARGE_SCALE = 1.20

    # --------------------------
    # Thickness
    # --------------------------

    TITLE_THICKNESS = 2

    NORMAL_THICKNESS = 1

    LARGE_THICKNESS = 3

    # --------------------------
    # Colors (BGR)
    # --------------------------

    WHITE = (255,255,255)

    GREEN = (0,255,0)

    RED = (0,0,255)

    YELLOW = (0,255,255)

    BLUE = (255,150,0)

    LIGHT_GRAY = (180,180,180)

    DARK_PANEL = (40,40,40)

    # --------------------------
    # Margins
    # --------------------------

    PADDING = 15

    LINE_SPACING = 28