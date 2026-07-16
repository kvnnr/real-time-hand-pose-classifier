from enum import Enum, auto



class VisionEvent(Enum):

    """
    VisionEvent Contract

    What:
        Defines all application events.

    Why:
        Removes direct dependency between
        keyboard input and application logic.

    Example:

        Keyboard:
            "="
        becomes:
            TRAIN_MODEL event

    """

    NONE = auto()

    EXIT = auto()

    CREATE_FOLDER = auto()

    SAVE_POSE = auto()

    TRAIN_MODEL = auto()