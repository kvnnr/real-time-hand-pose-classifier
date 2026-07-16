from src.schemas.event_schema import VisionEvent

class EventManager:

    """
    EventManager Contract

    What:

        Converts raw keyboard input
        into application events.

    Why:
        Separates input handling from
        application behavior.

    Input:
        Keyboard ASCII value.

    Process:
        Read key.
        Convert key into event.

    Output:
        VisionEvent.

    """

    # Keyboard mapping.

    CLOSE_KEY = 27
    ENTER_KEY = 13
    SAVE_KEY = 9
    TRAIN_KEY = 61

    def __init__(self) -> None:

        """
        Initialize EventManager.
        """
        pass

    def process_key(self, key: int) -> VisionEvent:

        """
        Convert keyboard key into event.

        Input:
            key:
                OpenCV keyboard value.

        Output:
            VisionEvent.
        """

        if key == self.CLOSE_KEY:
            return VisionEvent.EXIT

        if key == self.ENTER_KEY:
            return VisionEvent.CREATE_FOLDER

        if key == self.SAVE_KEY:
            return VisionEvent.SAVE_POSE

        if key == self.TRAIN_KEY:
            return VisionEvent.TRAIN_MODEL

        return VisionEvent.NONE