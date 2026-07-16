from __future__ import annotations
import numpy as np
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.schemas.event_schema import VisionEvent

if TYPE_CHECKING:

    # Camera
    from src.camera.camera import Camera

    # Frame Operations
    from src.feature.frame_ops.frame_operations import FrameOps

    # Detection
    from src.detection.hand_detector import HandDetector

    # Prediction
    from src.Inference.live_model import PredictionModel

    # UI
    from src.ui.ui_renderer import UIRenderer

    # Keyboard
    from src.feature.textbox.text_box_input import TextBoxInput

    # Visualization
    from src.visuals.hand_skeleton_drawer import HandSkeletonDrawer

    # Pipeline
    from src.pipeline.extract_landmark import ExtractLandmarkPipeline
    from src.model.train_pipeline import TrainModelPipeline
    from src.pipeline.extract_and_save_pose_dataset import SavePoseDatasetsPipeline
    from src.pipeline.dataset_filesystem import PoseDatasetFileSystemPipeline

    # File System
    from src.feature.filesystem.create_folder import FolderCreation
    from src.feature.filesystem.metadata_manager import MetaDataManager
    from src.feature.save_pose_image.save_image import SavePoseImage

    # State
    from src.schemas.state_schema import VisionState
    from src.schemas.ui_schema import UiState

    #Event handler.
    from src.controller.event_manager import EventManager

class VisionController:

    """
    VisionController Contract

    What:
        Controls the complete real-time hand pose
        recognition application workflow.

    Why:
        Separates application execution logic
        from main.py.

        The controller manages communication
        between independent system modules:

            Camera
            Hand Detector
            Landmark Extractor
            Prediction Model
            UI Renderer
            Dataset Pipeline
            Training Pipeline

    Input:
        Injected system dependencies.

        Includes:

            Hardware modules.

            Vision processing modules.

            User interface modules.

            Dataset modules.

            Application state.

    Process:
        1. Capture camera frame.

        2. Convert image format.

        3. Detect hand landmarks.

        4. Extract normalized features.

        5. Predict hand gesture.

        6. Update UI state.

        7. Render application interface.

        8. Handle user events.

    Output:
        None.

        Runs the real-time application lifecycle.

    Failure Conditions:

        - Camera cannot initialize.

        - Frame is invalid.

        - Detector fails.

        - Model is unavailable.

    Invariants:

        - UI always receives latest application state.

        - Camera resources are released correctly.

        - State objects remain synchronized.

    """

    # Root directory where pose datasets (folders + images) are stored.
    # NOTE: adjust this if your project keeps the dataset root somewhere else
    DATASET_ROOT = Path("data/raw")

    def __init__(
        self,
        cam: Camera,
        view: FrameOps,
        detector: HandDetector,
        predictor: PredictionModel,
        ui: UIRenderer,
        keyboard_ops: TextBoxInput,
        event_manager: EventManager,
        skeleton_drawer: HandSkeletonDrawer,
        extract_landmarks: ExtractLandmarkPipeline,
        train: TrainModelPipeline,
        save_pose_landmark: SavePoseDatasetsPipeline,
        filesystem: PoseDatasetFileSystemPipeline,
        save_image: SavePoseImage,
        metadata_manager: MetaDataManager,
        vision_state: VisionState,
        ui_state: UiState
    ) -> None:

        """
        Initialize VisionController.

        Input:
            cam:
                Camera capture module.

            view:
                Frame display and keyboard handler.

            detector:
                MediaPipe hand detection module.

            predictor:
                Machine learning inference model.

            ui:
                Application UI renderer.

            keyboard_ops:
                Keyboard input processor.

            skeleton_drawer:
                Draws hand landmark skeleton.

            extract_landmarks:
                Extracts normalized landmark features.

            train:
                Model training pipeline.

            save_pose_landmark:
                Dataset landmark saving pipeline.

            filesystem:
                Dataset folder manager.

            save_image:
                Raw image saving module.

            metadata_manager:
                Dataset metadata handler.

            vision_state:
                Stores vision processing data.

            ui_state:
                Stores UI related information.

        Output:
            None.

        """

        # Hardware modules.
        self.cam: Camera = cam
        self.view: FrameOps = view

        # Vision processing modules.
        self.detector: HandDetector = detector
        self.predictor: PredictionModel = predictor
        self.extract_landmarks: ExtractLandmarkPipeline = extract_landmarks

        # Visualization modules.
        self.ui: UIRenderer = ui
        self.skeleton_drawer: HandSkeletonDrawer = skeleton_drawer

        # User input.
        self.keyboard_ops: TextBoxInput = keyboard_ops

        # Dataset and training modules.
        self.train: TrainModelPipeline = train
        self.save_pose_landmark: SavePoseDatasetsPipeline = (save_pose_landmark)
        self.filesystem: PoseDatasetFileSystemPipeline = filesystem
        self.save_image: SavePoseImage = save_image
        self.metadata_manager: MetaDataManager = metadata_manager

        # Application state.
        self.vision_state: VisionState = vision_state
        self.ui_state: UiState = ui_state

        #Event Manager.
        self.event_manager = event_manager

        # FPS tracking
        self.previous_time: float = time.time()
        
        # Check if model is loaded.
        self.ui_state.model_loaded = self.predictor.loaded

    def process_frame(self) -> None:

        """
        Process one complete vision cycle.

        What:
            Executes one frame processing pipeline.

        Process:
            Camera Frame
                ↓
            Hand Detection
                ↓
            Landmark Extraction
                ↓
            Prediction
                ↓
            UI Rendering

        Output:
            None.
            Updates internal state objects.

        Failure Conditions:
            Invalid camera frame.

        """

        # Capture frame from camera.
        self.vision_state.frame = (self.cam.get_frames())

        # Convert BGR image to RGB.
        self.vision_state.rgb_frame = (self.cam.bgr_to_rgb(self.vision_state.frame))

        # Reset prediction state.
        self.ui_state.prediction = "Waiting..."
        self.ui_state.confidence = 0.0

        # Detect hand landmarks.
        self.vision_state.landmarks = (self.detector.detect_hands(self.vision_state.rgb_frame))

        # Update hand detection status.
        self.ui_state.hand_detected = (self.vision_state.landmarks is not None)

        # Run prediction pipeline.
        if self.vision_state.landmarks is not None:
            
            #Extract 21 landmarks.
            self.vision_state.features = (self.extract_landmarks.extract(self.vision_state.rgb_frame))

            #Check if features are detected.
            if self.vision_state.features is not None:

                CONFIDENCE_THRESHOLD = 0.60

                #Get Prediction and confidence.
                (prediction, confidence) = self.predictor.predict(np.array(self.vision_state.features))

                if confidence < CONFIDENCE_THRESHOLD:

                    self.ui_state.prediction = "Unknown"
                    
                else:
                    self.ui_state.prediction = prediction

                self.ui_state.confidence = confidence
    

        # Draw hand skeleton.
        self.vision_state.frame = (self.skeleton_drawer.draw(self.vision_state.frame, self.vision_state.landmarks))

        # Render UI.
        self.vision_state.frame = (self.ui.draw(self.vision_state.frame, self.ui_state))

    def handle_events(self) -> bool:

        """
        Handle application events.

        What:
            Receives keyboard input
            and converts it into events.

        Output:
            True:
                Continue application.
            False:

                Shutdown application.
        """

        key = self.view.get_key()
        event = self.event_manager.process_key(key)
        self.execute_event(event, key)

        if event == VisionEvent.EXIT:
            return False

        return True
    
    def update_fps(self) -> None:
    
        """
        Calculate current frames per second.

        Formula:
            FPS = 1 / frame_time_difference

        Updates:
            ui_state.fps
        """

        current_time = time.time()

        elapsed_time = current_time - self.previous_time

        if elapsed_time > 0:
            self.ui_state.fps = 1 / elapsed_time

        self.previous_time = current_time

    def run(self) -> None:

        """
        Start application.

        Process:
            1. Initialize camera.

            2. Run frame processing loop.

            3. Handle events.

            4. Shutdown resources.

        Output:
            None.

        """

        self.cam.open_camera()
        running: bool = True

        while running:

            self.update_fps()
            self.process_frame()
            self.view.update_frame(self.vision_state.frame)
            running = self.handle_events()

        self.shutdown()

    def shutdown(self) -> None:

        """
        Shutdown application resources.

        What:
            Releases hardware and UI resources.

        Why:
            Prevents resource leakage.

        Output:
            None.

        """

        self.detector.close_mediapipe()
        self.cam.release_camera()
        self.view.close_windows()

    def _pose_folder_path(self) -> Path:
        """
        Build the dataset folder path for the pose label currently
        typed into the UI textbox.
        """
        return self.DATASET_ROOT / self.ui_state.pose_label

    def execute_event(self, event: VisionEvent, key: int) -> None:
        
        """
        Execute application event.

        Input:
            VisionEvent.

        Process:
            Route event to correct action.

        Output:
            None.
        """

        if event == VisionEvent.CREATE_FOLDER:

            label = self.ui_state.pose_label

            if not label or not label.strip():
                print("Cannot create folder: pose label is empty.")
                return

            folder_path = self._pose_folder_path()

            # Creates the folder AND metadata.json (counter starts at 0).
            self.filesystem.initiate_pose_dataset_folder(folder_path, label)

        elif event == VisionEvent.SAVE_POSE:

            label = self.ui_state.pose_label

            if not label or not label.strip():
                print("Cannot save pose: pose label is empty.")
                return

            folder_path = self._pose_folder_path()

            if not folder_path.exists():
                print(f"Cannot save pose: {folder_path} does not exist. "
                      f"Create the folder first.")
                return

            # Pull current counter, save the raw frame under it, then bump it.
            counter = self.metadata_manager.get_metadata(folder_path, "counter")

            self.save_image.save_image(folder_path, self.vision_state.frame, label, counter)

            self.metadata_manager.update_metadata(folder_path, "counter", counter + 1)

            # Also extract + persist the normalized landmark vector for training.
            self.save_pose_landmark.extract_and_save_landmark(label, self.vision_state.rgb_frame)

        elif event == VisionEvent.TRAIN_MODEL:
            print("Training event triggered")
            self.vision_state.accuracy = self.train.train_and_save_model()
            print(f"Accuracy: {self.vision_state.accuracy * 100:.2f}%")

        elif event == VisionEvent.TYPE_CHAR:
            self.ui_state.pose_label = self.keyboard_ops.update_text_input(key, self.ui_state.pose_label)

        elif event == VisionEvent.BACKSPACE:
            self.ui_state.pose_label = self.keyboard_ops.update_text_input(key, self.ui_state.pose_label)