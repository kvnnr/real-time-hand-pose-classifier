# Controller
from src.controller.vision_controller import VisionController


# Pipelines
from src.pipeline.extract_and_save_pose_dataset import SavePoseDatasetsPipeline
from src.pipeline.dataset_filesystem import PoseDatasetFileSystemPipeline
from src.model.train_pipeline import TrainModelPipeline
from src.pipeline.extract_landmark import ExtractLandmarkPipeline


# Models
from sklearn.ensemble import RandomForestClassifier
from src.Inference.live_model import PredictionModel


# Camera
from src.camera.camera import Camera

# Frame Controls
from src.feature.frame_ops.frame_operations import FrameOps

# Detection
from src.detection.hand_detector import HandDetector

# Keyboard
from src.feature.textbox.text_box_input import TextBoxInput

# UI
from src.ui.ui_renderer import UIRenderer

# File System
from src.feature.filesystem.create_folder import FolderCreation
from src.feature.filesystem.metadata_manager import MetaDataManager

# State
from src.schemas.state_schema import VisionState
from src.schemas.ui_schema import UiState

# Dataset
from src.feature.save_pose_image.save_image import SavePoseImage

# Model Config
from src.model.model_config.model_config_loader import load_config

# Visualization
from src.visuals.hand_skeleton_drawer import HandSkeletonDrawer

# Feature
from src.feature.landmark_normalizer.normalizer import Normalizer

#Event Manager
from src.controller.event_manager import EventManager

from pathlib import Path
import mediapipe as mp


"""
Main Operation of the system.

Responsible for:

    1. Creating system components.
    2. Injecting dependencies.
    3. Starting controller.

The application logic is handled
inside VisionController.
"""

def main():

    # Project directory.
    BASE_DIR = Path(__file__).resolve().parent

    # MediaPipe skeleton topology.
    connections = (mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS)

    # -----------------------------
    # Core Modules
    # -----------------------------

    cam = Camera()

    view = FrameOps()

    detector = HandDetector()

    normalizer = Normalizer()

    keyboard_ops = TextBoxInput()

    ui = UIRenderer()

    event_manager = EventManager()


    # -----------------------------
    # State
    # -----------------------------

    vision_state = VisionState()

    ui_state = UiState()


    # -----------------------------
    # Visualization
    # -----------------------------

    skeleton_drawer = HandSkeletonDrawer(connections)

    # -----------------------------
    # Feature Pipeline
    # -----------------------------

    extract_landmarks = ExtractLandmarkPipeline(detector,normalizer)

    # -----------------------------
    # Dataset Pipeline
    # -----------------------------

    folder_creator = FolderCreation()
    metadata_manager = MetaDataManager()
    filesystem = PoseDatasetFileSystemPipeline(folder_creator,metadata_manager)
    save_pose_landmark = SavePoseDatasetsPipeline()
    save_image = SavePoseImage()

    # -----------------------------
    # Model
    # -----------------------------

    model_config = load_config()

    model = RandomForestClassifier(n_estimators=200, random_state=model_config.random_state)

    train = TrainModelPipeline(model, model_config)

    predictor = PredictionModel()

    # -----------------------------
    # Controller
    # -----------------------------

    controller = VisionController(

        cam=cam,

        view=view,

        detector=detector,

        predictor=predictor,

        ui=ui,

        keyboard_ops=keyboard_ops,
        
        event_manager=event_manager,

        skeleton_drawer=skeleton_drawer,

        extract_landmarks=extract_landmarks,

        train=train,

        save_pose_landmark=save_pose_landmark,

        filesystem=filesystem,

        save_image=save_image,

        metadata_manager=metadata_manager,

        vision_state=vision_state,

        ui_state=ui_state
    )

    # Start application.
    controller.run()

if __name__ == "__main__":
    main()