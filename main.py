#Pipelines:
from src.pipeline.save_pose_dataset import SavePoseDatasetsPipeline
from src.pipeline.dataset_filesystem import PoseDatasetFileSystemPipeline

#Models.
from sklearn.ensemble import RandomForestClassifier

#Module:
#Camera
from src.camera.camera import Camera
#FrameControls
from src.feature.frame_ops.frame_operations import FrameOps
#Hand detector
from src.detection.hand_detector import HandDetector
#Keyboard_operations
from src.feature.textbox.text_box_input import TextBoxInput
#text_box
from src.ui.text_box import TextBox
#create_folder
from src.feature.filesystem.create_folder import FolderCreation
#metadata_manager
from src.feature.filesystem.metadata_manager import MetaDataManager
#state
from src.core.state import AppState
#save_image
from src.feature.save_pose_image.save_image import SavePoseImage
#train_model_pipeline.py
from src.model.train_pipeline import TrainModelPipeline
#model_config.json
from src.model.model_config.model_config_loader import load_config

from pathlib import Path

"""
Main Operation of the system.
"""
def main():

    #CONSTANTS
    CLOSE_KEY = 27 #ESC Key.
    SAVE_POSE_KEY = 9 #TAB Key.
    ENTER_KEY = 13 #ENTER Key.
    TRAIN_KEY = 61 #Equal key.

    #Set Base directory of project folder.
    BASE_DIR = Path(__file__).resolve().parent

    #Object creations.
    cam = Camera()
    view = FrameOps()
    detect = HandDetector()
    keyboard_ops = TextBoxInput()
    text_box = TextBox()
    folder_creator = FolderCreation()
    metadata_manager = MetaDataManager()
    state = AppState()
    save_image = SavePoseImage()

    #Model creation:
    model_config = load_config()
    model = RandomForestClassifier(n_estimators=200, random_state= model_config.random_state)
    train = TrainModelPipeline(model, model_config)
    
    #Pipelines.
    save_pose_landmark = SavePoseDatasetsPipeline()
    filesystem = PoseDatasetFileSystemPipeline(folder_creator, metadata_manager)
    
    #Variables:
    COUNTER = 0

    #Folder path for datasets (PNG or JPEG).
    dataset_folder_path = BASE_DIR / "data" / "raw" / state.pose_label

    #Initialize the camera.
    cam.open_camera()

    """
    Why:
        A continuous loop is used to fetch frames from 
        the video stream in real time. The loop runs 
        until an exit condition is met, 
        allowing controlled termination of the capture process.
    """
    while True:

        #Folder creation for datasets (PNG or JPEG).
        dataset_folder_path = BASE_DIR / "data" / "raw" / state.pose_label
        
        #Grab each frame. Return: NDarray frames (BGR)
        state.frame = cam.get_frames()

        #Draw Text box with updated pose_label (text).
        state.frame = text_box.create_text_box(state.frame, state.pose_label)

        #Show the latest window Screen
        view.update_frame(state.frame)

        #Convert the BRG -> RGB frames. | Return: RGB frames.
        state.rgb_frame = cam.bgr_to_rgb(state.frame)

        #Grab the Operation key.
        state.key = view.get_key()

        #Handle text box input and update the texts.
        state.pose_label = keyboard_ops.update_text_input(state.key, state.pose_label)

        #Close the window.
        if state.key == CLOSE_KEY:
            break

        #When ENTER is pressed, it create a folder for the pose dataset (PNG or JPEG) and metadata file.
        #And Update the text to default empty.
        if state.key == ENTER_KEY:  # ENTER key
            filesystem.initiate_pose_dataset_folder(dataset_folder_path, state.pose_label)

        #Get landmarks
        if state.key == SAVE_POSE_KEY:

            #Save the 21 landmarks to NPZ file.
            result = save_pose_landmark.extract_and_save_landmark(state.pose_label, state.rgb_frame)
            
            #Check if landmark exists.
            if result is None:
                print(
                    "\nWARNING:"
                    "\nNo landmarks detected."
                    "\nPlease show your hand clearly and press TAB again."
                )

                continue
            
            #Get the Counter key in the metadatafile of specified pose folder.
            COUNTER = metadata_manager.get_metadata(dataset_folder_path, "counter")

            """
            NOTE: HELPER FUNCTION TO .save_image
            Convert RGB → BGR 
            Why:
                To save a frame(Image copy) in normal color.
            """
            state.bgr_frame = cam.rgb_to_bgr(state.rgb_frame)

            #Save the RAW FRAME (image) to pose folder.
            save_image.save_image(dataset_folder_path, state.bgr_frame, state.pose_label, COUNTER)

            """
            Why counter?:
                Counter act as a metadata for naming the numbers of frame(image) dataset inside 
                respective pose folders.
            """
            COUNTER += 1

            #Update the metadata counter.
            metadata_manager.update_metadata(dataset_folder_path, 'counter', COUNTER)

            print(f"\nInfos:\nData type: {type(result)}\nNumber of Landmark:{len(result)}")

        #Train the model.
        if state.key == TRAIN_KEY:
            state.accuracy = train.train_and_save_model()
            print(
                "Training was successful:" \
                f"Accuracy: {state.accuracy * 100:.2f}%"
            )

    #Release resources opened.   
    detect.close_mediapipe() 
    cam.release_camera()
    view.close_windows()

if __name__ == "__main__":
    main()
