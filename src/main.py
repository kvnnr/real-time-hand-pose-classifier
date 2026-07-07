#Pipelines.
from pipeline.landmark_pipeline import LandmarkPipeline
#Camera Module
from camera.camera import Camera, FrameControl
#Hand detector Module
from detection.hand_detector import HandDetector
#Normalizer Module
from src.feature.landmark_normalizer.normalizer import Normalizer
#Keyboard_operations Module
from src.feature.textbox.text_box_input import TextBoxInput
#filesystem Module
from src.feature.foldercreation.create_folder import FolderCreation
#text_box Module
from ui.text_box import TextBox

from pathlib import Path

"""
Main Operation of the system.
"""
def main():

    #CONSTANTS
    CLOSE_KEY = 27 #ESC Key.
    SAVE_POSE_KEY = 9 #TAB Key.
    ENTER_KEY = 13 #ENTER Key.

    #Set Base directory of project folder.
    BASE_DIR = Path(__file__).resolve().parent.parent

    #Object creations.
    cam = Camera()
    view = FrameControl()
    detect = HandDetector()
    normalizer = Normalizer()
    keyboard_ops = TextBoxInput()
    text_box = TextBox()
    filesystem = FolderCreation()

    #Pipelines.
    landmark_pipeline = LandmarkPipeline(detect, normalizer)
    
    #Variables:
    pose_label = ""

    #Folder path for datasets (PNG or JPEG).
    dataset_folder_path = BASE_DIR / "data" / "raw" / pose_label

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
        dataset_folder_path = BASE_DIR / "data" / "raw" / pose_label
        
        #Grab each frame. Return: NDarray frames (BGR)
        frame = cam.get_frames()

        #Draw Text box with updated pose_label (text).
        frame = text_box.create_text_box(frame, pose_label)

        #Show the latest window Screen
        view.show(frame)

        #Convert the BRG -> RGB frames. | Return: RGB frames.
        rgb_frames = cam.bgr_to_rgb(frame)

        #Grab the Operation key.
        key = view.operation_key()

        #Handle text box input and update the texts.
        pose_label = keyboard_ops.update_text_input(key, pose_label)

        #Close the window.
        if key == CLOSE_KEY:
            break

        #When ENTER is pressed, it create a folder for the pose dataset (PNG or JPEG).
        #And Update the text to default empty.
        if key == ENTER_KEY:  # ENTER key
           frame = filesystem.create_folder(dataset_folder_path, pose_label)
           
        #Get landmarks
        if key == SAVE_POSE_KEY:
            #Get the 21 Normalized Landmarks
            result = landmark_pipeline.extract(rgb_frames)
            print(f"\nLandmark is successfully extracted\n The landmarks:\n {result}")
            print(f"\nInfos:\nData type: {type(result)}\nNumber of Landmark:{len(result)}")

    #Release resources opened.   
    detect.close_mediapipe() 
    cam.release_camera()
    view.close()

if __name__ == "__main__":
    main()
