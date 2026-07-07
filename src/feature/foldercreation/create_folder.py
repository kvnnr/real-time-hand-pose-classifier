from pathlib import Path

class FolderCreation:
    """
    FileSystem Contract

    What:
        A utility class responsible for managing filesystem operations related to dataset
        or pose storage directories.

    Why:
        To ensure structured and automated creation of required project folders for
        data collection and organization in computer vision pipelines.

    Inputs:
        folder_path (Path)
            Target directory path where the folder should be created.

        text (str)
            Label or identifier associated with the folder creation process.

    Process:
        Receive target folder path
        Attempt to create directory using mkdir with parent creation enabled
        If folder already exists, handle FileExistsError gracefully
        If system error occurs, capture and report OSError
        Reset text value after processing
        Return updated text state

    Output:
        str
            Updated text value (reset to empty string after folder operation)
    """
    
    #Handle creating Poses folder.
    #Folder creation. | Return: text
    def create_folder(self, folder_path: Path, text: str) -> str:
    
        #Create folder for dataset.
        try: 
            folder_path.mkdir(parents=True)
            print(f"New Pose folder created at {folder_path} named {text}")

        #Check if the folder already exist.
        except FileExistsError:
            print(f"Folder already exists: {folder_path}")
        
        #Check for any Error in creating folder.
        except OSError as e:
            print(f"Error creating folder: {e}")

        #Update the text to default empty after pressing enter.
        text = ""

        return text
