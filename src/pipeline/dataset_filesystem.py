from src.feature.filesystem.create_folder import FolderCreation
from src.feature.filesystem.metadata_manager import MetaDataManager
from pathlib import Path


class PoseDatasetFileSystemPipeline:
    """
    What:
        Pipeline responsible for initializing a pose dataset 
        directory structure and generating its metadata file

    Pipeline:
        create_folder.py
            ↓
        metadata_manager.py
    
    Dependencies:
        FolderCreation:
            create_folder(label: str) -> Path

        MetaDataManager:
            create_metadata_file(folder_path: Path, label: str) -> None
            
    Responsibilities:
        →Create Folder named after the label of pose.
        →Create metadata.json inside the folders
    
    Input:
        folder_path: Path
            Target directory path where the folder should be created.
        label: str
            Label or identifier associated with the folder creation process.
    
    Process:
        Create folder named after the pose.
        Create metadata.json file in each folders

    Output:
        None:
            Folder named after the pose and metadata.json should
            exist

    Invariants:
        →Folder must be named after the pose label.
        →Folders must have metadata.json
    """

    #Initiate objects.
    def __init__(self, folder_creator: FolderCreation, metadata_manager: MetaDataManager) -> None:
        
        self._validate_dependencies(folder_creator, metadata_manager)
        
        self.folder_creator = folder_creator
        self.metadata_manager =  metadata_manager
    
    def _validate_dependencies(self,folder_creator: FolderCreation,metadata_manager: MetaDataManager) -> None:
        """Validate injected dependencies."""

        if not isinstance(folder_creator, FolderCreation):
            raise TypeError("folder_creator must be a FolderCreation instance." )

        if not isinstance(metadata_manager, MetaDataManager):
            raise TypeError("metadata_manager must be a MetaDataManager instance." )

    def _validate_folder_path(self, folder_path: Path) -> None:
        """Validate the target folder path."""

        if not isinstance(folder_path, Path):
            raise TypeError("folder_path must be a pathlib.Path object.")
        
    def _validate_label(self, label: str) -> None:
        """Validate the pose label."""

        if not isinstance(label, str):
            raise TypeError("label must be a string.")

        if not label.strip():
            raise ValueError("label cannot be empty.")
        
    #Main Pipeline method.
    def initiate_pose_dataset_folder(self, folder_path: Path, label: str) -> None:
        
        #VALIDATIONS!
        self._validate_folder_path(folder_path)
        self._validate_label(label)
        
        #Create folder directory for pose dataset.
        self.folder_creator.create_folder(folder_path, label)

        #Create metadata.json on pose dataset folder.
        self.metadata_manager.create_metadata_file(folder_path, label)

        