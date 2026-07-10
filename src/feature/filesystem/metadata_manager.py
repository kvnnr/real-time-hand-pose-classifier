import json
from typing import Any
from pathlib import Path

class MetaDataManager:
    """
    What:
        Handle all MetaData Operations.
        
    Responsibilities:
        → Create metadata.json
        → Read metadata.json
        → Update metadata.json
        → Validate metadata structure
    """

    # -------------------------
    # Validation Methods
    # -------------------------

    def _validate_folder(self, folder_path: Path) -> None:
        """Validate that the dataset folder exists."""

        if not isinstance(folder_path, Path):
            raise TypeError("folder_path must be a pathlib.Path object.")

        if not folder_path.exists():
            raise FileNotFoundError(f"{folder_path} does not exist.")

        if not folder_path.is_dir():
            raise NotADirectoryError(f"{folder_path} is not a directory.")

    def _validate_key(self, key: str) -> None:
        """Validate a metadata key."""

        if not isinstance(key, str):
            raise TypeError("Metadata key must be a string.")

        if not key.strip():
            raise ValueError("Metadata key cannot be empty.")

    def _validate_metadata_exists(self, metadata_path: Path) -> None:
        """Validate that metadata.json exists."""

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"{metadata_path} does not exist."
            )

        if not metadata_path.is_file():
            raise ValueError(
                f"{metadata_path} is not a file."
            )

    def _validate_key_exists(self, metadata: dict, key: str) -> None:
        """Validate that a metadata key exists."""

        if key not in metadata:
            raise KeyError(f"'{key}' does not exist in metadata.json")
    
    # -------------------------
    # Metadata Operations
    # -------------------------

    #Create Metadata file in every Pose folder instances.
    def create_metadata_file(self, folder_path: Path, label: str) -> None:

        """
        What:
            Initializes a metadata.json file inside a dataset directory.

        Why:
            To ensure every dataset folder contains the metadata required
            for managing sample counters and other dataset information.

        Inputs:
            folder_path (Path)
                Path to an existing dataset directory where the metadata
                file should be initialized.
            label (str)
                Name of the pose folder.

        Process:
            Verify that the target directory exists.
            Check whether metadata.json already exists.
            If the file does not exist:
                Create metadata.json with the default metadata structure.
            If the file already exists:
                Raise FileExistError.
            Handle filesystem-related errors gracefully.

        Output:
            metadata.json
                A metadata file initialized with the default schema.
        """

        #VALIDATION!
        self._validate_folder(folder_path)
        self._validate_key(label)

        #Path where the metadatafile will be created.
        metadata_path = folder_path / "metadata.json"

        #Check if the metadata.json file already exist.
        if metadata_path.exists():
            raise FileExistsError(f"metadata.json file already exist at {folder_path}")

        #Set the attributes of metadata.
        metadata = {
            "counter": 0
        }

        #Create the metadata.json file.
        with open(metadata_path, 'w') as file:
            json.dump(metadata, file, indent=4)

        print(f"{label} Metadata was successfully created at {folder_path}.")

    #-------------------
    #NOTE: HELPER METHODS
    #--------------------

    #Get the value of specific metadata. | Return: metadata value.
    def get_metadata(self, folder_path: Path, key: str) -> Any:

        """
        What:
            Retrieves the value associated with a specified key
            from a metadata.json file.
        
        Input: 
            →folder_path: Path
                Path to the dataset folder that contains metadata.json.
            →key: str
                Name of the metadata field to retrieve
                (e.g., "count", "label")
        
        Process:
            → Validate that the metadata file exists.
            → Open and parse the JSON file.
            → Validate that the requested key exists.
            → Return the value associated with the key.
        
        Output:
            →Any
                →The value stored under the requested metadata key.
        
        Invariants:
            →metadata_path refers to a metadata.json file.
            →All metadata keys are strings.
            →The file contain valid .json object.
            →Only return value and not modify.
            →Metadata key must exist in the metadata.json file.
        """
        #VALIDATION!
        self._validate_folder(folder_path)
        self._validate_key(key)

        #Initiate the metadata_path.
        metadata_path = folder_path / "metadata.json"
        
        #VALIDATION!
        self._validate_metadata_exists(metadata_path)
        
        try:
            #Open and load the metadata file.
            with open(metadata_path, 'r') as file:
                metadata = json.load(file)
            
            #VALIDATION!
            self._validate_key_exists(metadata, key)

        except json.JSONDecodeError as e:
            raise RuntimeError("metadata.json contains invalid JSON.") from e

        except PermissionError as e:
            raise RuntimeError("Cannot access metadata.json.") from e
        
        #Return the value in metadata file using the key.
        return metadata[key]

    #Update the Value of key.
    def update_metadata(self, folder_path: Path, key: str, updated_value: Any) -> None:

        """
        What:
            Update the value associated with a specified key
            from a metadata.json file.
        
        Input: 
            →folder_path: Path
                Path to the dataset folder that contains metadata.json.
            →key: str
                Name of the metadata field to retrieve
                (e.g., "count", "label")
        
        Process:
            → Validate that the metadata file exists.
            → Open and parse the JSON file.
            → Validate that the key exists.
            → Update the value associated with the key.
        
        Output:
            →None
                →The updated value stored under the requested metadata key.
        
        Invariants:
            →metadata_path refers to a metadata.json file.
            →All metadata keys are strings.
            →The method modifies metadata.json in place.
            →The method returns None.
        """
        #VALIDATION!
        self._validate_folder(folder_path)
        self._validate_key(key)

        #Initiate the metadata_path.
        metadata_path = folder_path / "metadata.json"
        
        #VALIDATION!
        self._validate_metadata_exists(metadata_path)
        
        try:
            #Open and load the metadata file.
            with open(metadata_path, 'r') as file:
                metadata = json.load(file)
            
            #VALIDATION!
            self._validate_key_exists(metadata, key)

            #Update the value in metadata file using the key.
            metadata[key] = updated_value

            #Save the updated metadata changes.
            with open(metadata_path, "w") as file:
                json.dump(metadata, file, indent=4)

        except json.JSONDecodeError as e:
            raise RuntimeError("metadata.json contains invalid JSON.") from e

        except PermissionError as e:
            raise RuntimeError("Cannot update metadata.json.") from e