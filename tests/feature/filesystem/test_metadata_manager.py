import json
import pytest

from pathlib import Path
from unittest.mock import patch, mock_open

from src.feature.filesystem.metadata_manager import MetaDataManager


class TestValidateFolder:

    def setup_method(self):

        """
        Variables:
            self.manager: MetaDataManager instance
        """

        self.manager = MetaDataManager()

    #Test valid existing directory passes without raising case.
    def test_validate_folder_valid_directory_passes(self, tmp_path: Path):

        self.manager._validate_folder(tmp_path)

    #Test non-Path folder_path raises TypeError case.
    def test_validate_folder_not_path_raises_type_error(self):

        with pytest.raises(TypeError):
            self.manager._validate_folder("not a path")

    #Test nonexistent folder raises FileNotFoundError case.
    def test_validate_folder_missing_raises_file_not_found_error(self, tmp_path: Path):

        folder_path = tmp_path / "missing"

        with pytest.raises(FileNotFoundError):
            self.manager._validate_folder(folder_path)

    #Test path that is a file, not a directory, raises NotADirectoryError case.
    def test_validate_folder_not_directory_raises_not_a_directory_error(self, tmp_path: Path):

        file_path = tmp_path / "file.txt"
        file_path.write_text("data")

        with pytest.raises(NotADirectoryError):
            self.manager._validate_folder(file_path)


class TestValidateKey:

    def setup_method(self):

        """
        Variables:
            self.manager: MetaDataManager instance
        """

        self.manager = MetaDataManager()

    #Test valid string key passes without raising case.
    def test_validate_key_valid_string_passes(self):

        self.manager._validate_key("counter")

    #Test non-string key raises TypeError case.
    def test_validate_key_not_string_raises_type_error(self):

        with pytest.raises(TypeError):
            self.manager._validate_key(123)

    #Test empty or whitespace key raises ValueError case.
    def test_validate_key_empty_raises_value_error(self):

        with pytest.raises(ValueError):
            self.manager._validate_key("   ")


class TestValidateMetadataExists:

    def setup_method(self):

        """
        Variables:
            self.manager: MetaDataManager instance
        """

        self.manager = MetaDataManager()

    #Test existing metadata file passes without raising case.
    def test_validate_metadata_exists_valid_file_passes(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text("{}")

        self.manager._validate_metadata_exists(metadata_path)

    #Test missing metadata file raises FileNotFoundError case.
    def test_validate_metadata_exists_missing_raises_file_not_found_error(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"

        with pytest.raises(FileNotFoundError):
            self.manager._validate_metadata_exists(metadata_path)

    #Test metadata path that is a directory raises ValueError case.
    def test_validate_metadata_exists_not_a_file_raises_value_error(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"
        metadata_path.mkdir()

        with pytest.raises(ValueError):
            self.manager._validate_metadata_exists(metadata_path)


class TestValidateKeyExists:

    def setup_method(self):

        """
        Variables:
            self.manager: MetaDataManager instance
        """

        self.manager = MetaDataManager()

    #Test existing key passes without raising case.
    def test_validate_key_exists_valid_key_passes(self):

        self.manager._validate_key_exists({"counter": 0}, "counter")

    #Test missing key raises KeyError case.
    def test_validate_key_exists_missing_key_raises_key_error(self):

        with pytest.raises(KeyError):
            self.manager._validate_key_exists({"counter": 0}, "label")


class TestCreateMetadataFile:

    def setup_method(self):

        """
        Variables:
            self.manager: MetaDataManager instance
        """

        self.manager = MetaDataManager()

    #Test creating metadata.json writes the default schema case.
    def test_create_metadata_file_creates_default_schema(self, tmp_path: Path):

        self.manager.create_metadata_file(tmp_path, "fist")

        metadata_path = tmp_path / "metadata.json"
        assert metadata_path.exists()

        metadata = json.loads(metadata_path.read_text())
        assert metadata == {"counter": 0}

    #Test creating metadata.json when it already exists raises FileExistsError case.
    def test_create_metadata_file_existing_file_raises_file_exists_error(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text("{}")

        with pytest.raises(FileExistsError):
            self.manager.create_metadata_file(tmp_path, "fist")

    #Test creating metadata.json in a nonexistent folder raises FileNotFoundError case.
    def test_create_metadata_file_missing_folder_raises_file_not_found_error(self, tmp_path: Path):

        folder_path = tmp_path / "missing"

        with pytest.raises(FileNotFoundError):
            self.manager.create_metadata_file(folder_path, "fist")

    #Test creating metadata.json with an invalid label raises ValueError case.
    def test_create_metadata_file_invalid_label_raises_value_error(self, tmp_path: Path):

        with pytest.raises(ValueError):
            self.manager.create_metadata_file(tmp_path, "   ")


class TestGetMetadata:

    def setup_method(self):

        """
        Variables:
            self.manager: MetaDataManager instance
        """

        self.manager = MetaDataManager()

    #Test getting an existing key returns its value case.
    def test_get_metadata_returns_value(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(json.dumps({"counter": 5}))

        result = self.manager.get_metadata(tmp_path, "counter")

        assert result == 5

    #Test getting metadata when metadata.json is missing raises FileNotFoundError case.
    def test_get_metadata_missing_file_raises_file_not_found_error(self, tmp_path: Path):

        with pytest.raises(FileNotFoundError):
            self.manager.get_metadata(tmp_path, "counter")

    #Test getting a key that does not exist raises KeyError case.
    def test_get_metadata_missing_key_raises_key_error(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(json.dumps({"counter": 5}))

        with pytest.raises(KeyError):
            self.manager.get_metadata(tmp_path, "label")

    #Test getting metadata from a corrupted JSON file raises RuntimeError case.
    def test_get_metadata_invalid_json_raises_runtime_error(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text("not valid json")

        with pytest.raises(RuntimeError):
            self.manager.get_metadata(tmp_path, "counter")

    #Test getting metadata when file access is denied raises RuntimeError case.
    def test_get_metadata_permission_error_raises_runtime_error(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(json.dumps({"counter": 5}))

        with patch("builtins.open", side_effect=PermissionError):
            with pytest.raises(RuntimeError):
                self.manager.get_metadata(tmp_path, "counter")


class TestUpdateMetadata:

    def setup_method(self):

        """
        Variables:
            self.manager: MetaDataManager instance
        """

        self.manager = MetaDataManager()

    #Test updating an existing key writes the new value to disk case.
    def test_update_metadata_updates_value_on_disk(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(json.dumps({"counter": 0}))

        self.manager.update_metadata(tmp_path, "counter", 10)

        metadata = json.loads(metadata_path.read_text())
        assert metadata["counter"] == 10

    #Test updating metadata when metadata.json is missing raises FileNotFoundError case.
    def test_update_metadata_missing_file_raises_file_not_found_error(self, tmp_path: Path):

        with pytest.raises(FileNotFoundError):
            self.manager.update_metadata(tmp_path, "counter", 10)

    #Test updating a key that does not exist raises KeyError case.
    def test_update_metadata_missing_key_raises_key_error(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(json.dumps({"counter": 0}))

        with pytest.raises(KeyError):
            self.manager.update_metadata(tmp_path, "label", "fist")

    #Test updating metadata in a corrupted JSON file raises RuntimeError case.
    def test_update_metadata_invalid_json_raises_runtime_error(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text("not valid json")

        with pytest.raises(RuntimeError):
            self.manager.update_metadata(tmp_path, "counter", 10)

    #Test updating metadata when file access is denied raises RuntimeError case.
    def test_update_metadata_permission_error_raises_runtime_error(self, tmp_path: Path):

        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(json.dumps({"counter": 0}))

        with patch("builtins.open", mock_open(read_data=json.dumps({"counter": 0}))) as mocked_open:
            mocked_open.side_effect = [mocked_open.return_value, PermissionError()]

            with pytest.raises(RuntimeError):
                self.manager.update_metadata(tmp_path, "counter", 10)