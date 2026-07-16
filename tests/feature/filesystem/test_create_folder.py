import pytest

from pathlib import Path
from unittest.mock import patch

from src.feature.filesystem.create_folder import FolderCreation


class TestCreateFolder:

    def setup_method(self):

        """
        Variables:
            self.creator: FolderCreation instance
        """

        self.creator = FolderCreation()

    #Test creating a new folder makes the directory case.
    def test_create_folder_creates_new_directory(self, tmp_path: Path, capsys):

        folder_path = tmp_path / "fist"

        self.creator.create_folder(folder_path, "fist")

        assert folder_path.exists()
        assert folder_path.is_dir()

        captured = capsys.readouterr()
        assert "New Pose folder created" in captured.out

    #Test creating nested folders makes all parent directories case.
    def test_create_folder_creates_nested_parents(self, tmp_path: Path):

        folder_path = tmp_path / "poses" / "fist"

        self.creator.create_folder(folder_path, "fist")

        assert folder_path.exists()
        assert folder_path.parent.exists()

    #Test folder that already exists is handled gracefully case.
    def test_create_folder_existing_folder_handled_gracefully(self, tmp_path: Path, capsys):

        folder_path = tmp_path / "fist"
        folder_path.mkdir(parents=True)

        self.creator.create_folder(folder_path, "fist")

        captured = capsys.readouterr()
        assert "Pose folder exists" in captured.out

    #Test OSError during folder creation is caught and reported case.
    def test_create_folder_os_error_handled_gracefully(self, tmp_path: Path, capsys):

        folder_path = tmp_path / "fist"

        with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
            self.creator.create_folder(folder_path, "fist")

        captured = capsys.readouterr()
        assert "Error creating folder" in captured.out
        assert not folder_path.exists()