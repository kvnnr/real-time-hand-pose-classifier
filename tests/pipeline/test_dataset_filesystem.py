
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.feature.filesystem.create_folder import FolderCreation
from src.feature.filesystem.metadata_manager import MetaDataManager
from src.pipeline.dataset_filesystem import PoseDatasetFileSystemPipeline


@pytest.fixture
def mock_folder_creator() -> MagicMock:
    
    """
    Provides a fake FolderCreation.

    Output:
        MagicMock:
            Mimics create_folder(), passes isinstance checks.
    """

    return MagicMock(spec=FolderCreation)


@pytest.fixture
def mock_metadata_manager() -> MagicMock:
    
    """
    Provides a fake MetaDataManager.

    Output:
        MagicMock:
            Mimics create_metadata_file(), passes isinstance
            checks.
    """

    return MagicMock(spec=MetaDataManager)


@pytest.fixture
def pipeline(mock_folder_creator: MagicMock, mock_metadata_manager: MagicMock) -> PoseDatasetFileSystemPipeline:
    
    """
    Provides a PoseDatasetFileSystemPipeline with mocked
    dependencies.

    Input:
        mock_folder_creator: MagicMock
        mock_metadata_manager: MagicMock

    Output:
        PoseDatasetFileSystemPipeline:
            Instance ready for testing.
    """

    return PoseDatasetFileSystemPipeline(mock_folder_creator, mock_metadata_manager)


class TestInit:
    """
    Tests pipeline construction and dependency validation.
    """

    def test_sets_folder_creator_and_metadata_manager(
        self,
        mock_folder_creator: MagicMock,
        mock_metadata_manager: MagicMock,
        ) -> None:
        
        """
        Confirms dependencies are stored as-is.
        """

        pipeline = PoseDatasetFileSystemPipeline(mock_folder_creator, mock_metadata_manager)

        assert pipeline.folder_creator is mock_folder_creator
        assert pipeline.metadata_manager is mock_metadata_manager

    def test_raises_type_error_when_folder_creator_invalid(self, mock_metadata_manager: MagicMock) -> None:
        
        """
        Confirms an invalid folder_creator raises TypeError.
        """

        with pytest.raises(TypeError):
            PoseDatasetFileSystemPipeline("not_a_folder_creator", mock_metadata_manager)

    def test_raises_type_error_when_metadata_manager_invalid(self, mock_folder_creator: MagicMock) -> None:
        
        """
        Confirms an invalid metadata_manager raises TypeError.
        """

        with pytest.raises(TypeError):
            PoseDatasetFileSystemPipeline(mock_folder_creator, "not_a_metadata_manager")


class TestValidateFolderPath:
    """
    Tests _validate_folder_path().
    """

    def test_raises_type_error_when_folder_path_not_path(self, pipeline: PoseDatasetFileSystemPipeline) -> None:
        
        """
        Confirms non-Path folder_path raises TypeError.
        """

        with pytest.raises(TypeError):
            pipeline._validate_folder_path("dataset/open_palm")

    def test_accepts_valid_folder_path(self, pipeline: PoseDatasetFileSystemPipeline) -> None:
        
        """
        Confirms a valid Path does not raise.
        """

        pipeline._validate_folder_path(Path("dataset/open_palm"))


class TestValidateLabel:
    """
    Tests _validate_label().
    """

    def test_raises_type_error_when_label_not_str(self, pipeline: PoseDatasetFileSystemPipeline) -> None:
        
        """
        Confirms non-string label raises TypeError.
        """

        with pytest.raises(TypeError):
            pipeline._validate_label(123)

    def test_raises_value_error_when_label_empty(self, pipeline: PoseDatasetFileSystemPipeline) -> None:
        
        """
        Confirms empty/whitespace label raises ValueError.
        """

        with pytest.raises(ValueError):
            pipeline._validate_label("   ")

    def test_accepts_valid_label(self, pipeline: PoseDatasetFileSystemPipeline) -> None:
        
        """
        Confirms a valid label does not raise.
        """

        pipeline._validate_label("open_palm")


class TestInitiatePoseDatasetFolder:
    """
    Tests initiate_pose_dataset_folder().
    """

    def test_creates_folder_and_metadata_file(
        self,
        pipeline: PoseDatasetFileSystemPipeline,
        mock_folder_creator: MagicMock,
        mock_metadata_manager: MagicMock,
        ) -> None:
        
        """
        Confirms create_folder() and create_metadata_file()
        are called with folder_path and label.
        """

        folder_path = Path("dataset/open_palm")
        label = "open_palm"

        pipeline.initiate_pose_dataset_folder(folder_path, label)

        mock_folder_creator.create_folder.assert_called_once_with(folder_path, label)
        mock_metadata_manager.create_metadata_file.assert_called_once_with(folder_path, label)

    def test_creates_metadata_after_folder(
        self,
        pipeline: PoseDatasetFileSystemPipeline,
        mock_folder_creator: MagicMock,
        mock_metadata_manager: MagicMock,
        ) -> None:
        
        """
        Confirms create_folder() runs before
        create_metadata_file().
        """

        # Track call order.
        call_order = []

        mock_folder_creator.create_folder.side_effect = lambda *_: call_order.append("create_folder")
        mock_metadata_manager.create_metadata_file.side_effect = lambda *_: call_order.append("create_metadata_file")

        pipeline.initiate_pose_dataset_folder(Path("dataset/open_palm"), "open_palm")

        assert call_order == ["create_folder", "create_metadata_file"]

    def test_returns_none(self, pipeline: PoseDatasetFileSystemPipeline) -> None:
        
        """
        Confirms the method returns None.
        """

        result = pipeline.initiate_pose_dataset_folder(Path("dataset/open_palm"), "open_palm")

        assert result is None

    def test_raises_type_error_when_folder_path_invalid(
        self,
        pipeline: PoseDatasetFileSystemPipeline,
        mock_folder_creator: MagicMock,
        ) -> None:
        
        """
        Confirms invalid folder_path short-circuits before
        folder creation.
        """

        with pytest.raises(TypeError):
            pipeline.initiate_pose_dataset_folder("dataset/open_palm", "open_palm")

        mock_folder_creator.create_folder.assert_not_called()

    def test_raises_value_error_when_label_invalid(
        self,
        pipeline: PoseDatasetFileSystemPipeline,
        mock_folder_creator: MagicMock,
        ) -> None:
        
        """
        Confirms invalid label short-circuits before folder
        creation.
        """

        with pytest.raises(ValueError):
            pipeline.initiate_pose_dataset_folder(Path("dataset/open_palm"), "   ")

        mock_folder_creator.create_folder.assert_not_called()

    def test_does_not_create_metadata_when_folder_creation_fails(
        self,
        pipeline: PoseDatasetFileSystemPipeline,
        mock_folder_creator: MagicMock,
        mock_metadata_manager: MagicMock,
        ) -> None:
        
        """
        Confirms create_metadata_file() is not called when
        create_folder() raises.
        """

        mock_folder_creator.create_folder.side_effect = OSError

        with pytest.raises(OSError):
            pipeline.initiate_pose_dataset_folder(Path("dataset/open_palm"), "open_palm")

        mock_metadata_manager.create_metadata_file.assert_not_called()