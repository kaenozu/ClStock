import shutil
from pathlib import Path

import pytest


@pytest.fixture()
def temp_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    yield workspace
    if workspace.exists():
        shutil.rmtree(workspace)


def create_file(path: Path):
    path.write_text("sample")
    return path


def test_file_organizer_groups_files_by_extension(temp_workspace):
    from utils.file_organizer import organize_files

    create_file(temp_workspace / "report.txt")
    create_file(temp_workspace / "notes.md")
    create_file(temp_workspace / "photo.JPG")

    organize_files(
        temp_workspace,
        rules={"documents": [".txt", ".md"], "images": [".jpg"]},
        default_folder="misc",
    )

    documents_dir = temp_workspace / "documents"
    images_dir = temp_workspace / "images"
    misc_dir = temp_workspace / "misc"

    assert (documents_dir / "report.txt").exists()
    assert (documents_dir / "notes.md").exists()
    assert (images_dir / "photo.JPG").exists()
    assert not misc_dir.exists()


def test_file_organizer_places_unknown_extensions_in_default(temp_workspace):
    from utils.file_organizer import organize_files

    create_file(temp_workspace / "archive.zip")

    organize_files(temp_workspace, rules={"documents": [".txt"]}, default_folder="others")

    assert (temp_workspace / "others" / "archive.zip").exists()


def test_file_organizer_skips_directories(temp_workspace):
    from utils.file_organizer import organize_files

    nested_dir = temp_workspace / "nested"
    nested_dir.mkdir()
    create_file(nested_dir / "keep.txt")
    create_file(temp_workspace / "root.log")

    organize_files(temp_workspace, rules={"logs": [".log"]}, default_folder="uncategorized")

    # Nested directory should remain untouched
    assert (nested_dir / "keep.txt").exists()
    assert (temp_workspace / "logs" / "root.log").exists()
