import tempfile
from pathlib import Path

import pytest

from utils import temp_cleanup


class DummyShutdownManager:
    def __init__(self):
        self.handlers = []

    def register_shutdown_handler(self, handler):
        self.handlers.append(handler)


@pytest.fixture
def cleanup_manager(monkeypatch):
    dummy_manager = DummyShutdownManager()
    monkeypatch.setattr(temp_cleanup, "get_shutdown_manager", lambda: dummy_manager)
    return temp_cleanup.TempFileCleanup()


def test_cleanup_all_removes_missing_paths_from_registry(cleanup_manager):
    missing_file = Path(tempfile.gettempdir()) / "nonexistent_temp_file.txt"
    cleanup_manager.register_temp_file(str(missing_file))

    cleanup_manager.cleanup_all()

    assert cleanup_manager.temp_files == []


def test_cleanup_all_removes_directories_and_files(cleanup_manager, tmp_path):
    temp_file = tmp_path / "temp.txt"
    temp_file.write_text("data")
    temp_dir = tmp_path / "tempdir"
    temp_dir.mkdir()
    (temp_dir / "nested.txt").write_text("nested")

    cleanup_manager.register_temp_file(str(temp_file))
    cleanup_manager.register_temp_dir(str(temp_dir))

    cleanup_manager.cleanup_all()

    assert cleanup_manager.temp_files == []
    assert cleanup_manager.temp_dirs == []
    assert not temp_file.exists()
    assert not temp_dir.exists()


def test_cleanup_unnecessary_files_removes_non_whitelisted(cleanup_manager, tmp_path):
    base_dir = tmp_path / "artifacts"
    base_dir.mkdir()

    keep_file = base_dir / "keep.txt"
    keep_file.write_text("keep")
    extra_file = base_dir / "remove.log"
    extra_file.write_text("remove")
    nested_dir = base_dir / "subdir"
    nested_dir.mkdir()
    (nested_dir / "nested.txt").write_text("nested")

    removed = cleanup_manager.cleanup_unnecessary_files(
        str(base_dir), required_entries={"keep.txt", "subdir"}
    )

    assert removed == [str(extra_file)]
    assert keep_file.exists()
    assert not extra_file.exists()
    assert nested_dir.exists()


def test_cleanup_unnecessary_files_handles_missing_directory(cleanup_manager, tmp_path):
    missing_dir = tmp_path / "does_not_exist"

    removed = cleanup_manager.cleanup_unnecessary_files(
        str(missing_dir), required_entries={"keep.txt"}
    )

    assert removed == []
