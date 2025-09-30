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
