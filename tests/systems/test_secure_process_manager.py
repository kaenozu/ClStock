import io
import types
import time
import subprocess
from pathlib import Path

import pytest

from systems.process_manager_fixed import (
    SecureProcessManager,
    ProcessStatus,
    OutputReader,
)


class DummyPipe:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        return ""


class DummyProcess:
    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = 1234
        self.stdout = DummyPipe(["stdout line\n", ""])
        self.stderr = DummyPipe(["stderr line\n", ""])
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class TimeoutProcess(DummyProcess):
    def __init__(self, command, **kwargs):
        super().__init__(command, **kwargs)
        self.wait_called_without_timeout = False

    def wait(self, timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired(cmd=self.command, timeout=timeout)
        self.wait_called_without_timeout = True
        return 0


@pytest.fixture
def manager(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(Path, "cwd", staticmethod(lambda: tmp_path))
    mgr = SecureProcessManager(config_file=str(config_path))
    return mgr


def test_sanitize_and_validate(manager):
    sanitized = manager._sanitize_input("abc; rm &xyz")
    assert sanitized == "abc rm xyz"

    assert manager._validate_command(["python", "script.py"]) is True
    assert manager._validate_command(["python", "bad;rm"]) is False
    assert manager._validate_command(["rm", "-rf", "/"]) is False


def test_start_service_with_safe_command(manager, monkeypatch):
    popen_calls = {}

    def fake_popen(command, **kwargs):
        popen_calls["command"] = command
        popen_calls["kwargs"] = kwargs
        return DummyProcess(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    process_info = manager.processes["dashboard"]
    assert manager.start_service("dashboard", symbol="7203") is True
    assert popen_calls["command"][-1] == "7203"
    assert process_info.status == ProcessStatus.RUNNING


def test_start_service_rejects_invalid_symbol(manager, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: DummyProcess(args[0], **kwargs))

    assert manager.start_service("dashboard", symbol="72A3") is False
    assert manager.processes["dashboard"].status != ProcessStatus.RUNNING


def test_start_service_rejects_invalid_command(manager, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: DummyProcess(args[0], **kwargs))
    monkeypatch.setattr(manager, "_validate_command", lambda cmd: False)

    assert manager.start_service("dashboard") is False
    assert manager.processes["dashboard"].status == ProcessStatus.FAILED


def test_start_service_rejects_dangerous_symbol(manager, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: DummyProcess(args[0], **kwargs))

    assert manager.start_service("dashboard", symbol="7203;rm") is False


def test_stop_service_with_timeout(manager, monkeypatch):
    popen = TimeoutProcess(["python", "app.py"])
    process_info = manager.processes["dashboard"]
    process_info.process = popen
    process_info.status = ProcessStatus.RUNNING
    process_info.output_reader = types.SimpleNamespace(stop=lambda: None)

    assert manager.stop_service("dashboard") is True
    assert popen.killed is True
    assert popen.wait_called_without_timeout is True
    assert process_info.status == ProcessStatus.STOPPED


def test_execute_safe_command(monkeypatch):
    manager = SecureProcessManager()

    def fake_run(command, capture_output, text, timeout, shell):
        assert command == ["echo", "hello"]
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    success, stdout, stderr = manager.execute_safe_command(["echo", "hello"])
    assert success is True
    assert stdout == "ok"
    assert stderr == ""

    success, stdout, stderr = manager.execute_safe_command(["echo", "bad;rm"])
    assert success is False
    assert stderr == "安全でないコマンドです"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=[], timeout=1)))
    success, stdout, stderr = manager.execute_safe_command(["python", "script.py"], timeout=1)
    assert success is False
    assert stderr == "コマンドタイムアウト"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    success, stdout, stderr = manager.execute_safe_command(["python", "script.py"])
    assert success is False
    assert "boom" in stderr


def test_predict_stock_safe(monkeypatch):
    manager = SecureProcessManager()

    responses = iter([
        types.SimpleNamespace(returncode=0, stdout="done", stderr=""),
        types.SimpleNamespace(returncode=1, stdout="", stderr="fail"),
    ])

    def fake_run(command, capture_output, text, timeout, shell):
        return next(responses)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = manager.predict_stock_safe("7203")
    assert result == {"status": "success", "output": "done"}

    result = manager.predict_stock_safe("7203")
    assert result == {"status": "error", "error": "fail"}

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd=[], timeout=1)))
    result = manager.predict_stock_safe("7203")
    assert result == {"status": "error", "error": "コマンドタイムアウト"}

    assert manager.predict_stock_safe("ABCD") is None


def test_output_reader_queue_handling(tmp_path):
    pipe = io.StringIO("line1\nline2\n")
    reader = OutputReader(pipe, log_file=str(tmp_path / "out.log"), pipe_name="stdout")
    reader.start()

    time.sleep(0.1)
    reader.stop()
    reader.join(timeout=1)

    lines = reader.get_recent_lines()
    assert "line1" in lines[0]
    assert reader.is_alive() is False

