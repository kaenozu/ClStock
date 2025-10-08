import io
from pathlib import Path
from unittest.mock import Mock

import pytest

from systems.full_auto.script_service import DataRetrievalScriptService


@pytest.fixture
def dummy_logger():
    class _Logger:
        def __init__(self):
            self.messages = []

        def info(self, msg, *args, **kwargs):
            self.messages.append(("info", msg % args if args else msg))

        def warning(self, msg, *args, **kwargs):
            self.messages.append(("warning", msg % args if args else msg))

        def error(self, msg, *args, **kwargs):
            self.messages.append(("error", msg % args if args else msg))

        def debug(self, msg, *args, **kwargs):
            self.messages.append(("debug", msg % args if args else msg))

    return _Logger()


@pytest.fixture
def printer():
    buffer = io.StringIO()

    def _printer(message: str) -> None:
        buffer.write(message + "\n")

    _printer.buffer = buffer  # type: ignore[attr-defined]
    return _printer


def test_generate_skips_when_no_failed_symbols(tmp_path: Path, dummy_logger, printer):
    script_generator = Mock()
    service = DataRetrievalScriptService(
        output_dir=tmp_path,
        script_generator=script_generator,
        logger=dummy_logger,
        printer=printer,
    )

    service.generate([])

    assert not list(tmp_path.iterdir())
    script_generator.assert_not_called()


def test_generate_writes_script_when_failed_symbols_present(tmp_path: Path, dummy_logger, printer):
    script_content = "print('hello from script')\n"
    script_generator = Mock(return_value=script_content)
    service = DataRetrievalScriptService(
        output_dir=tmp_path,
        script_generator=script_generator,
        logger=dummy_logger,
        printer=printer,
    )

    service.generate(["7203.T", "6758.T"])

    generated_files = list(tmp_path.iterdir())
    assert len(generated_files) == 1
    script_path = generated_files[0]
    assert script_path.name.endswith(".py")
    assert script_path.read_text(encoding="utf-8-sig") == script_content
    script_generator.assert_called_once_with(["7203.T", "6758.T"])
