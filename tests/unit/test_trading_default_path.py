"""Tests for TradeRecorder default database path handling."""

import importlib
from pathlib import Path

import pytest


@pytest.mark.usefixtures("tmp_path")
def test_trade_recorder_uses_home_directory_for_default_path(monkeypatch, tmp_path):
    """TradeRecorder should place its default database inside the user's home directory."""
    module = importlib.import_module("trading.trade_recorder")
    module = importlib.reload(module)

    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)

    recorder = module.TradeRecorder()

    expected = tmp_path / "gemini-desktop" / "ClStock" / "data" / "trading_records.db"
    assert Path(recorder.db_path) == expected
    assert expected.parent.exists(), "Default database directory should be created automatically."
