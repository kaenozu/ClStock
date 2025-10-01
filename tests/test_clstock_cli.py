import sys
from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner

import clstock_cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def reset_logger(monkeypatch):
    """Ensure logger calls can be observed without side effects."""
    logger = Mock()
    monkeypatch.setattr(clstock_cli, "logger", logger)
    return logger


def _patch_manager(monkeypatch, **behaviors):
    manager = Mock()
    for name, value in behaviors.items():
        setattr(manager, name, value)
    monkeypatch.setattr(clstock_cli, "get_process_manager", lambda: manager)
    return manager


def test_service_start_failure_raises_click_exception(runner, monkeypatch):
    _patch_manager(
        monkeypatch,
        start_service=Mock(return_value=False),
        list_services=Mock(return_value=[]),
    )

    result = runner.invoke(
        clstock_cli.cli, ["service", "start", "demo"], standalone_mode=False
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, click.ClickException)
    assert "サービス開始失敗" in result.exception.format_message()


def test_service_stop_failure_raises_click_exception(runner, monkeypatch):
    _patch_manager(
        monkeypatch,
        stop_service=Mock(return_value=False),
        list_services=Mock(return_value=[]),
    )

    result = runner.invoke(
        clstock_cli.cli, ["service", "stop", "demo"], standalone_mode=False
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, click.ClickException)
    assert "サービス停止失敗" in result.exception.format_message()


def test_predict_rejects_non_digit_symbol(runner):
    result = runner.invoke(
        clstock_cli.cli,
        ["system", "predict", "--symbol", "abc"],
        standalone_mode=False,
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, click.BadParameter)
    assert "銘柄コード" in result.exception.format_message()


def test_predict_internal_error_is_wrapped(runner, monkeypatch):
    class DummySystem:
        def predict_with_87_precision(self, symbol):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "models_new.precision.precision_87_system", Mock())
    module = sys.modules["models_new.precision.precision_87_system"]
    module.Precision87BreakthroughSystem = Mock(return_value=DummySystem())

    result = runner.invoke(
        clstock_cli.cli,
        ["system", "predict", "--symbol", "7203"],
        standalone_mode=False,
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, click.ClickException)
    assert "予測実行エラー" in result.exception.format_message()


def test_service_start_success_returns_cleanly(runner, monkeypatch):
    _patch_manager(
        monkeypatch,
        start_service=Mock(return_value=True),
        list_services=Mock(return_value=[]),
    )

    result = runner.invoke(clstock_cli.cli, ["service", "start", "demo"])

    assert result.exit_code == 0
    assert result.exception is None
    assert "サービス開始" in result.output
