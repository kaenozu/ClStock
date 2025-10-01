import click
from click.testing import CliRunner
import pytest

import clstock_cli


class DummyManager:
    def __init__(self, start_ok=True, stop_ok=True):
        self._start_ok = start_ok
        self._stop_ok = stop_ok

    def start_service(self, name):
        return self._start_ok

    def list_services(self):
        return []

    def stop_service(self, name, force=False):
        return self._stop_ok


@pytest.fixture
def runner():
    return CliRunner()


def test_service_start_failure_raises_click_exception(monkeypatch, runner):
    dummy_manager = DummyManager(start_ok=False)
    monkeypatch.setattr(clstock_cli, "get_process_manager", lambda: dummy_manager)

    with pytest.raises(click.ClickException) as exc_info:
        runner.invoke(
            clstock_cli.cli,
            ["service", "start", "demo"],
            catch_exceptions=False,
            standalone_mode=False,
        )

    assert exc_info.value.exit_code == 1
    assert "[失敗] サービス開始失敗" in str(exc_info.value)


def test_service_stop_failure_raises_click_exception(monkeypatch, runner):
    dummy_manager = DummyManager(stop_ok=False)
    monkeypatch.setattr(clstock_cli, "get_process_manager", lambda: dummy_manager)

    with pytest.raises(click.ClickException) as exc_info:
        runner.invoke(
            clstock_cli.cli,
            ["service", "stop", "demo"],
            catch_exceptions=False,
            standalone_mode=False,
        )

    assert exc_info.value.exit_code == 1
    assert "[失敗] サービス停止失敗" in str(exc_info.value)


def test_predict_invalid_symbol_raises_bad_parameter(runner):
    with pytest.raises(click.BadParameter) as exc_info:
        runner.invoke(
            clstock_cli.cli,
            ["system", "predict", "--symbol", "abc"],
            catch_exceptions=False,
            standalone_mode=False,
        )

    assert exc_info.value.exit_code == 2
    assert "銘柄コードは数値のみ有効です" in str(exc_info.value)
