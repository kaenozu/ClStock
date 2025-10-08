import builtins
import sys
from types import SimpleNamespace

import pytest

from menu import (
    optimization_history_menu,
    rollback_to_record,
    show_history_list,
    show_history_statistics,
)


@pytest.fixture
def history_manager_stub(monkeypatch):
    class Stub:
        def __init__(self):
            self.records = []
            self.rollback_result = True
            self.stats = {
                "total_runs": 0,
                "average_accuracy": 0.0,
                "max_accuracy": 0.0,
                "latest_run": "N/A",
            }
            self.last_record_id = None

        def list_optimization_records(self):
            return self.records

        def rollback_to_configuration(self, record_id):
            self.last_record_id = record_id
            return self.rollback_result

        def get_optimization_statistics(self):
            return self.stats

    stub = Stub()
    monkeypatch.setitem(
        sys.modules,
        "systems.optimization_history",
        SimpleNamespace(OptimizationHistoryManager=lambda: stub),
    )
    monkeypatch.setattr("menu.clear_screen", lambda: None)
    monkeypatch.setattr("menu.time.sleep", lambda *_args, **_kwargs: None)
    return stub


@pytest.fixture
def no_wait_input(monkeypatch):
    def _set_inputs(responses):
        iterator = iter(responses)
        monkeypatch.setattr(builtins, "input", lambda *_args: next(iterator))

    return _set_inputs


def test_optimization_history_menu_shows_history_list(
    history_manager_stub,
    no_wait_input,
    capsys,
):
    history_manager_stub.records = [
        {
            "record_id": "rec-001",
            "timestamp": "2024-05-01 12:00",
            "accuracy": 91.2,
            "stocks": ["AAA", "BBB"],
        },
    ]
    no_wait_input(["1", ""])  # menu choice, enter to continue

    optimization_history_menu()
    output = capsys.readouterr().out

    assert "【最適化履歴管理】" in output
    assert "rec-001" in output
    assert "【最適化履歴一覧】" in output


def test_optimization_history_menu_rolls_back_successfully(
    history_manager_stub,
    no_wait_input,
    capsys,
):
    history_manager_stub.rollback_result = True
    no_wait_input(["2", "target-record", ""])  # menu choice, record id, enter

    optimization_history_menu()
    output = capsys.readouterr().out

    assert history_manager_stub.last_record_id == "target-record"
    assert "復元が完了しました" in output


def test_optimization_history_menu_shows_statistics(
    history_manager_stub,
    no_wait_input,
    capsys,
):
    history_manager_stub.stats = {
        "total_runs": 5,
        "average_accuracy": 88.5,
        "max_accuracy": 95.1,
        "latest_run": "2024-05-02 09:00",
    }
    no_wait_input(["3", ""])  # menu choice, enter

    optimization_history_menu()
    output = capsys.readouterr().out

    assert "【最適化履歴統計】" in output
    assert "総実行回数: 5 回" in output
    assert "平均精度: 88.5%" in output
    assert "最新実行: 2024-05-02 09:00" in output


def test_optimization_history_menu_handles_invalid_choice(
    history_manager_stub,
    no_wait_input,
    capsys,
):
    no_wait_input(["9"])  # invalid menu choice

    optimization_history_menu()
    output = capsys.readouterr().out

    assert "無効な選択です" in output


def test_show_history_list_with_records(history_manager_stub, no_wait_input, capsys):
    history_manager_stub.records = [
        {
            "record_id": "rec-001",
            "timestamp": "2024-05-01 12:00",
            "accuracy": 91.2,
            "stocks": ["AAA", "BBB"],
        },
    ]
    no_wait_input([""])

    show_history_list()
    output = capsys.readouterr().out

    assert "【最適化履歴一覧】" in output
    assert "rec-001" in output
    assert "2 銘柄" in output


def test_show_history_list_without_records(history_manager_stub, no_wait_input, capsys):
    history_manager_stub.records = []
    no_wait_input([""])

    show_history_list()
    output = capsys.readouterr().out

    assert "保存された最適化履歴がありません" in output


def test_rollback_to_record_success(history_manager_stub, no_wait_input, capsys):
    history_manager_stub.rollback_result = True
    history_manager_stub.last_record_id = None
    no_wait_input([""])

    rollback_to_record("rec-001")
    output = capsys.readouterr().out

    assert history_manager_stub.last_record_id == "rec-001"
    assert "復元が完了しました" in output


def test_rollback_to_record_failure(history_manager_stub, no_wait_input, capsys):
    history_manager_stub.rollback_result = False
    history_manager_stub.last_record_id = None
    no_wait_input([""])

    rollback_to_record("rec-404")
    output = capsys.readouterr().out

    assert history_manager_stub.last_record_id == "rec-404"
    assert "が見つかりません" in output


def test_show_history_statistics(history_manager_stub, no_wait_input, capsys):
    history_manager_stub.stats = {
        "total_runs": 7,
        "average_accuracy": 90.3,
        "max_accuracy": 97.8,
        "latest_run": "2024-05-03 15:30",
    }
    no_wait_input([""])

    show_history_statistics()
    output = capsys.readouterr().out

    assert "【最適化履歴統計】" in output
    assert "総実行回数: 7 回" in output
    assert "平均精度: 90.3%" in output
    assert "最高精度: 97.8%" in output
    assert "最新実行: 2024-05-03 15:30" in output
