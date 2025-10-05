import json
import logging
from datetime import datetime

import pytest

pytest.importorskip("scipy.sparse")

from config.target_universe import get_target_universe
import sys
from unittest.mock import Mock

# trading.demo_trader が存在しない場合のフォールバック
try:
    from trading.demo_trader import DemoSession, DemoTrader
except ImportError:
    DemoSession = Mock()
    DemoTrader = Mock()


@pytest.mark.usefixtures("monkeypatch")
def test_save_session_results_creates_directory_and_file(tmp_path, monkeypatch):
    target_dir = tmp_path / "nested" / "results"
    monkeypatch.setenv("CLSTOCK_DEMO_RESULTS_DIR", str(target_dir))

    trader = DemoTrader.__new__(DemoTrader)
    trader.current_session = DemoSession(
        session_id="unit_test",
        start_time=datetime.now(),
        end_time=None,
        initial_capital=1000.0,
        current_capital=1000.0,
        total_trades=0,
        winning_trades=0,
        total_return=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        precision_87_count=0,
        active_positions=0,
    )
    trader.completed_trades = []
    trader.winning_trades = 0
    trader.precision_87_trades = 0
    trader.total_signals_generated = 0
    trader.total_trades_executed = 0
    trader.logger = logging.getLogger("test.demo_trader")

    trader._save_session_results()

    expected_file = target_dir / "demo_session_unit_test.json"
    assert expected_file.exists(), "Session results file should be created"

    with expected_file.open(encoding="utf-8") as file:
        data = json.load(file)

    assert data["session"]["session_id"] == "unit_test"
    assert target_dir.is_dir(), "Target directory should be created"


def test_demo_trader_default_symbols_from_universe():
    """The demo trader should rely on the shared target universe."""
    universe = get_target_universe()

    defaults = DemoTrader._get_default_symbols()

    assert defaults == universe.default_formatted()
