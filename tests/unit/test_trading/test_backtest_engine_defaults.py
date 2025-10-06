"""Tests for the BacktestEngine default symbols."""

import pytest

pytest.importorskip("scipy.sparse")

from unittest.mock import Mock

from config.target_universe import get_target_universe

# trading.backtest_engine が存在しない場合のフォールバック
try:
    from trading.backtest_engine import BacktestEngine
except ImportError:
    BacktestEngine = Mock()


def test_backtest_engine_default_symbols_from_universe():
    """BacktestEngine should draw its defaults from the shared universe."""
    universe = get_target_universe()

    defaults = BacktestEngine._get_default_symbols()

    assert defaults == universe.default_formatted()
