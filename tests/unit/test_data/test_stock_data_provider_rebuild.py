"""Unit tests for the reconstructed :mod:`data.stock_data` module.

The tests exercise the public API that other parts of the project rely on
while ensuring we never perform real network or filesystem I/O.  They work in
concert with the fallback ``yf.Ticker`` stub that will be implemented in
``data/stock_data.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List

import pandas as pd
import pytest
from unittest.mock import patch

from data.stock_data import StockDataProvider


class _MemoryCache:
    """Simple in-memory cache used to short-circuit disk access in tests."""

    def __init__(self) -> None:
        self._storage: Dict[str, object] = {}

    def get(self, key: str) -> object | None:
        return self._storage.get(key)

    def set(self, key: str, value: object, ttl: int | None = None) -> bool:
        self._storage[key] = value
        return True


@pytest.fixture()
def history_frame() -> pd.DataFrame:
    """Return a deterministic OHLCV frame used across the tests."""

    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "Open": pd.Series(range(100, 130), dtype="float"),
            "High": pd.Series(range(101, 131), dtype="float"),
            "Low": pd.Series(range(99, 129), dtype="float"),
            "Close": pd.Series(range(100, 130), dtype="float"),
            "Volume": pd.Series([10_000 + i * 100 for i in range(30)], dtype="int"),
        },
        index=idx,
    )


@pytest.fixture()
def ticker_factory(history_frame: pd.DataFrame):
    """Factory returning lightweight ``yfinance.Ticker`` stand-ins."""

    def _factory(symbol: str) -> SimpleNamespace:
        ticker = SimpleNamespace()
        ticker.symbol = symbol
        ticker.info = {
            "shortName": f"{symbol} Corp",
            "marketCap": 123_456_789,
            "forwardPE": 15.0,
            "trailingPE": 16.5,
            "dividendYield": 0.012,
        }
        ticker.fast_info = SimpleNamespace(
            last_price=history_frame["Close"].iloc[-1],
            previous_close=history_frame["Close"].iloc[-2],
            ten_day_average_volume=history_frame["Volume"].tail(10).mean(),
        )

        def _history(period: str = "1y", interval: str = "1d") -> pd.DataFrame:
            ticker.last_history_args = {"period": period, "interval": interval}
            return history_frame.copy()

        ticker.history = _history
        return ticker

    return _factory


def test_get_stock_data_uses_cache(history_frame: pd.DataFrame, ticker_factory) -> None:
    provider = StockDataProvider()
    cache = _MemoryCache()
    ticker_calls: List[str] = []

    def _ticker_side_effect(symbol: str):
        ticker_calls.append(symbol)
        return ticker_factory(symbol)

    with patch("data.stock_data.get_cache", return_value=cache), patch(
        "data.stock_data.yf.Ticker", side_effect=_ticker_side_effect
    ):
        first = provider.get_stock_data("7203", "1mo")
        second = provider.get_stock_data("7203", "1mo")

    assert ticker_calls == ["7203.T"]  # only the first call should touch yfinance
    assert list(first.columns[:5]) == ["Open", "High", "Low", "Close", "Volume"]
    assert {"Symbol", "ActualTicker", "CompanyName"}.issubset(first.columns)
    assert first.equals(second)
    assert first["Symbol"].unique().tolist() == ["7203"]
    assert first["ActualTicker"].iloc[-1] == "7203.T"


def test_calculate_technical_indicators_schema(history_frame: pd.DataFrame) -> None:
    provider = StockDataProvider()
    enriched = provider.calculate_technical_indicators(history_frame)

    required = {"SMA_20", "SMA_50", "RSI", "MACD", "MACD_Signal", "ATR"}
    assert required.issubset(enriched.columns)
    assert enriched.index.equals(history_frame.index)


def test_get_all_stock_symbols_sorted() -> None:
    provider = StockDataProvider()
    symbols = provider.get_all_stock_symbols()

    assert isinstance(symbols, list)
    assert symbols == sorted(symbols)
    assert "7203" in symbols
