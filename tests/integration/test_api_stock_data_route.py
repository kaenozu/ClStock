"""Integration test hitting the stock data route implementation directly."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import pandas as pd
from api import endpoints


class _MemoryCache:
    def __init__(self) -> None:
        self._storage = {}

    def get(self, key):
        return self._storage.get(key)

    def set(self, key, value, ttl=None):
        self._storage[key] = value
        return True


def _build_history() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    base = pd.Series(range(120, 140), dtype="float", index=idx)
    # introduce a small dip to avoid flat RSI calculations
    base.iloc[10:12] = base.iloc[10:12] - 2
    return pd.DataFrame(
        {
            "Open": base + 0.5,
            "High": base + 1.0,
            "Low": base - 1.0,
            "Close": base,
            "Volume": pd.Series(
                [20_000 + i * 50 for i in range(20)],
                dtype="int",
                index=idx,
            ),
        },
    )


def _ticker_stub(symbol: str) -> SimpleNamespace:
    frame = _build_history()
    ticker = SimpleNamespace()
    ticker.symbol = symbol
    ticker.info = {
        "shortName": f"{symbol} Incorporated",
        "marketCap": 987_654_321,
        "forwardPE": 14.2,
        "trailingPE": 16.1,
        "dividendYield": 0.015,
        "beta": 1.05,
    }
    ticker.fast_info = SimpleNamespace(
        last_price=float(frame["Close"].iloc[-1]),
        previous_close=float(frame["Close"].iloc[-2]),
        ten_day_average_volume=float(frame["Volume"].tail(10).mean()),
    )

    def _history(period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        ticker.last_request = {"period": period, "interval": interval}
        return frame.copy()

    ticker.history = _history
    return ticker


@pytest.mark.asyncio
async def test_stock_route_works_with_real_provider() -> None:
    with patch("data.stock_data.get_cache", return_value=_MemoryCache()), patch(
        "data.stock_data.yf.Ticker",
        side_effect=_ticker_stub,
    ):
        payload = await endpoints.get_stock_data("7203", "1mo")

    assert payload["symbol"] == "7203"
    assert payload["company_name"]
    assert not math.isnan(payload["current_price"])
    assert "technical_indicators" in payload
    assert "financial_metrics" in payload
