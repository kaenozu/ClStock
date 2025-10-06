"""Regression style checks for :mod:`data.stock_data`."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_real_stock_data_module():
    module_path = Path(__file__).resolve().parents[2] / "data" / "stock_data.py"
    spec = importlib.util.spec_from_file_location("real_stock_data", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stock_data_module = _load_real_stock_data_module()
StockDataProvider = stock_data_module.StockDataProvider


def test_stock_data_provider_returns_enriched_frame(monkeypatch):
    provider = StockDataProvider()

    dummy_index = pd.date_range("2024-01-01", periods=5, freq="B")
    dummy_frame = pd.DataFrame(
        {
            "Open": [100 + i for i in range(5)],
            "High": [102 + i for i in range(5)],
            "Low": [99 + i for i in range(5)],
            "Close": [101 + i for i in range(5)],
            "Volume": [100_000 + i for i in range(5)],
        },
        index=dummy_index,
    )

    class _CacheStub:
        def __init__(self):
            self.storage = {}

        def get(self, key):
            return self.storage.get(key)

        def set(self, key, value, ttl=None):
            self.storage[key] = value

    cache = _CacheStub()
    monkeypatch.setattr(stock_data_module, "get_cache", lambda: cache)
    monkeypatch.setattr(provider, "_should_use_local_first", lambda symbol: False)
    monkeypatch.setattr(
        provider,
        "_download_via_yfinance",
        lambda symbol, period: (dummy_frame, f"{symbol}.T"),
    )

    frame = provider.get_stock_data("6758", period="1mo")

    assert isinstance(frame, pd.DataFrame)
    assert not frame.empty
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(frame.columns)
    assert (frame["Symbol"] == "6758").all()
    assert frame.index.is_monotonic_increasing


def test_get_multiple_stocks_skips_failures(monkeypatch):
    provider = StockDataProvider()

    cache = {}

    class _CacheStub:
        def get(self, key):
            return cache.get(key)

        def set(self, key, value, ttl=None):
            cache[key] = value

    monkeypatch.setattr(stock_data_module, "get_cache", lambda: _CacheStub())

    def fallback(symbol: str, period: str = "1y"):
        return (
            pd.DataFrame(
                {
                    "Open": [1.0],
                    "High": [1.5],
                    "Low": [0.5],
                    "Close": [1.2],
                    "Volume": [10],
                },
                index=pd.date_range("2024-01-01", periods=1, freq="D"),
            ),
            f"{symbol}.T",
        )

    monkeypatch.setattr(provider, "_should_use_local_first", lambda symbol: False)
    monkeypatch.setattr(provider, "_download_via_yfinance", fallback)

    def failing_get_stock_data(symbol: str, period: str = "1y"):
        if symbol == "FAIL":
            raise stock_data_module.DataFetchError(symbol, "boom")
        return original(symbol, period)

    original = provider.get_stock_data
    monkeypatch.setattr(provider, "get_stock_data", failing_get_stock_data)

    result = provider.get_multiple_stocks(["6758", "FAIL"], period="1mo")
    assert set(result.keys()) == {"6758"}
