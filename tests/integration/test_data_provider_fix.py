"""Integration checks for the stock data provider fixes."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

# ``tests/conftest.py`` injects a lightweight stub for ``data.stock_data`` to
# accelerate legacy unit tests.  These integration checks exercise the real
# implementation by loading the module directly from its source file.
module_path = Path(__file__).resolve().parents[2] / "data" / "stock_data.py"
spec = spec_from_file_location("data.stock_data_real", module_path)
stock_data_module = module_from_spec(spec)
assert spec and spec.loader  # narrow mypy/pylint concerns
spec.loader.exec_module(stock_data_module)
StockDataProvider = stock_data_module.StockDataProvider


@pytest.fixture(autouse=True)
def stub_yfinance(monkeypatch):
    """Provide deterministic data instead of performing live downloads."""

    def _fake_download(self, symbol: str, period: str):
        index = pd.date_range("2024-01-01", periods=30, freq="B")
        base = 100 + (hash(symbol) % 10)
        frame = pd.DataFrame(
            {
                "Open": np.linspace(base, base + 1, len(index)),
                "High": np.linspace(base + 0.5, base + 1.5, len(index)),
                "Low": np.linspace(base - 0.5, base + 0.5, len(index)),
                "Close": np.linspace(base + 0.25, base + 1.25, len(index)),
                "Volume": np.arange(1, len(index) + 1) * 1000,
            },
            index=index,
        )
        return frame, f"{symbol}.T"

    monkeypatch.setattr(StockDataProvider, "_download_via_yfinance", _fake_download)
    monkeypatch.setattr(
        StockDataProvider, "_should_use_local_first", lambda self, symbol: False,
    )


@pytest.mark.integration
def test_stock_data_provider_returns_data_frame() -> None:
    """Fetching data for a known symbol should return a populated frame."""
    provider = StockDataProvider()
    frame = provider.get_stock_data("6758", period="1mo")

    assert isinstance(frame, pd.DataFrame)
    assert not frame.empty
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(frame.columns)
    assert frame.index.is_monotonic_increasing
    assert frame["Symbol"].iloc[0] == "6758"


@pytest.mark.integration
def test_multiple_symbols_fetch_uses_cache_for_consistency() -> None:
    """Repeated calls should rely on cached data and keep metadata stable."""
    provider = StockDataProvider()
    symbols = ["7203", "8306"]

    first_batch = provider.get_multiple_stocks(symbols, period="1mo")
    second_batch = provider.get_multiple_stocks(symbols, period="1mo")

    assert set(first_batch.keys()) == set(symbols)
    assert set(second_batch.keys()) == set(symbols)

    for symbol in symbols:
        first = first_batch[symbol]
        second = second_batch[symbol]

        assert not first.empty
        assert not second.empty
        assert first["CompanyName"].iloc[0] == second["CompanyName"].iloc[0]
        assert first.index[0] == second.index[0]


@pytest.mark.integration
def test_provider_generates_deterministic_history() -> None:
    """The pseudo-random fallback should produce deterministic results."""
    provider = StockDataProvider()
    history_a = provider.get_stock_data("8031", period="1mo")
    history_b = provider.get_stock_data("8031", period="1mo")

    assert history_a.equals(history_b)
    assert history_a["ActualTicker"].iloc[0] == "8031.T"
    latest_timestamp = history_a.index[-1]
    if getattr(latest_timestamp, "tzinfo", None) is None:
        latest_timestamp = latest_timestamp.tz_localize(UTC)
    assert latest_timestamp <= datetime.now(UTC)
