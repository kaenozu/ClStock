"""Trusted market data provider integration tests.

These tests ensure that the application keeps serving real historical
time-series data even if the optional ``yfinance`` dependency is missing.
They also confirm that pseudo-random fallbacks are not invoked when the
configured trusted data sources fail to provide records.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

import pandas as pd
from config.settings import load_from_env


def _load_stock_data_module():
    module_name = "tests.data.real_stock_data_module"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).resolve().parents[2] / "data" / "stock_data.py",
    )
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    sys.modules[module_name] = module
    return module


stock_data_module = _load_stock_data_module()
StockDataProvider = stock_data_module.StockDataProvider
DataFetchError = stock_data_module.DataFetchError


class _NullCache:
    def get(self, key):
        return None

    def set(self, key, value, ttl=None):
        return None


@pytest.fixture
def local_csv_env(tmp_path: Path) -> Iterator[Path]:
    env = {
        "CLSTOCK_MARKET_DATA_PROVIDER": "local_csv",
        "CLSTOCK_MARKET_DATA_LOCAL_CACHE": str(tmp_path),
    }
    load_from_env(env)
    try:
        yield tmp_path
    finally:
        load_from_env({})


def test_real_local_source_used_when_yfinance_missing(local_csv_env: Path, monkeypatch):
    csv_path = local_csv_env / "REAL1.csv"
    frame = pd.DataFrame(
        {
            "Open": [101.0, 102.5, 103.0],
            "High": [102.0, 103.5, 104.0],
            "Low": [100.0, 101.5, 102.4],
            "Close": [101.5, 103.0, 103.2],
            "Volume": [1200, 1300, 1400],
        },
        index=pd.date_range("2024-01-02", periods=3, freq="D"),
    )
    frame.to_csv(csv_path)

    monkeypatch.setattr(stock_data_module, "YFINANCE_AVAILABLE", False, raising=False)
    monkeypatch.setattr(stock_data_module, "yf", None, raising=False)

    provider = StockDataProvider()

    with patch.object(
        stock_data_module, "get_cache", return_value=_NullCache(),
    ), patch.object(provider, "_download_via_yfinance") as mocked_download:
        result = provider.get_stock_data("REAL1", period="1mo")

    assert not result.empty
    pd.testing.assert_index_equal(result.index, frame.index)
    pd.testing.assert_series_equal(
        result["Close"], frame["Close"], check_names=False, check_freq=False,
    )
    assert (result["ActualTicker"] == "REAL1").all()
    mocked_download.assert_not_called()


def test_missing_trusted_source_raises_without_fallback(
    local_csv_env: Path, monkeypatch,
):
    monkeypatch.setattr(stock_data_module, "YFINANCE_AVAILABLE", False, raising=False)
    monkeypatch.setattr(stock_data_module, "yf", None, raising=False)

    provider = StockDataProvider()

    with patch.object(
        stock_data_module, "get_cache", return_value=_NullCache(),
    ), patch.object(provider, "_download_via_yfinance") as mocked_download:
        with pytest.raises(DataFetchError):
            provider.get_stock_data("UNKNOWN", period="1mo")

    mocked_download.assert_not_called()
