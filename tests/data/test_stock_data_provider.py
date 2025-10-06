import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "data" / "stock_data.py"

spec = importlib.util.spec_from_file_location("stock_data_under_test", MODULE_PATH)
stock_data = importlib.util.module_from_spec(spec)
sys.modules["stock_data_under_test"] = stock_data
assert spec.loader is not None
spec.loader.exec_module(stock_data)

StockDataProvider = stock_data.StockDataProvider
from utils.cache import clear_cache, get_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_cache()
    yield
    clear_cache()


def test_get_financial_metrics_returns_defaults_when_yfinance_missing(monkeypatch):
    monkeypatch.setattr(stock_data, "YFINANCE_AVAILABLE", False, raising=False)
    monkeypatch.setattr(stock_data, "yf", None, raising=False)

    provider = StockDataProvider()
    provider.jp_stock_codes = {"TEST": "Test Corp"}

    call_count = {"count": 0}

    def fail_helper(self, ticker, metrics):
        call_count["count"] += 1
        raise AssertionError(
            "helper should not be invoked when yfinance is unavailable",
        )

    monkeypatch.setattr(
        StockDataProvider,
        "_fetch_financial_metrics_via_yfinance",
        fail_helper,
        raising=False,
    )

    metrics = provider.get_financial_metrics("TEST")

    assert metrics["symbol"] == "TEST"
    assert metrics["company_name"] == "Test Corp"
    assert metrics["market_cap"] is None
    assert metrics["pe_ratio"] is None
    assert metrics["actual_ticker"] is None
    assert call_count["count"] == 0

    cache_key = "financial::TEST"
    cached = get_cache().get(cache_key)
    assert cached == metrics

    metrics_again = provider.get_financial_metrics("TEST")
    assert metrics_again == metrics


def test_get_financial_metrics_uses_yfinance_helper_when_available(monkeypatch):
    monkeypatch.setattr(stock_data, "YFINANCE_AVAILABLE", True, raising=False)

    provider = StockDataProvider()
    provider.jp_stock_codes = {"TEST": "Test Corp"}

    calls = []

    def fake_helper(self, ticker, metrics):
        calls.append(ticker)
        metrics.update(
            {
                "market_cap": 123,
                "pe_ratio": 45.6,
                "dividend_yield": 0.01,
                "beta": 1.2,
                "last_price": 100.5,
                "previous_close": 99.5,
                "ten_day_average_volume": 1000,
                "actual_ticker": ticker,
            },
        )
        return True

    monkeypatch.setattr(
        StockDataProvider,
        "_fetch_financial_metrics_via_yfinance",
        fake_helper,
        raising=False,
    )

    metrics = provider.get_financial_metrics("TEST")

    assert calls == ["TEST"]
    assert metrics["market_cap"] == 123
    assert metrics["pe_ratio"] == 45.6
    assert metrics["dividend_yield"] == 0.01
    assert metrics["beta"] == 1.2
    assert metrics["last_price"] == 100.5
    assert metrics["previous_close"] == 99.5
    assert metrics["ten_day_average_volume"] == 1000
    assert metrics["actual_ticker"] == "TEST"

    metrics_again = provider.get_financial_metrics("TEST")
    assert metrics_again == metrics
    assert len(calls) == 1
