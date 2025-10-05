import pandas as pd
import importlib.util
import pickle
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def _load_stock_data_module():
    module_name = "tests.real_stock_data"
    if module_name in sys.modules:
        return sys.modules[module_name]

    if "joblib" not in sys.modules:
        joblib_stub = types.ModuleType("joblib")

        def _dump(obj, file):
            pickle.dump(obj, file)

        def _load(file):
            return pickle.load(file)

        joblib_stub.dump = _dump
        joblib_stub.load = _load
        sys.modules["joblib"] = joblib_stub

    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).resolve().parents[3] / "data" / "stock_data.py"
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
    def __init__(self):
        self.storage = {}

    def get(self, key):
        return None

    def set(self, key, value, ttl=None):
        self.storage[key] = value


class TestStockDataProvider:
    """StockDataProvider の挙動を検証するユニットテスト"""

    def test_init(self):
        provider = StockDataProvider()
        assert len(provider.jp_stock_codes) >= 10
        assert "7203" in provider.jp_stock_codes
        assert provider.jp_stock_codes["7203"]

    def test_get_all_stock_symbols(self):
        provider = StockDataProvider()
        symbols = provider.get_all_stock_symbols()
        assert len(symbols) >= 10
        assert "7203" in symbols
        assert "6758" in symbols

    def test_get_stock_data_enriches_columns(self):
        provider = StockDataProvider()
        dummy_df = pd.DataFrame(
            {"Close": [100.0], "Volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        with patch.object(stock_data_module, "get_cache", return_value=_NullCache()), \
             patch.object(provider, "_should_use_local_first", return_value=False), \
             patch.object(provider, "_load_first_available_csv", return_value=None), \
             patch.object(provider, "_download_via_yfinance", return_value=(dummy_df.copy(), "TEST1.T")):
            data = provider.get_stock_data("TEST1", "1mo")
            assert not data.empty
            assert data["Symbol"].iloc[0] == "TEST1"
            assert data["ActualTicker"].iloc[0] == "TEST1.T"

    def test_get_stock_data_invalid_symbol_raises(self):
        provider = StockDataProvider()
        empty_df = pd.DataFrame()
        with patch.object(stock_data_module, "get_cache", return_value=_NullCache()), \
             patch.object(provider, "_should_use_local_first", return_value=False), \
             patch.object(provider, "_load_first_available_csv", return_value=None), \
             patch.object(provider, "_download_via_yfinance", return_value=(empty_df, None)):
            with pytest.raises(DataFetchError):
                provider.get_stock_data("TEST_INVALID", "1mo")

    def test_get_stock_data_prefers_local_when_available(self):
        provider = StockDataProvider()
        dummy_df = pd.DataFrame(
            {"Close": [100.0], "Volume": [1000]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        with patch.object(stock_data_module, "get_cache", return_value=_NullCache()), \
             patch.object(provider, "_should_use_local_first", return_value=True), \
             patch.object(provider, "_load_first_available_csv", return_value=(dummy_df.copy(), "local")), \
             patch.object(provider, "_download_via_yfinance") as mock_download:
            data = provider.get_stock_data("TEST2", "1mo")
            assert not mock_download.called
            assert data["Symbol"].iloc[0] == "TEST2"
            assert data["ActualTicker"].iloc[0] == "local"

    def test_get_stock_data_raises_when_no_sources(self):
        provider = StockDataProvider()
        empty_df = pd.DataFrame()
        with patch.object(stock_data_module, "get_cache", return_value=_NullCache()), \
             patch.object(provider, "_should_use_local_first", return_value=False), \
             patch.object(provider, "_load_first_available_csv", return_value=None), \
             patch.object(provider, "_download_via_yfinance", return_value=(empty_df, None)):
            with pytest.raises(DataFetchError):
                provider.get_stock_data("TEST3", "1mo")

    def test_calculate_technical_indicators(self, mock_stock_data):
        provider = StockDataProvider()
        result = provider.calculate_technical_indicators(mock_stock_data)
        assert {"SMA_20", "SMA_50", "RSI", "MACD", "ATR"}.issubset(result.columns)

    def test_calculate_technical_indicators_empty_data(self):
        provider = StockDataProvider()
        empty_data = pd.DataFrame()
        result = provider.calculate_technical_indicators(empty_data)
        assert result.empty

    def test_get_financial_metrics_with_mock(self, mock_yfinance):
        provider = StockDataProvider()
        with patch("yfinance.Ticker", return_value=mock_yfinance):
            metrics = provider.get_financial_metrics("7203")
            required_keys = {"symbol", "company_name", "market_cap", "pe_ratio"}
            assert required_keys.issubset(metrics.keys())
            assert metrics["symbol"] == "7203"

    def test_get_financial_metrics_returns_defaults_when_yfinance_unavailable(
        self, monkeypatch
    ):
        provider = StockDataProvider()
        monkeypatch.setattr(stock_data_module, "YFINANCE_AVAILABLE", False)

        class _SentinelYF:
            def __init__(self):
                self.called = False

            def Ticker(self, *_args, **_kwargs):
                self.called = True
                raise AssertionError("yfinance should not be accessed when unavailable")

        sentinel = _SentinelYF()
        monkeypatch.setattr(stock_data_module, "yf", sentinel)
        monkeypatch.setattr(stock_data_module, "get_cache", lambda: _NullCache())

        metrics = provider.get_financial_metrics("7203")

        assert metrics == {
            "symbol": "7203",
            "company_name": provider.jp_stock_codes.get("7203", "7203"),
            "market_cap": None,
            "pe_ratio": None,
            "dividend_yield": None,
            "beta": None,
            "last_price": None,
            "previous_close": None,
            "ten_day_average_volume": None,
            "actual_ticker": None,
        }
        assert not sentinel.called

    def test_get_multiple_stocks(self, mock_yfinance):
        provider = StockDataProvider()
        dummy_yf = types.SimpleNamespace(Ticker=lambda ticker: mock_yfinance)
        with patch.object(stock_data_module, "YFINANCE_AVAILABLE", True), \
             patch.object(stock_data_module, "yf", dummy_yf):
            symbols = ["7203", "6758"]
            result = provider.get_multiple_stocks(symbols, "1mo")
            assert len(result) == 2
            assert "7203" in result
            assert "6758" in result

    def test_get_dividend_data_returns_defaults_when_yfinance_unavailable(
        self, monkeypatch
    ):
        provider = StockDataProvider()
        monkeypatch.setattr(stock_data_module, "YFINANCE_AVAILABLE", False)

        class _SentinelYF:
            def __init__(self):
                self.called = False

            def Ticker(self, *_args, **_kwargs):
                self.called = True
                raise AssertionError("yfinance should not be accessed when unavailable")

        sentinel = _SentinelYF()
        monkeypatch.setattr(stock_data_module, "yf", sentinel)
        monkeypatch.setattr(stock_data_module, "get_cache", lambda: _NullCache())

        dividend_data = provider.get_dividend_data("7203")

        assert dividend_data == {
            "symbol": "7203",
            "dividend_rate": None,
            "dividend_yield": None,
            "ex_dividend_date": None,
            "actual_ticker": None,
        }
        assert not sentinel.called

    def test_get_news_data_returns_empty_when_yfinance_unavailable(
        self, monkeypatch
    ):
        provider = StockDataProvider()
        monkeypatch.setattr(stock_data_module, "YFINANCE_AVAILABLE", False)

        class _SentinelYF:
            def __init__(self):
                self.called = False

            def Ticker(self, *_args, **_kwargs):
                self.called = True
                raise AssertionError("yfinance should not be accessed when unavailable")

        sentinel = _SentinelYF()
        monkeypatch.setattr(stock_data_module, "yf", sentinel)
        monkeypatch.setattr(stock_data_module, "get_cache", lambda: _NullCache())

        news_data = provider.get_news_data("7203")

        assert news_data == []
        assert not sentinel.called

    def test_ticker_formats_for_japanese_stock(self):
        provider = StockDataProvider()
        formats = provider._ticker_formats("7203")
        assert formats[0] == "7203"
        assert formats[1:] == ["7203.T", "7203.TO"]

    def test_ticker_formats_preserve_suffix_priority(self):
        provider = StockDataProvider()
        formats = provider._ticker_formats("6758.to")
        assert formats == ["6758.TO", "6758.T", "6758"]

    def test_prepare_history_frame_defaults_to_first_ticker_format(self):
        provider = StockDataProvider()
        base_df = pd.DataFrame(
            {"Close": [100.0], "Open": [99.0], "High": [101.0], "Low": [98.5], "Volume": [1500]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        prepared = provider._prepare_history_frame(base_df, "6758.to", actual_ticker=None)
        assert (prepared["ActualTicker"] == "6758.TO").all()

    def test_load_first_available_csv_prefers_suffix(self, tmp_path):
        provider = StockDataProvider()
        provider._history_dirs = [tmp_path]

        preferred_path = tmp_path / "6758.TO.csv"
        fallback_path = tmp_path / "6758.T.csv"

        pd.DataFrame({"Close": [1.0]}).to_csv(preferred_path)
        pd.DataFrame({"Close": [2.0]}).to_csv(fallback_path)

        loaded = provider._load_first_available_csv("6758.to")
        assert loaded is not None
        df, ticker = loaded
        assert ticker == "6758.TO"
        expected = pd.read_csv(preferred_path, index_col=0, parse_dates=True)
        pd.testing.assert_frame_equal(df, expected)

    def test_download_via_yfinance_prefers_suffix(self, monkeypatch):
        provider = StockDataProvider()

        history_df = pd.DataFrame({"Close": [1.0], "Volume": [1000]}, index=pd.date_range("2024-01-01", periods=1))

        class DummyTicker:
            def __init__(self, ticker):
                self.ticker = ticker

            def history(self, period="1y"):
                if self.ticker == "6758.TO":
                    return history_df
                return pd.DataFrame()

        dummy_yf = types.SimpleNamespace(Ticker=lambda ticker: DummyTicker(ticker))
        monkeypatch.setattr(stock_data_module, "YFINANCE_AVAILABLE", True)
        monkeypatch.setattr(stock_data_module, "yf", dummy_yf)

        history, actual = provider._download_via_yfinance("6758.to", "1mo")
        assert actual == "6758.TO"
        pd.testing.assert_frame_equal(history, history_df)

    def test_download_via_yfinance_retries_on_transient_error(self, monkeypatch):
        provider = StockDataProvider()

        history_df = pd.DataFrame(
            {"Close": [123.0], "Volume": [2000]},
            index=pd.date_range("2024-01-01", periods=1),
        )

        attempt_counter = {"count": 0}

        class FlakyTicker:
            def __init__(self, ticker):
                self.ticker = ticker

            def history(self, period="1y", start=None, end=None):
                attempt_counter["count"] += 1
                if attempt_counter["count"] == 1:
                    raise RuntimeError("temporary network glitch")
                return history_df

        dummy_yf = types.SimpleNamespace(Ticker=lambda ticker: FlakyTicker(ticker))

        monkeypatch.setattr(stock_data_module, "YFINANCE_AVAILABLE", True)
        monkeypatch.setattr(stock_data_module, "yf", dummy_yf)
        monkeypatch.setattr(
            stock_data_module,
            "time",
            types.SimpleNamespace(sleep=lambda *_: None),
            raising=False,
        )

        history, actual_ticker = provider._download_via_yfinance("6758", "1mo")

        assert attempt_counter["count"] == 2
        assert actual_ticker is not None
        pd.testing.assert_frame_equal(history, history_df)

    def test_download_via_yfinance_raises_with_last_error_details(self, monkeypatch):
        provider = StockDataProvider()

        attempt_counter = {"count": 0}
        last_message = "permanent outage"

        class FailingTicker:
            def __init__(self, ticker):
                self.ticker = ticker

            def history(self, period="1y", start=None, end=None):
                attempt_counter["count"] += 1
                raise RuntimeError(f"{last_message} #{attempt_counter['count']}")

        dummy_yf = types.SimpleNamespace(Ticker=lambda ticker: FailingTicker(ticker))

        monkeypatch.setattr(stock_data_module, "YFINANCE_AVAILABLE", True)
        monkeypatch.setattr(stock_data_module, "yf", dummy_yf)
        monkeypatch.setattr(
            stock_data_module,
            "time",
            types.SimpleNamespace(sleep=lambda *_: None),
            raising=False,
        )

        with pytest.raises(DataFetchError) as excinfo:
            provider._download_via_yfinance("6758", "1mo")

        assert attempt_counter["count"] >= 1
        assert excinfo.value.details is not None
        assert last_message in excinfo.value.details

    def test_real_stock_data_integration(self):
        provider = StockDataProvider()
        try:
            data = provider.get_stock_data("7203", "5d")
            if not data.empty:
                assert "Close" in data.columns
                assert "Volume" in data.columns
        except Exception:
            pytest.skip("Network connection required for integration test")

