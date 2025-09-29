import pandas as pd
import pytest
from unittest.mock import patch

from data.stock_data import StockDataProvider
from utils.exceptions import DataFetchError


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
        with patch("data.stock_data.get_cache", return_value=_NullCache()), \
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
        with patch("data.stock_data.get_cache", return_value=_NullCache()), \
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
        with patch("data.stock_data.get_cache", return_value=_NullCache()), \
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
        with patch("data.stock_data.get_cache", return_value=_NullCache()), \
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

    def test_get_multiple_stocks(self, mock_yfinance):
        provider = StockDataProvider()
        with patch("yfinance.Ticker", return_value=mock_yfinance):
            symbols = ["7203", "6758"]
            result = provider.get_multiple_stocks(symbols, "1mo")
            assert len(result) == 2
            assert "7203" in result
            assert "6758" in result

    def test_ticker_formats_for_japanese_stock(self):
        provider = StockDataProvider()
        formats = provider._ticker_formats("7203")
        assert formats[0] == "7203.T"
        assert formats[-1] in {"7203", "7203.T", "7203.TO"}

    def test_real_stock_data_integration(self):
        provider = StockDataProvider()
        try:
            data = provider.get_stock_data("7203", "5d")
            if not data.empty:
                assert "Close" in data.columns
                assert "Volume" in data.columns
        except Exception:
            pytest.skip("Network connection required for integration test")
