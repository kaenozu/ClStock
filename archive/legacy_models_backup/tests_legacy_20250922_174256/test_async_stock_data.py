"""
Integration tests for async functionality
"""

import pytest
import asyncio
import pandas as pd
from unittest.mock import patch, Mock
from data.async_stock_data import AsyncStockDataProvider, get_async_stock_data_provider


class TestAsyncStockDataProvider:
    """Async stock data provider integration tests"""

    @pytest.fixture
    def async_provider(self):
        """Create async stock data provider instance"""
        return get_async_stock_data_provider()

    @pytest.mark.asyncio
    async def test_get_stock_data_async(self, async_provider):
        """Test async stock data fetching"""
        # Mock the yfinance call
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [102, 103, 104],
                "Volume": [1000, 1100, 1200],
            }
        )

        with patch.object(async_provider, "_fetch_with_yfinance_async") as mock_fetch:
            mock_fetch.return_value = mock_data

            result = await async_provider.get_stock_data("7203", "1mo")

            assert not result.empty
            assert "Symbol" in result.columns
            assert result["Symbol"].iloc[0] == "7203"

    @pytest.mark.asyncio
    async def test_get_multiple_stocks_async(self, async_provider):
        """Test async multiple stocks fetching"""
        # Mock the yfinance calls
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [102, 103, 104],
                "Volume": [1000, 1100, 1200],
            }
        )

        with patch.object(async_provider, "_fetch_single_stock_async") as mock_fetch:
            mock_fetch.return_value = mock_data

            symbols = ["7203", "6758"]
            result = await async_provider.get_multiple_stocks(symbols, "1mo")

            assert len(result) == 2
            assert "7203" in result
            assert "6758" in result

    @pytest.mark.asyncio
    async def test_calculate_technical_indicators_async(self, async_provider):
        """Test async technical indicators calculation"""
        # Create test data
        test_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [95, 96, 97, 98, 99],
                "Close": [102, 103, 104, 105, 106],
                "Volume": [1000, 1100, 1200, 1300, 1400],
                "Symbol": ["7203"] * 5,
            }
        )

        result = await async_provider.calculate_technical_indicators_async(test_data)

        # Check that technical indicators were added
        assert "SMA_20" in result.columns
        assert "SMA_50" in result.columns
        assert "RSI" in result.columns
        assert "MACD" in result.columns
        assert "ATR" in result.columns

    @pytest.mark.asyncio
    async def test_error_handling_async(self, async_provider):
        """Test async error handling"""
        with patch.object(async_provider, "_fetch_with_yfinance_async") as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")

            with pytest.raises(Exception):
                await async_provider.get_stock_data("INVALID", "1mo")
