import pytest
import sys
import os
from unittest.mock import Mock, patch
try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency for tests
    pd = None
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import unittest.mock as _mock
from unittest.mock import MagicMock

_mock.Mock = MagicMock

@pytest.fixture
def mock_stock_data():
    """Mock stock data for testing (default 1 year)"""
    return generate_mock_stock_data()


def generate_mock_stock_data(period="1y", symbol="7203", company_name="トヨタ自動車"):
    """Generate mock data corresponding to period parameter

    Args:
        period: Period ('1y', '6mo', '3mo', '1mo', '5d' etc.)
        symbol: Stock symbol
        company_name: Company name

    Returns:
        pd.DataFrame: Stock price data
    """
    if pd is None:
        pytest.skip("pandas is required for mock stock data fixtures")
    # Parse period parameter
    if period == "1y":
        days = 365
    elif period == "6mo":
        days = 180
    elif period == "3mo":
        days = 90
    elif period == "1mo":
        days = 30
    elif period == "5d":
        days = 5
    elif period == "1d":
        days = 1
    else:
        # Default is 1 year
        days = 365

    # Calculate start date with end date as today
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    data = pd.DataFrame(
        {
            "Open": [100 + i * 0.1 for i in range(len(dates))],
            "High": [102 + i * 0.1 for i in range(len(dates))],
            "Low": [98 + i * 0.1 for i in range(len(dates))],
            "Close": [101 + i * 0.1 for i in range(len(dates))],
            "Volume": [1000000 + i * 1000 for i in range(len(dates))],
            "Symbol": [symbol] * len(dates),
            "CompanyName": [company_name] * len(dates),
        },
        index=dates,
    )
    return data


# StockRecommendation class has been removed, temporarily disabling test
# @pytest.fixture
# def sample_recommendation():
#     """Test recommendation data"""
#     # from models.recommendation import StockRecommendation
#     return None  # Removed: No longer needed after code refactoring


@pytest.fixture
def mock_yfinance():
    """Mock for yfinance"""
    with patch("yfinance.Ticker") as mock_ticker:
        if pd is None:
            pytest.skip("pandas is required for yfinance fixtures")
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [98, 99, 100],
                "Close": [101, 102, 103],
                "Volume": [1000000, 1100000, 1200000],
            }
        )
        mock_ticker_instance.info = {
            "marketCap": 1000000000,
            "trailingPE": 15.0,
            "priceToBook": 1.2,
            "dividendYield": 0.02,
            "returnOnEquity": 0.15,
            "currentPrice": 103.0,
            "targetMeanPrice": 110.0,
            "recommendationMean": 2.0,
        }
        mock_ticker.return_value = mock_ticker_instance
        yield mock_ticker_instance
