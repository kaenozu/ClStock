import os
import sys
from datetime import datetime, timedelta
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

import pandas as pd

# プロジェクトルートをパスに追加し、旧スタブをクリア
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

for legacy_module in ("models", "models_new", "models_refactored"):
    sys.modules.pop(legacy_module, None)

# data.stock_data は構文エラーを含むため、テストでは簡易スタブを注入する
if "data.stock_data" not in sys.modules:
    import data

    dummy_stock_data = ModuleType("data.stock_data")

    class _ConftestStockDataProvider:  # pragma: no cover - fixture用スタブ
        def __init__(self, *_, **__):
            pass

    dummy_stock_data.StockDataProvider = _ConftestStockDataProvider
    sys.modules["data.stock_data"] = dummy_stock_data
    data.stock_data = dummy_stock_data

# APIセキュリティ関連の環境変数とモジュールをモック
os.environ.setdefault("CLSTOCK_DEV_KEY", "test_dev_key")
os.environ.setdefault("CLSTOCK_ADMIN_KEY", "test_admin_key")


# config.secrets モジュールが存在しない場合のモック
class MockConfigSecrets:
    API_KEYS = {"test_dev_key": "developer", "test_admin_key": "administrator"}


sys.modules["config.secrets"] = MockConfigSecrets()

from models.recommendation import StockRecommendation


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


@pytest.fixture
def sample_recommendation() -> StockRecommendation:
    """Provide a representative stock recommendation object for API tests."""
    return StockRecommendation(
        rank=1,
        symbol="7203",
        company_name="トヨタ自動車",
        buy_timing="押し目買いを検討",
        target_price=2300.0,
        stop_loss=2000.0,
        profit_target_1=2400.0,
        profit_target_2=2500.0,
        holding_period="1～2か月",
        score=85.0,
        current_price=2100.0,
        recommendation_reason="テクニカル指標が上昇トレンドを示唆",
    )


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
            },
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
