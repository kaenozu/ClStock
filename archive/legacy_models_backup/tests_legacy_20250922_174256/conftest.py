import pytest
import sys
import os
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_stock_data():
    """テスト用の株価データのモック"""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    data = pd.DataFrame(
        {
            "Open": [100 + i * 0.1 for i in range(len(dates))],
            "High": [102 + i * 0.1 for i in range(len(dates))],
            "Low": [98 + i * 0.1 for i in range(len(dates))],
            "Close": [101 + i * 0.1 for i in range(len(dates))],
            "Volume": [1000000 + i * 1000 for i in range(len(dates))],
            "Symbol": ["7203"] * len(dates),
            "CompanyName": ["トヨタ自動車"] * len(dates),
        },
        index=dates,
    )
    return data


@pytest.fixture
def sample_recommendation():
    """テスト用の推奨データ"""
    from models.recommendation import StockRecommendation

    return StockRecommendation(
        rank=1,
        symbol="7203",
        company_name="トヨタ自動車",
        buy_timing="テスト用買いタイミング",
        target_price=3000.0,
        stop_loss=2800.0,
        profit_target_1=3100.0,
        profit_target_2=3200.0,
        holding_period="1～2か月",
        score=85.0,
        current_price=2900.0,
        recommendation_reason="テスト用推奨理由",
    )


@pytest.fixture
def mock_yfinance():
    """yfinanceのモック"""
    with patch("yfinance.Ticker") as mock_ticker:
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
