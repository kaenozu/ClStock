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
    """テスト用の株価データのモック（デフォルト1年間）"""
    return generate_mock_stock_data()

def generate_mock_stock_data(period="1y", symbol="7203", company_name="トヨタ自動車"):
    """期間パラメータに対応したモックデータ生成

    Args:
        period: 期間 ('1y', '6mo', '3mo', '1mo', '5d' など)
        symbol: 銘柄コード
        company_name: 会社名

    Returns:
        pd.DataFrame: 株価データ
    """
    # 期間パラメータを解析
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
        # デフォルトは1年
        days = 365

    # 終了日を今日として、開始日を計算
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
