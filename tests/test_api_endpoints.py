import os
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

os.environ.setdefault("CLSTOCK_DEV_KEY", "test-key")
os.environ.setdefault("CLSTOCK_ADMIN_KEY", "admin-key")

from api.endpoints import router
from models.recommendation import StockRecommendation


@patch("api.endpoints.StockDataProvider")
def test_get_stock_data_single_row(mock_provider_cls):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_provider = MagicMock()
    mock_provider.get_all_stock_symbols.return_value = {"7203": "Test Corp"}

    single_row_df = pd.DataFrame(
        {
            "Close": [100.0],
            "Volume": [1500],
            "SMA_20": [None],
            "SMA_50": [None],
            "RSI": [None],
            "MACD": [None],
        },
        index=pd.date_range("2024-01-01", periods=1),
    )

    mock_provider.get_stock_data.return_value = single_row_df
    mock_provider.calculate_technical_indicators.return_value = single_row_df
    mock_provider.get_financial_metrics.return_value = {}

    mock_provider_cls.return_value = mock_provider

    response = client.get("/stock/7203/data?period=1mo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["price_change"] == 0.0
    assert payload["price_change_percent"] == 0.0
    assert payload["current_price"] == 100.0


@patch("api.endpoints.StockDataProvider")
def test_get_stock_data_accepts_suffix_symbols(mock_provider_cls):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_provider = MagicMock()
    mock_provider.get_all_stock_symbols.return_value = {"7203": "Test Corp"}

    suffix_df = pd.DataFrame(
        {
            "Close": [200.0],
            "Volume": [2500],
            "SMA_20": [None],
            "SMA_50": [None],
            "RSI": [None],
            "MACD": [None],
        },
        index=pd.date_range("2024-02-01", periods=1),
    )

    mock_provider.get_stock_data.return_value = suffix_df
    mock_provider.calculate_technical_indicators.return_value = suffix_df
    mock_provider.get_financial_metrics.return_value = {
        "symbol": "7203",
        "company_name": "Test Corp",
        "actual_ticker": "7203",
    }

    mock_provider_cls.return_value = mock_provider

    response = client.get("/stock/7203.T/data?period=1mo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["financial_metrics"]["symbol"] == "7203.T"
    assert payload["financial_metrics"]["actual_ticker"] == "7203"


@patch("api.endpoints.StockDataProvider")
def test_get_stock_data_actual_ticker_prefers_technical_data(mock_provider_cls):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_provider = MagicMock()
    mock_provider.get_all_stock_symbols.return_value = {"7203": "Test Corp"}

    base_df = pd.DataFrame(
        {
            "Close": [300.0],
            "Volume": [3500],
            "SMA_20": [None],
            "SMA_50": [None],
            "RSI": [None],
            "MACD": [None],
        },
        index=pd.date_range("2024-03-01", periods=1),
    )

    technical_df = base_df.copy()
    technical_df["ActualTicker"] = ["7203.T"]

    mock_provider.get_stock_data.return_value = base_df
    mock_provider.calculate_technical_indicators.return_value = technical_df
    mock_provider.get_financial_metrics.return_value = {}

    mock_provider_cls.return_value = mock_provider

    response = client.get("/stock/7203.T/data?period=1mo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["financial_metrics"]["actual_ticker"] == "7203.T"


@patch("api.endpoints.StockDataProvider")
def test_get_stock_data_invalid_period_returns_400(mock_provider_cls):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/stock/7203/data?period=invalid")

    assert response.status_code == 400
    assert "Invalid period" in response.json()["detail"]
    mock_provider_cls.assert_not_called()


@patch("api.endpoints.MLStockPredictor")
@patch("api.endpoints.verify_token")
def test_get_recommendations_allows_top_n_50(mock_verify_token, mock_predictor_cls):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_predictor = MagicMock()
    mock_predictor.get_top_recommendations.return_value = []
    mock_predictor_cls.return_value = mock_predictor

    response = client.get(
        "/recommendations?top_n=50",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    mock_verify_token.assert_called_once_with("test-token")
    mock_predictor.get_top_recommendations.assert_called_once_with(50)


def test_get_recommendations_returns_closed_after_15(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class DummyDateTime:
        @classmethod
        def now(cls, tz=None):
            assert tz == ZoneInfo("Asia/Tokyo")
            return datetime(2024, 1, 1, 15, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))

    monkeypatch.setattr("api.endpoints.datetime", DummyDateTime)

    class DummyPredictor:
        def __init__(self) -> None:
            self.received_top_n = None

        def get_top_recommendations(self, top_n):
            self.received_top_n = top_n
            return []

    dummy_predictor = DummyPredictor()
    monkeypatch.setattr("api.endpoints.MLStockPredictor", lambda: dummy_predictor)
    monkeypatch.setattr("api.endpoints.verify_token", lambda token: None)

    response = client.get(
        "/recommendations",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    assert dummy_predictor.received_top_n == 10
    assert response.json()["market_status"] == "市場営業時間外"


def test_health_endpoint_includes_security_headers():
    from app.main import app as main_app

    client = TestClient(main_app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert (
        response.headers.get("Strict-Transport-Security")
        == "max-age=31536000; includeSubDomains"
    )


@patch("api.endpoints.verify_token")
@patch("api.endpoints.MLStockPredictor")
@patch("api.endpoints.StockDataProvider")
def test_get_single_recommendation_accepts_suffix(
    mock_provider_cls, mock_predictor_cls, mock_verify_token
):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_verify_token.return_value = None

    mock_provider = MagicMock()
    mock_provider.get_all_stock_symbols.return_value = {"7203": "Test Corp"}
    mock_provider.jp_stock_codes = {"7203": "Test Corp"}
    mock_provider_cls.return_value = mock_provider

    sample_recommendation = StockRecommendation(
        rank=1,
        symbol="7203",
        company_name="Test Corp",
        buy_timing="Now",
        target_price=120.0,
        stop_loss=90.0,
        profit_target_1=110.0,
        profit_target_2=130.0,
        holding_period="1m",
        score=75.0,
        current_price=100.0,
        recommendation_reason="Test",
        recommendation_level="buy",
    )

    mock_predictor = MagicMock()
    mock_predictor.generate_recommendation.return_value = sample_recommendation
    mock_predictor_cls.return_value = mock_predictor

    response = client.get(
        "/recommendation/7203.T",
        headers={"Authorization": "Bearer token"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "7203.T"
    assert payload["company_name"] == "Test Corp"