import os
import pandas as pd
from datetime import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

os.environ.setdefault("CLSTOCK_DEV_KEY", "test-key")
os.environ.setdefault("CLSTOCK_ADMIN_KEY", "admin-key")

from api.endpoints import router


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
    mock_provider.get_financial_metrics.return_value = {}

    mock_provider_cls.return_value = mock_provider

    response = client.get("/stock/7203.T/data?period=1mo")

    assert response.status_code == 200


@patch("api.endpoints.StockDataProvider")
def test_get_stock_data_invalid_period_returns_400(mock_provider_cls):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/stock/7203/data?period=invalid")

    assert response.status_code == 400
    assert "Invalid period" in response.json()["detail"]
    mock_provider_cls.assert_not_called()


def test_get_recommendations_uses_single_datetime_call(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    call_counter = {"count": 0}

    class DummyDateTime:
        @classmethod
        def now(cls, tz=None):
            call_counter["count"] += 1
            return datetime(2024, 1, 1, 10, 0, 0)

    monkeypatch.setattr("api.endpoints.datetime", DummyDateTime)

    class DummyPredictor:
        def get_top_recommendations(self, top_n):
            return []

    monkeypatch.setattr("api.endpoints.MLStockPredictor", lambda: DummyPredictor())
    monkeypatch.setattr("api.endpoints.verify_token", lambda token: None)

    response = client.get(
        "/recommendations",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    assert call_counter["count"] == 1


def test_get_recommendations_returns_closed_after_15(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class DummyDateTime:
        @classmethod
        def now(cls, tz=None):
            return datetime(2024, 1, 1, 15, 0, 0)

    monkeypatch.setattr("api.endpoints.datetime", DummyDateTime)

    class DummyPredictor:
        def get_top_recommendations(self, top_n):
            return []

    monkeypatch.setattr("api.endpoints.MLStockPredictor", lambda: DummyPredictor())
    monkeypatch.setattr("api.endpoints.verify_token", lambda token: None)

    response = client.get(
        "/recommendations",
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 200
    assert response.json()["market_status"] == "市場営業時間外"
