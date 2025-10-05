import os
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime as dt_datetime
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

os.environ.setdefault("CLSTOCK_DEV_KEY", "test-key")
os.environ.setdefault("CLSTOCK_ADMIN_KEY", "admin-key")


class _StubMLStockPredictor:
    pass


@dataclass
class _StubStockRecommendation:
    rank: int = 0
    symbol: str = ""
    company_name: str = ""
    buy_timing: str = ""
    target_price: float = 0.0
    stop_loss: float = 0.0
    profit_target_1: float = 0.0
    profit_target_2: float = 0.0
    holding_period: str = ""
    score: float = 0.0
    current_price: float = 0.0
    recommendation_reason: str = ""
    recommendation_level: str = "neutral"
    generated_at: dt_datetime = field(default_factory=dt_datetime.utcnow)

    def to_dict(self):
        return {
            "rank": self.rank,
            "symbol": self.symbol,
            "company_name": self.company_name,
            "buy_timing": self.buy_timing,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "profit_target_1": self.profit_target_1,
            "profit_target_2": self.profit_target_2,
            "holding_period": self.holding_period,
            "score": self.score,
            "current_price": self.current_price,
            "recommendation_reason": self.recommendation_reason,
            "recommendation_level": self.recommendation_level,
            "generated_at": self.generated_at.isoformat(),
        }


sys.modules.setdefault("joblib", types.ModuleType("joblib"))
_models_pkg = sys.modules.setdefault("models", types.ModuleType("models"))
sys.modules.setdefault("models.core", types.ModuleType("models.core"))
sys.modules.setdefault("models.legacy_core", types.ModuleType("models.legacy_core"))
sys.modules.setdefault(
    "models.recommendation", types.ModuleType("models.recommendation")
)
setattr(sys.modules["models.core"], "MLStockPredictor", _StubMLStockPredictor)
setattr(sys.modules["models.legacy_core"], "MLStockPredictor", _StubMLStockPredictor)
setattr(_models_pkg, "core", sys.modules["models.core"])
setattr(_models_pkg, "legacy_core", sys.modules["models.legacy_core"])
setattr(
    sys.modules["models.recommendation"],
    "StockRecommendation",
    _StubStockRecommendation,
)
setattr(_models_pkg, "recommendation", sys.modules["models.recommendation"])

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
            "SMA_20": [210.0],
            "SMA_50": [220.0],
            "RSI": [55.5],
            "MACD": [1.5],
        },
        index=pd.date_range("2024-02-01", periods=1),
    )

    mock_provider.get_stock_data.return_value = suffix_df
    mock_provider.calculate_technical_indicators.return_value = suffix_df
    mock_provider.get_financial_metrics.return_value = {
        "symbol": "7203",
        "company_name": "Test Corp",
        "actual_ticker": "7203.TO",
    }

    mock_provider_cls.return_value = mock_provider

    response = client.get("/stock/7203.TO/data?period=1mo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["financial_metrics"]["symbol"] == "7203.TO"
    assert payload["financial_metrics"]["actual_ticker"] == "7203.TO"
    assert payload["technical_indicators"] == {
        "sma_20": 210.0,
        "sma_50": 220.0,
        "rsi": 55.5,
        "macd": 1.5,
    }


@patch("api.endpoints.StockDataProvider")
def test_get_stock_data_uses_actual_ticker_from_technical_data(mock_provider_cls):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_provider = MagicMock()
    mock_provider.get_all_stock_symbols.return_value = {"7203": "Test Corp"}

    technical_df = pd.DataFrame(
        {
            "Close": [300.0, 310.0],
            "Volume": [3500, 3600],
            "SMA_20": [None, None],
            "SMA_50": [None, None],
            "RSI": [None, None],
            "MACD": [None, None],
            "ActualTicker": ["7203.T", "7203.T"],
        },
        index=pd.date_range("2024-03-01", periods=2),
    )

    mock_provider.get_stock_data.return_value = technical_df
    mock_provider.calculate_technical_indicators.return_value = technical_df
    mock_provider.get_financial_metrics.return_value = {}

    mock_provider_cls.return_value = mock_provider

    response = client.get("/stock/7203.T/data?period=1mo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["financial_metrics"]["actual_ticker"] == "7203.T"


@patch("api.endpoints.StockDataProvider")
def test_get_stock_data_includes_actual_ticker(mock_provider_cls):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    mock_provider = MagicMock()
    mock_provider.get_all_stock_symbols.return_value = {"7203": "Test Corp"}

    fallback_ticker = "7203.F"
    technical_df = pd.DataFrame(
        {
            "Close": [400.0, 405.0],
            "Volume": [4500, 4600],
            "SMA_20": [None, None],
            "SMA_50": [None, None],
            "RSI": [None, None],
            "MACD": [None, None],
            "ActualTicker": [fallback_ticker, fallback_ticker],
        },
        index=pd.date_range("2024-04-01", periods=2),
    )

    mock_provider.get_stock_data.return_value = technical_df
    mock_provider.calculate_technical_indicators.return_value = technical_df
    mock_provider.get_financial_metrics.return_value = {
        "symbol": "7203",
        "company_name": "Test Corp",
        "actual_ticker": None,
    }

    mock_provider_cls.return_value = mock_provider

    response = client.get("/stock/7203/data?period=1mo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["financial_metrics"]["actual_ticker"] == fallback_ticker


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


def test_get_recommendations_returns_closed_after_weekend(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    class DummyDateTime:
        @classmethod
        def now(cls, tz=None):
            assert tz == ZoneInfo("Asia/Tokyo")
            return datetime(2024, 1, 6, 10, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))

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


@patch("api.endpoints.verify_token")
@patch("api.endpoints.MLStockPredictor")
@patch("api.endpoints.StockDataProvider")
def test_get_single_recommendation_entry_price_centered_on_current_price(
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
        target_price=150.0,
        stop_loss=90.0,
        profit_target_1=110.0,
        profit_target_2=130.0,
        holding_period="1m",
        score=80.0,
        current_price=100.0,
        recommendation_reason="Test",
        recommendation_level="buy",
    )

    mock_predictor = MagicMock()
    mock_predictor.generate_recommendation.return_value = sample_recommendation
    mock_predictor_cls.return_value = mock_predictor

    response = client.get(
        "/recommendation/7203",
        headers={"Authorization": "Bearer token"},
    )

    assert response.status_code == 200
    payload = response.json()

    entry_price = payload["entry_price"]
    expected_min = sample_recommendation.current_price * 0.97
    expected_max = sample_recommendation.current_price * 1.03

    assert entry_price["min"] == pytest.approx(expected_min)
    assert entry_price["max"] == pytest.approx(expected_max)
