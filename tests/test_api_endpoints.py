import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

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
