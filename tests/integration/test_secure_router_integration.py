"""Integration test ensuring secure routes are mounted on the main FastAPI app."""

from __future__ import annotations

from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import pandas as pd
from app.main import app

client = TestClient(app)


@patch("api.secure_endpoints.verify_token", return_value="user")
@patch("api.secure_endpoints.StockDataProvider")
def test_secure_stock_endpoint_mounted(mock_provider_cls, mock_verify_token):
    """The secure stock endpoint should respond when routed through /api/v1."""
    mock_provider = Mock()
    mock_provider.get_stock_data.return_value = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100, 101, 102],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    mock_provider_cls.return_value = mock_provider

    response = client.get(
        "/api/v1/secure/stock/7203/data",
        params={"period": "1mo"},
        headers={"Authorization": "Bearer placeholder"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "7203"
    assert payload["period"] == "1mo"
    assert payload["data_points"] == 3
