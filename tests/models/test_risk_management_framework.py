import importlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.modules.pop("models", None)
importlib.import_module("models")

from models.advanced.risk_management_framework import (
    PortfolioRisk,
    RiskLevel,
    RiskManager,
)


def _generate_price_frame(start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.now(), periods=90, freq="D")
    close = np.linspace(start_price, start_price * 1.05, len(dates))
    return pd.DataFrame(
        {
            "Close": close,
            "Volume": np.full(len(dates), 1_000_000),
            "High": close * 1.01,
            "Low": close * 0.99,
        },
        index=dates,
    )


def test_risk_manager_provides_summary_for_balanced_portfolio():
    manager = RiskManager()

    portfolio = {"positions": {"AAPL": 60_000, "GOOGL": 40_000}}
    price_data = {symbol: _generate_price_frame(100 + idx * 5) for idx, symbol in enumerate(portfolio["positions"].keys())}

    risk_report = manager.analyze_portfolio_risk(portfolio, price_data)

    assert isinstance(risk_report, PortfolioRisk)
    assert isinstance(risk_report.risk_level, RiskLevel)

    summary = manager.get_risk_summary()
    assert summary["current_risk_level"] in {level.value for level in RiskLevel}
    assert summary["max_safe_position_size"] > 0
    assert summary["risk_breakdown"].keys() >= {"market", "liquidity", "concentration", "volatility", "correlation"}
