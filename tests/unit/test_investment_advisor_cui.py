"""Tests for the investment advisor CUI target symbol handling."""

import sys
import types
from types import SimpleNamespace

import pytest

from config.target_universe import get_target_universe


@pytest.fixture
def advisor(monkeypatch):
    """Provide an instance of InvestmentAdvisorCUI with lightweight dependencies."""
    stub_modules = {
        "models": types.ModuleType("models"),
        "models.precision": types.ModuleType("models.precision"),
        "models.precision.precision_87_system": types.ModuleType(
            "models.precision.precision_87_system",
        ),
        "models.hybrid": types.ModuleType("models.hybrid"),
        "models.hybrid.hybrid_predictor": types.ModuleType(
            "models.hybrid.hybrid_predictor",
        ),
        "models.hybrid.prediction_modes": types.ModuleType(
            "models.hybrid.prediction_modes",
        ),
        "archive": types.ModuleType("archive"),
        "archive.old_systems": types.ModuleType("archive.old_systems"),
        "archive.old_systems.medium_term_prediction": types.ModuleType(
            "archive.old_systems.medium_term_prediction",
        ),
        "data": types.ModuleType("data"),
        "data.stock_data": types.ModuleType("data.stock_data"),
        "data.sector_classification": types.ModuleType("data.sector_classification"),
    }

    for name, module in stub_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if name in {
            "models",
            "models.precision",
            "models.hybrid",
            "archive",
            "archive.old_systems",
            "data",
        }:
            module.__path__ = []  # type: ignore[attr-defined]

    stub_modules[
        "models.precision.precision_87_system"
    ].Precision87BreakthroughSystem = (  # type: ignore[attr-defined]
        lambda: SimpleNamespace()
    )
    stub_modules["models.hybrid.hybrid_predictor"].HybridStockPredictor = (  # type: ignore[attr-defined]
        lambda: SimpleNamespace()
    )
    stub_modules["models.hybrid.prediction_modes"].PredictionMode = SimpleNamespace()  # type: ignore[attr-defined]
    stub_modules[
        "archive.old_systems.medium_term_prediction"
    ].MediumTermPredictionSystem = (  # type: ignore[attr-defined]
        lambda: SimpleNamespace()
    )
    stub_modules["data.stock_data"].StockDataProvider = lambda: SimpleNamespace()  # type: ignore[attr-defined]
    stub_modules["data.sector_classification"].SectorClassification = SimpleNamespace(  # type: ignore[attr-defined]
        get_sector_risk=lambda symbol: 0.0,
    )

    from investment_advisor_cui import InvestmentAdvisorCUI

    return InvestmentAdvisorCUI()


def test_investment_advisor_targets_from_universe(advisor):
    """The advisor should mirror the target universe definitions."""
    universe = get_target_universe()

    assert advisor.target_symbols == universe.all_formatted()
    assert set(advisor.symbol_names) == set(universe.base_codes)
