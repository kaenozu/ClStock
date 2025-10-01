import sys
import types
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from types import SimpleNamespace

import pandas as pd
import pytest


def _ensure_module(path: str) -> types.ModuleType:
    parts = path.split(".")
    module: types.ModuleType | None = None
    for idx in range(len(parts)):
        name = ".".join(parts[: idx + 1])
        if name not in sys.modules:
            module = types.ModuleType(name)
            sys.modules[name] = module
            if idx > 0:
                parent = sys.modules[".".join(parts[:idx])]
                setattr(parent, parts[idx], module)
        else:
            module = sys.modules[name]
    assert module is not None
    return module


hybrid_module = _ensure_module("models_new.hybrid.hybrid_predictor")


class _StubHybridStockPredictor:
    def predict(self, symbol: str):  # pragma: no cover - default stub behaviour
        return SimpleNamespace(
            prediction=0.0,
            confidence=0.0,
            accuracy=0.0,
            metadata={},
            symbol=symbol,
            timestamp=datetime.now(),
        )


hybrid_module.HybridStockPredictor = _StubHybridStockPredictor


analysis_module = _ensure_module("analysis.sentiment_analyzer")


class _StubMarketSentimentAnalyzer:
    def analyze_news_sentiment(self, symbol: str):  # pragma: no cover - stub
        return {"sentiment_score": 0.0}


analysis_module.MarketSentimentAnalyzer = _StubMarketSentimentAnalyzer


trading_analysis_module = _ensure_module("trading.tse.analysis")


@dataclass
class _StubStockProfile:
    symbol: str
    sector: str
    market_cap: float
    volatility: float
    profit_potential: float
    diversity_score: float
    combined_score: float


trading_analysis_module.StockProfile = _StubStockProfile


trading_optimizer_module = _ensure_module("trading.tse.optimizer")


class _StubPortfolioOptimizer:
    def optimize_portfolio(self, profiles):  # pragma: no cover - stub
        return {"selected_stocks": [p.symbol for p in profiles[:5]]}


trading_optimizer_module.PortfolioOptimizer = _StubPortfolioOptimizer


strategy_module = _ensure_module("models_new.advanced.trading_strategy_generator")


class _StubActionType(Enum):
    BUY = "buy"
    SELL = "sell"


class _StubStrategyGenerator:
    def generate_strategy(self, *_, **__):  # pragma: no cover - stub
        return {}


class _StubSignalGenerator:
    def generate_signals(self, *_, **__):  # pragma: no cover - stub
        return []


strategy_module.ActionType = _StubActionType
strategy_module.StrategyGenerator = _StubStrategyGenerator
strategy_module.SignalGenerator = _StubSignalGenerator


risk_module = _ensure_module("models_new.advanced.risk_management_framework")


class _StubRiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class _StubPortfolioRisk:
    total_risk_score: float = 0.5
    risk_level: _StubRiskLevel = _StubRiskLevel.MEDIUM
    individual_metrics: dict | None = None
    risk_breakdown: dict | None = None
    recommendations: list | None = None
    max_safe_position_size: float = 0.1
    timestamp: datetime = datetime.now()


class _StubRiskManager:
    def analyze_portfolio_risk(self, *_, **__):  # pragma: no cover - stub
        return _StubPortfolioRisk()


risk_module.RiskLevel = _StubRiskLevel
risk_module.PortfolioRisk = _StubPortfolioRisk
risk_module.RiskManager = _StubRiskManager


archive_module = _ensure_module("archive.old_systems.medium_term_prediction")


class _StubMediumTermPredictionSystem:
    pass


archive_module.MediumTermPredictionSystem = _StubMediumTermPredictionSystem


script_module = _ensure_module("data_retrieval_script_generator")
script_module.generate_colab_data_retrieval_script = lambda *_, **__: ""


settings_module = _ensure_module("config.settings")
settings_module.get_settings = lambda: SimpleNamespace()


from full_auto_system import FullAutoInvestmentSystem, RiskAssessment
from models_new.advanced.risk_management_framework import RiskLevel


@pytest.mark.asyncio
async def test_analyze_single_stock_uses_risk_assessment_score():
    system = FullAutoInvestmentSystem()

    system.predictor = SimpleNamespace(
        predict=lambda symbol, data: {
            "predicted_price": float(data["Close"].iloc[-1]) + 5.0,
            "confidence": 0.31,
        }
    )

    distinctive_risk_score = 0.23
    risk_assessment = RiskAssessment(
        risk_score=distinctive_risk_score,
        risk_level=RiskLevel.LOW,
        max_safe_position_size=0.5,
        recommendations=["Hold"],
        raw=SimpleNamespace(total_risk_score=0.81),
    )
    system.risk_manager = SimpleNamespace(analyze_risk=lambda *args, **kwargs: risk_assessment)

    system.sentiment_analyzer = SimpleNamespace(
        analyze_sentiment=lambda symbol: {"sentiment_score": 0.12}
    )

    system.strategy_generator = SimpleNamespace(
        generate_strategy=lambda *args, **kwargs: {
            "entry_price": 100.0,
            "confidence_score": 0.7,
            "expected_return": 0.15,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.15,
        }
    )

    data = pd.DataFrame(
        {"Close": [95.0, 100.0]},
        index=pd.date_range(datetime(2024, 1, 1), periods=2),
    )

    recommendation = await system._analyze_single_stock("MOCK", data)

    assert recommendation is not None
    expected_confidence = (0.7 + (1 - distinctive_risk_score) + 0.31) / 3
    assert recommendation.confidence == pytest.approx(expected_confidence)
