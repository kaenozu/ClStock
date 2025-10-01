import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
import types

import pandas as pd
import pytest

def _stub_module(module_path: str, class_name: str) -> None:
    parts = module_path.split(".")
    parent_name = parts[0]
    if parent_name not in sys.modules:
        sys.modules[parent_name] = types.ModuleType(parent_name)

    parent = sys.modules[parent_name]

    for idx in range(1, len(parts)):
        module_name = ".".join(parts[: idx + 1])
        if module_name not in sys.modules:
            sys.modules[module_name] = types.ModuleType(module_name)
            setattr(parent, parts[idx], sys.modules[module_name])
        parent = sys.modules[module_name]

    if not hasattr(parent, class_name):
        stub_class = type(
            class_name,
            (),
            {"__init__": lambda self, *args, **kwargs: None},
        )
        setattr(parent, class_name, stub_class)


for module_name, class_name in [
    ("ml_models.hybrid_predictor", "HybridPredictor"),
    ("optimization.tse_optimizer", "TSEPortfolioOptimizer"),
    ("sentiment.sentiment_analyzer", "SentimentAnalyzer"),
    ("strategies.strategy_generator", "StrategyGenerator"),
    ("risk.risk_manager", "RiskManager"),
]:
    _stub_module(module_name, class_name)

from full_auto_system import (
    AutoRecommendation,
    FullAutoInvestmentSystem,
    RiskManagerAdapter,
)
from models_new.base.interfaces import PredictionResult
from models_new.advanced.risk_management_framework import (
    PortfolioRisk,
    RiskLevel,
)
from models_new.advanced.trading_strategy_generator import (
    TradingStrategy,
    StrategyType,
)


@pytest.mark.asyncio
async def test_analyze_single_stock_uses_new_components(monkeypatch):
    system = FullAutoInvestmentSystem()

    prediction_result = PredictionResult(
        prediction=110.0,
        confidence=0.8,
        accuracy=0.9,
        timestamp=datetime.now(),
        symbol="TEST",
        metadata={},
    )
    predictor = SimpleNamespace()
    predictor.predict = MagicMock(return_value=prediction_result)  # type: ignore[attr-defined]
    system.predictor = predictor  # type: ignore[assignment]

    portfolio_risk = PortfolioRisk(
        total_risk_score=0.3,
        risk_level=RiskLevel.LOW,
        individual_metrics={},
        risk_breakdown={},
        recommendations=["Diversify"],
        max_safe_position_size=0.1,
        timestamp=datetime.now(),
    )
    risk_manager = SimpleNamespace()
    risk_manager.analyze_portfolio_risk = MagicMock(return_value=portfolio_risk)  # type: ignore[attr-defined]
    system.risk_manager = risk_manager  # type: ignore[assignment]

    sentiment_analyzer = SimpleNamespace()
    sentiment_analyzer.analyze_news_sentiment = MagicMock(return_value=0.25)  # type: ignore[attr-defined]
    system.sentiment_analyzer = sentiment_analyzer  # type: ignore[assignment]

    trading_strategy = TradingStrategy(
        name="TEST_Momentum",
        strategy_type=StrategyType.MOMENTUM,
        parameters={},
        entry_conditions=[],
        exit_conditions=[],
        risk_management={"stop_loss_pct": 0.05, "take_profit_pct": 0.1},
        expected_return=0.12,
        max_drawdown=0.08,
        sharpe_ratio=1.4,
        win_rate=0.66,
        created_at=datetime.now(),
    )
    strategy_generator = SimpleNamespace()
    strategy_generator.generate_momentum_strategy = MagicMock(return_value=trading_strategy)  # type: ignore[attr-defined]
    system.strategy_generator = strategy_generator  # type: ignore[assignment]

    data = pd.DataFrame(
        {"Close": [100.0, 102.0], "High": [101.0, 103.0], "Low": [99.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    data.attrs.setdefault("info", {})["longName"] = "Test Corp"

    recommendation = await system._analyze_single_stock("TEST", data)

    assert isinstance(recommendation, AutoRecommendation)
    assert recommendation.entry_price == pytest.approx(data["Close"].iloc[-1])
    assert recommendation.target_price == pytest.approx(
        recommendation.entry_price * (1 + trading_strategy.expected_return)
    )
    assert recommendation.stop_loss == pytest.approx(
        recommendation.entry_price * (1 - trading_strategy.risk_management["stop_loss_pct"])
    )
    assert recommendation.expected_return == pytest.approx(trading_strategy.expected_return)
    assert recommendation.risk_level == portfolio_risk.risk_level.value

    # Risk-adjusted confidence combines strategy win rate and risk score
    expected_confidence = (
        trading_strategy.win_rate
        + (1.0 - portfolio_risk.total_risk_score)
        + prediction_result.confidence
    ) / 3
    assert recommendation.confidence == pytest.approx(expected_confidence)

    system.predictor.predict.assert_called_once_with("TEST")  # type: ignore[attr-defined]
    system.risk_manager.analyze_portfolio_risk.assert_called_once()  # type: ignore[attr-defined]
    portfolio_args = system.risk_manager.analyze_portfolio_risk.call_args[0]  # type: ignore[attr-defined]
    assert portfolio_args[0]["positions"]["TEST"] == pytest.approx(data["Close"].iloc[-1])
    assert portfolio_args[1]["TEST"] is data

    system.sentiment_analyzer.analyze_news_sentiment.assert_called_once()  # type: ignore[attr-defined]
    system.strategy_generator.generate_momentum_strategy.assert_called_once_with("TEST", data)  # type: ignore[attr-defined]

@pytest.mark.asyncio
async def test_analyze_single_stock_with_empty_strategy_returns_none(monkeypatch):
    system = FullAutoInvestmentSystem()

    prediction_result = PredictionResult(
        prediction=110.0,
        confidence=0.8,
        accuracy=0.9,
        timestamp=datetime.now(),
        symbol="TEST",
        metadata={},
    )
    predictor = SimpleNamespace()
    predictor.predict = MagicMock(return_value=prediction_result)  # type: ignore[attr-defined]
    system.predictor = predictor  # type: ignore[assignment]

    portfolio_risk = PortfolioRisk(
        total_risk_score=0.3,
        risk_level=RiskLevel.LOW,
        individual_metrics={},
        risk_breakdown={},
        recommendations=["Diversify"],
        max_safe_position_size=0.1,
        timestamp=datetime.now(),
    )
    risk_manager = SimpleNamespace()
    risk_manager.analyze_portfolio_risk = MagicMock(return_value=portfolio_risk)  # type: ignore[attr-defined]
    system.risk_manager = risk_manager  # type: ignore[assignment]

    sentiment_analyzer = SimpleNamespace()
    sentiment_analyzer.analyze_news_sentiment = MagicMock(return_value=0.25)  # type: ignore[attr-defined]
    system.sentiment_analyzer = sentiment_analyzer  # type: ignore[assignment]

    # 空の辞書を返すようにモックを設定
    strategy_generator = SimpleNamespace()
    strategy_generator.generate_strategy = MagicMock(return_value={})  # type: ignore[attr-defined]
    system.strategy_generator = strategy_generator  # type: ignore[assignment]

    data = pd.DataFrame(
        {"Close": [100.0, 102.0], "High": [101.0, 103.0], "Low": [99.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    data.attrs.setdefault("info", {})["longName"] = "Test Corp"

    recommendation = await system._analyze_single_stock("TEST", data)

    # recommendation が None であることを確認
    assert recommendation is None

    # generate_strategy が呼び出されたことを確認
    system.strategy_generator.generate_strategy.assert_called_once()  # type: ignore[attr-defined]


def test_perform_portfolio_risk_analysis_delegates_to_adapter():
    system = FullAutoInvestmentSystem()

    mock_manager = SimpleNamespace()
    mock_manager.analyze_portfolio_risk = MagicMock(return_value={"risk": "ok"})  # type: ignore[attr-defined]
    system.risk_manager = RiskManagerAdapter(manager=mock_manager)  # type: ignore[assignment]

    price_data = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )

    result = system._perform_portfolio_risk_analysis(
        "TEST",
        current_price=100.0,
        price_data=price_data,
        predicted_price=110.0,
    )

    assert result == {"risk": "ok"}
    mock_manager.analyze_portfolio_risk.assert_called_once()  # type: ignore[attr-defined]
