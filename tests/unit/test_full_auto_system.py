import sys
import types
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import pandas as pd


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

if "sklearn" not in sys.modules:
    sklearn_module = types.ModuleType("sklearn")
    sys.modules["sklearn"] = sklearn_module
else:
    sklearn_module = sys.modules["sklearn"]

sklearn_preprocessing = types.ModuleType("sklearn.preprocessing")


class _DummyStandardScaler:
    def fit_transform(self, data):
        return data

    def transform(self, data):
        return data


sklearn_preprocessing.StandardScaler = _DummyStandardScaler
sys.modules["sklearn.preprocessing"] = sklearn_preprocessing
sklearn_module.preprocessing = sklearn_preprocessing

sklearn_metrics = types.ModuleType("sklearn.metrics")


def _dummy_mean_squared_error(*args, **kwargs):
    return 0.0


sklearn_metrics.mean_squared_error = _dummy_mean_squared_error
sklearn_metrics.mean_absolute_error = _dummy_mean_squared_error
sklearn_metrics.mean_absolute_percentage_error = _dummy_mean_squared_error
sklearn_metrics.r2_score = _dummy_mean_squared_error
sklearn_metrics.explained_variance_score = _dummy_mean_squared_error


def _metrics_getattr(name):
    return _dummy_mean_squared_error


sklearn_metrics.__getattr__ = _metrics_getattr
sys.modules["sklearn.metrics"] = sklearn_metrics
sklearn_module.metrics = sklearn_metrics

sklearn_model_selection = types.ModuleType("sklearn.model_selection")


class _DummyTimeSeriesSplit:
    def split(self, X, y=None, groups=None):
        indices = list(range(len(X))) if hasattr(X, "__len__") else [0]
        yield indices, indices


sklearn_model_selection.TimeSeriesSplit = _DummyTimeSeriesSplit
sys.modules["sklearn.model_selection"] = sklearn_model_selection
sklearn_module.model_selection = sklearn_model_selection

if "scipy" not in sys.modules:
    scipy_module = types.ModuleType("scipy")
    sys.modules["scipy"] = scipy_module
else:
    scipy_module = sys.modules["scipy"]

scipy_sparse = types.ModuleType("scipy.sparse")


def _dummy_csr_matrix(*args, **kwargs):
    return []


scipy_sparse.csr_matrix = _dummy_csr_matrix
scipy_sparse.issparse = lambda matrix: False
sys.modules["scipy.sparse"] = scipy_sparse
scipy_module.sparse = scipy_sparse

if "torch" not in sys.modules:
    torch_module = types.ModuleType("torch")
    sys.modules["torch"] = torch_module
else:
    torch_module = sys.modules["torch"]

if not hasattr(torch_module, "tensor"):
    torch_module.tensor = lambda *args, **kwargs: None
    torch_module.no_grad = types.SimpleNamespace(
        __enter__=lambda self: None,
        __exit__=lambda self, exc_type, exc, tb: False,
    )

torch_cuda = types.ModuleType("torch.cuda")
torch_cuda.is_available = lambda: False
sys.modules["torch.cuda"] = torch_cuda
torch_module.cuda = torch_cuda

torch_nn = types.ModuleType("torch.nn")
torch_module.nn = torch_nn
sys.modules["torch.nn"] = torch_nn

torch_module.device = lambda *args, **kwargs: "cpu"

import full_auto_system
from full_auto_system import (
    AutoRecommendation,
    FullAutoInvestmentSystem,
    RiskManagerAdapter,
)
from models.advanced.risk_management_framework import (
    PortfolioRisk,
    RiskLevel,
)
from models.base.interfaces import PredictionResult


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
    prediction_payload = {
        "predicted_price": prediction_result.prediction,
        "confidence": prediction_result.confidence,
        "accuracy": prediction_result.accuracy,
        "timestamp": prediction_result.timestamp,
        "symbol": prediction_result.symbol,
        "metadata": prediction_result.metadata,
    }
    predictor = SimpleNamespace()
    predictor.predict = MagicMock(return_value=prediction_payload)  # type: ignore[attr-defined]
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
    risk_manager.analyze_risk = MagicMock(return_value=portfolio_risk)  # type: ignore[attr-defined]
    system.risk_manager = risk_manager  # type: ignore[assignment]

    sentiment_analyzer = SimpleNamespace()
    sentiment_payload = {"sentiment_score": 0.25}
    sentiment_analyzer.analyze_news_sentiment = MagicMock(
        return_value=sentiment_payload,
    )  # type: ignore[attr-defined]
    sentiment_analyzer.analyze_sentiment = MagicMock(return_value=sentiment_payload)  # type: ignore[attr-defined]
    system.sentiment_analyzer = sentiment_analyzer  # type: ignore[assignment]

    strategy_payload = {
        "entry_price": 102.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.1,
        "expected_return": 0.12,
        "confidence_score": 0.66,
    }
    strategy_calls = []

    def fake_generate_strategy(self, symbol, price_data, predictions, risk, sentiment):
        strategy_calls.append((symbol, price_data, predictions, risk, sentiment))
        return strategy_payload

    monkeypatch.setattr(
        full_auto_system.StrategyGeneratorAdapter,
        "generate_strategy",
        fake_generate_strategy,
        raising=False,
    )

    data = pd.DataFrame(
        {"Close": [100.0, 102.0], "High": [101.0, 103.0], "Low": [99.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    data.attrs.setdefault("info", {})["longName"] = "Test Corp"

    recommendation = await system._analyze_single_stock("TEST", data)

    assert isinstance(recommendation, AutoRecommendation)
    assert recommendation.entry_price == pytest.approx(data["Close"].iloc[-1])
    assert recommendation.target_price == pytest.approx(
        recommendation.entry_price * (1 + strategy_payload["expected_return"]),
    )
    assert recommendation.stop_loss == pytest.approx(
        recommendation.entry_price * (1 - strategy_payload["stop_loss_pct"]),
    )
    assert recommendation.expected_return == pytest.approx(
        strategy_payload["expected_return"],
    )
    assert recommendation.risk_level == portfolio_risk.risk_level.value

    # Risk-adjusted confidence combines strategy win rate and risk score
    expected_confidence = (
        strategy_payload["confidence_score"]
        + (1.0 - portfolio_risk.total_risk_score)
        + prediction_result.confidence
    ) / 3
    assert recommendation.confidence == pytest.approx(expected_confidence)

    system.predictor.predict.assert_called_once_with("TEST", data)  # type: ignore[attr-defined]
    system.risk_manager.analyze_risk.assert_called_once()  # type: ignore[attr-defined]
    risk_args = system.risk_manager.analyze_risk.call_args[0]  # type: ignore[attr-defined]
    assert risk_args[0] == "TEST"
    assert risk_args[1] is data

    system.sentiment_analyzer.analyze_sentiment.assert_called_once_with("TEST")  # type: ignore[attr-defined]
    assert strategy_calls == [
        ("TEST", data, prediction_payload, portfolio_risk, sentiment_payload),
    ]


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
    prediction_payload = {
        "predicted_price": prediction_result.prediction,
        "confidence": prediction_result.confidence,
        "accuracy": prediction_result.accuracy,
        "timestamp": prediction_result.timestamp,
        "symbol": prediction_result.symbol,
        "metadata": prediction_result.metadata,
    }
    predictor = SimpleNamespace()
    predictor.predict = MagicMock(return_value=prediction_payload)  # type: ignore[attr-defined]
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
    risk_manager.analyze_risk = MagicMock(return_value=portfolio_risk)  # type: ignore[attr-defined]
    system.risk_manager = risk_manager  # type: ignore[assignment]

    sentiment_analyzer = SimpleNamespace()
    sentiment_payload = {"sentiment_score": 0.25}
    sentiment_analyzer.analyze_news_sentiment = MagicMock(
        return_value=sentiment_payload,
    )  # type: ignore[attr-defined]
    sentiment_analyzer.analyze_sentiment = MagicMock(return_value=sentiment_payload)  # type: ignore[attr-defined]
    system.sentiment_analyzer = sentiment_analyzer  # type: ignore[assignment]

    # 空の辞書を返すようにモックを設定
    strategy_calls = []

    def fake_generate_strategy_empty(self, *args, **kwargs):
        strategy_calls.append(args)
        return {}

    monkeypatch.setattr(
        full_auto_system.StrategyGeneratorAdapter,
        "generate_strategy",
        fake_generate_strategy_empty,
        raising=False,
    )

    data = pd.DataFrame(
        {"Close": [100.0, 102.0], "High": [101.0, 103.0], "Low": [99.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    data.attrs.setdefault("info", {})["longName"] = "Test Corp"

    recommendation = await system._analyze_single_stock("TEST", data)

    # recommendation が None であることを確認
    assert recommendation is None

    # generate_strategy が呼び出されたことを確認
    assert len(strategy_calls) == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_run_full_auto_analysis_prefers_highest_return_portfolio(monkeypatch):
    """Regression test ensuring we keep the best performing portfolio."""
    system = FullAutoInvestmentSystem()
    system.settings.target_stocks = {"AAA": "AAA Corp", "BBB": "BBB Corp"}

    base_data = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )

    def fake_get_stock_data(symbol: str, period: str = "2y") -> pd.DataFrame:
        data = base_data.copy()
        data.attrs.setdefault("info", {})["longName"] = f"{symbol} Corp"
        return data

    monkeypatch.setattr(system.data_provider, "get_stock_data", fake_get_stock_data)

    system.portfolio_sizes = [1, 2]

    class DummyOptimizer:
        def optimize_portfolio(self, profiles, target_size=20):
            sorted_profiles = sorted(profiles, key=lambda p: p.symbol)
            return sorted_profiles[:target_size]

    system.optimizer = DummyOptimizer()

    class DummyBacktester:
        def __init__(self, data_provider):
            self.data_provider = data_provider

        def backtest_portfolio(self, selected_symbols):
            return {"return_rate": 5.0 if len(selected_symbols) == 1 else 15.0}

    monkeypatch.setattr(full_auto_system, "PortfolioBacktester", DummyBacktester)

    recommendation = AutoRecommendation(
        symbol="DUMMY",
        company_name="Dummy Corp",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        expected_return=0.1,
        confidence=0.8,
        risk_level="low",
        buy_date=datetime.now(),
        sell_date=datetime.now(),
        reasoning="test",
    )

    async def fake_analyze(symbol: str, data: pd.DataFrame):
        return AutoRecommendation(
            symbol=symbol,
            company_name=f"{symbol} Corp",
            entry_price=recommendation.entry_price,
            target_price=recommendation.target_price,
            stop_loss=recommendation.stop_loss,
            expected_return=recommendation.expected_return,
            confidence=recommendation.confidence,
            risk_level=recommendation.risk_level,
            buy_date=recommendation.buy_date,
            sell_date=recommendation.sell_date,
            reasoning=recommendation.reasoning,
        )

    monkeypatch.setattr(system, "_analyze_single_stock", fake_analyze)

    recommendations = await system.run_full_auto_analysis()

    assert {rec.symbol for rec in recommendations} == {"AAA", "BBB"}
    assert all(rec.company_name.endswith("Corp") for rec in recommendations)


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
