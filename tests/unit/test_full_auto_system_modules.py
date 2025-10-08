from datetime import datetime
from unittest.mock import Mock, patch

import pytest

import importlib.machinery
import pandas as pd
import sys
import types
from models.advanced.risk_management_framework import (
    PortfolioRisk,
    RiskLevel,
)
from models.advanced.trading_strategy_generator import (
    ActionType,
    StrategyType,
    TradingSignal,
    TradingStrategy,
)
from models.base.interfaces import PredictionResult
from trading.tse.analysis import StockProfile


def _install_sklearn_dependency_stubs() -> None:
    if "scipy" in sys.modules:
        return

    scipy_module = types.ModuleType("scipy")
    scipy_module.__path__ = []  # type: ignore[attr-defined]
    scipy_module.__spec__ = importlib.machinery.ModuleSpec(  # type: ignore[attr-defined]
        "scipy",
        loader=None,
        is_package=True,
    )
    sparse_module = types.ModuleType("scipy.sparse")
    sparse_module.__path__ = []  # type: ignore[attr-defined]
    sparse_module.__spec__ = importlib.machinery.ModuleSpec(  # type: ignore[attr-defined]
        "scipy.sparse",
        loader=None,
        is_package=True,
    )

    class _DummyCSRMatrix:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def _dummy_issparse(_value: object) -> bool:
        return False

    sparse_module.csr_matrix = _DummyCSRMatrix  # type: ignore[attr-defined]
    sparse_module.issparse = _dummy_issparse  # type: ignore[attr-defined]

    scipy_module.sparse = sparse_module  # type: ignore[attr-defined]

    sys.modules.setdefault("scipy", scipy_module)
    sys.modules.setdefault("scipy.sparse", sparse_module)


_install_sklearn_dependency_stubs()


def _make_trading_strategy(name: str = "momentum") -> TradingStrategy:
    return TradingStrategy(
        name=name,
        strategy_type=StrategyType.MOMENTUM,
        parameters={},
        entry_conditions=[],
        exit_conditions=[],
        risk_management={
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.1,
            "max_position_size": 0.2,
        },
        expected_return=0.12,
        max_drawdown=0.08,
        sharpe_ratio=1.4,
        win_rate=0.6,
        created_at=datetime.now(),
    )


@pytest.fixture
def sample_processed_data():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    base_frame = pd.DataFrame(
        {
            "Close": [100, 102, 101, 103, 104],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=index,
    )
    return {"AAA": base_frame, "BBB": base_frame * 1.01}


class TestFullAutoInvestmentSystemInitialisation:
    def test_initialises_with_new_component_classes(self):
        with patch("full_auto_system.StockDataProvider") as mock_provider, patch(
            "full_auto_system.HybridStockPredictor",
        ) as mock_predictor, patch(
            "full_auto_system.PortfolioOptimizer",
        ) as mock_optimizer, patch(
            "full_auto_system.MarketSentimentAnalyzer",
        ) as mock_sentiment, patch(
            "full_auto_system.StrategyGenerator",
        ) as mock_strategy, patch(
            "full_auto_system.RiskManager"
        ) as mock_risk:
            from full_auto_system import FullAutoInvestmentSystem

            system = FullAutoInvestmentSystem()

        assert mock_provider.called
        assert mock_predictor.called
        assert mock_optimizer.called
        assert mock_sentiment.called
        assert mock_strategy.called
        assert mock_risk.called
        # Optimizer helper should be available for later use
        assert hasattr(system, "_optimize_portfolio")


class TestPortfolioOptimizationHelper:
    def test_optimizer_helper_returns_selected_stocks_structure(
        self,
        sample_processed_data,
    ):
        class _DynamicModule(types.ModuleType):
            def __init__(self, name: str) -> None:
                super().__init__(name)
                self.__path__ = []  # type: ignore[attr-defined]

        class _SparseModule(_DynamicModule):
            def __init__(self) -> None:
                super().__init__("scipy.sparse")

                class _DummyCSRMatrix:
                    def __init__(self, *args, **kwargs):
                        self.args = args
                        self.kwargs = kwargs

                def _dummy_issparse(_value: object) -> bool:
                    return False

                self.csr_matrix = _DummyCSRMatrix  # type: ignore[attr-defined]
                self.csc_matrix = _DummyCSRMatrix  # type: ignore[attr-defined]
                self.coo_matrix = _DummyCSRMatrix  # type: ignore[attr-defined]
                self.lil_matrix = _DummyCSRMatrix  # type: ignore[attr-defined]
                self.dok_matrix = _DummyCSRMatrix  # type: ignore[attr-defined]
                self.issparse = _dummy_issparse  # type: ignore[attr-defined]
                self.linalg = _DynamicModule("scipy.sparse.linalg")

                class _DummyLinearOperator:
                    pass

                self.linalg.LinearOperator = _DummyLinearOperator  # type: ignore[attr-defined]
                sys.modules[self.linalg.__name__] = self.linalg

            def __getattr__(self, item: str):
                if item.endswith("_matrix"):
                    return self.csr_matrix
                module_name = f"{self.__name__}.{item}"
                module = _DynamicModule(module_name)
                sys.modules[module_name] = module
                setattr(self, item, module)
                return module

        class _SpecialModule(_DynamicModule):
            def __init__(self) -> None:
                super().__init__("scipy.special")

                def _dummy_expit(value):
                    return value

                self.expit = _dummy_expit  # type: ignore[attr-defined]
                self.gammaln = lambda *_args, **_kwargs: 0.0  # type: ignore[attr-defined]
                self.boxcox = lambda x, *_args, **_kwargs: x  # type: ignore[attr-defined]
                self.comb = lambda n, k, exact=False: 1 if exact else 1.0  # type: ignore[attr-defined]

        class _ScipyModule(_DynamicModule):
            def __init__(self) -> None:
                super().__init__("scipy")
                self.__version__ = "0.0.0"
                self.sparse = _SparseModule()
                self.special = _SpecialModule()
                sys.modules[self.sparse.__name__] = self.sparse
                sys.modules[self.special.__name__] = self.special

                def _dummy_line_search(*_args, **_kwargs):
                    return None

                self.optimize = _DynamicModule("scipy.optimize")
                line_search_module = _DynamicModule("scipy.optimize.linesearch")
                line_search_module.line_search_wolfe1 = _dummy_line_search  # type: ignore[attr-defined]
                line_search_module.line_search_wolfe2 = _dummy_line_search  # type: ignore[attr-defined]

                private_line_search = _DynamicModule("scipy.optimize._linesearch")
                private_line_search.line_search_wolfe1 = _dummy_line_search  # type: ignore[attr-defined]
                private_line_search.line_search_wolfe2 = _dummy_line_search  # type: ignore[attr-defined]

                self.optimize.linesearch = line_search_module  # type: ignore[attr-defined]
                self.optimize._linesearch = private_line_search  # type: ignore[attr-defined]
                self.optimize.linear_sum_assignment = (  # type: ignore[attr-defined]
                    lambda *_args, **_kwargs: ([], [])
                )

                sys.modules[self.optimize.__name__] = self.optimize
                sys.modules[line_search_module.__name__] = line_search_module
                sys.modules[private_line_search.__name__] = private_line_search

                def _dummy_trapezoid(*_args, **_kwargs):
                    return 0.0

                self.integrate = _DynamicModule("scipy.integrate")
                self.integrate.trapezoid = _dummy_trapezoid  # type: ignore[attr-defined]
                self.integrate.trapz = _dummy_trapezoid  # type: ignore[attr-defined]
                sys.modules[self.integrate.__name__] = self.integrate

                class _DummyBSpline:
                    def __init__(self, *args, **kwargs):
                        self.args = args
                        self.kwargs = kwargs

                self.interpolate = _DynamicModule("scipy.interpolate")
                self.interpolate.BSpline = _DummyBSpline  # type: ignore[attr-defined]
                sys.modules[self.interpolate.__name__] = self.interpolate

                self.spatial = _DynamicModule("scipy.spatial")
                distance_module = _DynamicModule("scipy.spatial.distance")
                distance_module.cdist = lambda *_args, **_kwargs: 0.0  # type: ignore[attr-defined]
                distance_module.pdist = lambda *_args, **_kwargs: 0.0  # type: ignore[attr-defined]
                self.spatial.distance = distance_module  # type: ignore[attr-defined]
                sys.modules[self.spatial.__name__] = self.spatial
                sys.modules[distance_module.__name__] = distance_module

            def __getattr__(self, item: str) -> types.ModuleType:
                module_name = f"scipy.{item}"
                module = _DynamicModule(module_name)
                sys.modules[module_name] = module
                setattr(self, item, module)
                return module

        scipy_stub = _ScipyModule()

        module_patch = patch.dict(
            "sys.modules",
            {
                "scipy": scipy_stub,
                "scipy.sparse": scipy_stub.sparse,
                "scipy.sparse.linalg": scipy_stub.sparse.linalg,
                "scipy.special": scipy_stub.special,
                "scipy.optimize": scipy_stub.optimize,
                "scipy.optimize.linesearch": scipy_stub.optimize.linesearch,
                "scipy.optimize._linesearch": scipy_stub.optimize._linesearch,
                "scipy.integrate": scipy_stub.integrate,
                "scipy.interpolate": scipy_stub.interpolate,
                "scipy.spatial": scipy_stub.spatial,
                "scipy.spatial.distance": scipy_stub.spatial.distance,
            },
        )

        hybrid_package = types.ModuleType("models.hybrid")
        hybrid_package.__path__ = []  # type: ignore[attr-defined]

        hybrid_predictor_module = types.ModuleType("models.hybrid.hybrid_predictor")

        class _DummyHybridPredictor:
            def predict(self, symbol: str):
                return PredictionResult(
                    prediction=0.0,
                    confidence=0.5,
                    accuracy=0.5,
                    timestamp=datetime.now(),
                    symbol=symbol,
                    metadata={},
                )

        hybrid_predictor_module.HybridStockPredictor = _DummyHybridPredictor
        sys.modules[hybrid_package.__name__] = hybrid_package
        sys.modules[hybrid_predictor_module.__name__] = hybrid_predictor_module

        predictor_patch = patch.dict(
            "sys.modules",
            {
                "models.hybrid": hybrid_package,
                "models.hybrid.hybrid_predictor": hybrid_predictor_module,
            },
        )

        with module_patch, predictor_patch, patch(
            "full_auto_system.StockDataProvider"
        ), patch(
            "full_auto_system.HybridStockPredictor",
        ), patch(
            "full_auto_system.MarketSentimentAnalyzer"
        ), patch(
            "full_auto_system.StrategyGenerator",
        ), patch(
            "full_auto_system.RiskManager"
        ), patch(
            "full_auto_system.PortfolioBacktester",
        ) as mock_backtester:
            from full_auto_system import FullAutoInvestmentSystem

            system = FullAutoInvestmentSystem()

        mock_backtester.return_value.backtest_portfolio.return_value = {}

        fake_selected = [
            StockProfile(
                symbol="AAA",
                sector="tech",
                market_cap=1.0,
                volatility=0.2,
                profit_potential=0.3,
                diversity_score=0.4,
                combined_score=0.9,
            ),
        ]

        with patch.object(
            system.optimizer,
            "optimize_portfolio",
            return_value=fake_selected,
        ) as mock_optimize:
            result = system._optimize_portfolio(sample_processed_data)

        assert result["selected_stocks"] == ["AAA"]

        call_args, _ = mock_optimize.call_args
        assert len(call_args) == 1
        sent_profiles = call_args[0]
        assert all(isinstance(profile, StockProfile) for profile in sent_profiles)
        assert {profile.symbol for profile in sent_profiles} == set(
            sample_processed_data.keys(),
        )


class TestHybridPredictorAdapter:
    def test_predictor_adapter_maps_prediction_result(self, sample_processed_data):
        prediction = PredictionResult(
            prediction=123.45,
            confidence=0.82,
            accuracy=0.77,
            timestamp=datetime.now(),
            symbol="AAA",
            metadata={"mode": "auto"},
        )

        mock_predictor = Mock()
        mock_predictor.predict.return_value = prediction

        from full_auto_system import HybridPredictorAdapter

        adapter = HybridPredictorAdapter(predictor=mock_predictor)
        result = adapter.predict("AAA", sample_processed_data["AAA"])

        mock_predictor.predict.assert_called_once_with("AAA")
        assert result["predicted_price"] == pytest.approx(prediction.prediction)
        assert result["confidence"] == pytest.approx(prediction.confidence)
        assert result["accuracy"] == pytest.approx(prediction.accuracy)
        assert result["metadata"] == prediction.metadata


class TestRiskManagerAdapter:
    def test_risk_manager_adapter_normalises_score(self, sample_processed_data):
        portfolio_risk = PortfolioRisk(
            total_risk_score=2.5,
            risk_level=RiskLevel.MEDIUM,
            individual_metrics={},
            risk_breakdown={},
            recommendations=["keep monitoring"],
            max_safe_position_size=0.08,
            timestamp=datetime.now(),
        )

        mock_manager = Mock()
        mock_manager.analyze_portfolio_risk.return_value = portfolio_risk

        from full_auto_system import RiskManagerAdapter

        adapter = RiskManagerAdapter(manager=mock_manager)
        predictions = {"predicted_price": 120.0}

        result = adapter.analyze_risk("AAA", sample_processed_data["AAA"], predictions)

        mock_manager.analyze_portfolio_risk.assert_called_once()
        expected_score = (portfolio_risk.total_risk_score - 1.0) / 3.0
        assert result.risk_score == pytest.approx(expected_score)
        assert result.risk_level == portfolio_risk.risk_level
        assert result.recommendations == portfolio_risk.recommendations


class TestStrategyGeneratorAdapter:
    def test_strategy_adapter_selects_best_buy_signal(self, sample_processed_data):
        mock_generator = Mock()
        mock_signal_generator = Mock()

        trading_strategy = _make_trading_strategy()
        mock_generator.generate_momentum_strategy.return_value = trading_strategy
        mock_generator.generate_mean_reversion_strategy.return_value = None
        mock_generator.generate_breakout_strategy.return_value = None

        buy_signal = TradingSignal(
            symbol="AAA",
            action=ActionType.BUY,
            confidence=0.75,
            entry_price=100.0,
            stop_loss=94.0,
            take_profit=112.0,
            position_size=0.2,
            reasoning="test",
            timestamp=datetime.now(),
            metadata={},
        )
        weaker_signal = TradingSignal(
            symbol="AAA",
            action=ActionType.BUY,
            confidence=0.55,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=108.0,
            position_size=0.2,
            reasoning="weak",
            timestamp=datetime.now(),
            metadata={},
        )

        mock_signal_generator.generate_signals.return_value = [
            weaker_signal,
            buy_signal,
        ]

        from full_auto_system import RiskAssessment, StrategyGeneratorAdapter

        risk_assessment = RiskAssessment(
            risk_score=0.3,
            risk_level=RiskLevel.LOW,
            max_safe_position_size=0.08,
            recommendations=["ok"],
            raw=None,
        )

        adapter = StrategyGeneratorAdapter(
            generator=mock_generator,
            signal_generator=mock_signal_generator,
        )

        sentiment = {"sentiment_score": 0.25}
        predictions = {"predicted_price": 110.0}

        result = adapter.generate_strategy(
            "AAA",
            sample_processed_data["AAA"],
            predictions,
            risk_assessment,
            sentiment,
        )

        mock_generator.generate_momentum_strategy.assert_called_once()
        mock_signal_generator.generate_signals.assert_called()
        assert result["entry_price"] == pytest.approx(buy_signal.entry_price)
        assert result["target_price"] == pytest.approx(buy_signal.take_profit)
        assert result["stop_loss"] == pytest.approx(buy_signal.stop_loss)
        expected_return = (
            buy_signal.take_profit - buy_signal.entry_price
        ) / buy_signal.entry_price
        assert result["expected_return"] == pytest.approx(expected_return)
