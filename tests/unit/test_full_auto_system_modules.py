from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from trading.tse.analysis import StockProfile
from models.base.interfaces import PredictionResult
from models.advanced.risk_management_framework import (
    PortfolioRisk,
    RiskLevel,
)
from models.advanced.trading_strategy_generator import (
    TradingStrategy,
    StrategyType,
    TradingSignal,
    ActionType,
)


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
            "full_auto_system.HybridStockPredictor"
        ) as mock_predictor, patch(
            "full_auto_system.PortfolioOptimizer"
        ) as mock_optimizer, patch(
            "full_auto_system.MarketSentimentAnalyzer"
        ) as mock_sentiment, patch(
            "full_auto_system.StrategyGenerator"
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
    def test_optimizer_helper_returns_selected_stocks_structure(self, sample_processed_data):
        with patch("full_auto_system.StockDataProvider"), patch(
            "full_auto_system.HybridStockPredictor"
        ), patch("full_auto_system.MarketSentimentAnalyzer"), patch(
            "full_auto_system.StrategyGenerator"
        ), patch("full_auto_system.RiskManager"):
            from full_auto_system import FullAutoInvestmentSystem

            system = FullAutoInvestmentSystem()

        fake_selected = [
            StockProfile(
                symbol="AAA",
                sector="tech",
                market_cap=1.0,
                volatility=0.2,
                profit_potential=0.3,
                diversity_score=0.4,
                combined_score=0.9,
            )
        ]

        with patch.object(
            system.optimizer,
            "optimize_portfolio",
            return_value=fake_selected,
        ) as mock_optimize:
            result = system._optimize_portfolio(sample_processed_data)

        assert result == {"selected_stocks": ["AAA"]}

        call_args, _ = mock_optimize.call_args
        assert len(call_args) == 1
        sent_profiles = call_args[0]
        assert all(isinstance(profile, StockProfile) for profile in sent_profiles)
        assert {profile.symbol for profile in sent_profiles} == set(
            sample_processed_data.keys()
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

        result = adapter.analyze_risk(
            "AAA", sample_processed_data["AAA"], predictions
        )

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

        mock_signal_generator.generate_signals.return_value = [weaker_signal, buy_signal]

        from full_auto_system import RiskAssessment, StrategyGeneratorAdapter

        risk_assessment = RiskAssessment(
            risk_score=0.3,
            risk_level=RiskLevel.LOW,
            max_safe_position_size=0.08,
            recommendations=["ok"],
            raw=None,
        )

        adapter = StrategyGeneratorAdapter(
            generator=mock_generator, signal_generator=mock_signal_generator
        )

        sentiment = {"sentiment_score": 0.25}
        predictions = {"predicted_price": 110.0}

        result = adapter.generate_strategy(
            "AAA", sample_processed_data["AAA"], predictions, risk_assessment, sentiment
        )

        mock_generator.generate_momentum_strategy.assert_called_once()
        mock_signal_generator.generate_signals.assert_called()
        assert result["entry_price"] == pytest.approx(buy_signal.entry_price)
        assert result["target_price"] == pytest.approx(buy_signal.take_profit)
        assert result["stop_loss"] == pytest.approx(buy_signal.stop_loss)
        expected_return = (buy_signal.take_profit - buy_signal.entry_price) / buy_signal.entry_price
        assert result["expected_return"] == pytest.approx(expected_return)
