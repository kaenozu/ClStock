"""完全自動投資システム向けアダプタ群。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from analysis.sentiment_analyzer import MarketSentimentAnalyzer
from models.advanced.risk_management_framework import (
    PortfolioRisk,
    RiskLevel,
    RiskManager,
)
from models.advanced.trading_strategy_generator import (
    ActionType,
    SignalGenerator,
    StrategyGenerator,
)
from models.base.interfaces import PredictionResult
from models.hybrid.hybrid_predictor import HybridStockPredictor

__all__ = [
    "AutoRecommendation",
    "RiskAssessment",
    "HybridPredictorAdapter",
    "SentimentAnalyzerAdapter",
    "RiskManagerAdapter",
    "StrategyGeneratorAdapter",
]


class AutoRecommendation:
    """自動推奨結果を表現する単純なコンテナ。"""

    def __init__(
        self,
        symbol: str,
        company_name: str,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        expected_return: float,
        confidence: float,
        risk_level: str,
        buy_date: datetime,
        sell_date: datetime,
        reasoning: str,
    ) -> None:
        self.symbol = symbol
        self.company_name = company_name
        self.entry_price = entry_price
        self.target_price = target_price
        self.stop_loss = stop_loss
        self.expected_return = expected_return
        self.confidence = confidence
        self.risk_level = risk_level
        self.buy_date = buy_date
        self.sell_date = sell_date
        self.reasoning = reasoning


@dataclass
class RiskAssessment:
    """Adapter friendly risk analysis result."""

    risk_score: float
    risk_level: RiskLevel
    max_safe_position_size: float
    recommendations: List[str]
    raw: Optional[PortfolioRisk] = None


class HybridPredictorAdapter:
    """Wrap HybridStockPredictor to expose the legacy predict interface."""

    def __init__(self, predictor: Optional[HybridStockPredictor] = None) -> None:
        self._predictor = predictor or HybridStockPredictor()

    def predict(self, symbol: str, data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        result = self._predictor.predict(symbol)

        if not isinstance(result, PredictionResult):
            logging.getLogger(__name__).warning(
                "%s: HybridStockPredictor.predict が PredictionResult 以外の型を返しました: %s",
                symbol,
                type(result),
            )
            return {}

        return {
            "predicted_price": float(result.prediction),
            "confidence": float(result.confidence),
            "accuracy": float(result.accuracy),
            "metadata": dict(result.metadata),
            "symbol": result.symbol,
            "timestamp": result.timestamp,
        }


class SentimentAnalyzerAdapter:
    """Provide an analyze_sentiment shim for the news sentiment analyzer."""

    def __init__(self, analyzer: Optional[MarketSentimentAnalyzer] = None) -> None:
        self._analyzer = analyzer or MarketSentimentAnalyzer()

    def analyze_sentiment(self, symbol: str) -> Dict[str, Any]:
        try:
            sentiment = self._analyzer.analyze_news_sentiment(symbol)
        except AttributeError:
            sentiment = self._analyzer.analyze_sentiment(symbol)  # type: ignore[attr-defined]
        except Exception:
            sentiment = {}

        sentiment_score = 0.0
        if isinstance(sentiment, dict):
            sentiment_score = float(sentiment.get("sentiment_score", 0.0))
        else:
            sentiment = {}

        sentiment.setdefault("sentiment_score", sentiment_score)
        return sentiment


class RiskManagerAdapter:
    """Translate portfolio level risk outputs into the legacy structure."""

    def __init__(self, manager: Optional[RiskManager] = None) -> None:
        self._manager = manager or RiskManager()
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_portfolio_risk(
        self,
        portfolio_data: Dict[str, Any],
        price_map: Dict[str, pd.DataFrame],
    ) -> Optional[RiskAssessment]:
        try:
            return self._manager.analyze_portfolio_risk(portfolio_data, price_map)
        except AttributeError:
            self.logger.debug(
                "RiskManager missing analyze_portfolio_risk, falling back",
                exc_info=True,
            )
            if not price_map:
                return None

            symbol, price_data = next(iter(price_map.items()))
            return self.analyze_risk(symbol, price_data, {})
        except Exception:
            self.logger.exception("Portfolio risk analysis failed")
            return None

    def analyze_risk(
        self,
        symbol: str,
        price_data: Optional[pd.DataFrame],
        predictions: Dict[str, Any],
    ) -> Optional[RiskAssessment]:
        if price_data is None or price_data.empty:
            return None

        try:
            portfolio_risk = self._manager.analyze_portfolio_risk(
                {"positions": {symbol: float(price_data["Close"].iloc[-1])}},
                {symbol: price_data},
            )
        except AttributeError:
            self.logger.debug(
                "RiskManager missing analyze_portfolio_risk, falling back",
                exc_info=True,
            )
            portfolio_risk = self._manager.analyze_risk(price_data, predictions)  # type: ignore[attr-defined]
        except Exception:
            self.logger.exception("Risk analysis failed for %s", symbol)
            return None

        if portfolio_risk is None:
            return None

        if isinstance(portfolio_risk, RiskAssessment):
            return portfolio_risk

        total_score = getattr(portfolio_risk, "total_risk_score", 2.0)
        min_risk_score = 1.0
        max_risk_score = 4.0
        normalized_score = max(
            0.0,
            min(
                (float(total_score) - min_risk_score)
                / (max_risk_score - min_risk_score),
                1.0,
            ),
        )
        risk_level = getattr(portfolio_risk, "risk_level", RiskLevel.MEDIUM)
        max_position = getattr(portfolio_risk, "max_safe_position_size", 0.05)
        recommendations = getattr(portfolio_risk, "recommendations", [])

        return RiskAssessment(
            risk_score=float(normalized_score),
            risk_level=risk_level,
            max_safe_position_size=float(max_position),
            recommendations=list(recommendations),
            raw=portfolio_risk if isinstance(portfolio_risk, PortfolioRisk) else None,
        )


class StrategyGeneratorAdapter:
    """Leverage the advanced generator to produce legacy-friendly strategies."""

    def __init__(
        self,
        generator: Optional[StrategyGenerator] = None,
        signal_generator: Optional[SignalGenerator] = None,
    ) -> None:
        self._generator = generator or StrategyGenerator()
        self._signal_generator = signal_generator or SignalGenerator()
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_strategy(
        self,
        symbol: str,
        price_data: Optional[pd.DataFrame],
        predictions: Dict[str, Any],
        risk_assessment: Optional[RiskAssessment],
        sentiment: Dict[str, Any],
    ) -> Dict[str, Any]:
        if price_data is None or price_data.empty:
            return {}

        sentiment_score = 0.0
        if isinstance(sentiment, dict):
            sentiment_score = float(sentiment.get("sentiment_score", 0.0))

        sentiment_payload = {"current_sentiment": {"score": sentiment_score}}

        best_signal = None
        for strategy in self._collect_strategies(symbol, price_data):
            try:
                signals = self._signal_generator.generate_signals(
                    symbol,
                    price_data,
                    strategy,
                    sentiment_payload,
                )
            except Exception:
                self.logger.debug(
                    "Signal generation failed for %s strategy",
                    strategy.name,
                    exc_info=True,
                )
                continue

            for signal in signals:
                if signal.action != ActionType.BUY:
                    continue
                if best_signal is None or signal.confidence > best_signal.confidence:
                    best_signal = signal

        if best_signal is None:
            return {}

        entry_price = float(best_signal.entry_price)
        target_price = float(best_signal.take_profit or entry_price)
        stop_loss = float(best_signal.stop_loss or entry_price)
        expected_return = 0.0
        if entry_price:
            expected_return = (target_price - entry_price) / entry_price

        strategy_dict = {
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "confidence_score": float(best_signal.confidence),
            "expected_return": float(expected_return),
            "reasoning": best_signal.reasoning,
            "metadata": best_signal.metadata,
        }

        if risk_assessment:
            strategy_dict["max_safe_position_size"] = (
                risk_assessment.max_safe_position_size
            )

        return strategy_dict

    def _collect_strategies(self, symbol: str, price_data: pd.DataFrame) -> List[Any]:
        candidate_methods = [
            getattr(self._generator, "generate_momentum_strategy", None),
            getattr(self._generator, "generate_mean_reversion_strategy", None),
            getattr(self._generator, "generate_breakout_strategy", None),
        ]

        strategies: List[Any] = []
        for method in candidate_methods:
            if not callable(method):
                continue
            try:
                strategy = method(symbol, price_data)
            except Exception:
                self.logger.debug("Strategy generation failed", exc_info=True)
                continue
            if strategy:
                strategies.append(strategy)

        return strategies
