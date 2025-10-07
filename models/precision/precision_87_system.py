#!/usr/bin/env python3
"""Baseline precision system with transparent evaluation.

This replaces the previous pseudo-"87% precision" stack with a lightweight
momentum/mean-reversion hybrid that can report genuine hit-rate statistics
based on recent data.  It keeps the external interface identical so the
surrounding trading strategy can continue to consume prediction metadata
without changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..core.base_predictor import BaseStockPredictor
from ..core.interfaces import PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class SignalEvaluation:
    """Container for recent signal quality statistics."""

    accuracy: float
    sample_size: int
    avg_positive_return: float
    avg_negative_return: float

    @property
    def confidence(self) -> float:
        """Translate accuracy into a 0–1 confidence score.

        We cap the range so downstream position sizing never sees a perfect
        (1.0) signal and keeps some capital discipline.
        """

        if self.sample_size == 0:
            return 0.1
        # Blend hit-rate with how much data we have: more samples -> more trust.
        size_bonus = min(self.sample_size / 120.0, 1.0)
        base = self.accuracy / 100.0
        score = 0.2 + 0.6 * base + 0.2 * size_bonus
        return float(np.clip(score, 0.05, 0.95))


class Precision87BreakthroughSystem(BaseStockPredictor):
    """Momentum/mean-reversion hybrid with verifiable evaluation."""

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 40,
        evaluation_window: int = 160,
        evaluation_horizon: int = 5,
    ) -> None:
        self.short_window = short_window
        self.long_window = max(long_window, short_window + 1)
        self.evaluation_window = evaluation_window
        self.evaluation_horizon = evaluation_horizon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, symbol: str) -> PredictionResult:
        result = self.predict_with_87_precision(symbol)
        return PredictionResult(
            prediction=result["final_prediction"],
            confidence=result["final_confidence"],
            accuracy=result["final_accuracy"],
            timestamp=pd.Timestamp.utcnow().to_pydatetime(),
            symbol=symbol,
            metadata=result,
        )

    def predict_batch(self, symbols: list[str]) -> list[PredictionResult]:
        return [self.predict(symbol) for symbol in symbols]

    def get_confidence(self, symbol: str) -> float:
        result = self.predict_with_87_precision(symbol)
        return float(result.get("final_confidence", 0.0))

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "BaselineMomentumReversion",
            "version": "2.0.0",
            "short_window": self.short_window,
            "long_window": self.long_window,
            "evaluation_window": self.evaluation_window,
            "evaluation_horizon": self.evaluation_horizon,
        }

    # ------------------------------------------------------------------
    # Core prediction logic
    # ------------------------------------------------------------------
    def predict_with_87_precision(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            from data.stock_data import StockDataProvider

            provider = StockDataProvider()
            history = self._load_history(provider, symbol, start, end)
            if history is None:
                return self._default_prediction(symbol, "insufficient_data")

            enriched = provider.calculate_technical_indicators(history)
            signal_series = self._build_signal_series(enriched)
            if signal_series is None or signal_series.empty:
                return self._default_prediction(symbol, "signal_generation_failed")

            evaluation = self._evaluate_signal(enriched, signal_series)
            last_signal = float(signal_series.iloc[-1])
            last_price = float(enriched["Close"].iloc[-1])
            predicted_change_rate = float(np.clip(last_signal, -0.12, 0.12))
            predicted_price = last_price * (1.0 + predicted_change_rate)

            confidence = evaluation.confidence
            # Blend volatility to avoid overconfident sizing when signal is small.
            recent_vol = enriched["Close"].pct_change().rolling(self.long_window).std().iloc[-1]
            if np.isfinite(recent_vol) and recent_vol > 0:
                confidence *= float(np.clip(abs(predicted_change_rate) / (recent_vol * 2.0), 0.3, 1.2))
            confidence = float(np.clip(confidence, 0.05, 0.95))

            accuracy = float(evaluation.accuracy)
            precision_flag = accuracy >= 87.0 and evaluation.sample_size >= 30

            return {
                "symbol": symbol,
                "final_prediction": float(predicted_price),
                "final_confidence": confidence,
                "final_accuracy": accuracy,
                "precision_87_achieved": precision_flag,
                "current_price": last_price,
                "predicted_change_rate": predicted_change_rate,
                "evaluation": {
                    "sample_size": evaluation.sample_size,
                    "avg_positive_return": evaluation.avg_positive_return,
                    "avg_negative_return": evaluation.avg_negative_return,
                },
            }

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Prediction failed for %s", symbol)
            return self._default_prediction(symbol, str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_history(
        self,
        provider,
        symbol: str,
        start: Optional[str],
        end: Optional[str],
    ) -> Optional[pd.DataFrame]:
        try:
            if start or end:
                history = provider.get_stock_data(symbol, start=start, end=end)
            else:
                history = provider.get_stock_data(symbol, period="2y")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to fetch history for %s: %s", symbol, exc)
            return None

        if history is None or len(history) < self.long_window + self.evaluation_horizon + 5:
            return None

        return history.copy()

    def _build_signal_series(self, data: pd.DataFrame) -> Optional[pd.Series]:
        close = data["Close"].astype(float)
        if close.isna().any():
            close = close.ffill()

        short_ma = close.rolling(self.short_window).mean()
        long_ma = close.rolling(self.long_window).mean()
        if long_ma.isna().all():
            return None

        momentum_component = short_ma - long_ma
        momentum_component = momentum_component / close

        if "RSI" in data.columns:
            rsi = data["RSI"].fillna(50.0)
        else:
            rsi = self._calculate_rsi(close)
        rsi_component = (50.0 - rsi) / 50.0

        signal = 0.6 * momentum_component + 0.4 * rsi_component
        signal_values = np.tanh(signal.to_numpy())
        return pd.Series(signal_values, index=data.index, name="baseline_signal")

    def _evaluate_signal(
        self, data: pd.DataFrame, signal_series: pd.Series,
    ) -> SignalEvaluation:
        signal = signal_series.dropna().iloc[-self.evaluation_window :].copy()
        if signal.empty:
            return SignalEvaluation(accuracy=0.0, sample_size=0, avg_positive_return=0.0, avg_negative_return=0.0)

        forward_returns = (
            data["Close"].pct_change(self.evaluation_horizon).shift(-self.evaluation_horizon)
        )
        aligned = signal.index.intersection(forward_returns.index)
        signal = signal.loc[aligned]
        returns = forward_returns.loc[aligned]
        mask = (~signal.isna()) & (~returns.isna())
        if mask.sum() == 0:
            return SignalEvaluation(accuracy=0.0, sample_size=0, avg_positive_return=0.0, avg_negative_return=0.0)

        signed_signal = np.sign(signal[mask])
        signed_returns = np.sign(returns[mask])
        hits = (signed_signal == signed_returns).sum()
        sample_size = int(mask.sum())
        accuracy = (hits / sample_size) * 100.0

        positive_mask = signed_signal > 0
        negative_mask = signed_signal < 0
        avg_positive = float(returns[mask][positive_mask].mean()) if positive_mask.any() else 0.0
        avg_negative = float(returns[mask][negative_mask].mean()) if negative_mask.any() else 0.0

        return SignalEvaluation(
            accuracy=float(accuracy),
            sample_size=sample_size,
            avg_positive_return=avg_positive,
            avg_negative_return=avg_negative,
        )

    @staticmethod
    def _calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = -delta.clip(upper=0).rolling(window).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def _default_prediction(self, symbol: str, reason: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "final_prediction": 0.0,
            "final_confidence": 0.1,
            "final_accuracy": 0.0,
            "precision_87_achieved": False,
            "current_price": 0.0,
            "predicted_change_rate": 0.0,
            "error": reason,
            "evaluation": {
                "sample_size": 0,
                "avg_positive_return": 0.0,
                "avg_negative_return": 0.0,
            },
        }


class MockMetaLearner:  # Backwards compatibility for modules still importing the symbol.
    def create_symbol_profile(self, symbol, data):  # pragma: no cover - legacy shim
        logger.warning("MockMetaLearner is deprecated; real meta-learning has been removed.")
        return {"current_price": data["Close"].iloc[-1] if not data.empty else 0.0}

    def adapt_model_parameters(self, symbol, profile, params):  # pragma: no cover - legacy shim
        return {"adapted_prediction": params.get("prediction", 0.0), "adapted_confidence": 0.0}


class MockDQNAgent:  # Backwards compatibility shim.
    def get_trading_signal(self, symbol, data):  # pragma: no cover - legacy shim
        logger.warning("MockDQNAgent is deprecated; reinforcement learner removed in baseline model.")
        return {"signal_strength": 0.0, "confidence": 0.0}
