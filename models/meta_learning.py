"""Meta-learning optimizers for adaptive stock prediction."""

import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetaLearningOptimizer:
    """Unified meta-learning optimizer with symbol profiling and adaptation."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.symbol_profiles: Dict[str, Dict[str, Any]] = {}
        self.adaptation_memory: Dict[str, Dict[str, Any]] = {}
        self.model_performance_history: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.meta_features: Dict[str, Dict[str, float]] = {}
        self.best_model_for_symbol: Dict[str, str] = {}

    def extract_meta_features(
        self, symbol: str, data: pd.DataFrame,
    ) -> Dict[str, float]:
        """Extract meta features that describe symbol specific characteristics."""
        if data.empty:
            return {}
        meta_features = {
            "data_length": float(len(data)),
            "missing_ratio": float(
                data.isnull().sum().sum() / max(data.shape[0] * data.shape[1], 1),
            ),
            "price_volatility": float(data["Close"].std() / data["Close"].mean()),
            "price_trend": float(
                (data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0],
            ),
            "price_skewness": float(data["Close"].skew()),
            "price_kurtosis": float(data["Close"].kurtosis()),
            "volume_volatility": float(data["Volume"].std() / data["Volume"].mean()),
            "volume_trend": float(
                (data["Volume"].iloc[-20:].mean() - data["Volume"].iloc[:20].mean())
                / max(data["Volume"].iloc[:20].mean(), 1),
            ),
            "rsi_avg": float(data.get("RSI", pd.Series([50])).mean()),
            "macd_trend": float(data.get("MACD", pd.Series([0])).tail(10).mean()),
            "sector_correlation": 0.5,
            "market_cap_category": 1.0,
        }
        self.meta_features[symbol] = meta_features
        return meta_features

    def select_best_model(self, symbol: str, data: pd.DataFrame) -> str:
        """Select the most suitable model based on historical performance and meta features."""
        meta_features = self.extract_meta_features(symbol, data)
        if symbol in self.best_model_for_symbol:
            return self.best_model_for_symbol[symbol]
        if meta_features.get("price_volatility", 0) > 0.05:
            return "ensemble"
        if meta_features.get("data_length", 0) > 500:
            return "deep_learning"
        return "xgboost"

    def update_model_performance(
        self, symbol: str, model_name: str, performance: float,
    ) -> None:
        """Record model performance statistics and update preferred model."""
        history = self.model_performance_history.setdefault(symbol, {})
        history.setdefault(model_name, []).append(
            {"timestamp": datetime.now(), "performance": performance},
        )
        recent_performances = {
            model: float(np.mean([h["performance"] for h in records[-10:]]))
            for model, records in history.items()
            if records
        }
        if recent_performances:
            self.best_model_for_symbol[symbol] = max(
                recent_performances, key=recent_performances.get,
            )

    def create_symbol_profile(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Create a rich symbol profile combining statistical and behavioural insights."""
        try:
            if data.empty or len(data) < 30:
                profile = {
                    "volatility_regime": "normal",
                    "trend_persistence": 0.5,
                    "volume_pattern": "stable",
                    "price_momentum": 0.0,
                    "seasonal_factor": 1.0,
                    "sector_strength": 0.5,
                    "current_price": float(data["Close"].iloc[-1])
                    if not data.empty
                    else 100.0,
                }
            else:
                close = data["Close"].astype(float)
                volume = data["Volume"].astype(float)
                volatility = close.pct_change().rolling(20).std()
                avg_vol = volatility.mean()
                if avg_vol > 0.03:
                    volatility_regime = "high"
                elif avg_vol < 0.015:
                    volatility_regime = "low"
                else:
                    volatility_regime = "normal"
                returns = close.pct_change()
                trend_periods: List[int] = []
                current_trend = 0
                for ret in returns:
                    if pd.isna(ret):
                        continue
                    if ret > 0.005:
                        current_trend = current_trend + 1 if current_trend > 0 else 1
                    elif ret < -0.005:
                        current_trend = current_trend - 1 if current_trend < 0 else -1
                    else:
                        if abs(current_trend) >= 3:
                            trend_periods.append(abs(current_trend))
                        current_trend = 0
                trend_persistence = (
                    float(np.mean(trend_periods) / 10.0) if trend_periods else 0.5
                )
                trend_persistence = float(np.clip(trend_persistence, 0.0, 1.0))
                volume_sma = volume.rolling(20).mean()
                recent_volume_mean = volume.iloc[-10:].mean()
                volume_sma_last = (
                    volume_sma.iloc[-1]
                    if not pd.isna(volume_sma.iloc[-1])
                    else recent_volume_mean
                )
                volume_ratio = (
                    recent_volume_mean / volume_sma_last if volume_sma_last > 0 else 1.0
                )
                if volume_ratio > 1.2:
                    volume_pattern = "increasing"
                elif volume_ratio < 0.8:
                    volume_pattern = "decreasing"
                else:
                    volume_pattern = "stable"
                price_momentum = (
                    float((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20])
                    if len(close) >= 20
                    else 0.0
                )
                seasonal_factor = float(
                    1.0 + 0.1 * np.sin(2 * np.pi * (datetime.now().month - 1) / 12),
                )
                first_digit = int(symbol[0]) if symbol and symbol[0].isdigit() else 5
                sector_strength = 0.3 + (first_digit % 5) * 0.1
                profile = {
                    "volatility_regime": volatility_regime,
                    "trend_persistence": trend_persistence,
                    "volume_pattern": volume_pattern,
                    "price_momentum": price_momentum,
                    "seasonal_factor": seasonal_factor,
                    "sector_strength": float(sector_strength),
                    "avg_volatility": float(avg_vol),
                    "volume_ratio": float(volume_ratio),
                    "current_price": float(close.iloc[-1]),
                }
            profile.update(
                {
                    "volatility_regime_score": self._analyze_volatility_regime(data),
                    "trend_persistence_score": self._analyze_trend_persistence(data),
                    "volume_correlation": self._analyze_volume_pattern(data),
                    "sector_correlation": self._analyze_sector_correlation(symbol),
                    "liquidity_score": self._calculate_liquidity_score(data),
                    "momentum_sensitivity": self._analyze_momentum_sensitivity(data),
                },
            )
            self.symbol_profiles[symbol] = profile
            return profile
        except Exception as error:
            self.logger.error(f"Symbol profile creation error {symbol}: {error}")
            fallback = {
                "volatility_regime": "normal",
                "trend_persistence": 0.5,
                "volume_pattern": "stable",
                "price_momentum": 0.0,
                "seasonal_factor": 1.0,
                "sector_strength": 0.5,
                "current_price": 100.0,
                "error": str(error),
            }
            self.symbol_profiles[symbol] = fallback
            return fallback

    def adapt_model_parameters(
        self, symbol: str, symbol_profile: Dict[str, Any], base_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Adapt base model parameters based on the symbol profile."""
        try:
            adapted = base_params.copy()
            base_prediction = base_params.get("prediction", 50.0)
            base_confidence = base_params.get("confidence", 0.5)
            volatility_regime = symbol_profile.get("volatility_regime", "normal")
            volatility_boost = 0.0
            if volatility_regime == "high":
                adapted["learning_rate"] = base_params.get("learning_rate", 0.01) * 0.7
                adapted["regularization"] = (
                    base_params.get("regularization", 0.01) * 1.3
                )
                volatility_boost = -3.0
            elif volatility_regime == "low":
                adapted["learning_rate"] = base_params.get("learning_rate", 0.01) * 1.2
                adapted["regularization"] = (
                    base_params.get("regularization", 0.01) * 0.8
                )
                volatility_boost = 2.0
            else:
                volatility_boost = 1.0
            trend_persistence = symbol_profile.get("trend_persistence", 0.5)
            if trend_persistence > 0.7:
                adapted["momentum_factor"] = 1.5
                adapted["trend_weight"] = 1.4
                trend_boost = 4.0
            elif trend_persistence < 0.3:
                adapted["momentum_factor"] = 0.6
                adapted["trend_weight"] = 0.7
                trend_boost = -1.0
            else:
                adapted["momentum_factor"] = 1.0
                adapted["trend_weight"] = 1.0
                trend_boost = 1.5
            volume_pattern = symbol_profile.get("volume_pattern", "stable")
            if volume_pattern == "increasing":
                adapted["volume_weight"] = 1.5
                volume_boost = 2.5
            elif volume_pattern == "decreasing":
                adapted["volume_weight"] = 0.6
                volume_boost = -1.5
            else:
                adapted["volume_weight"] = 1.0
                volume_boost = 0.5
            sector_strength = symbol_profile.get("sector_strength", 0.5)
            sector_boost = (sector_strength - 0.5) * 6
            adapted["sector_adjustment"] = 0.7 + sector_strength * 0.6
            seasonal_factor = symbol_profile.get("seasonal_factor", 1.0)
            seasonal_boost = (seasonal_factor - 1.0) * 3
            total_boost = (
                volatility_boost
                + trend_boost
                + volume_boost
                + sector_boost
                + seasonal_boost
            )
            adapted_prediction = base_prediction + total_boost
            adaptation_strength = abs(total_boost) / 10.0
            adapted_confidence = min(base_confidence + adaptation_strength * 0.1, 0.9)
            adapted["adapted_prediction"] = float(np.clip(adapted_prediction, 10, 90))
            adapted["adapted_confidence"] = float(adapted_confidence)
            adapted["meta_boost_applied"] = float(total_boost)
            adapted["adaptation_details"] = {
                "volatility_boost": volatility_boost,
                "trend_boost": trend_boost,
                "volume_boost": volume_boost,
                "sector_boost": sector_boost,
                "seasonal_boost": seasonal_boost,
            }
            self.adaptation_memory[symbol] = adapted
            return adapted
        except Exception as error:
            self.logger.error(f"Model parameter adaptation error for {symbol}: {error}")
            fallback = base_params.copy()
            fallback["adapted_prediction"] = base_params.get("prediction", 50.0) + 1.0
            fallback["adapted_confidence"] = min(
                base_params.get("confidence", 0.5) + 0.05, 0.9,
            )
            return fallback

    def _analyze_volatility_regime(self, data: pd.DataFrame) -> float:
        if data.empty or "Close" not in data:
            return 0.5
        returns = data["Close"].pct_change().dropna()
        volatility = returns.rolling(20).std()
        high_vol_periods = (volatility > volatility.quantile(0.8)).astype(int)
        persistence = high_vol_periods.rolling(10).sum().mean() / 10
        return float(np.nan_to_num(persistence, nan=0.5))

    def _analyze_trend_persistence(self, data: pd.DataFrame) -> float:
        if data.empty or "Close" not in data:
            return 0.5
        prices = data["Close"].astype(float)
        sma_5 = prices.rolling(5).mean()
        sma_20 = prices.rolling(20).mean()
        sma_60 = prices.rolling(60).mean()
        trend_5_20 = (sma_5 > sma_20).astype(int)
        trend_20_60 = (sma_20 > sma_60).astype(int)
        consistency = (trend_5_20 == trend_20_60).mean()
        return float(np.nan_to_num(consistency, nan=0.5))

    def _analyze_volume_pattern(self, data: pd.DataFrame) -> float:
        if data.empty or "Volume" not in data or "Close" not in data:
            return 0.5
        volume = data["Volume"].astype(float)
        price_change = data["Close"].pct_change().abs()
        correlation = price_change.corr(volume)
        return float(np.nan_to_num(correlation, nan=0.5))

    def _analyze_sector_correlation(self, symbol: str) -> float:
        tech_stocks = ["6758", "9984", "4519"]
        finance_stocks = ["8306", "8035"]
        if symbol in tech_stocks:
            return 0.8
        if symbol in finance_stocks:
            return 0.7
        return 0.5

    def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        if data.empty or "Volume" not in data:
            return 0.5
        volume = data["Volume"].astype(float)
        avg_volume = volume.mean()
        if avg_volume <= 0:
            return 0.5
        volume_stability = 1.0 - (volume.std() / avg_volume)
        return float(np.clip(volume_stability, 0.0, 1.0))

    def _analyze_momentum_sensitivity(self, data: pd.DataFrame) -> float:
        if data.empty or "Close" not in data:
            return 0.5
        returns = data["Close"].pct_change()
        momentum_3 = returns.rolling(3).sum()
        momentum_10 = returns.rolling(10).sum()
        momentum_persistence = (momentum_3 * momentum_10 > 0).mean()
        return float(np.nan_to_num(momentum_persistence, nan=0.5))