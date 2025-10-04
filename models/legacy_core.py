from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Callable, ClassVar
import logging

import hashlib
logger = logging.getLogger(__name__) # 修正

import json

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    _xgb_available = True
except ImportError:
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None

        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            # Avoid division by zero
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
            return self

        def transform(self, X):
            if self.mean_ is None or self.scale_ is None:
                raise ValueError("Scaler has not been fitted yet.")
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)
    _xgb_available = False

from data.stock_data import StockDataProvider
from models.recommendation import StockRecommendation


@dataclass
class PredictionResult:
    """Simple prediction result container shared across tests."""

    prediction: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    accuracy: float = 0.0
    symbol: str = ""


class PredictorInterface(ABC):
    """Minimal predictor protocol used by the unit tests."""

    @abstractmethod
    def predict(
        self, symbol: str, data: Optional[pd.DataFrame] = None
    ) -> PredictionResult: ...

    @abstractmethod
    def get_confidence(self) -> float: ...

    @abstractmethod
    def is_trained(self) -> bool: ...


class StockPredictor(PredictorInterface):
    """Base predictor that mimics the legacy behaviour expected by the tests."""

    @staticmethod
    def _default_data_provider() -> StockDataProvider:
        from data.stock_data import StockDataProvider as Provider

        return Provider()

    data_provider_factory: ClassVar[Callable[[], StockDataProvider]] = _default_data_provider

    def __init__(
        self,
        model_type: str = "base",
        data_provider: Optional[StockDataProvider] = None,
    ) -> None:
        self.model_type = model_type
        self._is_trained = False
        self.data_provider = data_provider or self.data_provider_factory()

    def predict(
        self, symbol: str, data: Optional[pd.DataFrame] = None
    ) -> PredictionResult:
        raise NotImplementedError

    def get_confidence(self) -> float:
        return 0.5

    def is_trained(self) -> bool:
        return self._is_trained

    # Convenience wrappers -------------------------------------------------
    def predict_score(self, symbol: str, data: Optional[pd.DataFrame] = None) -> float:
        return float(self.predict(symbol, data).prediction)

    def predict_return_rate(self, symbol: str, days: int = 5) -> float:
        result = self.predict(symbol)
        value = float(result.prediction)
        if value > 1:
            value /= 100.0
        limit = 0.006 * max(days, 1)
        return max(min(value, limit), -limit)

    def get_model_info(self) -> Dict[str, Any]:  # pragma: no cover - helper only
        return {"model_type": self.model_type, "trained": self.is_trained()}


class EnsembleStockPredictor(StockPredictor):
    """Simple ensemble container with weighted averaging."""

    def __init__(self, data_provider: Optional[StockDataProvider] = None) -> None:
        super().__init__(model_type="ensemble", data_provider=data_provider)
        self.models: List[PredictorInterface] = []
        self.weights: List[float] = []

    def add_model(self, model: PredictorInterface, weight: float = 1.0) -> None:
        self.models.append(model)
        self.weights.append(weight)

    def train(self, data: pd.DataFrame, target: Iterable[float]) -> None:
        if not self.models:
            self._is_trained = True
            return
        for model in self.models:
            train = getattr(model, "train", None)
            if callable(train):
                train(data, target)
        self._is_trained = all(
            getattr(model, "is_trained", lambda: True)() for model in self.models
        )

    def predict(
        self, symbol: str, data: Optional[pd.DataFrame] = None
    ) -> PredictionResult:
        if not self.is_trained():
            raise ValueError("Ensemble must be trained before prediction")

        if data is None and hasattr(self.data_provider, "get_stock_data"):
            data = self.data_provider.get_stock_data(symbol, "1mo")

        valid_predictions: List[PredictionResult] = []
        valid_weights: List[float] = []
        default_weight = 1.0 if not self.weights else None

        for index, model in enumerate(self.models):
            weight = self.weights[index] if default_weight is None else default_weight
            try:
                prediction = model.predict(symbol, data)
            except Exception as e:
                logger.warning(f"Prediction failed for symbol {symbol} with model {model.__class__.__name__}: {e}")
                continue
            valid_predictions.append(prediction)
            valid_weights.append(weight)

        if not valid_predictions:
            raise ValueError("All models failed to produce predictions")

        total_weight = sum(valid_weights) or len(valid_predictions)
        prediction_value = (
            sum(pr.prediction * w for pr, w in zip(valid_predictions, valid_weights))
            / total_weight
        )
        confidence = (
            sum(pr.confidence * w for pr, w in zip(valid_predictions, valid_weights))
            / total_weight
        )

        metadata = {"model_type": self.model_type, "symbol": symbol}
        return PredictionResult(prediction_value, confidence, datetime.now(), metadata)

    def get_confidence(self) -> float:
        if not self.models:
            return 0.0
        weights = self.weights or [1.0] * len(self.models)
        total_weight = sum(weights)
        if not total_weight:
            return 0.0
        return (
            sum(
                getattr(model, "get_confidence", lambda: 0.5)() * w
                for model, w in zip(self.models, weights)
            )
            / total_weight
        )


class CacheablePredictor(StockPredictor):
    """Prediction cache helper used by several tests."""

    def __init__(
        self,
        cache_size: int = 1000,
        data_provider: Optional[StockDataProvider] = None,
    ) -> None:
        super().__init__(model_type="cacheable", data_provider=data_provider)
        self.cache_size = cache_size
        self._prediction_cache: "OrderedDict[str, PredictionResult]" = OrderedDict()

    def _get_cache_key(self, symbol: str, data_hash: str) -> str:
        return f"{symbol}_{data_hash}_cacheable"

    def _get_data_hash(self, data: Optional[pd.DataFrame]) -> str:
        if data is None or data.empty:
            return "empty"
        try:
            json_repr = data.round(8).to_json(date_format="iso")
        except Exception:
            json_repr = json.dumps(data.to_dict(), default=str)
        return hashlib.sha256(json_repr.encode("utf-8")).hexdigest()

    def cache_prediction(
        self, symbol: str, data: Optional[pd.DataFrame], result: PredictionResult
    ) -> None:
        data_hash = self._get_data_hash(data)
        cache_key = self._get_cache_key(symbol, data_hash)
        self._prediction_cache[cache_key] = result
        self._prediction_cache.move_to_end(cache_key)
        while len(self._prediction_cache) > self.cache_size:
            self._prediction_cache.popitem(last=False)

    def get_cached_prediction(
        self, symbol: str, data: Optional[pd.DataFrame]
    ) -> Optional[PredictionResult]:
        data_hash = self._get_data_hash(data)
        cache_key = self._get_cache_key(symbol, data_hash)
        result = self._prediction_cache.get(cache_key)
        if result is not None:
            self._prediction_cache.move_to_end(cache_key)
        return result


class _SimpleRegressor:
    """Small helper returning the mean of the training target."""

    def __init__(self) -> None:
        self._mean = 50.0

    def fit(self, X: np.ndarray, y: Iterable[float]) -> None:
        arr = np.asarray(list(y), dtype=float)
        if arr.size:
            self._mean = float(arr.mean())

    def predict(self, X: np.ndarray) -> np.ndarray:
        length = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(length, self._mean, dtype=float)


class MLStockPredictor(CacheablePredictor):
    """Refactored ML predictor for the compatibility layer."""

    def __init__(
        self,
        model_type: str = "xgboost",
        data_provider: Optional[StockDataProvider] = None,
        cache_size: int = 512,
    ) -> None:
        super().__init__(cache_size=cache_size, data_provider=data_provider)
        self.model_type = model_type
        self.scaler: StandardScaler = StandardScaler()
        # xgboostが利用可能かを判定して初期化
        if _xgb_available and model_type == "xgboost":
            self.model: Optional[Any] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            # フォールバック用のモデル
            if _xgb_available:
                # xgboostは利用可能だが、model_typeが"xgboost"でない場合
                # ここでは _SimpleRegressor を使用する例
                # 必要に応じて他のモデルタイプも考慮
                self.model: Optional[_SimpleRegressor] = _SimpleRegressor()
            else:
                # xgboostが利用不可能な場合
                self.model = _SimpleRegressor()
        self.feature_names: List[str] = []
        self.model_directory = Path("models_cache")
        self.model_directory.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------
    def _ensure_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=[np.number]).copy()
        numeric.fillna(method="ffill", inplace=True)
        numeric.fillna(method="bfill", inplace=True)
        numeric.fillna(0.0, inplace=True)
        return numeric

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()

        df = data.copy()
        columns = {col.lower(): col for col in df.columns}
        close = columns.get("close")
        volume = columns.get("volume")
        high = columns.get("high")
        low = columns.get("low")
        open_col = columns.get("open")
        if close is None:
            raise ValueError("Input data must contain a Close column")

        features = pd.DataFrame(index=df.index)
        features["price_change"] = df[close].pct_change().fillna(0.0) * 100.0
        if volume:
            features["volume_change"] = df[volume].pct_change().fillna(0.0)
        else:
            features["volume_change"] = 0.0

        if high and low:
            with np.errstate(divide="ignore", invalid="ignore"):
                features["high_low_ratio"] = (
                    (df[high] - df[low]) / df[close].replace(0, np.nan)
                ).fillna(0.0)
        else:
            features["high_low_ratio"] = 0.0

        if open_col:
            with np.errstate(divide="ignore", invalid="ignore"):
                features["close_open_ratio"] = (
                    (df[close] - df[open_col]) / df[open_col].replace(0, np.nan)
                ).fillna(0.0)
        else:
            features["close_open_ratio"] = 0.0

        price_diff = df[close].diff().fillna(0.0)
        gain = price_diff.where(price_diff > 0, 0.0)
        loss = (-price_diff).where(price_diff < 0, 0.0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        features["rsi"] = 100 - (100 / (1 + rs.replace(np.nan, 0)))

        features["sma_20"] = df[close].rolling(window=20, min_periods=1).mean()
        features["sma_50"] = df[close].rolling(window=50, min_periods=1).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            features["sma_20_ratio"] = (
                df[close] / features["sma_20"].replace(0, np.nan)
            ).fillna(0.0)
            features["sma_50_ratio"] = (
                df[close] / features["sma_50"].replace(0, np.nan)
            ).fillna(0.0)
        features["macd"] = (
            df[close].ewm(span=12, adjust=False).mean()
            - df[close].ewm(span=26, adjust=False).mean()
        )
        features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
        features["atr"] = (df[close] - df[close].shift(1)).abs().fillna(0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            features["atr_ratio"] = (
                features["atr"] / df[close].replace(0, np.nan)
            ).fillna(0.0)

        features["price_lag_1"] = df[close].shift(1).fillna(0.0)
        features["price_lag_5"] = df[close].shift(5).fillna(0.0)
        if volume:
            features["volume_lag_1"] = df[volume].shift(1).fillna(0.0)
        else:
            features["volume_lag_1"] = 0.0

        return self._ensure_numeric(features)

    def create_targets(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if data is None or data.empty:
            raise ValueError("Data is required for creating targets")
        df = data.copy()
        close = next((col for col in df.columns if col.lower() == "close"), None)
        if close is None:
            raise ValueError("Input data must contain a Close column")

        returns_1d = df[close].pct_change().shift(-1) * 100.0
        returns_5d = df[close].pct_change(periods=5).shift(-5) * 100.0

        targets_reg = pd.DataFrame(
            {
                "return_1d": returns_1d,
                "return_5d": returns_5d,
                "recommendation_score": returns_5d.clip(-20, 20) + 50,
            }
        ).dropna()

        targets_cls = pd.DataFrame(
            {
                "direction_1d": (returns_1d > 0).astype(int),
                "direction_5d": (returns_5d > 0).astype(int),
            }
        ).loc[targets_reg.index]

        return targets_reg, targets_cls

    def _calculate_future_performance_score(self, data: pd.DataFrame) -> pd.Series:
        if data is None or data.empty:
            return pd.Series(dtype=float)
        df = data.copy()
        close = next((col for col in df.columns if col.lower() == "close"), None)
        if close is None:
            return pd.Series(dtype=float)
        future_returns = df[close].pct_change(periods=5).shift(-5) * 100.0
        future_returns = future_returns.clip(-20, 20).fillna(0.0)
        return future_returns + 50

    def prepare_dataset(self, symbols: List[str]) -> pd.DataFrame:
        if not symbols:
            raise ValueError("No valid data available")
        feature_frames = []
        for symbol in symbols:
            raw = self.data_provider.get_stock_data(symbol, "1y")
            if raw is None or raw.empty:
                continue
            features = self.prepare_features(raw)
            if not features.empty:
                features["symbol"] = symbol
                feature_frames.append(features)
        if not feature_frames:
            raise ValueError("No valid data available")
        combined = pd.concat(feature_frames).dropna()
        self.feature_names = [col for col in combined.columns if col != "symbol"]
        return combined

    # ------------------------------------------------------------------
    # Training and persistence
    # ------------------------------------------------------------------
    def train(self, features: pd.DataFrame, targets: Iterable[float]) -> None:
        if features is None or features.empty:
            raise ValueError("Training features cannot be empty")
        numeric = self._ensure_numeric(features)
        self.feature_names = list(numeric.columns)
        self.scaler.fit(numeric.values)
        X_scaled = self.scaler.transform(numeric.values)
        # self.model = _SimpleRegressor()  # XGBoostなどの場合、インスタンスは__init__で作成済み
        self.model.fit(X_scaled, list(targets))
        self._is_trained = True

    def _model_files(self) -> Dict[str, Path]:
        base = self.model_directory / self.model_type
        return {
            "model": base.with_name(f"{base.name}_model.joblib"),
            "scaler": base.with_name(f"{base.name}_scaler.joblib"),
            "features": base.with_name(f"{base.name}_features.joblib"),
        }

    def save_model(self) -> None:
        if not self.model:
            raise ValueError("No trained model available to save")
        files = self._model_files()
        joblib.dump(self.model, files["model"])
        joblib.dump(self.scaler, files["scaler"])
        joblib.dump(self.feature_names, files["features"])

    def load_model(self) -> bool:
        files = self._model_files()
        if not all(path.exists() for path in files.values()):
            return False
        self.model = joblib.load(files["model"])
        self.scaler = joblib.load(files["scaler"])
        self.feature_names = joblib.load(files["features"])
        self._is_trained = True
        return True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self, symbol: str, data: Optional[pd.DataFrame] = None
    ) -> PredictionResult:
        if not self.is_trained():
            raise ValueError("Model must be trained before prediction")
        if data is None:
            data = self.data_provider.get_stock_data(symbol, "3mo")
        if data is None or data.empty:
            raise ValueError("No data available for prediction")

        cached = self.get_cached_prediction(symbol, data)
        if cached is not None:
            return cached

        features = self.prepare_features(data)
        if features.empty:
            raise ValueError("Unable to prepare features for prediction")

        if not self.feature_names:
            self.feature_names = list(features.columns)

        aligned = features[self.feature_names].iloc[-1:]
        if not hasattr(self.scaler, "scale_"):
            self.scaler.fit(aligned.values)
        scaled = self.scaler.transform(aligned.values)
        prediction_value = 50.0
        if self.model is not None:
            prediction_value = float(
                np.asarray(self.model.predict(scaled)).flatten()[0]
            )
        confidence = min(max(abs(prediction_value - 50.0) / 50.0, 0.0), 1.0)
        metadata = {
            "model_type": self.model_type,
            "symbol": symbol,
            "features_used": len(self.feature_names),
        }
        result = PredictionResult(
            prediction_value, confidence, datetime.now(), metadata
        )
        self.cache_prediction(symbol, data, result)
        return result

    def get_feature_importance(self) -> Dict[str, float]:
        if not self.is_trained() or not self.model:
            return {}
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None or not self.feature_names:
            return {}
        return {
            name: float(value) for name, value in zip(self.feature_names, importances)
        }

    def get_confidence(self) -> float:
        return 0.8 if self.is_trained() else 0.3

    # ------------------------------------------------------------------
    # Recommendation helpers expected by API/tests
    # ------------------------------------------------------------------
    def calculate_score(self, symbol: str) -> float:
        """Compute a simple heuristic score in the 0-100 range."""

        try:
            raw_data = self.data_provider.get_stock_data(symbol, "6mo")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to fetch data for %s: %s", symbol, exc)
            return 0.0

        if raw_data is None or raw_data.empty:
            return 0.0

        try:
            data = self.data_provider.calculate_technical_indicators(raw_data)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Failed to calculate technical indicators for %s: %s", symbol, exc
            )
            data = raw_data

        if data is None or data.empty:
            return 0.0

        close_col = next((c for c in data.columns if c.lower() == "close"), None)
        sma20_col = next((c for c in data.columns if c.lower() == "sma_20"), None)
        sma50_col = next((c for c in data.columns if c.lower() == "sma_50"), None)
        rsi_col = next((c for c in data.columns if c.lower() == "rsi"), None)

        score = 50.0
        if close_col and sma20_col:
            current_price = data[close_col].iloc[-1]
            sma20 = data[sma20_col].iloc[-1]
            if pd.notna(current_price) and pd.notna(sma20) and sma20:
                score += float((current_price - sma20) / sma20) * 100

        if sma20_col and sma50_col:
            sma20 = data[sma20_col].iloc[-1]
            sma50 = data[sma50_col].iloc[-1]
            if pd.notna(sma20) and pd.notna(sma50) and sma50:
                score += float((sma20 - sma50) / sma50) * 80

        if rsi_col:
            rsi = data[rsi_col].iloc[-1]
            if pd.notna(rsi):
                score += (50 - abs(50 - float(rsi))) / 2

        volume_col = next((c for c in data.columns if c.lower() == "volume"), None)
        if volume_col:
            short = data[volume_col].rolling(window=5, min_periods=1).mean().iloc[-1]
            long = data[volume_col].rolling(window=20, min_periods=1).mean().iloc[-1]
            if pd.notna(short) and pd.notna(long) and long:
                score += float((short - long) / long) * 40

        return float(min(max(score, 0.0), 100.0))

    def _score_to_holding_period(self, score: float) -> str:
        if score >= 80:
            return "1～2か月"
        if score >= 60:
            return "2～3か月"
        return "3～4か月"

    def _score_to_level(self, score: float) -> str:
        if score >= 80:
            return "strong_buy"
        if score >= 60:
            return "buy"
        if score >= 40:
            return "neutral"
        if score >= 20:
            return "watch"
        return "avoid"

    def generate_recommendation(self, symbol: str) -> StockRecommendation:
        data = self.data_provider.get_stock_data(symbol, "6mo")
        if data is None or data.empty:
            raise ValueError(f"No data available for symbol {symbol}")

        technical_data = self.data_provider.calculate_technical_indicators(data)
        if technical_data is None or technical_data.empty:
            raise ValueError(f"Unable to prepare technical indicators for {symbol}")

        close_col = next((c for c in technical_data.columns if c.lower() == "close"), None)
        if close_col is None:
            raise ValueError("Technical data must contain a Close column")

        latest_price = float(technical_data[close_col].iloc[-1])
        score = self.calculate_score(symbol)

        base_multiplier = max(score - 40.0, 0.0) / 200.0
        target_price = latest_price * (1.05 + base_multiplier)
        stop_loss = max(latest_price * (1 - max(0.1, (100 - score) / 200.0)), 0.01)

        company_name = self.data_provider.jp_stock_codes.get(symbol)
        name_column = next(
            (
                col
                for col in technical_data.columns
                if col.lower() in {"companyname", "company_name"}
            ),
            None,
        )
        if name_column is not None:
            candidate = technical_data[name_column].iloc[-1]
            if pd.notna(candidate) and str(candidate):
                company_name = str(candidate)
        if not company_name:
            company_name = symbol

        recommendation = StockRecommendation(
            rank=0,
            symbol=symbol,
            company_name=company_name,
            buy_timing="押し目買いを検討",
            target_price=float(target_price),
            stop_loss=float(stop_loss),
            profit_target_1=float(target_price * 0.98),
            profit_target_2=float(target_price * 1.05),
            holding_period=self._score_to_holding_period(score),
            score=float(score),
            current_price=float(latest_price),
            recommendation_reason=(
                f"テクニカル指標から算出したスコアが {score:.1f} 点のため"
            ),
            recommendation_level=self._score_to_level(score),
        )

        try:
            financial_metrics = self.data_provider.get_financial_metrics(symbol)
            if financial_metrics:
                recommendation.recommendation_reason += "。財務データも安定しています"
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to fetch financial metrics for %s: %s", symbol, exc)

        return recommendation

    def get_top_recommendations(self, top_n: int = 10) -> List[StockRecommendation]:
        symbols = self.data_provider.get_all_stock_symbols()
        recommendations: List[StockRecommendation] = []

        for symbol in symbols:
            try:
                recommendation = self.generate_recommendation(symbol)
            except Exception as exc:
                logger.debug("Skipping symbol %s due to error: %s", symbol, exc)
                continue
            recommendations.append(recommendation)

        recommendations.sort(key=lambda rec: rec.score, reverse=True)

        limited = recommendations[:max(top_n, 0)]
        for idx, recommendation in enumerate(limited, start=1):
            recommendation.rank = idx

        return limited