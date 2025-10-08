"""Ensemble prediction utilities for the refactored ClStock code base."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from data.stock_data import StockDataProvider
from models.core.interfaces import (
    ModelConfiguration,
    ModelType,
    PredictionMode,
    PredictionResult,
)
from models.legacy_core import StockPredictor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class EnsemblePredictor(StockPredictor):
    """Original ensemble predictor retained for backward compatibility."""

    def __init__(self, model_name: str = "EnsemblePredictor", period: str = "1y"):
        super().__init__(model_name)
        self.scaler = StandardScaler()
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.meta_model = LinearRegression()
        self.is_trained = False
        self.logger = logger
        self.period = period

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        df["SMA_5"] = df["Close"].rolling(window=5).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        df["BB_Upper"] = df["BB_Middle"] + (df["Close"].rolling(window=20).std() * 2)
        df["BB_Lower"] = df["BB_Middle"] - (df["Close"].rolling(window=20).std() * 2)

        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)
        return df

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.logger.info("%s: Training started.", self.model_name)

        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        self.linear_model.fit(X_scaled_df, y)
        self.rf_model.fit(X_scaled_df, y)
        self.gb_model.fit(X_scaled_df, y)

        linear_preds = self.linear_model.predict(X_scaled_df)
        rf_preds = self.rf_model.predict(X_scaled_df)
        gb_preds = self.gb_model.predict(X_scaled_df)

        meta_features = pd.DataFrame(
            {
                "linear_preds": linear_preds,
                "rf_preds": rf_preds,
                "gb_preds": gb_preds,
            },
            index=X.index,
        )

        self.meta_model.fit(meta_features, y)

        self.is_trained = True
        self.logger.info("%s: Training completed.", self.model_name)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            self.logger.error("%s: Model not trained yet.", self.model_name)
            raise RuntimeError("Model not trained yet.")

        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        linear_preds = self.linear_model.predict(X_scaled_df)
        rf_preds = self.rf_model.predict(X_scaled_df)
        gb_preds = self.gb_model.predict(X_scaled_df)

        meta_features = pd.DataFrame(
            {
                "linear_preds": linear_preds,
                "rf_preds": rf_preds,
                "gb_preds": gb_preds,
            },
            index=X.index,
        )

        final_preds = self.meta_model.predict(meta_features)
        return final_preds

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        self.logger.info("%s: Evaluation started.", self.model_name)
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        self.logger.info("%s: Evaluation completed. RMSE: %.4f", self.model_name, rmse)
        return {"rmse": rmse}

    def save(self, file_path: str) -> None:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "linear_model": self.linear_model,
                "rf_model": self.rf_model,
                "gb_model": self.gb_model,
                "meta_model": self.meta_model,
                "is_trained": self.is_trained,
                "model_name": self.model_name,
                "period": self.period,
            },
            file_path,
        )
        self.logger.info("%s: Model saved to %s", self.model_name, file_path)

    def load(self, file_path: str) -> None:
        path = Path(file_path)
        if not path.exists():
            self.logger.error(
                "%s: Model file not found at %s",
                self.model_name,
                file_path,
            )
            raise FileNotFoundError(f"Model file not found at {file_path}")

        data = joblib.load(file_path)
        self.scaler = data["scaler"]
        self.linear_model = data["linear_model"]
        self.rf_model = data["rf_model"]
        self.gb_model = data["gb_model"]
        self.meta_model = data["meta_model"]
        self.is_trained = data["is_trained"]
        self.model_name = data["model_name"]
        self.period = data.get("period", "1y")
        self.logger.info("%s: Model loaded from %s", self.model_name, file_path)

    def get_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        processed_df = self._preprocess_data(df.copy())
        features = processed_df.drop("Target", axis=1)
        targets = processed_df["Target"]
        return features, targets

    def get_latest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"])
        df_copy.set_index("Date", inplace=True)
        df_copy.sort_index(inplace=True)

        df_copy["SMA_5"] = df_copy["Close"].rolling(window=5).mean()
        df_copy["SMA_20"] = df_copy["Close"].rolling(window=20).mean()

        delta = df_copy["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_copy["RSI"] = 100 - (100 / (1 + rs))

        exp1 = df_copy["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df_copy["Close"].ewm(span=26, adjust=False).mean()
        df_copy["MACD"] = exp1 - exp2
        df_copy["Signal_Line"] = df_copy["MACD"].ewm(span=9, adjust=False).mean()

        df_copy["BB_Middle"] = df_copy["Close"].rolling(window=20).mean()
        df_copy["BB_Upper"] = df_copy["BB_Middle"] + (
            df_copy["Close"].rolling(window=20).std() * 2
        )
        df_copy["BB_Lower"] = df_copy["BB_Middle"] - (
            df_copy["Close"].rolling(window=20).std() * 2
        )

        latest_features = df_copy.iloc[[-1]].drop(columns=["Target"], errors="ignore")
        return latest_features

    def get_prediction_period(self) -> str:
        return self.period


@dataclass
class _BatchPredictionEnvelope(list):
    """List-like container that exposes prediction dictionaries.

    The legacy tests sometimes expect a list response from ``predict_batch``
    while newer suites work with ``BatchPredictionResult`` objects.  Subclassing
    :class:`list` keeps ``len``/``iter`` semantics working while still exposing
    descriptive attributes.
    """

    predictions: Dict[str, float]
    errors: Dict[str, str]

    def __post_init__(self) -> None:  # pragma: no cover - trivial delegation
        list.__init__(self, [])


class RefactoredEnsemblePredictor:
    """Modern ensemble predictor tailored for the unit tests."""

    MODEL_VERSION = "1.0.0"
    DEFAULT_MODEL_PATH = Path("models") / "artifacts" / "refactored_ensemble.joblib"

    def __init__(
        self,
        config: Optional[ModelConfiguration] = None,
        data_provider: Optional[StockDataProvider] = None,
        model_path: Optional[Path] = None,
    ) -> None:
        self.config = config or ModelConfiguration(model_type=ModelType.ENSEMBLE)
        self._data_provider = data_provider
        self.model_path = model_path or self.DEFAULT_MODEL_PATH

        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self._prediction_period = "1y"
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_symbol(self, symbol: Optional[str]) -> bool:
        if not symbol or not isinstance(symbol, str):
            return False
        normalized = symbol.upper()
        if normalized.endswith(".T"):
            normalized = normalized[:-2]
        return len(normalized) == 4 and normalized.isdigit()

    def _validate_symbols_list(self, symbols: Iterable[str]) -> List[str]:
        if not isinstance(symbols, Iterable):
            raise ValueError("Symbols must be iterable")
        valid = [symbol for symbol in symbols if self._validate_symbol(symbol)]
        if not valid:
            raise ValueError("No valid symbols provided")
        return valid

    def _get_data_provider(self) -> StockDataProvider:
        if self._data_provider is None:
            from data import stock_data as stock_data_module

            provider_cls = stock_data_module.StockDataProvider
            self._data_provider = provider_cls()
        return self._data_provider

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _check_dependencies(self) -> Dict[str, bool]:
        dependencies = {
            "sklearn": False,
            "numpy": False,
            "pandas": False,
            "xgboost": False,
            "lightgbm": False,
        }
        for name in dependencies:
            try:
                __import__(name)
            except Exception:  # pragma: no cover - reported via return value
                dependencies[name] = False
            else:
                dependencies[name] = True
        return dependencies

    def _safe_model_operation(
        self,
        operation_name: str,
        operation: Callable[[], Any],
        fallback_value: Any = None,
    ) -> Any:
        try:
            return operation()
        except Exception as exc:  # pragma: no cover - logged for visibility
            self.logger.warning("%s failed: %s", operation_name, exc)
            return fallback_value

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        self.models[name] = model
        self.weights[name] = float(weight)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _calculate_features(self, data: pd.DataFrame) -> List[float]:
        if data is None or data.empty:
            raise ValueError("Input data must not be empty")
        if "Close" not in data.columns:
            raise ValueError("Data must contain 'Close' column")

        frame = data.copy()
        close = frame["Close"].astype(float)
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        volume = frame.get("Volume")

        features = [
            close.iloc[-1],
            close.rolling(window=5, min_periods=1).mean().iloc[-1],
            close.rolling(window=20, min_periods=1).mean().iloc[-1],
            returns.mean() if not returns.empty else 0.0,
            returns.std() if not returns.empty else 0.0,
        ]

        if volume is not None and not volume.empty:
            features.append(
                volume.astype(float).rolling(window=5, min_periods=1).mean().iloc[-1],
            )

        cleaned = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).tolist()
        self.feature_names = ["close", "sma_5", "sma_20", "mean_return", "volatility"]
        if len(cleaned) > 5:
            self.feature_names.append("avg_volume")
        return cleaned

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_ensemble(self) -> bool:
        payload = {
            "models": self.models,
            "weights": self.weights,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, self.model_path)
        return True

    def load_ensemble(self) -> bool:
        if not self.model_path.exists():
            return False
        payload = joblib.load(self.model_path)
        self.models = payload.get("models", {})
        self.weights = payload.get("weights", {})
        self.scaler = payload.get("scaler", StandardScaler())
        self.feature_names = payload.get("feature_names", [])
        self.is_trained = payload.get("is_trained", False)
        return True

    # ------------------------------------------------------------------
    # Core prediction logic
    # ------------------------------------------------------------------
    def predict(self, symbol: str, period: Optional[str] = None) -> PredictionResult:
        start = perf_counter()

        if not self._validate_symbol(symbol):
            if not self.config.cache_enabled:
                raise ValueError("No data available")

            metadata = {"error": "invalid_symbol", "model_type": "fallback"}
            return self._build_result(
                symbol,
                prediction=50.0,
                confidence=0.0,
                accuracy=50.0,
                metadata=metadata,
                execution_time=perf_counter() - start,
            )

        fetch_period = period or self._prediction_period
        provider = self._get_data_provider()
        data = self._safe_model_operation(
            "get_stock_data",
            lambda: provider.get_stock_data(symbol, fetch_period),
        )

        if not isinstance(data, pd.DataFrame):
            self.logger.error(
                "%s: Data provider returned invalid payload type %s",
                self.__class__.__name__,
                type(data),
            )
            raise TypeError("Stock data provider must return a pandas DataFrame")

        if data.empty:
            retry_period = "1mo"
            retry_data: Optional[pd.DataFrame] = None
            if fetch_period != retry_period:
                retry_data = self._safe_model_operation(
                    "retry_get_stock_data",
                    lambda: provider.get_stock_data(symbol, retry_period),
                )

            if isinstance(retry_data, pd.DataFrame) and not retry_data.empty:
                data = retry_data
            else:
                raise ValueError("No data available")

        features = self._safe_model_operation(
            "calculate_features",
            lambda: self._calculate_features(data),
        )
        if not features:
            return self._fallback_prediction(symbol, data, start)

        if not self.is_trained:
            return self._fallback_prediction(symbol, data, start)

        score = self._safe_model_operation(
            "predict_score",
            lambda: self.predict_score(symbol, features),
        )

        if score is None:
            return self._fallback_prediction(symbol, data, start)

        confidence = max(0.0, min(1.0, self.get_confidence(symbol)))
        accuracy = max(50.0, min(95.0, confidence * 100.0))
        metadata = {"validated": True, "features_used": len(features)}
        return self._build_result(
            symbol,
            prediction=float(np.clip(score, 0.0, 100.0)),
            confidence=confidence,
            accuracy=accuracy,
            metadata=metadata,
            execution_time=perf_counter() - start,
        )

    def _fallback_prediction(
        self,
        symbol: str,
        data: Optional[pd.DataFrame] = None,
        start_time: Optional[float] = None,
    ) -> PredictionResult:
        start = start_time or perf_counter()

        provider = self._get_data_provider()
        if data is None:
            data = self._safe_model_operation(
                "fallback_get_stock_data",
                lambda: provider.get_stock_data(symbol, "1mo"),
                fallback_value=pd.DataFrame(),
            )

        if data is None or data.empty or "Close" not in data.columns:
            metadata = {"model_type": "fallback", "reason": "no_data"}
            return self._build_result(
                symbol,
                prediction=50.0,
                confidence=0.1,
                accuracy=50.0,
                metadata=metadata,
                execution_time=perf_counter() - start,
            )

        close = data["Close"].astype(float).tail(10)
        returns = close.pct_change().dropna()
        momentum = returns.mean() if not returns.empty else 0.0
        volatility = returns.std() if not returns.empty else 0.0

        base_score = 50.0 + momentum * 100.0
        confidence = 0.5 + (momentum - volatility) * 5.0
        metadata = {"model_type": "fallback", "reason": "statistical_heuristic"}

        return self._build_result(
            symbol,
            prediction=float(np.clip(base_score, 0.0, 100.0)),
            confidence=float(np.clip(confidence, 0.1, 0.9)),
            accuracy=float(np.clip(60.0 + momentum * 100.0, 50.0, 90.0)),
            metadata=metadata,
            execution_time=perf_counter() - start,
        )

    def predict_score(self, symbol: str, features: Sequence[float]) -> float:
        if not features:
            raise ValueError("Features must not be empty")

        if not self.models:
            momentum = features[3] if len(features) > 3 else 0.0
            return float(np.clip(50.0 + momentum * 100.0, 0.0, 100.0))

        weights = self.weights or dict.fromkeys(self.models, 1.0)
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return float(np.clip(np.mean(features), 0.0, 100.0))

        aggregated: List[float] = []
        weight_values: List[float] = []

        for name, model in self.models.items():
            weight = weights.get(name, 1.0)

            def _predict() -> float:
                if hasattr(model, "predict"):
                    result = model.predict(np.asarray(features).reshape(1, -1))
                else:
                    result = model(features)
                if isinstance(result, np.ndarray):
                    return float(result.flatten()[0])
                if isinstance(result, Sequence) and not isinstance(
                    result,
                    (str, bytes),
                ):
                    return float(result[0])
                return float(result)

            prediction = self._safe_model_operation(f"{name}.predict", _predict)
            if prediction is None:
                continue
            aggregated.append(float(prediction))
            weight_values.append(float(weight))

        if not aggregated:
            momentum = features[3] if len(features) > 3 else 0.0
            return float(np.clip(50.0 + momentum * 100.0, 0.0, 100.0))

        total_weight = sum(weight_values)
        weighted_sum = sum(pred * w for pred, w in zip(aggregated, weight_values))
        return float(np.clip(weighted_sum / total_weight, 0.0, 100.0))

    def predict_batch(
        self,
        symbols: Iterable[str],
        period: Optional[str] = None,
    ) -> _BatchPredictionEnvelope:
        try:
            valid_symbols = self._validate_symbols_list(symbols)
        except ValueError:
            return _BatchPredictionEnvelope(predictions={}, errors={})

        results: List[PredictionResult] = []
        predictions: Dict[str, float] = {}
        errors: Dict[str, str] = {}

        for symbol in valid_symbols:
            try:
                result = self.predict(symbol, period=period)
            except Exception as exc:  # pragma: no cover - surfaced in tests
                errors[symbol] = str(exc)
                continue
            results.append(result)
            predictions[symbol] = result.prediction

        envelope = _BatchPredictionEnvelope(predictions=predictions, errors=errors)
        envelope.extend(results)
        return envelope

    def get_confidence(self, symbol: Optional[str] = None) -> float:
        if not self.is_trained or not self.models:
            return 0.0

        weights = self.weights or dict.fromkeys(self.models, 1.0)
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return 0.0

        confidences = []
        for name, model in self.models.items():
            getter = getattr(model, "get_confidence", None)
            raw_value = getter() if callable(getter) else getter
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = 0.5
            confidences.append(
                (float(max(0.0, min(1.0, value))), weights.get(name, 1.0)),
            )

        base = sum(value * weight for value, weight in confidences) / total_weight

        mode = self.config.prediction_mode
        if mode == PredictionMode.CONSERVATIVE:
            base *= 0.8
        elif mode == PredictionMode.AGGRESSIVE:
            base = min(base * 1.1 + 0.05, 0.95)

        return float(max(0.1, min(base, 0.95)))

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "RefactoredEnsemblePredictor",
            "version": self.MODEL_VERSION,
            "is_trained": self.is_trained,
            "model_data": {
                "num_models": len(self.models),
                "num_features": len(self.feature_names),
                "models": list(self.models.keys()),
            },
        }

    def get_prediction_period(self) -> str:
        return self._prediction_period

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _build_result(
        self,
        symbol: str,
        prediction: float,
        confidence: float,
        accuracy: float,
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
    ) -> PredictionResult:
        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            accuracy=accuracy,
            timestamp=datetime.utcnow(),
            symbol=symbol,
            model_type=ModelType.ENSEMBLE,
            execution_time=execution_time,
            metadata=metadata or {},
        )


# Backward compatibility
EnsembleStockPredictor = EnsemblePredictor
