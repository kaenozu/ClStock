"""Base classes and interfaces for stock prediction models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class PredictionResult:
    """Prediction result with metadata."""

    prediction: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]


class PredictorInterface(ABC):
    """Interface for all stock predictors."""

    @abstractmethod
    def predict(
        self, symbol: str, data: Optional[pd.DataFrame] = None,
    ) -> PredictionResult:
        """Predict stock performance for given symbol."""

    @abstractmethod
    def get_confidence(self) -> float:
        """Get model confidence score."""

    @abstractmethod
    def is_trained(self) -> bool:
        """Check if model is trained."""


class StockPredictor(PredictorInterface):
    """Base class for stock prediction models."""

    def __init__(self, model_type: str = "base"):
        self.model_type = model_type
        self._is_trained = False
        self.model = None
        self.feature_names: List[str] = []

    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained

    def get_confidence(self) -> float:
        """Get base confidence score."""
        return 0.5  # Default neutral confidence

    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        if data is None or data.empty:
            return False

        required_columns = ["open", "high", "low", "close", "volume"]
        return all(col in data.columns for col in required_columns)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement prepare_features")

    def train(self, data: pd.DataFrame, target: pd.Series) -> None:
        """Train the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement train")

    def predict(
        self, symbol: str, data: Optional[pd.DataFrame] = None,
    ) -> PredictionResult:
        """Default prediction implementation."""
        if not self.is_trained():
            raise ValueError("Model must be trained before making predictions")

        if data is None or not self.validate_input(data):
            raise ValueError("Invalid input data")

        # Default implementation returns neutral prediction
        return PredictionResult(
            prediction=0.0,
            confidence=self.get_confidence(),
            timestamp=datetime.now(),
            metadata={"model_type": self.model_type, "symbol": symbol},
        )


class EnsemblePredictor(StockPredictor):
    """Base class for ensemble prediction models."""

    def __init__(self, model_type: str = "ensemble"):
        super().__init__(model_type)
        self.models: List[StockPredictor] = []
        self.weights: List[float] = []

    def add_model(self, model: StockPredictor, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models.append(model)
        self.weights.append(weight)

    def get_confidence(self) -> float:
        """Get ensemble confidence as weighted average of model confidences."""
        if not self.models:
            return 0.0

        total_weight = sum(self.weights)
        if total_weight == 0:
            return 0.0

        weighted_confidence = sum(
            model.get_confidence() * weight
            for model, weight in zip(self.models, self.weights)
        )
        return weighted_confidence / total_weight

    def is_trained(self) -> bool:
        """Check if all models in ensemble are trained."""
        return all(model.is_trained() for model in self.models)


class CacheablePredictor(StockPredictor):
    """Base class for predictors with caching capabilities."""

    def __init__(self, model_type: str = "cacheable", cache_size: int = 1000):
        super().__init__(model_type)
        self.cache_size = cache_size
        self._prediction_cache: Dict[str, PredictionResult] = {}

    def _get_cache_key(self, symbol: str, data_hash: str) -> str:
        """Generate cache key for prediction."""
        return f"{symbol}_{data_hash}_{self.model_type}"

    def _get_data_hash(self, data: pd.DataFrame) -> str:
        """Generate hash for input data."""
        if data is None or data.empty:
            return "empty"
        return str(hash(tuple(data.iloc[-1].values)))

    def get_cached_prediction(
        self, symbol: str, data: pd.DataFrame,
    ) -> Optional[PredictionResult]:
        """Get cached prediction if available."""
        cache_key = self._get_cache_key(symbol, self._get_data_hash(data))
        return self._prediction_cache.get(cache_key)

    def cache_prediction(
        self, symbol: str, data: pd.DataFrame, result: PredictionResult,
    ) -> None:
        """Cache prediction result."""
        if len(self._prediction_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._prediction_cache))
            del self._prediction_cache[oldest_key]

        cache_key = self._get_cache_key(symbol, self._get_data_hash(data))
        self._prediction_cache[cache_key] = result
