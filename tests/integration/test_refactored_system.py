"""Tests covering the refactored predictor factory infrastructure."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np
import pytest


def _install_sklearn_stub():
    """Provide a lightweight stub for ``sklearn`` dependencies used in tests."""

    import sys
    import types

    if "sklearn" in sys.modules:
        return

    sklearn = types.ModuleType("sklearn")
    sklearn.__path__ = []

    class _DummyEstimator:
        def __init__(self, *args, **kwargs):
            self._fitted = False

        def fit(self, X, y):  # pragma: no cover - behaviour not asserted directly
            self._fitted = True
            return self

        def predict(self, X):
            length = len(X) if hasattr(X, "__len__") else 1
            return np.zeros(length)

    def _identity_transform(X):
        return np.asarray(X)

    class _StandardScaler:
        def fit(self, X):
            return self

        def transform(self, X):
            return _identity_transform(X)

        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)

    def _mean_squared_error(y_true, y_pred):
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_true_arr - y_pred_arr) ** 2))

    ensemble = types.ModuleType("sklearn.ensemble")
    ensemble.GradientBoostingRegressor = _DummyEstimator
    ensemble.RandomForestRegressor = _DummyEstimator

    linear_model = types.ModuleType("sklearn.linear_model")
    linear_model.LinearRegression = _DummyEstimator

    metrics = types.ModuleType("sklearn.metrics")
    metrics.mean_squared_error = _mean_squared_error

    preprocessing = types.ModuleType("sklearn.preprocessing")
    preprocessing.StandardScaler = _StandardScaler

    sklearn.ensemble = ensemble
    sklearn.linear_model = linear_model
    sklearn.metrics = metrics
    sklearn.preprocessing = preprocessing

    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.ensemble"] = ensemble
    sys.modules["sklearn.linear_model"] = linear_model
    sys.modules["sklearn.metrics"] = metrics
    sys.modules["sklearn.preprocessing"] = preprocessing


_install_sklearn_stub()

factory_mod = pytest.importorskip(
    "models_refactored.core.factory",
    reason="Refactored models require optional heavy dependencies",
)
interfaces_mod = pytest.importorskip(
    "models_refactored.core.interfaces",
    reason="Refactored models interfaces unavailable",
)

from data.stock_data import StockDataProvider

PredictorFactory = factory_mod.PredictorFactory
ModelConfiguration = interfaces_mod.ModelConfiguration
ModelType = interfaces_mod.ModelType
PredictionMode = interfaces_mod.PredictionMode
PredictionResult = interfaces_mod.PredictionResult
StockPredictor = interfaces_mod.StockPredictor


class DummyPredictor(StockPredictor):
    """Simple ``StockPredictor`` implementation for factory testing."""

    def __init__(self, config: ModelConfiguration, data_provider=None, **_kwargs):
        super().__init__(config)
        self.data_provider = data_provider

    def predict(self, symbol: str) -> PredictionResult:
        return PredictionResult(
            prediction=60.0,
            confidence=0.85,
            accuracy=88.0,
            timestamp=datetime(2024, 1, 1),
            symbol=symbol,
            model_type=self.config.model_type,
        )

    def predict_batch(self, symbols: List[str]) -> List[PredictionResult]:
        return [self.predict(symbol) for symbol in symbols]

    def train(self, data) -> bool:  # pragma: no cover - unused in tests
        self.is_trained = True
        return True

    def get_confidence(self, symbol: str) -> float:
        return 0.85

    def get_model_info(self) -> Dict[str, str]:
        return {
            "mode": self.config.prediction_mode.value,
            "model_type": self.config.model_type.value,
        }


@pytest.fixture(autouse=True)
def reset_predictor_factory():
    """Ensure the global factory registry is reset between tests."""

    original_registry = PredictorFactory._registered_predictors.copy()
    original_instances = PredictorFactory._instances.copy()
    yield
    PredictorFactory._registered_predictors = original_registry
    PredictorFactory._instances = original_instances


def test_register_and_create_custom_predictor():
    factory = PredictorFactory()
    factory.register_predictor(ModelType.PARALLEL, DummyPredictor)

    config = ModelConfiguration(
        model_type=ModelType.PARALLEL, prediction_mode=PredictionMode.AGGRESSIVE
    )
    provider = StockDataProvider()

    predictor = factory.create_predictor(
        ModelType.PARALLEL, config=config, data_provider=provider
    )

    result = predictor.predict("7203")
    assert result.symbol == "7203"
    assert result.model_type is ModelType.PARALLEL
    assert predictor.get_model_info()["mode"] == PredictionMode.AGGRESSIVE.value


def test_get_or_create_returns_cached_instance():
    factory = PredictorFactory()
    factory.register_predictor(ModelType.PARALLEL, DummyPredictor)

    first = factory.get_or_create_predictor(ModelType.PARALLEL, instance_name="custom")
    second = factory.get_or_create_predictor(ModelType.PARALLEL, instance_name="custom")
    assert first is second


def test_list_available_types_includes_custom_registration():
    factory = PredictorFactory()
    factory.register_predictor(ModelType.PARALLEL, DummyPredictor)

    available_types = factory.list_available_types()
    assert ModelType.PARALLEL in available_types
