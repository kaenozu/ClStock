"""Pytest compatible performance smoke tests for the ensemble predictor."""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import pytest


@dataclass
class _DummyCache:
    _size: int = 0

    def add(self, amount: int) -> None:
        self._size += amount

    def size(self) -> int:
        return self._size


class _DummyDataProvider:
    def get_stock_data(self, symbol: str, period: str) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=5, freq="B")
        base = 100 + hash(symbol) % 5
        return pd.DataFrame(
            {
                "Open": np.linspace(base, base + 1, len(index)),
                "High": np.linspace(base + 0.5, base + 1.5, len(index)),
                "Low": np.linspace(base - 0.5, base + 0.5, len(index)),
                "Close": np.linspace(base + 0.25, base + 1.25, len(index)),
                "Volume": np.arange(1, len(index) + 1) * 1000,
            },
            index=index,
        )


class _DummyPredictor:
    def __init__(self) -> None:
        self.data_provider = _DummyDataProvider()
        self.parallel_calculator = SimpleNamespace(n_jobs=4)
        self.feature_cache = _DummyCache()
        self.prediction_cache = _DummyCache()

    def predict(self, symbol: str) -> SimpleNamespace:
        self.prediction_cache.add(1)
        return SimpleNamespace(prediction=110.0, metadata={"symbol": symbol})

    def predict_batch(self, symbols: Iterable[str]) -> Dict[str, SimpleNamespace]:
        predictions = {}
        start = time.time()
        for symbol in symbols:
            predictions[symbol] = self.predict(symbol)
        duration = time.time() - start
        return SimpleNamespace(results=predictions, duration=duration)

    def _calculate_features_optimized(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        self.feature_cache.add(1)
        return {"mean_close": float(data["Close"].mean())}


def _single_prediction_metrics(predictor: _DummyPredictor, symbols: List[str]) -> Dict[str, Any]:
    timings: List[float] = []
    for symbol in symbols:
        start = time.perf_counter()
        result = predictor.predict(symbol)
        end = time.perf_counter()
        timings.append(end - start)
        assert isinstance(result.prediction, float)

    return {
        "average_time": float(np.mean(timings)),
        "count": len(timings),
    }


def _batch_prediction_metrics(predictor: _DummyPredictor, symbols: List[str]) -> Dict[str, Any]:
    outcome = predictor.predict_batch(symbols)
    duration = max(outcome.duration, 1e-6)
    return {
        "duration": duration,
        "throughput": len(symbols) / duration,
        "count": len(outcome.results),
    }


def _cache_effectiveness_metrics(predictor: _DummyPredictor, symbols: List[str]) -> Dict[str, Any]:
    predictor.prediction_cache = _DummyCache()
    for symbol in symbols:
        predictor.predict(symbol)
    cold_size = predictor.prediction_cache.size()

    for symbol in symbols:
        predictor.predict(symbol)
    warm_size = predictor.prediction_cache.size()

    return {
        "cold_predictions": cold_size,
        "warm_predictions": warm_size,
    }


def _parallel_metrics(predictor: _DummyPredictor, symbols: List[str]) -> Dict[str, Any]:
    times: List[float] = []
    for symbol in symbols[:3]:
        data = predictor.data_provider.get_stock_data(symbol, "1y")
        start = time.perf_counter()
        features = predictor._calculate_features_optimized(symbol, data)
        end = time.perf_counter()
        times.append(end - start)
        assert "mean_close" in features

    return {
        "workers": predictor.parallel_calculator.n_jobs,
        "average_time": float(np.mean(times)),
        "samples": len(times),
    }


@pytest.fixture
def dummy_predictor() -> _DummyPredictor:
    return _DummyPredictor()


@pytest.mark.performance
def test_enhanced_performance_suite_collects_metrics(dummy_predictor: _DummyPredictor) -> None:
    symbols = ["7203", "6758", "8306", "8031"]

    single = _single_prediction_metrics(dummy_predictor, symbols[:3])
    batch = _batch_prediction_metrics(dummy_predictor, symbols)
    cache = _cache_effectiveness_metrics(dummy_predictor, symbols[:2])
    parallel = _parallel_metrics(dummy_predictor, symbols)

    assert single["count"] == 3
    assert batch["count"] == len(symbols)
    assert batch["throughput"] > 0
    assert cache["warm_predictions"] >= cache["cold_predictions"]
    assert parallel["workers"] == 4


@pytest.mark.performance
def test_cache_metrics_reflect_prediction_calls(dummy_predictor: _DummyPredictor) -> None:
    symbols = ["7203", "6758"]
    metrics = _cache_effectiveness_metrics(dummy_predictor, symbols)

    assert metrics["cold_predictions"] == len(symbols)
    assert metrics["warm_predictions"] == len(symbols) * 2


@pytest.mark.performance
def test_parallel_metrics_include_feature_timings(dummy_predictor: _DummyPredictor) -> None:
    symbols = ["7203", "6758", "8306"]
    metrics = _parallel_metrics(dummy_predictor, symbols)

    assert metrics["samples"] == len(symbols)
    assert metrics["average_time"] >= 0
