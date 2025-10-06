"""Integration style tests for ``integration_test_enhanced`` helpers.

The legacy ``integration_test_enhanced.py`` script bundles a sizeable set of
utility functions that orchestrate comparisons between the so called
"87% precision" system and the newer ensemble based implementation.  The
original script was written as an executable and relied heavily on printing
results to STDOUT.  These tests exercise the pure helper functions so we can
confidently refactor the script in the future while keeping the public
behaviour stable.

All tests deliberately avoid touching the real predictor implementations.
Instead we use light-weight stubs so the tests remain deterministic and quick.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

import integration_test_enhanced as ite


class DummyPrecisionSystem:
    """Small stub mimicking the API of ``Precision87BreakthroughSystem``."""

    def __init__(self, predictions):
        self._predictions = predictions

    def predict_with_87_precision(self, symbol: str):
        return self._predictions[symbol]


class DummyEnhancedSystem:
    """Stub for ``EnsembleStockPredictor`` returning deterministic objects."""

    def __init__(self, predictions, batch=None):
        self._predictions = predictions
        self._batch = batch or {
            symbol: SimpleNamespace(
                symbol=symbol,
                prediction=value.prediction,
                confidence=value.confidence,
                accuracy=value.accuracy,
            )
            for symbol, value in predictions.items()
        }
        self.feature_cache = SimpleNamespace(size=lambda: 2)
        self.prediction_cache = SimpleNamespace(size=lambda: 1)
        self.parallel_calculator = SimpleNamespace(n_jobs=3)

    def predict(self, symbol: str):
        return self._predictions[symbol]

    def predict_batch(self, symbols):
        return [self._batch[symbol] for symbol in symbols]


@pytest.fixture
def deterministic_time(monkeypatch):
    """Provide a predictable ``time.time`` implementation for the module."""
    timestamps = iter([0.0, 0.2, 0.2, 0.5, 1.0, 1.4, 1.4, 1.9])
    monkeypatch.setattr(ite.time, "time", lambda: next(timestamps))


def _prediction_namespace(
    value: float, confidence: float, accuracy: float, symbol: str,
):
    return SimpleNamespace(
        prediction=value,
        confidence=confidence,
        accuracy=accuracy,
        timestamp=datetime(2024, 1, 1),
        symbol=symbol,
    )


def test_compare_prediction_systems_returns_metrics_and_symbols(deterministic_time):
    test_symbols = ["6758.T", "7203.T"]
    precision_system = DummyPrecisionSystem(
        {
            "6758.T": {
                "final_prediction": 58.0,
                "final_confidence": 0.82,
                "final_accuracy": 87.5,
            },
            "7203.T": {
                "final_prediction": 61.0,
                "final_confidence": 0.78,
                "final_accuracy": 88.1,
            },
        },
    )

    enhanced_system = DummyEnhancedSystem(
        {
            "6758.T": _prediction_namespace(60.0, 0.9, 89.0, "6758.T"),
            "7203.T": _prediction_namespace(59.0, 0.85, 88.5, "7203.T"),
        },
    )

    comparison = ite.compare_prediction_systems(
        precision_system, enhanced_system, test_symbols,
    )

    assert comparison["symbols"] == test_symbols
    assert len(comparison["precision_87_results"]) == 2
    assert len(comparison["enhanced_results"]) == 2

    metrics = comparison["comparison_metrics"]
    assert metrics["avg_prediction_time"]["precision_87"] == pytest.approx(0.3)
    assert metrics["avg_prediction_time"]["enhanced"] == pytest.approx(0.4)
    assert metrics["avg_confidence"]["difference"] == pytest.approx(0.075)
    assert metrics["prediction_consistency"]["max_difference"] == pytest.approx(2.0)


def test_compare_system_performance_reports_batch_metrics_and_psutil_flag(monkeypatch):
    test_symbols = ["6758.T", "7203.T", "8306.T"]

    enhanced_system = DummyEnhancedSystem(
        {
            symbol: _prediction_namespace(55.0, 0.8, 86.0, symbol)
            for symbol in test_symbols
        },
    )

    class _FakeProcess:
        def __init__(self, _pid):
            self._calls = 0

        def memory_info(self):
            self._calls += 1
            rss = 110 * 1024 * 1024 if self._calls > 1 else 100 * 1024 * 1024
            return SimpleNamespace(rss=rss)

    import sys

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=_FakeProcess))

    performance = ite.compare_system_performance(None, enhanced_system, test_symbols)

    assert performance["tested_symbols"] == test_symbols
    assert performance["psutil_available"] is True

    batch_metrics = performance["batch_processing"]["enhanced"]
    assert batch_metrics["success_count"] == len(test_symbols)
    assert batch_metrics["throughput"] > 0

    memory_metrics = performance["memory_efficiency"]["memory_usage"]
    assert memory_metrics["increase_mb"] == pytest.approx(10, abs=0.5)


def test_compare_system_features_includes_availability_flags():
    precision_system = object()
    enhanced_system = object()

    features = ite.compare_system_features(precision_system, enhanced_system)

    assert features["systems_available"] == {
        "precision_87": True,
        "enhanced": True,
    }
    assert "advantage_analysis" in features


def test_display_comprehensive_results_handles_partial_payload(capsys):
    payload = {
        "prediction_comparison": {
            "symbols": ["6758.T"],
            "precision_87_results": [],
            "enhanced_results": [],
            "comparison_metrics": {},
        },
        "performance_comparison": {
            "batch_processing": {},
            "memory_efficiency": {},
            "tested_symbols": [],
            "psutil_available": False,
        },
        "feature_comparison": {
            "systems_available": {"precision_87": False, "enhanced": False},
            "advantage_analysis": {
                "precision_87_advantages": ["High accuracy"],
                "enhanced_advantages": ["Faster processing"],
                "complementary_potential": ["Combine strengths"],
            },
        },
        "test_symbols": ["6758.T"],
    }

    ite.display_comprehensive_results(payload)
    captured = capsys.readouterr()
    assert "拡張機能 vs 87%精度システム 統合比較結果" in captured.out
    assert "Combine strengths" in captured.out
