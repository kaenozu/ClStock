"""Tests for the auto retraining scheduler stack."""

import sys
from datetime import datetime, timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

import numpy as np
import pandas as pd

sklearn_stub = ModuleType("sklearn")
metrics_stub = ModuleType("sklearn.metrics")
models_stub = ModuleType("models")
stock_specific_stub = ModuleType("models.stock_specific_predictor")
predictor_stub = ModuleType("models.predictor")


def _dummy_accuracy_score(*_, **__):  # pragma: no cover - safety net
    return 0.0


metrics_stub.accuracy_score = _dummy_accuracy_score
sklearn_stub.metrics = metrics_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.metrics", metrics_stub)
sys.modules.setdefault("models", models_stub)
sys.modules.setdefault("models.stock_specific_predictor", stock_specific_stub)
sys.modules.setdefault("models.predictor", predictor_stub)


class _DummyStockSpecificPredictor:  # pragma: no cover - minimal stub
    def train_symbol_model(self, symbol, period):
        return {"symbol": symbol, "period": period, "accuracy": 0.5}


class _DummyStockPredictor:  # pragma: no cover - minimal stub
    pass


stock_specific_stub.StockSpecificPredictor = _DummyStockSpecificPredictor
predictor_stub.StockPredictor = _DummyStockPredictor


from systems import auto_retraining_system as auto_module


@pytest.fixture
def dummy_settings(monkeypatch):
    """Provide deterministic settings for tests."""
    settings = SimpleNamespace(
        target_stocks={"AAA": "Alpha", "BBB": "Beta", "CCC": "Gamma"},
    )
    monkeypatch.setattr(auto_module, "get_settings", lambda: settings)
    return settings


def test_detect_performance_decline_requires_minimum_history(
    monkeypatch, dummy_settings,
):
    monitor = auto_module.ModelPerformanceMonitor()

    now = datetime.now()
    monitor.performance_history["AAA"] = [
        {
            "timestamp": now - timedelta(days=1),
            "prediction": {"signal": 1},
            "actual_result": True,
            "verified": True,
        }
        for _ in range(5)
    ]

    decline_info = monitor.detect_performance_decline("AAA")

    assert decline_info["needs_retraining"] is False
    assert decline_info["reason"] == "insufficient_data"
    assert monitor.get_retraining_candidates() == []


def test_detect_performance_decline_triggers_candidate(monkeypatch, dummy_settings):
    monitor = auto_module.ModelPerformanceMonitor()

    monkeypatch.setattr(
        monitor,
        "_get_historical_best_accuracy",
        lambda symbol: 0.9,
    )

    now = datetime.now()
    records = []
    for i in range(12):
        predicted_positive = i % 2 == 0
        records.append(
            {
                "timestamp": now - timedelta(days=2),
                "prediction": {"signal": 1 if predicted_positive else 0},
                "actual_result": True,
                "verified": True,
            },
        )
    monitor.performance_history["AAA"] = records

    decline_info = monitor.detect_performance_decline("AAA")

    assert decline_info["needs_retraining"] is True
    assert decline_info["reason"] == "performance_decline"

    candidates = monitor.get_retraining_candidates()
    assert len(candidates) == 1
    assert candidates[0]["symbol"] == "AAA"
    assert candidates[0]["reason"] == "performance_decline"


def test_data_drift_detector_identifies_significant_shift(monkeypatch):
    baseline_dates = pd.date_range("2023-01-01", periods=40, freq="D")
    baseline_close = pd.Series(
        100 + 0.5 * np.sin(np.linspace(0, 6, 40)), index=baseline_dates,
    )
    baseline_volume = pd.Series(
        100 + 2 * np.sin(np.linspace(0, 6, 40)), index=baseline_dates,
    )
    baseline_df = pd.DataFrame({"Close": baseline_close, "Volume": baseline_volume})

    current_dates = pd.date_range("2023-03-01", periods=40, freq="D")
    current_close = pd.Series(
        100 + 3 * np.sin(np.linspace(0, 6, 40)), index=current_dates,
    )
    current_volume = pd.Series(
        160 + 2 * np.sin(np.linspace(0, 6, 40)), index=current_dates,
    )
    current_df = pd.DataFrame({"Close": current_close, "Volume": current_volume})

    class DummyProvider:
        def get_stock_data(self, symbol, period):
            if period == "6mo":
                return baseline_df
            return current_df

    monkeypatch.setattr(auto_module, "StockDataProvider", lambda: DummyProvider())

    detector = auto_module.DataDriftDetector()
    baseline_stats = detector.establish_baseline("AAA")

    assert set(baseline_stats) == {
        "volatility",
        "avg_volume",
        "price_trend",
        "volume_trend",
    }

    drift_info = detector.detect_drift("AAA")

    assert drift_info["has_drift"], drift_info
    assert drift_info["symbol"] == "AAA"
    assert drift_info["recommendation"] == "retrain"
    assert drift_info["volatility_drift"] > 0.3


def test_data_drift_detector_reports_errors(monkeypatch):
    class FailingProvider:
        def get_stock_data(self, symbol, period):
            raise ValueError("data unavailable")

    monkeypatch.setattr(auto_module, "StockDataProvider", lambda: FailingProvider())

    detector = auto_module.DataDriftDetector()
    drift_info = detector.detect_drift("AAA")

    assert drift_info["has_drift"] is False
    assert "error" in drift_info
    assert drift_info["error"] == "data unavailable"


def test_merge_candidates_combines_and_limits(monkeypatch, dummy_settings):
    scheduler = auto_module.AutoRetrainingScheduler()
    scheduler.retraining_config["max_concurrent_retraining"] = 2

    performance_candidates = [
        {"symbol": "AAA", "priority": 1.0, "reason": "performance_decline"},
        {"symbol": "BBB", "priority": 0.5, "reason": "performance_decline"},
    ]
    drift_candidates = [
        {
            "symbol": "AAA",
            "priority": 0.8,
            "reason": "data_drift",
            "drift_info": {"has_drift": True},
        },
        {
            "symbol": "CCC",
            "priority": 0.9,
            "reason": "data_drift",
            "drift_info": {"has_drift": True},
        },
    ]

    merged = scheduler._merge_candidates(performance_candidates, drift_candidates)

    assert len(merged) == 2
    assert merged[0]["symbol"] == "AAA"
    assert merged[0]["priority"] == pytest.approx(1.8)
    assert merged[0]["drift_info"]["has_drift"] is True
    assert {candidate["symbol"] for candidate in merged} == {"AAA", "CCC"}


def test_execute_retraining_uses_concurrency_limit_and_backup(
    monkeypatch, dummy_settings,
):
    scheduler = auto_module.AutoRetrainingScheduler()
    scheduler.retraining_config.update(
        {
            "max_concurrent_retraining": 2,
            "backup_models": True,
            "retraining_data_period": "1y",
        },
    )

    backup_mock = MagicMock()
    scheduler._backup_existing_model = backup_mock
    scheduler.stock_predictor = MagicMock()
    scheduler.stock_predictor.train_symbol_model.return_value = {"accuracy": 0.88}

    created_executors = []

    class DummyExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.submitted = []
            created_executors.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, candidate):
            from concurrent.futures import Future

            future = Future()
            try:
                result = fn(candidate)
                future.set_result(result)
            except Exception as exc:  # pragma: no cover - defensive
                future.set_exception(exc)
            self.submitted.append(candidate["symbol"])
            return future

    monkeypatch.setattr(auto_module, "ThreadPoolExecutor", DummyExecutor)

    candidates = [
        {"symbol": "AAA", "reason": "performance_decline", "priority": 1.0},
        {"symbol": "BBB", "reason": "data_drift", "priority": 0.9},
    ]

    scheduler._execute_retraining(candidates)

    assert created_executors[0].max_workers == 2
    assert backup_mock.call_count == 2
    scheduler.stock_predictor.train_symbol_model.assert_called()


def test_manual_retrain_delegates_to_execute(monkeypatch, dummy_settings):
    scheduler = auto_module.AutoRetrainingScheduler()
    execute_mock = MagicMock()
    scheduler._execute_retraining = execute_mock

    result = scheduler.manual_retrain(["AAA", "BBB"], reason="manual_check")

    execute_mock.assert_called_once()
    candidates = execute_mock.call_args.args[0]
    assert all(candidate["priority"] == 1.0 for candidate in candidates)
    assert {candidate["symbol"] for candidate in candidates} == {"AAA", "BBB"}
    assert result["status"] == "completed"
    assert isinstance(result["timestamp"], datetime)


def test_retraining_orchestrator_initialization_establishes_baselines(
    monkeypatch, dummy_settings,
):
    orchestrator = auto_module.RetrainingOrchestrator()
    baseline_mock = MagicMock()
    orchestrator.drift_detector.establish_baseline = baseline_mock

    orchestrator.initialize_system()

    expected_calls = min(5, len(dummy_settings.target_stocks))
    assert baseline_mock.call_count == expected_calls


def test_retraining_orchestrator_comprehensive_status(monkeypatch, dummy_settings):
    orchestrator = auto_module.RetrainingOrchestrator()
    orchestrator.scheduler = MagicMock()
    orchestrator.monitor = MagicMock()

    orchestrator.scheduler.get_system_status.return_value = {
        "scheduler_running": False,
        "config": {},
    }
    orchestrator.monitor.get_retraining_candidates.return_value = [
        {"symbol": "AAA", "priority": 1.2, "reason": "performance_decline"},
    ]

    status = orchestrator.get_comprehensive_status()

    assert "auto_retraining" in status
    assert status["auto_retraining"]["scheduler_running"] is False
    assert status["retraining_candidates"][0]["symbol"] == "AAA"
    assert status["system_health"] == "operational"
    assert isinstance(status["recommendations"], list)
