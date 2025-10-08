"""Tests ensuring refactored ml model modules expose expected interfaces."""

from __future__ import annotations

import pytest

import numpy as np


def test_legacy_shim_matches_new_modules():
    from models import (
        AdvancedPrecisionBreakthrough87System,
        Precision87BreakthroughSystem,
        UltraHighPerformanceSystem,
    )
    from models.advanced_precision import (
        AdvancedPrecisionBreakthrough87System as AdvancedPrecisionModule,
    )
    from models.ml_models import (
        AdvancedPrecisionBreakthrough87System as AdvancedPrecisionShim,
    )
    from models.ml_models import Precision87BreakthroughSystem as PrecisionShim
    from models.ml_models import (
        UltraHighPerformancePredictor as UltraHighPerformanceShim,
    )
    from models.precision_breakthrough import (
        Precision87BreakthroughSystem as PrecisionModule,
    )
    from models.ultra_high_performance import (
        UltraHighPerformancePredictor as UltraHighPerformanceModule,
    )

    assert (
        AdvancedPrecisionModule
        is AdvancedPrecisionShim
        is AdvancedPrecisionBreakthrough87System
    )
    assert PrecisionModule is PrecisionShim is Precision87BreakthroughSystem
    assert UltraHighPerformanceModule is UltraHighPerformanceShim
    assert UltraHighPerformanceSystem is UltraHighPerformanceModule


def test_advanced_precision_prediction_uses_factories(monkeypatch):
    from models.advanced_precision import AdvancedPrecisionBreakthrough87System

    recorded_calls = {
        "create_market_state": False,
        "convert_action_to_score": None,
        "market_state_values": None,
    }

    class DummyDQN:
        def predict_with_dqn(self, market_state):
            recorded_calls["market_state_values"] = tuple(market_state.tolist())
            return {"action": 1, "confidence": 0.6}

    class DummyMultiModal:
        def predict_multimodal(self, price_data, volume_data=None):
            return {"prediction_score": 62.0, "confidence": 0.7}

    class DummyMeta:
        def meta_predict(self, symbol, base_prediction):
            return {"adjusted_prediction": base_prediction + 5, "confidence_boost": 0.1}

    class DummyEnsemble:
        def ensemble_predict(self, predictions, confidences):
            assert "dqn" in predictions
            assert "multimodal" in predictions
            return {"ensemble_prediction": 58.0, "ensemble_confidence": 0.75}

    class DummyTransformer:
        def transformer_predict(self, price_data, volume_data=None):
            return {"prediction_score": 61.0, "confidence": 0.65}

    monkeypatch.setattr(
        "models.advanced_precision.create_dqn_agent",
        lambda logger: DummyDQN(),
    )
    monkeypatch.setattr(
        "models.advanced_precision.create_multimodal_analyzer",
        lambda logger: DummyMultiModal(),
    )
    monkeypatch.setattr(
        "models.advanced_precision.create_meta_learning_optimizer",
        lambda: DummyMeta(),
    )
    monkeypatch.setattr(
        "models.advanced_precision.create_advanced_ensemble",
        lambda: DummyEnsemble(),
    )
    monkeypatch.setattr(
        "models.advanced_precision.create_market_transformer",
        lambda: DummyTransformer(),
    )

    def fake_create_market_state(price_data, volume_data):
        recorded_calls["create_market_state"] = True
        return np.ones(50)

    def fake_convert_action_to_score(action):
        recorded_calls["convert_action_to_score"] = action
        return 77.0

    monkeypatch.setattr(
        "models.advanced_precision.create_market_state",
        fake_create_market_state,
    )
    monkeypatch.setattr(
        "models.advanced_precision.convert_action_to_score",
        fake_convert_action_to_score,
    )

    system = AdvancedPrecisionBreakthrough87System()

    monkeypatch.setattr(
        AdvancedPrecisionBreakthrough87System,
        "_get_market_data",
        lambda self, symbol: (np.linspace(100, 120, 60), None),
    )

    result = system.predict_87_percent_accuracy("TEST")

    assert recorded_calls["create_market_state"] is True
    assert recorded_calls["convert_action_to_score"] == 1
    assert recorded_calls["market_state_values"] == tuple(np.ones(50))
    assert result["ensemble_result"]["ensemble_prediction"] == pytest.approx(58.0)
    assert result["final_confidence"] >= 0
    assert "model_contributions" in result
