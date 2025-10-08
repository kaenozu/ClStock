"""Integration tests covering the Phase 2 hybrid predictor stack."""

from __future__ import annotations

from dataclasses import replace

import pytest

# Skip these integration checks when heavy numerical dependencies are missing.
pytest.importorskip("scipy", reason="SciPy is required for the refactored hybrid stack")
pytest.importorskip(
    "sklearn",
    reason="scikit-learn is required for the refactored hybrid stack",
)

from models_refactored.core.interfaces import (
    ModelConfiguration,
    ModelType,
    PredictionMode,
)
from models_refactored.hybrid.hybrid_predictor import RefactoredHybridPredictor


@pytest.mark.integration
def test_phase2_components_are_importable() -> None:
    """The refactored hybrid stack should expose the expected entry points."""
    predictor = RefactoredHybridPredictor(enable_streaming=False)

    assert isinstance(predictor, RefactoredHybridPredictor)
    assert predictor.enable_streaming is False
    assert PredictionMode.BALANCED.value == "balanced"


@pytest.mark.integration
def test_hybrid_basic_initialization_flags() -> None:
    """Flags passed to the constructor should be reflected on the instance."""
    hybrid = RefactoredHybridPredictor(
        enable_cache=True,
        enable_adaptive_optimization=False,
        enable_streaming=False,
        enable_multi_gpu=False,
        enable_real_time_learning=False,
    )

    assert hybrid.cache_enabled is True
    assert hybrid.adaptive_optimization_enabled is False
    assert hybrid.streaming_enabled is False
    assert hybrid.multi_gpu_enabled is False
    assert hybrid.real_time_learning_enabled is False


@pytest.mark.integration
def test_streaming_flag_controls_attribute() -> None:
    """Enabling streaming should flip the corresponding configuration flag."""
    hybrid = RefactoredHybridPredictor(enable_streaming=True)

    assert hybrid.streaming_enabled is True
    assert hybrid.enable_streaming is True


@pytest.mark.integration
def test_multi_gpu_flag_controls_attribute() -> None:
    """The multi-GPU switch should be propagated to the predictor state."""
    hybrid = RefactoredHybridPredictor(enable_multi_gpu=True)

    assert hybrid.multi_gpu_enabled is True
    assert hybrid.enable_multi_gpu is True


@pytest.mark.integration
def test_custom_configuration_is_preserved() -> None:
    """The provided configuration should survive the initialization process."""
    config = ModelConfiguration(model_type=ModelType.HYBRID)
    config.custom_params["foo"] = "bar"

    hybrid = RefactoredHybridPredictor(config=replace(config))

    assert hybrid.config.model_type is ModelType.HYBRID
    assert hybrid.config.custom_params["foo"] == "bar"
    assert (
        hybrid.ensemble_predictor.config.custom_params["parent_model_type"] == "hybrid"
    )


@pytest.mark.integration
def test_disabling_real_time_learning_leaves_predictor_idle() -> None:
    """When real-time learning is disabled, the learner should not be initialised."""
    hybrid = RefactoredHybridPredictor(enable_real_time_learning=False)

    assert hybrid.real_time_learning_enabled is False
    assert hybrid.enable_real_time_learning is False
    assert getattr(hybrid, "real_time_learner", None) is None
