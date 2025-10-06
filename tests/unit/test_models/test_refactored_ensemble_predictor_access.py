"""Lightweight behavioural checks for RefactoredEnsemblePredictor.

These tests ensure the modern ensemble predictor is importable and exposes
basic introspection helpers required by higher level integration tests.
"""

from models.core.interfaces import ModelConfiguration, ModelType
from models.ensemble.ensemble_predictor import RefactoredEnsemblePredictor


def test_predictor_is_instantiable_without_arguments():
    predictor = RefactoredEnsemblePredictor()

    assert predictor.config.model_type == ModelType.ENSEMBLE
    assert predictor.get_prediction_period() == "1y"


def test_predictor_respects_explicit_configuration():
    config = ModelConfiguration(model_type=ModelType.ENSEMBLE, cache_enabled=False)

    predictor = RefactoredEnsemblePredictor(config=config)

    assert predictor.config is config
    assert predictor.config.cache_enabled is False
