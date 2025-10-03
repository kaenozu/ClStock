"""Compatibility module for ensemble predictors."""

from __future__ import annotations

from models.ensemble.ensemble_predictor import (  # type: ignore F401
    EnsemblePredictor,
    RefactoredEnsemblePredictor,
)

__all__ = ["EnsemblePredictor", "RefactoredEnsemblePredictor"]
