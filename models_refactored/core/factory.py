"""Compatibility wrapper for the refactored predictor factory."""

from __future__ import annotations

from models.core.factory import PredictorFactory  # type: ignore F401

__all__ = ["PredictorFactory"]
