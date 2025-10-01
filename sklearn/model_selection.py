"""Minimal sklearn.model_selection stub for test environment."""
from __future__ import annotations


def train_test_split(*_, **__):  # pragma: no cover - placeholder
    raise RuntimeError("sklearn.model_selection is unavailable in this test environment")


def cross_val_score(*_, **__):  # pragma: no cover - placeholder
    raise RuntimeError("sklearn.model_selection is unavailable in this test environment")


class TimeSeriesSplit:  # pragma: no cover - placeholder stub
    """Placeholder for compatibility in environments without scikit-learn."""

    def __init__(self, *args, **kwargs):
        self.n_splits = kwargs.get("n_splits", args[0] if args else None)

    def split(self, *_, **__):
        raise RuntimeError("TimeSeriesSplit.split is unavailable in this test environment")
