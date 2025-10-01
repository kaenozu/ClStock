"""Minimal sklearn.model_selection stub for test environment."""
from __future__ import annotations


def train_test_split(*_, **__):  # pragma: no cover - placeholder
    raise RuntimeError("sklearn.model_selection is unavailable in this test environment")


def cross_val_score(*_, **__):  # pragma: no cover - placeholder
    raise RuntimeError("sklearn.model_selection is unavailable in this test environment")
