"""Stub of sklearn.metrics with minimal numeric implementations."""

from __future__ import annotations

import numpy as np

__all__ = [
    "accuracy_score",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
]


def _to_arrays(y_true, y_pred):
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()
    return y_t, y_p


def mean_squared_error(y_true, y_pred):  # pragma: no cover - lightweight implementation
    y_t, y_p = _to_arrays(y_true, y_pred)
    return float(np.mean((y_t - y_p) ** 2))


def mean_absolute_error(y_true, y_pred):  # pragma: no cover - lightweight implementation
    y_t, y_p = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def r2_score(y_true, y_pred):  # pragma: no cover - lightweight implementation
    y_t, y_p = _to_arrays(y_true, y_pred)
    total = np.sum((y_t - np.mean(y_t)) ** 2)
    resid = np.sum((y_t - y_p) ** 2)
    if total == 0:
        return 0.0
    return float(1 - resid / total)


def accuracy_score(y_true, y_pred):  # pragma: no cover - lightweight implementation
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if y_t.size == 0:
        return 0.0
    return float(np.mean(y_t == y_p))
