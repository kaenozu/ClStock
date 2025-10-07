"""テスト向けの軽量な SHAP 互換スタブ。"""

from __future__ import annotations

import numpy as np

__all__ = [
    "KernelExplainer",
    "TreeExplainer",
    "LinearExplainer",
    "DeepExplainer",
    "summary_plot",
    "waterfall_plot",
    "force_plot",
    "Explanation",
]


class _BaseExplainer:
    def __init__(self, *_, **__):  # pragma: no cover - 単純スタブ
        self.expected_value = 0.0

    def shap_values(self, X):  # pragma: no cover - 単純スタブ
        array = np.asarray(X)
        return np.zeros_like(array, dtype=float)


class KernelExplainer(_BaseExplainer):
    pass


class TreeExplainer(_BaseExplainer):
    pass


class LinearExplainer(_BaseExplainer):
    pass


class DeepExplainer(_BaseExplainer):
    pass


def summary_plot(*_, **__):  # pragma: no cover - 単純スタブ
    return None


def waterfall_plot(*_, **__):  # pragma: no cover - 単純スタブ
    return None


def force_plot(*_, **__):  # pragma: no cover - 単純スタブ
    return None


class Explanation(np.ndarray):
    """最小限の `shap.Explanation` 互換オブジェクト。"""

    def __new__(cls, values=None, base_values=None, data=None, feature_names=None):
        values = np.asarray(values if values is not None else 0.0, dtype=float)
        obj = np.asarray(values, dtype=float).view(cls)
        obj.base_values = base_values
        obj.data = data
        obj.feature_names = feature_names
        return obj

