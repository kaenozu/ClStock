"""Stub of sklearn.preprocessing providing StandardScaler."""

from __future__ import annotations

__all__ = ["StandardScaler"]


class StandardScaler:
    def fit(self, X, y=None):  # pragma: no cover - stub
        return self

    def transform(self, X):  # pragma: no cover - stub
        return X

    def fit_transform(self, X, y=None):  # pragma: no cover - stub
        self.fit(X, y)
        return self.transform(X)
