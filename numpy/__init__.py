"""A lightweight fallback stub for numpy used during testing."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
_search_paths = [p for p in sys.path if p != _repo_root]
_spec = importlib.machinery.PathFinder.find_spec("numpy", _search_paths)

if _spec and _spec.loader:
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
else:
    import math
    import numbers
    from typing import Iterable

    class ndarray(list):
        def flatten(self):  # pragma: no cover - simple helper
            return ndarray(self)

    nan = float("nan")
    number = numbers.Number

    def asarray(data: Iterable, dtype=None):  # pragma: no cover - stub
        if isinstance(data, ndarray):
            return ndarray(data)
        return ndarray(list(data))

    def full(length: int, fill_value, dtype=None):  # pragma: no cover - stub
        return ndarray([fill_value for _ in range(length)])

    def mean(values: Iterable[float]):  # pragma: no cover - stub
        values = list(values)
        if not values:
            return math.nan
        return sum(values) / len(values)

    def log1p(value):  # pragma: no cover - stub
        return math.log1p(value)

    def tanh(value):  # pragma: no cover - stub
        return math.tanh(value)

    def std(values: Iterable[float]):  # pragma: no cover - stub
        values = list(values)
        if not values:
            return 0.0
        avg = mean(values)
        return (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5

    def clip(values: Iterable[float], min_value, max_value):  # pragma: no cover - stub
        return ndarray(max(min(v, max_value), min_value) for v in values)

    def array(values: Iterable):  # pragma: no cover - stub
        return asarray(values)

    def zeros(length: int):  # pragma: no cover - stub
        return ndarray([0.0] * length)

    def ones(length: int):  # pragma: no cover - stub
        return ndarray([1.0] * length)

    __all__ = [
        "ndarray",
        "nan",
        "number",
        "asarray",
        "full",
        "mean",
        "log1p",
        "tanh",
        "std",
        "clip",
        "array",
        "zeros",
        "ones",
    ]
