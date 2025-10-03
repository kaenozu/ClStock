"""Compatibility facade for the refactored models package.

This module re-exports the public symbols from :mod:`models` lazily so that the
refactored import paths used by the tests remain functional while the code base
transitions to the new package structure.
"""

from __future__ import annotations

from typing import Any, List

import models as _models

__all__: List[str] = list(getattr(_models, "__all__", []))


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(_models, name)
    raise AttributeError(f"module 'models_refactored' has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(__all__ + list(globals().keys())))
