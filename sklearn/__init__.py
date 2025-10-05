"""A lightweight fallback stub for sklearn used during testing."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_search_paths = [p for p in sys.path if Path(p).resolve() != _repo_root]
_spec = importlib.machinery.PathFinder.find_spec("sklearn", _search_paths)

if _spec and _spec.loader and getattr(_spec, "origin", None) != __file__:
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
else:
    __all__ = []
