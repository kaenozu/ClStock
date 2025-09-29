"""A lightweight fallback stub for pandas used during testing."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]


def _is_repo_path(entry: str) -> bool:
    try:
        return Path(entry).resolve() == _repo_root
    except Exception:
        return False


_search_paths = [p for p in sys.path if not _is_repo_path(p)]
_spec = importlib.machinery.PathFinder.find_spec("pandas", _search_paths)

if _spec and _spec.loader:
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
else:
    raise ImportError("Install pandas before running ClStock.")
