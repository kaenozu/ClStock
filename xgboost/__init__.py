"""XGBoost stub module for testing environment."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]


def _is_repo_path(entry: str) -> bool:
    try:
        return Path(entry).resolve() == _repo_root
    except Exception:
        return False


_search_paths = [p for p in sys.path if not _is_repo_path(p)]
_spec = importlib.machinery.PathFinder.find_spec("xgboost", _search_paths)

if _spec and _spec.loader:
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
else:
    stub_module = types.ModuleType(__name__)

    class Booster:  # pragma: no cover - placeholder type
        def __init__(self, *_, **__):
            raise RuntimeError("XGBoost is unavailable in this test environment")

    class DMatrix:  # pragma: no cover - placeholder type
        def __init__(self, *_, **__):
            raise RuntimeError("XGBoost is unavailable in this test environment")

    def train(*_, **__):  # pragma: no cover - placeholder function
        raise RuntimeError("XGBoost is unavailable in this test environment")

    stub_module.Booster = Booster
    stub_module.DMatrix = DMatrix
    stub_module.train = train

    sys.modules[__name__] = stub_module
    globals().update(stub_module.__dict__)
