"""Torch stub for testing environment."""

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
_spec = importlib.machinery.PathFinder.find_spec("torch", _search_paths)

if _spec and _spec.loader:
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
else:
    stub_module = types.ModuleType(__name__)

    class Tensor:  # pragma: no cover - placeholder type
        def __init__(self, *_, **__):
            raise RuntimeError("torch is unavailable in this test environment")

    def tensor(*_, **__):  # pragma: no cover - placeholder function
        raise RuntimeError("torch is unavailable in this test environment")

    stub_module.Tensor = Tensor
    stub_module.tensor = tensor

    sys.modules[__name__] = stub_module
    globals().update(stub_module.__dict__)
