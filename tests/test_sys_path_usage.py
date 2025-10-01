import importlib
import sys


class PathGuard(list):
    """List-like guard that fails on mutation."""

    def append(self, item):  # pragma: no cover - behaviour verified via exception
        raise AssertionError(f"sys.path.append called with {item!r}")

    def extend(self, iterable):  # pragma: no cover - behaviour verified via exception
        raise AssertionError(f"sys.path.extend called with {list(iterable)!r}")

    def insert(self, index, item):  # pragma: no cover - behaviour verified via exception
        raise AssertionError(f"sys.path.insert called with {index!r}, {item!r}")


def test_imports_do_not_modify_sys_path(monkeypatch):
    """Ensure importing modules does not attempt to mutate sys.path."""

    for module_name in (
        "clstock_cli",
        "ClStock",
        "ClStock.systems",
        "ClStock.systems.process_manager",
        "systems.process_manager",
    ):
        sys.modules.pop(module_name, None)

    guarded_path = PathGuard(sys.path)
    monkeypatch.setattr(sys, "path", guarded_path, raising=False)

    importlib.import_module("clstock_cli")
    importlib.import_module("ClStock.systems.process_manager")
