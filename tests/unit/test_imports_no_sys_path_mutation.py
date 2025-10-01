import importlib
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    ["clstock_cli", "ClStock.systems.process_manager"],
)
def test_import_does_not_modify_sys_path(monkeypatch, module_name):
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == module_name or name.startswith(f"{module_name}.")
    }
    if module_name.startswith("ClStock."):
        clstock_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "ClStock" or name.startswith("ClStock.")
        }
        original_modules.update(clstock_modules)
    for name in original_modules:
        sys.modules.pop(name, None)

    class GuardedPath(list):
        def append(self, item):
            raise AssertionError(
                f"sys.path.append was called with {item!r} during import of {module_name}"
            )

        def extend(self, items):
            raise AssertionError(
                f"sys.path.extend was called with {items!r} during import of {module_name}"
            )

        def insert(self, index, item):
            raise AssertionError(
                f"sys.path.insert was called with {item!r} during import of {module_name}"
            )

    guarded_path = GuardedPath(sys.path)
    monkeypatch.setattr(sys, "path", guarded_path)

    try:
        importlib.import_module(module_name)
    finally:
        sys.modules.update(original_modules)
