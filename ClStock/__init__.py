"""Top-level package for the ClStock project."""

from __future__ import annotations

from pathlib import Path

# The project is structured with most modules living directly under the
# repository root. By exposing the repository root as a package search path
# we can import modules such as ``ClStock.systems.process_manager`` without
# mutating ``sys.path`` during runtime initialisation.
_package_dir = Path(__file__).resolve().parent
_project_root = _package_dir.parent

# ``__path__`` defines where Python looks for submodules of ``ClStock``.
# We include both the package directory itself (to allow future package-local
# modules) and the repository root where the existing modules reside.
__path__ = [str(_package_dir), str(_project_root)]

__all__ = []
