"""System modules namespace for the ClStock package."""

from pathlib import Path

# Ensure both the local ``ClStock/systems`` modules and the original ``systems``
# package (used throughout the codebase) are importable via this namespace.
__path__ = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[1] / "systems"),
]

__all__ = [
    "process_manager",
]

