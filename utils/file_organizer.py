"""Utility helpers to organize files inside a directory."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional


def _normalize_extension(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext:
        return ""
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _normalize_rules(rules: Optional[MutableMapping[str, Iterable[str]]]) -> Dict[str, set[str]]:
    normalized: Dict[str, set[str]] = {}
    if not rules:
        return normalized

    for folder, extensions in rules.items():
        folder_name = folder.strip() if folder else ""
        if not folder_name:
            continue
        normalized[folder_name] = {
            ext for ext in (_normalize_extension(ext) for ext in extensions) if ext
        }
    return normalized


def organize_files(
    directory: Path | str,
    *,
    rules: Optional[MutableMapping[str, Iterable[str]]] = None,
    default_folder: str = "others",
) -> Dict[str, List[Path]]:
    """Organize files in ``directory`` based on their extensions.

    Parameters
    ----------
    directory:
        Base directory whose immediate files will be reorganized.
    rules:
        Mapping of folder name to iterable of extensions. Extensions are
        case-insensitive and may optionally include the leading dot.
    default_folder:
        Fallback folder for files that do not match any rule.

    Returns
    -------
    Dict[str, List[pathlib.Path]]
        Mapping of destination folder names to the moved files.
    """

    base_path = Path(directory)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_path}")

    if not base_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {base_path}")

    normalized_rules = _normalize_rules(rules)
    default_folder = default_folder.strip() or "others"

    moved: Dict[str, List[Path]] = {}

    for item in base_path.iterdir():
        if item.is_dir():
            continue

        extension = item.suffix.lower()
        destination_folder: Optional[str] = None

        if extension:
            for folder, extensions in normalized_rules.items():
                if extension in extensions:
                    destination_folder = folder
                    break

        if destination_folder is None:
            if not extension and "" in {ext for extensions in normalized_rules.values() for ext in extensions}:
                # Explicit rule for files without extension
                for folder, extensions in normalized_rules.items():
                    if "" in extensions:
                        destination_folder = folder
                        break
            else:
                destination_folder = default_folder

        destination_dir = base_path / destination_folder
        destination_dir.mkdir(exist_ok=True)
        destination_path = destination_dir / item.name
        item.replace(destination_path)

        moved.setdefault(destination_folder, []).append(destination_path)

    return moved

