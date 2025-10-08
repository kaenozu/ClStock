from pathlib import Path
from typing import Optional

from . import HTTPException


class StaticFiles:
    """極めて簡易的なStaticFilesスタブ"""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        if not self.directory.exists() or not self.directory.is_dir():
            raise RuntimeError(f"Static directory not found: {self.directory}")

    def get_response(self, path: Optional[str] = None) -> str:
        target = self.directory / (path or "")
        target = target.resolve()
        base = self.directory.resolve()
        if base not in target.parents and target != base:
            raise HTTPException(status_code=404, detail="Not Found")
        if target.is_dir():
            raise HTTPException(status_code=404, detail="Directory listing not supported")
        if not target.exists():
            raise HTTPException(status_code=404, detail="Not Found")
        return target.read_text(encoding="utf-8")

    def __call__(self, path: Optional[str] = None) -> str:
        return self.get_response(path)
