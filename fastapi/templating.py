from pathlib import Path
from typing import Any, Dict

from .responses import HTMLResponse


class Jinja2Templates:
    """Jinja2互換の極簡易スタブ。テンプレートは未評価で返す。"""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.env = type("_Env", (), {"filters": {}})()

    def TemplateResponse(self, name: str, context: Dict[str, Any]) -> HTMLResponse:  # noqa: N802
        template_path = self.directory / name
        content = template_path.read_text(encoding="utf-8") if template_path.exists() else ""
        return HTMLResponse(content=content, status_code=200)
