"""Blackフォーマット検証テスト"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_black_formatting() -> None:
    """Blackの--checkが失敗しないことを検証する。"""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "black", "--check", "--diff", "--fast", "."],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = (
            "Black formatting check failed.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        raise AssertionError(message)
