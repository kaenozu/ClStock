"""Tests for ensuring dashboard HTML fixtures are available for visualization tests."""

from pathlib import Path


def test_dashboard_fixture_exists():
    """The generated dashboard HTML should live under tests/resources for reuse."""
    repo_root = Path(__file__).resolve().parents[2]
    resource_path = repo_root / "tests" / "resources" / "test_dashboard.html"

    assert resource_path.exists(), f"Expected dashboard fixture at {resource_path}"
    assert resource_path.is_file(), "Dashboard fixture path should be a file"
