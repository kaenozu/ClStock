"""Repository integrity safeguards."""

from pathlib import Path


def test_development_database_not_committed():
    """Ensure the development SQLite database file is not committed."""
    repo_root = Path(__file__).resolve().parents[2]
    db_path = repo_root / "clstock.db"
    assert (
        not db_path.exists()
    ), "Development database file 'clstock.db' should not be committed to the repository."
