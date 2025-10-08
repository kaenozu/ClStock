import sqlite3
from datetime import datetime
from types import SimpleNamespace

import pytest


@pytest.fixture
def dummy_settings(tmp_path):
    db_path = tmp_path / "personal.db"
    database = SimpleNamespace(personal_portfolio_db=db_path)
    prediction = SimpleNamespace(achieved_accuracy=0)
    return SimpleNamespace(database=database, prediction=prediction)


@pytest.fixture(autouse=True)
def override_dependencies(monkeypatch, dummy_settings, fastapi_dependency_stubs):
    monkeypatch.setattr("app.personal_dashboard.get_settings", lambda: dummy_settings)
    monkeypatch.setattr(
        "app.personal_dashboard.StockDataProvider",
        lambda: SimpleNamespace(),
    )


@pytest.fixture
def personal_dashboard(fastapi_dependency_stubs):
    from app.personal_dashboard import PersonalDashboard

    return PersonalDashboard()


def test_get_recent_predictions_accepts_days_parameter(
    dummy_settings,
    personal_dashboard,
):
    dashboard = personal_dashboard

    conn = sqlite3.connect(dummy_settings.database.personal_portfolio_db)
    conn.execute(
        """
        INSERT INTO predictions (symbol, prediction_date, predicted_price, actual_price, confidence, accuracy, system_used)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("TEST", datetime.now(), 100.0, 101.0, 0.9, 0.8, "test"),
    )
    conn.commit()
    conn.close()

    try:
        result = dashboard.get_recent_predictions(days=3)
    except sqlite3.ProgrammingError as exc:  # pragma: no cover
        pytest.fail(f"Unexpected sqlite ProgrammingError: {exc}")

    assert isinstance(result, list)
