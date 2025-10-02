import sqlite3
import sys
import types
from types import SimpleNamespace
from datetime import datetime

import pytest


def _install_fastapi_stubs() -> None:
    """Inject lightweight stubs for optional web dependencies used in tests."""
    fastapi_stub = types.ModuleType("fastapi")

    class _FastAPI(SimpleNamespace):
        def mount(self, *_, **__):
            return None

        def get(self, *_, **__):
            def decorator(func):
                return func

            return decorator

        post = get
        put = get
        delete = get

    fastapi_stub.FastAPI = lambda *args, **kwargs: _FastAPI()
    fastapi_stub.Request = object

    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.HTMLResponse = object

    templating_stub = types.ModuleType("fastapi.templating")

    class _Templates(SimpleNamespace):
        def __init__(self, *_, **__):
            super().__init__(env=SimpleNamespace(filters={}))

    templating_stub.Jinja2Templates = lambda *args, **kwargs: _Templates()

    staticfiles_stub = types.ModuleType("fastapi.staticfiles")
    staticfiles_stub.StaticFiles = lambda *args, **kwargs: SimpleNamespace()

    uvicorn_stub = types.ModuleType("uvicorn")
    numpy_stub = types.ModuleType("numpy")
    models_stub = types.ModuleType("models")
    models_stub.__path__ = []
    precision_pkg = types.ModuleType("models.precision")
    precision_pkg.__path__ = []
    precision_module = types.ModuleType("models.precision.precision_87_system")

    class _Precision87BreakthroughSystem(SimpleNamespace):
        def predict_with_87_precision(self, symbol: str):
            return {}

    precision_module.Precision87BreakthroughSystem = _Precision87BreakthroughSystem
    models_stub.precision = precision_pkg
    precision_pkg.precision_87_system = precision_module

    sys.modules.setdefault("fastapi", fastapi_stub)
    sys.modules.setdefault("fastapi.responses", responses_stub)
    sys.modules.setdefault("fastapi.templating", templating_stub)
    sys.modules.setdefault("fastapi.staticfiles", staticfiles_stub)
    sys.modules.setdefault("uvicorn", uvicorn_stub)
    sys.modules.setdefault("numpy", numpy_stub)
    sys.modules.setdefault("models", models_stub)
    sys.modules.setdefault("models.precision", precision_pkg)
    sys.modules.setdefault("models.precision.precision_87_system", precision_module)


_install_fastapi_stubs()

from app.personal_dashboard import PersonalDashboard


@pytest.fixture
def dummy_settings(tmp_path):
    db_path = tmp_path / "personal.db"
    database = SimpleNamespace(personal_portfolio_db=db_path)
    prediction = SimpleNamespace(achieved_accuracy=0)
    return SimpleNamespace(database=database, prediction=prediction)


@pytest.fixture(autouse=True)
def override_dependencies(monkeypatch, dummy_settings):
    monkeypatch.setattr("app.personal_dashboard.get_settings", lambda: dummy_settings)
    monkeypatch.setattr("app.personal_dashboard.StockDataProvider", lambda: SimpleNamespace())


def test_get_recent_predictions_accepts_days_parameter(dummy_settings):
    dashboard = PersonalDashboard()

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
