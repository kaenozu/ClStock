import sys
import types
from importlib import import_module
from types import SimpleNamespace

import pytest


def _install_sklearn_stub(monkeypatch):
    class _DummyEstimator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def fit(self, *_args, **_kwargs):  # pragma: no cover - stubbed method
            return self

        def predict(self, *_args, **_kwargs):  # pragma: no cover - stubbed method
            return []

    ensemble_stub = SimpleNamespace(
        GradientBoostingRegressor=_DummyEstimator,
        RandomForestRegressor=_DummyEstimator,
    )
    linear_model_stub = SimpleNamespace(LinearRegression=_DummyEstimator)

    def _mean_squared_error(*_args, **_kwargs):  # pragma: no cover - stubbed function
        return 0.0

    metrics_stub = SimpleNamespace(mean_squared_error=_mean_squared_error)
    preprocessing_stub = SimpleNamespace(StandardScaler=_DummyEstimator)
    sklearn_stub = SimpleNamespace(
        ensemble=ensemble_stub,
        linear_model=linear_model_stub,
        metrics=metrics_stub,
        preprocessing=preprocessing_stub,
    )
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_stub)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", ensemble_stub)
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", linear_model_stub)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics_stub)
    monkeypatch.setitem(sys.modules, "sklearn.preprocessing", preprocessing_stub)


def _reload_full_auto_modules(monkeypatch):
    for name in list(sys.modules):
        if name.startswith("systems.full_auto"):
            monkeypatch.delitem(sys.modules, name, raising=False)


@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], None),
        (["--max-tickers", "7"], 7),
    ],
)
def test_main_delegates_to_asyncio_run(monkeypatch, argv, expected):
    """main() は asyncio.run に適切なコルーチンを渡すべき。"""
    _install_sklearn_stub(monkeypatch)
    _reload_full_auto_modules(monkeypatch)
    module = import_module("systems.full_auto.cli")

    captured = {}

    async def fake_run_full_auto(*, max_symbols=None):
        captured["args"] = max_symbols
        return "ok"

    def fake_asyncio_run(coro):
        captured["coro"] = coro
        try:
            result = coro.send(None)
        except StopIteration as exc:  # pragma: no cover - defensive
            return exc.value
        else:
            return result

    monkeypatch.setattr(module, "run_full_auto", fake_run_full_auto)
    monkeypatch.setattr(module.asyncio, "run", fake_asyncio_run)

    outcome = module.main(argv)

    assert outcome == 0
    assert captured["args"] == expected
    assert isinstance(captured["coro"], types.CoroutineType)
