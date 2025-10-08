import sys
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock


def _install_sklearn_ensemble_stub(monkeypatch):
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


def _reload_full_auto_system(monkeypatch):
    # 既存のキャッシュをクリアし、依存パッケージを再読み込み可能にする
    for name in list(sys.modules):
        if name.startswith("systems.full_auto"):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_system_accepts_dependency_injection(monkeypatch):
    _install_sklearn_ensemble_stub(monkeypatch)
    _reload_full_auto_system(monkeypatch)

    module = import_module("systems.full_auto.system")
    FullAutoInvestmentSystem = module.FullAutoInvestmentSystem

    dependencies = {
        "settings": Mock(name="settings"),
        "data_provider": Mock(name="data_provider"),
        "predictor": Mock(name="predictor"),
        "optimizer": Mock(name="optimizer"),
        "sentiment_analyzer": Mock(name="sentiment"),
        "strategy_generator": Mock(name="strategy"),
        "risk_manager": Mock(name="risk"),
        "medium_system": Mock(name="medium"),
        "backtester": Mock(name="backtester"),
        "script_generator": Mock(name="script_gen"),
    }

    system = FullAutoInvestmentSystem(**dependencies)

    for name, dependency in dependencies.items():
        assert getattr(system, name) is dependency
