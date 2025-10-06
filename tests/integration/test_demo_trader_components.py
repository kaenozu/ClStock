"""Integration wiring checks for the demo trading system.

This suite focuses on verifying that ``DemoTrader`` wires together the
trading strategy, risk manager, and other subsystems with the thresholds
documented in ``docs/DEMO_TRADING_SYSTEM_README.md``.  The real
``data.stock_data`` module in the repository is heavy and depends on
external services, so the tests replace it with a very small stub module
before importing ``DemoTrader``.
"""

from __future__ import annotations

import importlib
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest


@pytest.fixture
def stub_stock_data_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Provide a lightweight stand-in for :mod:`data.stock_data`.

    The production module performs network calls and is currently incomplete
    in the repository snapshot that backs these tests.  For the wiring checks
    we only need an importable ``StockDataProvider`` class that exposes the
    ``jp_stock_codes`` attribute used in a few convenience helpers.
    """
    stub_module = types.ModuleType("data.stock_data")

    class _StubStockDataProvider:  # pragma: no cover - behaviour verified indirectly
        def __init__(self) -> None:
            self.jp_stock_codes = {"7203": "トヨタ自動車", "6758": "ソニー"}

    stub_module.StockDataProvider = _StubStockDataProvider
    monkeypatch.setitem(sys.modules, "data.stock_data", stub_module)

    models_new_module = types.ModuleType("models_new")
    models_new_spec = ModuleSpec(name="models_new", loader=None, is_package=True)
    models_new_spec.submodule_search_locations = []
    models_new_module.__spec__ = models_new_spec
    models_new_module.__path__ = []

    models_new_precision_package = types.ModuleType("models_new.precision")
    precision_spec = ModuleSpec(
        name="models_new.precision", loader=None, is_package=True,
    )
    precision_spec.submodule_search_locations = []
    models_new_precision_package.__spec__ = precision_spec
    models_new_precision_package.__path__ = []

    precision_module = types.ModuleType("models.precision.precision_87_system")

    class _StubPrecision87BreakthroughSystem:  # pragma: no cover - behaviour verified indirectly
        def predict_with_87_precision(self, symbol: str) -> dict[str, float]:
            return {
                "final_prediction": 125.0,
                "current_price": 100.0,
                "final_confidence": 0.92,
                "final_accuracy": 0.9,
                "predicted_change_rate": 0.12,
            }

    precision_module.Precision87BreakthroughSystem = _StubPrecision87BreakthroughSystem

    monkeypatch.setitem(sys.modules, "models_new", models_new_module)
    monkeypatch.setitem(
        sys.modules, "models_new.precision", models_new_precision_package,
    )
    monkeypatch.setitem(
        sys.modules, "models_new.precision.precision_87_system", precision_module,
    )

    models_new_module.precision = models_new_precision_package
    models_new_precision_package.precision_87_system = precision_module

    # ``sklearn`` transitively depends on SciPy which is not available in the
    # execution environment for these tests.  Provide a very small stand-in so
    # the import machinery can complete successfully when ``DemoTrader`` pulls
    # in the production trading strategy module tree.
    scipy_module = types.ModuleType("scipy")
    scipy_sparse_module = types.ModuleType("scipy.sparse")

    def _identity(
        *args: object, **kwargs: object,
    ) -> None:  # pragma: no cover - trivial stub
        return None

    scipy_sparse_module.csr_matrix = _identity
    scipy_sparse_module.issparse = (
        lambda *args, **kwargs: False
    )  # pragma: no cover - trivial stub
    scipy_module.sparse = scipy_sparse_module

    monkeypatch.setitem(sys.modules, "scipy", scipy_module)
    monkeypatch.setitem(sys.modules, "scipy.sparse", scipy_sparse_module)

    # Provide a lightweight package spec for ``trading`` so we can import
    # ``trading.demo_trader`` without executing the heavy package ``__init__``.
    package_path = Path(__file__).resolve().parents[2] / "trading"
    trading_package = types.ModuleType("trading")
    package_spec = ModuleSpec(name="trading", loader=None, is_package=True)
    package_spec.submodule_search_locations = [str(package_path)]
    trading_package.__spec__ = package_spec
    trading_package.__path__ = [str(package_path)]
    monkeypatch.setitem(sys.modules, "trading", trading_package)

    # Ensure the repository root is on ``sys.path`` for the duration of the test.
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        monkeypatch.syspath_prepend(repo_root)

    return stub_module


def test_demo_trader_initializes_core_components(
    stub_stock_data_module: types.ModuleType,
) -> None:
    """``DemoTrader`` should respect risk/precision thresholds on init."""
    # ``trading.demo_trader`` imports ``data.stock_data`` at module import time,
    # so we reload it after injecting the stub to make sure the lightweight
    # provider is used during object construction.
    demo_trader_module = importlib.import_module("trading.demo_trader")
    demo_trader_module = importlib.reload(demo_trader_module)

    trader = demo_trader_module.DemoTrader(
        initial_capital=500_000,
        precision_threshold=87.0,
        confidence_threshold=0.8,
    )

    strategy = trader.trading_strategy
    assert strategy.precision_threshold == pytest.approx(87.0)
    assert strategy.confidence_threshold == pytest.approx(0.8)
    assert strategy.stop_loss_pct == pytest.approx(0.05)
    assert strategy.take_profit_pct == pytest.approx(0.15)

    risk_limits = trader.risk_manager.risk_limits
    assert risk_limits.max_position_size == pytest.approx(0.1)
    assert risk_limits.max_sector_exposure == pytest.approx(0.3)
    assert risk_limits.max_drawdown == pytest.approx(0.2)
