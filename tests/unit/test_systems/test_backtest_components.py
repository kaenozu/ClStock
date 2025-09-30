import importlib
import sys
import types
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trading.backtest import BacktestOptimizer, BacktestRunner, generate_backtest_charts, generate_recommendations


@pytest.fixture
def backtest_engine_module(monkeypatch):
    """Load ``trading.backtest_engine`` with lightweight stubs."""

    dummy_precision_module = types.ModuleType("models_new.precision.precision_87_system")

    class _Precision:
        pass

    dummy_precision_module.Precision87BreakthroughSystem = _Precision
    monkeypatch.setitem(
        sys.modules, "models_new.precision.precision_87_system", dummy_precision_module
    )

    dummy_data_module = types.ModuleType("data.stock_data")

    class _Provider:
        def get_stock_data(self, *args, **kwargs):  # pragma: no cover - not used
            raise NotImplementedError

        def calculate_technical_indicators(self, data):  # pragma: no cover - not used
            return data

    dummy_data_module.StockDataProvider = _Provider
    monkeypatch.setitem(sys.modules, "data.stock_data", dummy_data_module)

    sys.modules.pop("trading.backtest_engine", None)
    module = importlib.import_module("trading.backtest_engine")
    yield module
    sys.modules.pop("trading.backtest_engine", None)


class DummyPortfolioManager:
    def __init__(self, initial_capital: float):
        self.current_cash = initial_capital
        self.positions = {}

    def update_positions(self) -> None:
        pass

    def add_position(self, symbol, quantity, price, signal_type) -> None:
        self.positions[symbol] = types.SimpleNamespace(
            quantity=quantity, market_value=quantity * price
        )
        self.current_cash -= quantity * price


class DummyRiskManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

    def can_open_position(self, symbol, position_size):
        return position_size <= self.initial_capital


class DummyTradeRecorder:
    def __init__(self):
        self._trades = []

    def record_trade(self, trade):
        self._trades.append(trade)

    def get_completed_trades(self):
        return []


class DummyPerformanceTracker:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

    def update_performance(self, *args, **kwargs):
        pass


class DummyDataProvider:
    def __init__(self):
        idx = pd.date_range("2020-01-01", periods=120, freq="B")
        self.df = pd.DataFrame({"Close": np.linspace(100, 110, len(idx))}, index=idx)

    def get_stock_data(self, symbol, start_date=None, end_date=None):
        return self.df

    def calculate_technical_indicators(self, data):
        return data


@pytest.mark.usefixtures("backtest_engine_module")
def test_backtest_runner_produces_result(monkeypatch, backtest_engine_module):
    BacktestConfig = backtest_engine_module.BacktestConfig
    BacktestResult = backtest_engine_module.BacktestResult

    monkeypatch.setattr(
        "trading.backtest.runner.DemoPortfolioManager", DummyPortfolioManager
    )
    monkeypatch.setattr("trading.backtest.runner.DemoRiskManager", DummyRiskManager)
    monkeypatch.setattr("trading.backtest.runner.TradeRecorder", DummyTradeRecorder)
    monkeypatch.setattr(
        "trading.backtest.runner.PerformanceTracker", DummyPerformanceTracker
    )

    config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 1, 31),
        initial_capital=1_000_000,
        target_symbols=["AAA"],
    )

    runner = BacktestRunner(config, object(), DummyDataProvider())

    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(np.random, "random", lambda *args, **kwargs: 0.0)

    result = runner.run_backtest(["AAA"])

    assert isinstance(result, BacktestResult)
    assert result.config is config
    assert result.portfolio_values  # 簡易検証: 値が蓄積されていること


def test_optimizer_selects_best_parameters(backtest_engine_module):
    BacktestConfig = backtest_engine_module.BacktestConfig
    BacktestResult = backtest_engine_module.BacktestResult

    base_config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 2, 1),
        initial_capital=1_000_000,
        target_symbols=["AAA"],
    )

    optimizer = BacktestOptimizer()

    def engine_factory(config):
        score = config.precision_threshold / 100
        result = BacktestResult(
            config=config,
            start_date=config.start_date,
            end_date=config.end_date,
            total_return=score,
            annualized_return=score,
            volatility=1.0,
            sharpe_ratio=score,
            sortino_ratio=score,
            max_drawdown=0.1,
            calmar_ratio=1.0,
            total_trades=1,
            win_rate=50.0,
            profit_factor=1.0,
            precision_87_trades=1,
            precision_87_success_rate=50.0,
            final_value=config.initial_capital * (1 + score),
            benchmark_return=0.0,
            excess_return=score,
            beta=1.0,
            alpha=score,
            information_ratio=score,
            var_95=0.0,
            expected_shortfall=0.0,
            total_costs=0.0,
            total_tax=0.0,
            daily_returns=[score],
            trade_history=[],
            portfolio_values=[
                (config.start_date, config.initial_capital),
                (config.end_date, config.initial_capital * (1 + score)),
            ],
        )
        return types.SimpleNamespace(run_backtest=lambda: result)

    parameter_ranges = {"precision_threshold": [80.0, 90.0]}

    outcome = optimizer.optimize_parameters(
        base_config=base_config,
        parameter_ranges=parameter_ranges,
        optimization_metric="sharpe_ratio",
        engine_factory=engine_factory,
    )

    assert outcome["best_parameters"]["precision_threshold"] == 90.0
    assert len(outcome["all_results"]) == 2


def test_reporting_helpers(backtest_engine_module):
    BacktestResult = backtest_engine_module.BacktestResult
    BacktestConfig = backtest_engine_module.BacktestConfig

    config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 1, 10),
        initial_capital=1_000_000,
    )

    result = BacktestResult(
        config=config,
        start_date=config.start_date,
        end_date=config.end_date,
        total_return=0.1,
        annualized_return=0.1,
        volatility=0.1,
        sharpe_ratio=0.5,
        sortino_ratio=0.4,
        max_drawdown=0.25,
        calmar_ratio=0.4,
        total_trades=10,
        win_rate=40.0,
        profit_factor=1.0,
        precision_87_trades=2,
        precision_87_success_rate=50.0,
        final_value=1_100_000,
        benchmark_return=0.0,
        excess_return=0.1,
        beta=1.0,
        alpha=0.1,
        information_ratio=0.2,
        var_95=-0.05,
        expected_shortfall=-0.06,
        total_costs=1000.0,
        total_tax=2000.0,
        daily_returns=[0.01, -0.02, 0.03],
        trade_history=[{"position_size": 1000.0, "profit_loss": -100.0}],
        portfolio_values=[
            (config.start_date, config.initial_capital),
            (config.end_date, 1_100_000),
        ],
    )

    charts = generate_backtest_charts(result)
    assert "equity_curve" in charts

    recommendations = generate_recommendations(result)
    assert any("リスク管理" in msg for msg in recommendations)
    assert any("精度取引" in msg for msg in recommendations)
