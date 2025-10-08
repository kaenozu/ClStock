import importlib
import sys
import types
from datetime import datetime

import numpy as np
import pandas as pd

import trading.backtest as backtest_module
from trading.trading_strategy import SignalType, TradingSignal


class DummyDataProvider:
    def __init__(self):
        idx = pd.date_range("2020-01-01", periods=160, freq="B")
        self.df = pd.DataFrame({"Close": np.linspace(100, 110, len(idx))}, index=idx)

    def get_stock_data(self, symbol, start_date=None, end_date=None):
        data = self.df.copy()
        if start_date is not None:
            data = data.loc[start_date:]
        if end_date is not None:
            data = data.loc[:end_date]
        return data

    def calculate_technical_indicators(self, data):
        return data

    def price_at(self, as_of: datetime) -> float:
        if as_of is None:
            return float(self.df.iloc[-1]["Close"])
        series = self.df.loc[:as_of]
        return (
            float(series.iloc[-1]["Close"])
            if not series.empty
            else float(self.df.iloc[0]["Close"])
        )


class DummyTradingStrategy:
    def __init__(self, data_provider: DummyDataProvider):
        self.data_provider = data_provider
        self.precision_system = types.SimpleNamespace(
            evaluation_horizon=3, evaluation_window=60
        )
        self.commission_rate = 0.0
        self.spread_rate = 0.0
        self.slippage_rate = 0.0
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.15
        self.min_expected_return = 0.01
        self._opened_dates = set()

    def generate_trading_signal(
        self, symbol: str, current_capital: float, *, as_of: datetime | None = None
    ) -> TradingSignal | None:
        if current_capital <= 0:
            return None
        price = self.data_provider.price_at(as_of)
        expected_return = 0.05
        position_size = min(current_capital, 10_000.0)
        timestamp = as_of or datetime.utcnow()
        return TradingSignal(
            symbol=symbol,
            signal_type=SignalType.BUY,
            confidence=0.9,
            predicted_price=price * (1 + expected_return),
            current_price=price,
            expected_return=expected_return,
            position_size=position_size,
            timestamp=timestamp,
            reasoning="dummy",
            stop_loss_price=price * 0.95,
            take_profit_price=price * 1.10,
            precision_87_achieved=True,
        )

    def calculate_trading_costs(
        self, position_value: float, signal_type: SignalType
    ) -> dict[str, float]:
        return {"commission": 0.0, "spread": 0.0, "slippage": 0.0, "total_cost": 0.0}


def test_backtest_runner_produces_result(monkeypatch):
    dummy_data = DummyDataProvider()
    strategy = DummyTradingStrategy(dummy_data)
    module = importlib.import_module("trading.backtest_engine")
    BacktestConfig = module.BacktestConfig
    BacktestResult = module.BacktestResult

    config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 1, 31),
        initial_capital=100_000,
        target_symbols=["AAA"],
    )

    runner = backtest_module.BacktestRunner(config, strategy, dummy_data)
    result = runner.run_backtest(["AAA"])

    assert isinstance(result, BacktestResult)
    assert result.config is config
    assert result.portfolio_values
    assert result.trade_history


def test_optimizer_selects_best_parameters():
    module = importlib.import_module("trading.backtest_engine")
    BacktestConfig = module.BacktestConfig
    BacktestResult = module.BacktestResult

    base_config = BacktestConfig(
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 2, 1),
        initial_capital=1_000_000,
        target_symbols=["AAA"],
    )

    optimizer = backtest_module.BacktestOptimizer()

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


def test_reporting_helpers(monkeypatch):
    module = importlib.import_module("trading.backtest_engine")
    monkeypatch.setattr(
        backtest_module,
        "generate_backtest_charts",
        lambda result, logger=None: {"equity_curve": "chart"},
    )
    BacktestResult = module.BacktestResult
    BacktestConfig = module.BacktestConfig

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
        total_costs=1_000.0,
        total_tax=2_000.0,
        daily_returns=[0.01, -0.02, 0.03],
        trade_history=[
            {"action": "CLOSE", "position_size": 1000.0, "profit_loss": -100.0}
        ],
        portfolio_values=[
            (config.start_date, config.initial_capital),
            (config.end_date, 1_100_000),
        ],
    )

    charts = backtest_module.generate_backtest_charts(result)
    assert charts.get("equity_curve") == "chart"

    recommendations = backtest_module.generate_recommendations(result)
    assert len(recommendations) >= 2
