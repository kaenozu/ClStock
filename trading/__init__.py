"""
ClStock デモ運用システム

1週間のデモ取引による実際の利益・損失を正確にトレースする
87%精度システムと統合されたデモ運用システム
"""

from .base_investment_system import (
    BACKTEST_PERIOD,
    BaseInvestmentSystem,
    DATA_PERIOD,
    INITIAL_CAPITAL,
    LOSS_THRESHOLD,
    POSITION_SIZE_FACTOR,
    PROFIT_THRESHOLD,
)
from .demo_trader import DemoTrader
from .backtest_engine import BacktestEngine, BacktestConfig
from .performance_tracker import PerformanceTracker
from .risk_manager import DemoRiskManager
from .trading_strategy import TradingStrategy
from .portfolio_manager import DemoPortfolioManager
from .trade_recorder import TradeRecorder

__all__ = [
    "BaseInvestmentSystem",
    "BACKTEST_PERIOD",
    "DATA_PERIOD",
    "INITIAL_CAPITAL",
    "PROFIT_THRESHOLD",
    "LOSS_THRESHOLD",
    "POSITION_SIZE_FACTOR",
    "DemoTrader",
    "BacktestEngine",
    "BacktestConfig",
    "PerformanceTracker",
    "DemoRiskManager",
    "TradingStrategy",
    "DemoPortfolioManager",
    "TradeRecorder",
]
