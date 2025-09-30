from __future__ import annotations

"""共通データモデル定義"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .trading_strategy import SignalType


@dataclass
class TradeRecord:
    """取引記録"""

    trade_id: str
    session_id: str
    symbol: str
    action: str  # OPEN, CLOSE, PARTIAL_CLOSE
    quantity: int
    price: float
    timestamp: datetime
    signal_type: SignalType
    confidence: float
    precision: float
    precision_87_achieved: bool
    expected_return: float
    actual_return: Optional[float]
    profit_loss: Optional[float]
    trading_costs: Dict[str, float]
    position_size: float
    market_value: float
    reasoning: str
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    execution_details: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    total_return_pct: float
    average_return: float
    average_win: float
    average_loss: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    precision_87_trades: int
    precision_87_success_rate: float
    best_trade: float
    worst_trade: float
    average_holding_period: float


@dataclass
class TaxCalculation:
    """税務計算結果"""

    total_realized_gains: float
    total_realized_losses: float
    net_capital_gains: float
    short_term_gains: float
    long_term_gains: float
    wash_sales: float
    estimated_tax_liability: float
    deductible_expenses: float
