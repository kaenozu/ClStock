"""Common investment system infrastructure for live trading modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import pandas as pd

from data.stock_data import StockDataProvider
from utils.logger_config import setup_logger

__all__ = [
    "INITIAL_CAPITAL",
    "PROFIT_THRESHOLD",
    "LOSS_THRESHOLD",
    "POSITION_SIZE_FACTOR",
    "BACKTEST_PERIOD",
    "DATA_PERIOD",
    "BaseInvestmentSystem",
]


logger = setup_logger(__name__)

# Shared investment constants
INITIAL_CAPITAL = 1_000_000
PROFIT_THRESHOLD = 15.0
LOSS_THRESHOLD = -5.0
POSITION_SIZE_FACTOR = 0.15

# Backtest configuration
BACKTEST_PERIOD = "2y"
DATA_PERIOD = "1y"


@dataclass
class TradeRecord:
    action: str
    symbol: str
    shares: int
    price: float
    date: datetime
    profit: float | None = None


class BaseInvestmentSystem:
    """基底となる投資システムの共通ロジック。"""

    def __init__(self, initial_capital: int = INITIAL_CAPITAL):
        self.data_provider = StockDataProvider()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict[str, float]] = {}
        self.transaction_history: list[TradeRecord] = []

        # Threshold configuration
        self.min_profit_threshold = PROFIT_THRESHOLD
        self.max_loss_threshold = LOSS_THRESHOLD
        self.position_size_factor = POSITION_SIZE_FACTOR

    def _get_stock_data(self, symbol: str, period: str = DATA_PERIOD) -> pd.DataFrame:
        """株価データを取得する。"""
        try:
            return self.data_provider.get_stock_data(symbol, period)
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.warning("データ取得エラー %s: %s", symbol, exc)
            return pd.DataFrame()

    def _calculate_position_size(self, capital: float, confidence: float) -> float:
        """信頼度に基づいてポジションサイズを計算する。"""
        base_size = capital * self.position_size_factor
        confidence_multiplier = min(confidence / 100.0, 1.0)
        return base_size * confidence_multiplier

    def _execute_trade_order(
        self, symbol: str, action: str, shares: int, price: float, date: datetime
    ) -> None:
        """売買注文を実行する共通メソッド。"""
        if action == "BUY":
            self._execute_buy(symbol, shares, price, date)
        elif action == "SELL":
            self._execute_sell(symbol, shares, price, date)

    def _execute_buy(self, symbol: str, shares: int, price: float, date: datetime) -> None:
        cost = shares * price
        if self.current_capital >= cost:
            self.positions[symbol] = {
                "shares": shares,
                "buy_price": price,
                "buy_date": date,
            }
            self.current_capital -= cost
            self._record_transaction("BUY", symbol, shares, price, date)

    def _execute_sell(
        self, symbol: str, shares: int, price: float, date: datetime
    ) -> None:
        if symbol in self.positions:
            position = self.positions[symbol]
            profit = (price - position["buy_price"]) * shares

            self.current_capital += shares * price
            del self.positions[symbol]

            self._record_transaction("SELL", symbol, shares, price, date, profit)

    def _record_transaction(
        self,
        action: str,
        symbol: str,
        shares: int,
        price: float,
        date: datetime,
        profit: float | None = None,
    ) -> None:
        record = TradeRecord(action, symbol, shares, price, date, profit)
        self.transaction_history.append(record)

    def _should_sell_position(self, symbol: str, current_price: float) -> bool:
        if symbol not in self.positions:
            return False

        position = self.positions[symbol]
        change_rate = ((current_price - position["buy_price"]) / position["buy_price"]) * 100

        return (
            change_rate >= self.min_profit_threshold
            or change_rate <= self.max_loss_threshold
        )

    def _calculate_final_results(self) -> Dict[str, float]:
        for symbol in list(self.positions.keys()):
            try:
                current_data = self._get_stock_data(symbol, "1d")
                if not current_data.empty:
                    current_price = current_data["Close"].iloc[-1]
                    shares = self.positions[symbol]["shares"]
                    self.current_capital += shares * current_price
                    del self.positions[symbol]
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Failed to update position for %s: %s", symbol, exc)

        total_return = self.current_capital - self.initial_capital
        return_rate = (total_return / self.initial_capital) * 100

        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.current_capital,
            "total_return": total_return,
            "return_rate": return_rate,
            "total_trades": len(self.transaction_history),
        }
