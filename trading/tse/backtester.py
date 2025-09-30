"""Backtesting utilities for the TSE 4000 optimizer."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

INITIAL_CAPITAL = 1_000_000
POSITION_SIZE_PERCENTAGE = 0.05
SHORT_MA_PERIOD = 10
LONG_MA_PERIOD = 30


class PortfolioBacktester:
    """Executes the trading strategy to evaluate a portfolio."""

    def __init__(self, data_provider):
        self.data_provider = data_provider

    def backtest_portfolio(self, selected_symbols: List[str]) -> Dict:
        print(f"\nバックテスト実行中（{len(selected_symbols)}銘柄）...")

        portfolio_state = self._initialize_portfolio_state()

        for symbol in selected_symbols:
            try:
                self._backtest_single_symbol(symbol, portfolio_state)
            except Exception as exc:  # pragma: no cover - defensive logging
                import logging

                logging.warning(f"バックテスト失敗 {symbol}: {exc}")

        self._evaluate_remaining_positions(portfolio_state)
        return self._calculate_backtest_results(portfolio_state)

    def _initialize_portfolio_state(self) -> Dict:
        return {
            "current_capital": INITIAL_CAPITAL,
            "positions": {},
            "transaction_history": [],
        }

    def _backtest_single_symbol(self, symbol: str, portfolio_state: Dict):
        stock_data = self.data_provider.get_stock_data(symbol, "1y")
        if stock_data.empty:
            return

        self._execute_trading_strategy(symbol, stock_data, portfolio_state)

    def _execute_trading_strategy(
        self, symbol: str, stock_data: pd.DataFrame, portfolio_state: Dict
    ):
        close = stock_data["Close"]
        ma_short = close.rolling(SHORT_MA_PERIOD).mean()
        ma_long = close.rolling(LONG_MA_PERIOD).mean()

        position_size = portfolio_state["current_capital"] * POSITION_SIZE_PERCENTAGE

        for i in range(LONG_MA_PERIOD, len(close) - 1):
            current_price = close.iloc[i]

            if self._is_buy_signal(
                ma_short, ma_long, i, symbol, portfolio_state["positions"]
            ):
                self._execute_buy_order(
                    symbol,
                    current_price,
                    position_size,
                    stock_data.index[i],
                    portfolio_state,
                )
            elif self._is_sell_signal(
                ma_short, ma_long, i, symbol, portfolio_state["positions"]
            ):
                self._execute_sell_order(
                    symbol, current_price, stock_data.index[i], portfolio_state
                )

    def _is_buy_signal(
        self,
        ma_short: pd.Series,
        ma_long: pd.Series,
        index: int,
        symbol: str,
        positions: Dict,
    ) -> bool:
        return (
            ma_short.iloc[index] > ma_long.iloc[index]
            and ma_short.iloc[index - 1] <= ma_long.iloc[index - 1]
            and symbol not in positions
        )

    def _is_sell_signal(
        self,
        ma_short: pd.Series,
        ma_long: pd.Series,
        index: int,
        symbol: str,
        positions: Dict,
    ) -> bool:
        return (
            ma_short.iloc[index] < ma_long.iloc[index]
            and ma_short.iloc[index - 1] >= ma_long.iloc[index - 1]
            and symbol in positions
        )

    def _execute_buy_order(
        self,
        symbol: str,
        price: float,
        position_size: float,
        date,
        portfolio_state: Dict,
    ):
        shares = int(position_size / price)
        if shares <= 0:
            return

        cost = shares * price
        if portfolio_state["current_capital"] < cost:
            return

        portfolio_state["current_capital"] -= cost
        portfolio_state["positions"][symbol] = {
            "shares": shares,
            "entry_price": price,
            "entry_date": date,
        }
        self._record_transaction(
            portfolio_state,
            symbol,
            "buy",
            date,
            price,
            shares,
        )

    def _execute_sell_order(self, symbol: str, price: float, date, portfolio_state: Dict):
        position = portfolio_state["positions"].get(symbol)
        if not position:
            return

        shares = position["shares"]
        proceeds = shares * price
        portfolio_state["current_capital"] += proceeds
        self._record_transaction(
            portfolio_state,
            symbol,
            "sell",
            date,
            price,
            shares,
        )
        self._close_position(symbol, portfolio_state)

    def _close_position(self, symbol: str, portfolio_state: Dict):
        portfolio_state["positions"].pop(symbol, None)

    def _record_transaction(
        self,
        portfolio_state: Dict,
        symbol: str,
        action: str,
        date,
        price: float,
        shares: int,
    ):
        portfolio_state["transaction_history"].append(
            {
                "symbol": symbol,
                "action": action,
                "date": str(date),
                "price": price,
                "shares": shares,
            }
        )

    def _evaluate_remaining_positions(self, portfolio_state: Dict):
        for symbol, position in list(portfolio_state["positions"].items()):
            try:
                current_data = self.data_provider.get_stock_data(symbol, "1d")
            except Exception:  # pragma: no cover - defensive logging
                continue
            if current_data.empty:
                continue

            current_price = current_data["Close"].iloc[-1]
            portfolio_state["current_capital"] += position["shares"] * current_price
            self._close_position(symbol, portfolio_state)

    def _calculate_backtest_results(self, portfolio_state: Dict) -> Dict:
        total_return = portfolio_state["current_capital"] - INITIAL_CAPITAL
        return_rate = (total_return / INITIAL_CAPITAL) * 100

        return {
            "initial_capital": INITIAL_CAPITAL,
            "final_capital": portfolio_state["current_capital"],
            "total_return": total_return,
            "return_rate": return_rate,
            "total_trades": len(portfolio_state["transaction_history"]),
            "transaction_history": portfolio_state["transaction_history"],
        }
