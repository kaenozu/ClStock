"""Backtest execution helpers with cash-aware accounting."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..trading_strategy import SignalType, TradingSignal, TradingStrategy

if False:  # pragma: no cover - type checking only
    from data.stock_data import StockDataProvider
    from trading.backtest_engine import BacktestConfig, BacktestResult


@dataclass
class PositionRecord:
    """State for an open position inside the backtest."""

    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime
    entry_cost: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    direction: SignalType
    confidence: float
    expected_return: float
    precision_flag: bool


class BacktestRunner:
    """Execute the daily backtest loop for :class:`BacktestEngine`."""

    def __init__(
        self,
        config: BacktestConfig,
        trading_strategy: TradingStrategy,
        data_provider: StockDataProvider,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.trading_strategy = trading_strategy
        self.data_provider = data_provider
        self.logger = logger or logging.getLogger(__name__)
        self.historical_data: Dict[str, pd.DataFrame] = {}

        self.cash: float = 0.0
        self.positions: Dict[str, Optional[PositionRecord]] = {}
        self.trade_log: List[Dict[str, Any]] = []
        self.completed_trades: List[Dict[str, Any]] = []
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []

        self.holding_horizon: int = getattr(
            self.trading_strategy.precision_system,
            "evaluation_horizon",
            5,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_backtest(self, target_symbols: List[str]) -> BacktestResult:
        """Execute the backtest for the provided symbol universe."""
        try:
            self.logger.info("履歴データ読み込み開始")
            self._load_historical_data(target_symbols)

            self.cash = self.config.initial_capital
            self.positions = {symbol: None for symbol in target_symbols}
            self.trade_log = []
            self.completed_trades = []
            self.portfolio_values = []
            self.daily_returns = []

            previous_portfolio_value = self.cash
            current_date = self.config.start_date

            while current_date <= self.config.end_date:
                if not self._is_trading_day(current_date):
                    current_date += timedelta(days=1)
                    continue

                price_cache = self._build_price_cache(target_symbols, current_date)
                if not price_cache:
                    current_date += timedelta(days=1)
                    continue

                # Step 1: evaluate forced exits (stop-loss, take-profit, horizon)
                for symbol, position in list(self.positions.items()):
                    if position is None:
                        continue
                    price = price_cache.get(symbol)
                    if price is None:
                        continue
                    reason = self._should_force_close(position, price, current_date)
                    if reason:
                        self._close_position(symbol, price, current_date, reason)

                # Step 2: process model-driven signals per symbol
                for symbol in target_symbols:
                    price = price_cache.get(symbol)
                    if price is None:
                        continue

                    # Use available cash for sizing; prevents leverage until risk layer is rebuilt.
                    current_capital = max(self.cash, 0.0)
                    signal = self._generate_signal(
                        symbol, current_capital, current_date
                    )

                    if signal is not None:
                        self._maybe_close_on_signal(symbol, price, current_date, signal)
                        self._maybe_open_position(symbol, price, current_date, signal)

                # Step 3: record daily portfolio value
                portfolio_value = self._current_portfolio_value(price_cache)
                self.portfolio_values.append((current_date, portfolio_value))

                if previous_portfolio_value > 0:
                    daily_return = (
                        portfolio_value - previous_portfolio_value
                    ) / previous_portfolio_value
                else:
                    daily_return = 0.0
                self.daily_returns.append(daily_return)
                previous_portfolio_value = portfolio_value

                current_date += timedelta(days=1)

            # Liquidate any remaining positions at the final available price
            self._final_liquidation(target_symbols, self.config.end_date)

            result = self._calculate_backtest_results()
            self.logger.info(
                "バックテスト完了 %s → %s, リターン %.2f%%",
                self.config.start_date.date(),
                self.config.end_date.date(),
                result.total_return * 100,
            )
            return result

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"バックテスト実行エラー: {exc}")
            return self._empty_backtest_result()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_historical_data(self, symbols: List[str]) -> None:
        lookback_days = getattr(
            self.trading_strategy.precision_system,
            "evaluation_window",
            160,
        )
        buffer_days = max(lookback_days * 2, 360)
        extended_start = self.config.start_date - timedelta(days=buffer_days)

        for symbol in symbols:
            try:
                data = self.data_provider.get_stock_data(
                    symbol,
                    start_date=extended_start,
                    end_date=self.config.end_date,
                )
                if data is None or data.empty:
                    self.logger.warning("履歴データが取得できませんでした: %s", symbol)
                    continue
                enriched = self.data_provider.calculate_technical_indicators(data)
                self.historical_data[symbol] = enriched
                self.logger.info("%s: %d 本の足を読み込み", symbol, len(enriched))
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error("履歴データ取得エラー %s: %s", symbol, exc)

    def _build_price_cache(
        self, symbols: List[str], date: datetime
    ) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        for symbol in symbols:
            price = self._get_price(symbol, date)
            if price is not None:
                prices[symbol] = price
        return prices

    def _get_price(self, symbol: str, date: datetime) -> Optional[float]:
        frame = self.historical_data.get(symbol)
        if frame is None:
            return None
        if date not in frame.index:
            return None
        price = frame.loc[date, "Close"]
        if isinstance(price, pd.Series):
            price = price.iloc[-1]
        return float(price)

    def _current_portfolio_value(self, price_cache: Dict[str, float]) -> float:
        value = self.cash
        for symbol, position in self.positions.items():
            if position is None:
                continue
            price = price_cache.get(symbol)
            if price is None:
                continue
            value += position.quantity * price
        return float(value)

    def _generate_signal(
        self,
        symbol: str,
        current_capital: float,
        as_of: datetime,
    ) -> Optional[TradingSignal]:
        try:
            return self.trading_strategy.generate_trading_signal(
                symbol,
                current_capital,
                as_of=as_of,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("シグナル生成エラー %s: %s", symbol, exc)
            return None

    def _should_force_close(
        self,
        position: PositionRecord,
        price: float,
        as_of: datetime,
    ) -> Optional[str]:
        if position.direction != SignalType.BUY:
            return None
        if position.stop_loss is not None and price <= position.stop_loss:
            return "stop_loss"
        if position.take_profit is not None and price >= position.take_profit:
            return "take_profit"
        holding_days = (as_of - position.entry_date).days
        if holding_days >= self.holding_horizon:
            return "time_exit"
        return None

    def _maybe_close_on_signal(
        self,
        symbol: str,
        price: float,
        as_of: datetime,
        signal: TradingSignal,
    ) -> None:
        position = self.positions.get(symbol)
        if position is None:
            return
        if signal.signal_type in {
            SignalType.SELL,
            SignalType.STOP_LOSS,
            SignalType.TAKE_PROFIT,
        }:
            self._close_position(symbol, price, as_of, "signal_reverse")
        elif signal.expected_return <= 0:
            self._close_position(symbol, price, as_of, "negative_expectation")

    def _maybe_open_position(
        self,
        symbol: str,
        price: float,
        as_of: datetime,
        signal: TradingSignal,
    ) -> None:
        if signal.signal_type != SignalType.BUY:
            return
        if self.positions.get(symbol) is not None:
            return

        # Determine affordable quantity based on available cash after costs.
        approximate_cost_rate = (
            self.trading_strategy.commission_rate
            + self.trading_strategy.spread_rate
            + self.trading_strategy.slippage_rate
        )
        max_affordable_value = self.cash / (1.0 + approximate_cost_rate)
        target_value = min(signal.position_size, max_affordable_value)
        quantity = int(target_value / price)
        if quantity <= 0:
            return

        position_value = quantity * price
        trading_costs = self.trading_strategy.calculate_trading_costs(
            position_value,
            SignalType.BUY,
        )
        total_required = position_value + trading_costs["total_cost"]
        if total_required > self.cash:
            quantity = int(self.cash / (price * (1.0 + approximate_cost_rate)))
            if quantity <= 0:
                return
            position_value = quantity * price
            trading_costs = self.trading_strategy.calculate_trading_costs(
                position_value,
                SignalType.BUY,
            )
            total_required = position_value + trading_costs["total_cost"]
            if total_required > self.cash:
                return

        self.cash -= total_required
        position_record = PositionRecord(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_date=as_of,
            entry_cost=total_required,
            stop_loss=signal.stop_loss_price,
            take_profit=signal.take_profit_price,
            direction=signal.signal_type,
            confidence=signal.confidence,
            expected_return=signal.expected_return,
            precision_flag=signal.precision_87_achieved,
        )
        self.positions[symbol] = position_record

        trade_entry = {
            "trade_id": f"{symbol}_{as_of.strftime('%Y%m%d')}_OPEN",
            "symbol": symbol,
            "action": "OPEN",
            "quantity": quantity,
            "price": price,
            "timestamp": as_of.isoformat(),
            "confidence": signal.confidence,
            "expected_return": signal.expected_return,
            "precision_87_achieved": signal.precision_87_achieved,
            "position_size": position_value,
            "trading_costs": trading_costs,
            "stop_loss_price": signal.stop_loss_price,
            "take_profit_price": signal.take_profit_price,
            "reason": "model_signal",
        }
        self.trade_log.append(trade_entry)

    def _close_position(
        self,
        symbol: str,
        price: float,
        as_of: datetime,
        reason: str,
    ) -> None:
        position = self.positions.get(symbol)
        if position is None:
            return

        quantity = position.quantity
        gross_proceeds = quantity * price
        trading_costs = self.trading_strategy.calculate_trading_costs(
            gross_proceeds,
            SignalType.SELL,
        )
        net_proceeds = gross_proceeds - trading_costs["total_cost"]
        profit_loss = net_proceeds - position.entry_cost
        holding_days = max((as_of - position.entry_date).days, 1)

        self.cash += net_proceeds
        self.positions[symbol] = None

        trade_exit = {
            "trade_id": f"{symbol}_{as_of.strftime('%Y%m%d')}_CLOSE",
            "symbol": symbol,
            "action": "CLOSE",
            "quantity": quantity,
            "price": price,
            "timestamp": as_of.isoformat(),
            "profit_loss": profit_loss,
            "holding_days": holding_days,
            "precision_87_achieved": position.precision_flag,
            "trading_costs": trading_costs,
            "reason": reason,
        }
        self.trade_log.append(trade_exit)

        self.completed_trades.append(
            {
                "symbol": symbol,
                "entry_date": position.entry_date,
                "exit_date": as_of,
                "quantity": quantity,
                "entry_price": position.entry_price,
                "exit_price": price,
                "profit_loss": profit_loss,
                "precision_87_achieved": position.precision_flag,
                "holding_days": holding_days,
            }
        )

    def _final_liquidation(self, symbols: List[str], as_of: datetime) -> None:
        price_cache = self._build_price_cache(symbols, as_of)
        latest_time = as_of
        positions_closed = False
        for symbol, position in list(self.positions.items()):
            if position is None:
                continue
            price = price_cache.get(symbol)
            effective_time = as_of
            if price is None:
                history = self.historical_data.get(symbol)
                if history is None or history.empty:
                    continue
                price = float(history["Close"].iloc[-1])
                effective_time = history.index[-1].to_pydatetime()
            latest_time = max(latest_time, effective_time)
            self._close_position(symbol, price, effective_time, "final_liquidation")
            positions_closed = True
        if positions_closed:
            final_value = self.cash
            self.portfolio_values.append((latest_time, final_value))
            prior_value = (
                self.portfolio_values[-2][1]
                if len(self.portfolio_values) > 1
                else self.config.initial_capital
            )
            daily_return = (
                (final_value - prior_value) / prior_value if prior_value > 0 else 0.0
            )
            self.daily_returns.append(daily_return)

    # ------------------------------------------------------------------
    # Result aggregation
    # ------------------------------------------------------------------
    def _calculate_backtest_results(self) -> BacktestResult:
        try:
            if not self.portfolio_values:
                return self._empty_backtest_result()

            initial_value = self.config.initial_capital
            final_value = self.portfolio_values[-1][1]
            total_return = (
                (final_value - initial_value) / initial_value
                if initial_value > 0
                else 0.0
            )

            days = (self.config.end_date - self.config.start_date).days or 1
            annualized_return = (
                (1 + total_return) ** (252 / days) - 1 if total_return != -1 else -1
            )

            daily_returns = self.daily_returns
            volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0.0
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0

            downside_returns = [r for r in daily_returns if r < 0]
            downside_volatility = (
                np.std(downside_returns) * np.sqrt(252) if downside_returns else 0.0
            )
            sortino_ratio = (
                annualized_return / downside_volatility
                if downside_volatility > 0
                else 0.0
            )

            max_drawdown = self._calculate_max_drawdown(self.portfolio_values)
            calmar_ratio = (
                annualized_return / max_drawdown if max_drawdown > 0 else float("inf")
            )

            total_trades = len(self.completed_trades)
            winning_trades = len(
                [t for t in self.completed_trades if t["profit_loss"] > 0]
            )
            win_rate = (
                (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            )

            profits = [
                t["profit_loss"] for t in self.completed_trades if t["profit_loss"] > 0
            ]
            losses = [
                -t["profit_loss"] for t in self.completed_trades if t["profit_loss"] < 0
            ]
            total_profits = sum(profits)
            total_losses = sum(losses)
            profit_factor = (
                (total_profits / total_losses) if total_losses > 0 else float("inf")
            )

            precision_trades = [
                t for t in self.completed_trades if t["precision_87_achieved"]
            ]
            precision_87_trades = len(precision_trades)
            precision_87_success_rate = (
                len([t for t in precision_trades if t["profit_loss"] > 0])
                / precision_87_trades
                * 100
                if precision_87_trades > 0
                else 0.0
            )

            var_95 = np.percentile(daily_returns, 5) if daily_returns else 0.0
            tail_returns = [r for r in daily_returns if r <= var_95]
            expected_shortfall = (
                float(np.mean(tail_returns)) if tail_returns else float(var_95)
            )

            total_costs = sum(
                trade.get("trading_costs", {}).get("total_cost", 0.0)
                for trade in self.trade_log
            )
            total_tax = total_profits * self.config.tax_rate

            benchmark_return = 0.05
            excess_return = annualized_return - benchmark_return
            beta = 1.0
            alpha = excess_return
            information_ratio = excess_return / volatility if volatility > 0 else 0.0

            from trading.backtest_engine import BacktestResult  # local import

            return BacktestResult(
                config=self.config,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                total_trades=total_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                precision_87_trades=precision_87_trades,
                precision_87_success_rate=precision_87_success_rate,
                final_value=final_value,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                total_costs=total_costs,
                total_tax=total_tax,
                daily_returns=daily_returns,
                trade_history=self.trade_log,
                portfolio_values=self.portfolio_values,
            )

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"リザルト集計エラー: {exc}")
            return self._empty_backtest_result()

    def _calculate_max_drawdown(
        self,
        portfolio_values: List[Tuple[datetime, float]],
    ) -> float:
        if not portfolio_values:
            return 0.0
        values = [v[1] for v in portfolio_values]
        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            peak = max(peak, value)
            if peak <= 0:
                continue
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        return float(max_drawdown)

    def _is_trading_day(self, date: datetime) -> bool:
        return date.weekday() < 5

    def _empty_backtest_result(self) -> BacktestResult:
        from trading.backtest_engine import BacktestResult  # local import

        return BacktestResult(
            config=self.config,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            precision_87_trades=0,
            precision_87_success_rate=0.0,
            final_value=self.config.initial_capital,
            benchmark_return=0.0,
            excess_return=0.0,
            beta=0.0,
            alpha=0.0,
            information_ratio=0.0,
            var_95=0.0,
            expected_shortfall=0.0,
            total_costs=0.0,
            total_tax=0.0,
            daily_returns=[],
            trade_history=[],
            portfolio_values=[],
        )
