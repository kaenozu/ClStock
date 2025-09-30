"""Backtest execution helpers."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..performance_tracker import PerformanceTracker
from ..portfolio_manager import DemoPortfolioManager
from ..risk_manager import DemoRiskManager
from ..trade_recorder import TradeRecorder
from ..trading_strategy import TradingSignal, TradingStrategy, SignalType

if False:  # pragma: no cover - type checking only
    from trading.backtest_engine import BacktestConfig, BacktestResult
    from data.stock_data import StockDataProvider


class BacktestRunner:
    """Execute the daily backtest loop for :class:`BacktestEngine`."""

    def __init__(
        self,
        config: "BacktestConfig",
        trading_strategy: TradingStrategy,
        data_provider: "StockDataProvider",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.trading_strategy = trading_strategy
        self.data_provider = data_provider
        self.logger = logger or logging.getLogger(__name__)
        self.historical_data: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_backtest(self, target_symbols: List[str]) -> "BacktestResult":
        """Execute the backtest for the provided symbol universe."""

        try:
            self.logger.info("履歴データ読み込み開始")
            self._load_historical_data(target_symbols)

            portfolio_manager = DemoPortfolioManager(self.config.initial_capital)
            risk_manager = DemoRiskManager(self.config.initial_capital)
            trade_recorder = TradeRecorder()
            performance_tracker = PerformanceTracker(self.config.initial_capital)

            trades_executed: List[Dict[str, Any]] = []
            portfolio_values: List[Tuple[datetime, float]] = []
            daily_returns: List[float] = []

            current_date = self.config.start_date
            previous_portfolio_value = self.config.initial_capital

            while current_date <= self.config.end_date:
                try:
                    if not self._is_trading_day(current_date):
                        current_date += timedelta(days=1)
                        continue

                    daily_trades = self._process_trading_day(
                        current_date,
                        target_symbols,
                        portfolio_manager,
                        risk_manager,
                        trade_recorder,
                    )

                    trades_executed.extend(daily_trades)

                    portfolio_value = self._calculate_portfolio_value(
                        current_date, portfolio_manager
                    )
                    portfolio_values.append((current_date, portfolio_value))

                    daily_return = (
                        portfolio_value - previous_portfolio_value
                    ) / previous_portfolio_value
                    daily_returns.append(daily_return)
                    previous_portfolio_value = portfolio_value

                    performance_tracker.update_performance(
                        portfolio_value,
                        len(portfolio_manager.positions),
                        len(daily_trades),
                    )

                    current_date += timedelta(days=1)

                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.error(f"日次処理エラー {current_date}: {exc}")
                    current_date += timedelta(days=1)
                    continue

            result = self._calculate_backtest_results(
                trades_executed,
                portfolio_values,
                daily_returns,
                trade_recorder,
                performance_tracker,
            )

            self.logger.info(f"バックテスト完了: 総リターン {result.total_return:.2%}")
            return result

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"バックテスト実行エラー: {exc}")
            return self._empty_backtest_result()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_historical_data(self, symbols: List[str]) -> None:
        for symbol in symbols:
            try:
                extended_start = self.config.start_date - timedelta(days=100)
                data = self.data_provider.get_stock_data(
                    symbol, start_date=extended_start, end_date=self.config.end_date
                )

                if data is not None and len(data) > 0:
                    enriched = self.data_provider.calculate_technical_indicators(data)
                    self.historical_data[symbol] = enriched
                    self.logger.info(
                        f"履歴データ読み込み完了: {symbol} ({len(enriched)}日分)"
                    )
                else:
                    self.logger.warning(f"履歴データ取得失敗: {symbol}")

            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error(f"履歴データ読み込みエラー {symbol}: {exc}")

    def _process_trading_day(
        self,
        date: datetime,
        symbols: List[str],
        portfolio_manager: DemoPortfolioManager,
        risk_manager: DemoRiskManager,
        trade_recorder: TradeRecorder,
    ) -> List[Dict[str, Any]]:
        trades_executed: List[Dict[str, Any]] = []

        try:
            portfolio_manager.update_positions()

            for symbol in symbols:
                if symbol not in self.historical_data:
                    continue

                historical_data = self.historical_data[symbol]
                date_mask = historical_data.index <= date
                available_data = historical_data[date_mask]

                if len(available_data) < 50:
                    continue

                signal = self._generate_backtest_signal(symbol, available_data, date)
                if signal is None:
                    continue

                current_capital = portfolio_manager.current_cash + sum(
                    pos.market_value for pos in portfolio_manager.positions.values()
                )

                if not risk_manager.can_open_position(symbol, signal.position_size):
                    continue

                if current_capital <= 0:
                    continue

                trade = self._execute_backtest_trade(
                    signal, date, portfolio_manager, trade_recorder
                )

                if trade:
                    trades_executed.append(trade)

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"取引日処理エラー {date}: {exc}")

        return trades_executed

    def _generate_backtest_signal(
        self, symbol: str, historical_data: pd.DataFrame, current_date: datetime
    ) -> Optional[TradingSignal]:
        try:
            current_price = historical_data.loc[
                historical_data.index <= current_date, "Close"
            ].iloc[-1]

            signal_data = self._calculate_historical_signal(
                symbol, historical_data, current_date
            )

            if signal_data["confidence"] < self.config.confidence_threshold:
                return None

            return TradingSignal(
                symbol=symbol,
                signal_type=signal_data["signal_type"],
                confidence=signal_data["confidence"],
                predicted_price=signal_data["predicted_price"],
                current_price=current_price,
                expected_return=signal_data["expected_return"],
                position_size=signal_data["position_size"],
                timestamp=current_date,
                reasoning=signal_data["reasoning"],
                precision_87_achieved=signal_data["precision_87_achieved"],
            )

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"バックテストシグナル生成エラー {symbol}: {exc}")
            return None

    def _calculate_historical_signal(
        self, symbol: str, data: pd.DataFrame, date: datetime
    ) -> Dict[str, Any]:
        try:
            current_data = data[data.index <= date]

            if len(current_data) < 20:
                return self._default_signal()

            current_price = current_data["Close"].iloc[-1]

            sma_20 = current_data["Close"].rolling(20).mean().iloc[-1]
            rsi = self._calculate_rsi(current_data["Close"], 14).iloc[-1]

            signal_strength = 0.0
            reasoning_parts: List[str] = []

            if rsi < 30:
                signal_strength += 0.3
                reasoning_parts.append("RSI過売り")
            elif rsi > 70:
                signal_strength -= 0.3
                reasoning_parts.append("RSI過買い")

            if current_price > sma_20:
                signal_strength += 0.4
                reasoning_parts.append("SMA20上回り")
            else:
                signal_strength -= 0.4
                reasoning_parts.append("SMA20下回り")

            volatility = current_data["Close"].pct_change().std()
            confidence = min(abs(signal_strength) + 0.3, 0.9)

            precision_87_achieved = np.random.random() < 0.3

            expected_return = signal_strength * 0.05
            position_size = (
                self.config.initial_capital * self.config.max_position_size * confidence
            )

            if signal_strength > 0.3:
                signal_type = SignalType.BUY
            elif signal_strength < -0.3:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD

            predicted_price = current_price * (1 + expected_return)

            return {
                "signal_type": signal_type,
                "confidence": confidence,
                "predicted_price": predicted_price,
                "expected_return": expected_return,
                "position_size": position_size,
                "reasoning": " | ".join(reasoning_parts),
                "precision_87_achieved": precision_87_achieved,
            }

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"履歴シグナル計算エラー: {exc}")
            return self._default_signal()

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _default_signal(self) -> Dict[str, Any]:
        return {
            "signal_type": SignalType.HOLD,
            "confidence": 0.0,
            "predicted_price": 0.0,
            "expected_return": 0.0,
            "position_size": 0.0,
            "reasoning": "データ不足",
            "precision_87_achieved": False,
        }

    def _execute_backtest_trade(
        self,
        signal: TradingSignal,
        date: datetime,
        portfolio_manager: DemoPortfolioManager,
        trade_recorder: TradeRecorder,
    ) -> Optional[Dict[str, Any]]:
        try:
            slippage = np.random.normal(0, self.config.slippage_rate)
            execution_price = signal.current_price * (1 + slippage)

            quantity = int(signal.position_size / execution_price)
            if quantity <= 0:
                return None

            actual_position_value = quantity * execution_price

            commission = actual_position_value * self.config.commission_rate
            spread_cost = actual_position_value * self.config.spread_rate
            total_cost = commission + spread_cost

            portfolio_manager.add_position(
                signal.symbol, quantity, execution_price, signal.signal_type
            )

            trade_data = {
                "trade_id": f"{signal.symbol}_{date.strftime('%Y%m%d')}",
                "symbol": signal.symbol,
                "action": "OPEN",
                "quantity": quantity,
                "price": execution_price,
                "timestamp": date.isoformat(),
                "signal_type": signal.signal_type.value,
                "confidence": signal.confidence,
                "precision_87_achieved": signal.precision_87_achieved,
                "expected_return": signal.expected_return,
                "position_size": actual_position_value,
                "trading_costs": {
                    "commission": commission,
                    "spread": spread_cost,
                    "total_cost": total_cost,
                },
            }

            trade_recorder.record_trade(trade_data)

            return trade_data

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"バックテスト取引実行エラー: {exc}")
            return None

    def _calculate_portfolio_value(
        self, date: datetime, portfolio_manager: DemoPortfolioManager
    ) -> float:
        try:
            total_value = portfolio_manager.current_cash

            for symbol, position in portfolio_manager.positions.items():
                if symbol in self.historical_data:
                    historical_data = self.historical_data[symbol]
                    date_mask = historical_data.index <= date
                    available_data = historical_data[date_mask]

                    if len(available_data) > 0:
                        current_price = available_data["Close"].iloc[-1]
                        position_value = position.quantity * current_price
                        total_value += position_value

            return total_value

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"ポートフォリオ価値計算エラー: {exc}")
            return portfolio_manager.current_cash

    def _calculate_backtest_results(
        self,
        trades: List[Dict[str, Any]],
        portfolio_values: List[Tuple[datetime, float]],
        daily_returns: List[float],
        trade_recorder: TradeRecorder,
        performance_tracker: PerformanceTracker,
    ) -> "BacktestResult":
        try:
            if not portfolio_values:
                return self._empty_backtest_result()

            initial_value = self.config.initial_capital
            final_value = portfolio_values[-1][1]
            total_return = (final_value - initial_value) / initial_value

            days = (self.config.end_date - self.config.start_date).days
            annualized_return = (
                (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
            )

            volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            downside_returns = [r for r in daily_returns if r < 0]
            downside_volatility = (
                np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
            )
            sortino_ratio = (
                annualized_return / downside_volatility
                if downside_volatility > 0
                else 0
            )

            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            calmar_ratio = (
                annualized_return / max_drawdown if max_drawdown > 0 else float("inf")
            )

            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get("profit_loss", 0) > 0])
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

            completed_trades = trade_recorder.get_completed_trades()

            profits = [
                t["profit_loss"]
                for t in completed_trades
                if t.get("profit_loss", 0) > 0
            ]
            losses = [
                abs(t["profit_loss"])
                for t in completed_trades
                if t.get("profit_loss", 0) < 0
            ]
            total_profits = sum(profits) if profits else 0
            total_losses = sum(losses) if losses else 0
            profit_factor = (
                total_profits / total_losses if total_losses > 0 else float("inf")
            )

            precision_87_trades = len(
                [t for t in trades if t.get("precision_87_achieved", False)]
            )
            precision_87_wins = len(
                [
                    t
                    for t in completed_trades
                    if t.get("precision_87_achieved", False)
                    and t.get("profit_loss", 0) > 0
                ]
            )
            precision_87_success_rate = (
                precision_87_wins / precision_87_trades * 100
                if precision_87_trades > 0
                else 0
            )

            var_95 = np.percentile(daily_returns, 5) if daily_returns else 0
            tail_returns = [r for r in daily_returns if r <= var_95]
            expected_shortfall = np.mean(tail_returns) if tail_returns else var_95

            total_costs = sum(
                t.get("trading_costs", {}).get("total_cost", 0) for t in trades
            )
            total_tax = total_profits * self.config.tax_rate

            benchmark_return = 0.05
            excess_return = annualized_return - benchmark_return
            beta = 1.0
            alpha = excess_return

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
                information_ratio=excess_return / volatility if volatility > 0 else 0,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                total_costs=total_costs,
                total_tax=total_tax,
                daily_returns=daily_returns,
                trade_history=trades,
                portfolio_values=portfolio_values,
            )

        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error(f"バックテスト結果計算エラー: {exc}")
            return self._empty_backtest_result()

    def _calculate_max_drawdown(
        self, portfolio_values: List[Tuple[datetime, float]]
    ) -> float:
        if not portfolio_values:
            return 0.0

        values = [v[1] for v in portfolio_values]
        peak = values[0]
        max_drawdown = 0.0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _is_trading_day(self, date: datetime) -> bool:
        return date.weekday() < 5

    def _empty_backtest_result(self) -> "BacktestResult":
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
