from __future__ import annotations

"""パフォーマンス指標計算サービス"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trading.models import PerformanceMetrics, TradeRecord


class PerformanceMetricsService:
    """取引パフォーマンス指標の集計"""

    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital

    def generate_report(
        self,
        trades: List[TradeRecord],
    ) -> PerformanceMetrics:
        if not trades:
            return self._empty_metrics()

        completed_trades = [
            trade for trade in trades if trade.action == "CLOSE" and trade.profit_loss is not None
        ]

        if not completed_trades:
            return self._empty_metrics()

        total_trades = len(completed_trades)
        profits = [trade.profit_loss for trade in completed_trades]
        returns = [
            trade.actual_return for trade in completed_trades if trade.actual_return is not None
        ]

        winning_trades = len([p for p in profits if p > 0])
        losing_trades = len([p for p in profits if p < 0])
        win_rate = winning_trades / total_trades * 100 if total_trades else 0.0

        total_return = float(sum(profits))
        average_return = float(np.mean(returns)) if returns else 0.0

        winning_profits = [p for p in profits if p > 0]
        losing_profits = [p for p in profits if p < 0]

        average_win = float(np.mean(winning_profits)) if winning_profits else 0.0
        average_loss = abs(float(np.mean(losing_profits))) if losing_profits else 0.0

        total_wins = float(sum(winning_profits)) if winning_profits else 0.0
        total_losses = abs(float(sum(losing_profits))) if losing_profits else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        sharpe_ratio = 0.0
        if returns and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

        negative_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(negative_returns) if negative_returns else 0.0
        sortino_ratio = (
            (np.mean(returns) / downside_deviation) * np.sqrt(252)
            if downside_deviation > 0
            else 0.0
        )

        max_drawdown = self._calculate_max_drawdown(completed_trades)
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_streaks(
            profits
        )

        precision_87_trades = len([trade for trade in completed_trades if trade.precision_87_achieved])
        precision_87_success_rate = 0.0
        if precision_87_trades:
            precision_87_wins = len(
                [trade for trade in completed_trades if trade.precision_87_achieved and trade.profit_loss > 0]
            )
            precision_87_success_rate = precision_87_wins / precision_87_trades * 100

        best_trade = max(profits) if profits else 0.0
        worst_trade = min(profits) if profits else 0.0

        average_holding_period = self._calculate_average_holding_period(completed_trades)
        total_return_pct = total_return / self.initial_capital * 100 if self.initial_capital else 0.0

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            total_return_pct=total_return_pct,
            average_return=average_return,
            average_win=average_win,
            average_loss=average_loss,
            profit_factor=profit_factor,
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            max_drawdown=max_drawdown,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            precision_87_trades=precision_87_trades,
            precision_87_success_rate=precision_87_success_rate,
            best_trade=best_trade,
            worst_trade=worst_trade,
            average_holding_period=average_holding_period,
        )

    def get_precision_87_analysis(self, trades: List[TradeRecord]) -> Dict[str, Any]:
        completed_trades = [
            trade for trade in trades if trade.action == "CLOSE" and trade.profit_loss is not None
        ]

        if not completed_trades:
            return {}

        precision_87_trades = [trade for trade in completed_trades if trade.precision_87_achieved]
        regular_trades = [trade for trade in completed_trades if not trade.precision_87_achieved]

        precision_87_stats = self._calculate_trade_stats(precision_87_trades)
        regular_stats = self._calculate_trade_stats(regular_trades)

        return {
            "total_trades": len(completed_trades),
            "precision_87_trades": len(precision_87_trades),
            "precision_87_ratio": len(precision_87_trades) / len(completed_trades) * 100,
            "precision_87_performance": precision_87_stats,
            "regular_performance": regular_stats,
            "performance_comparison": {
                "win_rate_difference": precision_87_stats["win_rate"] - regular_stats["win_rate"],
                "avg_return_difference": precision_87_stats["avg_return"] - regular_stats["avg_return"],
                "profit_factor_ratio": (
                    precision_87_stats["profit_factor"] / regular_stats["profit_factor"]
                    if regular_stats["profit_factor"] > 0
                    else float("inf")
                ),
            },
        }

    def _calculate_max_drawdown(self, trades: List[TradeRecord]) -> float:
        if not trades:
            return 0.0

        cumulative_returns: List[float] = []
        cumulative = 0.0

        for trade in sorted(trades, key=lambda x: x.timestamp):
            cumulative += trade.profit_loss or 0.0
            cumulative_returns.append(cumulative)

        if not cumulative_returns:
            return 0.0

        peak = cumulative_returns[0]
        max_drawdown = 0.0

        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / abs(peak) if peak else 0
            max_drawdown = max(max_drawdown, drawdown)

        return float(max_drawdown * 100)

    def _calculate_consecutive_streaks(self, profits: List[Optional[float]]) -> Tuple[int, int]:
        if not profits:
            return 0, 0

        max_wins = max_losses = current_wins = current_losses = 0

        for profit in profits:
            if profit and profit > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _calculate_average_holding_period(self, trades: List[TradeRecord]) -> float:
        holding_periods: List[float] = []

        for trade in trades:
            if trade.action == "CLOSE":
                holding_periods.append(float(self._calculate_holding_days(trade)))

        return float(np.mean(holding_periods)) if holding_periods else 0.0

    def _calculate_holding_days(self, trade: TradeRecord) -> int:
        # 実際には対応するOPEN取引との時間差を計算する必要があるが、簡略化
        return 1

    def _calculate_trade_stats(self, trades: List[TradeRecord]) -> Dict[str, float]:
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "profit_factor": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
            }

        profits = [trade.profit_loss for trade in trades if trade.profit_loss is not None]
        if not profits:
            return {
                "total_trades": len(trades),
                "win_rate": 0.0,
                "avg_return": 0.0,
                "profit_factor": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
            }

        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        return {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(profits) * 100 if profits else 0.0,
            "avg_return": float(np.mean(profits)),
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else float("inf"),
            "best_trade": max(profits),
            "worst_trade": min(profits),
        }

    def _empty_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return=0.0,
            total_return_pct=0.0,
            average_return=0.0,
            average_win=0.0,
            average_loss=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            precision_87_trades=0,
            precision_87_success_rate=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            average_holding_period=0.0,
        )
