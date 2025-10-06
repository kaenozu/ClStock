from __future__ import annotations

"""チャート生成ユーティリティ"""

import base64
from io import BytesIO
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from trading.models import TradeRecord


class ChartGenerator:
    """取引パフォーマンスチャート生成"""

    def __init__(self) -> None:
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def create_equity_curve(self, trades: List[TradeRecord]) -> str:
        fig, ax = plt.subplots(figsize=(12, 6))
        sorted_trades = sorted(trades, key=lambda x: x.timestamp)

        cumulative = 0.0
        cumulative_pnl = []
        dates = []
        for trade in sorted_trades:
            cumulative += trade.profit_loss or 0.0
            cumulative_pnl.append(cumulative)
            dates.append(trade.timestamp)

        ax.plot(dates, cumulative_pnl, linewidth=2, label="累積損益")
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_title("資産曲線", fontsize=14, fontweight="bold")
        ax.set_xlabel("日付")
        ax.set_ylabel("累積損益 (円)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def create_monthly_returns_chart(self, trades: List[TradeRecord]) -> str:
        fig, ax = plt.subplots(figsize=(12, 6))

        monthly_returns: Dict[str, float] = {}
        for trade in trades:
            month_key = trade.timestamp.strftime("%Y-%m")
            monthly_returns.setdefault(month_key, 0.0)
            monthly_returns[month_key] += trade.profit_loss or 0.0

        months = sorted(monthly_returns.keys())
        returns = [monthly_returns[month] for month in months]
        colors = ["green" if r >= 0 else "red" for r in returns]

        ax.bar(months, returns, color=colors, alpha=0.7)
        ax.set_title("月次リターン", fontsize=14, fontweight="bold")
        ax.set_xlabel("月")
        ax.set_ylabel("リターン (円)")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def create_trade_distribution_chart(self, trades: List[TradeRecord]) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        profits = [trade.profit_loss or 0.0 for trade in trades]

        ax1.hist(profits, bins=20, alpha=0.7, color="blue", edgecolor="black")
        ax1.set_title("取引損益分布", fontsize=12, fontweight="bold")
        ax1.set_xlabel("損益 (円)")
        ax1.set_ylabel("取引数")
        ax1.axvline(x=0, color="red", linestyle="--", alpha=0.7)

        ax2.boxplot(profits)
        ax2.set_title("取引損益ボックスプロット", fontsize=12, fontweight="bold")
        ax2.set_ylabel("損益 (円)")
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.7)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def create_precision_analysis_chart(self, trades: List[TradeRecord]) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        precision_87_trades = [trade for trade in trades if trade.precision_87_achieved]
        regular_trades = [trade for trade in trades if not trade.precision_87_achieved]

        if precision_87_trades and regular_trades:
            precision_87_wins = len(
                [
                    trade
                    for trade in precision_87_trades
                    if trade.profit_loss and trade.profit_loss > 0
                ],
            )
            regular_wins = len(
                [
                    trade
                    for trade in regular_trades
                    if trade.profit_loss and trade.profit_loss > 0
                ],
            )

            precision_87_win_rate = precision_87_wins / len(precision_87_trades) * 100
            regular_win_rate = regular_wins / len(regular_trades) * 100

            categories = ["87%精度取引", "通常取引"]
            win_rates = [precision_87_win_rate, regular_win_rate]

            ax1.bar(categories, win_rates, color=["gold", "skyblue"], alpha=0.8)
            ax1.set_title("勝率比較", fontsize=12, fontweight="bold")
            ax1.set_ylabel("勝率 (%)")
            ax1.set_ylim(0, 100)

        if precision_87_trades and regular_trades:
            precision_87_avg = np.mean(
                [trade.profit_loss or 0.0 for trade in precision_87_trades],
            )
            regular_avg = np.mean(
                [trade.profit_loss or 0.0 for trade in regular_trades],
            )

            categories = ["87%精度取引", "通常取引"]
            avg_returns = [precision_87_avg, regular_avg]
            colors = ["green" if r >= 0 else "red" for r in avg_returns]

            ax2.bar(categories, avg_returns, color=colors, alpha=0.8)
            ax2.set_title("平均リターン比較", fontsize=12, fontweight="bold")
            ax2.set_ylabel("平均リターン (円)")
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64
