"""Reporting utilities for backtest results."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

if False:  # pragma: no cover - typing helper
    from trading.backtest_engine import BacktestResult


def generate_backtest_charts(
    result: "BacktestResult", logger: Optional[object] = None
) -> Dict[str, str]:
    """Create base64-encoded charts summarising the backtest."""

    charts: Dict[str, str] = {}

    try:
        if result.portfolio_values:
            fig, ax = plt.subplots(figsize=(12, 6))
            dates = [pv[0] for pv in result.portfolio_values]
            values = [pv[1] for pv in result.portfolio_values]

            ax.plot(dates, values, linewidth=2, label="ポートフォリオ価値")
            ax.axhline(
                y=result.config.initial_capital,
                color="red",
                linestyle="--",
                alpha=0.5,
                label="初期資本",
            )
            ax.set_title("バックテスト資産曲線", fontweight="bold")
            ax.set_ylabel("価値 (円)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()

            charts["equity_curve"] = _fig_to_base64(fig)
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger is not None:
            try:
                logger.error(f"チャート生成エラー: {exc}")
            except Exception:
                pass

    return charts


def generate_recommendations(result: "BacktestResult") -> List[str]:
    """Create a list of improvement suggestions for the strategy."""

    recommendations: List[str] = []

    if result.sharpe_ratio < 1.0:
        recommendations.append("シャープレシオが低いため、リスク調整が必要です")

    if result.max_drawdown > 0.2:
        recommendations.append(
            "最大ドローダウンが大きいため、リスク管理を強化してください"
        )

    if result.win_rate < 50:
        recommendations.append("勝率が低いため、エントリー条件の見直しを検討してください")

    if result.precision_87_success_rate < 60:
        recommendations.append(
            "87%精度取引の成功率が低いため、精度判定基準の調整が必要です"
        )

    return recommendations


def _fig_to_base64(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return image_base64
