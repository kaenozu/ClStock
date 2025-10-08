"""Portfolio optimization module for ClStock."""

from dataclasses import dataclass
from typing import Any, Dict, List

from trading.tse.analysis import StockProfile


@dataclass
class OptimizationResult:
    """Optimization result dataclass."""

    selected_stocks: List[str]
    weights: List[float]
    expected_return: float
    risk: float


class PortfolioOptimizer:
    """Portfolio optimizer class."""

    def __init__(self):
        pass

    def optimize_portfolio(
        self,
        profiles: List[StockProfile],
        target_size: int = 10,
    ) -> List[StockProfile]:
        """Optimize portfolio with given stock profiles."""
        # 単純なポートフォリオ最適化ロジック（ダミー）
        # volatility が低い順にソートして、上位 target_size を選択
        sorted_profiles = sorted(profiles, key=lambda x: x.volatility)
        return sorted_profiles[:target_size]

    def calculate_optimal_weights(
        self,
        selected_profiles: List[StockProfile],
    ) -> List[float]:
        """Calculate optimal weights for selected stocks."""
        n = len(selected_profiles)
        return [1.0 / n] * n  # 等重量

    def get_optimization_summary(self, result: OptimizationResult) -> Dict[str, Any]:
        """Get optimization summary."""
        return {
            "selected_stocks": result.selected_stocks,
            "total_expected_return": result.expected_return,
            "total_risk": result.risk,
        }
