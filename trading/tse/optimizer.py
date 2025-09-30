"""Portfolio optimisation utilities for the TSE 4000 project."""

from __future__ import annotations

from typing import Dict, List

from .analysis import StockProfile

PORTFOLIO_SIZES = [10, 15, 20, 25, 30]
DEFAULT_TARGET_SIZE = 20


class PortfolioOptimizer:
    """Selects balanced portfolios from analysed stock profiles."""

    def optimize_portfolio(
        self, profiles: List[StockProfile], target_size: int = DEFAULT_TARGET_SIZE
    ) -> List[StockProfile]:
        print(f"\n最適ポートフォリオ選択中（目標: {target_size}銘柄）...")

        sector_best = self._get_sector_best_stocks(profiles)
        selected = list(sector_best.values())
        remaining_slots = max(target_size - len(selected), 0)

        remaining_profiles = [p for p in profiles if p not in selected]
        remaining_profiles.sort(key=lambda x: x.combined_score, reverse=True)

        selected.extend(remaining_profiles[:remaining_slots])
        return selected[:target_size]

    def _get_sector_best_stocks(
        self, profiles: List[StockProfile]
    ) -> Dict[str, StockProfile]:
        sector_best: Dict[str, StockProfile] = {}
        for profile in profiles:
            if (
                profile.sector not in sector_best
                or profile.combined_score > sector_best[profile.sector].combined_score
            ):
                sector_best[profile.sector] = profile
        return sector_best
