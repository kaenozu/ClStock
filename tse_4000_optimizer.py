#!/usr/bin/env python3
"""東証4000銘柄最適組み合わせシステム."""

from __future__ import annotations

import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from data.stock_data import StockDataProvider
from trading.tse import (
    PORTFOLIO_SIZES,
    OptimizationReporter,
    PortfolioBacktester,
    PortfolioOptimizer,
    StockAnalyzer,
    StockProfile,
)
from utils.logger_config import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)


class TSE4000Optimizer:
    """Facade that orchestrates analysis, optimisation and reporting."""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.tse_universe = self._initialize_tse_universe()
        self.all_symbols = self._create_all_symbols_list()

        self.analyzer = StockAnalyzer(self.data_provider, self.tse_universe)
        self.portfolio_optimizer = PortfolioOptimizer()
        self.backtester = PortfolioBacktester(self.data_provider)
        self.reporter = OptimizationReporter()

        print(f"分析対象: {len(self.all_symbols)}銘柄（12セクター）")

    def _initialize_tse_universe(self) -> Dict[str, List[str]]:
        """東証主要銘柄リストの初期化（業界別分散）"""
        return {
            "finance": [
                "8306.T",
                "8411.T",
                "8316.T",
                "8031.T",
                "8002.T",
                "8001.T",
                "8058.T",
                "8750.T",
                "8725.T",
                "8771.T",
            ],
            "tech": [
                "6758.T",
                "9984.T",
                "4689.T",
                "9433.T",
                "9432.T",
                "6861.T",
                "6367.T",
                "6701.T",
                "8035.T",
                "4519.T",
                "3765.T",
                "4307.T",
                "4751.T",
                "2432.T",
                "4385.T",
            ],
            "automotive": [
                "7203.T",
                "7267.T",
                "7269.T",
                "6902.T",
                "7201.T",
                "7261.T",
                "9020.T",
                "9021.T",
                "9022.T",
                "5401.T",
                "5411.T",
                "7011.T",
            ],
            "manufacturing": [
                "6501.T",
                "6503.T",
                "6502.T",
                "6504.T",
                "7751.T",
                "6770.T",
                "6752.T",
                "6954.T",
                "6724.T",
                "6703.T",
                "7012.T",
                "5201.T",
            ],
            "consumer": [
                "9983.T",
                "3382.T",
                "8267.T",
                "3099.T",
                "2914.T",
                "2802.T",
                "2801.T",
                "4523.T",
                "4578.T",
                "4902.T",
            ],
            "energy": [
                "1605.T",
                "1332.T",
                "5020.T",
                "3865.T",
                "1801.T",
                "1802.T",
                "1803.T",
                "5101.T",
            ],
            "healthcare": [
                "4502.T",
                "4503.T",
                "4519.T",
                "4901.T",
                "4911.T",
                "4922.T",
                "4568.T",
                "4021.T",
            ],
            "realestate": [
                "8802.T",
                "1925.T",
                "1963.T",
                "1801.T",
                "1808.T",
                "1812.T",
                "1893.T",
                "1928.T",
            ],
            "telecom": ["9432.T", "9433.T", "9434.T", "4751.T", "2432.T", "4324.T"],
            "chemicals": [
                "4063.T",
                "4183.T",
                "4208.T",
                "4452.T",
                "3407.T",
                "4188.T",
                "4004.T",
                "4005.T",
            ],
            "food": ["2801.T", "2802.T", "2914.T", "1332.T", "2269.T", "2282.T"],
            "transport": ["9020.T", "9021.T", "9022.T", "9101.T", "9104.T"],
        }

    def _create_all_symbols_list(self) -> List[str]:
        all_symbols: List[str] = []
        for sector_symbols in self.tse_universe.values():
            all_symbols.extend(sector_symbols)
        return list(set(all_symbols))

    def run_comprehensive_optimization(self):
        """包括的最適化実行"""
        self._print_optimization_header()

        logger.info(
            "Starting comprehensive optimisation for %d symbols", len(self.all_symbols),
        )
        start_time = time.time()
        profiles = self.analyzer.parallel_analysis(self.all_symbols)
        analysis_time = time.time() - start_time

        print(f"\n分析完了: {len(profiles)}銘柄 ({analysis_time:.1f}秒)")

        if not profiles:
            print("分析可能な銘柄がありませんでした。")
            return None

        results = self._optimize_multiple_portfolio_sizes(profiles)
        self.display_optimization_results(results, profiles)
        return results

    def _print_optimization_header(self):
        print("=" * 80)
        print("東証4000銘柄最適組み合わせシステム")
        print("=" * 80)
        print(f"分析開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _optimize_multiple_portfolio_sizes(self, profiles: List[StockProfile]) -> Dict:
        results: Dict[int, Dict] = {}
        for portfolio_size in PORTFOLIO_SIZES:
            print(f"\n{portfolio_size}銘柄ポートフォリオ最適化...")

            selected = self.portfolio_optimizer.optimize_portfolio(
                profiles, portfolio_size,
            )
            selected_symbols = [p.symbol for p in selected]

            backtest_result = self.backtester.backtest_portfolio(selected_symbols)

            results[portfolio_size] = {
                "selected_profiles": selected,
                "backtest": backtest_result,
            }

            print(f"  利益率: {backtest_result['return_rate']:+.2f}%")

        return results

    def display_optimization_results(
        self, results: Dict, all_profiles: List[StockProfile],
    ):
        print("\n" + "=" * 80)
        print("最適化結果サマリー")
        print("=" * 80)

        best_result = self._find_best_result(results)
        self._display_portfolio_comparison(results)

        if best_result:
            self._display_optimal_portfolio_details(best_result)

        self._display_top_performers(all_profiles)
        self._finalize_results_display(results, all_profiles, best_result)

    def _find_best_result(self, results: Dict) -> Optional[Tuple[int, Dict]]:
        best_result: Optional[Tuple[int, Dict]] = None
        best_score = -float("inf")

        for size, result in results.items():
            backtest = result["backtest"]
            performance_score = backtest["return_rate"] * size * 0.1
            if performance_score > best_score:
                best_score = performance_score
                best_result = (size, result)

        return best_result

    def _display_portfolio_comparison(self, results: Dict):
        print("\nポートフォリオサイズ別パフォーマンス:")
        print("-" * 60)
        print(f"{'サイズ':>6} {'利益率':>8} {'総利益':>12} {'取引数':>8} {'最適性':>8}")
        print("-" * 60)

        for size, result in results.items():
            backtest = result["backtest"]
            performance_score = backtest["return_rate"] * size * 0.1

            print(
                f"{size:>6} {backtest['return_rate']:>+7.2f}% {backtest['total_return']:>+11,.0f}円 "
                f"{backtest['total_trades']:>7} {performance_score:>+7.1f}",
            )

    def _display_optimal_portfolio_details(self, best_result: Tuple[int, Dict]):
        best_size, best_data = best_result
        print(f"\n最適ポートフォリオ: {best_size}銘柄")
        print("=" * 50)

        self._display_sector_breakdown(best_data["selected_profiles"])

    def _display_sector_breakdown(self, selected_profiles: List[StockProfile]):
        print("\n選定銘柄（セクター別）:")
        sector_groups: Dict[str, List[StockProfile]] = {}
        for profile in selected_profiles:
            sector_groups.setdefault(profile.sector, []).append(profile)

        for sector, profiles in sector_groups.items():
            print(f"\n【{sector.upper()}】")
            for profile in profiles:
                print(
                    f"  {profile.symbol}: 総合スコア {profile.combined_score:.1f} "
                    f"(利益性: {profile.profit_potential:+.1f}%, 多様性: {profile.diversity_score:.1f})",
                )

    def _display_top_performers(self, all_profiles: List[StockProfile]):
        print("\n全体トップ10銘柄:")
        print("-" * 50)
        top_performers = sorted(
            all_profiles, key=lambda x: x.combined_score, reverse=True,
        )[:10]
        for i, profile in enumerate(top_performers, 1):
            print(
                f"{i:2d}. {profile.symbol} [{profile.sector}] スコア: {profile.combined_score:.1f}",
            )

    def _finalize_results_display(
        self,
        results: Dict,
        all_profiles: List[StockProfile],
        best_result: Optional[Tuple[int, Dict]],
    ):
        print(f"\n分析完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n[保存] 結果保存中...")
        save_result = self.reporter.save_optimization_results(
            best_result, results, all_profiles,
        )
        if save_result:
            self.reporter.print_save_success(save_result)
        else:
            print("[エラー] 保存に失敗しました")


def main():
    optimizer = TSE4000Optimizer()
    results = optimizer.run_comprehensive_optimization()

    if results:
        print("\n最適化完了！最高の多彩性と利益性の組み合わせを発見しました。")


if __name__ == "__main__":
    main()
