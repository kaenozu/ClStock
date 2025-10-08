"""Reporting helpers for the TSE 4000 optimizer."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .analysis import StockProfile


class OptimizationReporter:
    """Handles persistence and reporting side effects."""

    def save_optimization_results(
        self,
        best_result: Optional[Tuple[int, Dict]],
        results: Dict,
        all_profiles: List[StockProfile],
    ) -> Optional[Dict]:
        if not best_result:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        best_size, best_data = best_result
        selected_profiles = best_data["selected_profiles"]
        backtest_result = best_data["backtest"]

        return self._save_all_result_files(
            timestamp,
            best_size,
            selected_profiles,
            backtest_result,
            results,
            all_profiles,
        )

    def _save_all_result_files(
        self,
        timestamp: str,
        best_size: int,
        selected_profiles: List[StockProfile],
        backtest_result: Dict,
        results: Dict,
        all_profiles: List[StockProfile],
    ) -> Dict:
        csv_file = self._save_csv_file(timestamp, selected_profiles)
        json_file = self._save_json_file(
            timestamp,
            best_size,
            backtest_result,
            results,
            selected_profiles,
            all_profiles,
        )
        report_file = self._save_report_file(
            timestamp,
            best_size,
            backtest_result,
            selected_profiles,
            results,
        )

        return {
            "csv_file": csv_file,
            "json_file": json_file,
            "report_file": report_file,
            "optimal_portfolio_size": best_size,
            "expected_return": f"{backtest_result['return_rate']:+.2f}%",
        }

    def _save_csv_file(
        self,
        timestamp: str,
        selected_profiles: List[StockProfile],
    ) -> str:
        csv_filename = f"optimal_portfolio_{timestamp}.csv"
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Symbol",
                    "Sector",
                    "Combined_Score",
                    "Profit_Potential",
                    "Diversity_Score",
                    "Volatility",
                    "Market_Cap",
                ],
            )

            for profile in selected_profiles:
                writer.writerow(
                    [
                        profile.symbol,
                        profile.sector,
                        f"{profile.combined_score:.2f}",
                        f"{profile.profit_potential:.2f}",
                        f"{profile.diversity_score:.2f}",
                        f"{profile.volatility:.4f}",
                        f"{profile.market_cap:.0f}",
                    ],
                )

        print(f"[保存完了] 最適ポートフォリオCSV: {csv_filename}")
        return csv_filename

    def _save_json_file(
        self,
        timestamp: str,
        best_size: int,
        backtest_result: Dict,
        results: Dict,
        selected_profiles: List[StockProfile],
        all_profiles: List[StockProfile],
    ) -> str:
        json_data = self._create_json_data(
            timestamp,
            best_size,
            backtest_result,
            results,
            selected_profiles,
            all_profiles,
        )

        json_filename = f"tse_optimization_report_{timestamp}.json"
        with open(json_filename, "w", encoding="utf-8") as jsonfile:
            json.dump(json_data, jsonfile, ensure_ascii=False, indent=2)

        print(f"[保存完了] 詳細レポートJSON: {json_filename}")
        return json_filename

    def _create_json_data(
        self,
        timestamp: str,
        best_size: int,
        backtest_result: Dict,
        results: Dict,
        selected_profiles: List[StockProfile],
        all_profiles: List[StockProfile],
    ) -> Dict:
        return {
            "optimization_timestamp": timestamp,
            "optimization_summary": {
                "best_portfolio_size": best_size,
                "total_return_rate": f"{backtest_result['return_rate']:.2f}%",
                "total_profit": backtest_result["total_return"],
                "total_trades": backtest_result["total_trades"],
            },
            "portfolio_comparison": self._create_portfolio_comparison_data(results),
            "optimal_portfolio": self._create_optimal_portfolio_data(selected_profiles),
            "sector_analysis": self._create_sector_analysis_data(selected_profiles),
            "top_performers": self._create_top_performers_data(all_profiles),
        }

    def _create_portfolio_comparison_data(self, results: Dict) -> Dict:
        comparison = {}
        for size, result in results.items():
            comparison[f"{size}_stocks"] = {
                "return_rate": f"{result['backtest']['return_rate']:.2f}%",
                "total_profit": result["backtest"]["total_return"],
                "total_trades": result["backtest"]["total_trades"],
            }
        return comparison

    def _create_optimal_portfolio_data(
        self,
        selected_profiles: List[StockProfile],
    ) -> List[Dict]:
        portfolio_data = []
        for profile in selected_profiles:
            portfolio_data.append(
                {
                    "symbol": profile.symbol,
                    "sector": profile.sector,
                    "combined_score": round(profile.combined_score, 2),
                    "profit_potential": round(profile.profit_potential, 2),
                    "diversity_score": round(profile.diversity_score, 2),
                    "volatility": round(profile.volatility, 4),
                    "market_cap": int(profile.market_cap),
                },
            )
        return portfolio_data

    def _create_sector_analysis_data(
        self,
        selected_profiles: List[StockProfile],
    ) -> Dict:
        sector_groups: Dict[str, Dict] = {}
        for profile in selected_profiles:
            info = sector_groups.setdefault(
                profile.sector,
                {"count": 0, "total_score": 0.0, "symbols": []},
            )
            info["count"] += 1
            info["total_score"] += profile.combined_score
            info["symbols"].append(profile.symbol)

        for info in sector_groups.values():
            if info["count"]:
                info["avg_score"] = info["total_score"] / info["count"]
            else:
                info["avg_score"] = 0
        return sector_groups

    def _create_top_performers_data(
        self,
        all_profiles: List[StockProfile],
    ) -> List[Dict]:
        top_performers = sorted(
            all_profiles,
            key=lambda x: x.combined_score,
            reverse=True,
        )[:10]
        return [
            {
                "symbol": profile.symbol,
                "sector": profile.sector,
                "combined_score": round(profile.combined_score, 2),
            }
            for profile in top_performers
        ]

    def _save_report_file(
        self,
        timestamp: str,
        best_size: int,
        backtest_result: Dict,
        selected_profiles: List[StockProfile],
        results: Dict,
    ) -> str:
        report_filename = f"investment_recommendation_{timestamp}.txt"
        with open(report_filename, "w", encoding="utf-8") as reportfile:
            self._write_report_content(
                reportfile,
                timestamp,
                best_size,
                backtest_result,
                selected_profiles,
                results,
            )

        print(f"[保存] 投資推奨レポート保存: {report_filename}")
        return report_filename

    def _write_report_content(
        self,
        reportfile,
        timestamp: str,
        best_size: int,
        backtest_result: Dict,
        selected_profiles: List[StockProfile],
        results: Dict,
    ):
        reportfile.write("=" * 80 + "\n")
        reportfile.write("TSE 4000銘柄最適化投資推奨レポート\n")
        reportfile.write("=" * 80 + "\n")
        reportfile.write(
            f"作成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n",
        )

        self._write_strategy_summary(reportfile, best_size, backtest_result)
        self._write_sector_allocation(reportfile, selected_profiles, best_size, results)
        self._write_top_recommendations(reportfile, selected_profiles)
        self._write_risk_management(reportfile)
        self._write_execution_timing(reportfile)

    def _write_strategy_summary(
        self,
        reportfile,
        best_size: int,
        backtest_result: Dict,
    ):
        reportfile.write("📊 投資戦略サマリー\n")
        reportfile.write("-" * 40 + "\n")
        reportfile.write(f"推奨ポートフォリオサイズ: {best_size}銘柄\n")
        reportfile.write(f"期待利益率: {backtest_result['return_rate']:+.2f}%\n")
        reportfile.write(f"期待利益額: {backtest_result['total_return']:+,.0f}円\n")
        reportfile.write(f"予想取引回数: {backtest_result['total_trades']}回\n\n")

    def _write_sector_allocation(
        self,
        reportfile,
        selected_profiles: List[StockProfile],
        best_size: int,
        results: Dict,
    ):
        reportfile.write("🏆 セクター別投資配分\n")
        reportfile.write("-" * 40 + "\n")

        sector_analysis = self._create_sector_analysis_data(selected_profiles)
        for sector, info in sector_analysis.items():
            percentage = (info["count"] / best_size) * 100 if best_size else 0
            reportfile.write(
                f"{sector.upper()}: {info['count']}銘柄 ({percentage:.1f}%) ",
            )
            reportfile.write(f"- 平均スコア: {info['avg_score']:.1f}\n")
            reportfile.write(f"  推奨銘柄: {', '.join(info['symbols'])}\n\n")

    def _write_top_recommendations(
        self,
        reportfile,
        selected_profiles: List[StockProfile],
    ):
        reportfile.write("💎 トップ推奨銘柄（上位5銘柄）\n")
        reportfile.write("-" * 40 + "\n")
        top_5 = sorted(selected_profiles, key=lambda x: x.combined_score, reverse=True)[
            :5
        ]
        for i, profile in enumerate(top_5, 1):
            reportfile.write(f"{i}. {profile.symbol} [{profile.sector}] ")
            reportfile.write(f"スコア: {profile.combined_score:.1f}\n")
            reportfile.write(f"   期待利益: {profile.profit_potential:+.1f}% ")
            reportfile.write(f"多様性: {profile.diversity_score:.1f}\n\n")

    def _write_risk_management(self, reportfile):
        reportfile.write("⚠️  リスク管理指針\n")
        reportfile.write("-" * 40 + "\n")
        reportfile.write("• 各銘柄への投資比率は5-10%以内に制限\n")
        reportfile.write("• セクター集中リスクを避け、12セクターに分散\n")
        reportfile.write("• 利確目標: 2-3%、損切りライン: -1%\n")
        reportfile.write("• 四半期毎にポートフォリオの見直しを実施\n\n")

    def _write_execution_timing(self, reportfile):
        reportfile.write("📈 実行タイミング\n")
        reportfile.write("-" * 40 + "\n")
        reportfile.write("• 推奨実行期間: 今月中\n")
        reportfile.write("• 市場開始30分後の価格で順次投資\n")
        reportfile.write("• 1日2-3銘柄ずつ段階的に建玉\n")
        reportfile.write("• 全ポジション構築まで約2週間を予定\n")

    def print_save_success(self, save_result: Dict):
        print("\n[完了] 保存完了!")
        print(f"CSV: {save_result['csv_file']}")
        print(f"JSON: {save_result['json_file']}")
        print(f"レポート: {save_result['report_file']}")
        print(
            f"最適解: {save_result['optimal_portfolio_size']}銘柄で{save_result['expected_return']}期待利益",
        )
