#!/usr/bin/env python3
"""
東証4000銘柄最適組み合わせシステム
多彩性と利益性を両立する最適ポートフォリオを発見
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time
import json
import csv

warnings.filterwarnings("ignore")

from data.stock_data import StockDataProvider
import logging
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


@dataclass
class StockProfile:
    symbol: str
    sector: str
    market_cap: float
    volatility: float
    profit_potential: float
    diversity_score: float
    combined_score: float


# 最適化パラメータ定数
PORTFOLIO_SIZES = [10, 15, 20, 25, 30]
DEFAULT_TARGET_SIZE = 20
INITIAL_CAPITAL = 1000000
POSITION_SIZE_PERCENTAGE = 0.05

# パフォーマンス計算定数
ANNUAL_TRADING_DAYS = 252
LARGE_CAP_THRESHOLD = 1e12
MID_CAP_THRESHOLD = 1e11

# 移動平均期間定数
SHORT_MA_PERIOD = 10
LONG_MA_PERIOD = 30
TREND_MA_PERIOD_SHORT = 20
TREND_MA_PERIOD_LONG = 60

# スコア重み定数
PROFIT_WEIGHT = 0.5
DIVERSITY_WEIGHT = 0.3
STABILITY_WEIGHT = 0.2
STABILITY_MULTIPLIER = 10

# 並列処理定数
MAX_WORKERS = 10
PROGRESS_REPORT_INTERVAL = 10

# ボラティリティ分類閾値
LOW_VOLATILITY = 0.2
MID_VOLATILITY = 0.4


class TSE4000Optimizer:
    def __init__(self):
        self.data_provider = StockDataProvider()
        self.tse_universe = self._initialize_tse_universe()
        self.all_symbols = self._create_all_symbols_list()

        print(f"分析対象: {len(self.all_symbols)}銘柄（12セクター）")

    def _initialize_tse_universe(self) -> Dict[str, List[str]]:
        """東証主要銘柄リストの初期化（業界別分散）"""
        return {
            # 金融・銀行（10銘柄）
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
            # テクノロジー・IT（15銘柄）
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
            # 自動車・輸送（12銘柄）
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
            # 製造業・重工業（12銘柄）
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
            # 消費・小売（10銘柄）
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
            # エネルギー・資源（8銘柄）
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
            # ヘルスケア・製薬（8銘柄）
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
            # 不動産・建設（8銘柄）
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
            # 通信・メディア（6銘柄）
            "telecom": ["9432.T", "9433.T", "9434.T", "4751.T", "2432.T", "4324.T"],
            # 化学・材料（8銘柄）
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
            # 食品・農業（6銘柄）
            "food": ["2801.T", "2802.T", "2914.T", "1332.T", "2269.T", "2282.T"],
            # 航空・海運（5銘柄）
            "transport": ["9020.T", "9021.T", "9022.T", "9101.T", "9104.T"],
        }

    def _create_all_symbols_list(self) -> List[str]:
        """全銘柄リストの作成"""
        all_symbols = []
        for sector_symbols in self.tse_universe.values():
            all_symbols.extend(sector_symbols)
        return list(set(all_symbols))  # 重複除去

    def analyze_stock_profile(self, symbol: str) -> StockProfile:
        """個別銘柄の包括的分析"""
        try:
            stock_data = self._get_stock_data_for_analysis(symbol)
            if stock_data.empty:
                return None

            return self._create_stock_profile(symbol, stock_data)

        except Exception as e:
            logging.warning(f"Error analyzing {symbol}: {str(e)}")
            return None

    def _get_stock_data_for_analysis(self, symbol: str) -> pd.DataFrame:
        """分析用株価データ取得"""
        return self.data_provider.get_stock_data(symbol, "2y")

    def _create_stock_profile(
        self, symbol: str, stock_data: pd.DataFrame
    ) -> StockProfile:
        """株価プロファイルの作成"""
        close = stock_data["Close"]
        volume = stock_data["Volume"]

        # 基本指標計算
        volatility = self._calculate_volatility(close)
        profit_potential = self._calculate_profit_potential(close)
        sector = self.determine_sector(symbol)
        market_cap = self._estimate_market_cap(close, volume)
        diversity_score = self.calculate_diversity_score(symbol, volatility, market_cap)
        combined_score = self._calculate_combined_score(
            profit_potential, diversity_score, volatility
        )

        return StockProfile(
            symbol=symbol,
            sector=sector,
            market_cap=market_cap,
            volatility=volatility,
            profit_potential=profit_potential,
            diversity_score=diversity_score,
            combined_score=combined_score,
        )

    def _calculate_volatility(self, close: pd.Series) -> float:
        """年率ボラティリティ計算"""
        returns = close.pct_change().dropna()
        return returns.std() * np.sqrt(ANNUAL_TRADING_DAYS)

    def _calculate_profit_potential(self, close: pd.Series) -> float:
        """利益ポテンシャル計算（トレンド + モメンタム）"""
        ma_short = close.rolling(TREND_MA_PERIOD_SHORT).mean()
        ma_long = close.rolling(TREND_MA_PERIOD_LONG).mean()

        trend_strength = self._calculate_trend_strength(ma_short, ma_long)
        momentum = self._calculate_momentum(close)

        return (trend_strength + momentum) * 100

    def _calculate_trend_strength(
        self, ma_short: pd.Series, ma_long: pd.Series
    ) -> float:
        """トレンド強度計算"""
        if ma_long.iloc[-1] > 0:
            return (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
        return 0

    def _calculate_momentum(self, close: pd.Series) -> float:
        """モメンタム計算"""
        if close.iloc[-20] > 0:
            return (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
        return 0

    def _estimate_market_cap(self, close: pd.Series, volume: pd.Series) -> float:
        """時価総額推定（価格×出来高で近似）"""
        return close.iloc[-1] * volume.mean()

    def _calculate_combined_score(
        self, profit_potential: float, diversity_score: float, volatility: float
    ) -> float:
        """総合スコア計算"""
        stability = 1 / (volatility + 0.01)  # ボラティリティの逆数
        return (
            profit_potential * PROFIT_WEIGHT
            + diversity_score * DIVERSITY_WEIGHT
            + stability * STABILITY_MULTIPLIER * STABILITY_WEIGHT
        )

    def determine_sector(self, symbol: str) -> str:
        """銘柄のセクター判定"""
        for sector, symbols in self.tse_universe.items():
            if symbol in symbols:
                return sector
        return "other"

    def calculate_diversity_score(
        self, symbol: str, volatility: float, market_cap: float
    ) -> float:
        """多様性スコア計算"""
        sector_weight = self._get_sector_weight(symbol)
        cap_score = self._get_cap_score(market_cap)
        vol_score = self._get_volatility_score(volatility)

        return sector_weight * cap_score * vol_score

    def _get_sector_weight(self, symbol: str) -> float:
        """セクター重み取得"""
        sector = self.determine_sector(symbol)
        sector_weights = {
            "finance": 1.0,
            "tech": 1.2,
            "automotive": 1.0,
            "manufacturing": 0.9,
            "consumer": 1.1,
            "energy": 1.3,
            "healthcare": 1.2,
            "realestate": 0.8,
            "telecom": 1.0,
            "chemicals": 0.9,
            "food": 0.7,
            "transport": 1.1,
        }
        return sector_weights.get(sector, 1.0)

    def _get_cap_score(self, market_cap: float) -> float:
        """時価総額スコア取得"""
        if market_cap > LARGE_CAP_THRESHOLD:  # 大型株
            return 1.0
        elif market_cap > MID_CAP_THRESHOLD:  # 中型株
            return 1.2
        else:  # 小型株
            return 1.1

    def _get_volatility_score(self, volatility: float) -> float:
        """ボラティリティスコア取得"""
        if volatility < LOW_VOLATILITY:  # 低ボラ
            return 1.0
        elif volatility < MID_VOLATILITY:  # 中ボラ
            return 1.1
        else:  # 高ボラ
            return 1.2

    def parallel_analysis(self) -> List[StockProfile]:
        """並列処理でシンボル分析"""
        print("並列分析開始...")
        profiles = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_stock_profile, symbol): symbol
                for symbol in self.all_symbols
            }

            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    profile = future.result()
                    if profile:
                        profiles.append(profile)
                    completed += 1
                    if completed % PROGRESS_REPORT_INTERVAL == 0:
                        print(f"分析完了: {completed}/{len(self.all_symbols)}")
                except Exception as e:
                    logging.warning(f"分析失敗 {symbol}: {str(e)}")

        return profiles

    def optimize_portfolio(
        self, profiles: List[StockProfile], target_size: int = DEFAULT_TARGET_SIZE
    ) -> List[StockProfile]:
        """最適ポートフォリオ選択（遺伝的アルゴリズム風）"""
        print(f"\n最適ポートフォリオ選択中（目標: {target_size}銘柄）...")

        # セクター別に最低1銘柄は含める制約
        sector_best = self._get_sector_best_stocks(profiles)

        # 必須銘柄（各セクターのベスト）
        selected = list(sector_best.values())
        remaining_slots = target_size - len(selected)

        # 残りスロットを総合スコアで埋める
        remaining_profiles = [p for p in profiles if p not in selected]
        remaining_profiles.sort(key=lambda x: x.combined_score, reverse=True)

        selected.extend(remaining_profiles[:remaining_slots])
        return selected[:target_size]

    def _get_sector_best_stocks(
        self, profiles: List[StockProfile]
    ) -> Dict[str, StockProfile]:
        """セクター別ベスト銘柄の取得"""
        sector_best = {}
        for profile in profiles:
            if (
                profile.sector not in sector_best
                or profile.combined_score > sector_best[profile.sector].combined_score
            ):
                sector_best[profile.sector] = profile
        return sector_best

    def backtest_portfolio(self, selected_symbols: List[str]) -> Dict:
        """選択されたポートフォリオのバックテスト"""
        print(f"\nバックテスト実行中（{len(selected_symbols)}銘柄）...")

        portfolio_state = self._initialize_portfolio_state()

        for symbol in selected_symbols:
            try:
                self._backtest_single_symbol(symbol, portfolio_state)
            except Exception as e:
                logging.warning(f"バックテスト失敗 {symbol}: {str(e)}")

        # 残ポジションを現在価格で評価
        self._evaluate_remaining_positions(portfolio_state)

        return self._calculate_backtest_results(portfolio_state)

    def _initialize_portfolio_state(self) -> Dict:
        """ポートフォリオ状態の初期化"""
        return {
            "current_capital": INITIAL_CAPITAL,
            "positions": {},
            "transaction_history": [],
        }

    def _backtest_single_symbol(self, symbol: str, portfolio_state: Dict):
        """個別銘柄のバックテスト"""
        stock_data = self.data_provider.get_stock_data(symbol, "1y")
        if stock_data.empty:
            return

        self._execute_trading_strategy(symbol, stock_data, portfolio_state)

    def _execute_trading_strategy(
        self, symbol: str, stock_data: pd.DataFrame, portfolio_state: Dict
    ):
        """トレーディング戦略の実行"""
        close = stock_data["Close"]
        ma_short = close.rolling(SHORT_MA_PERIOD).mean()
        ma_long = close.rolling(LONG_MA_PERIOD).mean()

        position_size = portfolio_state["current_capital"] * POSITION_SIZE_PERCENTAGE

        for i in range(LONG_MA_PERIOD, len(close) - 1):
            current_price = close.iloc[i]

            # 買いシグナル処理
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

            # 売りシグナル処理
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
        i: int,
        symbol: str,
        positions: Dict,
    ) -> bool:
        """買いシグナルの判定"""
        return (
            ma_short.iloc[i] > ma_long.iloc[i]
            and ma_short.iloc[i - 1] <= ma_long.iloc[i - 1]
            and symbol not in positions
        )

    def _is_sell_signal(
        self,
        ma_short: pd.Series,
        ma_long: pd.Series,
        i: int,
        symbol: str,
        positions: Dict,
    ) -> bool:
        """売りシグナルの判定"""
        return (
            ma_short.iloc[i] < ma_long.iloc[i]
            and ma_short.iloc[i - 1] >= ma_long.iloc[i - 1]
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
        """買い注文の実行"""
        shares = int(position_size / price)
        if shares > 0 and portfolio_state["current_capital"] >= shares * price:
            portfolio_state["positions"][symbol] = {
                "shares": shares,
                "buy_price": price,
                "buy_date": date,
            }
            portfolio_state["current_capital"] -= shares * price
            portfolio_state["transaction_history"].append(
                {
                    "symbol": symbol,
                    "action": "BUY",
                    "shares": shares,
                    "price": price,
                    "date": date,
                }
            )

    def _execute_sell_order(
        self, symbol: str, price: float, date, portfolio_state: Dict
    ):
        """売り注文の実行"""
        position = portfolio_state["positions"][symbol]
        shares = position["shares"]
        profit = (price - position["buy_price"]) * shares

        portfolio_state["current_capital"] += shares * price
        del portfolio_state["positions"][symbol]

        portfolio_state["transaction_history"].append(
            {
                "symbol": symbol,
                "action": "SELL",
                "shares": shares,
                "price": price,
                "profit": profit,
                "date": date,
            }
        )

    def _evaluate_remaining_positions(self, portfolio_state: Dict):
        """残ポジションの評価"""
        for symbol, position in list(portfolio_state["positions"].items()):
            try:
                current_data = self.data_provider.get_stock_data(symbol, "1d")
                if not current_data.empty:
                    current_price = current_data["Close"].iloc[-1]
                    portfolio_state["current_capital"] += (
                        position["shares"] * current_price
                    )
            except Exception:
                pass

    def _calculate_backtest_results(self, portfolio_state: Dict) -> Dict:
        """バックテスト結果の計算"""
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

    def run_comprehensive_optimization(self):
        """包括的最適化実行"""
        self._print_optimization_header()

        # ステップ1: 全銘柄分析
        start_time = time.time()
        profiles = self.parallel_analysis()
        analysis_time = time.time() - start_time

        print(f"\n分析完了: {len(profiles)}銘柄 ({analysis_time:.1f}秒)")

        if not profiles:
            print("分析可能な銘柄がありませんでした。")
            return

        # ステップ2: 複数サイズでポートフォリオ最適化
        results = self._optimize_multiple_portfolio_sizes(profiles)

        # ステップ3: 結果表示
        self.display_optimization_results(results, profiles)
        return results

    def _print_optimization_header(self):
        """最適化ヘッダーの表示"""
        print("=" * 80)
        print("東証4000銘柄最適組み合わせシステム")
        print("=" * 80)
        print(f"分析開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _optimize_multiple_portfolio_sizes(self, profiles: List[StockProfile]) -> Dict:
        """複数サイズでのポートフォリオ最適化"""
        results = {}
        for portfolio_size in PORTFOLIO_SIZES:
            print(f"\n{portfolio_size}銘柄ポートフォリオ最適化...")

            selected = self.optimize_portfolio(profiles, portfolio_size)
            selected_symbols = [p.symbol for p in selected]

            # バックテスト
            backtest_result = self.backtest_portfolio(selected_symbols)

            results[portfolio_size] = {
                "selected_profiles": selected,
                "backtest": backtest_result,
            }

            print(f"  利益率: {backtest_result['return_rate']:+.2f}%")

        return results

    def display_optimization_results(
        self, results: Dict, all_profiles: List[StockProfile]
    ):
        """最適化結果の表示"""
        print("\n" + "=" * 80)
        print("最適化結果サマリー")
        print("=" * 80)

        best_result = self._find_best_result(results)
        self._display_portfolio_comparison(results)

        if best_result:
            self._display_optimal_portfolio_details(best_result, results)

        self._display_top_performers(all_profiles)
        self._finalize_results_display(results, all_profiles)

    def _find_best_result(self, results: Dict) -> Optional[Tuple[int, Dict]]:
        """最良結果の特定"""
        best_result = None
        best_score = -float("inf")

        for size, result in results.items():
            backtest = result["backtest"]
            performance_score = backtest["return_rate"] * size * 0.1
            if performance_score > best_score:
                best_score = performance_score
                best_result = (size, result)

        return best_result

    def _display_portfolio_comparison(self, results: Dict):
        """ポートフォリオサイズ別結果の表示"""
        print("\nポートフォリオサイズ別パフォーマンス:")
        print("-" * 60)
        print(f"{'サイズ':>6} {'利益率':>8} {'総利益':>12} {'取引数':>8} {'最適性':>8}")
        print("-" * 60)

        for size, result in results.items():
            backtest = result["backtest"]
            performance_score = backtest["return_rate"] * size * 0.1

            print(
                f"{size:>6} {backtest['return_rate']:>+7.2f}% {backtest['total_return']:>+11,.0f}円 "
                f"{backtest['total_trades']:>7} {performance_score:>+7.1f}"
            )

    def _display_optimal_portfolio_details(
        self, best_result: Tuple[int, Dict], results: Dict
    ):
        """最適ポートフォリオ詳細の表示"""
        best_size, best_data = best_result
        print(f"\n最適ポートフォリオ: {best_size}銘柄")
        print("=" * 50)

        self._display_sector_breakdown(best_data["selected_profiles"])

    def _display_sector_breakdown(self, selected_profiles: List[StockProfile]):
        """セクター別内訳の表示"""
        print("\n選定銘柄（セクター別）:")
        sector_groups = {}
        for profile in selected_profiles:
            if profile.sector not in sector_groups:
                sector_groups[profile.sector] = []
            sector_groups[profile.sector].append(profile)

        for sector, profiles in sector_groups.items():
            print(f"\n【{sector.upper()}】")
            for profile in profiles:
                print(
                    f"  {profile.symbol}: 総合スコア {profile.combined_score:.1f} "
                    f"(利益性: {profile.profit_potential:+.1f}%, 多様性: {profile.diversity_score:.1f})"
                )

    def _display_top_performers(self, all_profiles: List[StockProfile]):
        """トップパフォーマーの表示"""
        print(f"\n全体トップ10銘柄:")
        print("-" * 50)
        top_performers = sorted(
            all_profiles, key=lambda x: x.combined_score, reverse=True
        )[:10]
        for i, profile in enumerate(top_performers, 1):
            print(
                f"{i:2d}. {profile.symbol} [{profile.sector}] スコア: {profile.combined_score:.1f}"
            )

    def _finalize_results_display(
        self, results: Dict, all_profiles: List[StockProfile]
    ):
        """結果表示の最終処理"""
        print(f"\n分析完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 結果を保存
        print("\n[保存] 結果保存中...")
        save_result = self.save_optimization_results(results, all_profiles)
        if save_result:
            self._print_save_success(save_result)
        else:
            print("[エラー] 保存に失敗しました")

    def _print_save_success(self, save_result: Dict):
        """保存成功メッセージの表示"""
        print("\n[完了] 保存完了!")
        print(f"CSV: {save_result['csv_file']}")
        print(f"JSON: {save_result['json_file']}")
        print(f"レポート: {save_result['report_file']}")
        print(
            f"最適解: {save_result['optimal_portfolio_size']}銘柄で{save_result['expected_return']}期待利益"
        )

    def save_optimization_results(
        self, results: Dict, all_profiles: List[StockProfile]
    ):
        """最適化結果をファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        best_result = self._find_best_result(results)
        if not best_result:
            return None

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
        """全結果ファイルの保存"""
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
            timestamp, best_size, backtest_result, selected_profiles, results
        )

        return {
            "csv_file": csv_file,
            "json_file": json_file,
            "report_file": report_file,
            "optimal_portfolio_size": best_size,
            "expected_return": f"{backtest_result['return_rate']:+.2f}%",
        }

    def _save_csv_file(
        self, timestamp: str, selected_profiles: List[StockProfile]
    ) -> str:
        """CSVファイルの保存"""
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
                ]
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
                    ]
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
        """JSONファイルの保存"""
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
        """JSON用データの作成"""
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
        """ポートフォリオ比較データの作成"""
        comparison = {}
        for size, result in results.items():
            comparison[f"{size}_stocks"] = {
                "return_rate": f"{result['backtest']['return_rate']:.2f}%",
                "total_profit": result["backtest"]["total_return"],
                "total_trades": result["backtest"]["total_trades"],
            }
        return comparison

    def _create_optimal_portfolio_data(
        self, selected_profiles: List[StockProfile]
    ) -> List[Dict]:
        """最適ポートフォリオデータの作成"""
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
                }
            )
        return portfolio_data

    def _create_sector_analysis_data(
        self, selected_profiles: List[StockProfile]
    ) -> Dict:
        """セクター分析データの作成"""
        sector_groups = {}
        for profile in selected_profiles:
            if profile.sector not in sector_groups:
                sector_groups[profile.sector] = []
            sector_groups[profile.sector].append(profile)

        sector_analysis = {}
        for sector, profiles in sector_groups.items():
            sector_analysis[sector] = {
                "count": len(profiles),
                "avg_score": round(
                    sum(p.combined_score for p in profiles) / len(profiles), 2
                ),
                "avg_profit_potential": round(
                    sum(p.profit_potential for p in profiles) / len(profiles), 2
                ),
                "symbols": [p.symbol for p in profiles],
            }
        return sector_analysis

    def _create_top_performers_data(
        self, all_profiles: List[StockProfile]
    ) -> List[Dict]:
        """トップパフォーマーデータの作成"""
        top_performers = sorted(
            all_profiles, key=lambda x: x.combined_score, reverse=True
        )[:10]
        performers_data = []
        for i, profile in enumerate(top_performers, 1):
            performers_data.append(
                {
                    "rank": i,
                    "symbol": profile.symbol,
                    "sector": profile.sector,
                    "combined_score": round(profile.combined_score, 2),
                }
            )
        return performers_data

    def _save_report_file(
        self,
        timestamp: str,
        best_size: int,
        backtest_result: Dict,
        selected_profiles: List[StockProfile],
        results: Dict,
    ) -> str:
        """レポートファイルの保存"""
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

        print(f"✅ 投資推奨レポート保存: {report_filename}")
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
        """レポート内容の書き込み"""
        reportfile.write("=" * 80 + "\n")
        reportfile.write("TSE 4000銘柄最適化投資推奨レポート\n")
        reportfile.write("=" * 80 + "\n")
        reportfile.write(
            f"作成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n"
        )

        # 投資戦略サマリー
        self._write_strategy_summary(reportfile, best_size, backtest_result)

        # セクター別投資配分
        self._write_sector_allocation(reportfile, selected_profiles, best_size, results)

        # トップ推奨銘柄
        self._write_top_recommendations(reportfile, selected_profiles)

        # リスク管理指針
        self._write_risk_management(reportfile)

        # 実行タイミング
        self._write_execution_timing(reportfile)

    def _write_strategy_summary(
        self, reportfile, best_size: int, backtest_result: Dict
    ):
        """投資戦略サマリーの書き込み"""
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
        """セクター別投資配分の書き込み"""
        reportfile.write("🏆 セクター別投資配分\n")
        reportfile.write("-" * 40 + "\n")

        sector_analysis = self._create_sector_analysis_data(selected_profiles)
        for sector, info in sector_analysis.items():
            percentage = (info["count"] / best_size) * 100
            reportfile.write(
                f"{sector.upper()}: {info['count']}銘柄 ({percentage:.1f}%) "
            )
            reportfile.write(f"- 平均スコア: {info['avg_score']:.1f}\n")
            reportfile.write(f"  推奨銘柄: {', '.join(info['symbols'])}\n\n")

    def _write_top_recommendations(
        self, reportfile, selected_profiles: List[StockProfile]
    ):
        """トップ推奨銘柄の書き込み"""
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
        """リスク管理指針の書き込み"""
        reportfile.write("⚠️  リスク管理指針\n")
        reportfile.write("-" * 40 + "\n")
        reportfile.write("• 各銘柄への投資比率は5-10%以内に制限\n")
        reportfile.write("• セクター集中リスクを避け、12セクターに分散\n")
        reportfile.write("• 利確目標: 2-3%、損切りライン: -1%\n")
        reportfile.write("• 四半期毎にポートフォリオの見直しを実施\n\n")

    def _write_execution_timing(self, reportfile):
        """実行タイミングの書き込み"""
        reportfile.write("📈 実行タイミング\n")
        reportfile.write("-" * 40 + "\n")
        reportfile.write("• 推奨実行期間: 今月中\n")
        reportfile.write("• 市場開始30分後の価格で順次投資\n")
        reportfile.write("• 1日2-3銘柄ずつ段階的に建玉\n")
        reportfile.write("• 全ポジション構築まで約2週間を予定\n")


def main():
    optimizer = TSE4000Optimizer()
    results = optimizer.run_comprehensive_optimization()

    if results:
        print(f"\n最適化完了！最高の多彩性と利益性の組み合わせを発見しました。")


if __name__ == "__main__":
    main()
