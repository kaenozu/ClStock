"""ClStock バックテストエンジン

87%精度システムの過去データでの戦略検証
取引コスト、スリッページ、税金を考慮した高精度バックテスト
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from config.target_universe import get_target_universe

# 既存システム
from data.stock_data import StockDataProvider
from models.precision.precision_87_system import Precision87BreakthroughSystem

from .backtest import (
    BacktestOptimizer,
    BacktestRunner,
    generate_backtest_charts,
    generate_recommendations,
)
from .models import PerformanceMetrics

# 内部モジュール
from .trading_strategy import TradingStrategy


@dataclass
class BacktestConfig:
    """バックテスト設定"""

    start_date: datetime
    end_date: datetime
    initial_capital: float = 1000000
    commission_rate: float = 0.001  # 0.1%
    spread_rate: float = 0.0005  # 0.05%
    slippage_rate: float = 0.0002  # 0.02%
    tax_rate: float = 0.20315  # 20.315%
    precision_threshold: float = 85.0
    confidence_threshold: float = 0.7
    max_position_size: float = 0.1
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    target_symbols: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """バックテスト結果"""

    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    total_trades: int
    win_rate: float
    profit_factor: float
    precision_87_trades: int
    precision_87_success_rate: float
    final_value: float
    benchmark_return: float
    excess_return: float
    beta: float
    alpha: float
    information_ratio: float
    var_95: float
    expected_shortfall: float
    total_costs: float
    total_tax: float
    daily_returns: List[float]
    trade_history: List[Dict[str, Any]]
    portfolio_values: List[Tuple[datetime, float]]


@dataclass
class WalkForwardResult:
    """ウォークフォワード分析結果"""

    period_results: List[BacktestResult]
    combined_performance: PerformanceMetrics
    stability_metrics: Dict[str, float]
    degradation_analysis: Dict[str, float]


class BacktestEngine:
    """87%精度システム統合バックテストエンジン

    過去データを使用した戦略検証と最適化
    """

    def __init__(self, config: BacktestConfig):
        """Args:
        config: バックテスト設定

        """
        self.config = config

        # システムコンポーネント
        self.data_provider = StockDataProvider()
        self.precision_system = Precision87BreakthroughSystem()

        # バックテスト固有の設定
        self.trading_strategy = TradingStrategy(
            initial_capital=config.initial_capital,
            precision_threshold=config.precision_threshold,
            confidence_threshold=config.confidence_threshold,
        )

        # 取引コスト設定
        self.trading_strategy.commission_rate = config.commission_rate
        self.trading_strategy.spread_rate = config.spread_rate
        self.trading_strategy.slippage_rate = config.slippage_rate

        self.logger = logging.getLogger(__name__)

    def run_backtest(
        self,
        symbols: Optional[List[str]] = None,
        parallel: bool = True,
    ) -> BacktestResult:
        """バックテスト実行

        Args:
            symbols: 対象銘柄リスト（Noneの場合は設定から取得）
            parallel: 並列処理フラグ

        Returns:
            バックテスト結果

        """
        try:
            target_symbols = (
                symbols or self.config.target_symbols or self._get_default_symbols()
            )

            self.logger.info(
                f"バックテスト開始: {self.config.start_date} - {self.config.end_date}",
            )
            self.logger.info(f"対象銘柄数: {len(target_symbols)}")

            runner = BacktestRunner(
                self.config,
                self.trading_strategy,
                self.data_provider,
                self.logger,
            )
            result = runner.run_backtest(target_symbols)

            self.logger.info(f"バックテスト完了: 総リターン {result.total_return:.2%}")
            return result

        except Exception as e:
            self.logger.error(f"バックテスト実行エラー: {e}")
            return self._empty_backtest_result()

    def run_walk_forward_analysis(
        self,
        training_months: int = 6,
        testing_months: int = 1,
        step_months: int = 1,
    ) -> WalkForwardResult:
        """ウォークフォワード分析

        Args:
            training_months: 訓練期間（月数）
            testing_months: テスト期間（月数）
            step_months: ステップサイズ（月数）

        Returns:
            ウォークフォワード分析結果

        """
        try:
            period_results = []
            current_start = self.config.start_date

            while current_start <= self.config.end_date:
                # 訓練期間
                training_end = current_start + timedelta(days=training_months * 30)

                # テスト期間
                testing_start = training_end + timedelta(days=1)
                testing_end = testing_start + timedelta(days=testing_months * 30)

                if testing_end > self.config.end_date:
                    break

                # 期間バックテスト設定
                period_config = BacktestConfig(
                    start_date=testing_start,
                    end_date=testing_end,
                    initial_capital=self.config.initial_capital,
                    commission_rate=self.config.commission_rate,
                    spread_rate=self.config.spread_rate,
                    slippage_rate=self.config.slippage_rate,
                    precision_threshold=self.config.precision_threshold,
                    confidence_threshold=self.config.confidence_threshold,
                    target_symbols=self.config.target_symbols,
                )

                # 期間バックテスト実行
                period_engine = BacktestEngine(period_config)
                period_result = period_engine.run_backtest()
                period_results.append(period_result)

                self.logger.info(
                    f"期間完了: {testing_start.date()} - {testing_end.date()} "
                    f"リターン: {period_result.total_return:.2%}",
                )

                # 次の期間へ
                current_start += timedelta(days=step_months * 30)

            # 統合分析
            combined_performance = self._calculate_combined_performance(period_results)
            stability_metrics = self._calculate_stability_metrics(period_results)
            degradation_analysis = self._analyze_performance_degradation(period_results)

            return WalkForwardResult(
                period_results=period_results,
                combined_performance=combined_performance,
                stability_metrics=stability_metrics,
                degradation_analysis=degradation_analysis,
            )

        except Exception as e:
            self.logger.error(f"ウォークフォワード分析エラー: {e}")
            return WalkForwardResult(
                period_results=[],
                combined_performance=PerformanceMetrics(
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                stability_metrics={},
                degradation_analysis={},
            )

    def optimize_parameters(
        self,
        parameter_ranges: Dict[str, List[float]],
        optimization_metric: str = "sharpe_ratio",
    ) -> Dict[str, Any]:
        """パラメータ最適化

        Args:
            parameter_ranges: パラメータ範囲辞書
            optimization_metric: 最適化メトリクス

        Returns:
            最適化結果

        """
        optimizer = BacktestOptimizer(self.logger)
        return optimizer.optimize_parameters(
            base_config=self.config,
            parameter_ranges=parameter_ranges,
            optimization_metric=optimization_metric,
            engine_factory=BacktestEngine,
        )

    def generate_backtest_report(self, result: BacktestResult) -> Dict[str, Any]:
        """バックテストレポート生成"""
        try:
            # 基本統計
            basic_stats = {
                "period": f"{result.start_date.date()} - {result.end_date.date()}",
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "precision_87_trades": result.precision_87_trades,
                "precision_87_success_rate": result.precision_87_success_rate,
            }

            # リスク分析
            risk_analysis = {
                "var_95": result.var_95,
                "expected_shortfall": result.expected_shortfall,
                "beta": result.beta,
                "alpha": result.alpha,
                "information_ratio": result.information_ratio,
            }

            # コスト分析
            cost_analysis = {
                "total_costs": result.total_costs,
                "total_tax": result.total_tax,
                "cost_ratio": (
                    result.total_costs / result.final_value
                    if result.final_value > 0
                    else 0
                ),
                "net_return": result.total_return
                - (result.total_costs + result.total_tax)
                / result.config.initial_capital,
            }

            # 月次分析
            monthly_analysis = self._calculate_monthly_analysis(result)

            # 取引分析
            trade_analysis = self._analyze_trades(result.trade_history)

            # チャート生成
            charts = generate_backtest_charts(result, logger=self.logger)

            return {
                "basic_statistics": basic_stats,
                "risk_analysis": risk_analysis,
                "cost_analysis": cost_analysis,
                "monthly_analysis": monthly_analysis,
                "trade_analysis": trade_analysis,
                "charts": charts,
                "recommendations": generate_recommendations(result),
            }

        except Exception as e:
            self.logger.error(f"バックテストレポート生成エラー: {e}")
            return {}

    # --- プライベートメソッド ---

    def _is_trading_day(self, date: datetime) -> bool:
        """取引日判定"""
        # 平日のみ（簡略化）
        return date.weekday() < 5

    @staticmethod
    def _get_default_symbols() -> List[str]:
        """デフォルト銘柄リスト"""
        return get_target_universe().default_formatted()

    def _empty_backtest_result(self) -> BacktestResult:
        """空のバックテスト結果"""
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

    def _calculate_combined_performance(
        self,
        period_results: List[BacktestResult],
    ) -> PerformanceMetrics:
        """統合パフォーマンス計算"""
        if not period_results:
            return PerformanceMetrics(
                0,
                0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

        # 統計計算（簡略化）
        returns = [r.total_return for r in period_results]
        total_trades = sum(r.total_trades for r in period_results)
        winning_trades = sum(r.total_trades * r.win_rate / 100 for r in period_results)

        return PerformanceMetrics(
            total_trades=int(total_trades),
            winning_trades=int(winning_trades),
            losing_trades=int(total_trades - winning_trades),
            win_rate=winning_trades / total_trades * 100 if total_trades > 0 else 0,
            total_return=sum(returns),
            total_return_pct=sum(returns) * 100,
            average_return=np.mean(returns),
            average_win=(
                np.mean([r for r in returns if r > 0])
                if [r for r in returns if r > 0]
                else 0
            ),
            average_loss=(
                np.mean([r for r in returns if r < 0])
                if [r for r in returns if r < 0]
                else 0
            ),
            profit_factor=np.mean([r.profit_factor for r in period_results]),
            sharpe_ratio=np.mean([r.sharpe_ratio for r in period_results]),
            sortino_ratio=np.mean([r.sortino_ratio for r in period_results]),
            max_drawdown=max([r.max_drawdown for r in period_results]),
            max_consecutive_wins=0,  # 簡略化
            max_consecutive_losses=0,  # 簡略化
            precision_87_trades=sum(r.precision_87_trades for r in period_results),
            precision_87_success_rate=np.mean(
                [r.precision_87_success_rate for r in period_results],
            ),
            best_trade=max(
                [
                    max(r.daily_returns) if r.daily_returns else 0
                    for r in period_results
                ],
            ),
            worst_trade=min(
                [
                    min(r.daily_returns) if r.daily_returns else 0
                    for r in period_results
                ],
            ),
            average_holding_period=1.0,  # 簡略化
        )

    def _calculate_stability_metrics(
        self,
        period_results: List[BacktestResult],
    ) -> Dict[str, float]:
        """安定性メトリクス計算"""
        if not period_results:
            return {}

        returns = [r.total_return for r in period_results]
        sharpe_ratios = [r.sharpe_ratio for r in period_results]

        return {
            "return_stability": 1 / (np.std(returns) + 1e-6),
            "sharpe_stability": 1 / (np.std(sharpe_ratios) + 1e-6),
            "positive_periods_ratio": len([r for r in returns if r > 0]) / len(returns),
            "max_loss_period": min(returns) if returns else 0,
            "consistency_score": np.mean(returns) / (np.std(returns) + 1e-6),
        }

    def _analyze_performance_degradation(
        self,
        period_results: List[BacktestResult],
    ) -> Dict[str, float]:
        """パフォーマンス劣化分析"""
        if len(period_results) < 2:
            return {}

        # 時系列トレンド分析
        returns = [r.total_return for r in period_results]
        x = list(range(len(returns)))

        # 線形回帰で傾きを計算
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, returns)

        return {
            "trend_slope": slope,
            "trend_significance": 1 - p_value,
            "r_squared": r_value**2,
            "early_period_avg": (
                np.mean(returns[: len(returns) // 2]) if len(returns) >= 4 else 0
            ),
            "late_period_avg": (
                np.mean(returns[len(returns) // 2 :]) if len(returns) >= 4 else 0
            ),
            "degradation_detected": slope < -0.01 and p_value < 0.05,
        }

    def _calculate_monthly_analysis(self, result: BacktestResult) -> Dict[str, Any]:
        """月次分析"""
        # 簡略化実装
        return {
            "best_month": max(result.daily_returns) if result.daily_returns else 0,
            "worst_month": min(result.daily_returns) if result.daily_returns else 0,
            "positive_months": len([r for r in result.daily_returns if r > 0]),
            "negative_months": len([r for r in result.daily_returns if r < 0]),
        }

    def _analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """取引分析"""
        if not trades:
            return {}

        # 簡略化実装
        return {
            "avg_trade_size": np.mean([t.get("position_size", 0) for t in trades]),
            "largest_win": max(
                [
                    t.get("profit_loss", 0)
                    for t in trades
                    if t.get("profit_loss", 0) > 0
                ],
                default=0,
            ),
            "largest_loss": min(
                [
                    t.get("profit_loss", 0)
                    for t in trades
                    if t.get("profit_loss", 0) < 0
                ],
                default=0,
            ),
            "avg_holding_period": 1.0,  # 簡略化
        }
