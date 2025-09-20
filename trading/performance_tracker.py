"""
ClStock パフォーマンストラッカー

日次・週次・月次のP&L計算、リスク調整リターン、
ベンチマーク比較、アルファ・ベータ分析を提供
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# 既存システム
from data.stock_data import StockDataProvider
from .trade_recorder import TradeRecorder, PerformanceMetrics


@dataclass
class DailyPerformance:
    """日次パフォーマンス"""
    date: datetime
    portfolio_value: float
    daily_return: float
    daily_pnl: float
    benchmark_return: float
    excess_return: float
    cumulative_return: float
    drawdown: float
    volatility: float
    active_positions: int
    trades_count: int


@dataclass
class PeriodPerformance:
    """期間パフォーマンス"""
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    var_99: float
    skewness: float
    kurtosis: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    up_capture: float
    down_capture: float


@dataclass
class BenchmarkComparison:
    """ベンチマーク比較"""
    benchmark_symbol: str
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    correlation: float
    beta: float
    alpha: float
    r_squared: float
    tracking_error: float
    information_ratio: float
    up_market_performance: float
    down_market_performance: float


@dataclass
class RiskMetrics:
    """リスクメトリクス"""
    volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float
    var_99: float
    expected_shortfall: float
    skewness: float
    kurtosis: float
    tail_ratio: float


class PerformanceTracker:
    """
    パフォーマンス追跡・分析システム

    87%精度システムと統合されたパフォーマンス分析
    """

    def __init__(self,
                 initial_capital: float = 1000000,
                 benchmark_symbol: str = "^N225"):
        """
        Args:
            initial_capital: 初期資本
            benchmark_symbol: ベンチマーク銘柄（日経平均）
        """
        self.initial_capital = initial_capital
        self.benchmark_symbol = benchmark_symbol

        # データプロバイダー
        self.data_provider = StockDataProvider()

        # パフォーマンス履歴
        self.daily_performance: List[DailyPerformance] = []
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []

        # ベンチマークデータ
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.benchmark_returns: List[float] = []

        # 計算済みメトリクス
        self.current_metrics: Optional[PeriodPerformance] = None
        self.risk_metrics: Optional[RiskMetrics] = None

        self.logger = logging.getLogger(__name__)

    def update_performance(self,
                          current_portfolio_value: float,
                          active_positions: int,
                          trades_count: int = 0) -> bool:
        """
        パフォーマンス更新

        Args:
            current_portfolio_value: 現在のポートフォリオ価値
            active_positions: アクティブポジション数
            trades_count: 当日取引数

        Returns:
            更新成功フラグ
        """
        try:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

            # ベンチマークデータ更新
            self._update_benchmark_data()

            # 日次リターン計算
            daily_return = 0.0
            daily_pnl = 0.0
            if self.portfolio_values:
                previous_value = self.portfolio_values[-1][1]
                daily_return = (current_portfolio_value - previous_value) / previous_value
                daily_pnl = current_portfolio_value - previous_value
            else:
                # 初回
                daily_return = (current_portfolio_value - self.initial_capital) / self.initial_capital
                daily_pnl = current_portfolio_value - self.initial_capital

            # ベンチマークリターン取得
            benchmark_return = self._get_benchmark_return(today)

            # 超過リターン
            excess_return = daily_return - benchmark_return

            # 累積リターン
            cumulative_return = (current_portfolio_value - self.initial_capital) / self.initial_capital

            # ドローダウン計算
            drawdown = self._calculate_current_drawdown(current_portfolio_value)

            # ボラティリティ計算（過去30日）
            volatility = self._calculate_rolling_volatility()

            # 日次パフォーマンス記録
            daily_perf = DailyPerformance(
                date=today,
                portfolio_value=current_portfolio_value,
                daily_return=daily_return,
                daily_pnl=daily_pnl,
                benchmark_return=benchmark_return,
                excess_return=excess_return,
                cumulative_return=cumulative_return,
                drawdown=drawdown,
                volatility=volatility,
                active_positions=active_positions,
                trades_count=trades_count
            )

            self.daily_performance.append(daily_perf)
            self.portfolio_values.append((today, current_portfolio_value))
            self.daily_returns.append(daily_return)
            self.benchmark_returns.append(benchmark_return)

            # 過去1年分のみ保持
            if len(self.daily_performance) > 252:
                self.daily_performance.pop(0)
                self.portfolio_values.pop(0)
                self.daily_returns.pop(0)
                self.benchmark_returns.pop(0)

            # メトリクス更新
            self._update_metrics()

            return True

        except Exception as e:
            self.logger.error(f"パフォーマンス更新エラー: {e}")
            return False

    def get_period_performance(self,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> PeriodPerformance:
        """
        期間パフォーマンス取得

        Args:
            start_date: 開始日（Noneの場合は全期間）
            end_date: 終了日（Noneの場合は現在まで）

        Returns:
            期間パフォーマンス
        """
        try:
            # 期間フィルタリング
            filtered_performance = self._filter_performance_by_date(start_date, end_date)

            if not filtered_performance:
                return self._empty_period_performance()

            # データ準備
            returns = [p.daily_return for p in filtered_performance]
            benchmark_returns = [p.benchmark_return for p in filtered_performance]
            portfolio_values = [p.portfolio_value for p in filtered_performance]

            start_value = portfolio_values[0]
            end_value = portfolio_values[-1]
            days = len(filtered_performance)

            # 基本リターン計算
            total_return = (end_value - start_value) / start_value
            annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0.0

            # リスクメトリクス
            volatility = np.std(returns) * np.sqrt(252) if returns else 0.0
            downside_returns = [r for r in returns if r < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0.0

            # シャープレシオ（リスクフリーレート = 0と仮定）
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0

            # ソルティーノレシオ
            sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0.0

            # 最大ドローダウン
            max_drawdown = max([p.drawdown for p in filtered_performance], default=0.0)

            # カルマーレシオ
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

            # VaR計算
            var_95 = np.percentile(returns, 5) if returns else 0.0
            var_99 = np.percentile(returns, 1) if returns else 0.0

            # 歪度・尖度
            skewness = stats.skew(returns) if len(returns) > 2 else 0.0
            kurtosis = stats.kurtosis(returns) if len(returns) > 2 else 0.0

            # ベンチマーク関連
            beta, alpha, correlation, r_squared = self._calculate_beta_alpha(returns, benchmark_returns)

            # 情報レシオ
            excess_returns = [p.excess_return for p in filtered_performance]
            tracking_error = np.std(excess_returns) * np.sqrt(252) if excess_returns else 0.0
            information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0.0

            # アップ・ダウンキャプチャー
            up_capture, down_capture = self._calculate_capture_ratios(returns, benchmark_returns)

            return PeriodPerformance(
                start_date=filtered_performance[0].date,
                end_date=filtered_performance[-1].date,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                var_99=var_99,
                skewness=skewness,
                kurtosis=kurtosis,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                up_capture=up_capture,
                down_capture=down_capture
            )

        except Exception as e:
            self.logger.error(f"期間パフォーマンス計算エラー: {e}")
            return self._empty_period_performance()

    def get_benchmark_comparison(self,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> BenchmarkComparison:
        """
        ベンチマーク比較分析

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            ベンチマーク比較結果
        """
        try:
            filtered_performance = self._filter_performance_by_date(start_date, end_date)

            if not filtered_performance:
                return self._empty_benchmark_comparison()

            # リターンデータ取得
            portfolio_returns = [p.daily_return for p in filtered_performance]
            benchmark_returns = [p.benchmark_return for p in filtered_performance]

            # 累積リターン計算
            portfolio_cumulative = np.prod([1 + r for r in portfolio_returns]) - 1
            benchmark_cumulative = np.prod([1 + r for r in benchmark_returns]) - 1
            excess_return = portfolio_cumulative - benchmark_cumulative

            # 相関・ベータ・アルファ
            correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1] if len(portfolio_returns) > 1 else 0.0
            beta, alpha, _, r_squared = self._calculate_beta_alpha(portfolio_returns, benchmark_returns)

            # トラッキングエラー
            excess_returns = [p.excess_return for p in filtered_performance]
            tracking_error = np.std(excess_returns) * np.sqrt(252) if excess_returns else 0.0

            # 情報レシオ
            information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0.0

            # アップ・ダウンマーケットパフォーマンス
            up_market_performance, down_market_performance = self._calculate_market_performance(
                portfolio_returns, benchmark_returns
            )

            return BenchmarkComparison(
                benchmark_symbol=self.benchmark_symbol,
                portfolio_return=portfolio_cumulative,
                benchmark_return=benchmark_cumulative,
                excess_return=excess_return,
                correlation=correlation,
                beta=beta,
                alpha=alpha,
                r_squared=r_squared,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                up_market_performance=up_market_performance,
                down_market_performance=down_market_performance
            )

        except Exception as e:
            self.logger.error(f"ベンチマーク比較エラー: {e}")
            return self._empty_benchmark_comparison()

    def get_monthly_summary(self) -> List[Dict[str, Any]]:
        """月次サマリー取得"""
        try:
            monthly_data = defaultdict(lambda: {
                'month': '',
                'total_return': 0.0,
                'volatility': 0.0,
                'max_drawdown': 0.0,
                'trades_count': 0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'benchmark_return': 0.0,
                'excess_return': 0.0
            })

            # 月別データ集計
            for perf in self.daily_performance:
                month_key = perf.date.strftime('%Y-%m')
                monthly_data[month_key]['month'] = month_key
                monthly_data[month_key]['trades_count'] += perf.trades_count

            # 月別計算
            for month_key in monthly_data.keys():
                month_performance = [
                    p for p in self.daily_performance
                    if p.date.strftime('%Y-%m') == month_key
                ]

                if month_performance:
                    returns = [p.daily_return for p in month_performance]
                    benchmark_returns = [p.benchmark_return for p in month_performance]

                    # 月次リターン
                    monthly_return = np.prod([1 + r for r in returns]) - 1
                    benchmark_monthly_return = np.prod([1 + r for r in benchmark_returns]) - 1

                    monthly_data[month_key]['total_return'] = monthly_return
                    monthly_data[month_key]['benchmark_return'] = benchmark_monthly_return
                    monthly_data[month_key]['excess_return'] = monthly_return - benchmark_monthly_return
                    monthly_data[month_key]['volatility'] = np.std(returns) * np.sqrt(252)
                    monthly_data[month_key]['max_drawdown'] = max([p.drawdown for p in month_performance])

                    # 勝率計算（正のリターンの日数比率）
                    positive_days = len([r for r in returns if r > 0])
                    monthly_data[month_key]['win_rate'] = positive_days / len(returns) * 100

                    # シャープレシオ
                    if np.std(returns) > 0:
                        monthly_data[month_key]['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)

            return list(monthly_data.values())

        except Exception as e:
            self.logger.error(f"月次サマリーエラー: {e}")
            return []

    def get_risk_metrics(self) -> RiskMetrics:
        """リスクメトリクス取得"""
        try:
            if not self.daily_returns:
                return RiskMetrics(
                    volatility=0.0, downside_volatility=0.0, max_drawdown=0.0,
                    max_drawdown_duration=0, var_95=0.0, var_99=0.0,
                    expected_shortfall=0.0, skewness=0.0, kurtosis=0.0, tail_ratio=0.0
                )

            returns = self.daily_returns

            # ボラティリティ
            volatility = np.std(returns) * np.sqrt(252)

            # ダウンサイドボラティリティ
            downside_returns = [r for r in returns if r < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0.0

            # 最大ドローダウンと期間
            max_drawdown = max([p.drawdown for p in self.daily_performance], default=0.0)
            max_drawdown_duration = self._calculate_max_drawdown_duration()

            # VaR
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)

            # 期待ショートフォール（CVaR）
            tail_returns = [r for r in returns if r <= var_95]
            expected_shortfall = np.mean(tail_returns) if tail_returns else var_95

            # 歪度・尖度
            skewness = stats.skew(returns) if len(returns) > 2 else 0.0
            kurtosis = stats.kurtosis(returns) if len(returns) > 2 else 0.0

            # テールレシオ（95パーセンタイル / 5パーセンタイル）
            tail_ratio = abs(np.percentile(returns, 95) / np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 0.0

            return RiskMetrics(
                volatility=volatility,
                downside_volatility=downside_volatility,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                skewness=skewness,
                kurtosis=kurtosis,
                tail_ratio=tail_ratio
            )

        except Exception as e:
            self.logger.error(f"リスクメトリクスエラー: {e}")
            return RiskMetrics(
                volatility=0.0, downside_volatility=0.0, max_drawdown=0.0,
                max_drawdown_duration=0, var_95=0.0, var_99=0.0,
                expected_shortfall=0.0, skewness=0.0, kurtosis=0.0, tail_ratio=0.0
            )

    def generate_performance_charts(self) -> Dict[str, str]:
        """パフォーマンスチャート生成"""
        charts = {}

        try:
            if not self.daily_performance:
                return charts

            # 1. 累積リターン比較
            charts['cumulative_returns'] = self._create_cumulative_returns_chart()

            # 2. 月次リターン
            charts['monthly_returns'] = self._create_monthly_returns_chart()

            # 3. リスクリターン散布図
            charts['risk_return'] = self._create_risk_return_chart()

            # 4. ドローダウンチャート
            charts['drawdown'] = self._create_drawdown_chart()

            # 5. ローリングシャープレシオ
            charts['rolling_sharpe'] = self._create_rolling_sharpe_chart()

        except Exception as e:
            self.logger.error(f"チャート生成エラー: {e}")

        return charts

    def export_performance_report(self, filepath: str) -> bool:
        """パフォーマンスレポートエクスポート"""
        try:
            # 全期間パフォーマンス
            period_perf = self.get_period_performance()
            benchmark_comp = self.get_benchmark_comparison()
            risk_metrics = self.get_risk_metrics()
            monthly_summary = self.get_monthly_summary()

            # レポートデータ作成
            report_data = {
                'report_date': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
                'current_value': self.portfolio_values[-1][1] if self.portfolio_values else self.initial_capital,
                'period_performance': self._dataclass_to_dict(period_perf),
                'benchmark_comparison': self._dataclass_to_dict(benchmark_comp),
                'risk_metrics': self._dataclass_to_dict(risk_metrics),
                'monthly_summary': monthly_summary,
                'daily_performance': [self._dataclass_to_dict(p) for p in self.daily_performance]
            }

            # JSON出力
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

            return True

        except Exception as e:
            self.logger.error(f"レポートエクスポートエラー: {e}")
            return False

    def reset_performance(self):
        """パフォーマンスリセット"""
        self.daily_performance.clear()
        self.portfolio_values.clear()
        self.daily_returns.clear()
        self.benchmark_returns.clear()
        self.current_metrics = None
        self.risk_metrics = None

    # --- プライベートメソッド ---

    def _update_benchmark_data(self):
        """ベンチマークデータ更新"""
        try:
            if self.benchmark_data is None or len(self.benchmark_data) == 0:
                self.benchmark_data = self.data_provider.get_stock_data(
                    self.benchmark_symbol, period="1y"
                )
        except Exception as e:
            self.logger.error(f"ベンチマークデータ更新エラー: {e}")

    def _get_benchmark_return(self, date: datetime) -> float:
        """指定日のベンチマークリターン取得"""
        try:
            if self.benchmark_data is None or len(self.benchmark_data) < 2:
                return 0.0

            # 最新の2日分でリターン計算（簡略化）
            recent_prices = self.benchmark_data['Close'].tail(2)
            if len(recent_prices) >= 2:
                return (recent_prices.iloc[-1] - recent_prices.iloc[-2]) / recent_prices.iloc[-2]

            return 0.0

        except Exception as e:
            self.logger.error(f"ベンチマークリターン取得エラー: {e}")
            return 0.0

    def _calculate_current_drawdown(self, current_value: float) -> float:
        """現在のドローダウン計算"""
        if not self.portfolio_values:
            return 0.0

        peak_value = max([v[1] for v in self.portfolio_values] + [current_value])
        return (peak_value - current_value) / peak_value if peak_value > 0 else 0.0

    def _calculate_rolling_volatility(self, window: int = 30) -> float:
        """ローリングボラティリティ計算"""
        if len(self.daily_returns) < window:
            return np.std(self.daily_returns) * np.sqrt(252) if self.daily_returns else 0.0

        recent_returns = self.daily_returns[-window:]
        return np.std(recent_returns) * np.sqrt(252)

    def _calculate_beta_alpha(self,
                            portfolio_returns: List[float],
                            benchmark_returns: List[float]) -> Tuple[float, float, float, float]:
        """ベータ・アルファ計算"""
        try:
            if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
                return 0.0, 0.0, 0.0, 0.0

            # 線形回帰
            slope, intercept, r_value, p_value, std_err = stats.linregress(benchmark_returns, portfolio_returns)

            beta = slope
            alpha = intercept * 252  # 年率化
            correlation = r_value
            r_squared = r_value ** 2

            return beta, alpha, correlation, r_squared

        except Exception as e:
            self.logger.error(f"ベータ・アルファ計算エラー: {e}")
            return 0.0, 0.0, 0.0, 0.0

    def _calculate_capture_ratios(self,
                                portfolio_returns: List[float],
                                benchmark_returns: List[float]) -> Tuple[float, float]:
        """アップ・ダウンキャプチャー計算"""
        try:
            if len(portfolio_returns) != len(benchmark_returns):
                return 0.0, 0.0

            up_market_days = [(p, b) for p, b in zip(portfolio_returns, benchmark_returns) if b > 0]
            down_market_days = [(p, b) for p, b in zip(portfolio_returns, benchmark_returns) if b < 0]

            # アップキャプチャー
            if up_market_days:
                up_portfolio_avg = np.mean([p for p, b in up_market_days])
                up_benchmark_avg = np.mean([b for p, b in up_market_days])
                up_capture = up_portfolio_avg / up_benchmark_avg if up_benchmark_avg != 0 else 0.0
            else:
                up_capture = 0.0

            # ダウンキャプチャー
            if down_market_days:
                down_portfolio_avg = np.mean([p for p, b in down_market_days])
                down_benchmark_avg = np.mean([b for p, b in down_market_days])
                down_capture = down_portfolio_avg / down_benchmark_avg if down_benchmark_avg != 0 else 0.0
            else:
                down_capture = 0.0

            return up_capture, down_capture

        except Exception as e:
            self.logger.error(f"キャプチャー比率計算エラー: {e}")
            return 0.0, 0.0

    def _calculate_market_performance(self,
                                    portfolio_returns: List[float],
                                    benchmark_returns: List[float]) -> Tuple[float, float]:
        """上昇・下落市場でのパフォーマンス"""
        try:
            up_market_portfolio = [p for p, b in zip(portfolio_returns, benchmark_returns) if b > 0]
            down_market_portfolio = [p for p, b in zip(portfolio_returns, benchmark_returns) if b < 0]

            up_performance = np.mean(up_market_portfolio) if up_market_portfolio else 0.0
            down_performance = np.mean(down_market_portfolio) if down_market_portfolio else 0.0

            return up_performance, down_performance

        except Exception as e:
            self.logger.error(f"市場パフォーマンス計算エラー: {e}")
            return 0.0, 0.0

    def _calculate_max_drawdown_duration(self) -> int:
        """最大ドローダウン期間計算"""
        if not self.daily_performance:
            return 0

        max_duration = 0
        current_duration = 0
        peak_value = 0

        for perf in self.daily_performance:
            if perf.portfolio_value > peak_value:
                peak_value = perf.portfolio_value
                current_duration = 0
            else:
                current_duration += 1
                max_duration = max(max_duration, current_duration)

        return max_duration

    def _filter_performance_by_date(self,
                                  start_date: Optional[datetime],
                                  end_date: Optional[datetime]) -> List[DailyPerformance]:
        """日付でパフォーマンスフィルタリング"""
        filtered = self.daily_performance

        if start_date:
            filtered = [p for p in filtered if p.date >= start_date]

        if end_date:
            filtered = [p for p in filtered if p.date <= end_date]

        return filtered

    def _update_metrics(self):
        """メトリクス更新"""
        self.current_metrics = self.get_period_performance()
        self.risk_metrics = self.get_risk_metrics()

    def _empty_period_performance(self) -> PeriodPerformance:
        """空の期間パフォーマンス"""
        return PeriodPerformance(
            start_date=datetime.now(), end_date=datetime.now(),
            total_return=0.0, annualized_return=0.0, volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            max_drawdown=0.0, var_95=0.0, var_99=0.0,
            skewness=0.0, kurtosis=0.0, beta=0.0, alpha=0.0,
            information_ratio=0.0, tracking_error=0.0,
            up_capture=0.0, down_capture=0.0
        )

    def _empty_benchmark_comparison(self) -> BenchmarkComparison:
        """空のベンチマーク比較"""
        return BenchmarkComparison(
            benchmark_symbol=self.benchmark_symbol,
            portfolio_return=0.0, benchmark_return=0.0, excess_return=0.0,
            correlation=0.0, beta=0.0, alpha=0.0, r_squared=0.0,
            tracking_error=0.0, information_ratio=0.0,
            up_market_performance=0.0, down_market_performance=0.0
        )

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """データクラスを辞書に変換"""
        if hasattr(obj, '__dataclass_fields__'):
            result = {}
            for field_name, field_value in obj.__dict__.items():
                if isinstance(field_value, datetime):
                    result[field_name] = field_value.isoformat()
                else:
                    result[field_name] = field_value
            return result
        return obj

    # チャート作成メソッド（簡略化）
    def _create_cumulative_returns_chart(self) -> str:
        """累積リターンチャート"""
        fig, ax = plt.subplots(figsize=(12, 6))
        dates = [p.date for p in self.daily_performance]
        cumulative_returns = [p.cumulative_return * 100 for p in self.daily_performance]

        ax.plot(dates, cumulative_returns, label='ポートフォリオ', linewidth=2)
        ax.set_title('累積リターン', fontweight='bold')
        ax.set_ylabel('リターン (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_monthly_returns_chart(self) -> str:
        """月次リターンチャート"""
        fig, ax = plt.subplots(figsize=(12, 6))
        monthly_data = self.get_monthly_summary()

        months = [m['month'] for m in monthly_data]
        returns = [m['total_return'] * 100 for m in monthly_data]

        colors = ['green' if r >= 0 else 'red' for r in returns]
        ax.bar(months, returns, color=colors, alpha=0.7)
        ax.set_title('月次リターン', fontweight='bold')
        ax.set_ylabel('リターン (%)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_risk_return_chart(self) -> str:
        """リスクリターン散布図"""
        fig, ax = plt.subplots(figsize=(10, 8))
        monthly_data = self.get_monthly_summary()

        returns = [m['total_return'] * 100 for m in monthly_data]
        volatilities = [m['volatility'] for m in monthly_data]

        ax.scatter(volatilities, returns, alpha=0.7, s=50)
        ax.set_xlabel('ボラティリティ')
        ax.set_ylabel('リターン (%)')
        ax.set_title('リスク・リターン分析', fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_drawdown_chart(self) -> str:
        """ドローダウンチャート"""
        fig, ax = plt.subplots(figsize=(12, 6))
        dates = [p.date for p in self.daily_performance]
        drawdowns = [-p.drawdown * 100 for p in self.daily_performance]

        ax.fill_between(dates, drawdowns, 0, alpha=0.3, color='red')
        ax.plot(dates, drawdowns, color='red', linewidth=1)
        ax.set_title('ドローダウン', fontweight='bold')
        ax.set_ylabel('ドローダウン (%)')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _create_rolling_sharpe_chart(self) -> str:
        """ローリングシャープレシオ"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 30日ローリングシャープレシオ計算
        window = 30
        if len(self.daily_returns) >= window:
            rolling_sharpe = []
            dates = []

            for i in range(window, len(self.daily_returns)):
                window_returns = self.daily_returns[i-window:i]
                sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252) if np.std(window_returns) > 0 else 0
                rolling_sharpe.append(sharpe)
                dates.append(self.daily_performance[i].date)

            ax.plot(dates, rolling_sharpe, linewidth=2)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        ax.set_title('30日ローリングシャープレシオ', fontweight='bold')
        ax.set_ylabel('シャープレシオ')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """図をBase64エンコード"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return image_base64