#!/usr/bin/env python3
"""リスク管理フレームワーク
包括的なリスク分析・監視・制御システム
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class RiskLevel(Enum):
    """リスクレベル"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskType(Enum):
    """リスク種別"""

    MARKET_RISK = "market_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    VOLATILITY_RISK = "volatility_risk"
    CORRELATION_RISK = "correlation_risk"
    DRAWDOWN_RISK = "drawdown_risk"


@dataclass
class RiskMetric:
    """リスク指標"""

    metric_name: str
    current_value: float
    threshold_warning: float
    threshold_critical: float
    risk_level: RiskLevel
    description: str
    timestamp: datetime


@dataclass
class PortfolioRisk:
    """ポートフォリオリスク分析結果"""

    total_risk_score: float
    risk_level: RiskLevel
    individual_metrics: Dict[str, RiskMetric]
    risk_breakdown: Dict[str, float]
    recommendations: List[str]
    max_safe_position_size: float
    timestamp: datetime


class VolatilityAnalyzer:
    """ボラティリティ分析器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_historical_volatility(
        self, price_data: pd.Series, window: int = 30,
    ) -> float:
        """ヒストリカルボラティリティ計算"""
        try:
            returns = price_data.pct_change().dropna()
            volatility = returns.rolling(window=window).std().iloc[-1]
            annualized_volatility = volatility * np.sqrt(252)  # 年率換算
            return annualized_volatility
        except Exception as e:
            self.logger.error(f"Historical volatility calculation failed: {e!s}")
            return 0.3  # デフォルト値

    def calculate_garch_volatility(self, price_data: pd.Series) -> float:
        """GARCH ボラティリティ予測（簡略版）"""
        try:
            returns = price_data.pct_change().dropna()

            # 簡略GARCH: 指数加重移動平均
            decay_factor = 0.94
            weights = np.array([decay_factor**i for i in range(len(returns))])[::-1]
            weights = weights / weights.sum()

            variance = np.average(returns**2, weights=weights)
            volatility = np.sqrt(variance * 252)

            return volatility
        except Exception as e:
            self.logger.error(f"GARCH volatility calculation failed: {e!s}")
            return self.calculate_historical_volatility(price_data)

    def calculate_volatility_regime(self, price_data: pd.Series) -> str:
        """ボラティリティレジーム判定"""
        try:
            current_vol = self.calculate_historical_volatility(price_data, 30)
            long_term_vol = self.calculate_historical_volatility(price_data, 252)

            if current_vol > long_term_vol * 1.5:
                return "high_volatility"
            if current_vol < long_term_vol * 0.7:
                return "low_volatility"
            return "normal_volatility"
        except Exception:
            return "normal_volatility"


class VaRCalculator:
    """VaR（Value at Risk）計算器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_parametric_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ) -> float:
        """パラメトリックVaR計算"""
        try:
            mean_return = returns.mean()
            std_return = returns.std()

            # 正規分布を仮定
            from scipy import stats

            z_score = stats.norm.ppf(1 - confidence_level)

            var = -(mean_return + z_score * std_return) * np.sqrt(holding_period)
            return max(var, 0)  # 負の値は0にクリップ
        except Exception as e:
            self.logger.error(f"Parametric VaR calculation failed: {e!s}")
            return 0.05  # デフォルト5%

    def calculate_historical_var(
        self, returns: pd.Series, confidence_level: float = 0.95,
    ) -> float:
        """ヒストリカルVaR計算"""
        try:
            if len(returns) < 30:
                return self.calculate_parametric_var(returns, confidence_level)

            sorted_returns = returns.sort_values()
            var_index = int((1 - confidence_level) * len(sorted_returns))
            var = -sorted_returns.iloc[var_index]

            return max(var, 0)
        except Exception as e:
            self.logger.error(f"Historical VaR calculation failed: {e!s}")
            return 0.05

    def calculate_conditional_var(
        self, returns: pd.Series, confidence_level: float = 0.95,
    ) -> float:
        """条件付きVaR（CVaR/Expected Shortfall）計算"""
        try:
            var = self.calculate_historical_var(returns, confidence_level)

            # VaRを超える損失の平均
            tail_losses = returns[returns <= -var]
            if len(tail_losses) > 0:
                cvar = -tail_losses.mean()
            else:
                cvar = var * 1.3  # VaRの1.3倍を概算

            return cvar
        except Exception as e:
            self.logger.error(f"Conditional VaR calculation failed: {e!s}")
            return 0.07


class CorrelationAnalyzer:
    """相関分析器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_rolling_correlation(
        self, price_data: Dict[str, pd.Series], window: int = 60,
    ) -> pd.DataFrame:
        """ローリング相関計算"""
        try:
            # 価格データを結合
            combined_data = pd.DataFrame(price_data)
            returns = combined_data.pct_change().dropna()

            # ローリング相関
            rolling_corr = returns.rolling(window=window).corr()

            return rolling_corr
        except Exception as e:
            self.logger.error(f"Rolling correlation calculation failed: {e!s}")
            return pd.DataFrame()

    def detect_correlation_clusters(
        self, correlation_matrix: pd.DataFrame, threshold: float = 0.7,
    ) -> Dict[str, List[str]]:
        """相関クラスター検出"""
        try:
            clusters = {}
            processed = set()

            for symbol in correlation_matrix.columns:
                if symbol in processed:
                    continue

                # 高相関銘柄を見つける
                high_corr_symbols = []
                for other_symbol in correlation_matrix.columns:
                    if symbol != other_symbol:
                        corr_value = correlation_matrix.loc[symbol, other_symbol]
                        if abs(corr_value) > threshold:
                            high_corr_symbols.append(other_symbol)

                if high_corr_symbols:
                    cluster_name = f"cluster_{len(clusters) + 1}"
                    clusters[cluster_name] = [symbol] + high_corr_symbols
                    processed.update([symbol] + high_corr_symbols)

            return clusters
        except Exception as e:
            self.logger.error(f"Correlation cluster detection failed: {e!s}")
            return {}


class DrawdownAnalyzer:
    """ドローダウン分析器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """ドローダウン系列計算"""
        try:
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            return drawdown
        except Exception as e:
            self.logger.error(f"Drawdown series calculation failed: {e!s}")
            return pd.Series()

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """最大ドローダウン計算"""
        try:
            drawdown_series = self.calculate_drawdown_series(equity_curve)
            max_drawdown = drawdown_series.min()
            return abs(max_drawdown)
        except Exception as e:
            self.logger.error(f"Max drawdown calculation failed: {e!s}")
            return 0.0

    def calculate_drawdown_duration(self, equity_curve: pd.Series) -> Dict[str, int]:
        """ドローダウン期間計算"""
        try:
            drawdown_series = self.calculate_drawdown_series(equity_curve)

            # ドローダウン期間の検出
            in_drawdown = drawdown_series < 0
            drawdown_periods = []
            current_period = 0

            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                elif current_period > 0:
                    drawdown_periods.append(current_period)
                    current_period = 0

            if current_period > 0:
                drawdown_periods.append(current_period)

            return {
                "max_duration": max(drawdown_periods) if drawdown_periods else 0,
                "avg_duration": np.mean(drawdown_periods) if drawdown_periods else 0,
                "current_duration": current_period,
            }
        except Exception as e:
            self.logger.error(f"Drawdown duration calculation failed: {e!s}")
            return {"max_duration": 0, "avg_duration": 0, "current_duration": 0}


class LiquidityAnalyzer:
    """流動性分析器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_liquidity_metrics(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """流動性指標計算"""
        try:
            metrics = {}

            if "Volume" in price_data.columns:
                # 平均出来高
                avg_volume = price_data["Volume"].mean()
                metrics["avg_daily_volume"] = avg_volume

                # 出来高の変動係数
                volume_cv = (
                    price_data["Volume"].std() / avg_volume if avg_volume > 0 else 0
                )
                metrics["volume_volatility"] = volume_cv

                # Amihud流動性指標（価格インパクト）
                if "High" in price_data.columns and "Low" in price_data.columns:
                    daily_returns = price_data["Close"].pct_change().abs()
                    amihud = (daily_returns / price_data["Volume"]).mean()
                    metrics["amihud_illiquidity"] = amihud * 1000000  # スケール調整

            # ビッド・アスクスプレッド（簡略版：高値-安値）
            if "High" in price_data.columns and "Low" in price_data.columns:
                spread = (
                    (price_data["High"] - price_data["Low"]) / price_data["Close"]
                ).mean()
                metrics["avg_spread"] = spread

            return metrics
        except Exception as e:
            self.logger.error(f"Liquidity metrics calculation failed: {e!s}")
            return {}

    def assess_liquidity_risk(self, liquidity_metrics: Dict[str, float]) -> RiskLevel:
        """流動性リスク評価"""
        try:
            risk_score = 0

            # 出来高ボラティリティチェック
            volume_vol = liquidity_metrics.get("volume_volatility", 0)
            if volume_vol > 2.0:
                risk_score += 2
            elif volume_vol > 1.0:
                risk_score += 1

            # スプレッドチェック
            spread = liquidity_metrics.get("avg_spread", 0)
            if spread > 0.05:  # 5%以上
                risk_score += 2
            elif spread > 0.02:  # 2%以上
                risk_score += 1

            # Amihud指標チェック
            amihud = liquidity_metrics.get("amihud_illiquidity", 0)
            if amihud > 10:
                risk_score += 2
            elif amihud > 5:
                risk_score += 1

            if risk_score >= 4:
                return RiskLevel.CRITICAL
            if risk_score >= 3:
                return RiskLevel.HIGH
            if risk_score >= 2:
                return RiskLevel.MEDIUM
            return RiskLevel.LOW

        except Exception:
            return RiskLevel.MEDIUM


class RiskManager:
    """リスク管理フレームワーク

    特徴:
    - 包括的リスク分析
    - リアルタイム監視
    - 自動アラート
    - ポジションサイジング
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # 分析器初期化
        self.volatility_analyzer = VolatilityAnalyzer()
        self.var_calculator = VaRCalculator()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.drawdown_analyzer = DrawdownAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()

        # リスク設定
        self.risk_thresholds = {
            "max_portfolio_var": 0.15,  # 15%
            "max_single_position": 0.10,  # 10%
            "max_correlation": 0.80,  # 80%
            "max_drawdown": 0.20,  # 20%
            "min_liquidity_volume": 100000,  # 最小出来高
        }

        # リスク履歴
        self.risk_history = []

        self.logger.info("RiskManager initialized")

    def analyze_portfolio_risk(
        self, portfolio_data: Dict[str, Any], price_data: Dict[str, pd.DataFrame],
    ) -> PortfolioRisk:
        """ポートフォリオリスク分析"""
        try:
            risk_metrics = {}
            risk_scores = []

            # 1. 市場リスク（VaR）
            market_risk = self._analyze_market_risk(portfolio_data, price_data)
            risk_metrics["market_risk"] = market_risk
            risk_scores.append(self._risk_level_to_score(market_risk.risk_level))

            # 2. 流動性リスク
            liquidity_risk = self._analyze_liquidity_risk(portfolio_data, price_data)
            risk_metrics["liquidity_risk"] = liquidity_risk
            risk_scores.append(self._risk_level_to_score(liquidity_risk.risk_level))

            # 3. 集中リスク
            concentration_risk = self._analyze_concentration_risk(portfolio_data)
            risk_metrics["concentration_risk"] = concentration_risk
            risk_scores.append(self._risk_level_to_score(concentration_risk.risk_level))

            # 4. ボラティリティリスク
            volatility_risk = self._analyze_volatility_risk(portfolio_data, price_data)
            risk_metrics["volatility_risk"] = volatility_risk
            risk_scores.append(self._risk_level_to_score(volatility_risk.risk_level))

            # 5. 相関リスク
            correlation_risk = self._analyze_correlation_risk(
                portfolio_data, price_data,
            )
            risk_metrics["correlation_risk"] = correlation_risk
            risk_scores.append(self._risk_level_to_score(correlation_risk.risk_level))

            # 総合リスクスコア計算
            total_risk_score = np.mean(risk_scores)
            overall_risk_level = self._score_to_risk_level(total_risk_score)

            # リスク内訳
            risk_breakdown = {
                "market": self._risk_level_to_score(market_risk.risk_level),
                "liquidity": self._risk_level_to_score(liquidity_risk.risk_level),
                "concentration": self._risk_level_to_score(
                    concentration_risk.risk_level,
                ),
                "volatility": self._risk_level_to_score(volatility_risk.risk_level),
                "correlation": self._risk_level_to_score(correlation_risk.risk_level),
            }

            # 推奨事項生成
            recommendations = self._generate_risk_recommendations(
                risk_metrics, total_risk_score,
            )

            # 安全なポジションサイズ計算
            max_safe_position = self._calculate_max_safe_position_size(total_risk_score)

            portfolio_risk = PortfolioRisk(
                total_risk_score=total_risk_score,
                risk_level=overall_risk_level,
                individual_metrics=risk_metrics,
                risk_breakdown=risk_breakdown,
                recommendations=recommendations,
                max_safe_position_size=max_safe_position,
                timestamp=datetime.now(),
            )

            # 履歴更新
            self._update_risk_history(portfolio_risk)

            return portfolio_risk

        except Exception as e:
            self.logger.error(f"Portfolio risk analysis failed: {e!s}")
            return self._create_default_portfolio_risk()

    def _analyze_market_risk(
        self, portfolio_data: Dict[str, Any], price_data: Dict[str, pd.DataFrame],
    ) -> RiskMetric:
        """市場リスク分析"""
        try:
            # ポートフォリオリターン計算
            portfolio_returns = self._calculate_portfolio_returns(
                portfolio_data, price_data,
            )

            if len(portfolio_returns) < 30:
                return RiskMetric(
                    metric_name="Market Risk (VaR)",
                    current_value=0.05,
                    threshold_warning=0.10,
                    threshold_critical=0.15,
                    risk_level=RiskLevel.MEDIUM,
                    description="Insufficient data for accurate VaR calculation",
                    timestamp=datetime.now(),
                )

            # 95% VaR計算
            var_95 = self.var_calculator.calculate_historical_var(
                portfolio_returns, 0.95,
            )

            # リスクレベル判定
            if var_95 > self.risk_thresholds["max_portfolio_var"]:
                risk_level = RiskLevel.CRITICAL
            elif var_95 > self.risk_thresholds["max_portfolio_var"] * 0.8:
                risk_level = RiskLevel.HIGH
            elif var_95 > self.risk_thresholds["max_portfolio_var"] * 0.5:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW

            return RiskMetric(
                metric_name="Market Risk (VaR)",
                current_value=var_95,
                threshold_warning=self.risk_thresholds["max_portfolio_var"] * 0.8,
                threshold_critical=self.risk_thresholds["max_portfolio_var"],
                risk_level=risk_level,
                description=f"95% Value at Risk: {var_95:.2%}",
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Market risk analysis failed: {e!s}")
            return RiskMetric(
                metric_name="Market Risk (VaR)",
                current_value=0.10,
                threshold_warning=0.10,
                threshold_critical=0.15,
                risk_level=RiskLevel.MEDIUM,
                description="Market risk calculation error",
                timestamp=datetime.now(),
            )

    def _analyze_liquidity_risk(
        self, portfolio_data: Dict[str, Any], price_data: Dict[str, pd.DataFrame],
    ) -> RiskMetric:
        """流動性リスク分析"""
        try:
            liquidity_scores = []

            for symbol, position in portfolio_data.get("positions", {}).items():
                if symbol in price_data:
                    metrics = self.liquidity_analyzer.calculate_liquidity_metrics(
                        price_data[symbol],
                    )
                    risk_level = self.liquidity_analyzer.assess_liquidity_risk(metrics)
                    liquidity_scores.append(self._risk_level_to_score(risk_level))

            if not liquidity_scores:
                avg_liquidity_score = 2.0  # Medium risk
            else:
                avg_liquidity_score = np.mean(liquidity_scores)

            risk_level = self._score_to_risk_level(avg_liquidity_score)

            return RiskMetric(
                metric_name="Liquidity Risk",
                current_value=avg_liquidity_score,
                threshold_warning=2.5,
                threshold_critical=3.5,
                risk_level=risk_level,
                description=f"Average liquidity risk score: {avg_liquidity_score:.2f}",
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Liquidity risk analysis failed: {e!s}")
            return RiskMetric(
                metric_name="Liquidity Risk",
                current_value=2.0,
                threshold_warning=2.5,
                threshold_critical=3.5,
                risk_level=RiskLevel.MEDIUM,
                description="Liquidity risk calculation error",
                timestamp=datetime.now(),
            )

    def _analyze_concentration_risk(self, portfolio_data: Dict[str, Any]) -> RiskMetric:
        """集中リスク分析"""
        try:
            positions = portfolio_data.get("positions", {})
            total_value = sum(positions.values())

            if total_value == 0:
                return RiskMetric(
                    metric_name="Concentration Risk",
                    current_value=0.0,
                    threshold_warning=0.15,
                    threshold_critical=0.25,
                    risk_level=RiskLevel.LOW,
                    description="No positions",
                    timestamp=datetime.now(),
                )

            # 最大ポジション比率
            max_position_ratio = max(positions.values()) / total_value

            # ハーフィンダール指数（集中度指標）
            hhi = sum((value / total_value) ** 2 for value in positions.values())

            # リスクレベル判定
            concentration_score = max(max_position_ratio, hhi)

            if concentration_score > 0.25:
                risk_level = RiskLevel.CRITICAL
            elif concentration_score > 0.15:
                risk_level = RiskLevel.HIGH
            elif concentration_score > 0.10:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW

            return RiskMetric(
                metric_name="Concentration Risk",
                current_value=concentration_score,
                threshold_warning=0.15,
                threshold_critical=0.25,
                risk_level=risk_level,
                description=f"Max position: {max_position_ratio:.1%}, HHI: {hhi:.3f}",
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Concentration risk analysis failed: {e!s}")
            return RiskMetric(
                metric_name="Concentration Risk",
                current_value=0.10,
                threshold_warning=0.15,
                threshold_critical=0.25,
                risk_level=RiskLevel.MEDIUM,
                description="Concentration risk calculation error",
                timestamp=datetime.now(),
            )

    def _analyze_volatility_risk(
        self, portfolio_data: Dict[str, Any], price_data: Dict[str, pd.DataFrame],
    ) -> RiskMetric:
        """ボラティリティリスク分析"""
        try:
            volatilities = []

            for symbol, position in portfolio_data.get("positions", {}).items():
                if symbol in price_data and "Close" in price_data[symbol].columns:
                    vol = self.volatility_analyzer.calculate_historical_volatility(
                        price_data[symbol]["Close"],
                    )
                    volatilities.append(vol)

            if not volatilities:
                avg_volatility = 0.3  # デフォルト値
            else:
                avg_volatility = np.mean(volatilities)

            # リスクレベル判定
            if avg_volatility > 0.6:
                risk_level = RiskLevel.CRITICAL
            elif avg_volatility > 0.4:
                risk_level = RiskLevel.HIGH
            elif avg_volatility > 0.25:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW

            return RiskMetric(
                metric_name="Volatility Risk",
                current_value=avg_volatility,
                threshold_warning=0.4,
                threshold_critical=0.6,
                risk_level=risk_level,
                description=f"Average portfolio volatility: {avg_volatility:.1%}",
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Volatility risk analysis failed: {e!s}")
            return RiskMetric(
                metric_name="Volatility Risk",
                current_value=0.3,
                threshold_warning=0.4,
                threshold_critical=0.6,
                risk_level=RiskLevel.MEDIUM,
                description="Volatility risk calculation error",
                timestamp=datetime.now(),
            )

    def _analyze_correlation_risk(
        self, portfolio_data: Dict[str, Any], price_data: Dict[str, pd.DataFrame],
    ) -> RiskMetric:
        """相関リスク分析"""
        try:
            symbols = list(portfolio_data.get("positions", {}).keys())

            if len(symbols) < 2:
                return RiskMetric(
                    metric_name="Correlation Risk",
                    current_value=0.0,
                    threshold_warning=0.7,
                    threshold_critical=0.85,
                    risk_level=RiskLevel.LOW,
                    description="Insufficient positions for correlation analysis",
                    timestamp=datetime.now(),
                )

            # 価格データ準備
            price_series = {}
            for symbol in symbols:
                if symbol in price_data and "Close" in price_data[symbol].columns:
                    price_series[symbol] = price_data[symbol]["Close"]

            if len(price_series) < 2:
                return RiskMetric(
                    metric_name="Correlation Risk",
                    current_value=0.0,
                    threshold_warning=0.7,
                    threshold_critical=0.85,
                    risk_level=RiskLevel.LOW,
                    description="Insufficient price data for correlation analysis",
                    timestamp=datetime.now(),
                )

            # 相関行列計算
            corr_matrix = self.correlation_analyzer.calculate_rolling_correlation(
                price_series, 60,
            )

            if corr_matrix.empty:
                max_correlation = 0.5
            else:
                # 最新の相関行列を取得
                latest_corr = corr_matrix.iloc[-len(symbols) :, -len(symbols) :]

                # 対角成分を除いた最大相関
                np.fill_diagonal(latest_corr.values, np.nan)
                max_correlation = np.nanmax(np.abs(latest_corr.values))

            # リスクレベル判定
            if max_correlation > self.risk_thresholds["max_correlation"]:
                risk_level = RiskLevel.CRITICAL
            elif max_correlation > 0.7:
                risk_level = RiskLevel.HIGH
            elif max_correlation > 0.5:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW

            return RiskMetric(
                metric_name="Correlation Risk",
                current_value=max_correlation,
                threshold_warning=0.7,
                threshold_critical=self.risk_thresholds["max_correlation"],
                risk_level=risk_level,
                description=f"Maximum correlation: {max_correlation:.2f}",
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Correlation risk analysis failed: {e!s}")
            return RiskMetric(
                metric_name="Correlation Risk",
                current_value=0.5,
                threshold_warning=0.7,
                threshold_critical=0.85,
                risk_level=RiskLevel.MEDIUM,
                description="Correlation risk calculation error",
                timestamp=datetime.now(),
            )

    def _calculate_portfolio_returns(
        self, portfolio_data: Dict[str, Any], price_data: Dict[str, pd.DataFrame],
    ) -> pd.Series:
        """ポートフォリオリターン計算"""
        try:
            positions = portfolio_data.get("positions", {})
            total_value = sum(positions.values())

            if total_value == 0:
                return pd.Series()

            portfolio_returns = None

            for symbol, position_value in positions.items():
                if symbol in price_data and "Close" in price_data[symbol].columns:
                    weight = position_value / total_value
                    returns = price_data[symbol]["Close"].pct_change().dropna()
                    weighted_returns = returns * weight

                    if portfolio_returns is None:
                        portfolio_returns = weighted_returns
                    else:
                        portfolio_returns = portfolio_returns.add(
                            weighted_returns, fill_value=0,
                        )

            return portfolio_returns if portfolio_returns is not None else pd.Series()

        except Exception as e:
            self.logger.error(f"Portfolio returns calculation failed: {e!s}")
            return pd.Series()

    def _risk_level_to_score(self, risk_level: RiskLevel) -> float:
        """リスクレベルをスコアに変換"""
        mapping = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 2.0,
            RiskLevel.HIGH: 3.0,
            RiskLevel.CRITICAL: 4.0,
        }
        return mapping.get(risk_level, 2.0)

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """スコアをリスクレベルに変換"""
        if score >= 3.5:
            return RiskLevel.CRITICAL
        if score >= 2.5:
            return RiskLevel.HIGH
        if score >= 1.5:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _generate_risk_recommendations(
        self, risk_metrics: Dict[str, RiskMetric], total_risk_score: float,
    ) -> List[str]:
        """リスク推奨事項生成"""
        recommendations = []

        # 総合リスクスコアに基づく推奨
        if total_risk_score >= 3.5:
            recommendations.append(
                "【緊急】リスクレベルが危険域です。ポジション縮小を強く推奨",
            )
        elif total_risk_score >= 2.5:
            recommendations.append(
                "リスクレベルが高いです。ポジション調整を検討してください",
            )

        # 個別メトリクスに基づく推奨
        for metric_name, metric in risk_metrics.items():
            if metric.risk_level == RiskLevel.CRITICAL:
                if metric_name == "market_risk":
                    recommendations.append(
                        "市場リスクが危険レベル：ヘッジ取引やポジション縮小を検討",
                    )
                elif metric_name == "concentration_risk":
                    recommendations.append(
                        "集中リスクが高い：ポートフォリオの分散化が必要",
                    )
                elif metric_name == "correlation_risk":
                    recommendations.append(
                        "相関リスクが高い：異なるセクター・地域への分散投資を検討",
                    )
                elif metric_name == "liquidity_risk":
                    recommendations.append(
                        "流動性リスクが高い：より流動性の高い銘柄への切り替えを検討",
                    )
                elif metric_name == "volatility_risk":
                    recommendations.append(
                        "ボラティリティが危険レベル：防御的ポジション構築を推奨",
                    )

        if not recommendations:
            recommendations.append("現在のリスクレベルは管理可能な範囲内です")

        return recommendations

    def _calculate_max_safe_position_size(self, total_risk_score: float) -> float:
        """安全なポジションサイズ計算"""
        base_max_position = self.risk_thresholds["max_single_position"]

        # リスクスコアに基づく調整
        if total_risk_score >= 3.5:
            return base_max_position * 0.3  # 70%削減
        if total_risk_score >= 2.5:
            return base_max_position * 0.5  # 50%削減
        if total_risk_score >= 1.5:
            return base_max_position * 0.7  # 30%削減
        return base_max_position

    def _update_risk_history(self, portfolio_risk: PortfolioRisk):
        """リスク履歴更新"""
        self.risk_history.append(portfolio_risk)

        # 履歴サイズ制限（最新100件）
        if len(self.risk_history) > 100:
            self.risk_history = self.risk_history[-100:]

    def _create_default_portfolio_risk(self) -> PortfolioRisk:
        """デフォルトポートフォリオリスク作成"""
        return PortfolioRisk(
            total_risk_score=2.0,
            risk_level=RiskLevel.MEDIUM,
            individual_metrics={},
            risk_breakdown={},
            recommendations=["リスク分析でエラーが発生しました"],
            max_safe_position_size=0.05,
            timestamp=datetime.now(),
        )

    def get_risk_summary(self) -> Dict[str, Any]:
        """リスクサマリー取得"""
        if not self.risk_history:
            return {"status": "no_data"}

        latest_risk = self.risk_history[-1]

        return {
            "current_risk_level": latest_risk.risk_level.value,
            "total_risk_score": latest_risk.total_risk_score,
            "risk_breakdown": latest_risk.risk_breakdown,
            "recommendations": latest_risk.recommendations,
            "max_safe_position_size": latest_risk.max_safe_position_size,
            "risk_trend": self._calculate_risk_trend(),
            "timestamp": latest_risk.timestamp,
        }

    def _calculate_risk_trend(self) -> str:
        """リスクトレンド計算"""
        if len(self.risk_history) < 2:
            return "insufficient_data"

        recent_scores = [r.total_risk_score for r in self.risk_history[-5:]]

        if len(recent_scores) >= 3:
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

            if trend > 0.1:
                return "increasing"
            if trend < -0.1:
                return "decreasing"
            return "stable"

        return "stable"
