"""
ClStock リスクマネージャー

ポジションサイジング、ストップロス、VAR計算など
包括的なリスク管理システム
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats
import math

# 既存システム
from data.stock_data import StockDataProvider
from .trading_strategy import SignalType


@dataclass
class RiskLimit:
    """リスク制限"""

    max_position_size: float = 0.1  # 最大ポジションサイズ（資本比）
    max_sector_exposure: float = 0.3  # 最大セクターエクスポージャー
    max_single_loss: float = 0.02  # 単一取引最大損失（資本比）
    max_daily_loss: float = 0.05  # 1日最大損失
    max_drawdown: float = 0.2  # 最大ドローダウン
    max_correlation: float = 0.7  # 最大ポジション間相関
    min_diversification: int = 3  # 最小分散銘柄数
    var_confidence: float = 0.95  # VaR信頼水準


@dataclass
class PositionRisk:
    """ポジションリスク情報"""

    symbol: str
    current_exposure: float
    var_1day: float
    var_5day: float
    volatility: float
    beta: float
    sector: str
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float


@dataclass
class PortfolioRisk:
    """ポートフォリオリスク情報"""

    total_var: float
    marginal_var: Dict[str, float]
    component_var: Dict[str, float]
    expected_shortfall: float
    portfolio_volatility: float
    portfolio_beta: float
    diversification_ratio: float
    sector_concentrations: Dict[str, float]
    correlation_matrix: pd.DataFrame


class DemoRiskManager:
    """
    デモ取引リスクマネージャー

    87%精度システムと統合されたリスク管理
    """

    def __init__(
        self, initial_capital: float = 1000000, risk_limits: Optional[RiskLimit] = None
    ):
        """
        Args:
            initial_capital: 初期資本
            risk_limits: リスク制限設定
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimit()

        # データプロバイダー
        self.data_provider = StockDataProvider()

        # リスク追跡
        self.daily_pnl_history: List[float] = []
        self.position_history: List[Dict[str, float]] = []
        self.max_capital = initial_capital
        self.current_drawdown = 0.0

        # ポジション管理
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.sector_exposure: Dict[str, float] = defaultdict(float)

        # リスクメトリクス履歴
        self.var_history: List[Tuple[datetime, float]] = []
        self.risk_alerts: List[Dict[str, Any]] = []

        # セクター分類
        self.sector_map = self._load_sector_map()

        self.logger = logging.getLogger(__name__)

    def can_open_position(
        self,
        symbol: str,
        position_size: float,
        confidence: float = 0.0,
        precision: float = 85.0,
    ) -> bool:
        """
        新規ポジション開設可能性判定

        Args:
            symbol: 銘柄コード
            position_size: ポジションサイズ
            confidence: 取引信頼度
            precision: 予測精度

        Returns:
            開設可能フラグ
        """
        try:
            # 基本的な資金チェック
            if position_size > self.current_capital:
                self.logger.info(f"資金不足: {symbol}")
                return False

            # ポジションサイズ制限
            position_ratio = position_size / self.current_capital
            max_allowed = self._calculate_dynamic_position_limit(confidence, precision)

            if position_ratio > max_allowed:
                self.logger.info(
                    f"ポジションサイズ超過: {symbol} {position_ratio:.1%} > {max_allowed:.1%}"
                )
                return False

            # セクター集中度チェック
            sector = self.sector_map.get(symbol, "その他")
            current_sector_exposure = self.sector_exposure.get(sector, 0.0)
            new_sector_exposure = current_sector_exposure + position_ratio

            if new_sector_exposure > self.risk_limits.max_sector_exposure:
                self.logger.info(
                    f"セクター集中度超過: {sector} {new_sector_exposure:.1%}"
                )
                return False

            # 相関リスクチェック
            if not self._check_correlation_risk(symbol, position_size):
                self.logger.info(f"相関リスク超過: {symbol}")
                return False

            # 流動性リスクチェック
            if not self._check_liquidity_risk(symbol, position_size):
                self.logger.info(f"流動性リスク: {symbol}")
                return False

            # ドローダウンチェック
            if (
                self.current_drawdown > self.risk_limits.max_drawdown * 0.8
            ):  # 80%に達したら新規停止
                self.logger.info(f"ドローダウン制限: {self.current_drawdown:.1%}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"ポジション可否判定エラー {symbol}: {e}")
            return False

    def calculate_optimal_position_size(
        self,
        symbol: str,
        expected_return: float,
        confidence: float,
        precision: float,
        stop_loss_pct: float = 0.05,
    ) -> float:
        """
        最適ポジションサイズ計算（Kelly基準 + リスク調整）

        Args:
            symbol: 銘柄コード
            expected_return: 期待リターン
            confidence: 信頼度
            precision: 予測精度
            stop_loss_pct: ストップロス比率

        Returns:
            推奨ポジションサイズ（金額）
        """
        try:
            # Kelly基準による基本計算
            win_prob = confidence
            avg_win = abs(expected_return)
            avg_loss = stop_loss_pct

            # Kelly比率 = (勝率 × 平均利益 - 敗率 × 平均損失) / 平均利益
            kelly_ratio = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win

            # リスク調整
            # 87%精度達成時は積極的
            if precision >= 87.0:
                precision_multiplier = 1.2
            elif precision >= 85.0:
                precision_multiplier = 1.0
            else:
                precision_multiplier = 0.8

            # 信頼度調整
            confidence_multiplier = confidence

            # 市場環境調整
            market_multiplier = self._get_market_risk_multiplier()

            # 最終ポジションサイズ
            adjusted_kelly = (
                kelly_ratio
                * precision_multiplier
                * confidence_multiplier
                * market_multiplier
            )

            # 上限制限
            max_position_ratio = self._calculate_dynamic_position_limit(
                confidence, precision
            )
            position_ratio = min(abs(adjusted_kelly), max_position_ratio)

            # Kelly基準が負の場合は取引しない
            if kelly_ratio <= 0:
                return 0.0

            position_size = self.current_capital * position_ratio

            self.logger.info(
                f"最適ポジションサイズ計算: {symbol} "
                f"Kelly:{kelly_ratio:.3f} 調整後:{position_ratio:.3f} "
                f"サイズ:{position_size:,.0f}円"
            )

            return position_size

        except Exception as e:
            self.logger.error(f"ポジションサイズ計算エラー {symbol}: {e}")
            return 0.0

    def calculate_var(
        self,
        positions: Dict[str, Dict[str, Any]],
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ) -> float:
        """
        ポートフォリオVaR計算

        Args:
            positions: ポジション情報
            confidence_level: 信頼水準
            holding_period: 保有期間（日数）

        Returns:
            VaR値
        """
        try:
            if not positions:
                return 0.0

            # 各銘柄の収益率データ取得
            returns_data = {}
            for symbol, position in positions.items():
                try:
                    data = self.data_provider.get_stock_data(symbol, period="1y")
                    if data is not None and len(data) > 30:
                        returns = data["Close"].pct_change().dropna()
                        returns_data[symbol] = returns.tolist()
                except Exception:
                    continue

            if not returns_data:
                return 0.0

            # リターン行列作成
            min_length = min(len(returns) for returns in returns_data.values())
            if min_length < 30:
                return 0.0

            returns_matrix = np.array(
                [returns[-min_length:] for returns in returns_data.values()]
            ).T

            # ポートフォリオ重み計算
            total_value = sum(pos["market_value"] for pos in positions.values())
            weights = np.array(
                [
                    positions[symbol]["market_value"] / total_value
                    for symbol in returns_data.keys()
                ]
            )

            # ポートフォリオリターン計算
            portfolio_returns = np.dot(returns_matrix, weights)

            # VaR計算（ヒストリカル法）
            var_percentile = (1 - confidence_level) * 100
            var_1day = np.percentile(portfolio_returns, var_percentile)

            # 保有期間調整
            var_nday = var_1day * math.sqrt(holding_period)

            # 金額換算
            var_amount = abs(var_nday * total_value)

            return var_amount

        except Exception as e:
            self.logger.error(f"VaR計算エラー: {e}")
            return 0.0

    def calculate_expected_shortfall(
        self, positions: Dict[str, Dict[str, Any]], confidence_level: float = 0.95
    ) -> float:
        """
        期待ショートフォール（CVaR）計算

        Args:
            positions: ポジション情報
            confidence_level: 信頼水準

        Returns:
            期待ショートフォール
        """
        try:
            # VaR計算と同様の前処理
            returns_data = {}
            for symbol, position in positions.items():
                try:
                    data = self.data_provider.get_stock_data(symbol, period="1y")
                    if data is not None and len(data) > 30:
                        returns = data["Close"].pct_change().dropna()
                        returns_data[symbol] = returns.tolist()
                except Exception:
                    continue

            if not returns_data:
                return 0.0

            min_length = min(len(returns) for returns in returns_data.values())
            if min_length < 30:
                return 0.0

            returns_matrix = np.array(
                [returns[-min_length:] for returns in returns_data.values()]
            ).T

            total_value = sum(pos["market_value"] for pos in positions.values())
            weights = np.array(
                [
                    positions[symbol]["market_value"] / total_value
                    for symbol in returns_data.keys()
                ]
            )

            portfolio_returns = np.dot(returns_matrix, weights)

            # VaR閾値計算
            var_percentile = (1 - confidence_level) * 100
            var_threshold = np.percentile(portfolio_returns, var_percentile)

            # VaR以下のリターンの平均（期待ショートフォール）
            tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
            expected_shortfall = (
                np.mean(tail_returns) if len(tail_returns) > 0 else var_threshold
            )

            # 金額換算
            es_amount = abs(expected_shortfall * total_value)

            return es_amount

        except Exception as e:
            self.logger.error(f"期待ショートフォール計算エラー: {e}")
            return 0.0

    def check_stop_loss_conditions(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        position_type: SignalType,
        stop_loss_pct: float = 0.05,
    ) -> bool:
        """
        ストップロス条件チェック

        Args:
            symbol: 銘柄コード
            current_price: 現在価格
            entry_price: エントリー価格
            position_type: ポジションタイプ
            stop_loss_pct: ストップロス比率

        Returns:
            ストップロス実行フラグ
        """
        try:
            if position_type == SignalType.BUY:
                # 買いポジション
                loss_pct = (entry_price - current_price) / entry_price
                return loss_pct >= stop_loss_pct
            elif position_type == SignalType.SELL:
                # 売りポジション
                loss_pct = (current_price - entry_price) / entry_price
                return loss_pct >= stop_loss_pct

            return False

        except Exception as e:
            self.logger.error(f"ストップロスチェックエラー {symbol}: {e}")
            return False

    def update_risk_metrics(
        self, current_positions: Dict[str, Dict[str, Any]], daily_pnl: float
    ):
        """
        リスクメトリクス更新

        Args:
            current_positions: 現在のポジション
            daily_pnl: 日次損益
        """
        try:
            # PnL履歴更新
            self.daily_pnl_history.append(daily_pnl)
            if len(self.daily_pnl_history) > 252:  # 1年分保持
                self.daily_pnl_history.pop(0)

            # 最大資本更新
            current_total = self.current_capital + sum(
                pos.get("market_value", 0) for pos in current_positions.values()
            )
            if current_total > self.max_capital:
                self.max_capital = current_total

            # ドローダウン計算
            self.current_drawdown = (
                self.max_capital - current_total
            ) / self.max_capital

            # VaR計算・記録
            current_var = self.calculate_var(current_positions)
            self.var_history.append((datetime.now(), current_var))
            if len(self.var_history) > 30:  # 30日分保持
                self.var_history.pop(0)

            # セクターエクスポージャー更新
            self.sector_exposure.clear()
            total_value = sum(
                pos.get("market_value", 0) for pos in current_positions.values()
            )

            if total_value > 0:
                for symbol, position in current_positions.items():
                    sector = self.sector_map.get(symbol, "その他")
                    exposure = position.get("market_value", 0) / total_value
                    self.sector_exposure[sector] += exposure

            # リスクアラートチェック
            self._check_risk_alerts(current_positions)

        except Exception as e:
            self.logger.error(f"リスクメトリクス更新エラー: {e}")

    def get_risk_report(self) -> Dict[str, Any]:
        """リスクレポート生成"""
        try:
            current_var = self.var_history[-1][1] if self.var_history else 0.0

            return {
                "timestamp": datetime.now().isoformat(),
                "current_capital": self.current_capital,
                "max_capital": self.max_capital,
                "current_drawdown": self.current_drawdown,
                "var_1day_95": current_var,
                "var_5day_95": current_var * math.sqrt(5),
                "sector_exposure": dict(self.sector_exposure),
                "risk_limits": {
                    "max_position_size": self.risk_limits.max_position_size,
                    "max_sector_exposure": self.risk_limits.max_sector_exposure,
                    "max_drawdown": self.risk_limits.max_drawdown,
                },
                "active_alerts": self.risk_alerts[-10:],  # 最新10件
                "risk_utilization": {
                    "drawdown_usage": self.current_drawdown
                    / self.risk_limits.max_drawdown,
                    "max_sector_usage": (
                        max(self.sector_exposure.values())
                        / self.risk_limits.max_sector_exposure
                        if self.sector_exposure
                        else 0.0
                    ),
                },
                "daily_pnl_stats": {
                    "mean": (
                        np.mean(self.daily_pnl_history)
                        if self.daily_pnl_history
                        else 0.0
                    ),
                    "std": (
                        np.std(self.daily_pnl_history)
                        if self.daily_pnl_history
                        else 0.0
                    ),
                    "min": (
                        min(self.daily_pnl_history) if self.daily_pnl_history else 0.0
                    ),
                    "max": (
                        max(self.daily_pnl_history) if self.daily_pnl_history else 0.0
                    ),
                },
            }

        except Exception as e:
            self.logger.error(f"リスクレポート生成エラー: {e}")
            return {}

    def reset_risk_metrics(self):
        """リスクメトリクスリセット"""
        self.current_capital = self.initial_capital
        self.daily_pnl_history.clear()
        self.position_history.clear()
        self.max_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.positions.clear()
        self.sector_exposure.clear()
        self.var_history.clear()
        self.risk_alerts.clear()

    def _calculate_dynamic_position_limit(
        self, confidence: float, precision: float
    ) -> float:
        """動的ポジション制限計算"""
        base_limit = self.risk_limits.max_position_size

        # 87%精度達成時は制限緩和
        if precision >= 87.0:
            precision_multiplier = 1.5
        elif precision >= 85.0:
            precision_multiplier = 1.2
        else:
            precision_multiplier = 1.0

        # 信頼度による調整
        confidence_multiplier = 0.5 + confidence * 0.5

        # ドローダウンによる制限
        if self.current_drawdown > 0.1:  # 10%超過時は制限強化
            drawdown_multiplier = 0.5
        else:
            drawdown_multiplier = 1.0

        adjusted_limit = (
            base_limit
            * precision_multiplier
            * confidence_multiplier
            * drawdown_multiplier
        )

        return min(adjusted_limit, 0.15)  # 最大15%に制限

    def _check_correlation_risk(self, symbol: str, position_size: float) -> bool:
        """相関リスクチェック"""
        try:
            if len(self.positions) < 2:
                return True

            # 既存ポジションとの相関チェック（簡略化）
            sector = self.sector_map.get(symbol, "その他")
            same_sector_count = sum(
                1
                for pos_symbol in self.positions.keys()
                if self.sector_map.get(pos_symbol, "その他") == sector
            )

            # 同一セクター内ポジションが3つ以上の場合は制限
            return same_sector_count < 3

        except Exception as e:
            self.logger.error(f"相関リスクチェックエラー: {e}")
            return True

    def _check_liquidity_risk(self, symbol: str, position_size: float) -> bool:
        """流動性リスクチェック"""
        try:
            # 簡略化された流動性チェック
            data = self.data_provider.get_stock_data(symbol, period="1mo")
            if data is None or len(data) < 10:
                return False

            # 平均出来高チェック
            avg_volume = data["Volume"].mean()
            current_price = data["Close"].iloc[-1]
            avg_turnover = avg_volume * current_price

            # ポジションサイズが平均出来高の10%以下かチェック
            position_turnover = position_size
            return position_turnover <= avg_turnover * 0.1

        except Exception as e:
            self.logger.error(f"流動性リスクチェックエラー: {e}")
            return True

    def _get_market_risk_multiplier(self) -> float:
        """市場リスク乗数取得"""
        try:
            # 簡略化された市場リスク評価
            # 実際にはVIX指数や市場ボラティリティを使用
            if self.current_drawdown > 0.1:
                return 0.5  # 高リスク時は縮小
            elif self.current_drawdown < 0.05:
                return 1.0  # 低リスク時は通常
            else:
                return 0.8  # 中リスク時は少し縮小

        except Exception as e:
            self.logger.error(f"市場リスク乗数エラー: {e}")
            return 0.8

    def _check_risk_alerts(self, current_positions: Dict[str, Dict[str, Any]]):
        """リスクアラートチェック"""
        alerts = []

        try:
            # ドローダウンアラート
            if self.current_drawdown > self.risk_limits.max_drawdown * 0.8:
                alerts.append(
                    {
                        "type": "DRAWDOWN_WARNING",
                        "severity": "HIGH",
                        "message": f"ドローダウン {self.current_drawdown:.1%} が制限値の80%に接近",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # セクター集中度アラート
            for sector, exposure in self.sector_exposure.items():
                if exposure > self.risk_limits.max_sector_exposure * 0.9:
                    alerts.append(
                        {
                            "type": "SECTOR_CONCENTRATION",
                            "severity": "MEDIUM",
                            "message": f"{sector}セクターの集中度 {exposure:.1%} が制限値の90%に接近",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            # VaRアラート
            if self.var_history:
                current_var = self.var_history[-1][1]
                var_ratio = current_var / self.current_capital
                if var_ratio > 0.1:  # 資本の10%超過
                    alerts.append(
                        {
                            "type": "VAR_EXCEEDANCE",
                            "severity": "HIGH",
                            "message": f"VaR {var_ratio:.1%} が資本の10%を超過",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

            # 新しいアラートのみ追加
            for alert in alerts:
                if not any(
                    existing["message"] == alert["message"]
                    for existing in self.risk_alerts[-5:]
                ):
                    self.risk_alerts.append(alert)

        except Exception as e:
            self.logger.error(f"リスクアラートチェックエラー: {e}")

    def _load_sector_map(self) -> Dict[str, str]:
        """セクター分類マップ読み込み"""
        return {
            "6758.T": "電機",
            "7203.T": "自動車",
            "8306.T": "銀行",
            "9984.T": "小売",
            "6861.T": "電機",
            "4502.T": "化学",
            "6503.T": "電機",
            "7201.T": "自動車",
            "8001.T": "商社",
            "9022.T": "運輸",
            "1332.T": "建設",
            "1605.T": "建設",
            "1803.T": "建設",
            "1808.T": "建設",
            "1812.T": "建設",
            "1893.T": "建設",
            "2282.T": "食品",
            "3099.T": "小売",
            "4004.T": "化学",
            "4005.T": "化学",
            "4188.T": "電機",
            "4324.T": "電機",
            "4519.T": "化学",
            "4523.T": "化学",
            "5020.T": "石油",
            "5101.T": "繊維",
            "5401.T": "鉄鋼",
            "6504.T": "電機",
            "6701.T": "電機",
            "6770.T": "電機",
            "6902.T": "電機",
            "6954.T": "電機",
            "6981.T": "電機",
            "7261.T": "自動車",
            "7267.T": "自動車",
            "7269.T": "自動車",
            "7974.T": "ゲーム",
            "8002.T": "商社",
            "8031.T": "商社",
            "8035.T": "商社",
            "8058.T": "商社",
            "8306.T": "銀行",
            "8802.T": "不動産",
            "9101.T": "運輸",
            "9022.T": "運輸",
        }
