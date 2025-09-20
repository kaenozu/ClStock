"""
ClStock ポートフォリオマネージャー

デモ取引におけるポートフォリオ管理と資産追跡システム
リアルタイム資産価値計算、リスク分析、分散投資管理
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import numpy as np
import json

# 既存システム
from data.stock_data import StockDataProvider
from .trading_strategy import SignalType


@dataclass
class Position:
    """ポジション情報"""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    position_type: SignalType
    entry_date: datetime
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    weight: float = 0.0  # ポートフォリオ内重み
    sector: str = ""
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioMetrics:
    """ポートフォリオ指標"""
    total_value: float
    cash_value: float
    invested_value: float
    total_return: float
    total_return_pct: float
    day_change: float
    day_change_pct: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    position_count: int
    concentration_risk: float  # 最大ポジション比率
    sector_concentration: Dict[str, float]
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float  # 日経平均に対するベータ


@dataclass
class RiskMetrics:
    """リスクメトリクス"""
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    expected_shortfall: float  # 期待ショートフォール
    portfolio_volatility: float
    correlation_risk: float
    sector_risk: Dict[str, float]
    individual_stock_risk: Dict[str, float]


class DemoPortfolioManager:
    """
    デモ取引ポートフォリオマネージャー

    仮想ポートフォリオの管理と分析を行う
    """

    def __init__(self, initial_capital: float = 1000000):
        """
        Args:
            initial_capital: 初期資本
        """
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        self.positions: Dict[str, Position] = {}

        # データプロバイダー
        self.data_provider = StockDataProvider()

        # 履歴データ（パフォーマンス追跡用）
        self.value_history: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []

        # セクター分類マップ
        self.sector_map = self._load_sector_map()

        # ベンチマーク（日経平均）
        self.benchmark_symbol = "^N225"
        self.benchmark_data = None

        self.logger = logging.getLogger(__name__)

    def add_position(self,
                    symbol: str,
                    quantity: int,
                    price: float,
                    position_type: SignalType) -> bool:
        """
        ポジション追加

        Args:
            symbol: 銘柄コード
            quantity: 数量
            price: 価格
            position_type: ポジションタイプ

        Returns:
            成功フラグ
        """
        try:
            position_value = quantity * price

            # 既存ポジションがある場合は平均化
            if symbol in self.positions:
                existing = self.positions[symbol]
                total_quantity = existing.quantity + quantity
                total_value = (existing.quantity * existing.average_price +
                             quantity * price)
                average_price = total_value / total_quantity

                self.positions[symbol].quantity = total_quantity
                self.positions[symbol].average_price = average_price
            else:
                # 新規ポジション
                sector = self.sector_map.get(symbol, "その他")
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    average_price=price,
                    current_price=price,
                    position_type=position_type,
                    entry_date=datetime.now(),
                    sector=sector
                )

            # キャッシュ更新
            self.current_cash -= position_value

            self.logger.info(f"ポジション追加: {symbol} {quantity}株 @{price:.2f}")
            return True

        except Exception as e:
            self.logger.error(f"ポジション追加エラー {symbol}: {e}")
            return False

    def remove_position(self, symbol: str, quantity: Optional[int] = None) -> bool:
        """
        ポジション削除

        Args:
            symbol: 銘柄コード
            quantity: 削除数量（Noneの場合は全量）

        Returns:
            成功フラグ
        """
        try:
            if symbol not in self.positions:
                return False

            position = self.positions[symbol]

            if quantity is None or quantity >= position.quantity:
                # 全量削除
                position_value = position.quantity * position.current_price
                self.current_cash += position_value
                del self.positions[symbol]
                self.logger.info(f"ポジション全量削除: {symbol}")
            else:
                # 部分削除
                position_value = quantity * position.current_price
                self.current_cash += position_value
                position.quantity -= quantity
                self.logger.info(f"ポジション部分削除: {symbol} {quantity}株")

            return True

        except Exception as e:
            self.logger.error(f"ポジション削除エラー {symbol}: {e}")
            return False

    def update_positions(self) -> bool:
        """
        全ポジションの価格更新

        Returns:
            成功フラグ
        """
        try:
            for symbol, position in self.positions.items():
                # 現在価格取得
                current_data = self.data_provider.get_stock_data(symbol, period="1d")
                if current_data is not None and len(current_data) > 0:
                    position.current_price = current_data['Close'].iloc[-1]
                    position.market_value = position.quantity * position.current_price

                    # 損益計算
                    if position.position_type == SignalType.BUY:
                        position.unrealized_pnl = (position.current_price - position.average_price) * position.quantity
                    else:  # SELL
                        position.unrealized_pnl = (position.average_price - position.current_price) * position.quantity

                    position.unrealized_pnl_pct = (position.unrealized_pnl /
                                                 (position.average_price * position.quantity) * 100)
                    position.last_updated = datetime.now()

            # ポートフォリオ重み計算
            self._calculate_weights()

            # 価値履歴記録
            total_value = self.get_total_value()
            self.value_history.append((datetime.now(), total_value))

            # 日次リターン計算
            if len(self.value_history) > 1:
                prev_value = self.value_history[-2][1]
                daily_return = (total_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)

            return True

        except Exception as e:
            self.logger.error(f"ポジション更新エラー: {e}")
            return False

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """ポートフォリオ指標取得"""
        try:
            total_value = self.get_total_value()
            invested_value = sum(pos.market_value for pos in self.positions.values())
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

            # リターン計算
            total_return = total_value - self.initial_capital
            total_return_pct = total_return / self.initial_capital * 100

            # 日次変化計算
            day_change = 0.0
            day_change_pct = 0.0
            if len(self.value_history) > 1:
                yesterday_value = self.value_history[-2][1] if len(self.value_history) > 1 else total_value
                day_change = total_value - yesterday_value
                day_change_pct = day_change / yesterday_value * 100

            # 集中リスク（最大ポジション比率）
            max_position_weight = max([pos.weight for pos in self.positions.values()], default=0.0)

            # セクター集中度
            sector_weights = defaultdict(float)
            for position in self.positions.values():
                sector_weights[position.sector] += position.weight

            # ボラティリティとシャープレシオ
            volatility = np.std(self.daily_returns) * np.sqrt(252) if self.daily_returns else 0.0
            avg_return = np.mean(self.daily_returns) if self.daily_returns else 0.0
            sharpe_ratio = avg_return / volatility * np.sqrt(252) if volatility > 0 else 0.0

            # 最大ドローダウン
            max_drawdown = self._calculate_max_drawdown()

            # ベータ計算
            beta = self._calculate_beta()

            return PortfolioMetrics(
                total_value=total_value,
                cash_value=self.current_cash,
                invested_value=invested_value,
                total_return=total_return,
                total_return_pct=total_return_pct,
                day_change=day_change,
                day_change_pct=day_change_pct,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl / invested_value * 100 if invested_value > 0 else 0.0,
                position_count=len(self.positions),
                concentration_risk=max_position_weight,
                sector_concentration=dict(sector_weights),
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                beta=beta
            )

        except Exception as e:
            self.logger.error(f"ポートフォリオ指標計算エラー: {e}")
            return PortfolioMetrics(
                total_value=self.current_cash,
                cash_value=self.current_cash,
                invested_value=0.0,
                total_return=0.0,
                total_return_pct=0.0,
                day_change=0.0,
                day_change_pct=0.0,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                position_count=0,
                concentration_risk=0.0,
                sector_concentration={},
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                beta=0.0
            )

    def calculate_risk_metrics(self) -> RiskMetrics:
        """リスクメトリクス計算"""
        try:
            if not self.daily_returns:
                return RiskMetrics(
                    var_95=0.0, var_99=0.0, expected_shortfall=0.0,
                    portfolio_volatility=0.0, correlation_risk=0.0,
                    sector_risk={}, individual_stock_risk={}
                )

            returns = np.array(self.daily_returns)

            # VaR計算
            var_95 = np.percentile(returns, 5) * self.get_total_value()
            var_99 = np.percentile(returns, 1) * self.get_total_value()

            # Expected Shortfall（CVaR）
            threshold_95 = np.percentile(returns, 5)
            tail_returns = returns[returns <= threshold_95]
            expected_shortfall = np.mean(tail_returns) * self.get_total_value() if len(tail_returns) > 0 else 0.0

            # ポートフォリオボラティリティ
            portfolio_volatility = np.std(returns) * np.sqrt(252)

            # 相関リスク（平均相関）
            correlation_risk = self._calculate_correlation_risk()

            # セクターリスク
            sector_risk = self._calculate_sector_risk()

            # 個別株リスク
            individual_stock_risk = self._calculate_individual_stock_risk()

            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                portfolio_volatility=portfolio_volatility,
                correlation_risk=correlation_risk,
                sector_risk=sector_risk,
                individual_stock_risk=individual_stock_risk
            )

        except Exception as e:
            self.logger.error(f"リスクメトリクス計算エラー: {e}")
            return RiskMetrics(
                var_95=0.0, var_99=0.0, expected_shortfall=0.0,
                portfolio_volatility=0.0, correlation_risk=0.0,
                sector_risk={}, individual_stock_risk={}
            )

    def get_rebalancing_suggestions(self) -> List[Dict[str, Any]]:
        """リバランシング提案"""
        suggestions = []

        try:
            metrics = self.get_portfolio_metrics()

            # 1. 集中リスクチェック
            if metrics.concentration_risk > 0.2:  # 20%超過
                max_position = max(self.positions.items(), key=lambda x: x[1].weight)
                suggestions.append({
                    'type': 'CONCENTRATION_RISK',
                    'priority': 'HIGH',
                    'message': f'{max_position[0]}の比率が{max_position[1].weight:.1%}と高すぎます',
                    'action': f'{max_position[0]}のポジションを縮小し、分散を図ってください'
                })

            # 2. セクター集中度チェック
            for sector, weight in metrics.sector_concentration.items():
                if weight > 0.4:  # 40%超過
                    suggestions.append({
                        'type': 'SECTOR_CONCENTRATION',
                        'priority': 'MEDIUM',
                        'message': f'{sector}セクターの比率が{weight:.1%}と高すぎます',
                        'action': f'{sector}以外のセクターへの投資を検討してください'
                    })

            # 3. ドローダウンチェック
            if metrics.max_drawdown > 0.15:  # 15%超過
                suggestions.append({
                    'type': 'DRAWDOWN_RISK',
                    'priority': 'HIGH',
                    'message': f'最大ドローダウンが{metrics.max_drawdown:.1%}と大きいです',
                    'action': '損切りルールの見直しやポジションサイズの縮小を検討してください'
                })

            # 4. キャッシュ比率チェック
            cash_ratio = metrics.cash_value / metrics.total_value
            if cash_ratio > 0.2:  # 20%超過
                suggestions.append({
                    'type': 'CASH_MANAGEMENT',
                    'priority': 'LOW',
                    'message': f'キャッシュ比率が{cash_ratio:.1%}と高いです',
                    'action': '投資機会を探して資金効率を向上させてください'
                })

            # 5. 低パフォーマンス銘柄
            for symbol, position in self.positions.items():
                if position.unrealized_pnl_pct < -10:  # 10%以上の含み損
                    suggestions.append({
                        'type': 'UNDERPERFORMING_STOCK',
                        'priority': 'MEDIUM',
                        'message': f'{symbol}が{position.unrealized_pnl_pct:.1f}%の含み損',
                        'action': f'{symbol}の投資判断を見直し、損切りを検討してください'
                    })

        except Exception as e:
            self.logger.error(f"リバランシング提案エラー: {e}")

        return suggestions

    def get_total_value(self) -> float:
        """総資産価値取得"""
        invested_value = sum(pos.market_value for pos in self.positions.values())
        return self.current_cash + invested_value

    def get_position_summary(self) -> List[Dict[str, Any]]:
        """ポジション要約取得"""
        return [
            {
                'symbol': pos.symbol,
                'quantity': pos.quantity,
                'average_price': pos.average_price,
                'current_price': pos.current_price,
                'market_value': pos.market_value,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'weight': pos.weight,
                'sector': pos.sector,
                'entry_date': pos.entry_date.isoformat(),
                'days_held': (datetime.now() - pos.entry_date).days
            }
            for pos in self.positions.values()
        ]

    def reset_portfolio(self):
        """ポートフォリオリセット"""
        self.current_cash = self.initial_capital
        self.positions.clear()
        self.value_history.clear()
        self.daily_returns.clear()
        self.logger.info("ポートフォリオリセット完了")

    def _calculate_weights(self):
        """ポートフォリオ重み計算"""
        total_value = self.get_total_value()
        if total_value > 0:
            for position in self.positions.values():
                position.weight = position.market_value / total_value

    def _calculate_max_drawdown(self) -> float:
        """最大ドローダウン計算"""
        if len(self.value_history) < 2:
            return 0.0

        values = [v[1] for v in self.value_history]
        peak = values[0]
        max_drawdown = 0.0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_beta(self) -> float:
        """ベータ計算"""
        try:
            if not self.daily_returns:
                return 0.0

            # ベンチマークデータ取得
            if self.benchmark_data is None:
                self.benchmark_data = self.data_provider.get_stock_data(
                    self.benchmark_symbol, period="1y"
                )

            if self.benchmark_data is None:
                return 0.0

            # ベンチマークリターン計算
            benchmark_returns = self.benchmark_data['Close'].pct_change().dropna().tolist()

            # リターン期間を合わせる
            min_length = min(len(self.daily_returns), len(benchmark_returns))
            if min_length < 10:
                return 0.0

            portfolio_returns = self.daily_returns[-min_length:]
            benchmark_returns = benchmark_returns[-min_length:]

            # ベータ計算（共分散 / ベンチマーク分散）
            covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)

            return covariance / benchmark_variance if benchmark_variance > 0 else 0.0

        except Exception as e:
            self.logger.error(f"ベータ計算エラー: {e}")
            return 0.0

    def _calculate_correlation_risk(self) -> float:
        """相関リスク計算"""
        # 簡略化された相関リスク指標
        # 実際にはポジション間の相関行列を計算する必要がある
        symbol_count = len(self.positions)
        if symbol_count <= 1:
            return 0.0

        # セクター内相関を考慮した簡易計算
        sector_groups = defaultdict(int)
        for position in self.positions.values():
            sector_groups[position.sector] += 1

        # セクター内銘柄数の比率が高いほど相関リスクが高い
        max_sector_count = max(sector_groups.values())
        correlation_risk = max_sector_count / symbol_count

        return correlation_risk

    def _calculate_sector_risk(self) -> Dict[str, float]:
        """セクターリスク計算"""
        sector_risk = {}
        sector_weights = defaultdict(float)

        for position in self.positions.values():
            sector_weights[position.sector] += position.weight

        for sector, weight in sector_weights.items():
            # セクター集中度をリスクとして計算
            sector_risk[sector] = weight

        return sector_risk

    def _calculate_individual_stock_risk(self) -> Dict[str, float]:
        """個別株リスク計算"""
        stock_risk = {}

        for symbol, position in self.positions.items():
            # ポジション重みと期間をリスク要因として計算
            days_held = (datetime.now() - position.entry_date).days
            time_risk = min(days_held / 30, 1.0)  # 30日で最大
            position_risk = position.weight

            stock_risk[symbol] = position_risk * (1 + time_risk)

        return stock_risk

    def _load_sector_map(self) -> Dict[str, str]:
        """セクター分類マップ読み込み"""
        # 簡易セクター分類
        return {
            "6758.T": "電機", "7203.T": "自動車", "8306.T": "銀行",
            "9984.T": "小売", "6861.T": "電機", "4502.T": "化学",
            "6503.T": "電機", "7201.T": "自動車", "8001.T": "商社",
            "9022.T": "運輸", "1332.T": "建設", "1605.T": "建設",
            "1803.T": "建設", "1808.T": "建設", "1812.T": "建設",
            "1893.T": "建設", "2282.T": "食品", "3099.T": "小売",
            "4004.T": "化学", "4005.T": "化学", "4188.T": "電機",
            "4324.T": "電機", "4519.T": "化学", "4523.T": "化学",
            "5020.T": "石油", "5101.T": "繊維", "5401.T": "鉄鋼",
            "6504.T": "電機", "6701.T": "電機", "6770.T": "電機",
            "6902.T": "電機", "6954.T": "電機", "6981.T": "電機",
            "7261.T": "自動車", "7267.T": "自動車", "7269.T": "自動車",
            "7974.T": "ゲーム", "8002.T": "商社", "8031.T": "商社",
            "8035.T": "商社", "8058.T": "商社", "8306.T": "銀行",
            "8802.T": "不動産", "9101.T": "運輸", "9022.T": "運輸"
        }

    def export_portfolio_data(self) -> Dict[str, Any]:
        """ポートフォリオデータエクスポート"""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(self.get_portfolio_metrics()),
            'positions': self.get_position_summary(),
            'risk_metrics': asdict(self.calculate_risk_metrics()),
            'rebalancing_suggestions': self.get_rebalancing_suggestions(),
            'value_history': [(dt.isoformat(), val) for dt, val in self.value_history],
            'daily_returns': self.daily_returns
        }


def asdict(obj) -> Dict[str, Any]:
    """dataclass を辞書に変換（datetime対応）"""
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name, field_value in obj.__dict__.items():
            if isinstance(field_value, datetime):
                result[field_name] = field_value.isoformat()
            elif isinstance(field_value, dict):
                result[field_name] = field_value
            else:
                result[field_name] = field_value
        return result
    return obj