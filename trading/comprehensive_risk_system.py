"""
リスク管理コンポーネント統合モジュール
VaR、動的ポジショニング、ストップロス等を統合した総合リスク管理システム
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass

from trading.risk_management import (
    PortfolioRiskManager,
    DynamicRiskManager,
    RiskMetrics,
)
from trading.advanced_risk_management import AdvancedRiskManager, AdvancedRiskMetrics
from trading.position_sizing import AdaptivePositionSizer, PositionSizeResult
from trading.stop_loss_taking import SmartExitStrategy, ExitStrategyResult

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ComprehensiveRiskAssessment:
    """包括的リスク評価"""

    basic_metrics: RiskMetrics
    advanced_metrics: AdvancedRiskMetrics
    position_size: PositionSizeResult
    exit_strategy: ExitStrategyResult
    risk_level: str  # 'low', 'medium', 'high', 'very_high'
    risk_score: float  # 0.0-1.0のリスクスコア


class ComprehensiveRiskManager:
    """包括的リスク管理システム"""

    def __init__(self, initial_capital: float = 1000000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # 各サブコンポーネントの初期化
        self.basic_risk_manager = PortfolioRiskManager(initial_capital)
        self.advanced_risk_manager = AdvancedRiskManager(initial_capital)
        self.dynamic_risk_manager = DynamicRiskManager(initial_capital)
        self.position_sizer = AdaptivePositionSizer(initial_capital)
        self.exit_strategy_manager = SmartExitStrategy()

        # リスク閾値の設定
        self.risk_thresholds = {
            "var_95": -0.05,  # 95%VaRが-5%を超えると高いリスク
            "var_99": -0.10,  # 99%VaRが-10%を超えると非常に高いリスク
            "max_drawdown": -0.15,  # 最大ドローダウンが-15%を超えると高いリスク
            "volatility": 0.30,  # ボラティリティが30%を超えると高いリスク
        }

    def update_position(
        self, symbol: str, quantity: int, price: float, cash_flow: float = 0.0
    ):
        """保有状況を更新し、現金残高も更新"""
        self.basic_risk_manager.update_position(symbol, quantity, price, cash_flow)
        self.advanced_risk_manager.update_position(symbol, quantity, price, cash_flow)
        self.position_sizer.update_capital(
            self.basic_risk_manager.calculate_portfolio_value()
        )

    def update_portfolio_history(self):
        """ポートフォリオ履歴を更新"""
        self.basic_risk_manager.update_portfolio_history()
        current_value = self.basic_risk_manager.calculate_portfolio_value()
        self.advanced_risk_manager.historical_portfolio_values.append(current_value)

    def calculate_comprehensive_risk(
        self,
        symbol: str,
        price: float,
        returns_data: Optional[pd.DataFrame] = None,
        market_data: Optional[Dict] = None,
    ) -> ComprehensiveRiskAssessment:
        """
        包括的リスク評価を計算

        Args:
            symbol: 銘柄コード
            price: 現在価格
            returns_data: リターンデータ
            market_data: 市場データ (OHLCなど)

        Returns:
            ComprehensiveRiskAssessment
        """
        # 基本リスク指標の計算
        basic_metrics = self.basic_risk_manager.calculate_risk_metrics()

        # 高度リスク指標の計算
        advanced_metrics = self.advanced_risk_manager.calculate_advanced_risk_metrics()

        # ポジションサイズの計算（デフォルトパラメータで）
        position_size = self.position_sizer.calculate_optimal_position_size(
            symbol=symbol,
            price=price,
            win_rate=0.55,  # デフォルト勝率55%
            avg_win_rate=0.05,  # 平均利益率5%
            avg_loss_rate=-0.03,  # 平均損失率3%
            stop_loss_price=price * 0.97,  # 3%損切り
            volatility=0.25,  # 25%ボラティリティ
            correlation_with_portfolio=0.2,  # 20%相関
            strategy_confidence=0.7,  # 70%信頼度
        )

        # 退出戦略の計算
        if (
            market_data is not None
            and "High" in market_data
            and "Low" in market_data
            and "Close" in market_data
        ):
            # pandas DataFrame形式に変換
            if isinstance(market_data, dict):
                df = pd.DataFrame(market_data)
            else:
                df = market_data
        else:
            # ダミーの市場データを生成
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=30), periods=30, freq="D"
            )
            prices = [price * (1 + np.random.normal(0, 0.02)) for _ in range(30)]
            df = pd.DataFrame(
                {
                    "Close": prices,
                    "High": [p * 1.02 for p in prices],
                    "Low": [p * 0.98 for p in prices],
                },
                index=dates,
            )

        exit_strategy = self.exit_strategy_manager.calculate_exit_strategy(
            entry_price=price, df=df, direction="long", market_condition="normal"
        )

        # リスクレベルの計算
        risk_score = self._calculate_risk_score(basic_metrics, advanced_metrics)
        risk_level = self._risk_score_to_level(risk_score)

        return ComprehensiveRiskAssessment(
            basic_metrics=basic_metrics,
            advanced_metrics=advanced_metrics,
            position_size=position_size,
            exit_strategy=exit_strategy,
            risk_level=risk_level,
            risk_score=risk_score,
        )

    def _calculate_risk_score(
        self, basic_metrics: RiskMetrics, advanced_metrics: AdvancedRiskMetrics
    ) -> float:
        """
        総合リスクスコアを計算 (0.0-1.0)

        Args:
            basic_metrics: 基本リスク指標
            advanced_metrics: 高度リスク指標

        Returns:
            リスクスコア (0.0-1.0)
        """
        scores = []

        # VaRスコア (負の値が大きいほどリスクが高い)
        var_95_score = max(
            0.0,
            min(1.0, (basic_metrics.var_95 - self.risk_thresholds["var_95"]) / 0.10),
        )
        scores.append(var_95_score)

        var_99_score = max(
            0.0,
            min(1.0, (basic_metrics.var_99 - self.risk_thresholds["var_99"]) / 0.15),
        )
        scores.append(var_99_score)

        # 最大ドローダウンスコア
        max_drawdown_score = max(
            0.0,
            min(
                1.0,
                (basic_metrics.max_drawdown - self.risk_thresholds["max_drawdown"])
                / 0.20,
            ),
        )
        scores.append(max_drawdown_score)

        # ボラティリティスコア
        vol_score = max(
            0.0,
            min(
                1.0,
                (basic_metrics.volatility - self.risk_thresholds["volatility"]) / 0.20,
            ),
        )
        scores.append(vol_score)

        # シャープレシオスコア (低いほどリスクが高い)
        sharpe_score = max(0.0, min(1.0, (0.3 - basic_metrics.sharpe_ratio) / 0.5))
        scores.append(sharpe_score)

        # 期待ショートフォールスコア
        es_score = max(
            0.0, min(1.0, (advanced_metrics.expected_shortfall_95 + 0.05) / 0.15)
        )
        scores.append(es_score)

        # ストレステスト損失スコア
        stress_score = max(
            0.0, min(1.0, (advanced_metrics.stress_test_loss + 0.05) / 0.20)
        )
        scores.append(stress_score)

        # 平均を取るが、最大値に重みを置く（最もリスクの高い要素を重視）
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0

        # 最終リスクスコア = 平均 + (最大 - 平均) * 0.5
        final_score = avg_score + (max_score - avg_score) * 0.5

        return final_score

    def _risk_score_to_level(self, risk_score: float) -> str:
        """
        リスクスコアをリスクレベルに変換

        Args:
            risk_score: 0.0-1.0のリスクスコア

        Returns:
            リスクレベル文字列
        """
        if risk_score < 0.25:
            return "low"
        elif risk_score < 0.50:
            return "medium"
        elif risk_score < 0.75:
            return "high"
        else:
            return "very_high"

    def should_restrict_trading(self) -> bool:
        """取引制限が必要かどうか判断"""
        # 高度リスクマネージャーの上限チェックを使用
        checks = self.advanced_risk_manager.check_risk_limits()

        # いずれかのリスク上限を超えている場合は取引を制限
        for check_passed, _ in checks.values():
            if not check_passed:
                return True

        return False

    def get_risk_adjusted_parameters(
        self, base_stop_loss_pct: float = 0.05, base_take_profit_pct: float = 0.10
    ) -> Tuple[float, float]:
        """
        リスク調整された損切り・利確パラメータを取得

        Args:
            base_stop_loss_pct: 基本損切り率
            base_take_profit_pct: 基本利確率

        Returns:
            (調整後損切り率, 調整後利確率)
        """
        # 現在のリスク状況を取得
        basic_metrics = self.basic_risk_manager.calculate_risk_metrics()

        # ボラティリティが高い場合は損切りを広く、利確を狭く
        volatility_factor = (
            basic_metrics.volatility / 0.2 if basic_metrics.volatility > 0 else 1.0
        )  # 20%を基準

        # 最大ドローダウンが大きい場合はより保守的に
        drawdown_factor = (
            1.0 - (basic_metrics.max_drawdown / -0.20)
            if basic_metrics.max_drawdown < 0
            else 1.0
        )  # 20%ドローダウンを基準
        drawdown_factor = max(0.5, drawdown_factor)  # 最低でも50%

        # 調整後のパラメータ
        adjusted_stop_loss = (
            base_stop_loss_pct * (1 + volatility_factor * 0.5) * drawdown_factor
        )
        adjusted_take_profit = (
            base_take_profit_pct * (1 - volatility_factor * 0.2) / drawdown_factor
        )

        return adjusted_stop_loss, adjusted_take_profit

    def generate_risk_report(self) -> str:
        """リスクレポート生成"""
        return self.advanced_risk_manager.generate_risk_report()


class RiskManagedPortfolio:
    """リスク管理型ポートフォリオ"""

    def __init__(self, initial_capital: float = 1000000):
        self.risk_manager = ComprehensiveRiskManager(initial_capital)
        self.trade_history: List[Dict] = []

    def execute_trade_safely(
        self, symbol: str, action: str, price: float, market_data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        リスク管理下で取引を実行

        Args:
            symbol: 銘柄コード
            action: "buy" or "sell"
            price: 現在価格
            market_data: 市場データ

        Returns:
            取引結果またはNone (取引が制限された場合)
        """
        # リスク状況を評価
        risk_assessment = self.risk_manager.calculate_comprehensive_risk(
            symbol, price, market_data=market_data
        )

        # 取引制限が必要かチェック
        if self.risk_manager.should_restrict_trading():
            print(f"リスク上限超過のため取引を制限: {risk_assessment.risk_level}")
            return None

        # ポジションサイズ計算
        position_size = risk_assessment.position_size

        if action.lower() == "buy":
            # 購入の場合の制限
            max_investment = self.risk_manager.current_capital * 0.1  # 最大10%まで投資
            if position_size.position_value > max_investment:
                position_size.shares = int(max_investment / price)
                position_size.position_value = position_size.shares * price

            # 購入に必要な金額を計算
            purchase_amount = position_size.shares * price
            # 現金を減らしてポジションを更新
            self.risk_manager.update_position(
                symbol, position_size.shares, price, cash_flow=-purchase_amount
            )

            trade_result = {
                "symbol": symbol,
                "action": "buy",
                "shares": position_size.shares,
                "price": price,
                "value": position_size.position_value,
                "risk_level": risk_assessment.risk_level,
                "risk_score": risk_assessment.risk_score,
                "stop_loss": risk_assessment.exit_strategy.stop_loss_price,
                "take_profit": risk_assessment.exit_strategy.take_profit_price,
            }

        elif action.lower() == "sell":
            # 売却の場合は現在の保有状況を確認
            if symbol in self.risk_manager.basic_risk_manager.positions:
                current_pos = self.risk_manager.basic_risk_manager.positions[symbol]
                shares = current_pos["quantity"]

                # 売却による収入を計算
                sale_amount = shares * price
                # 現金を増やしてポジションを更新
                self.risk_manager.update_position(
                    symbol, 0, price, cash_flow=sale_amount
                )

                trade_result = {
                    "symbol": symbol,
                    "action": "sell",
                    "shares": shares,
                    "price": price,
                    "value": shares * price,
                    "risk_level": risk_assessment.risk_level,
                    "risk_score": risk_assessment.risk_score,
                }
            else:
                return None  # 保有していない銘柄は売却不可

        else:
            return None  # 無効なアクション

        # 取引履歴に追加
        self.trade_history.append(trade_result)

        # ポートフォリオ履歴を更新
        self.risk_manager.update_portfolio_history()

        return trade_result


# 使用例とテスト
if __name__ == "__main__":
    print("包括的リスク管理システムのテスト")

    # 包括的リスクマネージャーの初期化
    comp_risk_manager = ComprehensiveRiskManager(initial_capital=1000000)

    # サンプル市場データを生成
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    prices = 3000 + np.cumsum(np.random.normal(0, 20, 30))
    high_prices = prices + np.random.uniform(10, 30, 30)
    low_prices = prices - np.random.uniform(10, 30, 30)

    market_data = {"Close": prices, "High": high_prices, "Low": low_prices}

    # リスク評価を実行
    risk_assessment = comp_risk_manager.calculate_comprehensive_risk(
        symbol="7203", price=3200, market_data=market_data
    )

    print(f"リスク評価結果:")
    print(f"  リスクレベル: {risk_assessment.risk_level}")
    print(f"  リスクスコア: {risk_assessment.risk_score:.3f}")
    print(f"  推奨株数: {risk_assessment.position_size.shares}")
    print(f"  ポジション価値: {risk_assessment.position_size.position_value:,.0f}円")
    print(f"  損切り価格: {risk_assessment.exit_strategy.stop_loss_price:.0f}円")
    print(f"  利確価格: {risk_assessment.exit_strategy.take_profit_price:.0f}円")
    print(f"  戦略タイプ: {risk_assessment.exit_strategy.exit_strategy_type}")

    # リスク調整パラメータの取得
    stop_pct, profit_pct = comp_risk_manager.get_risk_adjusted_parameters()
    print(f"  リスク調整損切り率: {stop_pct:.3f} ({stop_pct*100:.1f}%)")
    print(f"  リスク調整利確率: {profit_pct:.3f} ({profit_pct*100:.1f}%)")

    # ポートフォリオテスト
    print(f"\nリスク管理型ポートフォリオテスト:")
    portfolio = RiskManagedPortfolio(initial_capital=1000000)

    # 購入取引テスト
    buy_result = portfolio.execute_trade_safely(
        symbol="7203", action="buy", price=3200, market_data=market_data
    )

    if buy_result:
        print(f"購入成功: {buy_result['shares']}株, 価値: {buy_result['value']:,.0f}円")
        print(
            f"リスクレベル: {buy_result['risk_level']}, スコア: {buy_result['risk_score']:.3f}"
        )

        # 売却取引テスト
        sell_result = portfolio.execute_trade_safely(
            symbol="7203",
            action="sell",
            price=3250,  # 価格上昇
            market_data=market_data,
        )

        if sell_result:
            print(
                f"売却成功: {sell_result['shares']}株, 価値: {sell_result['value']:,.0f}円"
            )

    # リスクレポート生成
    report = comp_risk_manager.generate_risk_report()
    print(f"\n{report}")
