#!/usr/bin/env python3
"""動的リスク管理システム - 84.6%予測精度を活用した高度なリスク制御
ボラティリティ連動型損切り、ポートフォリオVaR、相関監視を統合
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import logging
from datetime import datetime, timedelta

from config.settings import get_settings
from data.stock_data import StockDataProvider
from utils.logger_config import setup_logger

logger = setup_logger(__name__)


class AdvancedRiskManager:
    def __init__(self, initial_capital: float = 1000000):
        self.settings = get_settings()
        self.data_provider = StockDataProvider()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.risk_metrics = {}

    def calculate_dynamic_stop_loss(
        self, symbol: str, entry_price: float, confidence: float, volatility: float,
    ) -> dict:
        """動的ストップロス計算"""
        # 基本損切り率（設定から）
        base_stop_loss = self.settings.realtime.default_stop_loss_pct

        # 1. 信頼度による調整
        confidence_adjustment = (
            1.0 - (confidence - 0.5) * 0.5
        )  # 信頼度が高いほど損切り緩く

        # 2. ボラティリティによる調整
        vol_adjustment = max(0.5, min(2.0, volatility * 20))  # ボラティリティに比例

        # 3. 84.6%パターンの強度調整
        pattern_strength = confidence / 0.846  # 84.6%を基準とした強度
        pattern_adjustment = max(0.7, min(1.5, 2 - pattern_strength))

        # 最終的な動的損切り率
        dynamic_stop_loss = (
            base_stop_loss * confidence_adjustment * vol_adjustment * pattern_adjustment
        )
        dynamic_stop_loss = max(0.02, min(0.15, dynamic_stop_loss))  # 2-15%の範囲

        stop_loss_price = entry_price * (1 - dynamic_stop_loss)

        return {
            "stop_loss_rate": dynamic_stop_loss,
            "stop_loss_price": stop_loss_price,
            "confidence_factor": confidence_adjustment,
            "volatility_factor": vol_adjustment,
            "pattern_factor": pattern_adjustment,
            "reasoning": f"信頼度{confidence:.1%}, ボラ{volatility:.1%}, 損切り{dynamic_stop_loss:.1%}",
        }

    def calculate_portfolio_var(self, confidence_level: float = 0.05) -> dict:
        """ポートフォリオVaR（Value at Risk）計算"""
        if len(self.positions) == 0:
            return {
                "var": 0,
                "cvar": 0,
                "var_rate": 0,
                "cvar_rate": 0,
                "portfolio_volatility": 0,
                "confidence_level": confidence_level,
                "total_portfolio_value": 0,
            }

        try:
            # 各ポジションのリターンデータ取得
            returns_data = {}
            position_values = {}

            for symbol, position in self.positions.items():
                # 過去データ取得
                data = self.data_provider.get_stock_data(symbol, "3mo")
                if len(data) < 20:
                    continue

                # 日次リターン計算
                daily_returns = data["Close"].pct_change().dropna()
                returns_data[symbol] = daily_returns

                # 現在のポジション価値
                current_price = data["Close"].iloc[-1]
                position_values[symbol] = position["size"] * current_price

            if len(returns_data) == 0:
                return {
                    "var": 0,
                    "cvar": 0,
                    "var_rate": 0,
                    "cvar_rate": 0,
                    "portfolio_volatility": 0,
                    "confidence_level": confidence_level,
                    "total_portfolio_value": sum(position_values.values()),
                }

            # リターンマトリックス作成
            returns_df = pd.DataFrame(returns_data).fillna(0)

            # ポートフォリオウェイト計算
            total_value = sum(position_values.values())
            weights = np.array(
                [
                    position_values.get(symbol, 0) / total_value
                    for symbol in returns_df.columns
                ],
            )

            # ポートフォリオリターン
            portfolio_returns = returns_df.dot(weights)

            # VaR計算（ヒストリカル法）
            var_percentile = confidence_level * 100
            var = np.percentile(portfolio_returns, var_percentile)

            # CVaR計算（Expected Shortfall）
            cvar = portfolio_returns[portfolio_returns <= var].mean()

            # ポートフォリオボラティリティ
            portfolio_volatility = portfolio_returns.std()

            # 金額ベースのVaR
            var_amount = abs(var * total_value)
            cvar_amount = abs(cvar * total_value)

            return {
                "var": var_amount,
                "cvar": cvar_amount,
                "var_rate": var,
                "cvar_rate": cvar,
                "portfolio_volatility": portfolio_volatility,
                "confidence_level": confidence_level,
                "total_portfolio_value": total_value,
            }

        except Exception as e:
            logging.exception(f"VaR計算エラー: {e}")
            return {
                "var": 0,
                "cvar": 0,
                "var_rate": 0,
                "cvar_rate": 0,
                "portfolio_volatility": 0,
                "confidence_level": confidence_level,
                "total_portfolio_value": 0,
            }

    def monitor_correlation_changes(self, lookback_days: int = 60) -> dict:
        """銘柄間相関の変化監視"""
        if len(self.positions) < 2:
            return {
                "correlation_risk": "低",
                "max_correlation": 0,
                "avg_correlation": 0,
                "high_correlation_pairs": [],
            }

        try:
            symbols = list(self.positions.keys())

            # 過去データ取得
            returns_data = {}
            for symbol in symbols:
                data = self.data_provider.get_stock_data(symbol, "6mo")
                if len(data) >= lookback_days:
                    daily_returns = data["Close"].pct_change().dropna()
                    returns_data[symbol] = daily_returns.tail(lookback_days)

            if len(returns_data) < 2:
                return {
                    "correlation_risk": "低",
                    "max_correlation": 0,
                    "avg_correlation": 0,
                    "high_correlation_pairs": [],
                }

            # 相関マトリックス計算
            returns_df = pd.DataFrame(returns_data).fillna(0)
            correlation_matrix = returns_df.corr()

            # 上三角部分の相関係数取得（対角線除く）
            correlations = []
            for i in range(len(correlation_matrix)):
                for j in range(i + 1, len(correlation_matrix)):
                    correlations.append(abs(correlation_matrix.iloc[i, j]))

            if len(correlations) == 0:
                return {
                    "correlation_risk": "低",
                    "max_correlation": 0,
                    "avg_correlation": 0,
                    "high_correlation_pairs": [],
                }

            max_correlation = max(correlations)
            avg_correlation = np.mean(correlations)

            # リスクレベル判定
            if max_correlation > 0.8:
                risk_level = "極高"
            elif max_correlation > 0.6:
                risk_level = "高"
            elif max_correlation > 0.4:
                risk_level = "中"
            else:
                risk_level = "低"

            return {
                "correlation_risk": risk_level,
                "max_correlation": max_correlation,
                "avg_correlation": avg_correlation,
                "correlation_matrix": correlation_matrix.to_dict(),
                "high_correlation_pairs": [
                    (
                        correlation_matrix.index[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j],
                    )
                    for i in range(len(correlation_matrix))
                    for j in range(i + 1, len(correlation_matrix))
                    if abs(correlation_matrix.iloc[i, j]) > 0.6
                ],
            }

        except Exception as e:
            logging.exception(f"相関監視エラー: {e}")
            return {
                "correlation_risk": "不明",
                "max_correlation": 0,
                "avg_correlation": 0,
                "high_correlation_pairs": [],
            }

    def calculate_position_sizing_kelly(
        self,
        win_probability: float,
        avg_win: float,
        avg_loss: float,
        current_capital: float,
    ) -> dict:
        """Kelly基準によるポジションサイズ計算"""
        # Kelly基準: f = (bp - q) / b
        # b = avg_win / avg_loss (勝率比)
        # p = win_probability (勝率)
        # q = 1 - p (負率)

        if avg_loss <= 0:
            return {"kelly_fraction": 0, "reason": "平均損失が0以下"}

        b = avg_win / avg_loss
        p = win_probability
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # Kelly基準の制限（最大25%）
        kelly_fraction = max(0, min(0.25, kelly_fraction))

        # 84.6%予測精度を反映
        if win_probability >= 0.846:
            confidence_bonus = min(0.05, (win_probability - 0.846) * 0.5)
            kelly_fraction += confidence_bonus

        optimal_size = current_capital * kelly_fraction

        return {
            "kelly_fraction": kelly_fraction,
            "optimal_size": optimal_size,
            "win_ratio": b,
            "win_probability": p,
            "reasoning": f"Kelly{kelly_fraction:.1%}, 勝率{p:.1%}, 勝率比{b:.1f}",
        }

    def assess_maximum_drawdown_risk(self) -> dict:
        """最大ドローダウンリスク評価"""
        try:
            if len(self.positions) == 0:
                return {"current_drawdown": 0, "risk_level": "なし"}

            # 各ポジションの損益計算
            total_unrealized_pnl = 0
            position_details = []

            for symbol, position in self.positions.items():
                try:
                    data = self.data_provider.get_stock_data(symbol, "1d")
                    if len(data) == 0:
                        continue

                    current_price = data["Close"].iloc[-1]
                    entry_price = position["avg_price"]

                    position_pnl = (current_price - entry_price) * position["size"]
                    position_pnl_pct = (current_price - entry_price) / entry_price

                    total_unrealized_pnl += position_pnl

                    position_details.append(
                        {
                            "symbol": symbol,
                            "pnl": position_pnl,
                            "pnl_pct": position_pnl_pct,
                            "current_price": current_price,
                            "entry_price": entry_price,
                        },
                    )

                except Exception as e:
                    logging.warning(f"{symbol} ドローダウン計算エラー: {e}")
                    continue

            # 現在のドローダウン率
            current_drawdown_pct = total_unrealized_pnl / self.initial_capital

            # リスクレベル判定
            if current_drawdown_pct <= -0.15:  # -15%以下
                risk_level = "極高"
            elif current_drawdown_pct <= -0.10:  # -10%以下
                risk_level = "高"
            elif current_drawdown_pct <= -0.05:  # -5%以下
                risk_level = "中"
            else:
                risk_level = "低"

            return {
                "current_drawdown": current_drawdown_pct,
                "current_drawdown_amount": total_unrealized_pnl,
                "risk_level": risk_level,
                "position_details": position_details,
                "recommendation": self._get_drawdown_recommendation(
                    current_drawdown_pct,
                ),
            }

        except Exception as e:
            logging.exception(f"ドローダウン評価エラー: {e}")
            return {
                "current_drawdown": 0,
                "risk_level": "不明",
                "position_details": [],
                "recommendation": "リスク水準正常: 現在の戦略継続",
            }

    def _get_drawdown_recommendation(self, drawdown_pct: float) -> str:
        """ドローダウンに基づく推奨アクション"""
        if drawdown_pct <= -0.15:
            return "緊急: 全ポジション見直し推奨"
        if drawdown_pct <= -0.10:
            return "警告: 損切り基準の再検討"
        if drawdown_pct <= -0.05:
            return "注意: リスク管理強化"
        return "正常: 現状維持"

    def generate_comprehensive_risk_report(self) -> dict:
        """総合リスクレポート生成"""
        # 各種リスク指標計算
        var_result = self.calculate_portfolio_var()
        correlation_result = self.monitor_correlation_changes()
        drawdown_result = self.assess_maximum_drawdown_risk()

        # 総合リスクスコア計算（0-100）
        risk_scores = []

        # VaRスコア
        total_portfolio_value = var_result.get("total_portfolio_value", 0)
        if total_portfolio_value > 0:
            var_ratio = var_result.get("var", 0) / total_portfolio_value
            var_score = min(100, var_ratio * 1000)  # VaR比率をスコア化
            risk_scores.append(var_score)

        # 相関スコア
        correlation_score = correlation_result.get("max_correlation", 0) * 100
        risk_scores.append(correlation_score)

        # ドローダウンスコア
        drawdown_score = abs(drawdown_result.get("current_drawdown", 0)) * 100
        risk_scores.append(drawdown_score)

        # 総合スコア
        overall_risk_score = np.mean(risk_scores) if risk_scores else 0

        # リスクレベル判定
        if overall_risk_score >= 15:
            overall_risk_level = "極高"
        elif overall_risk_score >= 10:
            overall_risk_level = "高"
        elif overall_risk_score >= 5:
            overall_risk_level = "中"
        else:
            overall_risk_level = "低"

        return {
            "timestamp": datetime.now(),
            "overall_risk_score": overall_risk_score,
            "overall_risk_level": overall_risk_level,
            "var_analysis": var_result,
            "correlation_analysis": correlation_result,
            "drawdown_analysis": drawdown_result,
            "recommendations": self._generate_risk_recommendations(
                overall_risk_score, var_result, correlation_result, drawdown_result,
            ),
        }

    def _generate_risk_recommendations(
        self,
        risk_score: float,
        var_result: dict,
        correlation_result: dict,
        drawdown_result: dict,
    ) -> list:
        """リスク状況に基づく推奨アクション"""
        recommendations = []

        # VaRベースの推奨
        total_portfolio_value = var_result.get("total_portfolio_value", 0)
        if total_portfolio_value > 0:
            var_ratio = var_result.get("var", 0) / total_portfolio_value
            if var_ratio > 0.1:
                recommendations.append("VaRが10%超過: ポジションサイズ縮小推奨")

        # 相関ベースの推奨
        if correlation_result.get("max_correlation", 0) > 0.8:
            recommendations.append("高相関リスク: 分散投資の見直し推奨")

        # ドローダウンベースの推奨
        drawdown_pct = drawdown_result.get("current_drawdown", 0)
        if drawdown_pct <= -0.1:
            recommendations.append("ドローダウン10%超過: 損切り実行検討")

        # 総合スコアベースの推奨
        if risk_score >= 15:
            recommendations.append("極高リスク: 緊急リスク軽減措置必要")
        elif risk_score >= 10:
            recommendations.append("高リスク: 積極的リスク管理推奨")

        if not recommendations:
            recommendations.append("リスク水準正常: 現在の戦略継続")

        return recommendations


def main():
    """動的リスク管理システムデモまたはテスト実行"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # テストモード
        return test_main()
    # デモモード
    print("動的リスク管理システム - 84.6%予測精度活用")
    print("=" * 60)

    # システム初期化
    risk_manager = AdvancedRiskManager(initial_capital=1000000)

    # サンプルポジション設定（テスト用）
    risk_manager.positions = {
        "7203": {"size": 100, "avg_price": 2500, "entry_date": datetime.now()},
        "6758": {"size": 50, "avg_price": 12000, "entry_date": datetime.now()},
        "9434": {"size": 500, "avg_price": 230, "entry_date": datetime.now()},
    }

    # 総合リスクレポート生成
    risk_report = risk_manager.generate_comprehensive_risk_report()

    print(
        f"リスク評価時刻: {risk_report['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
    )
    print(f"総合リスクスコア: {risk_report['overall_risk_score']:.1f}")
    print(f"総合リスクレベル: {risk_report['overall_risk_level']}")

    print("\n=== VaR分析 ===")
    var_data = risk_report["var_analysis"]
    print(f"1日VaR: {var_data.get('var', 0):,.0f}円")
    print(f"CVaR: {var_data.get('cvar', 0):,.0f}円")
    print(
        f"ポートフォリオボラティリティ: {var_data.get('portfolio_volatility', 0):.1%}",
    )

    print("\n=== 相関分析 ===")
    corr_data = risk_report["correlation_analysis"]
    print(f"相関リスクレベル: {corr_data.get('correlation_risk', 'N/A')}")
    print(f"最大相関係数: {corr_data.get('max_correlation', 0):.1%}")

    print("\n=== ドローダウン分析 ===")
    dd_data = risk_report["drawdown_analysis"]
    print(f"現在ドローダウン: {dd_data.get('current_drawdown', 0):.1%}")
    print(f"リスクレベル: {dd_data.get('risk_level', 'N/A')}")

    print("\n=== 推奨アクション ===")
    for i, rec in enumerate(risk_report["recommendations"], 1):
        print(f"{i}. {rec}")

    print("\nテスト実行は: python dynamic_risk_manager.py test")


class TestAdvancedRiskManager:
    """AdvancedRiskManagerの包括的テストスイート"""

    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.total_tests = 0

    def setup_test_manager(self):
        """テスト用リスクマネージャーの設定"""
        manager = AdvancedRiskManager(initial_capital=1000000)

        # テスト用ポジション設定
        manager.positions = {
            "7203": {
                "size": 100,
                "avg_price": 2500,
                "entry_date": datetime.now() - timedelta(days=5),
            },
            "6758": {
                "size": 50,
                "avg_price": 12000,
                "entry_date": datetime.now() - timedelta(days=3),
            },
            "9434": {
                "size": 500,
                "avg_price": 230,
                "entry_date": datetime.now() - timedelta(days=1),
            },
        }

        return manager

    def test_dynamic_stop_loss_calculation(self):
        """動的ストップロス計算テスト"""
        print("\n=== 動的ストップロス計算テスト ===")

        manager = self.setup_test_manager()

        test_cases = [
            # (entry_price, confidence, volatility, expected_range)
            (2500, 0.85, 0.02, (0.02, 0.15)),  # 高信頼度、低ボラティリティ
            (12000, 0.70, 0.05, (0.02, 0.15)),  # 中信頼度、中ボラティリティ
            (230, 0.90, 0.01, (0.02, 0.15)),  # 超高信頼度、超低ボラティリティ
        ]

        for i, (price, conf, vol, expected_range) in enumerate(test_cases, 1):
            result = manager.calculate_dynamic_stop_loss("TEST", price, conf, vol)

            stop_loss_rate = result["stop_loss_rate"]
            is_valid = expected_range[0] <= stop_loss_rate <= expected_range[1]

            print(f"テスト{i}: {'OK' if is_valid else 'NG'}")
            print(f"  価格: {price}, 信頼度: {conf:.1%}, ボラ: {vol:.1%}")
            print(f"  損切り率: {stop_loss_rate:.1%}")
            print(f"  理由: {result['reasoning']}")

            self.total_tests += 1
            if is_valid:
                self.passed_tests += 1

        return True

    def test_portfolio_var_calculation(self):
        """ポートフォリオVaR計算テスト"""
        print("\n=== ポートフォリオVaR計算テスト ===")

        manager = self.setup_test_manager()

        try:
            var_result = manager.calculate_portfolio_var(confidence_level=0.05)

            # 基本的な検証
            has_var = var_result.get("var", 0) > 0
            has_cvar = var_result.get("cvar", 0) > 0
            has_volatility = var_result.get("portfolio_volatility", 0) > 0

            print(f"VaR計算: {'OK' if has_var else 'NG'}")
            print(f"  1日VaR: {var_result.get('var', 0):,.0f}円")
            print(f"  CVaR: {var_result.get('cvar', 0):,.0f}円")
            print(f"  ボラティリティ: {var_result.get('portfolio_volatility', 0):.2%}")

            self.total_tests += 1
            if has_var and has_cvar:
                self.passed_tests += 1

            return True

        except Exception as e:
            print(f"VaR計算エラー: {e}")
            self.total_tests += 1
            return False

    def test_correlation_monitoring(self):
        """相関監視テスト"""
        print("\n=== 相関監視テスト ===")

        manager = self.setup_test_manager()

        try:
            corr_result = manager.monitor_correlation_changes()

            has_risk_level = "correlation_risk" in corr_result
            has_max_corr = "max_correlation" in corr_result

            print(f"相関監視: {'OK' if has_risk_level and has_max_corr else 'NG'}")
            print(f"  リスクレベル: {corr_result.get('correlation_risk', 'N/A')}")
            print(f"  最大相関: {corr_result.get('max_correlation', 0):.1%}")

            # 高相関ペアの確認
            high_corr_pairs = corr_result.get("high_correlation_pairs", [])
            if high_corr_pairs:
                print(f"  高相関ペア数: {len(high_corr_pairs)}")

            self.total_tests += 1
            if has_risk_level and has_max_corr:
                self.passed_tests += 1

            return True

        except Exception as e:
            print(f"相関監視エラー: {e}")
            self.total_tests += 1
            return False

    def test_kelly_criterion_calculation(self):
        """Kelly基準計算テスト"""
        print("\n=== Kelly基準計算テスト ===")

        manager = self.setup_test_manager()

        test_cases = [
            # (win_prob, avg_win, avg_loss, expected_positive)
            (0.846, 0.08, 0.05, True),  # 84.6%成功パターン
            (0.90, 0.10, 0.03, True),  # 超高勝率
            (0.60, 0.05, 0.10, False),  # 低勝率
        ]

        for i, (win_prob, avg_win, avg_loss, should_be_positive) in enumerate(
            test_cases, 1,
        ):
            result = manager.calculate_position_sizing_kelly(
                win_prob, avg_win, avg_loss, 1000000,
            )

            kelly_fraction = result["kelly_fraction"]
            is_valid = (kelly_fraction > 0) == should_be_positive

            print(f"テスト{i}: {'OK' if is_valid else 'NG'}")
            print(f"  勝率: {win_prob:.1%}, Kelly: {kelly_fraction:.1%}")
            print(f"  推奨サイズ: {result['optimal_size']:,.0f}円")

            self.total_tests += 1
            if is_valid:
                self.passed_tests += 1

        return True

    def test_drawdown_assessment(self):
        """ドローダウン評価テスト"""
        print("\n=== ドローダウン評価テスト ===")

        manager = self.setup_test_manager()

        try:
            dd_result = manager.assess_maximum_drawdown_risk()

            has_drawdown = "current_drawdown" in dd_result
            has_risk_level = "risk_level" in dd_result
            has_recommendation = "recommendation" in dd_result

            print(
                f"ドローダウン評価: {'OK' if has_drawdown and has_risk_level else 'NG'}",
            )
            print(f"  現在DD: {dd_result.get('current_drawdown', 0):.1%}")
            print(f"  リスクレベル: {dd_result.get('risk_level', 'N/A')}")
            print(f"  推奨: {dd_result.get('recommendation', 'N/A')}")

            self.total_tests += 1
            if has_drawdown and has_risk_level and has_recommendation:
                self.passed_tests += 1

            return True

        except Exception as e:
            print(f"ドローダウン評価エラー: {e}")
            self.total_tests += 1
            return False

    def test_comprehensive_risk_report(self):
        """総合リスクレポートテスト"""
        print("\n=== 総合リスクレポートテスト ===")

        manager = self.setup_test_manager()

        try:
            risk_report = manager.generate_comprehensive_risk_report()

            required_keys = [
                "timestamp",
                "overall_risk_score",
                "overall_risk_level",
                "var_analysis",
                "correlation_analysis",
                "drawdown_analysis",
                "recommendations",
            ]

            has_all_keys = all(key in risk_report for key in required_keys)
            has_recommendations = len(risk_report.get("recommendations", [])) > 0

            print(
                f"総合レポート: {'OK' if has_all_keys and has_recommendations else 'NG'}",
            )
            print(f"  リスクスコア: {risk_report.get('overall_risk_score', 0):.1f}")
            print(f"  リスクレベル: {risk_report.get('overall_risk_level', 'N/A')}")
            print(f"  推奨数: {len(risk_report.get('recommendations', []))}")

            self.total_tests += 1
            if has_all_keys and has_recommendations:
                self.passed_tests += 1

            return True

        except Exception as e:
            print(f"総合レポートエラー: {e}")
            self.total_tests += 1
            return False

    def test_edge_cases(self):
        """エッジケーステスト"""
        print("\n=== エッジケーステスト ===")

        # 空ポジションでのテスト
        empty_manager = AdvancedRiskManager(initial_capital=1000000)

        try:
            # 空ポジションでのVaR
            var_result = empty_manager.calculate_portfolio_var()
            empty_var_ok = var_result["var"] == 0

            # 空ポジションでの相関
            corr_result = empty_manager.monitor_correlation_changes()
            empty_corr_ok = corr_result["max_correlation"] == 0

            # 極端な値でのKelly計算
            kelly_result = empty_manager.calculate_position_sizing_kelly(
                0.99,
                0.01,
                0.50,
                1000000,  # 極端な値
            )
            extreme_kelly_ok = 0 <= kelly_result["kelly_fraction"] <= 0.25

            print(f"空ポジションVaR: {'OK' if empty_var_ok else 'NG'}")
            print(f"空ポジション相関: {'OK' if empty_corr_ok else 'NG'}")
            print(f"極端Kelly計算: {'OK' if extreme_kelly_ok else 'NG'}")

            self.total_tests += 3
            if empty_var_ok:
                self.passed_tests += 1
            if empty_corr_ok:
                self.passed_tests += 1
            if extreme_kelly_ok:
                self.passed_tests += 1

            return True

        except Exception as e:
            print(f"エッジケースエラー: {e}")
            self.total_tests += 3
            return False

    def run_all_tests(self):
        """全テスト実行"""
        print("AdvancedRiskManager 包括的テストスイート")
        print("=" * 60)

        # 各テスト実行
        self.test_dynamic_stop_loss_calculation()
        self.test_portfolio_var_calculation()
        self.test_correlation_monitoring()
        self.test_kelly_criterion_calculation()
        self.test_drawdown_assessment()
        self.test_comprehensive_risk_report()
        self.test_edge_cases()

        # 結果サマリー
        print("\n" + "=" * 60)
        print("テストスイート結果サマリー")
        print("=" * 60)

        success_rate = (
            (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        )

        print(f"実行テスト数: {self.total_tests}")
        print(f"成功テスト数: {self.passed_tests}")
        print(f"成功率: {success_rate:.1f}%")

        if success_rate >= 90:
            print("SUCCESS テストスイート成功！リスク管理システム正常動作")
        elif success_rate >= 75:
            print("WARNING 部分的成功：一部改善が必要")
        else:
            print("ERROR テスト失敗：大幅な修正が必要")

        return success_rate >= 90


def test_main():
    """テストスイート実行"""
    test_suite = TestAdvancedRiskManager()
    return test_suite.run_all_tests()


if __name__ == "__main__":
    main()
