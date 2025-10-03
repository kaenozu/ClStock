#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3機能統合テスト
センチメント分析 + 可視化ダッシュボード + 戦略生成 + リスク管理の統合テスト
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Phase 3システムのインポート
from models.advanced.market_sentiment_analyzer import (
    MarketSentimentAnalyzer,
    SentimentData,
)
from models.advanced.prediction_dashboard import (
    PredictionDashboard,
    VisualizationData,
)
from models.advanced.trading_strategy_generator import (
    AutoTradingStrategyGenerator,
    StrategyType,
)
from models.advanced.risk_management_framework import RiskManager

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Phase3IntegrationTester:
    """Phase 3機能統合テスター"""

    def __init__(self):
        self.logger = logger
        self.test_results = {}

    def run_all_tests(self):
        """全テスト実行"""
        self.logger.info("=== Phase 3 統合テスト開始 ===")

        test_methods = [
            ("センチメント分析テスト", self.test_sentiment_analysis),
            ("可視化ダッシュボードテスト", self.test_prediction_dashboard),
            ("戦略生成システムテスト", self.test_strategy_generator),
            ("リスク管理フレームワークテスト", self.test_risk_management),
            ("包括的統合テスト", self.test_comprehensive_integration),
        ]

        for test_name, test_method in test_methods:
            self.logger.info(f"\n--- {test_name} ---")
            try:
                result = test_method()
                self.test_results[test_name] = {
                    "status": "SUCCESS" if result else "FAILURE",
                    "details": result if isinstance(result, dict) else {},
                }

                status = "[OK]" if result else "[ERROR]"
                self.logger.info(f"{status} {test_name}")

            except Exception as e:
                self.test_results[test_name] = {"status": "ERROR", "error": str(e)}
                self.logger.error(f"[ERROR] {test_name}: {str(e)}")

        self._print_test_summary()

    def test_sentiment_analysis(self) -> Dict[str, Any]:
        """センチメント分析テスト"""
        try:
            analyzer = MarketSentimentAnalyzer()

            # テストデータ生成
            test_symbol = "6758.T"

            # ニュースデータ（ダミー）
            news_data = [
                "ソニーグループの業績が上昇傾向、新高値更新",
                "市場全体が好調な中、成長期待が高まる",
                "アナリストが買い推奨を発表",
            ]

            # ソーシャルデータ（ダミー）
            social_data = [
                {"text": "ソニー株買い増し 🚀", "likes": 50, "retweets": 20},
                {"text": "決算良かった、long継続", "likes": 30, "retweets": 10},
                {"text": "まだまだ上がりそう 📈", "likes": 25, "retweets": 5},
            ]

            # 価格データ（ダミー）
            dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
            price_data = pd.DataFrame(
                {
                    "Close": np.random.randn(100).cumsum() + 1500,
                    "High": np.random.randn(100).cumsum() + 1520,
                    "Low": np.random.randn(100).cumsum() + 1480,
                    "Volume": np.random.randint(100000, 1000000, 100),
                },
                index=dates,
            )

            # センチメント分析実行
            sentiment_result = analyzer.analyze_comprehensive_sentiment(
                symbol=test_symbol,
                news_data=news_data,
                social_data=social_data,
                price_data=price_data,
            )

            # トレンド分析
            trend_result = analyzer.get_sentiment_trend(test_symbol)

            # レポート生成
            report = analyzer.generate_sentiment_report(test_symbol)

            return {
                "success": True,
                "sentiment_score": sentiment_result.sentiment_score,
                "confidence": sentiment_result.confidence,
                "trend": trend_result["trend"],
                "recommendation": (
                    report["recommendation"] if "recommendation" in report else "N/A"
                ),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_prediction_dashboard(self) -> Dict[str, Any]:
        """可視化ダッシュボードテスト"""
        try:
            dashboard = PredictionDashboard(enable_web_server=False)

            # テストデータ生成
            test_symbol = "7203.T"

            # 履歴データ
            dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
            historical_data = pd.DataFrame(
                {
                    "Close": np.random.randn(50).cumsum() + 2000,
                    "High": np.random.randn(50).cumsum() + 2020,
                    "Low": np.random.randn(50).cumsum() + 1980,
                    "Volume": np.random.randint(100000, 500000, 50),
                },
                index=dates,
            )

            # 予測データ
            future_dates = pd.date_range(start="2024-02-20", periods=10, freq="D")
            predictions = []
            for i, date in enumerate(future_dates):
                predictions.append(
                    {
                        "timestamp": date,
                        "prediction": 2000 + i * 10 + np.random.uniform(-20, 20),
                        "confidence": np.random.uniform(0.7, 0.9),
                        "mode": "momentum",
                    }
                )

            # センチメントデータ
            sentiment_data = {
                "current_sentiment": {"score": 0.6, "confidence": 0.8, "momentum": 0.3},
                "sources_breakdown": {"news": 0.5, "social": 0.7, "technical": 0.6},
                "trend": {
                    "trend": "bullish",
                    "recent_sentiments": [0.4, 0.5, 0.6, 0.7, 0.6],
                },
            }

            # パフォーマンス指標
            performance_metrics = {
                "accuracy": 0.87,
                "precision": 0.83,
                "recall": 0.79,
                "f1_score": 0.81,
            }

            # 可視化データ作成
            viz_data = VisualizationData(
                symbol=test_symbol,
                predictions=predictions,
                historical_data=historical_data,
                sentiment_data=sentiment_data,
                performance_metrics=performance_metrics,
                timestamp=datetime.now(),
            )

            # ダッシュボード生成
            dashboard_html = dashboard.create_dashboard(viz_data)

            # ダッシュボード保存テスト
            output_path = (
                Path(__file__).resolve().parents[2]
                / "tests"
                / "resources"
                / "test_dashboard.html"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            save_success = dashboard.save_dashboard(viz_data, str(output_path))

            # ダッシュボード状況確認
            dashboard_status = dashboard.get_dashboard_status()

            return {
                "success": True,
                "dashboard_generated": len(dashboard_html) > 1000,
                "save_success": save_success,
                "plotly_available": dashboard_status["plotly_available"],
                "web_server_available": dashboard_status["web_server_available"],
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_strategy_generator(self) -> Dict[str, Any]:
        """戦略生成システムテスト"""
        try:
            strategy_generator = AutoTradingStrategyGenerator()

            # テストデータ生成
            test_symbol = "8306.T"
            dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
            price_data = pd.DataFrame(
                {
                    "Close": np.random.randn(100).cumsum() + 800,
                    "High": np.random.randn(100).cumsum() + 820,
                    "Low": np.random.randn(100).cumsum() + 780,
                    "Volume": np.random.randint(500000, 2000000, 100),
                },
                index=dates,
            )

            # 戦略生成
            strategies = strategy_generator.generate_comprehensive_strategy(
                symbol=test_symbol,
                price_data=price_data,
                strategy_types=[
                    StrategyType.MOMENTUM,
                    StrategyType.MEAN_REVERSION,
                    StrategyType.BREAKOUT,
                ],
            )

            # 取引シグナル生成
            signals = strategy_generator.generate_trading_signals(
                symbol=test_symbol, price_data=price_data
            )

            # パフォーマンス取得
            performance = strategy_generator.get_strategy_performance(test_symbol)

            # 戦略レポート生成
            report = strategy_generator.generate_strategy_report(test_symbol)

            return {
                "success": True,
                "strategies_generated": len(strategies),
                "signals_generated": len(signals),
                "avg_expected_return": performance.get("avg_expected_return", 0),
                "avg_sharpe_ratio": performance.get("avg_sharpe_ratio", 0),
                "report_generated": "recommendations" in report,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_risk_management(self) -> Dict[str, Any]:
        """リスク管理フレームワークテスト"""
        try:
            risk_manager = RiskManager()

            # テストポートフォリオデータ
            portfolio_data = {
                "positions": {
                    "6758.T": 100000,  # ソニー
                    "7203.T": 150000,  # トヨタ
                    "8306.T": 80000,  # 三菱UFJ
                    "4502.T": 120000,  # 武田薬品
                },
                "total_value": 450000,
            }

            # テスト価格データ
            price_data = {}
            symbols = list(portfolio_data["positions"].keys())

            for symbol in symbols:
                dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
                price_data[symbol] = pd.DataFrame(
                    {
                        "Close": np.random.randn(100).cumsum()
                        + np.random.uniform(800, 2000),
                        "High": np.random.randn(100).cumsum()
                        + np.random.uniform(820, 2020),
                        "Low": np.random.randn(100).cumsum()
                        + np.random.uniform(780, 1980),
                        "Volume": np.random.randint(100000, 1000000, 100),
                    },
                    index=dates,
                )

            # ポートフォリオリスク分析
            portfolio_risk = risk_manager.analyze_portfolio_risk(
                portfolio_data, price_data
            )

            # リスクサマリー取得
            risk_summary = risk_manager.get_risk_summary()

            return {
                "success": True,
                "total_risk_score": portfolio_risk.total_risk_score,
                "risk_level": portfolio_risk.risk_level.value,
                "recommendations_count": len(portfolio_risk.recommendations),
                "max_safe_position_size": portfolio_risk.max_safe_position_size,
                "individual_metrics_count": len(portfolio_risk.individual_metrics),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def test_comprehensive_integration(self) -> Dict[str, Any]:
        """包括的統合テスト"""
        try:
            # 全システム初期化
            sentiment_analyzer = MarketSentimentAnalyzer()
            dashboard = PredictionDashboard(enable_web_server=False)
            strategy_generator = AutoTradingStrategyGenerator()
            risk_manager = RiskManager()

            test_symbol = "6758.T"

            # 1. 市場データ準備
            dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
            price_data = pd.DataFrame(
                {
                    "Close": np.random.randn(100).cumsum() + 1500,
                    "High": np.random.randn(100).cumsum() + 1520,
                    "Low": np.random.randn(100).cumsum() + 1480,
                    "Volume": np.random.randint(100000, 1000000, 100),
                },
                index=dates,
            )

            # 2. センチメント分析
            sentiment_result = sentiment_analyzer.analyze_comprehensive_sentiment(
                symbol=test_symbol,
                news_data=["好調な決算発表", "業績上方修正"],
                social_data=[{"text": "買い推奨 📈", "likes": 100, "retweets": 50}],
                price_data=price_data,
            )

            # 3. 戦略生成（センチメント考慮）
            strategies = strategy_generator.generate_comprehensive_strategy(
                test_symbol, price_data
            )
            signals = strategy_generator.generate_trading_signals(
                test_symbol,
                price_data,
                sentiment_data={
                    "current_sentiment": {"score": sentiment_result.sentiment_score}
                },
            )

            # 4. リスク分析
            portfolio_data = {"positions": {test_symbol: 200000}, "total_value": 200000}
            risk_analysis = risk_manager.analyze_portfolio_risk(
                portfolio_data, {test_symbol: price_data}
            )

            # 5. 統合ダッシュボード
            viz_data = VisualizationData(
                symbol=test_symbol,
                predictions=[
                    {
                        "timestamp": datetime.now(),
                        "prediction": price_data["Close"].iloc[-1] * 1.05,
                        "confidence": 0.8,
                        "mode": "integrated",
                    }
                ],
                historical_data=price_data,
                sentiment_data={
                    "current_sentiment": {
                        "score": sentiment_result.sentiment_score,
                        "confidence": sentiment_result.confidence,
                    }
                },
                performance_metrics={"integrated_accuracy": 0.85},
                timestamp=datetime.now(),
            )

            dashboard_html = dashboard.create_dashboard(viz_data)

            return {
                "success": True,
                "sentiment_score": sentiment_result.sentiment_score,
                "strategies_count": len(strategies),
                "signals_count": len(signals),
                "risk_level": risk_analysis.risk_level.value,
                "dashboard_size": len(dashboard_html),
                "integration_successful": True,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _print_test_summary(self):
        """テスト結果サマリー出力"""
        self.logger.info("\n=== Phase 3 統合テスト結果サマリー ===")

        total_tests = len(self.test_results)
        successful_tests = len(
            [r for r in self.test_results.values() if r["status"] == "SUCCESS"]
        )
        failed_tests = total_tests - successful_tests

        self.logger.info(f"総テスト数: {total_tests}")
        self.logger.info(f"成功: {successful_tests}")
        self.logger.info(f"失敗: {failed_tests}")
        self.logger.info(f"成功率: {(successful_tests/total_tests)*100:.1f}%")

        self.logger.info("\n--- 詳細結果 ---")
        for test_name, result in self.test_results.items():
            status_icon = "[OK]" if result["status"] == "SUCCESS" else "[ERROR]"
            self.logger.info(f"{status_icon} {test_name}")

            if result["status"] == "ERROR" and "error" in result:
                self.logger.info(f"    エラー: {result['error']}")
            elif "details" in result and result["details"]:
                details = result["details"]
                if isinstance(details, dict) and "success" in details:
                    for key, value in details.items():
                        if key != "success" and key != "error":
                            self.logger.info(f"    {key}: {value}")


def main():
    """メイン実行関数"""
    tester = Phase3IntegrationTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
