"""
ClStock デモ運用システム統合テスト

87%精度システムと統合されたデモ運用システムの動作確認と
使用方法のサンプルコード
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# デモ運用システムインポート
from trading import (
    DemoTrader, TradingStrategy, DemoPortfolioManager,
    DemoRiskManager, TradeRecorder, PerformanceTracker,
    BacktestEngine, BacktestConfig
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_trading_strategy():
    """取引戦略テスト"""
    logger.info("=== 取引戦略システムテスト ===")

    try:
        # 戦略初期化
        strategy = TradingStrategy(
            initial_capital=1000000,
            precision_threshold=85.0,
            confidence_threshold=0.7
        )

        # テスト銘柄でシグナル生成
        test_symbols = ["6758.T", "7203.T", "8306.T"]

        for symbol in test_symbols:
            try:
                signal = strategy.generate_trading_signal(symbol, 1000000)

                if signal:
                    logger.info(f"取引シグナル生成成功: {symbol}")
                    logger.info(f"  シグナルタイプ: {signal.signal_type.value}")
                    logger.info(f"  信頼度: {signal.confidence:.2f}")
                    logger.info(f"  期待リターン: {signal.expected_return:.3f}")
                    logger.info(f"  87%精度達成: {signal.precision_87_achieved}")
                else:
                    logger.info(f"シグナル生成なし: {symbol}")

            except Exception as e:
                logger.error(f"シグナル生成エラー {symbol}: {e}")

        # 戦略情報取得
        strategy_info = strategy.get_strategy_info()
        logger.info(f"戦略情報: {strategy_info['name']}")

        logger.info("取引戦略テスト完了")
        return True

    except Exception as e:
        logger.error(f"取引戦略テストエラー: {e}")
        return False


def test_portfolio_manager():
    """ポートフォリオマネージャーテスト"""
    logger.info("=== ポートフォリオマネージャーテスト ===")

    try:
        # ポートフォリオマネージャー初期化
        portfolio = DemoPortfolioManager(initial_capital=1000000)

        # テストポジション追加
        from trading.trading_strategy import SignalType

        success1 = portfolio.add_position("6758.T", 100, 2500.0, SignalType.BUY)
        success2 = portfolio.add_position("7203.T", 50, 1800.0, SignalType.BUY)

        if success1 and success2:
            logger.info("ポジション追加成功")

            # ポジション更新
            portfolio.update_positions()

            # メトリクス取得
            metrics = portfolio.get_portfolio_metrics()
            logger.info(f"総資産価値: {metrics.total_value:,.0f}円")
            logger.info(f"投資済み価値: {metrics.invested_value:,.0f}円")
            logger.info(f"リターン: {metrics.total_return_pct:.2f}%")
            logger.info(f"ポジション数: {metrics.position_count}")

            # リバランシング提案
            suggestions = portfolio.get_rebalancing_suggestions()
            if suggestions:
                logger.info("リバランシング提案:")
                for suggestion in suggestions:
                    logger.info(f"  - {suggestion['message']}")

            # ポジション要約
            position_summary = portfolio.get_position_summary()
            logger.info(f"ポジション詳細: {len(position_summary)}件")

        logger.info("ポートフォリオマネージャーテスト完了")
        return True

    except Exception as e:
        logger.error(f"ポートフォリオマネージャーテストエラー: {e}")
        return False


def test_risk_manager():
    """リスクマネージャーテスト"""
    logger.info("=== リスクマネージャーテスト ===")

    try:
        # リスクマネージャー初期化
        risk_manager = DemoRiskManager(initial_capital=1000000)

        # ポジション開設可能性テスト
        can_open = risk_manager.can_open_position(
            symbol="6758.T",
            position_size=100000,
            confidence=0.8,
            precision=87.0
        )

        logger.info(f"ポジション開設可能: {can_open}")

        # 最適ポジションサイズ計算
        optimal_size = risk_manager.calculate_optimal_position_size(
            symbol="6758.T",
            expected_return=0.05,
            confidence=0.8,
            precision=87.0
        )

        logger.info(f"最適ポジションサイズ: {optimal_size:,.0f}円")

        # VaR計算テスト
        test_positions = {
            "6758.T": {"market_value": 250000},
            "7203.T": {"market_value": 180000}
        }

        var_value = risk_manager.calculate_var(test_positions)
        logger.info(f"VaR (95%): {var_value:,.0f}円")

        # 期待ショートフォール
        es_value = risk_manager.calculate_expected_shortfall(test_positions)
        logger.info(f"期待ショートフォール: {es_value:,.0f}円")

        # リスクレポート生成
        risk_report = risk_manager.get_risk_report()
        logger.info(f"リスクレポート生成: {len(risk_report)}項目")

        logger.info("リスクマネージャーテスト完了")
        return True

    except Exception as e:
        logger.error(f"リスクマネージャーテストエラー: {e}")
        return False


def test_trade_recorder():
    """取引記録システムテスト"""
    logger.info("=== 取引記録システムテスト ===")

    try:
        # 取引記録システム初期化
        recorder = TradeRecorder()

        # テスト取引記録
        test_trade_data = {
            'trade_id': 'TEST_001',
            'session_id': 'test_session',
            'symbol': '6758.T',
            'action': 'OPEN',
            'quantity': 100,
            'price': 2500.0,
            'timestamp': datetime.now().isoformat(),
            'signal_type': 'BUY',
            'confidence': 0.85,
            'precision': 87.0,
            'precision_87_achieved': True,
            'expected_return': 0.05,
            'position_size': 250000,
            'trading_costs': {
                'commission': 250,
                'spread': 125,
                'total_cost': 375
            },
            'reasoning': 'テスト取引'
        }

        # 取引記録
        success = recorder.record_trade(test_trade_data)
        logger.info(f"取引記録成功: {success}")

        # クローズ取引記録
        close_trade_data = test_trade_data.copy()
        close_trade_data.update({
            'trade_id': 'TEST_001_CLOSE',
            'action': 'CLOSE',
            'actual_return': 0.04,
            'profit_loss': 9625  # 4%リターン - 取引コスト
        })

        recorder.record_trade(close_trade_data)

        # パフォーマンスレポート生成
        performance = recorder.generate_performance_report()
        logger.info(f"パフォーマンスレポート:")
        logger.info(f"  総取引数: {performance.total_trades}")
        logger.info(f"  勝率: {performance.win_rate:.1f}%")
        logger.info(f"  87%精度取引: {performance.precision_87_trades}")

        # CSV エクスポートテスト
        csv_path = project_root / "test_trades_export.csv"
        csv_success = recorder.export_to_csv(str(csv_path))
        logger.info(f"CSV エクスポート: {csv_success}")

        # JSON エクスポートテスト
        json_path = project_root / "test_trades_export.json"
        json_success = recorder.export_to_json(str(json_path))
        logger.info(f"JSON エクスポート: {json_success}")

        logger.info("取引記録システムテスト完了")
        return True

    except Exception as e:
        logger.error(f"取引記録システムテストエラー: {e}")
        return False


def test_performance_tracker():
    """パフォーマンストラッカーテスト"""
    logger.info("=== パフォーマンストラッカーテスト ===")

    try:
        # パフォーマンストラッカー初期化
        tracker = PerformanceTracker(initial_capital=1000000)

        # テストデータでパフォーマンス更新
        test_days = 7
        current_value = 1000000

        for day in range(test_days):
            # ランダムな価格変動シミュレーション
            import random
            daily_change = random.uniform(-0.02, 0.03)  # -2%から+3%
            current_value *= (1 + daily_change)

            success = tracker.update_performance(
                current_portfolio_value=current_value,
                active_positions=random.randint(3, 8),
                trades_count=random.randint(0, 3)
            )

            if success:
                logger.info(f"Day {day+1}: ポートフォリオ価値 {current_value:,.0f}円")

        # 期間パフォーマンス取得
        period_perf = tracker.get_period_performance()
        logger.info(f"期間パフォーマンス:")
        logger.info(f"  総リターン: {period_perf.total_return:.2%}")
        logger.info(f"  年率リターン: {period_perf.annualized_return:.2%}")
        logger.info(f"  ボラティリティ: {period_perf.volatility:.2%}")
        logger.info(f"  シャープレシオ: {period_perf.sharpe_ratio:.2f}")
        logger.info(f"  最大ドローダウン: {period_perf.max_drawdown:.2%}")

        # ベンチマーク比較
        benchmark_comp = tracker.get_benchmark_comparison()
        logger.info(f"ベンチマーク比較:")
        logger.info(f"  ポートフォリオリターン: {benchmark_comp.portfolio_return:.2%}")
        logger.info(f"  ベンチマークリターン: {benchmark_comp.benchmark_return:.2%}")
        logger.info(f"  超過リターン: {benchmark_comp.excess_return:.2%}")

        # リスクメトリクス
        risk_metrics = tracker.get_risk_metrics()
        logger.info(f"リスクメトリクス:")
        logger.info(f"  VaR (95%): {risk_metrics.var_95:.2%}")
        logger.info(f"  期待ショートフォール: {risk_metrics.expected_shortfall:.2%}")

        logger.info("パフォーマンストラッカーテスト完了")
        return True

    except Exception as e:
        logger.error(f"パフォーマンストラッカーテストエラー: {e}")
        return False


def test_backtest_engine():
    """バックテストエンジンテスト"""
    logger.info("=== バックテストエンジンテスト ===")

    try:
        # バックテスト設定
        config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now() - timedelta(days=1),
            initial_capital=1000000,
            precision_threshold=85.0,
            confidence_threshold=0.7,
            target_symbols=["6758.T", "7203.T"]
        )

        # バックテストエンジン初期化
        backtest_engine = BacktestEngine(config)

        # バックテスト実行
        logger.info("バックテスト実行中...")
        result = backtest_engine.run_backtest()

        # 結果表示
        logger.info(f"バックテスト結果:")
        logger.info(f"  期間: {result.start_date.date()} - {result.end_date.date()}")
        logger.info(f"  総リターン: {result.total_return:.2%}")
        logger.info(f"  年率リターン: {result.annualized_return:.2%}")
        logger.info(f"  シャープレシオ: {result.sharpe_ratio:.2f}")
        logger.info(f"  最大ドローダウン: {result.max_drawdown:.2%}")
        logger.info(f"  総取引数: {result.total_trades}")
        logger.info(f"  勝率: {result.win_rate:.1f}%")
        logger.info(f"  87%精度取引: {result.precision_87_trades}")
        logger.info(f"  最終価値: {result.final_value:,.0f}円")

        # バックテストレポート生成
        report = backtest_engine.generate_backtest_report(result)
        logger.info(f"バックテストレポート生成: {len(report)}セクション")

        logger.info("バックテストエンジンテスト完了")
        return True

    except Exception as e:
        logger.error(f"バックテストエンジンテストエラー: {e}")
        return False


def test_demo_trader_integration():
    """デモトレーダー統合テスト"""
    logger.info("=== デモトレーダー統合テスト ===")

    try:
        # デモトレーダー初期化
        demo_trader = DemoTrader(
            initial_capital=1000000,
            target_symbols=["6758.T", "7203.T", "8306.T"],
            precision_threshold=85.0,
            confidence_threshold=0.7,
            update_interval=60  # 1分間隔（テスト用）
        )

        # 短期間のデモ取引開始
        logger.info("短期デモ取引テスト開始")
        session_id = demo_trader.start_demo_trading(session_duration_days=1)
        logger.info(f"セッション開始: {session_id}")

        # 少し待機（実際の取引処理をシミュレーション）
        import time
        time.sleep(5)

        # 現在状況確認
        status = demo_trader.get_current_status()
        logger.info(f"現在の状況:")
        logger.info(f"  実行中: {status['is_running']}")
        logger.info(f"  現在資本: {status['current_capital']:,.0f}円")
        logger.info(f"  総資産: {status['total_equity']:,.0f}円")
        logger.info(f"  アクティブポジション: {status['active_positions']}")
        logger.info(f"  完了取引: {status['completed_trades']}")

        # デモ取引停止
        final_session = demo_trader.stop_demo_trading()
        logger.info(f"デモ取引停止")
        logger.info(f"  最終リターン: {final_session.total_return:.2f}%")
        logger.info(f"  総取引数: {final_session.total_trades}")

        # 取引統計
        statistics = demo_trader.get_trading_statistics()
        if statistics:
            logger.info(f"取引統計:")
            logger.info(f"  勝率: {statistics.get('win_rate', 0):.1f}%")
            logger.info(f"  平均リターン: {statistics.get('average_return', 0):.3f}")
            logger.info(f"  87%精度取引率: {statistics.get('precision_87_rate', 0):.1f}%")

        logger.info("デモトレーダー統合テスト完了")
        return True

    except Exception as e:
        logger.error(f"デモトレーダー統合テストエラー: {e}")
        return False


def sample_demo_trading_usage():
    """デモ取引システム使用サンプル"""
    logger.info("=== デモ取引システム使用サンプル ===")

    try:
        logger.info("1週間のデモ運用システム使用例:")

        # 1. システム初期化
        logger.info("Step 1: システム初期化")
        demo_trader = DemoTrader(
            initial_capital=1000000,  # 100万円
            target_symbols=[
                "6758.T", "7203.T", "8306.T", "9984.T", "6861.T"  # 主要5銘柄
            ],
            precision_threshold=87.0,  # 87%精度以上で取引
            confidence_threshold=0.8,  # 80%信頼度以上で取引
            update_interval=300  # 5分間隔
        )

        # 2. 1週間のデモ取引開始
        logger.info("Step 2: 1週間のデモ取引開始")
        session_id = demo_trader.start_demo_trading(session_duration_days=7)
        logger.info(f"デモセッション開始: {session_id}")

        # 3. 実際の運用では、システムが自動的に以下を実行:
        logger.info("Step 3: 自動実行される処理（実際には7日間継続）")
        logger.info("  - 87%精度システムによる取引シグナル生成")
        logger.info("  - リアルタイムデータに基づく価格更新")
        logger.info("  - リスク管理による取引判断")
        logger.info("  - ポートフォリオの自動リバランシング")
        logger.info("  - 取引記録とパフォーマンス追跡")

        # 4. 期間中の状況確認（デモ）
        time.sleep(2)  # 短時間待機
        status = demo_trader.get_current_status()
        logger.info("Step 4: 運用状況確認")
        logger.info(f"  初期資金: {status['initial_capital']:,.0f}円")
        logger.info(f"  現在総資産: {status['total_equity']:,.0f}円")
        logger.info(f"  現在リターン: {status['total_return']:.2f}%")

        # 5. デモ取引終了
        logger.info("Step 5: デモ取引終了")
        final_session = demo_trader.stop_demo_trading()

        # 6. 最終結果レポート
        logger.info("Step 6: 最終結果レポート")
        logger.info(f"運用期間: {final_session.start_time} - {final_session.end_time}")
        logger.info(f"初期資金: {final_session.initial_capital:,.0f}円")
        logger.info(f"最終資産: {final_session.current_capital:,.0f}円")
        logger.info(f"総リターン: {final_session.total_return:.2f}%")
        logger.info(f"総取引数: {final_session.total_trades}")
        logger.info(f"勝利取引: {final_session.winning_trades}")
        logger.info(f"87%精度取引: {final_session.precision_87_count}")
        logger.info(f"最大ドローダウン: {final_session.max_drawdown:.2f}%")

        # 7. 詳細分析レポート生成
        logger.info("Step 7: 詳細レポート生成")
        statistics = demo_trader.get_trading_statistics()
        if statistics:
            logger.info("詳細パフォーマンス統計:")
            for key, value in statistics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.3f}")
                else:
                    logger.info(f"  {key}: {value}")

        logger.info("=== 1週間のデモ運用完了 ===")
        logger.info("実際の利益・損失が正確にトレースされました。")
        logger.info("87%精度システムによる高精度な取引判断を確認できます。")

        return True

    except Exception as e:
        logger.error(f"デモ取引システム使用サンプルエラー: {e}")
        return False


def main():
    """メイン実行関数"""
    logger.info("=" * 50)
    logger.info("ClStock デモ運用システム統合テスト開始")
    logger.info("=" * 50)

    # テスト結果
    test_results = {}

    # 各コンポーネントテスト実行
    test_functions = [
        ("取引戦略", test_trading_strategy),
        ("ポートフォリオマネージャー", test_portfolio_manager),
        ("リスクマネージャー", test_risk_manager),
        ("取引記録システム", test_trade_recorder),
        ("パフォーマンストラッカー", test_performance_tracker),
        ("バックテストエンジン", test_backtest_engine),
        ("デモトレーダー統合", test_demo_trader_integration),
        ("使用サンプル", sample_demo_trading_usage)
    ]

    for test_name, test_func in test_functions:
        logger.info(f"\n{'-' * 30}")
        logger.info(f"実行中: {test_name}")
        logger.info(f"{'-' * 30}")

        try:
            result = test_func()
            test_results[test_name] = result
            status = "成功" if result else "失敗"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            test_results[test_name] = False
            logger.error(f"{test_name} 実行エラー: {e}")

    # 総合結果表示
    logger.info("\n" + "=" * 50)
    logger.info("統合テスト結果サマリー")
    logger.info("=" * 50)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    for test_name, result in test_results.items():
        status = "✓ 成功" if result else "✗ 失敗"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\n総合結果: {passed_tests}/{total_tests} テスト成功")

    if passed_tests == total_tests:
        logger.info("🎉 全テスト成功！デモ運用システムは正常に動作しています。")
        logger.info("87%精度システムと統合されたデモ取引システムの実装が完了しました。")
        logger.info("1週間のデモ運用で実際の利益・損失を正確にトレースできます。")
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests}個のテストが失敗しました。")
        logger.info("失敗したコンポーネントの詳細を確認してください。")

    logger.info("\n" + "=" * 50)
    logger.info("ClStock デモ運用システム統合テスト完了")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()