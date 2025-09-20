#!/usr/bin/env python3
"""
リアルタイム取引システムテスト
84.6%パターン検出の動作確認とパフォーマンステスト
"""

import pandas as pd
from realtime_trading_system import (
    RealTimeTradingSystem,
    Pattern846Detector,
    RealTimeDataProvider,
)
from data.stock_data import StockDataProvider
import time
from datetime import datetime


def test_pattern_detection():
    """84.6%パターン検出テスト"""
    print("=== 84.6%パターン検出テスト ===")

    detector = Pattern846Detector()
    data_provider = StockDataProvider()

    test_symbols = ["7203", "6758", "9434", "8306", "6861"]  # テスト用5銘柄

    results = []

    for symbol in test_symbols:
        try:
            print(f"\n{symbol}: {data_provider.jp_stock_codes.get(symbol, symbol)}")

            # 履歴データ取得
            data = data_provider.get_stock_data(symbol, "3mo")
            if len(data) < 50:
                print("  データ不足")
                continue

            # パターン検出
            pattern_result = detector.detect_846_pattern(data)

            print(f"  シグナル: {pattern_result['signal']}")
            print(f"  信頼度: {pattern_result['confidence']:.1%}")
            print(f"  理由: {pattern_result['reason']}")
            print(f"  現在価格: {pattern_result['current_price']:.0f}円")

            results.append(
                {
                    "symbol": symbol,
                    "signal": pattern_result["signal"],
                    "confidence": pattern_result["confidence"],
                    "reason": pattern_result["reason"],
                }
            )

        except Exception as e:
            print(f"  エラー: {e}")

    return results


def test_realtime_data():
    """リアルタイムデータ取得テスト"""
    print("\n=== リアルタイムデータ取得テスト ===")

    provider = RealTimeDataProvider()
    test_symbols = ["7203", "6758"]  # 2銘柄でテスト

    for symbol in test_symbols:
        try:
            print(f"\n{symbol}: データ取得中...")

            # リアルタイムデータ
            realtime_data = provider.get_realtime_data(symbol)
            if realtime_data is not None:
                latest_price = realtime_data["Close"].iloc[-1]
                latest_time = realtime_data.index[-1]
                print(f"  最新価格: {latest_price:.0f}円")
                print(f"  取得時刻: {latest_time}")
                print(f"  データ点数: {len(realtime_data)}")
            else:
                print("  データ取得失敗")

            # 履歴コンテキスト
            historical = provider.get_historical_context(symbol)
            if historical is not None:
                print(f"  履歴データ点数: {len(historical)}")
            else:
                print("  履歴データ取得失敗")

        except Exception as e:
            print(f"  エラー: {e}")


def test_system_initialization():
    """システム初期化テスト"""
    print("\n=== システム初期化テスト ===")

    try:
        system = RealTimeTradingSystem(initial_capital=1000000)

        status = system.get_status_report()

        print(f"システム状態: {status['status']}")
        print(f"初期資金: {status['initial_capital']:,.0f}円")
        print(f"現金: {status['current_cash']:,.0f}円")
        print(f"ポジション数: {status['positions_count']}")
        print(f"本日取引数: {status['daily_trades']}")

        print("\nシステム初期化成功")
        return True

    except Exception as e:
        print(f"システム初期化エラー: {e}")
        return False


def test_order_simulation():
    """注文シミュレーションテスト"""
    print("\n=== 注文シミュレーションテスト ===")

    system = RealTimeTradingSystem(initial_capital=1000000)

    # シミュレーション注文
    test_orders = [
        {
            "symbol": "7203",
            "action": "BUY",
            "size": 100,
            "price": 2500,
            "confidence": 0.87,
        },
        {
            "symbol": "6758",
            "action": "BUY",
            "size": 200,
            "price": 12000,
            "confidence": 0.85,
        },
    ]

    for order in test_orders:
        try:
            result = system.order_executor.execute_order(
                order["symbol"],
                order["action"],
                order["size"],
                order["price"],
                order["confidence"],
            )

            if result["status"] == "executed":
                # ポジション更新
                system.risk_manager.update_positions(
                    order["symbol"], order["action"], order["size"], order["price"]
                )

                print(
                    f"注文成功: {order['symbol']} {order['action']} {order['size']}株"
                )
            else:
                print(f"注文失敗: {order['symbol']}")

        except Exception as e:
            print(f"注文エラー: {e}")

    # 最終状況
    status = system.get_status_report()
    print(f"\n最終現金: {status['current_cash']:,.0f}円")
    print(f"ポジション数: {status['positions_count']}")


def performance_test():
    """パフォーマンステスト"""
    print("\n=== パフォーマンステスト ===")

    detector = Pattern846Detector()
    data_provider = StockDataProvider()

    # 5銘柄での処理時間測定
    test_symbols = ["7203", "6758", "9434", "8306", "6861"]

    start_time = time.time()

    for symbol in test_symbols:
        try:
            data = data_provider.get_stock_data(symbol, "3mo")
            if len(data) >= 50:
                result = detector.detect_846_pattern(data)
        except:
            continue

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"5銘柄処理時間: {processing_time:.2f}秒")
    print(f"1銘柄あたり: {processing_time/5:.2f}秒")

    # 60秒間隔での処理可能性評価
    symbols_per_minute = 60 / (processing_time / 5)
    print(f"1分間で処理可能銘柄数: {symbols_per_minute:.0f}銘柄")

    if symbols_per_minute >= 50:
        print("OK 全50銘柄のリアルタイム監視可能")
    else:
        print("注意 処理速度改善が必要")


def main():
    """テスト実行"""
    print("リアルタイム取引システム 総合テスト")
    print("=" * 50)
    print(f"テスト開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. システム初期化テスト
    if not test_system_initialization():
        print("NG システム初期化失敗")
        return

    # 2. リアルタイムデータ取得テスト
    test_realtime_data()

    # 3. パターン検出テスト
    pattern_results = test_pattern_detection()

    # 4. 注文シミュレーションテスト
    test_order_simulation()

    # 5. パフォーマンステスト
    performance_test()

    # 結果サマリー
    print("\n" + "=" * 50)
    print("テスト結果サマリー")
    print("=" * 50)

    signal_count = sum(1 for r in pattern_results if r["signal"] != 0)
    high_confidence_count = sum(1 for r in pattern_results if r["confidence"] >= 0.846)

    print(f"パターン検出結果: {len(pattern_results)}銘柄中")
    print(f"  シグナル検出: {signal_count}銘柄")
    print(f"  高信頼度(>=84.6%): {high_confidence_count}銘柄")

    if high_confidence_count > 0:
        print("OK 84.6%パターン検出成功")
        print("OK リアルタイム取引システム準備完了")
    else:
        print("注意 より多くのデータで再テスト推奨")

    print(f"\nテスト終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
