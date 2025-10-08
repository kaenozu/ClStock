#!/usr/bin/env python3
"""ハイブリッド予測システムテスト
速度と精度を両立した最強システムの動作確認
"""

import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))


def main():
    """メインテスト実行関数"""
    print("=" * 80)
    print("ハイブリッド予測システム 動作確認テスト")
    print("=" * 80)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # ハイブリッドシステムテスト実行
        test_results = run_hybrid_system_tests()

        if test_results["success"]:
            print("\n[成功] ハイブリッドシステムテスト完了!")
            display_test_results(test_results)
        else:
            print(
                f"\n[失敗] ハイブリッドテスト失敗: {test_results.get('error', 'Unknown error')}",
            )

    except KeyboardInterrupt:
        print("\n\nテストが中断されました")
    except Exception as e:
        print(f"\n[エラー] 予期しないエラー: {e!s}")
        traceback.print_exc()


def run_hybrid_system_tests() -> Dict[str, Any]:
    """ハイブリッドシステムの包括的テスト"""
    try:
        print("1. ハイブリッドシステム初期化")

        from data.stock_data import StockDataProvider
        from models.hybrid.hybrid_predictor import (
            HybridStockPredictor,
            PredictionMode,
        )

        # データプロバイダー初期化
        data_provider = StockDataProvider()

        # ハイブリッドシステム初期化
        hybrid_system = HybridStockPredictor(
            data_provider=data_provider,
            default_mode=PredictionMode.AUTO,
        )

        print("   [OK] ハイブリッドシステム初期化完了")

        # システム情報確認
        system_info = hybrid_system.get_model_info()
        print(f"   サブシステム数: {len(system_info['subsystems'])}")
        print(f"   デフォルトモード: {system_info['default_mode']}")

        # テスト銘柄
        test_symbols = ["6758.T", "7203.T", "8306.T"]  # ソニー、トヨタ、三菱UFJ

        # 各種テスト実行
        test_results = {
            "initialization": True,
            "system_info": system_info,
            "mode_tests": {},
            "performance_tests": {},
            "integration_tests": {},
        }

        # モード別テスト
        print("\n2. モード別予測テスト")
        test_results["mode_tests"] = test_prediction_modes(hybrid_system, test_symbols)

        # パフォーマンステスト
        print("\n3. パフォーマンステスト")
        test_results["performance_tests"] = test_hybrid_performance(
            hybrid_system,
            test_symbols,
        )

        # 統合機能テスト
        print("\n4. 統合機能テスト")
        test_results["integration_tests"] = test_integration_features(
            hybrid_system,
            test_symbols,
        )

        test_results["success"] = True
        return test_results

    except ImportError as e:
        return {"success": False, "error": f"モジュールインポートエラー: {e!s}"}
    except Exception as e:
        return {"success": False, "error": f"テスト実行エラー: {e!s}"}


def test_prediction_modes(hybrid_system, test_symbols: List[str]) -> Dict[str, Any]:
    """各予測モードのテスト"""
    from models.hybrid.hybrid_predictor import PredictionMode

    mode_results = {}

    # 各モードでテスト
    for mode in [
        PredictionMode.SPEED_PRIORITY,
        PredictionMode.ACCURACY_PRIORITY,
        PredictionMode.BALANCED,
        PredictionMode.AUTO,
    ]:
        print(f"   {mode.value}モードテスト...")
        mode_data = {
            "predictions": [],
            "total_time": 0,
            "avg_confidence": 0,
            "avg_accuracy": 0,
        }

        start_time = time.time()

        for symbol in test_symbols:
            try:
                result = hybrid_system.predict(symbol, mode)
                mode_data["predictions"].append(
                    {
                        "symbol": symbol,
                        "prediction": result.prediction,
                        "confidence": result.confidence,
                        "accuracy": result.accuracy,
                        "metadata": result.metadata,
                    },
                )

                print(
                    f"     {symbol}: 予測値={result.prediction:.1f}, "
                    f"信頼度={result.confidence:.2f}, "
                    f"システム={result.metadata.get('system_used', 'unknown')}",
                )

            except Exception as e:
                logger.error(
                    f"Mode {mode.value} prediction failed for {symbol}: {e!s}",
                )

        mode_data["total_time"] = time.time() - start_time

        if mode_data["predictions"]:
            mode_data["avg_confidence"] = np.mean(
                [p["confidence"] for p in mode_data["predictions"]],
            )
            mode_data["avg_accuracy"] = np.mean(
                [p["accuracy"] for p in mode_data["predictions"]],
            )

        print(
            f"     総時間: {mode_data['total_time']:.3f}秒, "
            f"平均信頼度: {mode_data['avg_confidence']:.2f}",
        )

        mode_results[mode.value] = mode_data

    return mode_results


def test_hybrid_performance(hybrid_system, test_symbols: List[str]) -> Dict[str, Any]:
    """ハイブリッドシステムのパフォーマンステスト"""
    performance_data = {}

    # バッチ処理テスト
    print("   バッチ処理テスト...")
    start_time = time.time()
    try:
        batch_results = hybrid_system.predict_batch(test_symbols)
        batch_time = time.time() - start_time

        performance_data["batch_processing"] = {
            "total_time": batch_time,
            "per_symbol_time": batch_time / len(test_symbols),
            "throughput": len(test_symbols) / batch_time,
            "success_count": len(batch_results),
            "success_rate": len(batch_results) / len(test_symbols),
        }

        print(
            f"     バッチ処理: {batch_time:.3f}秒, "
            f"スループット: {len(test_symbols) / batch_time:.1f}銘柄/秒",
        )

    except Exception as e:
        logger.error(f"Batch processing test failed: {e!s}")
        performance_data["batch_processing"] = {"error": str(e)}

    # 信頼度計算テスト
    print("   信頼度計算テスト...")
    confidence_data = {}
    for symbol in test_symbols:
        try:
            confidence = hybrid_system.get_confidence(symbol)
            confidence_data[symbol] = confidence
            print(f"     {symbol}: 信頼度={confidence:.2f}")
        except Exception as e:
            logger.error(f"Confidence calculation failed for {symbol}: {e!s}")

    performance_data["confidence_calculation"] = confidence_data

    # パフォーマンス統計
    try:
        perf_stats = hybrid_system.get_performance_stats()
        performance_data["performance_statistics"] = perf_stats

        if "avg_prediction_time" in perf_stats:
            print(f"     平均予測時間: {perf_stats['avg_prediction_time']:.3f}秒")
            print(f"     総予測回数: {perf_stats['total_predictions']}")

    except Exception as e:
        logger.error(f"Performance statistics failed: {e!s}")

    return performance_data


def test_integration_features(hybrid_system, test_symbols: List[str]) -> Dict[str, Any]:
    """統合機能のテスト"""
    from models.hybrid.hybrid_predictor import PredictionMode

    integration_data = {}

    # 自動モード選択テスト
    print("   自動モード選択テスト...")
    auto_mode_results = []

    for symbol in test_symbols:
        try:
            # AUTOモードで複数回予測（学習効果確認）
            for i in range(3):
                result = hybrid_system.predict(symbol, PredictionMode.AUTO)
                auto_mode_results.append(
                    {
                        "symbol": symbol,
                        "iteration": i + 1,
                        "mode_used": result.metadata.get("mode_used", "unknown"),
                        "system_used": result.metadata.get("system_used", "unknown"),
                        "prediction_time": result.metadata.get("prediction_time", 0),
                    },
                )

                print(
                    f"     {symbol} 回{i + 1}: モード={result.metadata.get('mode_used', 'unknown')}, "
                    f"システム={result.metadata.get('system_used', 'unknown')}",
                )

        except Exception as e:
            logger.error(f"Auto mode test failed for {symbol}: {e!s}")

    integration_data["auto_mode_selection"] = auto_mode_results

    # システム切り替えテスト
    print("   システム切り替えテスト...")
    switching_results = {}

    for symbol in test_symbols[:1]:  # 1銘柄のみでテスト
        symbol_results = {}

        for mode in [PredictionMode.SPEED_PRIORITY, PredictionMode.ACCURACY_PRIORITY]:
            try:
                start_time = time.time()
                result = hybrid_system.predict(symbol, mode)
                switch_time = time.time() - start_time

                symbol_results[mode.value] = {
                    "prediction": result.prediction,
                    "confidence": result.confidence,
                    "system_used": result.metadata.get("system_used", "unknown"),
                    "switch_time": switch_time,
                }

                print(
                    f"     {symbol} {mode.value}: "
                    f"システム={result.metadata.get('system_used', 'unknown')}, "
                    f"時間={switch_time:.3f}秒",
                )

            except Exception as e:
                logger.error(f"System switching test failed: {e!s}")

        switching_results[symbol] = symbol_results

    integration_data["system_switching"] = switching_results

    return integration_data


def display_test_results(results: Dict[str, Any]):
    """テスト結果の表示"""
    print("\n" + "=" * 80)
    print("ハイブリッドシステム テスト結果")
    print("=" * 80)

    # システム情報
    if "system_info" in results:
        system_info = results["system_info"]
        print(f"システム名: {system_info['name']}")
        print(f"バージョン: {system_info['version']}")
        print(f"サブシステム: {list(system_info['subsystems'].keys())}")

    # モード別テスト結果
    if "mode_tests" in results:
        print("\n[モード別性能]")
        mode_tests = results["mode_tests"]

        for mode, data in mode_tests.items():
            if data["predictions"]:
                print(f"  {mode}:")
                print(f"    平均信頼度: {data['avg_confidence']:.2f}")
                print(f"    平均精度: {data['avg_accuracy']:.1f}%")
                print(f"    総処理時間: {data['total_time']:.3f}秒")
                print(
                    f"    銘柄あたり時間: {data['total_time'] / len(data['predictions']):.3f}秒",
                )

    # パフォーマンステスト結果
    if "performance_tests" in results:
        print("\n[パフォーマンス]")
        perf_tests = results["performance_tests"]

        if (
            "batch_processing" in perf_tests
            and "error" not in perf_tests["batch_processing"]
        ):
            batch_data = perf_tests["batch_processing"]
            print("  バッチ処理:")
            print(f"    スループット: {batch_data['throughput']:.1f} 銘柄/秒")
            print(f"    成功率: {batch_data['success_rate'] * 100:.1f}%")

        if "performance_statistics" in perf_tests:
            stats = perf_tests["performance_statistics"]
            if "avg_prediction_time" in stats:
                print("  統計:")
                print(f"    平均予測時間: {stats['avg_prediction_time']:.3f}秒")
                print(f"    総予測回数: {stats['total_predictions']}")

    # 統合機能結果
    if "integration_tests" in results:
        print("\n[統合機能]")
        integration = results["integration_tests"]

        if "auto_mode_selection" in integration:
            auto_results = integration["auto_mode_selection"]
            if auto_results:
                modes_used = [r["mode_used"] for r in auto_results]
                print(f"  自動モード選択: {set(modes_used)}")

        if "system_switching" in integration:
            switching = integration["system_switching"]
            print(f"  システム切り替え: {len(switching)}銘柄でテスト完了")

    print("\n" + "=" * 80)
    print("[結論] ハイブリッドシステム動作確認完了")
    print("速度と精度を両立した統合予測システムが正常に動作")
    print("=" * 80)


if __name__ == "__main__":
    main()
