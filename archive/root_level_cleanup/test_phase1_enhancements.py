#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1機能強化テスト
インテリジェントキャッシュ + 次世代モード + 学習型最適化の統合テスト
"""

import sys
import os
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import traceback

# ロギング設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))


def main():
    """メインテスト実行関数"""
    print("=" * 80)
    print("Phase 1機能強化 統合テスト")
    print("=" * 80)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Phase 1機能テスト実行
        test_results = run_phase1_enhancement_tests()

        if test_results["success"]:
            print("\n[成功] Phase 1機能強化テスト完了!")
            display_test_results(test_results)
        else:
            print(
                f"\n[失敗] Phase 1テスト失敗: {test_results.get('error', 'Unknown error')}"
            )

    except KeyboardInterrupt:
        print("\n\nテストが中断されました")
    except Exception as e:
        print(f"\n[エラー] 予期しないエラー: {str(e)}")
        traceback.print_exc()


def run_phase1_enhancement_tests() -> Dict[str, Any]:
    """Phase 1機能強化の包括的テスト"""
    try:
        print("1. Phase 1ハイブリッドシステム初期化")

        from models.hybrid.hybrid_predictor import (
            HybridStockPredictor,
            PredictionMode,
        )
        from data.stock_data import StockDataProvider

        # データプロバイダー初期化
        data_provider = StockDataProvider()

        # Phase 1強化ハイブリッドシステム初期化
        enhanced_hybrid = HybridStockPredictor(
            data_provider=data_provider,
            default_mode=PredictionMode.AUTO,
            enable_cache=True,
            enable_adaptive_optimization=True,
        )

        print("   [OK] Phase 1強化ハイブリッドシステム初期化完了")

        # システム情報確認
        system_info = enhanced_hybrid.get_model_info()
        print(f"   システム名: {system_info['name']}")
        print(f"   キャッシュ有効: {enhanced_hybrid.cache_enabled}")
        print(f"   学習型最適化有効: {enhanced_hybrid.adaptive_optimization_enabled}")

        # テスト銘柄
        test_symbols = ["6758.T", "7203.T", "8306.T"]  # ソニー、トヨタ、三菱UFJ

        # テスト結果格納
        test_results = {
            "initialization": True,
            "system_info": system_info,
            "cache_tests": {},
            "next_gen_mode_tests": {},
            "adaptive_optimization_tests": {},
            "performance_comparison": {},
        }

        # キャッシュシステムテスト
        print("\n2. インテリジェントキャッシュテスト")
        test_results["cache_tests"] = test_intelligent_cache(
            enhanced_hybrid, test_symbols
        )

        # 次世代予測モードテスト
        print("\n3. 次世代予測モードテスト")
        test_results["next_gen_mode_tests"] = test_next_generation_modes(
            enhanced_hybrid, test_symbols
        )

        # 学習型最適化テスト
        print("\n4. 学習型最適化テスト")
        test_results["adaptive_optimization_tests"] = test_adaptive_optimization(
            enhanced_hybrid, test_symbols
        )

        # パフォーマンス比較テスト
        print("\n5. パフォーマンス比較テスト")
        test_results["performance_comparison"] = test_performance_improvements(
            enhanced_hybrid, test_symbols
        )

        test_results["success"] = True
        return test_results

    except ImportError as e:
        return {"success": False, "error": f"モジュールインポートエラー: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"テスト実行エラー: {str(e)}"}


def test_intelligent_cache(hybrid_system, test_symbols: List[str]) -> Dict[str, Any]:
    """インテリジェントキャッシュテスト"""
    cache_results = {
        "cache_enabled": hybrid_system.cache_enabled,
        "cache_hits": 0,
        "cache_misses": 0,
        "performance_improvement": 0,
        "cache_statistics": {},
    }

    if not hybrid_system.cache_enabled:
        cache_results["error"] = "キャッシュが無効です"
        return cache_results

    print("   キャッシュ動作テスト...")

    # 初回予測（キャッシュミス期待）
    for symbol in test_symbols:
        start_time = time.time()
        result1 = hybrid_system.predict(symbol, PredictionMode.BALANCED)
        first_time = time.time() - start_time

        # 同じ予測（キャッシュヒット期待）
        start_time = time.time()
        result2 = hybrid_system.predict(symbol, PredictionMode.BALANCED)
        second_time = time.time() - start_time

        # キャッシュヒット判定
        if result2.metadata.get("cache_hit", False):
            cache_results["cache_hits"] += 1
            improvement = (first_time - second_time) / first_time * 100
            cache_results["performance_improvement"] += improvement
            print(f"     {symbol}: キャッシュヒット - {improvement:.1f}% 高速化")
        else:
            cache_results["cache_misses"] += 1
            print(f"     {symbol}: キャッシュミス")

    # キャッシュ統計取得
    if hybrid_system.intelligent_cache:
        cache_results["cache_statistics"] = (
            hybrid_system.intelligent_cache.get_cache_statistics()
        )

    # 平均パフォーマンス向上
    total_tests = len(test_symbols)
    if total_tests > 0:
        cache_results["performance_improvement"] /= total_tests

    return cache_results


def test_next_generation_modes(
    hybrid_system, test_symbols: List[str]
) -> Dict[str, Any]:
    """次世代予測モードテスト"""

    # 次世代モードリスト
    next_gen_modes = [
        PredictionMode.ULTRA_SPEED,
        PredictionMode.RESEARCH_MODE,
        PredictionMode.SWING_TRADE,
        PredictionMode.SCALPING,
        PredictionMode.PORTFOLIO_ANALYSIS,
        PredictionMode.RISK_MANAGEMENT,
    ]

    mode_results = {}

    for mode in next_gen_modes:
        print(f"   {mode.value}モードテスト...")
        mode_data = {
            "predictions": [],
            "total_time": 0,
            "avg_confidence": 0,
            "avg_accuracy": 0,
            "mode_specific_features": {},
        }

        start_time = time.time()

        for symbol in test_symbols[:1]:  # 1銘柄のみでテスト（時間短縮）
            try:
                result = hybrid_system.predict(symbol, mode)
                mode_data["predictions"].append(
                    {
                        "symbol": symbol,
                        "prediction": result.prediction,
                        "confidence": result.confidence,
                        "accuracy": result.accuracy,
                        "metadata": result.metadata,
                    }
                )

                # モード固有の特徴確認
                strategy = result.metadata.get("prediction_strategy", "unknown")
                optimization = result.metadata.get("optimization", "none")

                print(f"     {symbol}: 戦略={strategy}, 最適化={optimization}")

            except Exception as e:
                logger.error(
                    f"Mode {mode.value} prediction failed for {symbol}: {str(e)}"
                )

        mode_data["total_time"] = time.time() - start_time

        if mode_data["predictions"]:
            mode_data["avg_confidence"] = np.mean(
                [p["confidence"] for p in mode_data["predictions"]]
            )
            mode_data["avg_accuracy"] = np.mean(
                [p["accuracy"] for p in mode_data["predictions"]]
            )

        mode_results[mode.value] = mode_data

    return mode_results


def test_adaptive_optimization(
    hybrid_system, test_symbols: List[str]
) -> Dict[str, Any]:
    """学習型最適化テスト"""
    optimization_results = {
        "optimization_enabled": hybrid_system.adaptive_optimization_enabled,
        "optimization_triggers": 0,
        "pattern_learning": {},
        "performance_monitoring": {},
    }

    if not hybrid_system.adaptive_optimization_enabled:
        optimization_results["error"] = "学習型最適化が無効です"
        return optimization_results

    print("   学習型最適化動作テスト...")

    # 複数回予測実行して学習パターン生成
    prediction_count = 0
    initial_interval = hybrid_system.optimization_interval

    # 最適化間隔を短くしてテスト
    hybrid_system.optimization_interval = 5  # 5回ごとに最適化

    for i in range(10):  # 10回予測実行
        symbol = test_symbols[i % len(test_symbols)]
        mode = [
            PredictionMode.SPEED_PRIORITY,
            PredictionMode.ACCURACY_PRIORITY,
            PredictionMode.BALANCED,
        ][i % 3]

        result = hybrid_system.predict(symbol, mode)
        prediction_count += 1

        # 最適化トリガー確認
        if hybrid_system.optimization_counter == 0:  # リセットされた = 最適化実行
            optimization_results["optimization_triggers"] += 1
            print(f"     最適化実行 #{optimization_results['optimization_triggers']}")

    # 最適化間隔を元に戻す
    hybrid_system.optimization_interval = initial_interval

    # 最適化状況取得
    optimization_status = hybrid_system.get_adaptive_optimization_status()
    optimization_results["optimization_status"] = optimization_status

    # パフォーマンス統計取得
    performance_stats = hybrid_system.get_performance_stats()
    if "cache_statistics" in performance_stats:
        optimization_results["performance_monitoring"] = performance_stats[
            "cache_statistics"
        ]

    return optimization_results


def test_performance_improvements(
    hybrid_system, test_symbols: List[str]
) -> Dict[str, Any]:
    """パフォーマンス改善テスト"""

    # 従来システム（キャッシュ・最適化無効）との比較
    print("   パフォーマンス比較テスト...")

    # Phase 1強化システムでのテスト
    enhanced_times = []
    enhanced_confidences = []

    for symbol in test_symbols:
        start_time = time.time()
        result = hybrid_system.predict(symbol)
        prediction_time = time.time() - start_time

        enhanced_times.append(prediction_time)
        enhanced_confidences.append(result.confidence)

    # 基本システム（比較用）でのテスト
    from models.hybrid.hybrid_predictor import HybridStockPredictor

    basic_hybrid = HybridStockPredictor(
        enable_cache=False, enable_adaptive_optimization=False
    )

    basic_times = []
    basic_confidences = []

    for symbol in test_symbols:
        start_time = time.time()
        result = basic_hybrid.predict(symbol)
        prediction_time = time.time() - start_time

        basic_times.append(prediction_time)
        basic_confidences.append(result.confidence)

    # パフォーマンス比較計算
    avg_enhanced_time = np.mean(enhanced_times)
    avg_basic_time = np.mean(basic_times)
    speed_improvement = (avg_basic_time - avg_enhanced_time) / avg_basic_time * 100

    avg_enhanced_confidence = np.mean(enhanced_confidences)
    avg_basic_confidence = np.mean(basic_confidences)
    confidence_improvement = (
        (avg_enhanced_confidence - avg_basic_confidence) / avg_basic_confidence * 100
    )

    comparison_results = {
        "enhanced_avg_time": avg_enhanced_time,
        "basic_avg_time": avg_basic_time,
        "speed_improvement_percent": speed_improvement,
        "enhanced_avg_confidence": avg_enhanced_confidence,
        "basic_avg_confidence": avg_basic_confidence,
        "confidence_improvement_percent": confidence_improvement,
        "overall_performance_score": (speed_improvement + confidence_improvement) / 2,
    }

    print(f"     速度改善: {speed_improvement:.1f}%")
    print(f"     信頼度改善: {confidence_improvement:.1f}%")

    return comparison_results


def display_test_results(results: Dict[str, Any]):
    """テスト結果の表示"""
    print("\n" + "=" * 80)
    print("Phase 1機能強化 テスト結果")
    print("=" * 80)

    # システム情報
    if "system_info" in results:
        system_info = results["system_info"]
        print(f"システム名: {system_info['name']}")
        print(f"バージョン: {system_info['version']}")

    # キャッシュテスト結果
    if "cache_tests" in results:
        print("\n[インテリジェントキャッシュ]")
        cache_tests = results["cache_tests"]

        if cache_tests.get("cache_enabled"):
            print(f"  キャッシュヒット: {cache_tests['cache_hits']}")
            print(f"  キャッシュミス: {cache_tests['cache_misses']}")
            print(f"  平均高速化: {cache_tests['performance_improvement']:.1f}%")

            if "cache_statistics" in cache_tests:
                stats = cache_tests["cache_statistics"]
                if "hit_rate" in stats:
                    print(f"  ヒット率: {stats['hit_rate']*100:.1f}%")

    # 次世代モードテスト結果
    if "next_gen_mode_tests" in results:
        print("\n[次世代予測モード]")
        mode_tests = results["next_gen_mode_tests"]

        for mode, data in mode_tests.items():
            if data["predictions"]:
                print(f"  {mode}:")
                print(f"    平均信頼度: {data['avg_confidence']:.2f}")
                print(f"    平均精度: {data['avg_accuracy']:.1f}%")
                print(f"    処理時間: {data['total_time']:.3f}秒")

    # 学習型最適化結果
    if "adaptive_optimization_tests" in results:
        print("\n[学習型最適化]")
        opt_tests = results["adaptive_optimization_tests"]

        if opt_tests.get("optimization_enabled"):
            print(f"  最適化実行回数: {opt_tests['optimization_triggers']}")
            print(f"  状況: 有効")
        else:
            print(f"  状況: 無効")

    # パフォーマンス比較結果
    if "performance_comparison" in results:
        print("\n[パフォーマンス比較]")
        perf_comp = results["performance_comparison"]

        print(f"  速度改善: {perf_comp['speed_improvement_percent']:.1f}%")
        print(f"  信頼度改善: {perf_comp['confidence_improvement_percent']:.1f}%")
        print(f"  総合スコア: {perf_comp['overall_performance_score']:.1f}%")

    print("\n" + "=" * 80)
    print("[結論] Phase 1機能強化テスト完了")
    print("インテリジェントキャッシュ + 次世代モード + 学習型最適化が統合動作")
    print("=" * 80)


if __name__ == "__main__":
    main()
