#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
機能拡張後のパフォーマンステスト
並列処理、キャッシュ、マルチタイムフレーム統合の効果を測定
"""

import time
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))


def test_enhanced_ensemble_performance():
    """機能拡張後のアンサンブル予測性能テスト"""
    print("=" * 80)
    print("機能拡張後 EnsembleStockPredictor パフォーマンステスト")
    print("=" * 80)

    try:
        from models_new.ensemble.ensemble_predictor import EnsembleStockPredictor
        from data.stock_data import StockDataProvider

        # テスト用銘柄（小規模テスト）
        test_symbols = [
            "7203",
            "9984",
            "6758",
            "8306",
            "4523",
            "1803",
            "5101",
            "9022",
            "8031",
            "4004",
        ]

        print(f"テスト対象銘柄数: {len(test_symbols)}")
        print(f"テスト実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # データプロバイダー初期化
        data_provider = StockDataProvider()

        # 拡張版予測器初期化
        enhanced_predictor = EnsembleStockPredictor(data_provider=data_provider)

        # パフォーマンステスト実行
        results = run_performance_tests(enhanced_predictor, test_symbols)

        # 結果表示
        display_performance_results(results)

        return results

    except Exception as e:
        print(f"[エラー] テスト実行エラー: {str(e)}")
        traceback.print_exc()
        return None


def run_performance_tests(predictor, symbols: List[str]) -> Dict[str, Any]:
    """各種パフォーマンステストの実行"""
    results = {}

    # 1. 単一予測速度テスト
    print("1. 単一予測速度テスト")
    print("-" * 40)
    single_prediction_results = test_single_prediction_speed(predictor, symbols[:3])
    results["single_prediction"] = single_prediction_results

    # 2. バッチ予測速度テスト
    print("\n2. バッチ予測速度テスト")
    print("-" * 40)
    batch_prediction_results = test_batch_prediction_speed(predictor, symbols)
    results["batch_prediction"] = batch_prediction_results

    # 3. キャッシュ効果テスト
    print("\n3. キャッシュ効果テスト")
    print("-" * 40)
    cache_effectiveness_results = test_cache_effectiveness(predictor, symbols[:5])
    results["cache_effectiveness"] = cache_effectiveness_results

    # 4. 並列処理効果テスト
    print("\n4. 並列処理効果テスト")
    print("-" * 40)
    parallel_processing_results = test_parallel_processing_effect(predictor, symbols)
    results["parallel_processing"] = parallel_processing_results

    # 5. メモリ使用量テスト
    print("\n5. メモリ使用量テスト")
    print("-" * 40)
    memory_usage_results = test_memory_usage(predictor, symbols)
    results["memory_usage"] = memory_usage_results

    return results


def test_single_prediction_speed(predictor, symbols: List[str]) -> Dict[str, float]:
    """単一予測の速度テスト"""
    prediction_times = []
    successful_predictions = 0

    for symbol in symbols:
        try:
            start_time = time.time()
            result = predictor.predict(symbol)
            end_time = time.time()

            prediction_time = end_time - start_time
            prediction_times.append(prediction_time)
            successful_predictions += 1

            print(
                f"  {symbol}: {prediction_time:.3f}秒 (予測値: {result.prediction:.1f})"
            )

        except Exception as e:
            print(f"  {symbol}: エラー - {str(e)}")

    avg_time = np.mean(prediction_times) if prediction_times else 0
    print(f"\n平均予測時間: {avg_time:.3f}秒")
    print(
        f"成功率: {successful_predictions}/{len(symbols)} ({successful_predictions/len(symbols)*100:.1f}%)"
    )

    return {
        "average_time": avg_time,
        "success_rate": successful_predictions / len(symbols),
        "total_predictions": len(symbols),
    }


def test_batch_prediction_speed(predictor, symbols: List[str]) -> Dict[str, float]:
    """バッチ予測の速度テスト"""
    print(f"バッチ予測テスト ({len(symbols)}銘柄)")

    start_time = time.time()
    try:
        results = predictor.predict_batch(symbols)
        end_time = time.time()

        total_time = end_time - start_time
        per_symbol_time = total_time / len(symbols) if symbols else 0
        success_count = len(results)

        print(f"  総処理時間: {total_time:.3f}秒")
        print(f"  銘柄あたり時間: {per_symbol_time:.3f}秒")
        print(f"  成功予測数: {success_count}/{len(symbols)}")

        return {
            "total_time": total_time,
            "per_symbol_time": per_symbol_time,
            "success_count": success_count,
            "throughput": len(symbols) / total_time if total_time > 0 else 0,
        }

    except Exception as e:
        print(f"  バッチ予測エラー: {str(e)}")
        return {"error": str(e)}


def test_cache_effectiveness(predictor, symbols: List[str]) -> Dict[str, Any]:
    """キャッシュ効果のテスト"""
    print("キャッシュ効果測定")

    # 1回目の予測（キャッシュなし）
    first_run_times = []
    for symbol in symbols:
        try:
            start_time = time.time()
            predictor.predict(symbol)
            end_time = time.time()
            first_run_times.append(end_time - start_time)
        except:
            pass

    # 2回目の予測（キャッシュあり）
    second_run_times = []
    for symbol in symbols:
        try:
            start_time = time.time()
            predictor.predict(symbol)
            end_time = time.time()
            second_run_times.append(end_time - start_time)
        except:
            pass

    if first_run_times and second_run_times:
        avg_first = np.mean(first_run_times)
        avg_second = np.mean(second_run_times)
        improvement = (avg_first - avg_second) / avg_first * 100 if avg_first > 0 else 0

        print(f"  1回目平均時間: {avg_first:.3f}秒")
        print(f"  2回目平均時間: {avg_second:.3f}秒")
        print(f"  キャッシュ効果: {improvement:.1f}%改善")

        return {
            "first_run_avg": avg_first,
            "second_run_avg": avg_second,
            "improvement_percent": improvement,
        }

    return {"error": "キャッシュ効果を測定できませんでした"}


def test_parallel_processing_effect(predictor, symbols: List[str]) -> Dict[str, Any]:
    """並列処理効果のテスト"""
    print("並列処理効果測定")

    # 並列処理の設定確認
    parallel_workers = getattr(predictor.parallel_calculator, "n_jobs", 1)
    print(f"  並列ワーカー数: {parallel_workers}")

    # 特徴量計算時間の測定（少数のサンプルで）
    test_sample = symbols[:3]
    feature_calc_times = []

    for symbol in test_sample:
        try:
            start_time = time.time()
            data = predictor.data_provider.get_stock_data(symbol, "1y")
            if not data.empty:
                features = predictor._calculate_features_optimized(symbol, data)
            end_time = time.time()
            feature_calc_times.append(end_time - start_time)
        except Exception as e:
            print(f"    {symbol}の特徴量計算エラー: {str(e)}")

    if feature_calc_times:
        avg_feature_time = np.mean(feature_calc_times)
        print(f"  平均特徴量計算時間: {avg_feature_time:.3f}秒")

        return {
            "parallel_workers": parallel_workers,
            "avg_feature_calc_time": avg_feature_time,
            "samples_tested": len(feature_calc_times),
        }

    return {"error": "並列処理効果を測定できませんでした"}


def test_memory_usage(predictor, symbols: List[str]) -> Dict[str, Any]:
    """メモリ使用量のテスト"""
    print("メモリ使用量測定")

    try:
        import psutil
        import os

        # プロセス情報取得
        process = psutil.Process(os.getpid())

        # 初期メモリ使用量
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 予測実行
        for symbol in symbols[:5]:  # 少数のサンプルでテスト
            try:
                predictor.predict(symbol)
            except:
                pass

        # 実行後メモリ使用量
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # キャッシュサイズ確認
        feature_cache_size = predictor.feature_cache.size()
        prediction_cache_size = predictor.prediction_cache.size()

        print(f"  初期メモリ使用量: {initial_memory:.1f} MB")
        print(f"  最終メモリ使用量: {final_memory:.1f} MB")
        print(f"  メモリ増加量: {memory_increase:.1f} MB")
        print(f"  特徴量キャッシュサイズ: {feature_cache_size}")
        print(f"  予測キャッシュサイズ: {prediction_cache_size}")

        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": memory_increase,
            "feature_cache_size": feature_cache_size,
            "prediction_cache_size": prediction_cache_size,
        }

    except ImportError:
        print("  psutilが利用できないため、メモリ測定をスキップ")
        return {"error": "psutil not available"}
    except Exception as e:
        print(f"  メモリ測定エラー: {str(e)}")
        return {"error": str(e)}


def display_performance_results(results: Dict[str, Any]):
    """パフォーマンステスト結果の表示"""
    print("\n" + "=" * 80)
    print("パフォーマンステスト結果サマリー")
    print("=" * 80)

    # 単一予測性能
    if "single_prediction" in results:
        single = results["single_prediction"]
        print(f"単一予測性能:")
        print(f"  平均予測時間: {single.get('average_time', 0):.3f}秒")
        print(f"  成功率: {single.get('success_rate', 0)*100:.1f}%")

    # バッチ予測性能
    if "batch_prediction" in results:
        batch = results["batch_prediction"]
        if "error" not in batch:
            print(f"\nバッチ予測性能:")
            print(f"  スループット: {batch.get('throughput', 0):.1f} 銘柄/秒")
            print(f"  銘柄あたり時間: {batch.get('per_symbol_time', 0):.3f}秒")

    # キャッシュ効果
    if "cache_effectiveness" in results:
        cache = results["cache_effectiveness"]
        if "error" not in cache:
            print(f"\nキャッシュ効果:")
            print(f"  性能改善: {cache.get('improvement_percent', 0):.1f}%")

    # 並列処理効果
    if "parallel_processing" in results:
        parallel = results["parallel_processing"]
        if "error" not in parallel:
            print(f"\n並列処理:")
            print(f"  ワーカー数: {parallel.get('parallel_workers', 1)}")
            print(f"  特徴量計算時間: {parallel.get('avg_feature_calc_time', 0):.3f}秒")

    # メモリ使用量
    if "memory_usage" in results:
        memory = results["memory_usage"]
        if "error" not in memory:
            print(f"\nメモリ使用量:")
            print(f"  メモリ増加: {memory.get('memory_increase_mb', 0):.1f} MB")
            print(
                f"  キャッシュエントリ数: {memory.get('feature_cache_size', 0) + memory.get('prediction_cache_size', 0)}"
            )

    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)


def main():
    """メイン実行関数"""
    try:
        results = test_enhanced_ensemble_performance()
        if results:
            print("\n[成功] パフォーマンステスト完了")
            return True
        else:
            print("\n[失敗] パフォーマンステストが失敗しました")
            return False
    except KeyboardInterrupt:
        print("\n\nテストが中断されました")
        return False
    except Exception as e:
        print(f"\n[エラー] 予期しないエラー: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
