#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ハイブリッドシステム使い方デモ
実際の使用例とベストプラクティス
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

from models.hybrid.hybrid_predictor import HybridStockPredictor, PredictionMode
from data.stock_data import StockDataProvider


def demo_basic_usage():
    """基本的な使い方デモ"""
    print("=" * 60)
    print("1. 基本的な使い方 - シンプル予測")
    print("=" * 60)

    # システム初期化
    print("[OK] システム初期化...")
    hybrid_system = HybridStockPredictor()

    # 単一銘柄予測（自動モード）
    symbol = "6758.T"  # ソニー
    print(f"\n[PREDICT] {symbol} の予測実行...")

    result = hybrid_system.predict(symbol)
    print(f"予測値: {result.prediction:.1f}円")
    print(f"信頼度: {result.confidence:.2f}")
    print(f"使用システム: {result.metadata.get('system_used', 'unknown')}")
    print(f"処理時間: {result.metadata.get('prediction_time', 0):.3f}秒")


def demo_mode_selection():
    """モード選択デモ"""
    print("\n" + "=" * 60)
    print("2. モード選択 - 用途別使い分け")
    print("=" * 60)

    hybrid_system = HybridStockPredictor()
    symbol = "7203.T"  # トヨタ

    # 各モードでの予測
    modes = [
        (PredictionMode.SPEED_PRIORITY, "[SPEED] 高速取引向け"),
        (PredictionMode.ACCURACY_PRIORITY, "[ACCURACY] 精密分析向け"),
        (PredictionMode.BALANCED, "[BALANCED] バランス重視"),
        (PredictionMode.AUTO, "[AUTO] 自動最適化"),
    ]

    for mode, description in modes:
        print(f"\n{description}")
        result = hybrid_system.predict(symbol, mode)
        print(f"  予測値: {result.prediction:.1f}円")
        print(f"  信頼度: {result.confidence:.2f}")
        print(f"  処理時間: {result.metadata.get('prediction_time', 0):.3f}秒")


def demo_batch_processing():
    """バッチ処理デモ"""
    print("\n" + "=" * 60)
    print("3. バッチ処理 - 複数銘柄一括予測")
    print("=" * 60)

    hybrid_system = HybridStockPredictor()

    # 複数銘柄リスト
    symbols = ["6758.T", "7203.T", "8306.T", "9984.T", "9983.T"]
    print(f"[BATCH] {len(symbols)}銘柄の一括予測...")

    # バッチ予測実行
    import time

    start_time = time.time()

    results = hybrid_system.predict_batch(symbols)

    batch_time = time.time() - start_time

    print(f"\n処理時間: {batch_time:.3f}秒")
    print(f"スループット: {len(symbols)/batch_time:.1f}銘柄/秒")
    print(f"成功率: {len(results)/len(symbols)*100:.1f}%")

    print("\n個別結果:")
    for i, result in enumerate(results):
        print(
            f"  {symbols[i]}: {result.prediction:.1f}円 (信頼度{result.confidence:.2f})"
        )


def demo_real_time_scenario():
    """リアルタイム取引シナリオ"""
    print("\n" + "=" * 60)
    print("4. リアルタイム取引シナリオ")
    print("=" * 60)

    hybrid_system = HybridStockPredictor()

    # 高速取引シナリオ
    print("[SPEED] 高速取引モード（アルゴ取引想定）")
    symbols = ["6758.T", "7203.T", "8306.T"]

    total_time = 0
    for symbol in symbols:
        import time

        start = time.time()

        result = hybrid_system.predict(symbol, PredictionMode.SPEED_PRIORITY)

        elapsed = time.time() - start
        total_time += elapsed

        print(f"  {symbol}: {result.prediction:.1f}円 ({elapsed*1000:.1f}ms)")

    print(f"総処理時間: {total_time*1000:.1f}ms")
    print(f"平均レスポンス: {total_time*1000/len(symbols):.1f}ms/銘柄")


def demo_analysis_scenario():
    """深度分析シナリオ"""
    print("\n" + "=" * 60)
    print("5. 深度分析シナリオ")
    print("=" * 60)

    hybrid_system = HybridStockPredictor()
    symbol = "6758.T"

    print(f"[ANALYSIS] {symbol} の詳細分析...")

    # 高精度モードで予測
    result = hybrid_system.predict(symbol, PredictionMode.ACCURACY_PRIORITY)

    print(f"予測値: {result.prediction:.2f}円")
    print(f"信頼度: {result.confidence:.3f}")
    print(f"期待精度: {result.accuracy:.1f}%")

    # メタデータ詳細
    metadata = result.metadata
    print(f"\n詳細情報:")
    print(f"  使用システム: {metadata.get('system_used', 'unknown')}")
    print(f"  予測戦略: {metadata.get('prediction_strategy', 'unknown')}")

    if "meta_learning_contribution" in metadata:
        print(f"  メタ学習貢献度: {metadata.get('meta_learning_contribution', 0):.2f}")
    if "dqn_contribution" in metadata:
        print(f"  DQN貢献度: {metadata.get('dqn_contribution', 0):.2f}")


def demo_monitoring():
    """システム監視デモ"""
    print("\n" + "=" * 60)
    print("6. システム監視・パフォーマンス確認")
    print("=" * 60)

    hybrid_system = HybridStockPredictor()

    # いくつか予測を実行してから統計取得
    symbols = ["6758.T", "7203.T", "8306.T"]
    for symbol in symbols:
        hybrid_system.predict(symbol)

    # システム情報
    info = hybrid_system.get_model_info()
    print("[INFO] システム情報:")
    print(f"  システム名: {info['name']}")
    print(f"  バージョン: {info['version']}")
    print(f"  サブシステム数: {len(info['subsystems'])}")
    print(f"  予測履歴数: {info['prediction_history_size']}")

    # パフォーマンス統計
    stats = hybrid_system.get_performance_stats()
    if "error" not in stats:
        print(f"\n[PERF] パフォーマンス統計:")
        print(f"  総予測回数: {stats['total_predictions']}")
        print(f"  平均処理時間: {stats['avg_prediction_time']:.3f}秒")
        print(f"  平均信頼度: {stats['avg_confidence']:.2f}")


def main():
    """デモ実行"""
    print("[DEMO] ハイブリッドシステム使い方デモ")
    print("用途別の実践的な使用例をご紹介します\n")

    try:
        demo_basic_usage()
        demo_mode_selection()
        demo_batch_processing()
        demo_real_time_scenario()
        demo_analysis_scenario()
        demo_monitoring()

        print("\n" + "=" * 60)
        print("[SUCCESS] デモ完了！")
        print("=" * 60)
        print("[INFO] 実際の使用では、用途に応じてモードを選択してください:")
        print("  - 高速取引 -> SPEED_PRIORITY")
        print("  - 精密分析 -> ACCURACY_PRIORITY")
        print("  - 日常使用 -> AUTO (推奨)")
        print("  - カスタム -> BALANCED")

    except Exception as e:
        print(f"[ERROR] エラー: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
