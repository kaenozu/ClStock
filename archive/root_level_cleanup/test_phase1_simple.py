#!/usr/bin/env python3
"""Phase 1機能簡易テスト
インテリジェントキャッシュ + 次世代モード機能の動作確認
"""

import logging
import os
import sys
from datetime import datetime

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# プロジェクトパスの追加
sys.path.append(os.path.dirname(__file__))


def main():
    """メインテスト実行"""
    print("=" * 60)
    print("Phase 1機能 簡易動作確認テスト")
    print("=" * 60)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # 基本インポートテスト
        print("1. モジュールインポートテスト")
        test_imports()

        # キャッシュ単体テスト
        print("\n2. キャッシュ単体テスト")
        test_cache_standalone()

        # 次世代モード確認
        print("\n3. 次世代モード確認")
        test_prediction_modes()

        print("\n[成功] Phase 1機能簡易テスト完了!")

    except Exception as e:
        print(f"\n[エラー] テスト失敗: {e!s}")
        import traceback

        traceback.print_exc()


def test_imports():
    """インポートテスト"""
    try:
        from models.hybrid.intelligent_cache import IntelligentPredictionCache

        print("   [OK] IntelligentPredictionCache インポート成功")

        cache = IntelligentPredictionCache()
        print("   [OK] キャッシュシステム初期化成功")

        stats = cache.get_cache_statistics()
        print(f"   [OK] キャッシュ統計取得: ヒット率={stats.get('hit_rate', 0):.2f}")

    except Exception as e:
        print(f"   [ERROR] インポートエラー: {e!s}")
        raise


def test_cache_standalone():
    """キャッシュ単体テスト"""
    try:
        from datetime import datetime

        from models.base.interfaces import PredictionResult
        from models.hybrid.intelligent_cache import IntelligentPredictionCache

        cache = IntelligentPredictionCache()

        # テスト用予測結果作成
        test_result = PredictionResult(
            prediction=100.0,
            confidence=0.85,
            accuracy=90.0,
            timestamp=datetime.now(),
            symbol="TEST.T",
            metadata={"test": True},
        )

        # キャッシュ設定テスト
        cache._set_to_memory("test_key", test_result, 60)
        print("   [OK] メモリキャッシュ設定成功")

        # キャッシュ取得テスト
        cached_result = cache._get_from_memory("test_key")
        if cached_result:
            print(f"   [OK] キャッシュ取得成功: 予測値={cached_result.prediction}")
        else:
            print("   [WARNING] キャッシュ取得失敗")

        # 統計確認
        stats = cache.get_cache_statistics()
        print(f"   [OK] メモリキャッシュサイズ: {stats.get('memory_cache_size', 0)}")

    except Exception as e:
        print(f"   [ERROR] キャッシュテストエラー: {e!s}")
        raise


def test_prediction_modes():
    """予測モード確認テスト"""
    try:
        # Enumクラス直接インポート
        sys.path.append(os.path.join(os.path.dirname(__file__), "models_new", "hybrid"))

        # ファイル読み込みで次世代モード確認
        hybrid_file = os.path.join(
            os.path.dirname(__file__), "models_new", "hybrid", "hybrid_predictor.py",
        )

        if os.path.exists(hybrid_file):
            with open(hybrid_file, encoding="utf-8") as f:
                content = f.read()

            # 次世代モード確認
            next_gen_modes = [
                "ULTRA_SPEED",
                "RESEARCH_MODE",
                "SWING_TRADE",
                "SCALPING",
                "PORTFOLIO_ANALYSIS",
                "RISK_MANAGEMENT",
            ]

            found_modes = []
            for mode in next_gen_modes:
                if mode in content:
                    found_modes.append(mode)

            print(f"   [OK] 次世代モード確認: {len(found_modes)}/6 モード実装済み")
            for mode in found_modes:
                print(f"     - {mode}")

        # 学習型最適化確認
        adaptive_file = os.path.join(
            os.path.dirname(__file__), "models_new", "hybrid", "adaptive_optimizer.py",
        )
        if os.path.exists(adaptive_file):
            print("   [OK] 学習型最適化システムファイル確認")
        else:
            print("   [WARNING] 学習型最適化システムファイル未確認")

    except Exception as e:
        print(f"   [ERROR] 予測モードテストエラー: {e!s}")
        raise


if __name__ == "__main__":
    main()
