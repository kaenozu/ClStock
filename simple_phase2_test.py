#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2機能簡単テスト
基本的な機能確認のみ
"""

import logging
from datetime import datetime

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phase2_imports():
    """Phase 2モジュールインポートテスト"""
    try:
        from models_new.hybrid.hybrid_predictor import HybridStockPredictor
        from models_new.hybrid.prediction_modes import PredictionMode
        from models_new.hybrid.ultra_fast_streaming import UltraFastStreamingPredictor
        from models_new.hybrid.multi_gpu_processor import MultiGPUParallelPredictor, RealTimeLearningSystem

        logger.info("[OK] All Phase 2 modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Import failed: {str(e)}")
        return False

def test_hybrid_basic_initialization():
    """ハイブリッドシステム基本初期化テスト"""
    try:
        from models_new.hybrid.hybrid_predictor import HybridStockPredictor

        # キャッシュのみ有効にして初期化
        predictor = HybridStockPredictor(
            enable_cache=True,
            enable_adaptive_optimization=False,
            enable_streaming=False,
            enable_multi_gpu=False,
            enable_real_time_learning=False
        )

        logger.info("[OK] Basic hybrid predictor initialized")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Basic initialization failed: {str(e)}")
        return False

def test_streaming_basic():
    """ストリーミング基本テスト"""
    try:
        from models_new.hybrid.ultra_fast_streaming import UltraFastStreamingPredictor

        streaming_predictor = UltraFastStreamingPredictor(buffer_size=100)
        logger.info("[OK] Streaming predictor created")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Streaming test failed: {str(e)}")
        return False

def test_gpu_basic():
    """GPU基本テスト"""
    try:
        from models_new.hybrid.multi_gpu_processor import MultiGPUParallelPredictor

        gpu_predictor = MultiGPUParallelPredictor()
        logger.info("[OK] GPU predictor created")
        return True
    except Exception as e:
        logger.error(f"[ERROR] GPU test failed: {str(e)}")
        return False

def test_learning_basic():
    """実時間学習基本テスト"""
    try:
        from models_new.hybrid.multi_gpu_processor import RealTimeLearningSystem

        learning_system = RealTimeLearningSystem(learning_window_size=10)
        status = learning_system.get_learning_status()
        logger.info(f"[OK] Learning system created, status: {status['learning_active']}")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Learning test failed: {str(e)}")
        return False

def main():
    """メイン実行"""
    logger.info("=== Phase 2 簡単テスト開始 ===")

    tests = [
        ("モジュールインポートテスト", test_phase2_imports),
        ("ハイブリッド基本初期化テスト", test_hybrid_basic_initialization),
        ("ストリーミング基本テスト", test_streaming_basic),
        ("GPU基本テスト", test_gpu_basic),
        ("実時間学習基本テスト", test_learning_basic)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"テスト失敗: {test_name}")
        except Exception as e:
            logger.error(f"テスト例外: {test_name}: {str(e)}")

    logger.info(f"\n=== テスト結果 ===")
    logger.info(f"成功: {passed}/{total} ({(passed/total)*100:.1f}%)")

    if passed == total:
        logger.info("[SUCCESS] 全てのPhase 2基本機能が正常動作")
    else:
        logger.info("[PARTIAL] 一部の機能に問題があります")

if __name__ == "__main__":
    main()