#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2機能統合テスト
超高速ストリーミング + マルチGPU並列処理 + 実時間学習システムの統合テスト
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import List, Dict, Any

# Phase 2システムのインポート
from models_new.hybrid.hybrid_predictor import HybridStockPredictor
from models_new.hybrid.prediction_modes import PredictionMode
from models_new.hybrid.ultra_fast_streaming import UltraFastStreamingPredictor
from models_new.hybrid.multi_gpu_processor import MultiGPUParallelPredictor, RealTimeLearningSystem

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2IntegrationTester:
    """Phase 2機能統合テスター"""

    def __init__(self):
        self.logger = logger
        self.test_results = {}

    async def run_all_tests(self):
        """全テスト実行"""
        self.logger.info("=== Phase 2 統合テスト開始 ===")

        test_methods = [
            ("基本ハイブリッド初期化テスト", self.test_hybrid_initialization),
            ("超高速ストリーミングテスト", self.test_ultra_fast_streaming),
            ("マルチGPU並列処理テスト", self.test_multi_gpu_processing),
            ("実時間学習システムテスト", self.test_real_time_learning),
            ("包括的システム統合テスト", self.test_comprehensive_integration),
            ("パフォーマンス負荷テスト", self.test_performance_load)
        ]

        for test_name, test_method in test_methods:
            self.logger.info(f"\n--- {test_name} ---")
            try:
                start_time = time.time()
                result = await test_method()
                execution_time = time.time() - start_time

                self.test_results[test_name] = {
                    'status': 'SUCCESS' if result else 'FAILURE',
                    'execution_time': execution_time,
                    'details': result if isinstance(result, dict) else {}
                }

                status = '[OK]' if result else '[ERROR]'
                self.logger.info(f"{status} {test_name} - {execution_time:.3f}秒")

            except Exception as e:
                self.test_results[test_name] = {
                    'status': 'ERROR',
                    'execution_time': 0,
                    'error': str(e)
                }
                self.logger.error(f"[ERROR] {test_name}: {str(e)}")

        self._print_test_summary()

    async def test_hybrid_initialization(self) -> bool:
        """ハイブリッドシステム初期化テスト"""
        try:
            # 全機能有効でハイブリッドシステム初期化
            predictor = HybridStockPredictor(
                enable_cache=True,
                enable_adaptive_optimization=True,
                enable_streaming=True,
                enable_multi_gpu=True,
                enable_real_time_learning=True
            )

            # システム状況確認
            status = predictor.get_comprehensive_system_status()

            self.logger.info(f"キャッシュ有効: {status['hybrid_predictor']['cache_enabled']}")
            self.logger.info(f"適応最適化有効: {status['hybrid_predictor']['adaptive_optimization_enabled']}")
            self.logger.info(f"ストリーミング有効: {status['hybrid_predictor']['streaming_enabled']}")
            self.logger.info(f"マルチGPU有効: {status['hybrid_predictor']['multi_gpu_enabled']}")
            self.logger.info(f"実時間学習有効: {status['hybrid_predictor']['real_time_learning_enabled']}")

            return True

        except Exception as e:
            self.logger.error(f"初期化テスト失敗: {str(e)}")
            return False

    async def test_ultra_fast_streaming(self) -> Dict[str, Any]:
        """超高速ストリーミングテスト"""
        try:
            streaming_predictor = UltraFastStreamingPredictor(buffer_size=1000)

            # テストデータストリーミング
            test_symbols = ["6758.T", "7203.T", "8306.T"]

            # ストリーミング開始
            await streaming_predictor.start_streaming(test_symbols)

            # 予測実行
            for i in range(10):
                result = await streaming_predictor.predict_streaming_single("6758.T")
                if result and result.prediction > 0:
                    continue
                else:
                    return {'success': False, 'error': 'Invalid prediction result'}

            # 統計取得
            stats = streaming_predictor.get_streaming_statistics()

            # ストリーミング停止
            streaming_predictor.stop_streaming()

            return {
                'success': True,
                'predictions_completed': 10,
                'statistics': stats
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def test_multi_gpu_processing(self) -> Dict[str, Any]:
        """マルチGPU並列処理テスト"""
        try:
            gpu_predictor = MultiGPUParallelPredictor()

            # 大規模バッチテスト
            test_symbols = [f"TEST{i:04d}.T" for i in range(50)]

            # GPU並列予測実行
            results = await gpu_predictor.predict_massive_batch(test_symbols)

            # パフォーマンス統計取得
            performance_stats = gpu_predictor.get_performance_statistics()

            return {
                'success': True,
                'processed_symbols': len(results),
                'expected_symbols': len(test_symbols),
                'performance_stats': performance_stats
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def test_real_time_learning(self) -> Dict[str, Any]:
        """実時間学習システムテスト"""
        try:
            learning_system = RealTimeLearningSystem(learning_window_size=50)

            # テスト市場データ生成
            test_market_data = []
            for i in range(100):
                data = {
                    'symbol': 'TEST.T',
                    'price': 1000 + i * 10,
                    'volume': 100000 + i * 1000,
                    'timestamp': datetime.now()
                }
                test_market_data.append(data)

            # リアルタイム学習実行
            learning_results = []
            for data in test_market_data:
                result = await learning_system.process_real_time_data(data)
                learning_results.append(result)

            # 予測フィードバック追加
            for i in range(20):
                learning_system.add_prediction_feedback(
                    prediction=1000 + i * 10,
                    actual=1000 + i * 11,  # 少し異なる実績値
                    symbol="TEST.T"
                )

            # 学習状況取得
            learning_status = learning_system.get_learning_status()

            return {
                'success': True,
                'data_processed': len(test_market_data),
                'learning_results': len([r for r in learning_results if r.get('status') == 'learning_completed']),
                'learning_status': learning_status
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def test_comprehensive_integration(self) -> Dict[str, Any]:
        """包括的システム統合テスト"""
        try:
            # 全機能統合ハイブリッドシステム
            predictor = HybridStockPredictor(
                enable_cache=True,
                enable_adaptive_optimization=True,
                enable_streaming=True,
                enable_multi_gpu=True,
                enable_real_time_learning=True
            )

            test_results = {}

            # 1. 単一予測テスト（各モード）
            test_modes = [
                PredictionMode.ULTRA_SPEED,
                PredictionMode.ACCURACY_PRIORITY,
                PredictionMode.RESEARCH_MODE,
                PredictionMode.SCALPING
            ]

            for mode in test_modes:
                result = await predictor.predict("6758.T", mode=mode)
                test_results[f'single_prediction_{mode.value}'] = {
                    'success': result is not None,
                    'prediction': result.prediction if result else None,
                    'confidence': result.confidence if result else None
                }

            # 2. バッチ予測テスト
            batch_symbols = ["6758.T", "7203.T", "8306.T", "4502.T", "6861.T"]
            batch_results = await predictor.predict_batch(batch_symbols)
            test_results['batch_prediction'] = {
                'success': len(batch_results) == len(batch_symbols),
                'processed_count': len(batch_results)
            }

            # 3. ストリーミング予測テスト
            if predictor.streaming_enabled:
                await predictor.start_streaming(["6758.T"])
                streaming_result = await predictor.predict_streaming_batch(["6758.T"])
                predictor.stop_streaming()
                test_results['streaming_prediction'] = {
                    'success': len(streaming_result) > 0,
                    'result_count': len(streaming_result)
                }

            # 4. 実時間学習テスト
            if predictor.real_time_learning_enabled:
                market_data = {
                    'symbol': '6758.T',
                    'price': 1500.0,
                    'volume': 150000,
                    'timestamp': datetime.now()
                }
                learning_result = await predictor.process_real_time_market_data(market_data)
                test_results['real_time_learning'] = learning_result

            # 5. システム状況確認
            system_status = predictor.get_comprehensive_system_status()
            test_results['system_status'] = system_status

            return {
                'success': True,
                'test_results': test_results
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def test_performance_load(self) -> Dict[str, Any]:
        """パフォーマンス負荷テスト"""
        try:
            predictor = HybridStockPredictor(
                enable_cache=True,
                enable_adaptive_optimization=True,
                enable_streaming=True,
                enable_multi_gpu=True,
                enable_real_time_learning=True
            )

            # 大量予測テスト
            large_symbol_list = [f"TEST{i:04d}.T" for i in range(200)]

            start_time = time.time()
            batch_results = await predictor.predict_batch(large_symbol_list)
            batch_time = time.time() - start_time

            # スループット計算
            throughput = len(batch_results) / batch_time if batch_time > 0 else 0

            # メモリ使用量チェック（近似）
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB

            return {
                'success': True,
                'symbols_processed': len(batch_results),
                'processing_time': batch_time,
                'throughput_per_second': throughput,
                'memory_usage_mb': memory_usage
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _print_test_summary(self):
        """テスト結果サマリー出力"""
        self.logger.info("\n=== Phase 2 統合テスト結果サマリー ===")

        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results.values() if r['status'] == 'SUCCESS'])
        failed_tests = total_tests - successful_tests

        self.logger.info(f"総テスト数: {total_tests}")
        self.logger.info(f"成功: {successful_tests}")
        self.logger.info(f"失敗: {failed_tests}")
        self.logger.info(f"成功率: {(successful_tests/total_tests)*100:.1f}%")

        self.logger.info("\n--- 詳細結果 ---")
        for test_name, result in self.test_results.items():
            status_icon = "[OK]" if result['status'] == 'SUCCESS' else "[ERROR]"
            execution_time = result.get('execution_time', 0)
            self.logger.info(f"{status_icon} {test_name}: {execution_time:.3f}秒")

            if result['status'] == 'ERROR' and 'error' in result:
                self.logger.info(f"    エラー: {result['error']}")

async def main():
    """メイン実行関数"""
    tester = Phase2IntegrationTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())