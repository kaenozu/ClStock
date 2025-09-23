"""
統合リファクタリングシステムのテスト
新しい統一アーキテクチャの動作確認
"""

import sys
import logging
from pathlib import Path

# プロジェクトルート追加
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from models_refactored.core.interfaces import ModelType, PredictionMode, ModelConfiguration
from models_refactored.core.factory import PredictorFactory, create_predictor
from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class RefactoredSystemTest:
    """統合リファクタリングシステムのテストクラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_provider = StockDataProvider()
        self.test_symbols = ['7203', '9984', '8306']  # トヨタ、ソフトバンクG、三菱UFJ

    def test_factory_pattern(self):
        """ファクトリパターンのテスト"""
        self.logger.info("=" * 50)
        self.logger.info("ファクトリパターンテスト開始")

        try:
            # 利用可能なモデルタイプの確認
            available_types = PredictorFactory.list_available_types()
            self.logger.info(f"利用可能なモデルタイプ: {[t.value for t in available_types]}")

            # エンサンブル予測器の作成
            config = ModelConfiguration(
                model_type=ModelType.ENSEMBLE,
                prediction_mode=PredictionMode.BALANCED,
                cache_enabled=True,
                parallel_enabled=True
            )

            predictor = create_predictor(
                model_type=ModelType.ENSEMBLE,
                config=config,
                data_provider=self.data_provider
            )

            self.logger.info(f"予測器作成成功: {predictor.__class__.__name__}")

            # モデル情報取得
            model_info = predictor.get_model_info()
            self.logger.info(f"モデル情報: {model_info}")

            return True

        except Exception as e:
            self.logger.error(f"ファクトリパターンテスト失敗: {str(e)}")
            return False

    def test_ensemble_predictor(self):
        """統合エンサンブル予測器のテスト"""
        self.logger.info("=" * 50)
        self.logger.info("統合エンサンブル予測器テスト開始")

        try:
            # エンサンブル予測器の作成
            predictor = create_predictor(
                model_type=ModelType.ENSEMBLE,
                data_provider=self.data_provider
            )

            # 単一予測テスト
            symbol = self.test_symbols[0]
            self.logger.info(f"単一予測テスト: {symbol}")

            result = predictor.predict(symbol)
            self.logger.info(f"予測結果: {result.to_dict()}")

            # 信頼度テスト
            confidence = predictor.get_confidence(symbol)
            self.logger.info(f"信頼度: {confidence}")

            # バッチ予測テスト
            self.logger.info(f"バッチ予測テスト: {self.test_symbols}")
            batch_results = predictor.predict_batch(self.test_symbols)

            for result in batch_results:
                self.logger.info(f"  {result.symbol}: {result.prediction:.1f} (信頼度: {result.confidence:.3f})")

            # 性能指標テスト
            metrics = predictor.get_performance_metrics()
            self.logger.info(f"性能指標: {metrics}")

            return True

        except Exception as e:
            self.logger.error(f"エンサンブル予測器テスト失敗: {str(e)}")
            return False

    def test_parallel_feature_calculator(self):
        """並列特徴量計算のテスト"""
        self.logger.info("=" * 50)
        self.logger.info("並列特徴量計算テスト開始")

        try:
            from models_refactored.ensemble.parallel_feature_calculator import ParallelFeatureCalculator

            calculator = ParallelFeatureCalculator(n_jobs=4)

            # 並列特徴量計算実行
            features_df = calculator.calculate_features_parallel(
                self.test_symbols, self.data_provider
            )

            if not features_df.empty:
                self.logger.info(f"特徴量計算成功: {len(features_df)} 行, {len(features_df.columns)} 列")
                self.logger.info(f"特徴量名（一部）: {list(features_df.columns[:10])}")

                # 性能統計
                stats = calculator.get_performance_stats()
                self.logger.info(f"計算統計: {stats}")
            else:
                self.logger.warning("特徴量計算結果が空です")

            return not features_df.empty

        except Exception as e:
            self.logger.error(f"並列特徴量計算テスト失敗: {str(e)}")
            return False

    def test_unified_interfaces(self):
        """統一インターフェースのテスト"""
        self.logger.info("=" * 50)
        self.logger.info("統一インターフェーステスト開始")

        try:
            # 複数の予測器を作成してインターフェース統一性をテスト
            predictors = []

            # 利用可能な全タイプで予測器作成を試行
            available_types = PredictorFactory.list_available_types()

            for model_type in available_types:
                try:
                    predictor = create_predictor(
                        model_type=model_type,
                        data_provider=self.data_provider
                    )
                    predictors.append((model_type, predictor))
                    self.logger.info(f"{model_type.value} 予測器作成成功")
                except Exception as e:
                    self.logger.warning(f"{model_type.value} 予測器作成失敗: {str(e)}")

            # 各予測器でインターフェース統一性をテスト
            symbol = self.test_symbols[0]

            for model_type, predictor in predictors:
                try:
                    # 統一インターフェースのテスト
                    model_info = predictor.get_model_info()
                    is_ready = predictor.is_ready()
                    confidence = predictor.get_confidence(symbol)

                    self.logger.info(
                        f"{model_type.value}: ready={is_ready}, confidence={confidence:.3f}"
                    )

                    # 予測実行（訓練済みの場合のみ）
                    if is_ready:
                        result = predictor.predict(symbol)
                        self.logger.info(
                            f"  予測結果: {result.prediction:.1f}, "
                            f"実行時間: {result.execution_time:.3f}s"
                        )

                except Exception as e:
                    self.logger.warning(f"{model_type.value} インターフェーステスト失敗: {str(e)}")

            return len(predictors) > 0

        except Exception as e:
            self.logger.error(f"統一インターフェーステスト失敗: {str(e)}")
            return False

    def test_memory_and_performance(self):
        """メモリ使用量とパフォーマンスのテスト"""
        self.logger.info("=" * 50)
        self.logger.info("メモリ・パフォーマンステスト開始")

        try:
            import psutil
            import time

            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # エンサンブル予測器でストレステスト
            predictor = create_predictor(
                model_type=ModelType.ENSEMBLE,
                data_provider=self.data_provider
            )

            # 大量予測テスト
            test_symbols = ['7203', '9984', '8306', '9433', '8316'] * 10  # 50銘柄

            start_time = time.time()
            batch_results = predictor.predict_batch(test_symbols)
            execution_time = time.time() - start_time

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            self.logger.info(f"パフォーマンステスト結果:")
            self.logger.info(f"  処理銘柄数: {len(test_symbols)}")
            self.logger.info(f"  総実行時間: {execution_time:.2f}秒")
            self.logger.info(f"  銘柄あたり実行時間: {execution_time/len(test_symbols):.3f}秒")
            self.logger.info(f"  メモリ使用量増加: {memory_increase:.1f}MB")

            # 成功結果の確認
            successful_predictions = [r for r in batch_results if r.prediction > 0]
            success_rate = len(successful_predictions) / len(batch_results) * 100

            self.logger.info(f"  成功率: {success_rate:.1f}%")

            return success_rate > 50  # 50%以上成功すればOK

        except Exception as e:
            self.logger.error(f"メモリ・パフォーマンステスト失敗: {str(e)}")
            return False

    def run_all_tests(self):
        """全テスト実行"""
        self.logger.info("=" * 60)
        self.logger.info("統合リファクタリングシステム全テスト開始")
        self.logger.info("=" * 60)

        tests = [
            ("ファクトリパターン", self.test_factory_pattern),
            ("エンサンブル予測器", self.test_ensemble_predictor),
            ("並列特徴量計算", self.test_parallel_feature_calculator),
            ("統一インターフェース", self.test_unified_interfaces),
            ("メモリ・パフォーマンス", self.test_memory_and_performance)
        ]

        results = {}

        for test_name, test_func in tests:
            self.logger.info(f"\n{test_name}テスト実行中...")
            try:
                results[test_name] = test_func()
            except Exception as e:
                self.logger.error(f"{test_name}テスト実行中にエラー: {str(e)}")
                results[test_name] = False

        # 結果サマリー
        self.logger.info("=" * 60)
        self.logger.info("テスト結果サマリー")
        self.logger.info("=" * 60)

        passed = 0
        total = len(tests)

        for test_name, result in results.items():
            status = "PASS" if result else "FAIL"
            self.logger.info(f"{test_name:<20}: {status}")
            if result:
                passed += 1

        self.logger.info(f"\n総合結果: {passed}/{total} パス ({passed/total*100:.1f}%)")

        if passed == total:
            self.logger.info("🎉 全テスト成功！統合リファクタリングは正常に動作しています。")
        else:
            self.logger.warning("⚠️  一部テストが失敗しました。改善が必要です。")

        return passed == total


def main():
    """メイン実行"""
    print("統合リファクタリングシステムテスト")
    print("=" * 60)

    try:
        test_runner = RefactoredSystemTest()
        success = test_runner.run_all_tests()

        if success:
            print("\n✅ 統合リファクタリング完了 - システムは正常に動作しています")
            return 0
        else:
            print("\n❌ 統合リファクタリング不完全 - 追加修正が必要です")
            return 1

    except Exception as e:
        print(f"\n💥 テスト実行中に致命的エラー: {str(e)}")
        return 2


if __name__ == "__main__":
    exit(main())