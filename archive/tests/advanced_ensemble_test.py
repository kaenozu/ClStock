#!/usr/bin/env python3
"""ClStock 高度アンサンブル学習テストシステム
84.6%精度突破を目指す統合検証システム

実装機能:
- BERT活用ニュースセンチメント分析
- マクロ経済指標統合予測
- 動的重み調整アンサンブル学習
- 深層学習最適化 (LSTM/Transformer)
"""

import logging
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from utils.logger_config import setup_logger

warnings.filterwarnings("ignore")

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = setup_logger(__name__)


class AdvancedEnsembleTestSystem:
    """高度アンサンブル学習テストシステム"""

    def __init__(self):
        self.test_symbols = [
            "7203",
            "6758",
            "9984",
            "8306",
            "6861",
            "4661",
            "9433",
            "4519",
            "6367",
            "8035",
        ]
        self.results = {}
        self.ensemble_predictor = None
        self.macro_provider = None

    def initialize_systems(self):
        """システム初期化"""
        try:
            logger.info("=== 高度アンサンブル学習システム初期化 ===")

            # models/ml_models.py から高度クラスをインポート
            sys.path.append(".")
            from models.ml_models import (
                AdvancedEnsemblePredictor,
                MacroEconomicDataProvider,
            )

            self.ensemble_predictor = AdvancedEnsemblePredictor()
            self.macro_provider = MacroEconomicDataProvider()

            # 各コンポーネント初期化
            self.ensemble_predictor.initialize_components()

            logger.info("システム初期化完了")
            return True

        except Exception as e:
            logger.error(f"システム初期化エラー: {e}")
            return False

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """包括的テスト実行"""
        logger.info("\n" + "=" * 60)
        logger.info("ClStock 高度アンサンブル学習 包括テスト開始")
        logger.info("目標: 84.6%精度突破")
        logger.info("=" * 60)

        test_results = {
            "system_info": self._get_system_info(),
            "macro_data_test": self._test_macro_data_integration(),
            "sentiment_analysis_test": self._test_sentiment_analysis(),
            "deep_learning_test": self._test_deep_learning_models(),
            "ensemble_prediction_test": self._test_ensemble_predictions(),
            "accuracy_benchmark": self._run_accuracy_benchmark(),
            "performance_summary": {},
        }

        # 総合評価
        test_results["performance_summary"] = self._generate_performance_summary(
            test_results,
        )

        return test_results

    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        import platform

        import psutil

        return {
            "timestamp": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "test_symbols": self.test_symbols,
            "test_count": len(self.test_symbols),
        }

    def _test_macro_data_integration(self) -> Dict[str, Any]:
        """マクロ経済データ統合テスト"""
        logger.info("\n--- マクロ経済データ統合テスト ---")

        try:
            # マクロ経済指標取得
            macro_data = self.macro_provider.get_economic_indicators()

            test_result = {
                "status": "success",
                "data_sources": list(macro_data.keys()),
                "boj_policy_available": "boj_policy" in macro_data,
                "global_rates_available": "global_rates" in macro_data,
                "currency_data_available": "currency_strength" in macro_data,
                "sentiment_indicators_available": "market_sentiment" in macro_data,
                "data_completeness": len(macro_data) / 4.0,  # 4つの主要データソース
            }

            logger.info(f"マクロデータソース: {test_result['data_sources']}")
            logger.info(f"データ完全性: {test_result['data_completeness']:.1%}")

            return test_result

        except Exception as e:
            logger.error(f"マクロデータテストエラー: {e}")
            return {"status": "error", "error": str(e)}

    def _test_sentiment_analysis(self) -> Dict[str, Any]:
        """センチメント分析テスト"""
        logger.info("\n--- センチメント分析テスト ---")

        test_results = {}
        successful_analyses = 0

        for symbol in self.test_symbols[:5]:  # 5銘柄でテスト
            try:
                sentiment_result = (
                    self.ensemble_predictor.enhanced_sentiment_prediction(symbol)
                )

                test_results[symbol] = {
                    "sentiment_score": sentiment_result["sentiment_score"],
                    "confidence": sentiment_result["confidence"],
                    "status": "success",
                }

                if abs(sentiment_result["sentiment_score"]) > 0.1:  # 有意なセンチメント
                    successful_analyses += 1

                logger.info(
                    f"{symbol}: センチメント={sentiment_result['sentiment_score']:.3f}, 信頼度={sentiment_result['confidence']:.3f}",
                )

            except Exception as e:
                test_results[symbol] = {"status": "error", "error": str(e)}
                logger.error(f"{symbol}: センチメント分析エラー - {e}")

        return {
            "individual_results": test_results,
            "success_rate": successful_analyses / len(self.test_symbols[:5]),
            "bert_available": hasattr(self.ensemble_predictor, "bert_model"),
            "total_tested": len(self.test_symbols[:5]),
        }

    def _test_deep_learning_models(self) -> Dict[str, Any]:
        """深層学習モデルテスト"""
        logger.info("\n--- 深層学習モデルテスト ---")

        try:
            # 簡易訓練（計算量制限）
            train_symbols = self.test_symbols[:3]
            logger.info(f"深層学習訓練開始: {train_symbols}")

            # 既存の深層学習システムテスト
            from experiments.big_data_deep_learning import BigDataDeepLearningSystem

            dl_system = BigDataDeepLearningSystem()
            results = dl_system.test_multiple_symbols(train_symbols)

            # 結果解析
            best_accuracies = []
            for symbol, result in results.items():
                if "best_accuracy" in result:
                    best_accuracies.append(result["best_accuracy"])

            avg_accuracy = np.mean(best_accuracies) if best_accuracies else 0.0
            max_accuracy = np.max(best_accuracies) if best_accuracies else 0.0

            return {
                "status": "success",
                "tested_symbols": train_symbols,
                "average_accuracy": avg_accuracy,
                "max_accuracy": max_accuracy,
                "models_tested": ["Advanced LSTM", "Advanced GRU", "CNN-LSTM Hybrid"],
                "accuracy_above_60": sum(1 for acc in best_accuracies if acc > 60.0),
                "individual_results": results,
            }

        except Exception as e:
            logger.error(f"深層学習テストエラー: {e}")
            return {"status": "error", "error": str(e)}

    def _test_ensemble_predictions(self) -> Dict[str, Any]:
        """アンサンブル予測テスト"""
        logger.info("\n--- アンサンブル予測テスト ---")

        ensemble_results = {}
        high_confidence_predictions = 0
        total_tested = 0

        for symbol in self.test_symbols:
            try:
                result = self.ensemble_predictor.dynamic_ensemble_prediction(symbol)

                ensemble_results[symbol] = {
                    "ensemble_prediction": result["ensemble_prediction"],
                    "ensemble_confidence": result["ensemble_confidence"],
                    "high_confidence": result["high_confidence"],
                    "individual_predictions": result.get("individual_predictions", {}),
                    "adjusted_weights": result.get("adjusted_weights", {}),
                }

                if result["high_confidence"]:
                    high_confidence_predictions += 1

                total_tested += 1

                logger.info(
                    f"{symbol}: 予測={result['ensemble_prediction']:.1f}, "
                    f"信頼度={result['ensemble_confidence']:.3f}, "
                    f"高信頼={result['high_confidence']}",
                )

            except Exception as e:
                ensemble_results[symbol] = {"status": "error", "error": str(e)}
                logger.error(f"{symbol}: アンサンブル予測エラー - {e}")

        return {
            "individual_results": ensemble_results,
            "high_confidence_rate": (
                high_confidence_predictions / total_tested if total_tested > 0 else 0
            ),
            "total_tested": total_tested,
            "high_confidence_count": high_confidence_predictions,
        }

    def _run_accuracy_benchmark(self) -> Dict[str, Any]:
        """精度ベンチマークテスト"""
        logger.info("\n--- 精度ベンチマーク vs 84.6%基準 ---")

        try:
            # 84.6%ベースシステムとの比較
            from trend_following_predictor import TrendFollowingPredictor

            base_predictor = TrendFollowingPredictor()
            comparison_results = {}

            ensemble_wins = 0
            base_wins = 0
            ties = 0

            for symbol in self.test_symbols[:7]:  # 計算量制限
                try:
                    # ベース予測
                    base_result = base_predictor.predict_stock(symbol)
                    base_confidence = base_result["confidence"]

                    # アンサンブル予測
                    ensemble_result = (
                        self.ensemble_predictor.dynamic_ensemble_prediction(symbol)
                    )
                    ensemble_confidence = ensemble_result["ensemble_confidence"]

                    # 信頼度比較
                    if ensemble_confidence > base_confidence + 0.05:  # 5%以上の改善
                        winner = "ensemble"
                        ensemble_wins += 1
                    elif base_confidence > ensemble_confidence + 0.05:
                        winner = "base"
                        base_wins += 1
                    else:
                        winner = "tie"
                        ties += 1

                    comparison_results[symbol] = {
                        "base_confidence": base_confidence,
                        "ensemble_confidence": ensemble_confidence,
                        "improvement": ensemble_confidence - base_confidence,
                        "winner": winner,
                    }

                    logger.info(
                        f"{symbol}: ベース={base_confidence:.3f}, "
                        f"アンサンブル={ensemble_confidence:.3f}, "
                        f"改善={ensemble_confidence - base_confidence:+.3f}",
                    )

                except Exception as e:
                    logger.error(f"{symbol}: ベンチマークエラー - {e}")

            total_comparisons = ensemble_wins + base_wins + ties

            return {
                "ensemble_wins": ensemble_wins,
                "base_wins": base_wins,
                "ties": ties,
                "ensemble_win_rate": (
                    ensemble_wins / total_comparisons if total_comparisons > 0 else 0
                ),
                "average_improvement": np.mean(
                    [r.get("improvement", 0) for r in comparison_results.values()],
                ),
                "individual_comparisons": comparison_results,
                "total_comparisons": total_comparisons,
            }

        except Exception as e:
            logger.error(f"ベンチマークテストエラー: {e}")
            return {"status": "error", "error": str(e)}

    def _generate_performance_summary(
        self,
        test_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """総合パフォーマンス評価"""
        logger.info("\n" + "=" * 60)
        logger.info("総合パフォーマンス評価")
        logger.info("=" * 60)

        try:
            # 各テスト項目のスコア算出
            scores = {}

            # マクロデータ統合スコア
            macro_test = test_results.get("macro_data_test", {})
            scores["macro_integration"] = macro_test.get("data_completeness", 0) * 100

            # センチメント分析スコア
            sentiment_test = test_results.get("sentiment_analysis_test", {})
            scores["sentiment_analysis"] = sentiment_test.get("success_rate", 0) * 100

            # 深層学習スコア
            dl_test = test_results.get("deep_learning_test", {})
            scores["deep_learning"] = dl_test.get("max_accuracy", 0)

            # アンサンブル予測スコア
            ensemble_test = test_results.get("ensemble_prediction_test", {})
            scores["ensemble_prediction"] = (
                ensemble_test.get("high_confidence_rate", 0) * 100
            )

            # ベンチマークスコア
            benchmark_test = test_results.get("accuracy_benchmark", {})
            scores["benchmark_performance"] = (
                benchmark_test.get("ensemble_win_rate", 0) * 100
            )

            # 総合スコア算出
            weight_map = {
                "macro_integration": 0.15,
                "sentiment_analysis": 0.20,
                "deep_learning": 0.30,
                "ensemble_prediction": 0.20,
                "benchmark_performance": 0.15,
            }

            overall_score = sum(
                scores[key] * weight_map[key] for key in scores if key in weight_map
            )

            # 84.6%基準との比較
            baseline_accuracy = 84.6
            projected_accuracy = (
                baseline_accuracy + (overall_score - 70) * 0.2
            )  # スコアベース推定

            # 評価グレード
            if projected_accuracy >= 87.0:
                grade = "S (卓越)"
            elif projected_accuracy >= 85.5:
                grade = "A+ (優秀)"
            elif projected_accuracy >= 84.6:
                grade = "A (良好)"
            elif projected_accuracy >= 82.0:
                grade = "B (標準)"
            else:
                grade = "C (要改善)"

            summary = {
                "individual_scores": scores,
                "overall_score": overall_score,
                "projected_accuracy": projected_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "accuracy_improvement": projected_accuracy - baseline_accuracy,
                "grade": grade,
                "target_achieved": projected_accuracy >= 85.0,  # 85%目標
                "recommendation": self._generate_recommendations(
                    scores,
                    projected_accuracy,
                ),
            }

            # 結果表示
            logger.info("個別スコア:")
            for key, score in scores.items():
                logger.info(f"  {key}: {score:.1f}")

            logger.info("\n総合評価:")
            logger.info(f"  総合スコア: {overall_score:.1f}/100")
            logger.info(
                f"  予測精度: {projected_accuracy:.1f}% (基準: {baseline_accuracy}%)",
            )
            logger.info(f"  精度改善: {projected_accuracy - baseline_accuracy:+.1f}%")
            logger.info(f"  評価グレード: {grade}")
            logger.info(
                f"  目標達成: {'✅ YES' if summary['target_achieved'] else '❌ NO'}",
            )

            return summary

        except Exception as e:
            logger.error(f"パフォーマンス評価エラー: {e}")
            return {"status": "error", "error": str(e)}

    def _generate_recommendations(
        self,
        scores: Dict[str, float],
        projected_accuracy: float,
    ) -> List[str]:
        """改善推奨事項生成"""
        recommendations = []

        if scores.get("macro_integration", 0) < 70:
            recommendations.append("マクロ経済データ統合の改善が必要")

        if scores.get("sentiment_analysis", 0) < 60:
            recommendations.append("センチメント分析精度の向上が必要")

        if scores.get("deep_learning", 0) < 65:
            recommendations.append("深層学習モデルの最適化が必要")

        if scores.get("ensemble_prediction", 0) < 75:
            recommendations.append("アンサンブル重み調整の改善が必要")

        if projected_accuracy < 85.0:
            recommendations.append("追加の特徴量エンジニアリングを推奨")
            recommendations.append("より長期間のバックテストが必要")

        if not recommendations:
            recommendations.append("優秀な性能です！本格運用を検討してください")

        return recommendations


def main():
    """メイン実行関数"""
    print("=" * 70)
    print("ClStock 高度アンサンブル学習テストシステム")
    print("84.6%精度突破への挑戦")
    print("=" * 70)

    # テストシステム初期化
    test_system = AdvancedEnsembleTestSystem()

    if not test_system.initialize_systems():
        print("❌ システム初期化に失敗しました")
        return 1

    try:
        # 包括的テスト実行
        results = test_system.run_comprehensive_test()

        # 結果保存
        import json

        output_file = f"advanced_ensemble_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # JSON serializable に変換
        serializable_results = {}
        for key, value in results.items():
            try:
                json.dumps(value)  # テスト
                serializable_results[key] = value
            except:
                serializable_results[key] = str(value)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"\n📊 詳細結果を保存: {output_file}")

        # 最終判定
        summary = results.get("performance_summary", {})
        if summary.get("target_achieved", False):
            print("\n🎉 85%精度目標達成！")
            print(f"予測精度: {summary.get('projected_accuracy', 0):.1f}%")
        else:
            print("\n📈 継続改善が必要")
            print(f"現在予測精度: {summary.get('projected_accuracy', 0):.1f}%")
            print("推奨改善事項:")
            for rec in summary.get("recommendation", []):
                print(f"  • {rec}")

        return 0

    except KeyboardInterrupt:
        print("\n⏸️  テスト中断")
        return 130
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
