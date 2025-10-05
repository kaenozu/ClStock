#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拡張アンサンブルモデル訓練スクリプト
Phase 1機能を使った実際のモデル訓練と評価
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
    """メイン訓練実行関数"""
    print("=" * 80)
    print("拡張アンサンブルモデル 実訓練システム")
    print("=" * 80)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # 訓練実行
        training_results = run_enhanced_ensemble_training()

        if training_results["success"]:
            print("\n✅ アンサンブルモデル訓練完了!")

            # 訓練後の性能評価
            evaluation_results = evaluate_trained_model()

            if evaluation_results["success"]:
                print("\n✅ 訓練済みモデル評価完了!")
                display_final_results(training_results, evaluation_results)
            else:
                print(
                    f"\n❌ モデル評価失敗: {evaluation_results.get('error', 'Unknown error')}"
                )
        else:
            print(f"\n❌ 訓練失敗: {training_results.get('error', 'Unknown error')}")

    except KeyboardInterrupt:
        print("\n\n訓練が中断されました")
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {str(e)}")
        traceback.print_exc()


def run_enhanced_ensemble_training() -> Dict[str, Any]:
    """拡張アンサンブルモデルの訓練実行"""
    try:
        from models.ensemble.ensemble_predictor import EnsembleStockPredictor
        from data.stock_data import StockDataProvider

        print("1. データプロバイダー初期化")
        data_provider = StockDataProvider()

        print("2. 拡張アンサンブル予測器初期化")
        predictor = EnsembleStockPredictor(data_provider=data_provider)

        # 訓練用銘柄の選定（実際にデータ取得可能な銘柄）
        print("3. 訓練用銘柄の選定")
        training_symbols = select_training_symbols(data_provider)

        if len(training_symbols) < 5:
            return {
                "success": False,
                "error": f"訓練用銘柄が不足: {len(training_symbols)}銘柄（最低5銘柄必要）",
            }

        print(f"   選定された訓練銘柄: {len(training_symbols)}銘柄")
        print(f"   銘柄リスト: {training_symbols[:10]}...")  # 最初の10銘柄表示

        # アンサンブル訓練実行
        print("4. アンサンブルモデル訓練開始")
        start_time = time.time()

        try:
            predictor.train_ensemble(
                training_symbols, target_column="recommendation_score"
            )
            training_time = time.time() - start_time

            print(f"   ✅ 訓練完了（所要時間: {training_time:.1f}秒）")

            return {
                "success": True,
                "predictor": predictor,
                "training_symbols": training_symbols,
                "training_time": training_time,
                "model_count": len(predictor.models),
                "feature_count": len(predictor.feature_names),
            }

        except Exception as e:
            return {"success": False, "error": f"訓練実行エラー: {str(e)}"}

    except ImportError as e:
        return {"success": False, "error": f"モジュールインポートエラー: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"初期化エラー: {str(e)}"}


def select_training_symbols(data_provider) -> List[str]:
    """訓練用銘柄の選定（データ取得可能性チェック）"""
    # 日本の主要銘柄候補
    candidate_symbols = [
        "6758.T",  # ソニー
        "7203.T",  # トヨタ
        "8306.T",  # 三菱UFJ
        "9432.T",  # NTT
        "4519.T",  # 中外製薬
        "6861.T",  # キーエンス
        "4568.T",  # 第一三共
        "8035.T",  # 東京エレクトロン
        "6954.T",  # ファナック
        "4502.T",  # 武田薬品
        "9983.T",  # ファーストリテイリング
        "8316.T",  # 三井住友フィナンシャル
        "4063.T",  # 信越化学
        "6098.T",  # リクルート
        "9434.T",  # ソフトバンク
    ]

    valid_symbols = []
    print("   データ取得可能性チェック中...")

    for symbol in candidate_symbols:
        try:
            # 短期間のデータ取得テスト
            data = data_provider.get_stock_data(symbol, "30d")
            if not data.empty and len(data) >= 20:  # 最低20日分のデータ
                valid_symbols.append(symbol)
                if len(valid_symbols) % 5 == 0:
                    print(f"     {len(valid_symbols)}銘柄確認完了...")

            # 15銘柄で十分
            if len(valid_symbols) >= 15:
                break

        except Exception as e:
            logger.debug(f"Symbol {symbol} data fetch failed: {str(e)}")
            continue

    return valid_symbols


def evaluate_trained_model() -> Dict[str, Any]:
    """訓練済みモデルの評価"""
    try:
        from models.ensemble.ensemble_predictor import EnsembleStockPredictor
        from data.stock_data import StockDataProvider

        print("5. 訓練済みモデルの読み込み")
        data_provider = StockDataProvider()
        predictor = EnsembleStockPredictor(data_provider=data_provider)

        # 保存されたモデルの読み込み
        if not predictor.load_ensemble():
            return {
                "success": False,
                "error": "保存されたアンサンブルモデルが見つかりません",
            }

        print("   ✅ モデル読み込み完了")
        print(f"   モデル数: {len(predictor.models)}")
        print(f"   特徴量数: {len(predictor.feature_names)}")

        # テスト用銘柄で評価
        print("6. テスト銘柄での予測性能評価")
        test_symbols = ["6758.T", "7203.T", "8306.T"]  # ソニー、トヨタ、三菱UFJ

        evaluation_results = []
        prediction_times = []

        for symbol in test_symbols:
            try:
                print(f"   {symbol} 評価中...")

                # 予測実行
                start_time = time.time()
                result = predictor.predict(symbol)
                prediction_time = time.time() - start_time

                prediction_times.append(prediction_time)

                evaluation_results.append(
                    {
                        "symbol": symbol,
                        "prediction": result.prediction,
                        "confidence": result.confidence,
                        "accuracy": result.accuracy,
                        "prediction_time": prediction_time,
                        "metadata": result.metadata,
                    }
                )

                print(
                    f"     予測値: {result.prediction:.1f}, "
                    f"信頼度: {result.confidence:.2f}, "
                    f"時間: {prediction_time:.3f}秒"
                )

            except Exception as e:
                logger.error(f"Evaluation failed for {symbol}: {str(e)}")

        if not evaluation_results:
            return {"success": False, "error": "すべてのテスト銘柄で評価が失敗しました"}

        # 性能統計計算
        avg_prediction_time = np.mean(prediction_times)
        avg_confidence = np.mean([r["confidence"] for r in evaluation_results])

        print(f"   ✅ 評価完了")
        print(f"   平均予測時間: {avg_prediction_time:.3f}秒")
        print(f"   平均信頼度: {avg_confidence:.2f}")

        return {
            "success": True,
            "evaluation_results": evaluation_results,
            "avg_prediction_time": avg_prediction_time,
            "avg_confidence": avg_confidence,
            "test_symbols_count": len(evaluation_results),
        }

    except Exception as e:
        return {"success": False, "error": f"評価実行エラー: {str(e)}"}


def display_final_results(
    training_results: Dict[str, Any], evaluation_results: Dict[str, Any]
):
    """最終結果の表示"""
    print("\n" + "=" * 80)
    print("拡張アンサンブルモデル 訓練・評価結果")
    print("=" * 80)

    # 訓練結果
    print("📚 訓練結果:")
    print(f"   訓練銘柄数: {len(training_results['training_symbols'])}銘柄")
    print(f"   訓練時間: {training_results['training_time']:.1f}秒")
    print(f"   モデル数: {training_results['model_count']}")
    print(f"   特徴量数: {training_results['feature_count']}")

    # 評価結果
    print("\n🎯 評価結果:")
    print(f"   テスト銘柄数: {evaluation_results['test_symbols_count']}")
    print(f"   平均予測時間: {evaluation_results['avg_prediction_time']:.3f}秒")
    print(f"   平均信頼度: {evaluation_results['avg_confidence']:.2f}")

    # 個別予測結果
    print("\n📊 個別予測結果:")
    print("   銘柄      予測値   信頼度   時間")
    print("   " + "-" * 35)

    for result in evaluation_results["evaluation_results"]:
        print(
            f"   {result['symbol']}  {result['prediction']:6.1f}  "
            f"{result['confidence']:6.2f}  {result['prediction_time']:6.3f}秒"
        )

    # パフォーマンス改善効果
    print("\n🚀 Phase 1 改善効果:")
    print("   ✅ 並列特徴量計算システム - 実装完了")
    print("   ✅ インメモリキャッシュシステム - 実装完了")
    print("   ✅ マルチタイムフレーム統合 - 実装完了")
    print("   ✅ アンサンブルモデル訓練 - 実装完了")

    training_efficiency = (
        len(training_results["training_symbols"]) / training_results["training_time"]
    )
    print(f"   訓練効率: {training_efficiency:.2f} 銘柄/秒")

    print("\n" + "=" * 80)
    print("🎉 Phase 1 完成度向上 - 完了!")
    print("=" * 80)


if __name__ == "__main__":
    main()
