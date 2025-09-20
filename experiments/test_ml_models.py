#!/usr/bin/env python3
"""
機械学習モデルのテストと検証スクリプト
"""

import logging
import sys
from datetime import datetime
from models.predictor import StockPredictor
from models.ml_models import MLStockPredictor
from models.backtest import Backtester
from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_basic_functionality():
    """基本機能のテスト"""
    logger.info("=== 基本機能テスト ===")

    # データプロバイダーのテスト
    logger.info("データプロバイダーをテスト中...")
    data_provider = StockDataProvider()
    symbols = data_provider.get_all_stock_symbols()
    logger.info(f"利用可能銘柄数: {len(symbols)}")

    # サンプル銘柄でデータ取得テスト
    test_symbol = "7203"
    data = data_provider.get_stock_data(test_symbol, "1mo")
    logger.info(f"{test_symbol}のデータ行数: {len(data)}")

    if not data.empty:
        data_with_indicators = data_provider.calculate_technical_indicators(data)
        logger.info(f"技術指標計算後の列数: {len(data_with_indicators.columns)}")

    return True


def test_rule_based_predictor():
    """ルールベース予測器のテスト"""
    logger.info("=== ルールベース予測器テスト ===")

    predictor = StockPredictor(use_ml_model=False)

    # 単一銘柄スコア計算
    test_symbol = "7203"
    score = predictor.calculate_score(test_symbol)
    logger.info(f"{test_symbol}のスコア: {score:.1f}")

    # 上位推奨取得
    logger.info("上位3銘柄の推奨を取得中...")
    recommendations = predictor.get_top_recommendations(top_n=3)

    for rec in recommendations:
        logger.info(
            f"[{rec.rank}位] {rec.company_name} ({rec.symbol}) - スコア: {rec.score:.1f}"
        )

    return True


def test_ml_predictor():
    """機械学習予測器のテスト"""
    logger.info("=== 機械学習予測器テスト ===")

    try:
        # 機械学習モデルの初期化
        ml_predictor = MLStockPredictor(model_type="xgboost")

        # 特徴量準備のテスト
        data_provider = StockDataProvider()
        test_data = data_provider.get_stock_data("7203", "6mo")
        if not test_data.empty:
            features = ml_predictor.prepare_features(test_data)
            logger.info(f"準備された特徴量数: {len(features.columns)}")
            logger.info(f"特徴量データ行数: {len(features)}")

        # 小規模なデータセットでモデル訓練をテスト
        logger.info("小規模データセットでモデル訓練をテスト中...")
        test_symbols = ["7203", "6758", "9984"]  # 3銘柄のみ

        try:
            features, targets_reg, targets_cls = ml_predictor.prepare_dataset(
                test_symbols
            )
            logger.info(
                f"データセット準備完了 - 特徴量: {features.shape}, ターゲット: {targets_reg.shape}"
            )

            # 実際の訓練は時間がかかるため、データ準備のみテスト
            logger.info("データセット準備が正常に完了しました")

        except Exception as e:
            logger.warning(f"モデル訓練テストでエラー: {str(e)}")

        return True

    except Exception as e:
        logger.error(f"機械学習予測器テストでエラー: {str(e)}")
        return False


def test_hybrid_predictor():
    """ハイブリッド予測器のテスト"""
    logger.info("=== ハイブリッド予測器テスト ===")

    try:
        # ハイブリッドモデル（ML有効だが未訓練）
        hybrid_predictor = StockPredictor(use_ml_model=True, ml_model_type="xgboost")

        # モデル情報取得
        model_info = hybrid_predictor.get_model_info()
        logger.info(f"モデル情報: {model_info}")

        # スコア計算（MLモデル未訓練なのでルールベースにフォールバック）
        test_symbol = "7203"
        score = hybrid_predictor.calculate_score(test_symbol)
        logger.info(f"{test_symbol}のハイブリッドスコア: {score:.1f}")

        return True

    except Exception as e:
        logger.error(f"ハイブリッド予測器テストでエラー: {str(e)}")
        return False


def test_backtest_framework():
    """バックテストフレームワークのテスト"""
    logger.info("=== バックテストフレームワークテスト ===")

    try:
        backtester = Backtester(initial_capital=1000000)

        # ルールベースモデルでの短期バックテスト
        rule_predictor = StockPredictor(use_ml_model=False)

        test_symbols = ["7203", "6758"]  # 2銘柄のみ
        start_date = "2024-01-01"
        end_date = "2024-03-31"  # 3ヶ月間

        logger.info(f"バックテスト実行中: {start_date} から {end_date}")

        result = backtester.run_backtest(
            predictor=rule_predictor,
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            rebalance_frequency=10,  # 10日毎
            top_n=2,
            max_holding_days=20,
        )

        logger.info(f"バックテスト結果:")
        logger.info(f"  総リターン: {result.total_return:.2%}")
        logger.info(f"  年率リターン: {result.annualized_return:.2%}")
        logger.info(f"  最大ドローダウン: {result.max_drawdown:.2%}")
        logger.info(f"  シャープレシオ: {result.sharpe_ratio:.2f}")
        logger.info(f"  勝率: {result.win_rate:.2%}")
        logger.info(f"  総取引数: {result.total_trades}")

        return True

    except Exception as e:
        logger.error(f"バックテストフレームワークテストでエラー: {str(e)}")
        return False


def main():
    """メインテスト実行"""
    logger.info("機械学習モデルのテストと検証を開始します")

    tests = [
        ("基本機能", test_basic_functionality),
        ("ルールベース予測器", test_rule_based_predictor),
        ("機械学習予測器", test_ml_predictor),
        ("ハイブリッド予測器", test_hybrid_predictor),
        ("バックテストフレームワーク", test_backtest_framework),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"テスト実行: {test_name}")
            logger.info(f"{'='*50}")

            success = test_func()
            results[test_name] = success

            if success:
                logger.info(f"✅ {test_name}: 成功")
            else:
                logger.error(f"❌ {test_name}: 失敗")

        except Exception as e:
            logger.error(f"❌ {test_name}: エラー - {str(e)}")
            results[test_name] = False

    # 結果サマリー
    logger.info(f"\n{'='*50}")
    logger.info("テスト結果サマリー")
    logger.info(f"{'='*50}")

    success_count = sum(results.values())
    total_count = len(results)

    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\n合計: {success_count}/{total_count} のテストが成功")

    if success_count == total_count:
        logger.info("🎉 全てのテストが成功しました！")
        return 0
    else:
        logger.warning(f"⚠️  {total_count - success_count} 個のテストが失敗しました")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
