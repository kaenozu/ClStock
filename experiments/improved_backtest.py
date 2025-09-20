#!/usr/bin/env python3
"""
改善されたバックテストスクリプト
- 長期間のテスト
- 調整されたスコア閾値
- 機械学習モデルとルールベースモデルの比較
"""

import logging
from models.backtest import Backtester
from models.predictor import StockPredictor
from models.ml_models import MLStockPredictor
from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_improved_backtest():
    """改善されたバックテストを実行"""
    logger.info("=== 改善されたバックテストの実行 ===")

    try:
        # データプロバイダーと対象銘柄
        data_provider = StockDataProvider()
        symbols = list(data_provider.jp_stock_codes.keys())

        # バックテスターを初期化
        backtester = Backtester(initial_capital=1000000)

        # 予測器を準備
        rule_predictor = StockPredictor(use_ml_model=False)

        # 機械学習予測器（訓練済みモデルを読み込み）
        try:
            ml_predictor = MLStockPredictor()
            ml_predictor.load_model()
            logger.info("訓練済み機械学習モデルを読み込みました")
        except Exception as e:
            logger.warning(f"機械学習モデルの読み込みに失敗: {e}")
            ml_predictor = None

        # テスト設定
        configs = [
            {
                "name": "長期間ルールベース（スコア閾値50）",
                "predictor": rule_predictor,
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "score_threshold": 50,
                "rebalance_frequency": 7,
                "top_n": 3,
                "max_holding_days": 45,
            },
            {
                "name": "長期間ルールベース（スコア閾値40）",
                "predictor": rule_predictor,
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "score_threshold": 40,
                "rebalance_frequency": 7,
                "top_n": 3,
                "max_holding_days": 45,
            },
        ]

        # 機械学習モデルが利用可能な場合、追加
        if ml_predictor and ml_predictor.is_trained:
            configs.append(
                {
                    "name": "長期間機械学習（スコア閾値45）",
                    "predictor": ml_predictor,
                    "start_date": "2023-01-01",
                    "end_date": "2024-12-31",
                    "score_threshold": 45,
                    "rebalance_frequency": 7,
                    "top_n": 3,
                    "max_holding_days": 45,
                }
            )

        results = {}

        for config in configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"テスト実行: {config['name']}")
            logger.info(f"{'='*60}")

            try:
                # バックテスト期間の修正 - より現実的な期間に
                result = backtester.run_backtest(
                    predictor=config["predictor"],
                    symbols=symbols[:5],  # 最初の5銘柄でテスト（パフォーマンス向上）
                    start_date=config["start_date"],
                    end_date=config["end_date"],
                    rebalance_frequency=config["rebalance_frequency"],
                    top_n=config["top_n"],
                    max_holding_days=config["max_holding_days"],
                    score_threshold=config["score_threshold"],  # スコア閾値を設定
                )

                results[config["name"]] = result

                # 結果を表示
                logger.info(f"バックテスト結果:")
                logger.info(f"  総リターン: {result.total_return:.2%}")
                logger.info(f"  年率リターン: {result.annualized_return:.2%}")
                logger.info(f"  最大ドローダウン: {result.max_drawdown:.2%}")
                logger.info(f"  シャープレシオ: {result.sharpe_ratio:.2f}")
                logger.info(f"  勝率: {result.win_rate:.2%}")
                logger.info(f"  総取引数: {result.total_trades}")
                logger.info(f"  平均保有日数: {result.avg_holding_days:.1f}日")
                logger.info(f"  最良取引: {result.best_trade:.2%}")
                logger.info(f"  最悪取引: {result.worst_trade:.2%}")

            except Exception as e:
                logger.error(f"{config['name']}でエラー: {str(e)}")
                continue

        # 結果比較
        logger.info(f"\n{'='*60}")
        logger.info("結果比較")
        logger.info(f"{'='*60}")

        for name, result in results.items():
            logger.info(f"{name}:")
            logger.info(f"  年率リターン: {result.annualized_return:.2%}")
            logger.info(f"  シャープレシオ: {result.sharpe_ratio:.2f}")
            logger.info(f"  最大ドローダウン: {result.max_drawdown:.2%}")
            logger.info(f"  取引数: {result.total_trades}")
            logger.info("")

        return results

    except Exception as e:
        logger.error(f"バックテスト実行中にエラー: {str(e)}")
        raise


def modify_score_threshold():
    """スコア閾値を変更してバックテストの取引機会を増やす"""
    logger.info("=== スコア閾値の調整 ===")

    # バックテストコードでスコア閾値を変更
    # models/backtest.py の _rebalance_portfolio メソッドを確認
    logger.info("現在のスコア閾値: 60")
    logger.info("推奨される新しい閾値: 50, 45, 40")
    logger.info("より多くの取引機会を得るため、閾値を下げることを検討してください")


if __name__ == "__main__":
    try:
        # スコア閾値の調整に関する情報
        modify_score_threshold()

        # 改善されたバックテストを実行
        results = run_improved_backtest()

        logger.info("改善されたバックテストが正常に完了しました！")

    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        raise
