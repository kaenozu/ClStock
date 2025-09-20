#!/usr/bin/env python3
"""
機械学習モデルの訓練スクリプト
"""

import logging
from utils.logger_config import setup_logger
from models.ml_models import MLStockPredictor
from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = setup_logger(__name__)


def main():
    """メイン実行関数"""
    logger.info("機械学習モデルの訓練を開始します")

    try:
        # データプロバイダーと予測器を初期化
        data_provider = StockDataProvider()
        ml_predictor = MLStockPredictor()

        # 対象銘柄を取得
        symbols = list(data_provider.jp_stock_codes.keys())
        logger.info(f"対象銘柄数: {len(symbols)}")

        # モデルを訓練
        logger.info("機械学習モデルを訓練中...")
        ml_predictor.train_model(symbols)

        logger.info("訓練完了!")

        # 特徴量の重要度を表示
        try:
            importance = ml_predictor.get_feature_importance()
            logger.info("特徴量の重要度:")
            for feature, score in importance.items():
                logger.info(f"  {feature}: {score:.4f}")
        except Exception as e:
            logger.warning(f"特徴量重要度の取得に失敗: {e}")

        # 各銘柄のスコア予測をテスト
        logger.info("\n各銘柄のスコア予測テスト:")
        for symbol in symbols[:5]:  # 最初の5銘柄のみテスト
            try:
                score = ml_predictor.predict_score(symbol)
                company_name = data_provider.jp_stock_codes[symbol]
                logger.info(f"{symbol} ({company_name}): {score:.1f}")
            except Exception as e:
                logger.error(f"{symbol}のスコア予測でエラー: {e}")

        logger.info("機械学習モデルの訓練が正常に完了しました！")

    except Exception as e:
        logger.error(f"訓練中にエラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()
