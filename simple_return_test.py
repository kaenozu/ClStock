#!/usr/bin/env python3
"""
シンプルなリターン率予測テスト
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from data.stock_data import StockDataProvider
from models.predictor import StockPredictor

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_return_prediction():
    """シンプルなリターン率予測テスト"""

    print("=" * 50)
    print("シンプルなリターン率予測テスト")
    print("=" * 50)

    # 予測器初期化
    predictor = StockPredictor(prediction_days=5)
    data_provider = StockDataProvider()

    # テスト銘柄（最初の5銘柄）
    test_symbols = list(data_provider.jp_stock_codes.keys())[:5]

    predictions = []
    actuals = []
    symbols = []

    for symbol in test_symbols:
        try:
            print(f"\n{symbol} ({data_provider.jp_stock_codes[symbol]}) をテスト中...")

            # 予測実行
            predicted_return = predictor.predict_return_rate(symbol)
            print(f"予測リターン率: {predicted_return:.3f} ({predicted_return*100:.1f}%)")

            # 価格ターゲット予測
            current_price, target_price = predictor.predict_price_target(symbol)
            print(f"現在価格: {current_price:.0f}円")
            print(f"予測価格: {target_price:.0f}円")

            # 実際のリターンを計算（過去のデータから）
            data = data_provider.get_stock_data(symbol, "6mo")
            if len(data) >= 10:
                # 過去10営業日前から5営業日前までの実績リターン
                start_idx = len(data) - 10
                end_idx = len(data) - 5

                start_price = data.iloc[start_idx]['Close']
                end_price = data.iloc[end_idx]['Close']
                actual_return = (end_price - start_price) / start_price

                print(f"実績リターン率（参考）: {actual_return:.3f} ({actual_return*100:.1f}%)")

                predictions.append(predicted_return)
                actuals.append(actual_return)
                symbols.append(symbol)

                # 誤差計算
                error = abs(predicted_return - actual_return)
                print(f"絶対誤差: {error:.3f}")

        except Exception as e:
            print(f"エラー: {str(e)}")
            continue

    # 全体の統計
    if predictions:
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        print("\n" + "=" * 50)
        print("全体統計")
        print("=" * 50)

        # 基本統計
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        # MAPE計算（ゼロ除算回避）
        mask = np.abs(actuals) > 0.001
        if mask.sum() > 0:
            mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
        else:
            mape = float('inf')

        # 方向性精度
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        directional_accuracy = (pred_direction == actual_direction).mean() * 100

        print(f"テスト銘柄数: {len(predictions)}")
        print(f"MAE (平均絶対誤差): {mae:.4f}")
        print(f"RMSE (二乗平均平方根誤差): {rmse:.4f}")
        print(f"MAPE (平均絶対パーセント誤差): {mape:.2f}%")
        print(f"方向性精度: {directional_accuracy:.1f}%")

        print(f"\n予測範囲: {predictions.min():.3f} ～ {predictions.max():.3f}")
        print(f"実績範囲: {actuals.min():.3f} ～ {actuals.max():.3f}")
        print(f"予測平均: {predictions.mean():.3f}")
        print(f"実績平均: {actuals.mean():.3f}")

        # 評価
        print(f"\n" + "=" * 50)
        print("評価結果")
        print("=" * 50)

        if mape < 8:
            print("✓ プロフェッショナル使用レベル (MAPE < 8%)")
        elif mape < 15:
            print("✓ 実用レベル (MAPE < 15%)")
        elif mape < 25:
            print("△ 改善が必要 (MAPE < 25%)")
        else:
            print("✗ 大幅な改善が必要 (MAPE >= 25%)")

        if directional_accuracy > 60:
            print("✓ 優秀な方向性予測精度")
        elif directional_accuracy > 52:
            print("✓ 良好な方向性予測精度")
        else:
            print("△ 方向性予測精度の改善が必要")

    else:
        print("有効なテストデータが取得できませんでした。")

if __name__ == "__main__":
    test_simple_return_prediction()