#!/usr/bin/env python3
"""実用的なシンプル予測システム（MAPE最適化）
"""

import numpy as np
from data.stock_data import StockDataProvider
from utils.logger_config import setup_logger

# ログ設定
logger = setup_logger(__name__)


class PracticalPredictor:
    """実用的なシンプル予測システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()

    def predict_return_rate(self, symbol: str) -> float:
        """シンプルで実用的なリターン率予測"""
        try:
            # データ取得
            data = self.data_provider.get_stock_data(symbol, "3mo")
            if data.empty or len(data) < 30:
                return 0.0

            data = self.data_provider.calculate_technical_indicators(data)

            # 過去のパフォーマンス分析
            returns = data["Close"].pct_change().dropna()

            # シンプルな予測ロジック
            current_price = data["Close"].iloc[-1]

            # 1. 短期移動平均トレンド（強力なシグナル）
            sma_5 = data["Close"].rolling(5).mean().iloc[-1]
            sma_20 = data["SMA_20"].iloc[-1]

            # 基本トレンド
            if current_price > sma_5 > sma_20:
                base_return = 0.008  # 0.8%の上昇予測
            elif current_price < sma_5 < sma_20:
                base_return = -0.006  # -0.6%の下降予測
            else:
                base_return = 0.001  # 0.1%の微増予測

            # 2. RSI による調整（逆張り要素）
            rsi = data["RSI"].iloc[-1]
            if rsi < 30:  # 過売り
                base_return += 0.005
            elif rsi > 70:  # 過買い
                base_return -= 0.004

            # 3. 出来高による確信度調整
            volume_ratio = (
                data["Volume"].iloc[-1] / data["Volume"].rolling(20).mean().iloc[-1]
            )
            if volume_ratio > 1.5:  # 高出来高で確信度アップ
                base_return *= 1.2
            elif volume_ratio < 0.8:  # 低出来高で控えめに
                base_return *= 0.8

            # 4. 最近のボラティリティによる調整
            recent_vol = returns.rolling(5).std().iloc[-1]
            if recent_vol > 0.03:  # 高ボラティリティ
                base_return *= 0.7  # 控えめに
            elif recent_vol < 0.01:  # 低ボラティリティ
                base_return *= 1.1  # やや積極的に

            # 現実的な範囲に制限
            return max(-0.04, min(0.04, base_return))

        except Exception as e:
            logger.error(f"Error predicting return rate for {symbol}: {e!s}")
            return 0.0

    def test_prediction_accuracy(self, symbols: list, test_days: int = 30) -> dict:
        """予測精度テスト"""
        results = {"predictions": [], "actuals": [], "symbols": [], "errors": []}

        for symbol in symbols[:5]:  # 最初の5銘柄
            try:
                # より長期のデータを取得
                data = self.data_provider.get_stock_data(symbol, "1y")
                if len(data) < 60:
                    continue

                # 過去30日分のテスト
                for i in range(test_days, 5, -5):  # 5日ずつ戻る
                    # i日前の時点でのデータで予測
                    historical_data = data.iloc[:-i].copy()

                    # この時点での予測（簡易計算）
                    if len(historical_data) < 30:
                        continue

                    # 実際のリターン（i日前から5日後まで）
                    start_price = data.iloc[-i]["Close"]
                    end_price = (
                        data.iloc[-i + 5]["Close"] if i > 5 else data.iloc[-1]["Close"]
                    )
                    actual_return = (end_price - start_price) / start_price

                    # 簡易予測（この時点での）
                    historical_data = self.data_provider.calculate_technical_indicators(
                        historical_data,
                    )
                    current_price = historical_data["Close"].iloc[-1]
                    sma_5 = historical_data["Close"].rolling(5).mean().iloc[-1]
                    sma_20 = historical_data["SMA_20"].iloc[-1]

                    # シンプル予測
                    if current_price > sma_5 > sma_20:
                        predicted_return = 0.008
                    elif current_price < sma_5 < sma_20:
                        predicted_return = -0.006
                    else:
                        predicted_return = 0.001

                    results["predictions"].append(predicted_return)
                    results["actuals"].append(actual_return)
                    results["symbols"].append(symbol)
                    results["errors"].append(abs(predicted_return - actual_return))

            except Exception as e:
                logger.warning(f"Error testing {symbol}: {e!s}")
                continue

        return results


def calculate_metrics(results: dict) -> dict:
    """メトリクス計算"""
    if not results["predictions"]:
        return {"error": "No valid predictions"}

    predictions = np.array(results["predictions"])
    actuals = np.array(results["actuals"])

    # MAPE
    mask = np.abs(actuals) > 0.001
    if mask.sum() > 0:
        mape = (
            np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
        )
    else:
        mape = float("inf")

    # 他のメトリクス
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # 方向性精度
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    directional_accuracy = (pred_direction == actual_direction).mean() * 100

    return {
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "total_predictions": len(predictions),
    }


def main():
    """メイン実行関数"""
    print("=" * 50)
    print("実用的なシンプル予測システムテスト")
    print("=" * 50)

    predictor = PracticalPredictor()
    data_provider = StockDataProvider()

    # 現在の予測テスト
    print("\n現在の予測テスト:")
    print("-" * 30)

    test_symbols = list(data_provider.jp_stock_codes.keys())[:5]
    for symbol in test_symbols:
        predicted_return = predictor.predict_return_rate(symbol)
        print(
            f"{symbol}: 予測リターン率 {predicted_return:.3f} ({predicted_return * 100:.1f}%)",
        )

    # 精度テスト
    print("\n過去データでの精度テスト:")
    print("-" * 30)

    results = predictor.test_prediction_accuracy(test_symbols, test_days=30)
    metrics = calculate_metrics(results)

    if "error" not in metrics:
        print(f"テスト予測数: {metrics['total_predictions']}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"方向性精度: {metrics['directional_accuracy']:.1f}%")

        print("\n評価:")
        if metrics["mape"] < 15:
            print("✓ 実用レベル達成!")
        elif metrics["mape"] < 25:
            print("△ 改善が必要")
        else:
            print("✗ 大幅な改善が必要")

    else:
        print("有効なテストデータが取得できませんでした")


if __name__ == "__main__":
    main()
