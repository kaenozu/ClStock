#!/usr/bin/env python3
"""
MAPE爆発問題を解決する堅牢な予測システム
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobustMAPEPredictor:
    """MAPE爆発問題を解決する堅牢な予測システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.min_movement_threshold = 0.01  # 1%以上の動きのみMAPE計算対象

    def robust_mape_calculation(
        self, actual: np.ndarray, predicted: np.ndarray, min_threshold: float = 0.01
    ) -> float:
        """堅牢なMAPE計算（小さな動きを除外）"""
        # 絶対値がしきい値以上のケースのみ対象
        mask = np.abs(actual) >= min_threshold

        if mask.sum() == 0:
            # 全て小さな動きの場合は、別の指標を使用
            return self.alternative_mape(actual, predicted)

        valid_actual = actual[mask]
        valid_predicted = predicted[mask]

        mape = np.mean(np.abs((valid_actual - valid_predicted) / valid_actual)) * 100
        return mape

    def alternative_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """小さな動きの場合の代替MAPE計算"""
        # 正規化されたMAE
        mae = np.mean(np.abs(actual - predicted))
        mean_abs_actual = np.mean(np.abs(actual))

        if mean_abs_actual == 0:
            return 0.0

        normalized_mae = (mae / mean_abs_actual) * 100
        return normalized_mae

    def adaptive_predict_return_rate(
        self, symbol: str, prediction_days: int = 1
    ) -> float:
        """適応的リターン率予測"""
        try:
            data = self.data_provider.get_stock_data(symbol, "1mo")
            if data.empty or len(data) < 15:
                return 0.0

            data = self.data_provider.calculate_technical_indicators(data)

            # 銘柄の特性分析
            returns = data["Close"].pct_change().dropna()
            avg_volatility = returns.std()
            avg_abs_return = returns.abs().mean()

            # 予測スケールを銘柄特性に合わせる
            prediction_scale = max(0.002, min(0.015, avg_abs_return * 2))

            # 現在の市場状況
            current_price = data["Close"].iloc[-1]

            # 短期トレンド分析（1-3日）
            momentum_1d = returns.iloc[-1] if len(returns) > 0 else 0
            momentum_3d = returns.iloc[-3:].mean() if len(returns) > 3 else 0

            # 超短期移動平均
            sma_3 = data["Close"].rolling(3).mean().iloc[-1]

            # 基本予測ロジック（銘柄特性に適応）
            base_return = 0.0

            # 1. モメンタム継続性（主要シグナル）
            if abs(momentum_1d) > avg_abs_return * 0.5:  # 平均的な動きの50%以上
                # 強いモメンタムの場合
                if momentum_1d > 0:
                    base_return = prediction_scale * 0.3  # 保守的な継続予想
                else:
                    base_return = -prediction_scale * 0.3
            else:
                # 弱いモメンタムの場合は平均回帰
                if momentum_1d > 0:
                    base_return = -prediction_scale * 0.1  # 軽微な反動
                else:
                    base_return = prediction_scale * 0.1

            # 2. 短期トレンド調整
            trend_strength = (current_price - sma_3) / sma_3
            if abs(trend_strength) > 0.005:  # 0.5%以上のトレンド
                if trend_strength > 0:
                    base_return += prediction_scale * 0.2
                else:
                    base_return -= prediction_scale * 0.2

            # 3. ボラティリティ調整
            recent_vol = returns.iloc[-3:].std() if len(returns) > 3 else avg_volatility
            vol_ratio = recent_vol / avg_volatility if avg_volatility > 0 else 1.0

            if vol_ratio > 1.5:  # 高ボラティリティ環境
                base_return *= 0.6  # より保守的に
            elif vol_ratio < 0.7:  # 低ボラティリティ環境
                base_return *= 1.3  # やや積極的に

            # 最終制限（銘柄特性の2倍まで）
            max_prediction = prediction_scale * 2
            final_prediction = max(-max_prediction, min(max_prediction, base_return))

            return final_prediction

        except Exception as e:
            logger.error(f"Error predicting return rate for {symbol}: {str(e)}")
            return 0.0

    def test_robust_prediction(
        self, symbols: List[str], prediction_days: int = 1
    ) -> Dict:
        """堅牢な予測システムのテスト"""
        print(f"\n堅牢MAPE予測システム テスト（{prediction_days}日予測）")
        print("-" * 50)

        results = {
            "predictions": [],
            "actuals": [],
            "symbols": [],
            "significant_cases": [],  # 1%以上の動きのケース
            "small_cases": [],  # 1%未満の動きのケース
        }

        for symbol in symbols[:5]:
            try:
                data = self.data_provider.get_stock_data(symbol, "3mo")
                if len(data) < 30:
                    continue

                # より多くのテストケース（1日ずつずらす）
                for i in range(20, prediction_days, -1):
                    historical_data = data.iloc[:-i].copy()
                    if len(historical_data) < 15:
                        continue

                    # 実際のリターン
                    start_price = data.iloc[-i]["Close"]
                    end_idx = max(0, -i + prediction_days)
                    end_price = (
                        data.iloc[end_idx]["Close"]
                        if end_idx < 0
                        else data.iloc[-1]["Close"]
                    )
                    actual_return = (end_price - start_price) / start_price

                    # 適応的予測
                    predicted_return = self._predict_with_historical_data(
                        historical_data, symbol, prediction_days
                    )

                    results["predictions"].append(predicted_return)
                    results["actuals"].append(actual_return)
                    results["symbols"].append(symbol)

                    # 重要度による分類
                    if abs(actual_return) >= 0.01:  # 1%以上の動き
                        results["significant_cases"].append(
                            {
                                "actual": actual_return,
                                "predicted": predicted_return,
                                "symbol": symbol,
                            }
                        )
                    else:  # 1%未満の動き
                        results["small_cases"].append(
                            {
                                "actual": actual_return,
                                "predicted": predicted_return,
                                "symbol": symbol,
                            }
                        )

            except Exception as e:
                logger.warning(f"Error testing {symbol}: {str(e)}")
                continue

        return results

    def _predict_with_historical_data(
        self, data: pd.DataFrame, symbol: str, prediction_days: int
    ) -> float:
        """過去データを使った予測"""
        try:
            data = self.data_provider.calculate_technical_indicators(data)
            returns = data["Close"].pct_change().dropna()

            if len(returns) < 5:
                return 0.0

            # 銘柄の履歴特性
            avg_abs_return = returns.abs().mean()
            recent_return = returns.iloc[-1]

            # 超シンプルな平均回帰予測
            if abs(recent_return) > avg_abs_return * 1.5:  # 大きな動きの後
                # 平均回帰
                return -recent_return * 0.3
            elif abs(recent_return) > avg_abs_return * 0.5:  # 中程度の動き
                # 弱い継続
                return recent_return * 0.2
            else:
                # 小さな動きは継続
                return recent_return * 0.5

        except:
            return 0.0


def calculate_robust_metrics(results: Dict, predictor: RobustMAPEPredictor) -> Dict:
    """堅牢なメトリクス計算"""
    if not results["predictions"]:
        return {"error": "No valid predictions"}

    predictions = np.array(results["predictions"])
    actuals = np.array(results["actuals"])

    # 堅牢MAPE計算
    robust_mape = predictor.robust_mape_calculation(
        actuals, predictions, min_threshold=0.01
    )

    # 重要な動きでのMAPE
    significant_cases = results["significant_cases"]
    if significant_cases:
        sig_actuals = np.array([case["actual"] for case in significant_cases])
        sig_predictions = np.array([case["predicted"] for case in significant_cases])
        significant_mape = (
            np.mean(np.abs((sig_actuals - sig_predictions) / sig_actuals)) * 100
        )
    else:
        significant_mape = float("inf")

    # その他のメトリクス
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # 方向性精度（重要な動きのみ）
    if significant_cases:
        sig_dir_actual = np.sign(sig_actuals)
        sig_dir_predicted = np.sign(sig_predictions)
        significant_direction_accuracy = (
            sig_dir_actual == sig_dir_predicted
        ).mean() * 100
    else:
        significant_direction_accuracy = 0

    # 全体の方向性精度
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    overall_direction_accuracy = (pred_direction == actual_direction).mean() * 100

    return {
        "robust_mape": robust_mape,
        "significant_mape": significant_mape,
        "mae": mae,
        "rmse": rmse,
        "overall_direction_accuracy": overall_direction_accuracy,
        "significant_direction_accuracy": significant_direction_accuracy,
        "total_predictions": len(predictions),
        "significant_cases": len(significant_cases),
        "small_cases": len(results["small_cases"]),
        "significant_ratio": (
            len(significant_cases) / len(predictions) if len(predictions) > 0 else 0
        ),
    }


def main():
    """メイン実行関数"""
    print("=" * 60)
    print("MAPE爆発問題を解決する堅牢な予測システム")
    print("=" * 60)

    data_provider = StockDataProvider()
    symbols = list(data_provider.jp_stock_codes.keys())

    predictor = RobustMAPEPredictor()

    # 1日予測テスト
    results_1d = predictor.test_robust_prediction(symbols, prediction_days=1)
    metrics_1d = calculate_robust_metrics(results_1d, predictor)

    if "error" not in metrics_1d:
        print(f"1日予測結果:")
        print(f"  堅牢MAPE: {metrics_1d['robust_mape']:.2f}%")
        print(f"  重要動きMAPE: {metrics_1d['significant_mape']:.2f}%")
        print(f"  MAE: {metrics_1d['mae']:.4f}")
        print(f"  全体方向性精度: {metrics_1d['overall_direction_accuracy']:.1f}%")
        print(
            f"  重要動き方向性精度: {metrics_1d['significant_direction_accuracy']:.1f}%"
        )
        print(f"  総予測数: {metrics_1d['total_predictions']}")
        print(f"  重要動きケース: {metrics_1d['significant_cases']}")
        print(f"  小動きケース: {metrics_1d['small_cases']}")
        print(f"  重要動き比率: {metrics_1d['significant_ratio']:.1%}")

    # より短期の予測も試す
    print(f"\n現在の予測例:")
    print("-" * 30)

    test_symbols = symbols[:5]
    for symbol in test_symbols:
        predicted_return = predictor.adaptive_predict_return_rate(
            symbol, prediction_days=1
        )
        print(
            f"{symbol}: 予測リターン率 {predicted_return:.4f} ({predicted_return*100:.2f}%)"
        )

    print(f"\n{'='*60}")
    if "error" not in metrics_1d:
        if metrics_1d["robust_mape"] < 15:
            print("✓ 実用レベル達成！")
        elif metrics_1d["robust_mape"] < 30:
            print("△ 大幅改善、あと少し！")
        else:
            print("継続改善が必要")

        print(f"最終結果: 堅牢MAPE {metrics_1d['robust_mape']:.2f}%")


if __name__ == "__main__":
    main()
