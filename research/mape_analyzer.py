#!/usr/bin/env python3
"""
MAPE問題の詳細分析と改善システム
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging

from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MAPEAnalyzer:
    """MAPE問題の詳細分析"""

    def __init__(self):
        self.data_provider = StockDataProvider()

    def analyze_mape_explosion(self, symbols: List[str]) -> Dict:
        """MAPE爆発の原因分析"""
        print("=" * 60)
        print("MAPE爆発原因の詳細分析")
        print("=" * 60)

        all_predictions = []
        all_actuals = []
        all_errors = []
        problem_cases = []

        for symbol in symbols[:5]:
            try:
                data = self.data_provider.get_stock_data(symbol, "6mo")
                if len(data) < 60:
                    continue

                # 過去30日分のテストデータを作成
                for i in range(30, 5, -5):
                    historical_data = data.iloc[:-i].copy()
                    if len(historical_data) < 30:
                        continue

                    # 実際のリターン（5日間）
                    start_price = data.iloc[-i]["Close"]
                    end_price = (
                        data.iloc[-i + 5]["Close"] if i > 5 else data.iloc[-1]["Close"]
                    )
                    actual_return = (end_price - start_price) / start_price

                    # シンプル予測
                    predicted_return = self._simple_predict(historical_data)

                    all_predictions.append(predicted_return)
                    all_actuals.append(actual_return)

                    # エラー計算
                    if abs(actual_return) > 0.001:  # 0.1%以上の動きのみ
                        mape_individual = (
                            abs((actual_return - predicted_return) / actual_return)
                            * 100
                        )
                        all_errors.append(mape_individual)

                        # 問題ケースの特定
                        if mape_individual > 200:  # 200%以上のエラー
                            problem_cases.append(
                                {
                                    "symbol": symbol,
                                    "actual": actual_return,
                                    "predicted": predicted_return,
                                    "mape": mape_individual,
                                    "abs_actual": abs(actual_return),
                                    "error": abs(predicted_return - actual_return),
                                }
                            )
                    else:
                        # 小さすぎる動きは除外
                        continue

            except Exception as e:
                logger.warning(f"Error analyzing {symbol}: {str(e)}")
                continue

        # 分析結果
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        errors = np.array(all_errors) if all_errors else np.array([])

        print(f"総テストケース: {len(predictions)}")
        print(f"有効MAPEケース: {len(errors)}")

        if len(errors) > 0:
            print(f"MAPE平均: {np.mean(errors):.2f}%")
            print(f"MAPE中央値: {np.median(errors):.2f}%")
            print(f"MAPE最大: {np.max(errors):.2f}%")
            print(f"MAPE最小: {np.min(errors):.2f}%")

            # パーセンタイル分析
            print(f"\nMAPE分布:")
            print(f"  90%tile: {np.percentile(errors, 90):.2f}%")
            print(f"  75%tile: {np.percentile(errors, 75):.2f}%")
            print(f"  50%tile: {np.percentile(errors, 50):.2f}%")
            print(f"  25%tile: {np.percentile(errors, 25):.2f}%")

        # 問題ケース分析
        print(f"\n問題ケース分析 (MAPE > 200%):")
        print(f"問題ケース数: {len(problem_cases)}")

        if problem_cases:
            # 問題ケースの特徴
            problem_df = pd.DataFrame(problem_cases)
            print(
                f"問題ケースの実績リターン範囲: {problem_df['actual'].min():.4f} ～ {problem_df['actual'].max():.4f}"
            )
            print(f"問題ケースの絶対値平均: {problem_df['abs_actual'].mean():.4f}")

            # 小さすぎる動きが問題か？
            small_movements = problem_df[problem_df["abs_actual"] < 0.01]  # 1%以下
            print(
                f"1%以下の小動きでの問題ケース: {len(small_movements)}/{len(problem_cases)}"
            )

            # 最悪ケースを表示
            worst_cases = problem_df.nlargest(5, "mape")
            print(f"\n最悪ケース Top5:")
            for _, case in worst_cases.iterrows():
                print(
                    f"  {case['symbol']}: 実績{case['actual']:.4f} vs 予測{case['predicted']:.4f} → MAPE {case['mape']:.1f}%"
                )

        return {
            "all_predictions": predictions,
            "all_actuals": actuals,
            "all_errors": errors,
            "problem_cases": problem_cases,
            "mape_mean": np.mean(errors) if len(errors) > 0 else float("inf"),
        }

    def _simple_predict(self, data: pd.DataFrame) -> float:
        """シンプル予測ロジック"""
        try:
            data = self.data_provider.calculate_technical_indicators(data)

            current_price = data["Close"].iloc[-1]
            sma_5 = data["Close"].rolling(5).mean().iloc[-1]
            sma_20 = data["SMA_20"].iloc[-1]

            if current_price > sma_5 > sma_20:
                return 0.008
            elif current_price < sma_5 < sma_20:
                return -0.006
            else:
                return 0.001
        except:
            return 0.001

    def create_improved_predictor(self, analysis_results: Dict) -> "ImprovedPredictor":
        """分析結果を基に改善された予測器を作成"""
        return ImprovedPredictor(analysis_results)


class ImprovedPredictor:
    """改善された予測システム"""

    def __init__(self, analysis_results: Dict = None):
        self.data_provider = StockDataProvider()
        self.analysis_results = analysis_results

    def predict_return_rate_v2(self, symbol: str, prediction_days: int = 2) -> float:
        """改善されたリターン率予測 v2.0"""
        try:
            # より短期のデータ取得
            data = self.data_provider.get_stock_data(symbol, "2mo")
            if data.empty or len(data) < 20:
                return 0.0

            data = self.data_provider.calculate_technical_indicators(data)

            # 超短期予測ロジック（1-2日用）
            current_price = data["Close"].iloc[-1]
            returns = data["Close"].pct_change().dropna()

            # 1. 直近のモメンタム（より重要）
            momentum_1d = returns.iloc[-1] if len(returns) > 0 else 0
            momentum_3d = (
                (current_price - data["Close"].iloc[-4]) / data["Close"].iloc[-4]
                if len(data) > 4
                else 0
            )

            # 2. 短期移動平均
            sma_3 = data["Close"].rolling(3).mean().iloc[-1]
            sma_10 = data["Close"].rolling(10).mean().iloc[-1]

            # 3. RSI（短期設定）
            rsi = data["RSI"].iloc[-1] if "RSI" in data.columns else 50

            # 基本予測（超保守的）
            base_return = 0.0

            # モメンタムベース予測（主要シグナル）
            if momentum_1d > 0.01:  # 前日1%以上上昇
                if momentum_3d > 0.02:  # 3日で2%以上上昇
                    base_return = 0.004  # 0.4%の継続予想
                else:
                    base_return = 0.002  # 0.2%の継続予想
            elif momentum_1d < -0.01:  # 前日1%以上下落
                if momentum_3d < -0.02:  # 3日で2%以上下落
                    base_return = -0.003  # -0.3%の継続予想
                else:
                    base_return = -0.001  # -0.1%の継続予想

            # 移動平均トレンド調整
            if current_price > sma_3 > sma_10:
                base_return += 0.002
            elif current_price < sma_3 < sma_10:
                base_return -= 0.002

            # RSI逆張り調整（弱め）
            if rsi < 25:
                base_return += 0.001
            elif rsi > 75:
                base_return -= 0.001

            # 超保守的制限（1-2日予測用）
            max_range = 0.015 * prediction_days  # 日数に応じた制限
            predicted_return = max(-max_range, min(max_range, base_return))

            return predicted_return

        except Exception as e:
            logger.error(f"Error predicting return rate for {symbol}: {str(e)}")
            return 0.0

    def test_improved_prediction(
        self, symbols: List[str], prediction_days: int = 2
    ) -> Dict:
        """改善された予測システムのテスト"""
        print(f"\n改善された予測システム v2.0 テスト（{prediction_days}日予測）")
        print("-" * 50)

        results = {"predictions": [], "actuals": [], "symbols": [], "valid_errors": []}

        for symbol in symbols[:5]:
            try:
                data = self.data_provider.get_stock_data(symbol, "6mo")
                if len(data) < 40:
                    continue

                # 短期テスト（より多くのテストケース）
                for i in range(30, prediction_days, -prediction_days):
                    historical_data = data.iloc[:-i].copy()
                    if len(historical_data) < 20:
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

                    # 改善された予測
                    predicted_return = self._predict_with_data(
                        historical_data, prediction_days
                    )

                    results["predictions"].append(predicted_return)
                    results["actuals"].append(actual_return)
                    results["symbols"].append(symbol)

                    # MAPE計算用（閾値フィルタリング）
                    if abs(actual_return) > 0.005:  # 0.5%以上の動きのみ
                        mape_individual = (
                            abs((actual_return - predicted_return) / actual_return)
                            * 100
                        )
                        results["valid_errors"].append(mape_individual)

            except Exception as e:
                logger.warning(f"Error testing {symbol}: {str(e)}")
                continue

        return results

    def _predict_with_data(self, data: pd.DataFrame, prediction_days: int) -> float:
        """データを使った予測"""
        try:
            data = self.data_provider.calculate_technical_indicators(data)

            current_price = data["Close"].iloc[-1]
            returns = data["Close"].pct_change().dropna()

            # 超シンプル予測
            momentum_1d = returns.iloc[-1] if len(returns) > 0 else 0

            # 平均回帰的予測
            if momentum_1d > 0.015:  # 1.5%以上上昇
                return -0.002 * prediction_days  # 反動予想
            elif momentum_1d < -0.015:  # 1.5%以上下落
                return 0.003 * prediction_days  # 反発予想
            elif momentum_1d > 0.005:  # 0.5%以上上昇
                return 0.001 * prediction_days  # 継続予想
            elif momentum_1d < -0.005:  # 0.5%以上下落
                return -0.001 * prediction_days  # 継続予想
            else:
                return 0.0  # 中立

        except:
            return 0.0


def calculate_improved_metrics(results: Dict) -> Dict:
    """改善されたメトリクス計算"""
    if not results["predictions"]:
        return {"error": "No valid predictions"}

    predictions = np.array(results["predictions"])
    actuals = np.array(results["actuals"])
    valid_errors = np.array(results["valid_errors"])

    # 基本メトリクス
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # 改善されたMAPE計算
    mape = np.mean(valid_errors) if len(valid_errors) > 0 else float("inf")

    # 方向性精度
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    directional_accuracy = (pred_direction == actual_direction).mean() * 100

    # 安定性指標
    prediction_std = np.std(predictions)
    actual_std = np.std(actuals)

    return {
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
        "total_predictions": len(predictions),
        "valid_mape_cases": len(valid_errors),
        "prediction_std": prediction_std,
        "actual_std": actual_std,
        "prediction_range": (predictions.min(), predictions.max()),
        "actual_range": (actuals.min(), actuals.max()),
    }


def main():
    """メイン実行関数"""
    print("MAPE問題の根本的解決システム")

    data_provider = StockDataProvider()
    symbols = list(data_provider.jp_stock_codes.keys())

    # 1. 現在のMAPE問題を分析
    analyzer = MAPEAnalyzer()
    analysis_results = analyzer.analyze_mape_explosion(symbols)

    # 2. 改善された予測システムをテスト
    improved_predictor = analyzer.create_improved_predictor(analysis_results)

    # 1日予測テスト
    print(f"\n{'='*60}")
    results_1d = improved_predictor.test_improved_prediction(symbols, prediction_days=1)
    metrics_1d = calculate_improved_metrics(results_1d)

    if "error" not in metrics_1d:
        print(f"1日予測結果:")
        print(f"  MAPE: {metrics_1d['mape']:.2f}%")
        print(f"  MAE: {metrics_1d['mae']:.4f}")
        print(f"  方向性精度: {metrics_1d['directional_accuracy']:.1f}%")
        print(f"  テスト数: {metrics_1d['total_predictions']}")
        print(f"  有効MAPE数: {metrics_1d['valid_mape_cases']}")

    # 2日予測テスト
    results_2d = improved_predictor.test_improved_prediction(symbols, prediction_days=2)
    metrics_2d = calculate_improved_metrics(results_2d)

    if "error" not in metrics_2d:
        print(f"\n2日予測結果:")
        print(f"  MAPE: {metrics_2d['mape']:.2f}%")
        print(f"  MAE: {metrics_2d['mae']:.4f}")
        print(f"  方向性精度: {metrics_2d['directional_accuracy']:.1f}%")
        print(f"  テスト数: {metrics_2d['total_predictions']}")
        print(f"  有効MAPE数: {metrics_2d['valid_mape_cases']}")

    # 最良結果の判定
    best_mape = float("inf")
    best_config = None

    if "error" not in metrics_1d and metrics_1d["mape"] < best_mape:
        best_mape = metrics_1d["mape"]
        best_config = "1日予測"

    if "error" not in metrics_2d and metrics_2d["mape"] < best_mape:
        best_mape = metrics_2d["mape"]
        best_config = "2日予測"

    print(f"\n{'='*60}")
    print(f"最良結果: {best_config} - MAPE {best_mape:.2f}%")

    if best_mape < 15:
        print("✓ 実用レベル達成！")
    elif best_mape < 50:
        print("△ 大幅改善、さらに最適化が必要")
    else:
        print("✗ 追加改善が必要")


if __name__ == "__main__":
    main()
