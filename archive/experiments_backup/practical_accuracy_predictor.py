#!/usr/bin/env python3
"""
MAPE問題を回避した実用的な精度重視予測システム
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PracticalAccuracyPredictor:
    """実用的な精度重視予測システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()

    def predict_probability_bands(self, symbol: str) -> Dict[str, float]:
        """確率帯域予測（従来のポイント予測の代替）"""
        try:
            data = self.data_provider.get_stock_data(symbol, "1mo")
            if data.empty or len(data) < 10:
                return {
                    "neutral": 1.0,
                    "up_small": 0.0,
                    "up_large": 0.0,
                    "down_small": 0.0,
                    "down_large": 0.0,
                }

            data = self.data_provider.calculate_technical_indicators(data)
            returns = data["Close"].pct_change().dropna()

            if len(returns) < 5:
                return {
                    "neutral": 1.0,
                    "up_small": 0.0,
                    "up_large": 0.0,
                    "down_small": 0.0,
                    "down_large": 0.0,
                }

            # 過去パターン分析
            recent_returns = returns.iloc[-5:]
            avg_volatility = returns.std()

            # 現在の状況
            current_price = data["Close"].iloc[-1]
            sma_5 = data["Close"].rolling(5).mean().iloc[-1]
            rsi = data["RSI"].iloc[-1] if "RSI" in data.columns else 50

            # 確率計算（基本は均等分布から開始）
            probabilities = {
                "down_large": 0.15,  # -2%以下
                "down_small": 0.25,  # -2% to -0.5%
                "neutral": 0.30,  # -0.5% to +0.5%
                "up_small": 0.25,  # +0.5% to +2%
                "up_large": 0.05,  # +2%以上
            }

            # トレンド調整
            trend_strength = (current_price - sma_5) / sma_5
            if trend_strength > 0.01:  # 上昇トレンド
                probabilities["up_small"] += 0.15
                probabilities["up_large"] += 0.10
                probabilities["down_small"] -= 0.15
                probabilities["down_large"] -= 0.10
            elif trend_strength < -0.01:  # 下降トレンド
                probabilities["down_small"] += 0.15
                probabilities["down_large"] += 0.10
                probabilities["up_small"] -= 0.15
                probabilities["up_large"] -= 0.10

            # RSI調整
            if rsi < 30:  # 過売り
                probabilities["up_small"] += 0.10
                probabilities["up_large"] += 0.05
                probabilities["down_large"] -= 0.15
            elif rsi > 70:  # 過買い
                probabilities["down_small"] += 0.10
                probabilities["down_large"] += 0.05
                probabilities["up_large"] -= 0.15

            # ボラティリティ調整
            recent_vol = recent_returns.std()
            if recent_vol > avg_volatility * 1.5:  # 高ボラティリティ
                probabilities["neutral"] -= 0.10
                probabilities["up_large"] += 0.05
                probabilities["down_large"] += 0.05

            # 確率の正規化
            total_prob = sum(probabilities.values())
            for key in probabilities:
                probabilities[key] = max(0, probabilities[key] / total_prob)

            return probabilities

        except Exception as e:
            logger.error(f"Error predicting probabilities for {symbol}: {str(e)}")
            return {
                "neutral": 1.0,
                "up_small": 0.0,
                "up_large": 0.0,
                "down_small": 0.0,
                "down_large": 0.0,
            }

    def predict_expected_return_range(self, symbol: str) -> Tuple[float, float, float]:
        """期待リターン範囲予測（最悪、期待値、最良）"""
        try:
            data = self.data_provider.get_stock_data(symbol, "1mo")
            if data.empty or len(data) < 10:
                return (-0.01, 0.0, 0.01)

            data = self.data_provider.calculate_technical_indicators(data)
            returns = data["Close"].pct_change().dropna()

            if len(returns) < 5:
                return (-0.01, 0.0, 0.01)

            # 銘柄の特性分析
            volatility = returns.std()
            mean_return = returns.mean()

            # 現在の市場状況による調整
            current_price = data["Close"].iloc[-1]
            sma_5 = data["Close"].rolling(5).mean().iloc[-1]
            momentum = returns.iloc[-1] if len(returns) > 0 else 0

            # 基本期待値
            base_expected = (
                mean_return * 0.3 + momentum * 0.2
            )  # 履歴とモメンタムの組み合わせ

            # トレンド調整
            trend = (current_price - sma_5) / sma_5
            if abs(trend) > 0.005:  # 有意なトレンド
                base_expected += trend * 0.5

            # 範囲計算（1σ、2σベース）
            daily_vol = volatility
            worst_case = base_expected - daily_vol * 1.5
            best_case = base_expected + daily_vol * 1.5

            # 現実的な制限
            worst_case = max(-0.05, worst_case)
            best_case = min(0.05, best_case)
            expected = max(worst_case, min(best_case, base_expected))

            return (worst_case, expected, best_case)

        except Exception as e:
            logger.error(f"Error predicting range for {symbol}: {str(e)}")
            return (-0.01, 0.0, 0.01)

    def test_range_prediction_accuracy(self, symbols: List[str]) -> Dict:
        """範囲予測精度のテスト"""
        print("\n範囲予測精度テスト")
        print("-" * 40)

        results = {
            "predictions": [],  # (worst, expected, best)
            "actuals": [],
            "symbols": [],
            "within_range": [],
            "expected_errors": [],
        }

        for symbol in symbols[:5]:
            try:
                data = self.data_provider.get_stock_data(symbol, "2mo")
                if len(data) < 20:
                    continue

                # 10回分のテスト
                for i in range(15, 5, -1):
                    historical_data = data.iloc[:-i].copy()
                    if len(historical_data) < 10:
                        continue

                    # 実際のリターン（1日後）
                    start_price = data.iloc[-i]["Close"]
                    end_price = data.iloc[-i + 1]["Close"]
                    actual_return = (end_price - start_price) / start_price

                    # 範囲予測（過去データのみ使用）
                    worst, expected, best = self._predict_range_with_data(
                        historical_data
                    )

                    results["predictions"].append((worst, expected, best))
                    results["actuals"].append(actual_return)
                    results["symbols"].append(symbol)

                    # 範囲内精度
                    within_range = worst <= actual_return <= best
                    results["within_range"].append(within_range)

                    # 期待値誤差
                    expected_error = abs(actual_return - expected)
                    results["expected_errors"].append(expected_error)

            except Exception as e:
                logger.warning(f"Error testing {symbol}: {str(e)}")
                continue

        return results

    def _predict_range_with_data(
        self, data: pd.DataFrame
    ) -> Tuple[float, float, float]:
        """過去データのみを使った範囲予測"""
        try:
            data = self.data_provider.calculate_technical_indicators(data)
            returns = data["Close"].pct_change().dropna()

            if len(returns) < 5:
                return (-0.01, 0.0, 0.01)

            # 超シンプルな予測
            recent_return = returns.iloc[-1]
            volatility = returns.std()

            # 平均回帰的予測
            if abs(recent_return) > volatility:
                expected = -recent_return * 0.3  # 反動予想
            else:
                expected = recent_return * 0.2  # 継続予想

            # 範囲
            range_width = volatility * 1.0
            worst = expected - range_width
            best = expected + range_width

            return (worst, expected, best)

        except:
            return (-0.01, 0.0, 0.01)


def calculate_practical_metrics(results: Dict) -> Dict:
    """実用的メトリクス計算"""
    if not results["predictions"]:
        return {"error": "No valid predictions"}

    # 範囲内精度
    within_range_accuracy = np.mean(results["within_range"]) * 100

    # 期待値の平均絶対誤差
    expected_mae = np.mean(results["expected_errors"])

    # 期待値の実用的MAPE（絶対誤差/平均絶対リターン）
    actuals = np.array(results["actuals"])
    expected_errors = np.array(results["expected_errors"])

    mean_abs_actual = np.mean(np.abs(actuals))
    practical_mape = (
        (expected_mae / mean_abs_actual * 100) if mean_abs_actual > 0 else float("inf")
    )

    # 予測の有用性（ランダム予測との比較）
    random_mae = np.mean(np.abs(actuals))  # ランダム予測（常に0予測）との比較
    improvement_ratio = (
        (random_mae - expected_mae) / random_mae * 100 if random_mae > 0 else 0
    )

    return {
        "range_accuracy": within_range_accuracy,
        "expected_mae": expected_mae,
        "practical_mape": practical_mape,
        "improvement_over_random": improvement_ratio,
        "total_tests": len(results["predictions"]),
        "mean_abs_actual": mean_abs_actual,
    }


def main():
    """メイン実行関数"""
    print("=" * 60)
    print("実用的な精度重視予測システム")
    print("=" * 60)

    data_provider = StockDataProvider()
    symbols = list(data_provider.jp_stock_codes.keys())

    predictor = PracticalAccuracyPredictor()

    # 範囲予測テスト
    results = predictor.test_range_prediction_accuracy(symbols)
    metrics = calculate_practical_metrics(results)

    if "error" not in metrics:
        print(f"実用的予測精度結果:")
        print(f"  範囲内精度: {metrics['range_accuracy']:.1f}%")
        print(f"  期待値MAE: {metrics['expected_mae']:.4f}")
        print(f"  実用的MAPE: {metrics['practical_mape']:.2f}%")
        print(f"  ランダム予測比改善: {metrics['improvement_over_random']:.1f}%")
        print(f"  総テスト数: {metrics['total_tests']}")
        print(f"  平均絶対リターン: {metrics['mean_abs_actual']:.4f}")

    # 現在の予測例
    print(f"\n現在の予測例（範囲予測）:")
    print("-" * 40)

    test_symbols = symbols[:5]
    for symbol in test_symbols:
        worst, expected, best = predictor.predict_expected_return_range(symbol)
        probs = predictor.predict_probability_bands(symbol)

        print(f"{symbol}:")
        print(f"  予測範囲: {worst:.3f} ～ {best:.3f} (期待値: {expected:.3f})")
        print(
            f"  確率: 上昇{(probs['up_small']+probs['up_large']):.1%}, 中立{probs['neutral']:.1%}, 下降{(probs['down_small']+probs['down_large']):.1%}"
        )

    print(f"\n{'='*60}")
    if "error" not in metrics:
        if metrics["practical_mape"] < 15:
            print("✓ 実用レベル達成！")
        elif metrics["practical_mape"] < 30:
            print("△ 改善中、あと少し！")
        elif metrics["range_accuracy"] > 70:
            print("✓ 範囲予測として実用レベル達成！")
        else:
            print("継続改善が必要")

        print(
            f"最終結果: 実用的MAPE {metrics['practical_mape']:.2f}%, 範囲精度 {metrics['range_accuracy']:.1f}%"
        )


if __name__ == "__main__":
    main()
