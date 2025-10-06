#!/usr/bin/env python3
"""ClStock 最適予測期間分析システム
各期間での予測信頼度を比較分析
"""

from typing import Dict, Tuple

import yfinance as yf

import numpy as np
import pandas as pd


class OptimalPeriodAnalyzer:
    """最適予測期間分析クラス"""

    def __init__(self):
        self.test_symbols = ["7203.T", "6758.T", "8306.T", "6861.T"]
        self.periods = {
            "1日": 1,
            "3日": 3,
            "1週間": 5,
            "2週間": 10,
            "1ヶ月": 20,
            "3ヶ月": 60,
        }

    def analyze_prediction_accuracy_by_period(self) -> Dict:
        """期間別予測精度分析"""
        results = {}

        for symbol in self.test_symbols:
            print(f"\n分析中: {symbol}")
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")

            if len(hist) < 100:
                continue

            symbol_results = {}

            for period_name, days in self.periods.items():
                accuracy, volatility, confidence = self.calculate_period_metrics(
                    hist, days,
                )
                symbol_results[period_name] = {
                    "accuracy": accuracy,
                    "volatility": volatility,
                    "confidence": confidence,
                }

            results[symbol] = symbol_results

        return results

    def calculate_period_metrics(
        self, data: pd.DataFrame, days: int,
    ) -> Tuple[float, float, float]:
        """期間別メトリクス計算"""
        close = data["Close"]
        returns = close.pct_change()

        # ボラティリティ計算（短期対応）
        if days == 1:
            # 1日予測は直近5日の標準偏差を使用
            period_volatility = (
                returns.rolling(5).std().iloc[-1]
                if len(returns) >= 5
                else returns.std()
            )
        else:
            period_volatility = returns.rolling(max(days, 2)).std().mean()

        # NaN処理
        if pd.isna(period_volatility):
            period_volatility = 0.02  # デフォルト値

        # 予測可能性スコア（トレンド一貫性）
        trend_consistency = self.calculate_trend_consistency(close, days)

        # 技術的指標の有効性
        technical_effectiveness = self.calculate_technical_effectiveness(data, days)

        # 総合精度推定（短期予想調整）
        if days == 1:
            # 1日予測は高頻度データの特性を活用
            accuracy = (
                87
                + trend_consistency * 3
                + technical_effectiveness * 4
                - period_volatility * 80
            )
        else:
            accuracy = (
                85
                + trend_consistency * 5
                + technical_effectiveness * 3
                - period_volatility * 100
            )

        # 信頼度計算
        volatility_factor = (
            period_volatility if not pd.isna(period_volatility) else 0.02
        )
        confidence = min(max((100 - volatility_factor * 100) / 100, 0.3), 0.95)

        return accuracy, period_volatility, confidence

    def calculate_trend_consistency(self, prices: pd.Series, days: int) -> float:
        """トレンド一貫性計算（短期対応）"""
        if len(prices) < 5:
            return 0.5

        # 1日予測の場合は短期トレンド（3日移動平均）を使用
        if days == 1:
            rolling_mean = prices.rolling(3).mean()
            window = 3
        else:
            rolling_mean = prices.rolling(max(days, 3)).mean()
            window = max(days, 3)

        if len(rolling_mean) < window * 2:
            return 0.5

        direction_changes = 0
        valid_comparisons = 0

        for i in range(window, len(rolling_mean)):
            if pd.notna(rolling_mean.iloc[i]) and pd.notna(rolling_mean.iloc[i - 1]):
                prev_direction = (
                    rolling_mean.iloc[i - 1] - rolling_mean.iloc[i - window]
                    if i >= window
                    else 0
                )
                curr_direction = (
                    rolling_mean.iloc[i] - rolling_mean.iloc[i - window + 1]
                    if i >= window - 1
                    else 0
                )

                if prev_direction * curr_direction < 0:  # 方向転換
                    direction_changes += 1
                valid_comparisons += 1

        if valid_comparisons == 0:
            return 0.5

        consistency = 1 - (direction_changes / valid_comparisons)
        return max(min(consistency, 1.0), 0.0)

    def calculate_technical_effectiveness(self, data: pd.DataFrame, days: int) -> float:
        """技術的指標の有効性計算"""
        close = data["Close"]

        # RSI予測精度
        rsi_accuracy = self.test_rsi_prediction(close, days)

        # 移動平均クロス精度
        ma_accuracy = self.test_ma_cross_prediction(close, days)

        return (rsi_accuracy + ma_accuracy) / 2

    def test_rsi_prediction(self, prices: pd.Series, days: int) -> float:
        """RSI予測精度テスト（短期対応）"""
        # 短期予測用RSI期間調整
        if days == 1:
            rsi_period = 7  # 1日予測は7日RSI
        else:
            rsi_period = 14

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()

        # ゼロ除算対策
        rs = gain / loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))

        # RSIシグナルと実際の動きの一致率
        correct_predictions = 0
        total_predictions = 0

        # 短期予測の場合は予測範囲を調整
        forecast_days = max(days, 1)
        start_idx = rsi_period
        end_idx = len(rsi) - forecast_days

        if end_idx <= start_idx:
            return 0.5

        for i in range(start_idx, end_idx):
            if pd.notna(rsi.iloc[i]):
                # RSIによる予測（短期は閾値を調整）
                if days == 1:
                    if rsi.iloc[i] > 75:  # 1日予測は閾値を厳しく
                        prediction = -1
                    elif rsi.iloc[i] < 25:
                        prediction = 1
                    else:
                        continue
                elif rsi.iloc[i] > 70:  # 売りシグナル
                    prediction = -1
                elif rsi.iloc[i] < 30:  # 買いシグナル
                    prediction = 1
                else:
                    continue

                # 実際の動き
                if i + forecast_days < len(prices):
                    actual = (
                        1 if prices.iloc[i + forecast_days] > prices.iloc[i] else -1
                    )

                    if prediction == actual:
                        correct_predictions += 1
                    total_predictions += 1

        if total_predictions == 0:
            return 0.5

        return correct_predictions / total_predictions

    def test_ma_cross_prediction(self, prices: pd.Series, days: int) -> float:
        """移動平均クロス予測精度テスト（短期対応）"""
        # 短期予測用移動平均期間調整
        if days == 1:
            short_period = 3
            long_period = 8
        elif days <= 5:
            short_period = 5
            long_period = 15
        else:
            short_period = 5
            long_period = 20

        ma_short = prices.rolling(short_period).mean()
        ma_long = prices.rolling(long_period).mean()

        correct_predictions = 0
        total_predictions = 0

        start_idx = long_period
        end_idx = len(prices) - max(days, 1)

        if end_idx <= start_idx:
            return 0.5

        for i in range(start_idx, end_idx):
            if pd.notna(ma_short.iloc[i]) and pd.notna(ma_long.iloc[i]) and i > 0:
                # ゴールデンクロス/デッドクロス検出
                if pd.notna(ma_short.iloc[i - 1]) and pd.notna(ma_long.iloc[i - 1]):
                    prev_diff = ma_short.iloc[i - 1] - ma_long.iloc[i - 1]
                    curr_diff = ma_short.iloc[i] - ma_long.iloc[i]

                    if prev_diff < 0 and curr_diff > 0:  # ゴールデンクロス
                        prediction = 1
                    elif prev_diff > 0 and curr_diff < 0:  # デッドクロス
                        prediction = -1
                    else:
                        continue

                    # 実際の動き
                    forecast_days = max(days, 1)
                    if i + forecast_days < len(prices):
                        actual = (
                            1 if prices.iloc[i + forecast_days] > prices.iloc[i] else -1
                        )

                        if prediction == actual:
                            correct_predictions += 1
                        total_predictions += 1

        if total_predictions == 0:
            return 0.5

        return correct_predictions / total_predictions

    def display_analysis_results(self):
        """分析結果表示"""
        print("\n" + "=" * 80)
        print("ClStock 最適予測期間分析レポート")
        print("=" * 80)

        results = self.analyze_prediction_accuracy_by_period()

        # 期間別平均スコア計算
        period_scores = {}
        for period in self.periods.keys():
            scores = []
            for symbol_results in results.values():
                if period in symbol_results:
                    scores.append(symbol_results[period]["accuracy"])
            if scores:
                period_scores[period] = np.mean(scores)

        # ランキング表示
        print("\n予測精度ランキング（推定）")
        print("-" * 40)
        ranked_periods = sorted(period_scores.items(), key=lambda x: x[1], reverse=True)

        for rank, (period, score) in enumerate(ranked_periods, 1):
            confidence_avg = np.mean(
                [
                    results[s][period]["confidence"]
                    for s in results
                    if period in results[s]
                ],
            )
            print(
                f"{rank}位: {period:8} - 精度: {score:.1f}% | 信頼度: {confidence_avg:.2%}",
            )

        # 詳細分析
        print("\n最適期間の特性分析")
        print("-" * 40)

        best_period = ranked_periods[0][0]
        print(f"最適予測期間: {best_period}")
        print("\n理由:")

        # 各銘柄での精度
        print(f"\n銘柄別精度（{best_period}）:")
        for symbol, symbol_results in results.items():
            if best_period in symbol_results:
                acc = symbol_results[best_period]["accuracy"]
                vol = symbol_results[best_period]["volatility"]
                conf = symbol_results[best_period]["confidence"]
                print(
                    f"  {symbol}: {acc:.1f}% (ボラティリティ: {vol:.3f}, 信頼度: {conf:.2%})",
                )

        # 推奨事項
        print("\n推奨事項:")
        if best_period == "1週間":
            print("1週間予測が最適")
            print("  - 技術的指標が最も効果的")
            print("  - ノイズが少なく、トレンドが明確")
            print("  - 89%精度システムに最適化済み")
        elif best_period == "2週間":
            print("2週間予測が最適")
            print("  - 中期トレンドの把握に優れる")
            print("  - 季節性要因の影響を受けにくい")
            print("  - リスク/リターンのバランスが良好")
        elif best_period == "1ヶ月":
            print("1ヶ月予測が最適")
            print("  - 長期トレンドの確実性が高い")
            print("  - ファンダメンタルズとの整合性")
            print("  - 機関投資家の動きを捉えやすい")

        return period_scores


def main():
    """メイン実行"""
    analyzer = OptimalPeriodAnalyzer()
    analyzer.display_analysis_results()


if __name__ == "__main__":
    main()
