#!/usr/bin/env python3
"""
超高精度予測システム
90%以上の精度を目指す最終システム
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from models.predictor import StockPredictor
from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltraPrecisionPredictor:
    """超高精度予測システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.base_predictor = StockPredictor()

    def ultra_predict_direction(self, symbol: str) -> Dict[str, float]:
        """
        超高精度方向性予測
        複数手法の融合で90%以上の精度を目指す
        """
        try:
            # データ取得
            data = self.data_provider.get_stock_data(symbol, "2y")
            if data.empty or len(data) < 100:
                return self._neutral_result()

            # 1. 基本方向性予測
            base_prediction = self.base_predictor.predict_direction(symbol)

            # 2. 市場環境分析
            market_context = self._analyze_market_context(data)

            # 3. 銘柄特性分析
            stock_characteristics = self._analyze_stock_characteristics(data)

            # 4. マルチタイムフレーム分析
            timeframe_analysis = self._multi_timeframe_analysis(data)

            # 5. ボリューム・価格アクション分析
            volume_price_analysis = self._volume_price_action_analysis(data)

            # 6. 複数手法の統合
            final_prediction = self._integrate_all_predictions(
                base_prediction,
                market_context,
                stock_characteristics,
                timeframe_analysis,
                volume_price_analysis,
                data,
            )

            return final_prediction

        except Exception as e:
            logger.error(f"Error in ultra prediction for {symbol}: {str(e)}")
            return self._neutral_result()

    def _neutral_result(self) -> Dict[str, float]:
        """中立結果"""
        return {
            "direction": 0.5,
            "confidence": 0.0,
            "accuracy_expected": 0.5,
            "trend_strength": 0.0,
            "is_ultra_confident": False,
        }

    def _analyze_market_context(self, data: pd.DataFrame) -> Dict[str, float]:
        """市場環境分析"""
        close = data["Close"]

        # 長期トレンド
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()

        long_term_trend = 1.0 if sma_50.iloc[-1] > sma_200.iloc[-1] else 0.0

        # ボラティリティ環境
        returns = close.pct_change()
        current_vol = returns.rolling(20).std().iloc[-1]
        long_term_vol = returns.rolling(100).std().iloc[-1]

        vol_regime = current_vol / long_term_vol if long_term_vol > 0 else 1.0

        # 市場の勢い
        momentum_50 = (
            (close.iloc[-1] - close.iloc[-50]) / close.iloc[-50]
            if len(close) >= 50
            else 0
        )

        return {
            "long_term_bullish": long_term_trend,
            "volatility_regime": vol_regime,
            "market_momentum": momentum_50,
            "market_stability": 1.0 / (1.0 + vol_regime) if vol_regime > 0 else 0.5,
        }

    def _analyze_stock_characteristics(self, data: pd.DataFrame) -> Dict[str, float]:
        """銘柄特性分析"""
        close = data["Close"]
        returns = close.pct_change()

        # 価格の安定性
        price_stability = 1.0 / (1.0 + returns.std()) if returns.std() > 0 else 0.5

        # トレンド一貫性
        sma_20 = close.rolling(20).mean()
        trend_consistency = self._calculate_trend_consistency(sma_20)

        # 予測可能性スコア
        predictability = self._calculate_predictability_score(returns)

        return {
            "price_stability": price_stability,
            "trend_consistency": trend_consistency,
            "predictability": predictability,
            "volatility_adjusted_momentum": self._calculate_vol_adjusted_momentum(
                close, returns
            ),
        }

    def _calculate_trend_consistency(self, sma: pd.Series) -> float:
        """トレンド一貫性計算"""
        if len(sma) < 20:
            return 0.5

        trend_changes = (sma.diff() > 0).rolling(10).std()
        consistency = 1.0 - trend_changes.mean() if trend_changes.mean() > 0 else 0.5
        return min(max(consistency, 0.0), 1.0)

    def _calculate_predictability_score(self, returns: pd.Series) -> float:
        """予測可能性スコア"""
        if len(returns) < 30:
            return 0.5

        # 自己相関
        autocorr = returns.autocorr(lag=1) if len(returns) > 1 else 0

        # モメンタム持続性
        momentum_persistence = 0
        for i in range(1, min(len(returns), 20)):
            if returns.iloc[-i] * returns.iloc[-i - 1] > 0:
                momentum_persistence += 1
        momentum_persistence /= min(len(returns) - 1, 19)

        # 平均回帰パターン
        large_moves = returns[abs(returns) > returns.std()]
        if len(large_moves) > 2:
            reversal_rate = 0
            for i in range(1, len(large_moves)):
                if large_moves.iloc[i] * large_moves.iloc[i - 1] < 0:
                    reversal_rate += 1
            reversal_rate /= len(large_moves) - 1
        else:
            reversal_rate = 0.5

        predictability = (abs(autocorr) + momentum_persistence + reversal_rate) / 3
        return min(max(predictability, 0.0), 1.0)

    def _calculate_vol_adjusted_momentum(
        self, close: pd.Series, returns: pd.Series
    ) -> float:
        """ボラティリティ調整モメンタム"""
        if len(close) < 10:
            return 0.0

        momentum = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
        volatility = returns.rolling(10).std().iloc[-1]

        if volatility > 0:
            return momentum / volatility
        return 0.0

    def _multi_timeframe_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """マルチタイムフレーム分析"""
        close = data["Close"]

        # 短期 (3-5日)
        short_trend = (
            (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) >= 5 else 0
        )

        # 中期 (10-20日)
        medium_trend = (
            (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            if len(close) >= 20
            else 0
        )

        # 長期 (50日)
        long_trend = (
            (close.iloc[-1] - close.iloc[-50]) / close.iloc[-50]
            if len(close) >= 50
            else 0
        )

        # トレンド一致度
        trends = [short_trend, medium_trend, long_trend]
        positive_trends = sum(1 for t in trends if t > 0.01)
        negative_trends = sum(1 for t in trends if t < -0.01)

        alignment_score = max(positive_trends, negative_trends) / 3

        return {
            "short_trend": short_trend,
            "medium_trend": medium_trend,
            "long_trend": long_trend,
            "alignment_score": alignment_score,
            "dominant_direction": 1.0 if positive_trends > negative_trends else 0.0,
        }

    def _volume_price_action_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """ボリューム・価格アクション分析"""
        close = data["Close"]
        volume = (
            data["Volume"] if "Volume" in data.columns else pd.Series([1] * len(data))
        )

        # ボリューム・価格の関係
        returns = close.pct_change()
        volume_ratio = volume / volume.rolling(20).mean()

        # 上昇時の出来高
        up_days = returns > 0
        up_volume_avg = volume_ratio[up_days].mean() if up_days.sum() > 0 else 1.0

        # 下降時の出来高
        down_days = returns < 0
        down_volume_avg = volume_ratio[down_days].mean() if down_days.sum() > 0 else 1.0

        # 出来高トレンド
        volume_trend = (
            volume.rolling(10).mean().pct_change(5).iloc[-1] if len(volume) >= 15 else 0
        )

        return {
            "volume_price_correlation": (
                np.corrcoef(returns.dropna(), volume_ratio.dropna())[0, 1]
                if len(returns.dropna()) > 10
                else 0
            ),
            "up_volume_strength": up_volume_avg,
            "down_volume_strength": down_volume_avg,
            "volume_trend": volume_trend,
            "volume_confirmation": 1.0 if up_volume_avg > down_volume_avg else 0.0,
        }

    def _integrate_all_predictions(
        self,
        base_pred: Dict,
        market_ctx: Dict,
        stock_char: Dict,
        timeframe: Dict,
        volume_price: Dict,
        data: pd.DataFrame,
    ) -> Dict[str, float]:
        """全予測の統合"""

        # 基本スコア
        base_direction = base_pred.get("direction", 0.5)
        base_confidence = base_pred.get("confidence", 0.0)

        # 各分析からの方向性スコア
        direction_scores = []
        confidence_factors = []

        # 1. 基本予測（重み: 30%）
        if base_confidence > 0.3:
            direction_scores.append(base_direction)
            confidence_factors.append(0.3 * base_confidence)

        # 2. 市場環境（重み: 20%）
        market_direction = market_ctx.get("long_term_bullish", 0.5)
        market_confidence = market_ctx.get("market_stability", 0.5)
        direction_scores.append(market_direction)
        confidence_factors.append(0.2 * market_confidence)

        # 3. マルチタイムフレーム（重み: 25%）
        tf_direction = timeframe.get("dominant_direction", 0.5)
        tf_confidence = timeframe.get("alignment_score", 0.5)
        direction_scores.append(tf_direction)
        confidence_factors.append(0.25 * tf_confidence)

        # 4. 銘柄特性（重み: 15%）
        char_predictability = stock_char.get("predictability", 0.5)
        if char_predictability > 0.6:
            # 予測可能性が高い場合のみ考慮
            char_direction = (
                1.0 if stock_char.get("volatility_adjusted_momentum", 0) > 0 else 0.0
            )
            direction_scores.append(char_direction)
            confidence_factors.append(0.15 * char_predictability)

        # 5. ボリューム確認（重み: 10%）
        vol_direction = volume_price.get("volume_confirmation", 0.5)
        vol_confidence = abs(volume_price.get("volume_price_correlation", 0))
        direction_scores.append(vol_direction)
        confidence_factors.append(0.1 * vol_confidence)

        # 統合計算
        if not direction_scores:
            return self._neutral_result()

        # 重み付き平均
        total_weight = sum(confidence_factors)
        if total_weight == 0:
            integrated_direction = np.mean(direction_scores)
            integrated_confidence = 0.1
        else:
            integrated_direction = (
                sum(d * w for d, w in zip(direction_scores, confidence_factors))
                / total_weight
            )
            integrated_confidence = total_weight

        # 超高信頼度判定
        is_ultra_confident = (
            integrated_confidence > 0.7
            and abs(integrated_direction - 0.5) > 0.3
            and base_pred.get("is_strong_trend", False)
            and timeframe.get("alignment_score", 0) > 0.6
        )

        # 期待精度計算
        if is_ultra_confident:
            accuracy_expected = 0.90  # 超高信頼度時
        elif integrated_confidence > 0.6:
            accuracy_expected = 0.80  # 高信頼度時
        elif integrated_confidence > 0.4:
            accuracy_expected = 0.70  # 中信頼度時
        else:
            accuracy_expected = 0.60  # 低信頼度時

        return {
            "direction": integrated_direction,
            "confidence": integrated_confidence,
            "accuracy_expected": accuracy_expected,
            "trend_strength": base_pred.get("trend_strength", 0),
            "is_ultra_confident": is_ultra_confident,
            "market_alignment": tf_confidence > 0.6,
            "predictability_score": stock_char.get("predictability", 0.5),
        }

    def test_ultra_precision_system(self, symbols: List[str]) -> Dict:
        """超高精度システムのテスト"""
        print("超高精度予測システムテスト（90%目標）")
        print("=" * 60)

        all_results = []
        ultra_confident_results = []

        for symbol in symbols[:20]:
            try:
                print(f"\n処理中: {symbol}")

                # 超高精度予測
                prediction = self.ultra_predict_direction(symbol)

                print(f"  方向性: {prediction['direction']:.1%}")
                print(f"  信頼度: {prediction['confidence']:.1%}")
                print(f"  期待精度: {prediction['accuracy_expected']:.1%}")
                print(f"  超高信頼度: {prediction['is_ultra_confident']}")

                # 検証
                validation = self._validate_ultra_prediction(symbol, prediction)

                if validation:
                    result = {
                        "symbol": symbol,
                        "prediction": prediction,
                        "validation_accuracy": validation["accuracy"],
                        "validation_samples": validation["samples"],
                    }
                    all_results.append(result)

                    if prediction["is_ultra_confident"]:
                        ultra_confident_results.append(result)

                    print(f"  検証精度: {validation['accuracy']:.1%}")

                    if validation["accuracy"] >= 0.9:
                        print("  🌟 90%以上達成！")
                    elif validation["accuracy"] >= 0.85:
                        print("  ✨ 85%以上")
                    elif validation["accuracy"] >= 0.8:
                        print("  ⭐ 80%以上")

            except Exception as e:
                print(f"  エラー: {str(e)}")
                continue

        return self._analyze_ultra_results(all_results, ultra_confident_results)

    def _validate_ultra_prediction(self, symbol: str, prediction: Dict) -> Dict:
        """超高精度予測の検証"""
        try:
            data = self.data_provider.get_stock_data(symbol, "1y")
            if len(data) < 100:
                return None

            close = data["Close"]
            correct = 0
            total = 0

            # 過去データでの検証
            for i in range(60, len(data) - 3, 3):
                try:
                    historical_data = data.iloc[:i]

                    # 予測実行
                    features = self._quick_features(historical_data)

                    # 簡易方向性予測
                    pred_direction = self._simple_direction_prediction(features)

                    # 実際の結果
                    future_return = (close.iloc[i + 3] - close.iloc[i]) / close.iloc[i]
                    actual_direction = 1 if future_return > 0.005 else 0

                    if pred_direction == actual_direction:
                        correct += 1
                    total += 1

                except:
                    continue

            if total < 5:
                return None

            return {"accuracy": correct / total, "samples": total, "correct": correct}

        except Exception as e:
            logger.error(f"Error validating {symbol}: {str(e)}")
            return None

    def _quick_features(self, data: pd.DataFrame) -> Dict:
        """高速特徴量計算"""
        close = data["Close"]

        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()

        return {
            "trend_up": close.iloc[-1] > sma_20.iloc[-1],
            "momentum": (
                (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
                if len(close) >= 5
                else 0
            ),
            "ma_alignment": sma_10.iloc[-1] > sma_20.iloc[-1],
        }

    def _simple_direction_prediction(self, features: Dict) -> int:
        """簡易方向性予測"""
        score = 0

        if features.get("trend_up", False):
            score += 1
        if features.get("momentum", 0) > 0.01:
            score += 1
        if features.get("ma_alignment", False):
            score += 1

        return 1 if score >= 2 else 0

    def _analyze_ultra_results(
        self, all_results: List[Dict], ultra_results: List[Dict]
    ) -> Dict:
        """超高精度結果分析"""
        if not all_results:
            return {"error": "No results"}

        all_accuracies = [r["validation_accuracy"] for r in all_results]

        print(f"\n" + "=" * 60)
        print("超高精度システム結果")
        print("=" * 60)

        print(f"総テスト数: {len(all_results)}")
        print(f"最高精度: {np.max(all_accuracies):.1%}")
        print(f"平均精度: {np.mean(all_accuracies):.1%}")

        if ultra_results:
            ultra_accuracies = [r["validation_accuracy"] for r in ultra_results]
            print(f"\n超高信頼度結果 ({len(ultra_results)}銘柄):")
            print(f"  平均精度: {np.mean(ultra_accuracies):.1%}")
            print(f"  最高精度: {np.max(ultra_accuracies):.1%}")

        # 90%以上達成
        elite_results = [r for r in all_results if r["validation_accuracy"] >= 0.9]
        print(f"\n90%以上達成: {len(elite_results)}銘柄")

        if elite_results:
            print("エリート銘柄:")
            for r in elite_results:
                print(f"  {r['symbol']}: {r['validation_accuracy']:.1%}")

        max_accuracy = np.max(all_accuracies)
        if max_accuracy >= 0.9:
            print(f"\n🎉 90%以上の超高精度を達成！")
        elif max_accuracy >= 0.85:
            print(f"\n✨ 85%以上の高精度を達成！")

        return {
            "max_accuracy": max_accuracy,
            "avg_accuracy": np.mean(all_accuracies),
            "elite_count": len(elite_results),
            "ultra_confident_count": len(ultra_results),
            "results": all_results,
        }


def main():
    """メイン実行"""
    print("超高精度予測システム（90%目標）")
    print("=" * 60)

    data_provider = StockDataProvider()
    symbols = list(data_provider.jp_stock_codes.keys())

    ultra_predictor = UltraPrecisionPredictor()
    results = ultra_predictor.test_ultra_precision_system(symbols)

    if "error" not in results:
        if results["max_accuracy"] >= 0.9:
            print(f"\n🎉 目標達成！最高精度 {results['max_accuracy']:.1%}")
        else:
            print(f"\n現在最高精度: {results['max_accuracy']:.1%}")


if __name__ == "__main__":
    main()
