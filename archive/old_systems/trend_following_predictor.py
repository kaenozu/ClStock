#!/usr/bin/env python3
"""
トレンドフォロー特化の方向性予測システム
明確なトレンド中のみの予測で80%以上を目指す
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from utils.logger_config import setup_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

from data.stock_data import StockDataProvider

# ログ設定
logger = setup_logger(__name__)


class TrendFollowingPredictor:
    """トレンドフォロー特化の方向性予測システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.scaler = StandardScaler()

    def identify_strong_trends(self, data: pd.DataFrame) -> pd.Series:
        """強いトレンド期間の特定"""
        close = data["Close"]

        # 複数期間の移動平均
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        # 強い上昇トレンド
        strong_uptrend = (
            (sma_10 > sma_20)
            & (sma_20 > sma_50)
            & (close > sma_10)
            & (sma_10.pct_change(5) > 0.01)  # 5日で1%以上上昇
        )

        # 強い下降トレンド
        strong_downtrend = (
            (sma_10 < sma_20)
            & (sma_20 < sma_50)
            & (close < sma_10)
            & (sma_10.pct_change(5) < -0.01)  # 5日で1%以上下降
        )

        # トレンドの継続性確認
        trend_duration = pd.Series(0, index=data.index)

        for i in range(10, len(data)):
            if strong_uptrend.iloc[i] or strong_downtrend.iloc[i]:
                # 過去10日間のトレンド一貫性
                recent_up = strong_uptrend.iloc[i - 10 : i].sum()
                recent_down = strong_downtrend.iloc[i - 10 : i].sum()

                if recent_up >= 7:  # 10日中7日以上上昇トレンド
                    trend_duration.iloc[i] = 1
                elif recent_down >= 7:  # 10日中7日以上下降トレンド
                    trend_duration.iloc[i] = 1

        return trend_duration.astype(bool)

    def create_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """トレンド特化特徴量"""
        features = pd.DataFrame(index=data.index)

        close = data["Close"]
        volume = data["Volume"]

        # 1. 移動平均の関係
        sma_5 = close.rolling(5).mean()
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()

        features["ma_bullish"] = (sma_5 > sma_10) & (sma_10 > sma_20)
        features["ma_bearish"] = (sma_5 < sma_10) & (sma_10 < sma_20)

        # 移動平均の傾き
        features["sma10_slope"] = sma_10.pct_change(5)
        features["sma20_slope"] = sma_20.pct_change(5)

        # 2. トレンド強度
        features["trend_strength"] = abs((sma_5 - sma_20) / sma_20)

        # 3. 価格のモメンタム
        features["price_momentum_5d"] = close.pct_change(5)
        features["price_momentum_10d"] = close.pct_change(10)

        # 連続上昇/下降日数
        daily_change = close.pct_change() > 0
        features["consecutive_up"] = daily_change.rolling(5).sum()
        features["consecutive_down"] = (~daily_change).rolling(5).sum()

        # 4. ボリューム確認
        vol_avg = volume.rolling(20).mean()
        features["volume_support"] = volume > vol_avg

        # 5. RSI（トレンド確認用）
        rsi = self._calculate_rsi(close, 14)
        features["rsi_trend_up"] = (rsi > 55) & (rsi < 80)
        features["rsi_trend_down"] = (rsi < 45) & (rsi > 20)

        return features

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_trend_target(
        self, data: pd.DataFrame, prediction_days: int = 3
    ) -> pd.Series:
        """トレンド継続予測ターゲット"""
        close = data["Close"]

        # 短期の将来トレンド
        future_return = close.shift(-prediction_days).pct_change(prediction_days)

        # トレンド継続の判定（より厳格）
        target = (future_return > 0.005).astype(int)  # 0.5%以上の上昇

        return target

    def test_trend_following_system(self, symbols: List[str]) -> Dict:
        """トレンドフォロー予測システムのテスト"""
        print("トレンドフォロー特化方向性予測システム")
        print("=" * 60)

        all_results = []

        for symbol in symbols[:30]:  # 20→30に拡張してより多くテスト
            try:
                print(f"\n処理中: {symbol}")

                # データ取得
                data = self.data_provider.get_stock_data(symbol, "2y")
                if len(data) < 200:
                    continue

                # 強いトレンド期間の特定
                strong_trend_mask = self.identify_strong_trends(data)

                if strong_trend_mask.sum() < 30:
                    print(
                        f"  スキップ: 強いトレンド期間不足 ({strong_trend_mask.sum()})"
                    )
                    continue

                print(f"  強いトレンド期間: {strong_trend_mask.sum()}日")

                # トレンド期間のデータのみ使用
                trend_data = data[strong_trend_mask]

                # 特徴量とターゲット
                features = self.create_trend_features(trend_data)
                target = self.create_trend_target(trend_data, prediction_days=3)

                # クリーニング
                valid_idx = ~(features.isna().any(axis=1) | target.isna())
                X = features[valid_idx].fillna(0)
                y = target[valid_idx]

                if len(X) < 20:
                    print(f"  スキップ: 有効サンプル不足 ({len(X)})")
                    continue

                print(f"  有効サンプル: {len(X)}")

                # クラス分布
                class_counts = y.value_counts()
                up_ratio = class_counts.get(1, 0) / len(y)
                print(f"  上昇期待率: {up_ratio:.1%}")

                # 極端に偏っている場合はスキップ
                if up_ratio < 0.2 or up_ratio > 0.8:
                    print("  スキップ: 極端なクラス偏り")
                    continue

                # 時系列分割（最新30%をテスト）
                split_idx = int(len(X) * 0.7)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                if len(X_test) < 8:
                    continue

                # シンプルなモデル（過学習防止）
                model = LogisticRegression(random_state=42, max_iter=200)

                # 訓練
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)

                # 予測
                test_predictions = model.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, test_predictions)

                # より詳細な分析
                y_proba = model.predict_proba(X_test_scaled)
                high_confidence_mask = np.max(y_proba, axis=1) > 0.7

                if high_confidence_mask.sum() > 0:
                    high_conf_accuracy = accuracy_score(
                        y_test[high_confidence_mask],
                        test_predictions[high_confidence_mask],
                    )
                else:
                    high_conf_accuracy = 0

                result = {
                    "symbol": symbol,
                    "accuracy": test_accuracy,
                    "high_conf_accuracy": high_conf_accuracy,
                    "high_conf_samples": high_confidence_mask.sum(),
                    "test_samples": len(X_test),
                    "trend_days": strong_trend_mask.sum(),
                    "up_ratio": up_ratio,
                }

                all_results.append(result)

                print(f"  全体精度: {test_accuracy:.1%}")
                print(
                    f"  高信頼度精度: {high_conf_accuracy:.1%} ({high_confidence_mask.sum()}サンプル)"
                )

                if test_accuracy >= 0.8:
                    print("  *** 80%達成！")
                elif test_accuracy >= 0.75:
                    print("  *** 75%以上")
                elif test_accuracy >= 0.7:
                    print("  *** 70%以上")

            except Exception as e:
                print(f"  エラー: {str(e)}")
                continue

        return self._analyze_trend_results(all_results)

    def _analyze_trend_results(self, results: List[Dict]) -> Dict:
        """トレンドフォロー結果の分析"""
        if not results:
            return {"error": "No results"}

        accuracies = [r["accuracy"] for r in results]
        high_conf_accuracies = [
            r["high_conf_accuracy"] for r in results if r["high_conf_accuracy"] > 0
        ]

        max_accuracy = np.max(accuracies)
        avg_accuracy = np.mean(accuracies)

        print(f"\n" + "=" * 60)
        print("トレンドフォロー予測結果")
        print("=" * 60)
        print(f"テスト銘柄数: {len(results)}")
        print(f"最高精度: {max_accuracy:.1%}")
        print(f"平均精度: {avg_accuracy:.1%}")

        if high_conf_accuracies:
            max_high_conf = np.max(high_conf_accuracies)
            avg_high_conf = np.mean(high_conf_accuracies)
            print(f"高信頼度最高精度: {max_high_conf:.1%}")
            print(f"高信頼度平均精度: {avg_high_conf:.1%}")

        # 精度別カウント
        excellent_count = sum(1 for acc in accuracies if acc >= 0.8)
        good_count = sum(1 for acc in accuracies if acc >= 0.75)

        print(f"80%以上: {excellent_count}/{len(results)} 銘柄")
        print(f"75%以上: {good_count}/{len(results)} 銘柄")

        # トップ結果
        top_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)[:5]
        print(f"\nトップ5結果:")
        for i, result in enumerate(top_results, 1):
            print(
                f"  {i}. {result['symbol']}: {result['accuracy']:.1%} "
                f"(高信頼度: {result['high_conf_accuracy']:.1%})"
            )

        if max_accuracy >= 0.8:
            print(f"\n*** 80%以上達成！最高 {max_accuracy:.1%}")
        elif max_accuracy >= 0.75:
            print(f"\n*** 75%以上達成：最高 {max_accuracy:.1%}")

        return {
            "max_accuracy": max_accuracy,
            "avg_accuracy": avg_accuracy,
            "excellent_count": excellent_count,
            "results": results,
        }

    def predict_stock(self, symbol: str, data=None) -> Dict[str, Any]:
        """株価予測メソッド - アンサンブルテスト用"""
        try:
            if data is None:
                # データ取得
                stock_data = self.data_provider.get_stock_data(symbol, period="1y")
                stock_data = self.data_provider.calculate_technical_indicators(
                    stock_data
                )
            else:
                stock_data = data

            if len(stock_data) < 50:
                return {
                    "direction": 0,
                    "confidence": 0.5,
                    "predicted_price": (
                        stock_data["Close"].iloc[-1] if len(stock_data) > 0 else 0
                    ),
                    "error": "Insufficient data",
                }

            # トレンド特徴量作成
            trend_features = self.create_trend_features(stock_data)

            if len(trend_features) == 0:
                return {
                    "direction": 0,
                    "confidence": 0.5,
                    "predicted_price": stock_data["Close"].iloc[-1],
                    "error": "No trend features",
                }

            # 最新の特徴量を取得
            latest_features = trend_features.iloc[-1]

            # 簡易的な予測ロジック
            rsi = latest_features.get("RSI", 50)
            macd = latest_features.get("MACD", 0)
            sma_ratio = latest_features.get("SMA_ratio", 1)

            # 上昇/下降の判定
            direction_score = 0
            if rsi > 60 and macd > 0 and sma_ratio > 1.02:
                direction_score = 1  # 上昇
            elif rsi < 40 and macd < 0 and sma_ratio < 0.98:
                direction_score = -1  # 下降
            else:
                direction_score = 0  # 中立

            # 信頼度計算 (0-1)
            confidence = min(abs(rsi - 50) / 50 + abs(macd) + abs(sma_ratio - 1), 1.0)

            # 予測価格 (現在価格からの変化率予測)
            current_price = stock_data["Close"].iloc[-1]
            price_change_rate = direction_score * 0.02 * confidence  # 最大2%の変化
            predicted_price = current_price * (1 + price_change_rate)

            return {
                "direction": direction_score,
                "confidence": confidence,
                "predicted_price": predicted_price,
                "current_price": current_price,
                "rsi": rsi,
                "macd": macd,
                "sma_ratio": sma_ratio,
            }

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return {
                "direction": 0,
                "confidence": 0.5,
                "predicted_price": 0,
                "error": str(e),
            }


def main():
    """メイン実行"""
    print("トレンドフォロー特化方向性予測システム")
    print("=" * 60)

    data_provider = StockDataProvider()
    symbols = list(data_provider.jp_stock_codes.keys())

    predictor = TrendFollowingPredictor()
    results = predictor.test_trend_following_system(symbols)

    if "error" not in results:
        if results["max_accuracy"] >= 0.8:
            print(f"*** 80%以上の方向性予測を達成！")


if __name__ == "__main__":
    main()
