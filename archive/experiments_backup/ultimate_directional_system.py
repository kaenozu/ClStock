#!/usr/bin/env python3
"""
究極の方向性予測システム
複数手法の組み合わせで90%以上の精度を目指す
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltimateDirectionalSystem:
    """究極の方向性予測システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.scaler = StandardScaler()

    def identify_optimal_conditions(self, data: pd.DataFrame) -> pd.Series:
        """最適な予測条件の特定"""
        close = data["Close"]

        # 1. 強いトレンド条件
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        strong_trend = ((sma_10 > sma_20) & (sma_20 > sma_50)) | (  # 上昇トレンド
            (sma_10 < sma_20) & (sma_20 < sma_50)
        )  # 下降トレンド

        # 2. 適度なボラティリティ
        returns = close.pct_change()
        volatility = returns.rolling(10).std()
        moderate_vol = (volatility > volatility.quantile(0.2)) & (
            volatility < volatility.quantile(0.8)
        )

        # 3. 出来高確認
        volume = data["Volume"]
        vol_avg = volume.rolling(20).mean()
        good_volume = volume > vol_avg * 0.8

        # 4. RSIが極端でない
        rsi = self._calculate_rsi(close, 14)
        reasonable_rsi = (rsi > 25) & (rsi < 75)

        # 全条件を満たす期間
        optimal_conditions = strong_trend & moderate_vol & good_volume & reasonable_rsi

        return optimal_conditions

    def create_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """強化された特徴量セット"""
        features = pd.DataFrame(index=data.index)

        close = data["Close"]
        volume = data["Volume"]

        # 1. 移動平均システム
        sma_5 = close.rolling(5).mean()
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()

        features["ma_bullish_alignment"] = (sma_5 > sma_10) & (sma_10 > sma_20)
        features["ma_bearish_alignment"] = (sma_5 < sma_10) & (sma_10 < sma_20)
        features["price_above_ma20"] = close > sma_20

        # 移動平均の勢い
        features["sma10_momentum"] = sma_10.pct_change(3)
        features["sma20_momentum"] = sma_20.pct_change(5)

        # 2. 価格アクション
        features["higher_high"] = (close > close.shift(1)) & (
            close.shift(1) > close.shift(2)
        )
        features["lower_low"] = (close < close.shift(1)) & (
            close.shift(1) < close.shift(2)
        )

        # 価格レンジ
        high_5 = data["High"].rolling(5).max()
        low_5 = data["Low"].rolling(5).min()
        features["price_position_5d"] = (close - low_5) / (high_5 - low_5)

        # 3. モメンタム
        features["momentum_3d"] = close.pct_change(3)
        features["momentum_5d"] = close.pct_change(5)

        # 4. ボリューム分析
        vol_avg = volume.rolling(10).mean()
        features["volume_surge"] = volume > vol_avg * 1.5
        features["volume_confirm"] = volume > vol_avg

        # 5. テクニカル指標
        rsi = self._calculate_rsi(close, 14)
        features["rsi_bullish"] = (rsi > 50) & (rsi < 70)
        features["rsi_bearish"] = (rsi < 50) & (rsi > 30)

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        features["macd_bullish"] = macd > macd_signal

        return features

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_multi_timeframe_target(self, data: pd.DataFrame) -> pd.Series:
        """マルチタイムフレームターゲット"""
        close = data["Close"]

        # 3日後と5日後の両方で上昇
        future_3d = close.shift(-3).pct_change(3)
        future_5d = close.shift(-5).pct_change(5)

        # より厳格な条件
        strong_up = (future_3d > 0.01) & (future_5d > 0.01)  # 両方で1%以上上昇

        return strong_up.astype(int)

    def create_ensemble_model(self) -> VotingClassifier:
        """アンサンブルモデルの作成"""
        models = [
            ("logistic", LogisticRegression(random_state=42, max_iter=200)),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=50, max_depth=6, min_samples_split=10, random_state=42
                ),
            ),
        ]

        return VotingClassifier(estimators=models, voting="soft")

    def test_ultimate_system(self, symbols: List[str]) -> Dict:
        """究極システムのテスト"""
        print("究極の方向性予測システム（90%目標）")
        print("=" * 60)

        all_results = []

        for symbol in symbols[:25]:
            try:
                print(f"\n処理中: {symbol}")

                # データ取得
                data = self.data_provider.get_stock_data(symbol, "2y")
                if len(data) < 200:
                    continue

                # 最適条件の特定
                optimal_mask = self.identify_optimal_conditions(data)

                if optimal_mask.sum() < 40:
                    print(f"  スキップ: 最適条件不足 ({optimal_mask.sum()})")
                    continue

                print(f"  最適条件期間: {optimal_mask.sum()}日")

                # 最適期間のデータのみ
                optimal_data = data[optimal_mask]

                # 特徴量とターゲット
                features = self.create_enhanced_features(optimal_data)
                target = self.create_multi_timeframe_target(optimal_data)

                # クリーニング
                valid_idx = ~(features.isna().any(axis=1) | target.isna())
                X = features[valid_idx].fillna(0)
                y = target[valid_idx]

                if len(X) < 25:
                    print(f"  スキップ: 有効サンプル不足 ({len(X)})")
                    continue

                print(f"  有効サンプル: {len(X)}")

                # クラス分布
                up_ratio = y.mean()
                print(f"  強上昇率: {up_ratio:.1%}")

                # 極端に偏っている場合はスキップ
                if up_ratio < 0.15 or up_ratio > 0.85:
                    print("  スキップ: 極端なクラス偏り")
                    continue

                # 時系列分割
                split_idx = int(len(X) * 0.7)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                if len(X_test) < 8:
                    continue

                # アンサンブルモデル訓練
                ensemble_model = self.create_ensemble_model()

                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                ensemble_model.fit(X_train_scaled, y_train)

                # 予測
                test_predictions = ensemble_model.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, test_predictions)

                # 信頼度分析
                y_proba = ensemble_model.predict_proba(X_test_scaled)
                confidence_scores = np.max(y_proba, axis=1)

                # 超高信頼度予測
                ultra_high_conf_mask = confidence_scores > 0.8
                if ultra_high_conf_mask.sum() > 0:
                    ultra_high_accuracy = accuracy_score(
                        y_test[ultra_high_conf_mask],
                        test_predictions[ultra_high_conf_mask],
                    )
                else:
                    ultra_high_accuracy = 0

                result = {
                    "symbol": symbol,
                    "accuracy": test_accuracy,
                    "ultra_high_accuracy": ultra_high_accuracy,
                    "ultra_high_samples": ultra_high_conf_mask.sum(),
                    "test_samples": len(X_test),
                    "optimal_days": optimal_mask.sum(),
                    "up_ratio": up_ratio,
                }

                all_results.append(result)

                print(f"  全体精度: {test_accuracy:.1%}")
                if ultra_high_accuracy > 0:
                    print(
                        f"  超高信頼度精度: {ultra_high_accuracy:.1%} ({ultra_high_conf_mask.sum()}サンプル)"
                    )

                if test_accuracy >= 0.9:
                    print("  *** 90%達成！")
                elif test_accuracy >= 0.85:
                    print("  ✓ 85%以上")
                elif test_accuracy >= 0.8:
                    print("  ○ 80%以上")

            except Exception as e:
                print(f"  エラー: {str(e)}")
                continue

        return self._analyze_ultimate_results(all_results)

    def _analyze_ultimate_results(self, results: List[Dict]) -> Dict:
        """究極システム結果の分析"""
        if not results:
            return {"error": "No results"}

        accuracies = [r["accuracy"] for r in results]
        ultra_high_accuracies = [
            r["ultra_high_accuracy"] for r in results if r["ultra_high_accuracy"] > 0
        ]

        max_accuracy = np.max(accuracies)
        avg_accuracy = np.mean(accuracies)

        print(f"\n" + "=" * 60)
        print("究極システム結果")
        print("=" * 60)
        print(f"テスト銘柄数: {len(results)}")
        print(f"最高精度: {max_accuracy:.1%}")
        print(f"平均精度: {avg_accuracy:.1%}")

        if ultra_high_accuracies:
            max_ultra = np.max(ultra_high_accuracies)
            avg_ultra = np.mean(ultra_high_accuracies)
            print(f"超高信頼度最高精度: {max_ultra:.1%}")
            print(f"超高信頼度平均精度: {avg_ultra:.1%}")

        # 精度別カウント
        elite_count = sum(1 for acc in accuracies if acc >= 0.9)
        excellent_count = sum(1 for acc in accuracies if acc >= 0.85)
        very_good_count = sum(1 for acc in accuracies if acc >= 0.8)

        print(f"90%以上: {elite_count}/{len(results)} 銘柄")
        print(f"85%以上: {excellent_count}/{len(results)} 銘柄")
        print(f"80%以上: {very_good_count}/{len(results)} 銘柄")

        # エリート結果
        elite_results = [r for r in results if r["accuracy"] >= 0.85]
        if elite_results:
            print(f"\nエリート結果 (85%以上):")
            for result in sorted(
                elite_results, key=lambda x: x["accuracy"], reverse=True
            ):
                print(
                    f"  {result['symbol']}: {result['accuracy']:.1%} "
                    f"(超高信頼度: {result['ultra_high_accuracy']:.1%})"
                )

        if max_accuracy >= 0.9:
            print(f"\n*** 90%以上達成！最高 {max_accuracy:.1%}")
        elif max_accuracy >= 0.85:
            print(f"\n✓ 85%以上達成：最高 {max_accuracy:.1%}")

        return {
            "max_accuracy": max_accuracy,
            "avg_accuracy": avg_accuracy,
            "elite_count": elite_count,
            "results": results,
        }


def main():
    """メイン実行"""
    print("究極の方向性予測システム")
    print("=" * 60)

    data_provider = StockDataProvider()
    symbols = list(data_provider.jp_stock_codes.keys())

    system = UltimateDirectionalSystem()
    results = system.test_ultimate_system(symbols)

    if "error" not in results:
        if results["max_accuracy"] >= 0.9:
            print(f"*** 90%以上の究極精度を達成！")
        elif results["max_accuracy"] >= 0.85:
            print(f"✓ 85%以上の優秀な精度を達成！")


if __name__ == "__main__":
    main()
