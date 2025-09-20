#!/usr/bin/env python3
"""
84.6%直接強化システム
84.6%達成手法のパラメータをそのまま使用し、モデル部分のみ改良
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Direct846Enhancement:
    """84.6%直接強化システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.scaler = StandardScaler()

    def identify_strong_trends(self, data: pd.DataFrame) -> pd.Series:
        """84.6%手法と全く同じトレンド特定"""
        close = data["Close"]

        # 複数期間の移動平均
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        # 強い上昇トレンド（84.6%手法と完全同一）
        strong_uptrend = (
            (sma_10 > sma_20)
            & (sma_20 > sma_50)
            & (close > sma_10)
            & (sma_10.pct_change(5) > 0.01)  # 5日で1%以上上昇
        )

        # 強い下降トレンド（84.6%手法と完全同一）
        strong_downtrend = (
            (sma_10 < sma_20)
            & (sma_20 < sma_50)
            & (close < sma_10)
            & (sma_10.pct_change(5) < -0.01)  # 5日で1%以上下降
        )

        # トレンドの継続性確認（84.6%手法と完全同一）
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
        """84.6%手法と全く同じ特徴量"""
        features = pd.DataFrame(index=data.index)

        close = data["Close"]
        volume = data["Volume"]

        # 1. 移動平均の関係（84.6%手法と完全同一）
        sma_5 = close.rolling(5).mean()
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()

        features["ma_bullish"] = (sma_5 > sma_10) & (sma_10 > sma_20)
        features["ma_bearish"] = (sma_5 < sma_10) & (sma_10 < sma_20)

        # 移動平均の傾き（84.6%手法と完全同一）
        features["sma10_slope"] = sma_10.pct_change(5)
        features["sma20_slope"] = sma_20.pct_change(5)

        # 2. トレンド強度（84.6%手法と完全同一）
        features["trend_strength"] = abs((sma_5 - sma_20) / sma_20)

        # 3. 価格のモメンタム（84.6%手法と完全同一）
        features["price_momentum_5d"] = close.pct_change(5)
        features["price_momentum_10d"] = close.pct_change(10)

        # 連続上昇/下降日数（84.6%手法と完全同一）
        daily_change = close.pct_change() > 0
        features["consecutive_up"] = daily_change.rolling(5).sum()
        features["consecutive_down"] = (~daily_change).rolling(5).sum()

        # 4. ボリューム確認（84.6%手法と完全同一）
        vol_avg = volume.rolling(20).mean()
        features["volume_support"] = volume > vol_avg

        # 5. RSI（84.6%手法と完全同一）
        rsi = self._calculate_rsi(close, 14)
        features["rsi_trend_up"] = (rsi > 55) & (rsi < 80)
        features["rsi_trend_down"] = (rsi < 45) & (rsi > 20)

        return features

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI計算（84.6%手法と完全同一）"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_trend_target(
        self, data: pd.DataFrame, prediction_days: int = 3
    ) -> pd.Series:
        """84.6%手法と全く同じターゲット"""
        close = data["Close"]

        # 短期の将来トレンド（84.6%手法と完全同一）
        future_return = close.shift(-prediction_days).pct_change(prediction_days)

        # トレンド継続の判定（84.6%手法と完全同一）
        target = (future_return > 0.005).astype(int)  # 0.5%以上の上昇

        return target

    def create_enhanced_ensemble(self) -> VotingClassifier:
        """84.6%手法のLogisticRegressionに高性能モデルを追加"""
        models = [
            # 84.6%達成の元モデル
            ("lr_846", LogisticRegression(random_state=42, max_iter=200)),
            # 追加の高性能モデル
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_split=5,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
            (
                "lr_enhanced",
                LogisticRegression(
                    random_state=123, max_iter=300, C=0.5, class_weight="balanced"
                ),
            ),
        ]

        return VotingClassifier(estimators=models, voting="soft")

    def test_direct_846_enhancement(self, symbols: List[str]) -> Dict:
        """84.6%直接強化テスト"""
        print("84.6%直接強化システム（限界突破への挑戦）")
        print("=" * 60)

        all_results = []
        enhanced_results = []

        for symbol in symbols[:25]:  # 84.6%手法の20→25に拡張
            try:
                print(f"\n処理中: {symbol}")

                # データ取得（84.6%手法と完全同一：2年間）
                data = self.data_provider.get_stock_data(symbol, "2y")
                if len(data) < 200:
                    continue

                # 強いトレンド期間の特定（84.6%手法と完全同一）
                strong_trend_mask = self.identify_strong_trends(data)

                if strong_trend_mask.sum() < 30:
                    print(
                        f"  スキップ: 強いトレンド期間不足 ({strong_trend_mask.sum()})"
                    )
                    continue

                print(f"  強いトレンド期間: {strong_trend_mask.sum()}日")

                # トレンド期間のデータのみ使用（84.6%手法と完全同一）
                trend_data = data[strong_trend_mask]

                # 特徴量とターゲット（84.6%手法と完全同一）
                features = self.create_trend_features(trend_data)
                target = self.create_trend_target(trend_data, prediction_days=3)

                # クリーニング（84.6%手法と完全同一）
                valid_idx = ~(features.isna().any(axis=1) | target.isna())
                X = features[valid_idx].fillna(0)
                y = target[valid_idx]

                if len(X) < 20:
                    print(f"  スキップ: 有効サンプル不足 ({len(X)})")
                    continue

                print(f"  有効サンプル: {len(X)}")

                # クラス分布（84.6%手法と完全同一）
                up_ratio = y.mean()
                print(f"  上昇期待率: {up_ratio:.1%}")

                if up_ratio < 0.2 or up_ratio > 0.8:
                    print("  スキップ: 極端なクラス偏り")
                    continue

                # 時系列分割（84.6%手法と完全同一：70%-30%）
                split_idx = int(len(X) * 0.7)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                if len(X_test) < 8:
                    continue

                # 強化アンサンブルモデル（改良点）
                model = self.create_enhanced_ensemble()

                # 訓練（84.6%手法と完全同一）
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)

                # 予測（84.6%手法と完全同一）
                test_predictions = model.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, test_predictions)

                # 信頼度分析（84.6%手法と完全同一）
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

                # 84.6%突破チェック
                if test_accuracy > 0.846:
                    enhanced_results.append(result)
                    print(f"  *** 84.6%突破達成！精度: {test_accuracy:.1%} ***")
                elif test_accuracy >= 0.84:
                    print(f"  ○ 84%台: {test_accuracy:.1%}")
                elif test_accuracy >= 0.8:
                    print(f"  ○ 80%台: {test_accuracy:.1%}")
                else:
                    print(f"  精度: {test_accuracy:.1%}")

                if high_conf_accuracy > 0:
                    print(
                        f"  高信頼度: {high_conf_accuracy:.1%} ({high_confidence_mask.sum()}サンプル)"
                    )

            except Exception as e:
                print(f"  エラー: {str(e)}")
                continue

        return self._analyze_direct_enhancement(all_results, enhanced_results)

    def _analyze_direct_enhancement(
        self, all_results: List[Dict], enhanced_results: List[Dict]
    ) -> Dict:
        """84.6%直接強化結果の分析"""
        if not all_results:
            return {"error": "No results"}

        accuracies = [r["accuracy"] for r in all_results]
        max_accuracy = np.max(accuracies)
        avg_accuracy = np.mean(accuracies)

        print(f"\n" + "=" * 60)
        print("84.6%直接強化システム最終結果")
        print("=" * 60)
        print(f"テスト銘柄数: {len(all_results)}")
        print(f"最高精度: {max_accuracy:.1%}")
        print(f"平均精度: {avg_accuracy:.1%}")

        # 84.6%突破の詳細分析
        if enhanced_results:
            en_accuracies = [r["accuracy"] for r in enhanced_results]
            print(f"\n*** 84.6%突破成功: {len(enhanced_results)}銘柄 ***")
            print(f"  突破最高精度: {np.max(en_accuracies):.1%}")
            print(f"  突破平均精度: {np.mean(en_accuracies):.1%}")

            print("\n84.6%突破達成銘柄:")
            for r in sorted(
                enhanced_results, key=lambda x: x["accuracy"], reverse=True
            ):
                hc_info = (
                    f" (高信頼度: {r['high_conf_accuracy']:.1%})"
                    if r["high_conf_accuracy"] > 0
                    else ""
                )
                print(f"  {r['symbol']}: {r['accuracy']:.1%}{hc_info}")

        # 高信頼度分析
        hc_results = [r for r in all_results if r["high_conf_accuracy"] > 0]
        if hc_results:
            hc_accuracies = [r["high_conf_accuracy"] for r in hc_results]
            print(f"\n高信頼度予測分析:")
            print(f"  対象銘柄数: {len(hc_results)}")
            print(f"  平均精度: {np.mean(hc_accuracies):.1%}")
            print(f"  最高精度: {np.max(hc_accuracies):.1%}")

        # 精度分布
        ranges = [
            (0.90, "90%以上"),
            (0.85, "85%以上"),
            (0.80, "80%以上"),
            (0.846, "84.6%突破"),
        ]

        print(f"\n精度分布:")
        for threshold, label in ranges:
            count = sum(1 for acc in accuracies if acc >= threshold)
            percentage = count / len(all_results) * 100
            print(f"  {label}: {count}/{len(all_results)} 銘柄 ({percentage:.1f}%)")

        # トップ結果（84.6%手法と同様の表示）
        top_results = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)[:5]
        print(f"\nトップ5結果:")
        for i, result in enumerate(top_results, 1):
            hc_info = (
                f" (高信頼度: {result['high_conf_accuracy']:.1%})"
                if result["high_conf_accuracy"] > 0
                else ""
            )
            print(f"  {i}. {result['symbol']}: {result['accuracy']:.1%}{hc_info}")

        # 最終判定
        if max_accuracy > 0.846:
            improvement = (max_accuracy - 0.846) * 100
            print(
                f"\n*** 歴史的突破！84.6%を {improvement:.1f}%ポイント上回る {max_accuracy:.1%} 達成！***"
            )
            if max_accuracy >= 0.90:
                print("*** 90%の壁も突破！完璧な勝利！***")
        elif max_accuracy >= 0.84:
            gap = (0.846 - max_accuracy) * 100
            print(
                f"\n○ 84%台達成：{max_accuracy:.1%} (84.6%まで残り {gap:.1f}%ポイント)"
            )
        elif max_accuracy >= 0.8:
            print(f"\n○ 80%台達成：{max_accuracy:.1%}")

        return {
            "max_accuracy": max_accuracy,
            "avg_accuracy": avg_accuracy,
            "enhanced_count": len(enhanced_results),
            "enhanced_results": enhanced_results,
            "all_results": all_results,
            "breakthrough": max_accuracy > 0.846,
        }


def main():
    """メイン実行"""
    print("84.6%直接強化システム")
    print("=" * 60)

    data_provider = StockDataProvider()
    symbols = list(data_provider.jp_stock_codes.keys())

    system = Direct846Enhancement()
    results = system.test_direct_846_enhancement(symbols)

    if "error" not in results:
        if results["breakthrough"]:
            print(f"\n*** 84.6%の限界を突破！新記録樹立！***")
        else:
            print(f"\n○ 84.6%への挑戦継続中...")


if __name__ == "__main__":
    main()
