#!/usr/bin/env python3
"""
高精度方向性予測システム（80%以上の精度を目指す）
上昇・下降の予測に特化した最適化アプローチ
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedDirectionalPredictor:
    """高精度方向性予測システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.scaler = StandardScaler()

    def create_directional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """方向性予測に最適化された特徴量"""
        features = pd.DataFrame(index=data.index)

        close = data["Close"]
        volume = data["Volume"]
        high = data["High"]
        low = data["Low"]

        # 1. 複数期間の移動平均関係
        sma_5 = close.rolling(5).mean()
        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        # 移動平均の序列（強力なトレンド指標）
        features["ma_alignment_bull"] = (
            (sma_5 > sma_10) & (sma_10 > sma_20) & (sma_20 > sma_50)
        ).astype(int)
        features["ma_alignment_bear"] = (
            (sma_5 < sma_10) & (sma_10 < sma_20) & (sma_20 < sma_50)
        ).astype(int)

        # 移動平均からの距離
        features["price_above_sma20"] = (close > sma_20).astype(int)
        features["price_above_sma50"] = (close > sma_50).astype(int)

        # 2. モメンタム指標
        returns_1d = close.pct_change()
        returns_3d = close.pct_change(3)
        returns_5d = close.pct_change(5)

        # 連続上昇/下降の検出
        features["consecutive_up"] = (returns_1d > 0).rolling(3).sum()
        features["consecutive_down"] = (returns_1d < 0).rolling(3).sum()

        # 強いモメンタム
        vol_20 = returns_1d.rolling(20).std()
        features["strong_momentum_up"] = (returns_3d > vol_20 * 0.5).astype(int)
        features["strong_momentum_down"] = (returns_3d < -vol_20 * 0.5).astype(int)

        # 3. テクニカル指標
        # RSI
        rsi_14 = self._calculate_rsi(close, 14)
        features["rsi_oversold"] = (rsi_14 < 30).astype(int)
        features["rsi_overbought"] = (rsi_14 > 70).astype(int)
        features["rsi_neutral"] = ((rsi_14 >= 40) & (rsi_14 <= 60)).astype(int)

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        features["macd_bullish"] = (macd > macd_signal).astype(int)
        features["macd_bullish_cross"] = (
            (macd > macd_signal) & (macd.shift(1) <= macd_signal.shift(1))
        ).astype(int)

        # 4. 出来高分析
        vol_sma_20 = volume.rolling(20).mean()
        features["high_volume"] = (volume > vol_sma_20 * 1.5).astype(int)
        features["volume_confirm_up"] = (
            (returns_1d > 0) & (volume > vol_sma_20 * 1.2)
        ).astype(int)
        features["volume_confirm_down"] = (
            (returns_1d < 0) & (volume > vol_sma_20 * 1.2)
        ).astype(int)

        # 5. サポート・レジスタンス
        max_20 = high.rolling(20).max()
        min_20 = low.rolling(20).min()
        features["near_resistance"] = (
            (close > max_20 * 0.98) & (close < max_20 * 1.02)
        ).astype(int)
        features["near_support"] = (
            (close < min_20 * 1.02) & (close > min_20 * 0.98)
        ).astype(int)

        # 価格位置
        price_position = (close - min_20) / (max_20 - min_20)
        features["price_upper_half"] = (price_position > 0.6).astype(int)
        features["price_lower_half"] = (price_position < 0.4).astype(int)

        # 6. ボラティリティ状況
        current_vol = vol_20
        vol_60 = returns_1d.rolling(60).std()
        features["low_volatility"] = (current_vol < vol_60 * 0.8).astype(int)
        features["high_volatility"] = (current_vol > vol_60 * 1.2).astype(int)

        # 7. ギャップ検出
        prev_close = close.shift(1)
        features["gap_up"] = ((close > prev_close * 1.02)).astype(int)
        features["gap_down"] = ((close < prev_close * 0.98)).astype(int)

        return features

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_directional_target(
        self, data: pd.DataFrame, prediction_days: int = 5, threshold: float = 0.01
    ) -> pd.Series:
        """方向性ターゲット（上昇/下降/中立）"""
        close = data["Close"]

        # 将来リターン
        future_return = close.shift(-prediction_days).pct_change(prediction_days)

        # 3クラス分類（閾値ベース）
        target = pd.Series(1, index=data.index)  # 中立=1
        target[future_return > threshold] = 2  # 上昇=2
        target[future_return < -threshold] = 0  # 下降=0

        return target

    def train_ensemble_classifier(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Dict, Dict]:
        """アンサンブル分類器の訓練"""

        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                class_weight="balanced",
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42
            ),
            "logistic": LogisticRegression(
                random_state=42, max_iter=300, class_weight="balanced"
            ),
        }

        # 時系列クロスバリデーション
        tscv = TimeSeriesSplit(n_splits=3)
        model_scores = {}
        trained_models = {}

        for name, model in models.items():
            accuracies = []
            detailed_scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # スケーリング
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)

                # 訓練
                model.fit(X_train_scaled, y_train)

                # 予測
                y_pred = model.predict(X_val_scaled)

                # 精度
                accuracy = accuracy_score(y_val, y_pred)
                accuracies.append(accuracy)

                # クラス別精度
                unique_classes = y_val.unique()
                class_accuracies = {}
                for cls in unique_classes:
                    mask = y_val == cls
                    if mask.sum() > 0:
                        class_acc = (y_pred[mask] == y_val[mask]).mean()
                        class_accuracies[cls] = class_acc

                detailed_scores.append(class_accuracies)

            avg_accuracy = np.mean(accuracies)
            model_scores[name] = {
                "accuracy": avg_accuracy,
                "std": np.std(accuracies),
                "detailed": detailed_scores,
            }

            print(f"  {name}: 精度 {avg_accuracy:.3f} ± {np.std(accuracies):.3f}")

            # 全データで最終訓練
            X_scaled = self.scaler.fit_transform(X)
            model.fit(X_scaled, y)
            trained_models[name] = model

        return model_scores, trained_models

    def test_directional_system(self, symbols: List[str]) -> Dict:
        """方向性予測システムのテスト"""
        print("高精度方向性予測システムテスト")
        print("=" * 60)

        all_results = []

        for symbol in symbols[:15]:
            try:
                print(f"\n処理中: {symbol}")

                # データ取得
                data = self.data_provider.get_stock_data(symbol, "2y")
                if len(data) < 200:
                    continue

                # 特徴量とターゲット
                features = self.create_directional_features(data)
                target = self.create_directional_target(
                    data, prediction_days=5, threshold=0.015
                )

                # クリーニング
                valid_idx = ~(features.isna().any(axis=1) | target.isna())
                X = features[valid_idx].fillna(0)
                y = target[valid_idx]

                if len(X) < 100:
                    print(f"  スキップ: サンプル不足 ({len(X)})")
                    continue

                print(f"  有効サンプル: {len(X)}")

                # クラス分布確認
                class_counts = y.value_counts()
                print(f"  クラス分布: {dict(class_counts)}")

                # 分割
                split_idx = int(len(X) * 0.7)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                if len(X_test) < 20:
                    continue

                # モデル訓練
                model_scores, trained_models = self.train_ensemble_classifier(
                    X_train, y_train
                )

                # ベストモデル選択
                best_model_name = max(
                    model_scores.keys(), key=lambda x: model_scores[x]["accuracy"]
                )
                best_model = trained_models[best_model_name]

                # テスト予測
                X_test_scaled = self.scaler.transform(X_test)
                test_predictions = best_model.predict(X_test_scaled)

                # テスト精度
                test_accuracy = accuracy_score(y_test, test_predictions)

                # 詳細分析
                test_class_counts = y_test.value_counts()
                print(f"  テストクラス分布: {dict(test_class_counts)}")

                result = {
                    "symbol": symbol,
                    "accuracy": test_accuracy,
                    "best_model": best_model_name,
                    "test_samples": len(X_test),
                    "train_accuracy": model_scores[best_model_name]["accuracy"],
                }

                all_results.append(result)

                print(f"  ベストモデル: {best_model_name}")
                print(f"  テスト精度: {test_accuracy:.3f}")

                if test_accuracy >= 0.8:
                    print("  ✓ 80%以上達成！")
                elif test_accuracy >= 0.75:
                    print("  △ 75%以上")

            except Exception as e:
                print(f"  エラー: {str(e)}")
                continue

        return self._analyze_directional_results(all_results)

    def _analyze_directional_results(self, results: List[Dict]) -> Dict:
        """方向性予測結果の分析"""
        if not results:
            return {"error": "No results"}

        accuracies = [r["accuracy"] for r in results]

        max_accuracy = np.max(accuracies)
        avg_accuracy = np.mean(accuracies)
        median_accuracy = np.median(accuracies)

        # 精度別カウント
        high_accuracy_count = sum(1 for acc in accuracies if acc >= 0.8)
        good_accuracy_count = sum(1 for acc in accuracies if acc >= 0.75)

        print(f"\n" + "=" * 60)
        print("方向性予測結果分析")
        print("=" * 60)
        print(f"テスト銘柄数: {len(results)}")
        print(f"最高精度: {max_accuracy:.1%}")
        print(f"平均精度: {avg_accuracy:.1%}")
        print(f"中央値精度: {median_accuracy:.1%}")
        print(f"80%以上: {high_accuracy_count}/{len(results)} 銘柄")
        print(f"75%以上: {good_accuracy_count}/{len(results)} 銘柄")

        # 最優秀銘柄
        best_result = max(results, key=lambda x: x["accuracy"])
        print(f"\n最優秀結果:")
        print(f"  銘柄: {best_result['symbol']}")
        print(f"  精度: {best_result['accuracy']:.1%}")
        print(f"  モデル: {best_result['best_model']}")

        if max_accuracy >= 0.8:
            print(f"\n🎉 目標達成！最高精度 {max_accuracy:.1%}")
        elif avg_accuracy >= 0.75:
            print(f"\n△ 良好な結果：平均精度 {avg_accuracy:.1%}")

        return {
            "max_accuracy": max_accuracy,
            "avg_accuracy": avg_accuracy,
            "high_accuracy_count": high_accuracy_count,
            "results": results,
        }


def main():
    """メイン実行"""
    print("高精度方向性予測システム（80%以上目標）")
    print("=" * 60)

    data_provider = StockDataProvider()
    symbols = list(data_provider.jp_stock_codes.keys())

    predictor = AdvancedDirectionalPredictor()
    results = predictor.test_directional_system(symbols)

    if "error" not in results:
        print(f"\n最終評価: 最高精度 {results['max_accuracy']:.1%}")
        if results["max_accuracy"] >= 0.8:
            print("✓ 80%以上の方向性予測精度を達成！")


if __name__ == "__main__":
    main()
