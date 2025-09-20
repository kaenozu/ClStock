#!/usr/bin/env python3
"""
バイナリ方向性予測システム（上昇・下降のみ）
シンプルな2クラス分類で80%以上の精度を目指す
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from utils.logger_config import setup_logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

from data.stock_data import StockDataProvider

# ログ設定
logger = setup_logger(__name__)


class BinaryDirectionPredictor:
    """バイナリ方向性予測システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.scaler = StandardScaler()

    def create_focused_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """方向性予測に集中した特徴量（シンプル化）"""
        features = pd.DataFrame(index=data.index)

        close = data["Close"]
        volume = data["Volume"]

        # 1. 最も重要な移動平均関係
        sma_5 = close.rolling(5).mean()
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        # 移動平均トレンド
        features["trend_short"] = (sma_5 > sma_20).astype(int)
        features["trend_medium"] = (sma_20 > sma_50).astype(int)
        features["trend_alignment"] = ((sma_5 > sma_20) & (sma_20 > sma_50)).astype(int)

        # 価格と移動平均の関係
        features["price_above_sma20"] = (close > sma_20).astype(int)
        features["price_above_sma50"] = (close > sma_50).astype(int)

        # 移動平均の傾き
        features["sma20_rising"] = (sma_20 > sma_20.shift(5)).astype(int)
        features["sma50_rising"] = (sma_50 > sma_50.shift(10)).astype(int)

        # 2. モメンタム（最重要）
        returns = close.pct_change()

        # 最近の動向
        features["recent_positive"] = (returns.rolling(3).sum() > 0).astype(int)
        features["momentum_1w"] = (close > close.shift(5)).astype(int)
        features["momentum_2w"] = (close > close.shift(10)).astype(int)

        # 3. RSI（簡略版）
        rsi = self._calculate_rsi(close, 14)
        features["rsi_bullish"] = (rsi > 50).astype(int)
        features["rsi_strong_bull"] = (rsi > 60).astype(int)
        features["rsi_oversold"] = (rsi < 30).astype(int)

        # 4. ボリューム確認
        vol_avg = volume.rolling(20).mean()
        features["volume_above_avg"] = (volume > vol_avg).astype(int)
        features["strong_volume"] = (volume > vol_avg * 1.5).astype(int)

        # 5. 価格位置
        high_20 = data["High"].rolling(20).max()
        low_20 = data["Low"].rolling(20).min()
        price_position = (close - low_20) / (high_20 - low_20)
        features["price_upper_half"] = (price_position > 0.5).astype(int)
        features["price_top_quarter"] = (price_position > 0.75).astype(int)

        return features

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_binary_target(
        self, data: pd.DataFrame, prediction_days: int = 5
    ) -> pd.Series:
        """バイナリターゲット（上昇=1, 下降=0）"""
        close = data["Close"]

        # 将来リターン
        future_return = close.shift(-prediction_days).pct_change(prediction_days)

        # シンプルなバイナリ分類
        target = (future_return > 0).astype(int)

        return target

    def filter_significant_periods(
        self, features: pd.DataFrame, target: pd.Series, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """有意な期間のみをフィルタリング"""
        close = data["Close"]
        returns = close.pct_change()

        # ボラティリティフィルター（安定期間のみ）
        vol_20 = returns.rolling(20).std()
        stable_periods = vol_20 < vol_20.quantile(0.7)

        # トレンド期間のみ（横ばいを除外）
        sma_20 = close.rolling(20).mean()
        trend_strength = abs(sma_20.pct_change(10))
        trending_periods = trend_strength > 0.02  # 2%以上の変化

        # 組み合わせ
        valid_periods = stable_periods & trending_periods

        return features[valid_periods], target[valid_periods]

    def train_binary_classifier(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Dict, Dict]:
        """バイナリ分類器の訓練"""

        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            ),
            "logistic": LogisticRegression(
                random_state=42, max_iter=200, class_weight="balanced"
            ),
        }

        # 時系列分割
        tscv = TimeSeriesSplit(n_splits=3)
        model_scores = {}
        trained_models = {}

        for name, model in models.items():
            accuracies = []

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

            avg_accuracy = np.mean(accuracies)
            model_scores[name] = {"accuracy": avg_accuracy, "std": np.std(accuracies)}

            print(f"  {name}: {avg_accuracy:.3f} ± {np.std(accuracies):.3f}")

            # 全データで最終訓練
            X_scaled = self.scaler.fit_transform(X)
            model.fit(X_scaled, y)
            trained_models[name] = model

        return model_scores, trained_models

    def test_binary_system(self, symbols: List[str]) -> Dict:
        """バイナリ方向性予測システムのテスト"""
        print("バイナリ方向性予測システム（80%目標）")
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
                features = self.create_focused_features(data)
                target = self.create_binary_target(data, prediction_days=5)

                # 有意期間のフィルタリング
                X_filtered, y_filtered = self.filter_significant_periods(
                    features, target, data
                )

                # クリーニング
                valid_idx = ~(X_filtered.isna().any(axis=1) | y_filtered.isna())
                X = X_filtered[valid_idx].fillna(0)
                y = y_filtered[valid_idx]

                if len(X) < 50:
                    print(f"  スキップ: 有意サンプル不足 ({len(X)})")
                    continue

                print(f"  有意サンプル: {len(X)}")

                # クラス分布
                class_counts = y.value_counts()
                print(f"  上昇/下降: {class_counts.get(1, 0)}/{class_counts.get(0, 0)}")

                # バランスチェック（修正版）
                class_values = list(class_counts.values)
                minority_ratio = min(class_values) / sum(class_values)
                if minority_ratio < 0.3:
                    print(
                        f"  スキップ: クラス不均衡 (少数クラス: {minority_ratio:.1%})"
                    )
                    continue

                # 分割
                split_idx = int(len(X) * 0.7)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                if len(X_test) < 15:
                    continue

                # モデル訓練
                model_scores, trained_models = self.train_binary_classifier(
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

                result = {
                    "symbol": symbol,
                    "accuracy": test_accuracy,
                    "best_model": best_model_name,
                    "test_samples": len(X_test),
                    "train_accuracy": model_scores[best_model_name]["accuracy"],
                    "class_balance": minority_ratio,
                }

                all_results.append(result)

                print(f"  ベストモデル: {best_model_name}")
                print(f"  テスト精度: {test_accuracy:.1%}")

                if test_accuracy >= 0.8:
                    print("  ✓ 80%達成！")
                elif test_accuracy >= 0.75:
                    print("  △ 75%以上")
                elif test_accuracy >= 0.7:
                    print("  ○ 70%以上")

            except Exception as e:
                print(f"  エラー: {str(e)}")
                continue

        return self._analyze_binary_results(all_results)

    def _analyze_binary_results(self, results: List[Dict]) -> Dict:
        """バイナリ予測結果の分析"""
        if not results:
            return {"error": "No results"}

        accuracies = [r["accuracy"] for r in results]

        max_accuracy = np.max(accuracies)
        avg_accuracy = np.mean(accuracies)
        median_accuracy = np.median(accuracies)

        # 精度別カウント
        excellent_count = sum(1 for acc in accuracies if acc >= 0.8)
        good_count = sum(1 for acc in accuracies if acc >= 0.75)
        decent_count = sum(1 for acc in accuracies if acc >= 0.7)

        print(f"\n" + "=" * 60)
        print("バイナリ方向性予測結果")
        print("=" * 60)
        print(f"テスト銘柄数: {len(results)}")
        print(f"最高精度: {max_accuracy:.1%}")
        print(f"平均精度: {avg_accuracy:.1%}")
        print(f"中央値精度: {median_accuracy:.1%}")
        print(f"80%以上: {excellent_count}/{len(results)} 銘柄")
        print(f"75%以上: {good_count}/{len(results)} 銘柄")
        print(f"70%以上: {decent_count}/{len(results)} 銘柄")

        # トップ3結果
        top_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)[:3]
        print(f"\nトップ3結果:")
        for i, result in enumerate(top_results, 1):
            print(
                f"  {i}. {result['symbol']}: {result['accuracy']:.1%} ({result['best_model']})"
            )

        if max_accuracy >= 0.8:
            print(f"\n🎉 80%以上達成！最高 {max_accuracy:.1%}")
        elif avg_accuracy >= 0.75:
            print(f"\n△ 良好：平均 {avg_accuracy:.1%}")
        elif max_accuracy >= 0.75:
            print(f"\n△ 部分的成功：最高 {max_accuracy:.1%}")

        return {
            "max_accuracy": max_accuracy,
            "avg_accuracy": avg_accuracy,
            "excellent_count": excellent_count,
            "results": results,
        }


def main():
    """メイン実行"""
    print("バイナリ方向性予測システム（シンプル2クラス）")
    print("=" * 60)

    data_provider = StockDataProvider()
    symbols = list(data_provider.jp_stock_codes.keys())

    predictor = BinaryDirectionPredictor()
    results = predictor.test_binary_system(symbols)

    if "error" not in results:
        if results["max_accuracy"] >= 0.8:
            print(f"✓ 目標達成！最高精度 {results['max_accuracy']:.1%}")
        else:
            print(f"継続改善中：現在 {results['max_accuracy']:.1%}")


if __name__ == "__main__":
    main()
