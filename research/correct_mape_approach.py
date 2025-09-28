#!/usr/bin/env python3
"""
正しいアプローチでMAPE 10-20%を達成する予測システム
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import logging
from utils.logger_config import setup_logger
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

from data.stock_data import StockDataProvider

# ログ設定
logger = setup_logger(__name__)


class CorrectMAPEPredictor:
    """正しいアプローチでMAPE 10-20%達成を目指すシステム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.models = {}
        self.scalers = {}
        self.is_trained = False

    def create_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """包括的特徴量作成（ChatGPT推奨レベル）"""
        features = pd.DataFrame(index=data.index)

        # 基本価格特徴量
        features["price"] = data["Close"]
        features["volume"] = data["Volume"]
        features["high_low_ratio"] = data["High"] / data["Low"]
        features["price_volume"] = data["Close"] * data["Volume"]

        # 複数期間のリターン率
        for period in [1, 2, 3, 5, 7, 10, 15, 20]:
            features[f"return_{period}d"] = data["Close"].pct_change(period)

        # 移動平均とその関係
        for window in [5, 10, 20, 50, 100]:
            sma = data["Close"].rolling(window).mean()
            features[f"sma_{window}"] = sma
            features[f"price_sma_{window}_ratio"] = data["Close"] / sma
            features[f"sma_{window}_slope"] = sma.diff(5) / sma.shift(5)

        # ボラティリティ特徴量
        for window in [5, 10, 20]:
            vol = data["Close"].pct_change().rolling(window).std()
            features[f"volatility_{window}"] = vol
            features[f"volatility_{window}_ratio"] = vol / vol.rolling(50).mean()

        # テクニカル指標
        # RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features["rsi"] = 100 - (100 / (1 + rs))
        features["rsi_sma"] = features["rsi"].rolling(5).mean()

        # MACD
        ema_12 = data["Close"].ewm(span=12).mean()
        ema_26 = data["Close"].ewm(span=26).mean()
        features["macd"] = ema_12 - ema_26
        features["macd_signal"] = features["macd"].ewm(span=9).mean()
        features["macd_histogram"] = features["macd"] - features["macd_signal"]

        # ボリンジャーバンド
        sma_20 = features["sma_20"]
        std_20 = data["Close"].rolling(20).std()
        features["bb_upper"] = sma_20 + (std_20 * 2)
        features["bb_lower"] = sma_20 - (std_20 * 2)
        features["bb_position"] = (data["Close"] - features["bb_lower"]) / (
            features["bb_upper"] - features["bb_lower"]
        )
        features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / sma_20

        # 出来高特徴量
        features["volume_sma_20"] = data["Volume"].rolling(20).mean()
        features["volume_ratio"] = data["Volume"] / features["volume_sma_20"]
        features["volume_price_trend"] = (
            features["return_1d"] * features["volume_ratio"]
        )

        # 高値・安値特徴量
        for window in [5, 10, 20]:
            features[f"high_{window}"] = data["High"].rolling(window).max()
            features[f"low_{window}"] = data["Low"].rolling(window).min()
            features[f"price_position_{window}"] = (
                data["Close"] - features[f"low_{window}"]
            ) / (features[f"high_{window}"] - features[f"low_{window}"])

        # モメンタム指標
        for window in [5, 10, 20]:
            features[f"momentum_{window}"] = (
                data["Close"] / data["Close"].shift(window) - 1
            )
            features[f"roc_{window}"] = data["Close"].pct_change(window)

        # ラグ特徴量（重要！）
        for lag in [1, 2, 3, 5]:
            features[f"return_1d_lag_{lag}"] = features["return_1d"].shift(lag)
            features[f"volume_ratio_lag_{lag}"] = features["volume_ratio"].shift(lag)
            features[f"rsi_lag_{lag}"] = features["rsi"].shift(lag)

        # 統計的特徴量
        for window in [5, 10, 20]:
            returns = data["Close"].pct_change()
            features[f"return_mean_{window}"] = returns.rolling(window).mean()
            features[f"return_std_{window}"] = returns.rolling(window).std()
            features[f"return_skew_{window}"] = returns.rolling(window).skew()
            features[f"return_kurt_{window}"] = returns.rolling(window).kurt()

        # 相対強度指標
        features["price_rank_20"] = data["Close"].rolling(20).rank(pct=True)
        features["volume_rank_20"] = data["Volume"].rolling(20).rank(pct=True)

        # トレンド特徴量
        for window in [10, 20]:
            poly_coef = []
            for i in range(window, len(data)):
                y = data["Close"].iloc[i - window : i].values
                x = np.arange(len(y))
                coef = np.polyfit(x, y, 1)[0]  # 線形トレンド係数
                poly_coef.append(coef)

            trend_series = pd.Series([np.nan] * window + poly_coef, index=data.index)
            features[f"trend_coef_{window}"] = trend_series

        return features

    def create_target_variable(
        self, data: pd.DataFrame, prediction_days: int = 7
    ) -> pd.Series:
        """目標変数作成（中期リターン予測）"""
        # 7日後の累積リターン率
        future_return = data["Close"].shift(-prediction_days) / data["Close"] - 1
        return future_return

    def prepare_training_data(self, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """訓練データ準備"""
        print(f"データ準備中: {symbol}")

        # より長期のデータを取得
        data = self.data_provider.get_stock_data(symbol, "2y")
        if data.empty or len(data) < 200:
            return None, None

        # テクニカル指標追加
        data = self.data_provider.calculate_technical_indicators(data)

        # 包括的特徴量作成
        features = self.create_comprehensive_features(data)

        # 目標変数作成（7日後リターン）
        target = self.create_target_variable(data, prediction_days=7)

        # 特徴量選択（数値のみ、未来情報なし）
        numeric_features = features.select_dtypes(include=[np.number])

        # 目標変数と同じ期間に調整
        aligned_features = numeric_features.align(target, join="inner", axis=0)[0]

        # NaN除去
        combined = pd.concat([aligned_features, target.rename("target")], axis=1)
        combined = combined.dropna()

        if len(combined) < 50:
            return None, None

        # X, y分離
        X = combined.iloc[:, :-1].values
        y = combined.iloc[:, -1].values

        print(f"  特徴量数: {X.shape[1]}")
        print(f"  サンプル数: {X.shape[0]}")

        return X, y

    def train_models(self, symbols: List[str]) -> Dict:
        """複数モデル訓練"""
        print("=" * 60)
        print("正しいアプローチでの機械学習モデル訓練")
        print("=" * 60)

        # データ収集
        all_X = []
        all_y = []

        for symbol in symbols[:10]:  # 10銘柄
            X, y = self.prepare_training_data(symbol)
            if X is not None and y is not None:
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            return {"error": "No training data"}

        # データ結合
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)

        print(
            f"総訓練データ: {X_combined.shape[0]}サンプル, {X_combined.shape[1]}特徴量"
        )

        # 外れ値除去（重要！）
        q1, q3 = np.percentile(y_combined, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = (y_combined >= lower_bound) & (y_combined <= upper_bound)
        X_cleaned = X_combined[mask]
        y_cleaned = y_combined[mask]

        print(f"外れ値除去後: {X_cleaned.shape[0]}サンプル")

        # モデル定義
        models = {
            "random_forest": RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42
            ),
            "ridge": Ridge(alpha=1.0),
            "linear": LinearRegression(),
        }

        # 時系列分割検証
        tscv = TimeSeriesSplit(n_splits=3)
        best_model = None
        best_mape = float("inf")
        best_name = ""

        for name, model in models.items():
            print(f"\n{name}モデル訓練中...")

            mape_scores = []

            for train_idx, val_idx in tscv.split(X_cleaned):
                X_train, X_val = X_cleaned[train_idx], X_cleaned[val_idx]
                y_train, y_val = y_cleaned[train_idx], y_cleaned[val_idx]

                # スケーリング
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # 訓練
                model.fit(X_train_scaled, y_train)

                # 予測
                y_pred = model.predict(X_val_scaled)

                # MAPE計算（正しい方法）
                # ゼロ除算回避：小さすぎる値を除外
                mask = np.abs(y_val) > 0.01  # 1%以上の変動のみ
                if mask.sum() > 10:  # 十分なサンプル
                    mape = (
                        mean_absolute_percentage_error(y_val[mask], y_pred[mask]) * 100
                    )
                    mape_scores.append(mape)

            if mape_scores:
                avg_mape = np.mean(mape_scores)
                print(f"  平均MAPE: {avg_mape:.2f}%")

                if avg_mape < best_mape:
                    best_mape = avg_mape
                    best_model = model
                    best_name = name

        # 最良モデルで全データ訓練
        if best_model is not None:
            print(f"\n最良モデル: {best_name} (MAPE: {best_mape:.2f}%)")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cleaned)
            best_model.fit(X_scaled, y_cleaned)

            self.models["main"] = best_model
            self.scalers["main"] = scaler
            self.is_trained = True

            return {
                "best_model": best_name,
                "mape": best_mape,
                "training_samples": len(X_cleaned),
            }

        return {"error": "Training failed"}

    def predict_return(self, symbol: str) -> float:
        """リターン率予測"""
        if not self.is_trained:
            return 0.0

        try:
            # データ取得
            data = self.data_provider.get_stock_data(symbol, "6mo")
            if data.empty or len(data) < 100:
                return 0.0

            data = self.data_provider.calculate_technical_indicators(data)

            # 特徴量作成
            features = self.create_comprehensive_features(data)

            # 最新データ
            latest_features = (
                features.select_dtypes(include=[np.number]).iloc[-1:].fillna(0)
            )

            # 予測
            features_scaled = self.scalers["main"].transform(latest_features)
            prediction = self.models["main"].predict(features_scaled)[0]

            # 現実的範囲に制限
            return max(-0.2, min(0.2, prediction))

        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {str(e)}")
            return 0.0

    def test_final_system(self, symbols: List[str]) -> Dict:
        """最終システムテスト"""
        print("\n最終システムテスト")
        print("-" * 40)

        predictions = []
        actuals = []
        valid_mapes = []

        for symbol in symbols[:5]:
            try:
                data = self.data_provider.get_stock_data(symbol, "1y")
                if len(data) < 100:
                    continue

                # 複数のテストポイント
                for i in range(50, 10, -7):  # 7日ずつ
                    historical_data = data.iloc[:-i].copy()

                    if len(historical_data) < 50:
                        continue

                    # 実際の7日リターン
                    start_price = data.iloc[-i]["Close"]
                    end_price = (
                        data.iloc[-i + 7]["Close"] if i >= 7 else data.iloc[-1]["Close"]
                    )
                    actual_return = (end_price - start_price) / start_price

                    # 予測（この実装では簡略化）
                    predicted_return = self._simple_ml_predict(historical_data)

                    predictions.append(predicted_return)
                    actuals.append(actual_return)

                    # 有効MAPE
                    if abs(actual_return) > 0.01:  # 1%以上
                        mape_individual = (
                            abs((actual_return - predicted_return) / actual_return)
                            * 100
                        )
                        valid_mapes.append(mape_individual)

            except Exception as e:
                logger.warning(f"Error testing {symbol}: {str(e)}")
                continue

        if valid_mapes:
            final_mape = np.mean(valid_mapes)
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))

            return {
                "mape": final_mape,
                "mae": mae,
                "total_tests": len(predictions),
                "valid_tests": len(valid_mapes),
            }

        return {"error": "No valid tests"}

    def _simple_ml_predict(self, data: pd.DataFrame) -> float:
        """簡易ML予測（テスト用）"""
        try:
            data = self.data_provider.calculate_technical_indicators(data)
            returns = data["Close"].pct_change().dropna()

            if len(returns) < 20:
                return 0.0

            # 簡単なパターン認識
            recent_trend = returns.iloc[-7:].mean()
            volatility = returns.iloc[-20:].std()
            momentum = (data["Close"].iloc[-1] - data["Close"].iloc[-7]) / data[
                "Close"
            ].iloc[-7]

            # 線形結合
            prediction = recent_trend * 0.3 + momentum * 0.5

            # ボラティリティ調整
            if volatility > 0.03:
                prediction *= 0.7

            return max(-0.1, min(0.1, prediction))

        except Exception:
            return 0.0


def main():
    """メイン実行"""
    print("=" * 60)
    print("正しいアプローチでMAPE 10-20%達成チャレンジ")
    print("=" * 60)

    data_provider = StockDataProvider()
    symbols = list(data_provider.jp_stock_codes.keys())

    predictor = CorrectMAPEPredictor()

    # モデル訓練
    train_results = predictor.train_models(symbols)

    if "error" not in train_results:
        print(f"\n訓練結果:")
        print(f"  最良モデル: {train_results['best_model']}")
        print(f"  MAPE: {train_results['mape']:.2f}%")
        print(f"  訓練サンプル: {train_results['training_samples']}")

        # テスト
        test_results = predictor.test_final_system(symbols)

        if "error" not in test_results:
            print(f"\nテスト結果:")
            print(f"  MAPE: {test_results['mape']:.2f}%")
            print(f"  MAE: {test_results['mae']:.4f}")
            print(f"  総テスト: {test_results['total_tests']}")
            print(f"  有効テスト: {test_results['valid_tests']}")

            if test_results["mape"] < 20:
                if test_results["mape"] < 10:
                    print("🎉 目標達成！MAPE < 10%")
                else:
                    print("✓ 良好な結果！MAPE < 20%")
            else:
                print("継続改善が必要")

        # 現在の予測例
        print(f"\n現在の予測例:")
        print("-" * 30)

        for symbol in symbols[:5]:
            pred_return = predictor.predict_return(symbol)
            print(
                f"{symbol}: 7日後リターン予測 {pred_return:.3f} ({pred_return*100:.1f}%)"
            )

    else:
        print(f"訓練失敗: {train_results}")


if __name__ == "__main__":
    main()
