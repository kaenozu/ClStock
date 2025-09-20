#!/usr/bin/env python3
"""
機械学習ベースの改善されたリターン率予測システム
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from typing import Dict, List, Tuple
import pickle
import os

from data.stock_data import StockDataProvider

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMLPredictor:
    """機械学習ベースの改善された予測システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.is_trained = False

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量作成"""
        features = data.copy()

        # 基本的なリターン率
        features["return_1d"] = data["Close"].pct_change()
        features["return_3d"] = data["Close"].pct_change(3)
        features["return_7d"] = data["Close"].pct_change(7)

        # 移動平均系
        for window in [5, 10, 20, 50]:
            features[f"sma_{window}"] = data["Close"].rolling(window).mean()
            features[f"price_to_sma_{window}"] = (
                data["Close"] / features[f"sma_{window}"] - 1
            )

        # ボラティリティ
        for window in [5, 20]:
            features[f"volatility_{window}"] = (
                features["return_1d"].rolling(window).std()
            )

        # ボラティリティ比率
        features["volatility_ratio_5"] = features["volatility_5"] / (
            features["volatility_20"] + 1e-8
        )
        features["volatility_ratio_20"] = features["volatility_20"] / (
            features["volatility_20"].shift(20) + 1e-8
        )

        # RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features["rsi"] = 100 - (100 / (1 + rs))
        features["rsi_normalized"] = (features["rsi"] - 50) / 50

        # MACD
        ema_12 = data["Close"].ewm(span=12).mean()
        ema_26 = data["Close"].ewm(span=26).mean()
        features["macd"] = ema_12 - ema_26
        features["macd_signal"] = features["macd"].ewm(span=9).mean()
        features["macd_hist"] = features["macd"] - features["macd_signal"]

        # ボリューム関連
        features["volume_sma_20"] = data["Volume"].rolling(20).mean()
        features["volume_ratio"] = data["Volume"] / features["volume_sma_20"]
        features["volume_price_trend"] = (
            features["return_1d"] * features["volume_ratio"]
        )

        # トレンド強度
        for window in [5, 10, 20]:
            features[f"trend_strength_{window}"] = (
                data["Close"] - data["Close"].shift(window)
            ) / data["Close"].shift(window)

        # モメンタム
        features["momentum_3_7"] = features["return_3d"] / features["return_7d"]
        features["momentum_acceleration"] = (
            features["return_3d"] - features["return_7d"]
        )

        # サポート・レジスタンス
        for window in [20, 50]:
            features[f"high_{window}"] = data["High"].rolling(window).max()
            features[f"low_{window}"] = data["Low"].rolling(window).min()
            features[f"price_position_{window}"] = (
                data["Close"] - features[f"low_{window}"]
            ) / (features[f"high_{window}"] - features[f"low_{window}"])

        # ボリンジャーバンド
        sma_20 = features["sma_20"]
        std_20 = data["Close"].rolling(20).std()
        features["bb_upper"] = sma_20 + (std_20 * 2)
        features["bb_lower"] = sma_20 - (std_20 * 2)
        features["bb_position"] = (data["Close"] - features["bb_lower"]) / (
            features["bb_upper"] - features["bb_lower"]
        )
        features["bb_squeeze"] = std_20 / sma_20

        # 連続上昇/下降日数
        returns_sign = np.sign(features["return_1d"])
        features["consecutive_up"] = (
            returns_sign.groupby(
                (returns_sign != returns_sign.shift()).cumsum()
            ).cumcount()
            + 1
        ) * (returns_sign > 0)
        features["consecutive_down"] = (
            returns_sign.groupby(
                (returns_sign != returns_sign.shift()).cumsum()
            ).cumcount()
            + 1
        ) * (returns_sign < 0)

        # 時間的特徴量
        features.index = pd.to_datetime(features.index)
        features["day_of_week"] = features.index.dayofweek
        features["month"] = features.index.month
        features["quarter"] = features.index.quarter

        # ラグ特徴量
        for lag in [1, 2, 3, 5]:
            features[f"return_1d_lag_{lag}"] = features["return_1d"].shift(lag)
            features[f"volume_ratio_lag_{lag}"] = features["volume_ratio"].shift(lag)
            features[f"rsi_lag_{lag}"] = features["rsi"].shift(lag)

        return features

    def prepare_training_data(
        self, symbol: str, lookback_days: int = 252
    ) -> Tuple[np.ndarray, np.ndarray]:
        """訓練データの準備"""
        data = self.data_provider.get_stock_data(symbol, "2y")
        if data.empty or len(data) < 100:
            return None, None

        data = self.data_provider.calculate_technical_indicators(data)
        features = self.create_features(data)

        # 目標変数: 5日後のリターン率
        features["target"] = features["return_1d"].shift(-5).rolling(5).sum()

        # 特徴量カラムの選択（数値のみ）
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        feature_cols = [
            col
            for col in numeric_cols
            if col not in ["target", "Close", "Open", "High", "Low", "Volume"]
        ]
        feature_cols = [
            col
            for col in feature_cols
            if not col.startswith("return_1d") or "lag" in col
        ]  # 未来情報除去

        # データクリーニング
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()

        if len(features) < 50:
            return None, None

        X = features[feature_cols].values
        y = features["target"].values

        # 追加の無限大チェック
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

        self.feature_columns = feature_cols
        return X, y

    def train_ensemble_model(self, symbols: List[str]) -> Dict:
        """アンサンブルモデルの訓練"""
        print("アンサンブルモデルの訓練を開始...")

        all_X = []
        all_y = []
        symbol_features = {}

        # 各銘柄のデータを収集
        for symbol in symbols[:10]:  # 最初の10銘柄
            try:
                X, y = self.prepare_training_data(symbol)
                if X is not None and y is not None:
                    all_X.append(X)
                    all_y.append(y)
                    symbol_features[symbol] = (X, y)
                    print(f"{symbol}: {len(X)}件のデータ")
            except Exception as e:
                print(f"{symbol}でエラー: {str(e)}")
                continue

        if not all_X:
            print("訓練データが不足しています")
            return {"error": "No training data"}

        # データを結合
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)

        print(f"総訓練データ: {len(X_combined)}件")

        # 複数モデルの訓練
        models = {
            "rf": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "gb": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "ridge": Ridge(alpha=1.0),
        }

        # 時系列分割による検証
        tscv = TimeSeriesSplit(n_splits=3)
        results = {}

        for name, model in models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X_combined):
                X_train, X_val = X_combined[train_idx], X_combined[val_idx]
                y_train, y_val = y_combined[train_idx], y_combined[val_idx]

                # スケーリング
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # 訓練
                model.fit(X_train_scaled, y_train)

                # 予測
                y_pred = model.predict(X_val_scaled)

                # MAPE計算
                mask = np.abs(y_val) > 0.001
                if mask.sum() > 0:
                    mape = (
                        np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask]))
                        * 100
                    )
                else:
                    mape = float("inf")

                scores.append(mape)

            avg_mape = np.mean(scores)
            results[name] = {"model": model, "mape": avg_mape, "scores": scores}
            print(f"{name}: MAPE = {avg_mape:.2f}%")

        # 最良モデルを選択
        best_model_name = min(results.keys(), key=lambda x: results[x]["mape"])
        best_model = results[best_model_name]["model"]

        print(
            f"最良モデル: {best_model_name} (MAPE: {results[best_model_name]['mape']:.2f}%)"
        )

        # 全データで再訓練
        scaler = StandardScaler()
        X_combined_scaled = scaler.fit_transform(X_combined)
        best_model.fit(X_combined_scaled, y_combined)

        self.models["main"] = best_model
        self.scalers["main"] = scaler
        self.is_trained = True

        return {
            "best_model": best_model_name,
            "mape": results[best_model_name]["mape"],
            "training_samples": len(X_combined),
        }

    def predict_return_rate(self, symbol: str) -> float:
        """リターン率予測"""
        if not self.is_trained:
            return 0.0

        try:
            data = self.data_provider.get_stock_data(symbol, "6mo")
            if data.empty:
                return 0.0

            data = self.data_provider.calculate_technical_indicators(data)
            features = self.create_features(data)

            # 最新データの特徴量
            latest_features = features[self.feature_columns].iloc[-1:].fillna(0)

            # 予測
            features_scaled = self.scalers["main"].transform(latest_features)
            prediction = self.models["main"].predict(features_scaled)[0]

            # 現実的な範囲に制限
            prediction = max(-0.10, min(0.10, prediction))

            return prediction

        except Exception as e:
            logger.error(f"Error predicting return rate for {symbol}: {str(e)}")
            return 0.0

    def save_model(self, filepath: str = "enhanced_ml_model.pkl"):
        """モデル保存"""
        model_data = {
            "models": self.models,
            "scalers": self.scalers,
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"モデルを{filepath}に保存しました")

    def load_model(self, filepath: str = "enhanced_ml_model.pkl") -> bool:
        """モデル読み込み"""
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            self.models = model_data["models"]
            self.scalers = model_data["scalers"]
            self.feature_columns = model_data["feature_columns"]
            self.is_trained = model_data["is_trained"]

            print(f"モデルを{filepath}から読み込みました")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


def main():
    """メイン実行関数"""
    print("=" * 50)
    print("機械学習ベースの改善された予測システム")
    print("=" * 50)

    predictor = EnhancedMLPredictor()
    data_provider = StockDataProvider()

    # モデル訓練
    symbols = list(data_provider.jp_stock_codes.keys())
    results = predictor.train_ensemble_model(symbols)

    if "error" not in results:
        # モデル保存
        predictor.save_model()

        # テスト予測
        print("\n" + "=" * 50)
        print("テスト予測")
        print("=" * 50)

        test_symbols = symbols[:5]
        for symbol in test_symbols:
            predicted_return = predictor.predict_return_rate(symbol)
            print(
                f"{symbol}: 予測リターン率 {predicted_return:.3f} ({predicted_return*100:.1f}%)"
            )


if __name__ == "__main__":
    main()
