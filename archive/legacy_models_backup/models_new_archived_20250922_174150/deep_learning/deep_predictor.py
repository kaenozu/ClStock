#!/usr/bin/env python3
"""深層学習予測器モジュール
LSTM/Transformer技術による高精度時系列予測
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from data.stock_data import StockDataProvider
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DeepLearningPredictor:
    """LSTM/Transformer深層学習予測器"""

    def __init__(self, model_type="lstm", data_provider=None):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = 60
        self.is_trained = False
        self.feature_columns = []
        self.data_provider = data_provider or StockDataProvider()
        self.model_path = Path("models/saved_models")
        self.model_path.mkdir(exist_ok=True)

    def prepare_sequences(
        self,
        data: pd.DataFrame,
        target_col: str = "Close",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """時系列データをシーケンスに変換"""
        # 特徴量とターゲット分離
        feature_data = data.drop([target_col], axis=1).values
        target_data = data[target_col].values

        # 正規化
        feature_data = self.scaler.fit_transform(feature_data)

        X, y = [], []
        for i in range(self.sequence_length, len(feature_data)):
            X.append(feature_data[i - self.sequence_length : i])
            y.append(target_data[i])

        return np.array(X), np.array(y)

    def build_lstm_model(self, input_shape):
        """LSTM モデル構築"""
        try:
            import tensorflow as tf
            from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.optimizers import Adam

            model = Sequential(
                [
                    LSTM(128, return_sequences=True, input_shape=input_shape),
                    Dropout(0.3),
                    BatchNormalization(),
                    LSTM(128, return_sequences=True),
                    Dropout(0.3),
                    BatchNormalization(),
                    LSTM(64, return_sequences=False),
                    Dropout(0.3),
                    BatchNormalization(),
                    Dense(32, activation="relu"),
                    Dropout(0.2),
                    Dense(1, activation="linear"),
                ],
            )

            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="mse",
                metrics=["mae"],
            )
            return model

        except ImportError:
            logger.error("TensorFlow not available. Using fallback model.")
            return self._build_fallback_model(input_shape)

    def build_transformer_model(self, input_shape):
        """Transformer モデル構築"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers

            def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
                # Multi-head self-attention
                x = layers.MultiHeadAttention(
                    key_dim=head_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )(inputs, inputs)
                x = layers.Dropout(dropout)(x)
                x = layers.LayerNormalization(epsilon=1e-6)(x)
                res = x + inputs

                # Feed forward network
                x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
                x = layers.Dropout(dropout)(x)
                x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
                x = layers.LayerNormalization(epsilon=1e-6)(x)
                return x + res

            inputs = tf.keras.Input(shape=input_shape)
            x = inputs

            # Multi-layer transformer
            for _ in range(3):
                x = transformer_encoder(
                    x,
                    head_size=64,
                    num_heads=4,
                    ff_dim=128,
                    dropout=0.3,
                )

            x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(32, activation="relu")(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(1, activation="linear")(x)

            model = tf.keras.Model(inputs, outputs)
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            return model

        except ImportError:
            logger.error("TensorFlow not available. Using fallback model.")
            return self._build_fallback_model(input_shape)

    def _build_fallback_model(self, input_shape):
        """TensorFlow不可時のフォールバックモデル"""
        from sklearn.ensemble import RandomForestRegressor

        # シーケンス次元を平坦化するためのシンプルモデル
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

    def train_deep_model(self, symbols: List[str]):
        """深層学習モデル訓練"""
        # データ準備
        all_data = []

        from models.ml_stock_predictor import MLStockPredictor

        ml_predictor = MLStockPredictor()

        for symbol in symbols:
            data = self.data_provider.get_stock_data(symbol, "3y")
            if len(data) < 200:
                continue

            features = ml_predictor.prepare_features(data)
            # Closeカラムを追加
            features["Close"] = data["Close"]
            all_data.append(features.dropna())

        if not all_data:
            raise ValueError("No sufficient data for training")

        # 全データ結合
        combined_data = pd.concat(all_data, ignore_index=True)

        # シーケンス作成
        X, y = self.prepare_sequences(combined_data)

        # 訓練/テスト分割
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # モデル構築
        input_shape = (X_train.shape[1], X_train.shape[2])

        if self.model_type == "lstm":
            self.model = self.build_lstm_model(input_shape)
        else:  # transformer
            self.model = self.build_transformer_model(input_shape)

        # TensorFlowモデルの場合
        if hasattr(self.model, "fit") and hasattr(self.model, "predict"):
            try:
                # TensorFlowの場合
                from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

                callbacks = [
                    EarlyStopping(patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(patience=7, factor=0.5),
                ]

                history = self.model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_test, y_test),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=1,
                )

                # 評価
                test_pred = self.model.predict(X_test)
                test_mse = mean_squared_error(y_test, test_pred)

            except Exception as e:
                # フォールバック: sklearnモデルで訓練
                logger.warning(f"TensorFlow training failed: {e}. Using fallback.")
                # シーケンスを2D配列に変換
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)

                self.model.fit(X_train_flat, y_train)
                test_pred = self.model.predict(X_test_flat)
                test_mse = mean_squared_error(y_test, test_pred)
                history = None

        else:
            # sklearn フォールバック
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)

            self.model.fit(X_train_flat, y_train)
            test_pred = self.model.predict(X_test_flat)
            test_mse = mean_squared_error(y_test, test_pred)
            history = None

        self.is_trained = True
        self.save_deep_model()

        logger.info(f"Deep Learning {self.model_type} Test MSE: {test_mse:.4f}")
        return history

    def predict_deep(self, symbol: str) -> float:
        """深層学習予測"""
        if not self.is_trained:
            if not self.load_deep_model():
                return 50.0

        try:
            from models.ml_stock_predictor import MLStockPredictor

            ml_predictor = MLStockPredictor()

            data = self.data_provider.get_stock_data(symbol, "1y")
            features = ml_predictor.prepare_features(data)
            features["Close"] = data["Close"]

            if len(features) < self.sequence_length:
                return 50.0

            # 最新シーケンス準備
            recent_data = (
                features.tail(self.sequence_length).drop(["Close"], axis=1).values
            )
            recent_data = self.scaler.transform(recent_data)

            # 予測実行
            if hasattr(self.model, "predict") and recent_data.ndim == 2:
                # TensorFlowモデル
                sequence = recent_data.reshape(1, self.sequence_length, -1)
                pred = self.model.predict(sequence)[0][0]
            else:
                # sklearn フォールバック
                sequence_flat = recent_data.flatten().reshape(1, -1)
                pred = self.model.predict(sequence_flat)[0]

            # スコア変換 (価格予測→0-100スコア)
            current_price = data["Close"].iloc[-1]
            score = 50 + (pred - current_price) / current_price * 100
            return max(0, min(100, score))

        except Exception as e:
            logger.error(f"Deep learning prediction error for {symbol}: {e!s}")
            return 50.0

    def save_deep_model(self):
        """深層学習モデル保存"""
        try:
            model_file = self.model_path / f"deep_{self.model_type}_model"
            scaler_file = self.model_path / f"deep_{self.model_type}_scaler.joblib"

            # TensorFlowモデルの場合
            if hasattr(self.model, "save"):
                self.model.save(f"{model_file}.h5")
            else:
                # sklearn モデル
                joblib.dump(self.model, f"{model_file}.joblib")

            # スケーラー保存
            joblib.dump(self.scaler, scaler_file)

            logger.info(f"Deep {self.model_type} model saved")

        except Exception as e:
            logger.error(f"Error saving deep model: {e!s}")

    def load_deep_model(self) -> bool:
        """深層学習モデル読み込み"""
        try:
            model_file_h5 = self.model_path / f"deep_{self.model_type}_model.h5"
            model_file_joblib = self.model_path / f"deep_{self.model_type}_model.joblib"
            scaler_file = self.model_path / f"deep_{self.model_type}_scaler.joblib"

            # スケーラー読み込み
            if scaler_file.exists():
                self.scaler = joblib.load(scaler_file)
            else:
                return False

            # モデル読み込み
            if model_file_h5.exists():
                # TensorFlowモデル
                try:
                    import tensorflow as tf

                    self.model = tf.keras.models.load_model(model_file_h5)
                    self.is_trained = True
                    return True
                except ImportError:
                    logger.warning("TensorFlow not available for model loading")

            if model_file_joblib.exists():
                # sklearn モデル
                self.model = joblib.load(model_file_joblib)
                self.is_trained = True
                return True

            return False

        except Exception as e:
            logger.error(f"Error loading deep model: {e!s}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        return {
            "name": "DeepLearningPredictor",
            "version": "1.0.0",
            "model_type": self.model_type,
            "sequence_length": self.sequence_length,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_columns),
        }
