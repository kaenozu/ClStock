"""自己教師あり学習による特徴量抽出"""

import logging

from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SelfSupervisedModel:
    """自己教師あり学習モデル（Autoencoderベース）"""

    def __init__(self, input_shape, encoding_dim=64, learning_rate=0.001):
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.autoencoder = None
        self.encoder = None
        self.scaler = StandardScaler()
        self._build_model()

    def _build_model(self):
        """Autoencoderモデル構築"""
        # 入力層
        input_layer = Input(shape=self.input_shape)

        # 符号化（エンコーダー）
        encoded = Dense(128, activation="relu")(input_layer)
        encoded = Dense(64, activation="relu")(encoded)
        encoded = Dense(self.encoding_dim, activation="relu", name="encoded_layer")(
            encoded,
        )

        # 復号化（デコーダー）
        decoded = Dense(64, activation="relu")(encoded)
        decoded = Dense(128, activation="relu")(decoded)
        decoded = Dense(np.prod(self.input_shape), activation="linear")(decoded)
        decoded = Dense(np.prod(self.input_shape), activation="linear")(
            decoded,
        )  # 出力層のサイズ調整

        # Autoencoderモデル
        self.autoencoder = Model(input_layer, decoded)

        # エンコーダーモデル
        self.encoder = Model(
            input_layer, self.autoencoder.get_layer("encoded_layer").output,
        )

        # 編集：出力層の形状を入力層に一致させる
        # 元の入力形状に一致するように出力層を再構築
        output_layer = Dense(np.prod(self.input_shape), activation="linear")(decoded)
        reshaped_output = tf.reshape(output_layer, (-1,) + self.input_shape)

        # Autoencoderモデル（修正版）
        self.autoencoder = Model(input_layer, reshaped_output)

        # エンコーダーモデル（修正版）
        self.encoder = Model(
            input_layer, self.autoencoder.get_layer("encoded_layer").output,
        )

        self.autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        logger.info(f"自己教師あり学習モデル構築完了: {self.autoencoder.summary()}")

    def preprocess_data(self, X):
        """データ前処理"""
        # 次元数を確認し、必要に応じて調整
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return X

    def train(self, X_train, epochs=100, batch_size=32):
        """モデル訓練（再構成タスク）"""
        X_train = self.preprocess_data(X_train)

        # 正規化
        X_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1]),
        ).reshape(X_train.shape)

        # 再構成タスクのため、入力=出力
        history = self.autoencoder.fit(
            X_train_scaled,
            X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        logger.info("自己教師あり学習モデル訓練完了")
        return history

    def extract_features(self, X):
        """特徴量抽出（エンコーダーを使用）"""
        X = self.preprocess_data(X)
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        encoded_features = self.encoder.predict(X_scaled)
        return encoded_features

    def evaluate(self, X_test):
        """モデル評価"""
        X_test = self.preprocess_data(X_test)
        X_test_scaled = self.scaler.transform(
            X_test.reshape(-1, X_test.shape[-1]),
        ).reshape(X_test.shape)
        results = self.autoencoder.evaluate(X_test_scaled, X_test_scaled, verbose=0)
        return dict(zip(self.autoencoder.metrics_names, results))


class TemporalSelfSupervisedModel:
    """時系列データ向け自己教師あり学習モデル（LSTM Autoencoder）"""

    def __init__(self, input_shape, encoding_dim=64, learning_rate=0.001):
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.autoencoder = None
        self.encoder = None
        self.scaler = StandardScaler()
        self._build_model()

    def _build_model(self):
        """LSTM Autoencoderモデル構築"""
        input_layer = Input(shape=self.input_shape)

        # LSTMエンコーダー
        encoded = LSTM(128, return_sequences=True)(input_layer)
        encoded = LSTM(64, return_sequences=True)(encoded)
        encoded = LSTM(self.encoding_dim, return_sequences=False, name="encoded_layer")(
            encoded,
        )

        # LSTMデコーダー
        decoded = Dense(64)(encoded)
        decoded = tf.reshape(decoded, [-1, 1, 64])  # LSTM層に合わせるために形状変更
        decoded = LSTM(64, return_sequences=True)(decoded)
        decoded = LSTM(128, return_sequences=True)(decoded)
        decoded = LSTM(self.input_shape[1], return_sequences=False)(decoded)  # 出力層
        decoded = tf.reshape(decoded, [-1, 1, self.input_shape[1]])  # 元の形状に戻す

        # Autoencoderモデル
        self.autoencoder = Model(input_layer, decoded)

        # エンコーダーモデル
        self.encoder = Model(
            input_layer, self.autoencoder.get_layer("encoded_layer").output,
        )

        self.autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        logger.info(
            f"時系列自己教師あり学習モデル構築完了: {self.autoencoder.summary()}",
        )

    def train(self, X_train, epochs=100, batch_size=32):
        """モデル訓練（再構成タスク）"""
        # 正規化
        X_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1]),
        ).reshape(X_train.shape)

        # 再構成タスクのため、入力=出力
        history = self.autoencoder.fit(
            X_train_scaled,
            X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        logger.info("時系列自己教師あり学習モデル訓練完了")
        return history

    def extract_features(self, X):
        """特徴量抽出（エンコーダーを使用）"""
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        encoded_features = self.encoder.predict(X_scaled)
        return encoded_features


# 修正：TensorFlowのインポートが抜けていたため追加
import tensorflow as tf
