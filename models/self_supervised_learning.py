"""自己教師あり学習モジュールの軽量 NumPy 実装。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class _History:
    """Keras 互換の履歴オブジェクトを模した単純なスタブ。"""

    def __init__(self, loss: float, val_loss: float | None = None):
        history = {"loss": [loss]}
        if val_loss is not None:
            history["val_loss"] = [val_loss]
        self.history = history


class SelfSupervisedModel:
    """自己教師あり学習モデル（線形オートエンコーダ風）。"""

    def __init__(self, input_shape, encoding_dim=64, learning_rate=0.001):
        del learning_rate  # NumPy 実装では未使用
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.scaler = StandardScaler()
        self.components_: np.ndarray | None = None

        logger.info("自己教師あり学習モデル構築完了 (スタブ実装)")

    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array[:, np.newaxis]
        return array

    def _encode(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            return np.zeros((X.shape[0], self.encoding_dim))
        return X @ self.components_.T

    def _decode(self, encoded: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            return np.zeros((encoded.shape[0], np.prod(self.input_shape)))
        return encoded @ self.components_

    def train(
        self, X_train: np.ndarray, epochs: int = 100, batch_size: int = 32
    ) -> Any:
        del epochs, batch_size
        X_train = self.preprocess_data(X_train)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_scaled = self.scaler.fit_transform(X_train_flat)

        _, _, vh = np.linalg.svd(X_scaled, full_matrices=False)
        k = min(self.encoding_dim, vh.shape[0])
        components = np.zeros((self.encoding_dim, vh.shape[1]))
        components[:k] = vh[:k]
        self.components_ = components

        encoded = self._encode(X_scaled)
        reconstructed = self._decode(encoded)
        loss = float(np.mean((X_scaled - reconstructed) ** 2))

        logger.info("自己教師あり学習モデル訓練完了 (スタブ実装)")
        return _History(loss=loss)

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        X = self.preprocess_data(X)
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        return self._encode(X_scaled)

    def evaluate(self, X_test: np.ndarray) -> dict[str, float]:
        X_test = self.preprocess_data(X_test)
        X_flat = X_test.reshape(X_test.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        encoded = self._encode(X_scaled)
        reconstructed = self._decode(encoded)
        mse = float(np.mean((X_scaled - reconstructed) ** 2))
        mae = float(np.mean(np.abs(X_scaled - reconstructed)))
        return {"loss": mse, "mae": mae}


class TemporalSelfSupervisedModel:
    """時系列データ向け自己教師あり学習モデル (NumPy 版)。"""

    def __init__(self, input_shape, encoding_dim=64, learning_rate=0.001):
        del learning_rate
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.scaler = StandardScaler()
        self.components_: np.ndarray | None = None

        logger.info("時系列自己教師あり学習モデル構築完了 (スタブ実装)")

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim != 3:
            raise ValueError(
                "入力は (samples, timesteps, features) である必要があります。"
            )
        return array.reshape(array.shape[0], -1)

    def _encode(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            return np.zeros((X.shape[0], self.encoding_dim))
        return X @ self.components_.T

    def _decode(self, encoded: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            return np.zeros((encoded.shape[0], np.prod(self.input_shape)))
        return encoded @ self.components_

    def train(
        self, X_train: np.ndarray, epochs: int = 100, batch_size: int = 32
    ) -> Any:
        del epochs, batch_size
        X_flat = self._flatten(X_train)
        X_scaled = self.scaler.fit_transform(X_flat)

        _, _, vh = np.linalg.svd(X_scaled, full_matrices=False)
        k = min(self.encoding_dim, vh.shape[0])
        components = np.zeros((self.encoding_dim, vh.shape[1]))
        components[:k] = vh[:k]
        self.components_ = components

        encoded = self._encode(X_scaled)
        reconstructed = self._decode(encoded)
        loss = float(np.mean((X_scaled - reconstructed) ** 2))

        logger.info("時系列自己教師あり学習モデル訓練完了 (スタブ実装)")
        return _History(loss=loss)

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        X_flat = self._flatten(X)
        X_scaled = self.scaler.transform(X_flat)
        features = self._encode(X_scaled)
        return features
