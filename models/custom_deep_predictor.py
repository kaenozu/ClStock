"""独自の深層学習モデル（CNN + MLP）"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class _LinearRegressor:
    """単純な線形モデルで畳み込み・全結合層を代替する軽量実装。"""

    input_dim: int
    weights: Optional[np.ndarray] = None
    bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Iterable[float]]:
        features = np.asarray(X, dtype=float)
        targets = np.asarray(y, dtype=float).reshape(-1, 1)

        design = np.hstack([features, np.ones((features.shape[0], 1))])
        ridge_lambda = 1e-3
        identity = np.eye(design.shape[1])
        solution = np.linalg.solve(
            design.T @ design + ridge_lambda * identity,
            design.T @ targets,
        )

        self.weights = solution[:-1]
        self.bias = float(solution[-1])

        predictions = self.predict(features)
        loss = np.mean((predictions - targets.ravel()) ** 2)
        return {"loss": [float(loss)]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if self.weights is None:
            return np.zeros(features.shape[0])
        return (features @ self.weights).ravel() + self.bias

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        predictions = self.predict(X)
        targets = np.asarray(y, dtype=float).ravel()
        mse = float(np.mean((predictions - targets) ** 2))
        mae = float(np.mean(np.abs(predictions - targets)))
        denom = np.where(targets != 0, np.abs(targets), 1.0)
        mape = float(np.mean(np.abs((predictions - targets) / denom)))
        return mse, mae, mape


class _SimpleSequentialModel:
    """Keras `Sequential` の最小互換スタブ。"""

    def __init__(self, input_dim: int):
        self._regressor = _LinearRegressor(input_dim=input_dim)
        self.metrics_names = ["loss", "mae", "mape"]

    def compile(self, *_, **__):  # pragma: no cover - 非機能スタブ
        return None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 1,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> Any:
        del epochs, batch_size, verbose
        history = self._regressor.fit(X, y)
        if validation_data is not None:
            X_val, y_val = validation_data
            val_loss = np.mean(
                (
                    self._regressor.predict(X_val)
                    - np.asarray(y_val, dtype=float).ravel()
                )
                ** 2,
            )
            history.setdefault("val_loss", []).append(float(val_loss))
        return type("History", (), {"history": history})()

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self._regressor.predict(X)
        return predictions.reshape(-1, 1)

    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: int = 0) -> Tuple[float, float, float]:
        del verbose
        return self._regressor.evaluate(X, y)


class CustomDeepPredictor:
    """CNN + MLP 構成を模した軽量な深層学習モデル。"""

    def __init__(self, input_shape, learning_rate=0.001):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.model = _SimpleSequentialModel(np.prod(input_shape))
        self.model.compile(optimizer=None, loss="mse", metrics=["mae", "mape"])
        logger.info("独自深層学習モデル構築完了 (スタブ実装)")

    def preprocess_data(self, X):
        """データ前処理"""
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return X

    def _flatten_features(self, X: np.ndarray) -> np.ndarray:
        processed = self.preprocess_data(X)
        return processed.reshape(processed.shape[0], -1)

    def train(
        self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32,
    ):
        """モデル訓練"""
        X_train_flat = self._flatten_features(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_flat)

        if X_val is not None:
            X_val_flat = self._flatten_features(X_val)
            X_val_scaled = self.scaler.transform(X_val_flat)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None

        history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

        logger.info("独自深層学習モデル訓練完了 (スタブ実装)")
        return history

    def predict(self, X):
        """予測実行"""
        X_flat = self._flatten_features(X)
        X_scaled = self.scaler.transform(X_flat)
        predictions = self.model.predict(X_scaled)
        return predictions.flatten()

    def evaluate(self, X_test, y_test):
        """モデル評価"""
        X_test_flat = self._flatten_features(X_test)
        X_test_scaled = self.scaler.transform(X_test_flat)
        results = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        return dict(zip(self.model.metrics_names, results))
