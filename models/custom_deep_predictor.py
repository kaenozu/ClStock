"""独自の深層学習モデル（CNN + MLP）"""

import logging

from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Input, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CustomDeepPredictor:
    """CNN + MLPによる独自深層学習モデル"""

    def __init__(self, input_shape, learning_rate=0.001):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self._build_model()

    def _build_model(self):
        """モデル構築"""
        self.model = Sequential(
            [
                Input(shape=self.input_shape),
                # CNN層
                Conv1D(filters=64, kernel_size=3, activation="relu"),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=32, kernel_size=3, activation="relu"),
                MaxPooling1D(pool_size=2),
                Flatten(),
                # MLP層
                Dense(100, activation="relu"),
                Dropout(0.3),
                Dense(50, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="linear"),
            ],
        )

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae", "mape"],
        )

        logger.info(f"独自深層学習モデル構築完了: {self.model.summary()}")

    def preprocess_data(self, X):
        """データ前処理"""
        # 次元数を確認し、必要に応じて調整
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        return X

    def train(
        self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32,
    ):
        """モデル訓練"""
        X_train = self.preprocess_data(X_train)
        if X_val is not None:
            X_val = self.preprocess_data(X_val)

        # 正規化
        X_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1]),
        ).reshape(X_train.shape)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(
                X_val.reshape(-1, X_val.shape[-1]),
            ).reshape(X_val.shape)
        else:
            X_val_scaled = None

        validation_data = (X_val_scaled, y_val) if X_val_scaled is not None else None

        history = self.model.fit(
            X_train_scaled,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        logger.info("独自深層学習モデル訓練完了")
        return history

    def predict(self, X):
        """予測実行"""
        X = self.preprocess_data(X)
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = self.model.predict(X_scaled)
        return predictions.flatten()

    def evaluate(self, X_test, y_test):
        """モデル評価"""
        X_test = self.preprocess_data(X_test)
        X_test_scaled = self.scaler.transform(
            X_test.reshape(-1, X_test.shape[-1]),
        ).reshape(X_test.shape)
        results = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        return dict(zip(self.model.metrics_names, results))
