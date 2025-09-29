import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

from data.stock_data import StockData
from models.core import ModelCore
from utils.logger import setup_logging

# ロギング設定
setup_logging()
logger = logging.getLogger(__name__)


class EnsemblePredictor(ModelCore):
    """
    複数の機械学習モデルを組み合わせたアンサンブル予測器。
    線形回帰、ランダムフォレスト、勾配ブースティングを統合し、
    それぞれの予測を最終的な線形モデルで結合する。
    """

    def __init__(self, model_name: str = "EnsemblePredictor", period: str = "1y"):
        super().__init__(model_name)
        self.scaler = StandardScaler()
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.meta_model = LinearRegression()  # 各モデルの予測を結合するメタモデル
        self.is_trained = False
        self.logger = logger  # ロガーをインスタンス変数として保持
        self.period = period  # 予測期間を保持

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量エンジニアリングとデータの前処理を行う。
        """
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        # 終値の移動平均
        df["SMA_5"] = df["Close"].rolling(window=5).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()

        # RSI (Relative Strength Index)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        exp1 = df["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = exp1 - exp2
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # ボリンジャーバンド
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        df["BB_Upper"] = df["BB_Middle"] + (df["Close"].rolling(window=20).std() * 2)
        df["BB_Lower"] = df["BB_Middle"] - (df["Close"].rolling(window=20).std() * 2)

        # ターゲット変数の作成（翌日の終値）
        df["Target"] = df["Close"].shift(-1)

        df.dropna(inplace=True)
        return df

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        アンサンブルモデルを訓練する。
        """
        self.logger.info(f"{self.model_name}: Training started.")

        # データのスケーリング
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # 個別モデルの訓練
        self.linear_model.fit(X_scaled_df, y)
        self.rf_model.fit(X_scaled_df, y)
        self.gb_model.fit(X_scaled_df, y)

        # メタモデルのための予測を生成
        linear_preds = self.linear_model.predict(X_scaled_df)
        rf_preds = self.rf_model.predict(X_scaled_df)
        gb_preds = self.gb_model.predict(X_scaled_df)

        meta_features = pd.DataFrame({
            "linear_preds": linear_preds,
            "rf_preds": rf_preds,
            "gb_preds": gb_preds,
        }, index=X.index)

        # メタモデルの訓練
        self.meta_model.fit(meta_features, y)

        self.is_trained = True
        self.logger.info(f"{self.model_name}: Training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        新しいデータポイントに対して予測を行う。
        """
        if not self.is_trained:
            self.logger.error(f"{self.model_name}: Model not trained yet. Please train the model first.")
            raise RuntimeError("Model not trained yet.")

        # データのスケーリング
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        # 個別モデルの予測
        linear_preds = self.linear_model.predict(X_scaled_df)
        rf_preds = self.rf_model.predict(X_scaled_df)
        gb_preds = self.gb_model.predict(X_scaled_df)

        meta_features = pd.DataFrame({
            "linear_preds": linear_preds,
            "rf_preds": rf_preds,
            "gb_preds": gb_preds,
        }, index=X.index)

        # メタモデルによる最終予測
        final_preds = self.meta_model.predict(meta_features)
        return final_preds

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        モデルの性能を評価する。
        """
        self.logger.info(f"{self.model_name}: Evaluation started.")
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        self.logger.info(f"{self.model_name}: Evaluation completed. RMSE: {rmse:.4f}")
        return {"rmse": rmse}

    def save(self, file_path: str) -> None:
        """
        訓練済みモデルを保存する。
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "linear_model": self.linear_model,
                "rf_model": self.rf_model,
                "gb_model": self.gb_model,
                "meta_model": self.meta_model,
                "is_trained": self.is_trained,
                "model_name": self.model_name,
                "period": self.period,
            },
            file_path,
        )
        self.logger.info(f"{self.model_name}: Model saved to {file_path}")

    def load(self, file_path: str) -> None:
        """
        モデルをファイルからロードする。
        """
        if not os.path.exists(file_path):
            self.logger.error(f"{self.model_name}: Model file not found at {file_path}")
            raise FileNotFoundError(f"Model file not found at {file_path}")

        data = joblib.load(file_path)
        self.scaler = data["scaler"]
        self.linear_model = data["linear_model"]
        self.rf_model = data["rf_model"]
        self.gb_model = data["gb_model"]
        self.meta_model = data["meta_model"]
        self.is_trained = data["is_trained"]
        self.model_name = data["model_name"]
        self.period = data.get("period", "1y")  # 互換性のためperiodを追加
        self.logger.info(f"{self.model_name}: Model loaded from {file_path}")

    def get_features_targets(self, df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        """
        データフレームから特徴量とターゲットを抽出する。
        """
        processed_df = self._preprocess_data(df.copy())
        features = processed_df.drop("Target", axis=1)
        targets = processed_df["Target"]
        return features, targets

    def get_latest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        予測のための最新の特徴量セットを抽出する。
        """
        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"])
        df_copy.set_index("Date", inplace=True)
        df_copy.sort_index(inplace=True)

        # 終値の移動平均
        df_copy["SMA_5"] = df_copy["Close"].rolling(window=5).mean()
        df_copy["SMA_20"] = df_copy["Close"].rolling(window=20).mean()

        # RSI (Relative Strength Index)
        delta = df_copy["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_copy["RSI"] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        exp1 = df_copy["Close"].ewm(span=12, adjust=False).mean()
        exp2 = df_copy["Close"].ewm(span=26, adjust=False).mean()
        df_copy["MACD"] = exp1 - exp2
        df_copy["Signal_Line"] = df_copy["MACD"].ewm(span=9, adjust=False).mean()

        # ボリンジャーバンド
        df_copy["BB_Middle"] = df_copy["Close"].rolling(window=20).mean()
        df_copy["BB_Upper"] = df_copy["BB_Middle"] + (df_copy["Close"].rolling(window=20).std() * 2)
        df_copy["BB_Lower"] = df_copy["BB_Middle"] - (df_copy["Close"].rolling(window=20).std() * 2)

        # 最新の行を特徴量として使用
        latest_features = df_copy.iloc[[-1]].drop(columns=["Target"], errors='ignore')
        return latest_features

    def get_prediction_period(self) -> str:
        """
        この予測器が使用する期間を返す。
        """
        return self.period