"""Core ML prediction models."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import xgboost as xgb
from data.stock_data import StockDataProvider
from sklearn.preprocessing import StandardScaler

from .base import EnsemblePredictor, PredictionResult, StockPredictor

logger = logging.getLogger(__name__)


class MLStockPredictor(StockPredictor):
    """機械学習を使用した株価予測モデル"""

    def __init__(self, model_type: str = "xgboost"):
        super().__init__(model_type)
        self.data_provider = StockDataProvider()
        self.scaler = StandardScaler()
        self.model_path = Path("models/saved_models")
        self.model_path.mkdir(exist_ok=True)

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量を準備する"""
        if data.empty:
            return pd.DataFrame()

        # 技術指標を計算
        data = self.data_provider.calculate_technical_indicators(data)
        features = pd.DataFrame(index=data.index)

        # === 基本価格特徴量 ===
        features["price_change"] = data["Close"].pct_change()
        features["volume_change"] = data["Volume"].pct_change()
        features["high_low_ratio"] = (data["High"] - data["Low"]) / data["Close"]
        features["close_open_ratio"] = (data["Close"] - data["Open"]) / data["Open"]

        # === 移動平均関連（強化） ===
        features["sma_20_ratio"] = data["Close"] / data["SMA_20"]
        features["sma_50_ratio"] = data["Close"] / data["SMA_50"]
        features["sma_cross"] = (data["SMA_20"] > data["SMA_50"]).astype(int)
        features["sma_distance"] = (data["SMA_20"] - data["SMA_50"]) / data["Close"]

        # 指数移動平均
        ema_12 = data["Close"].ewm(span=12).mean()
        ema_26 = data["Close"].ewm(span=26).mean()
        features["ema_12_ratio"] = data["Close"] / ema_12
        features["ema_26_ratio"] = data["Close"] / ema_26
        features["ema_cross"] = (ema_12 > ema_26).astype(int)

        # === 高度な技術指標 ===
        features["rsi"] = data["RSI"]
        features["rsi_normalized"] = (data["RSI"] - 50) / 50  # -1 to 1
        features["rsi_overbought"] = (data["RSI"] > 70).astype(int)
        features["rsi_oversold"] = (data["RSI"] < 30).astype(int)

        features["macd"] = data["MACD"]
        features["macd_signal"] = data["MACD_Signal"]
        features["macd_histogram"] = data["MACD"] - data["MACD_Signal"]
        features["macd_bullish"] = (features["macd_histogram"] > 0).astype(int)
        features["atr_ratio"] = data["ATR"] / data["Close"]

        # === ボリンジャーバンド ===
        bb_middle = data["Close"].rolling(20).mean()
        bb_std = data["Close"].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)

        # ゼロ除算を防ぐ安全チェック
        bb_range = bb_upper - bb_lower
        features["bb_position"] = (data["Close"] - bb_lower) / bb_range.where(
            bb_range != 0,
            1,
        )
        features["bb_squeeze"] = bb_range / bb_middle.where(bb_middle != 0, 1)
        features["bb_breakout_up"] = (data["Close"] > bb_upper).astype(int)
        features["bb_breakout_down"] = (data["Close"] < bb_lower).astype(int)

        # === ストキャスティクス ===
        low_14 = data["Low"].rolling(14).min()
        high_14 = data["High"].rolling(14).max()
        features["stoch_k"] = 100 * (data["Close"] - low_14) / (high_14 - low_14)
        features["stoch_d"] = features["stoch_k"].rolling(3).mean()

        # === ウィリアムズ%R ===
        features["williams_r"] = -100 * (high_14 - data["Close"]) / (high_14 - low_14)

        # === CCIとROC ===
        typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: abs(x - x.mean()).mean())
        features["cci"] = (typical_price - sma_tp) / (0.015 * mad)
        features["roc_5"] = (data["Close"] / data["Close"].shift(5) - 1) * 100
        features["roc_10"] = (data["Close"] / data["Close"].shift(10) - 1) * 100

        # === ラグ特徴量（強化） ===
        for lag in [1, 2, 3, 5, 10, 20]:
            features[f"close_lag_{lag}"] = data["Close"].shift(lag)
            features[f"volume_lag_{lag}"] = data["Volume"].shift(lag)
            features[f"rsi_lag_{lag}"] = data["RSI"].shift(lag)
            features[f"price_change_lag_{lag}"] = data["Close"].pct_change().shift(lag)

        # === ローリング統計（拡張） ===
        for window in [3, 5, 10, 20, 50]:
            features[f"close_mean_{window}"] = data["Close"].rolling(window).mean()
            features[f"close_std_{window}"] = data["Close"].rolling(window).std()
            features[f"close_min_{window}"] = data["Close"].rolling(window).min()
            features[f"close_max_{window}"] = data["Close"].rolling(window).max()
            features[f"volume_mean_{window}"] = data["Volume"].rolling(window).mean()
            features[f"volume_std_{window}"] = data["Volume"].rolling(window).std()

        # === ボラティリティ特徴量 ===
        for window in [5, 10, 20, 50]:
            features[f"volatility_{window}d"] = data["Close"].rolling(window).std()
            features[f"volatility_ratio_{window}d"] = (
                features[f"volatility_{window}d"] / data["Close"]
            )

        # === トレンド強度とモメンタム ===
        for window in [5, 10, 20, 50]:
            returns = data["Close"].pct_change()
            features[f"trend_strength_{window}"] = returns.rolling(window).mean()
            features[f"momentum_{window}"] = (
                data["Close"] / data["Close"].shift(window) - 1
            )

        # === リターン特徴量 ===
        for period in [1, 2, 3, 5, 10, 15, 20]:
            features[f"return_{period}d"] = data["Close"].pct_change(period)

        # === 高度な統計特徴量 ===
        for window in [10, 20]:
            returns = data["Close"].pct_change()
            features[f"skewness_{window}"] = returns.rolling(window).skew()
            features[f"kurtosis_{window}"] = returns.rolling(window).kurt()

        # === ボリューム特徴量（強化） ===
        features["volume_price_trend"] = (
            (data["Close"] - data["Close"].shift()) / data["Close"].shift()
        ) * data["Volume"]
        features["on_balance_volume"] = (
            data["Volume"]
            * ((data["Close"] > data["Close"].shift()).astype(int) * 2 - 1)
        ).cumsum()
        features["volume_rate_change"] = data["Volume"].pct_change()

        # ボリューム移動平均
        for window in [5, 10, 20]:
            vol_ma = data["Volume"].rolling(window).mean()
            features[f"volume_ma_ratio_{window}"] = data["Volume"] / vol_ma

        # === 季節性・曜日効果 ===
        features["day_of_week"] = data.index.dayofweek
        features["day_of_month"] = data.index.day
        features["month"] = data.index.month
        features["quarter"] = data.index.quarter

        # === サポート・レジスタンス指標 ===
        for window in [10, 20, 50]:
            rolling_max = data["High"].rolling(window).max()
            rolling_min = data["Low"].rolling(window).min()
            features[f"resistance_distance_{window}"] = (
                rolling_max - data["Close"]
            ) / data["Close"]
            features[f"support_distance_{window}"] = (
                data["Close"] - rolling_min
            ) / data["Close"]

        # === 価格パターン指標 ===
        # ドージ判定
        body_size = abs(data["Close"] - data["Open"])
        candle_range = data["High"] - data["Low"]
        features["doji"] = (body_size / candle_range < 0.1).astype(int)

        # ハンマー・倒立ハンマー
        lower_shadow = data["Close"].combine(data["Open"], min) - data["Low"]
        upper_shadow = data["High"] - data["Close"].combine(data["Open"], max)
        features["hammer"] = (
            (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
        ).astype(int)
        features["hanging_man"] = (
            (upper_shadow > 2 * body_size) & (lower_shadow < body_size)
        ).astype(int)

        # === 相対パフォーマンス ===
        # 市場全体との相関を後で計算するプレースホルダー
        features["market_relative_strength"] = 0  # 後で実装

        # === ギャップ検出 ===
        prev_close = data["Close"].shift(1)
        features["gap_up"] = (data["Open"] > prev_close * 1.02).astype(int)
        features["gap_down"] = (data["Open"] < prev_close * 0.98).astype(int)
        features["gap_size"] = (data["Open"] - prev_close) / prev_close

        # 欠損値処理
        features = features.ffill().fillna(0)
        return features

    def create_targets(
        self,
        data: pd.DataFrame,
        prediction_days: int = 5,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """予測ターゲットを作成（分類と回帰の両方）"""
        targets_regression = pd.DataFrame(index=data.index)
        targets_classification = pd.DataFrame(index=data.index)

        # 回帰ターゲット: 将来の価格変化率
        for days in [1, 3, 5, 10]:
            future_return = data["Close"].shift(-days) / data["Close"] - 1
            targets_regression[f"return_{days}d"] = future_return

        # 分類ターゲット: 価格上昇/下降
        for days in [1, 3, 5, 10]:
            future_return = data["Close"].shift(-days) / data["Close"] - 1
            targets_classification[f"direction_{days}d"] = (future_return > 0).astype(
                int,
            )

        # 推奨スコアターゲット（0-100）
        targets_regression["recommendation_score"] = (
            self._calculate_future_performance_score(data)
        )

        return targets_regression, targets_classification

    def _calculate_future_performance_score(self, data: pd.DataFrame) -> pd.Series:
        """将来のパフォーマンスに基づくスコアを計算"""
        scores = pd.Series(index=data.index, dtype=float)

        for i in range(len(data) - 10):
            current_price = data["Close"].iloc[i]
            future_prices = data["Close"].iloc[i + 1 : i + 11]

            if len(future_prices) < 10:
                continue

            # 最大利益
            max_gain = (future_prices.max() - current_price) / current_price
            # 最大損失
            max_loss = (future_prices.min() - current_price) / current_price
            # 最終リターン
            final_return = (future_prices.iloc[-1] - current_price) / current_price

            # スコア計算
            score = 50  # ベーススコア
            score += max_gain * 100  # 最大利益を加算
            score += max_loss * 100  # 最大損失を減算（負の値）
            score += final_return * 50  # 最終リターンの影響

            # ボラティリティペナルティ
            volatility = future_prices.std() / current_price
            score -= volatility * 20

            scores.iloc[i] = max(0, min(100, score))

        return scores.fillna(50)

    def prepare_dataset(
        self,
        symbols: List[str],
        start_date: str = "2020-01-01",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """複数銘柄のデータセットを準備"""
        all_features = []
        all_targets_reg = []
        all_targets_cls = []

        for symbol in symbols:
            try:
                logger.info(f"Processing {symbol}...")
                data = self.data_provider.get_stock_data(symbol, "3y")
                if data.empty or len(data) < 100:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue

                features = self.prepare_features(data)
                targets_reg, targets_cls = self.create_targets(data)

                # 銘柄情報を追加
                features[f"symbol_{symbol}"] = 1

                all_features.append(features)
                all_targets_reg.append(targets_reg)
                all_targets_cls.append(targets_cls)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e!s}")
                continue

        if not all_features:
            raise ValueError("No valid data available")

        # データを結合
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets_reg = pd.concat(all_targets_reg, ignore_index=True)
        combined_targets_cls = pd.concat(all_targets_cls, ignore_index=True)

        # ワンホットエンコーディング用の銘柄列を調整
        symbol_columns = [
            col for col in combined_features.columns if col.startswith("symbol_")
        ]
        for symbol in symbols:
            col_name = f"symbol_{symbol}"
            if col_name not in combined_features.columns:
                combined_features[col_name] = 0

        combined_features = combined_features.fillna(0)
        return combined_features, combined_targets_reg, combined_targets_cls

    def train(self, data: pd.DataFrame, target: pd.Series) -> None:
        """Train the ML model"""
        if data.empty or target.empty:
            raise ValueError("Training data cannot be empty")

        # Prepare features
        features = self.prepare_features(data)

        # Create targets based on the type of target provided
        if target.dtype == "float64":
            # Regression task
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            # Classification task
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train model
        self.model.fit(features_scaled, target)
        self._is_trained = True
        self.feature_names = features.columns.tolist()

    def predict(
        self,
        symbol: str,
        data: Optional[pd.DataFrame] = None,
    ) -> PredictionResult:
        """Predict stock performance"""
        if not self.is_trained():
            raise ValueError("Model must be trained before making predictions")

        if data is None:
            data = self.data_provider.get_stock_data(symbol, "1y")

        if not self.validate_input(data):
            raise ValueError("Invalid input data")

        try:
            # Prepare features
            features = self.prepare_features(data)
            latest_features = features.iloc[-1:].copy()

            # Handle symbol encoding
            for feature_name in self.feature_names:
                if feature_name.startswith("symbol_"):
                    latest_features[feature_name] = (
                        1 if feature_name == f"symbol_{symbol}" else 0
                    )
                elif feature_name not in latest_features.columns:
                    latest_features[feature_name] = 0

            # Reorder features to match training
            latest_features = latest_features.reindex(
                columns=self.feature_names,
                fill_value=0,
            )

            # Scale features
            features_scaled = self.scaler.transform(latest_features)

            # Make prediction
            prediction = self.model.predict(features_scaled)[0]

            # Calculate confidence based on prediction
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(features_scaled)[0]
                confidence = max(proba)
            else:
                confidence = 0.8  # Default confidence for regression

            return PredictionResult(
                prediction=float(prediction),
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    "model_type": self.model_type,
                    "symbol": symbol,
                    "feature_count": len(self.feature_names),
                },
            )

        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e!s}")
            return PredictionResult(
                prediction=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={"error": str(e)},
            )

    def predict_score(self, symbol: str) -> float:
        """単一銘柄のスコアを予測"""
        result = self.predict(symbol)
        return max(0, min(100, result.prediction))

    def predict_return_rate(self, symbol: str, days: int = 5) -> float:
        """リターン率を直接予測（改善されたMAPE対応）"""
        result = self.predict(symbol)
        # Limit to realistic range based on days
        max_return = 0.006 * days  # 1日あたり最大0.6%
        return max(-max_return, min(max_return, result.prediction))

    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度を取得"""
        if not self.is_trained() or self.model is None:
            return {}

        try:
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                return dict(zip(self.feature_names, importances))
        except Exception as e:
            logger.error(f"Error getting feature importance: {e!s}")

        return {}

    def save_model(self):
        """モデルを保存"""
        try:
            model_file = (
                self.model_path / f"ml_stock_predictor_{self.model_type}.joblib"
            )
            scaler_file = self.model_path / f"scaler_{self.model_type}.joblib"
            features_file = self.model_path / f"features_{self.model_type}.joblib"

            joblib.dump(self.model, model_file)
            joblib.dump(self.scaler, scaler_file)
            joblib.dump(self.feature_names, features_file)

            logger.info(f"Model saved to {model_file}")
        except Exception as e:
            logger.error(f"Error saving model: {e!s}")

    def load_model(self) -> bool:
        """モデルを読み込み"""
        try:
            model_file = (
                self.model_path / f"ml_stock_predictor_{self.model_type}.joblib"
            )
            scaler_file = self.model_path / f"scaler_{self.model_type}.joblib"
            features_file = self.model_path / f"features_{self.model_type}.joblib"

            if not all(
                [model_file.exists(), scaler_file.exists(), features_file.exists()],
            ):
                return False

            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            self.feature_names = joblib.load(features_file)
            self._is_trained = True

            logger.info(f"Model loaded from {model_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e!s}")
            return False


class EnsembleStockPredictor(EnsemblePredictor):
    """アンサンブル株価予測モデル"""

    def __init__(self):
        super().__init__("ensemble")
        self.data_provider = StockDataProvider()

    def train(self, data: pd.DataFrame, target: pd.Series) -> None:
        """Train all models in the ensemble"""
        for model in self.models:
            model.train(data, target)
        self._is_trained = True

    def predict(
        self,
        symbol: str,
        data: Optional[pd.DataFrame] = None,
    ) -> PredictionResult:
        """Ensemble prediction"""
        if not self.is_trained():
            raise ValueError("Ensemble must be trained before making predictions")

        if data is None:
            data = self.data_provider.get_stock_data(symbol, "1y")

        predictions = []
        confidences = []

        for model, weight in zip(self.models, self.weights):
            try:
                result = model.predict(symbol, data)
                predictions.append(result.prediction * weight)
                confidences.append(result.confidence * weight)
            except Exception as e:
                logger.warning(f"Model {model.model_type} failed: {e!s}")
                continue

        if not predictions:
            raise ValueError("All models failed to make predictions")

        total_weight = sum(self.weights[: len(predictions)])
        ensemble_prediction = sum(predictions) / total_weight
        ensemble_confidence = sum(confidences) / total_weight

        return PredictionResult(
            prediction=ensemble_prediction,
            confidence=ensemble_confidence,
            timestamp=datetime.now(),
            metadata={
                "model_type": "ensemble",
                "symbol": symbol,
                "models_used": len(predictions),
            },
        )
