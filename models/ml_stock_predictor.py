"""Machine learning based stock predictor utilities."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from data.stock_data import StockDataProvider

try:  # pragma: no cover - optional dependency for enhanced parallelism
    from models_refactored.ensemble.parallel_feature_calculator import (
        ParallelFeatureCalculator,
    )
except Exception:  # pragma: no cover - calculator not available in all environments
    ParallelFeatureCalculator = None

logger = logging.getLogger(__name__)

class MLStockPredictor:
    """機械学習を使用した株価予測モデル"""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.data_provider = StockDataProvider()
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.feature_names = []
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
            bb_range != 0, 1
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
        features["gap_up"] = ((data["Open"] > prev_close * 1.02)).astype(int)
        features["gap_down"] = ((data["Open"] < prev_close * 0.98)).astype(int)
        features["gap_size"] = (data["Open"] - prev_close) / prev_close
        # 欠損値処理
        features = features.ffill().fillna(0)
        return features

    def create_targets(
        self, data: pd.DataFrame, prediction_days: int = 5
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
                int
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

    def _process_symbol(
        self, symbol: str
    ) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Retrieve data, build features and targets for a single symbol."""
        try:
            logger.info(f"Processing {symbol}...")
            data = self.data_provider.get_stock_data(symbol, "3y")
            if data.empty or len(data) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                return symbol, None, None, None
            features = self.prepare_features(data)
            targets_reg, targets_cls = self.create_targets(data)
            features[f"symbol_{symbol}"] = 1
            return symbol, features, targets_reg, targets_cls
        except Exception as exc:  # pragma: no cover - safety net for unexpected errors
            logger.error(f"Error processing {symbol}: {str(exc)}")
            return symbol, None, None, None

    def _aggregate_results(
        self,
        symbols: List[str],
        results: Dict[str, Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        all_features: List[pd.DataFrame] = []
        all_targets_reg: List[pd.DataFrame] = []
        all_targets_cls: List[pd.DataFrame] = []

        for symbol in symbols:
            features, targets_reg, targets_cls = results.get(symbol, (None, None, None))
            if features is None or targets_reg is None or targets_cls is None:
                continue
            all_features.append(features)
            all_targets_reg.append(targets_reg)
            all_targets_cls.append(targets_cls)

        if not all_features:
            raise ValueError("No valid data available")

        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets_reg = pd.concat(all_targets_reg, ignore_index=True)
        combined_targets_cls = pd.concat(all_targets_cls, ignore_index=True)
        for symbol in symbols:
            col_name = f"symbol_{symbol}"
            if col_name not in combined_features.columns:
                combined_features[col_name] = 0

        combined_features = combined_features.fillna(0)
        return combined_features, combined_targets_reg, combined_targets_cls

    def _prepare_dataset_sequential(
        self, symbols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        results: Dict[
            str,
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]],
        ] = {}
        for symbol in symbols:
            _, features, targets_reg, targets_cls = self._process_symbol(symbol)
            results[symbol] = (features, targets_reg, targets_cls)

        return self._aggregate_results(symbols, results)

    def _prepare_dataset_parallel(
        self, symbols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        results: Dict[
            str,
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]],
        ] = {}

        if ParallelFeatureCalculator is not None:
            try:
                calculator = ParallelFeatureCalculator()
                max_workers = getattr(calculator, "n_jobs", len(symbols) or 1)
                with ThreadPoolExecutor(max_workers=max_workers or 1) as executor:
                    futures = {
                        executor.submit(self._process_symbol, symbol): symbol
                        for symbol in symbols
                    }
                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            _, features, targets_reg, targets_cls = future.result()
                            results[symbol] = (features, targets_reg, targets_cls)
                        except Exception as exc:  # pragma: no cover - executor safety
                            logger.error(f"Parallel processing error for {symbol}: {str(exc)}")
                            results[symbol] = (None, None, None)
            except Exception as exc:  # pragma: no cover - creation failure
                logger.warning(
                    "ParallelFeatureCalculator unavailable, falling back to ThreadPoolExecutor: %s",
                    str(exc),
                )
                results = {}

        if not results:
            with ThreadPoolExecutor(max_workers=len(symbols) or 1) as executor:
                futures = {
                    executor.submit(self._process_symbol, symbol): symbol
                    for symbol in symbols
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        _, features, targets_reg, targets_cls = future.result()
                        results[symbol] = (features, targets_reg, targets_cls)
                    except Exception as exc:  # pragma: no cover - executor safety
                        logger.error(f"Parallel processing error for {symbol}: {str(exc)}")
                        results[symbol] = (None, None, None)

        return self._aggregate_results(symbols, results)

    def prepare_dataset(
        self,
        symbols: List[str],
        start_date: str = "2020-01-01",
        parallel: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """複数銘柄のデータセットを準備"""
        if not parallel:
            return self._prepare_dataset_sequential(symbols)

        try:
            return self._prepare_dataset_parallel(symbols)
        except Exception as exc:
            logger.warning(
                "Parallel dataset preparation failed, falling back to sequential: %s",
                str(exc),
            )
            return self._prepare_dataset_sequential(symbols)

    def train_model(
        self, symbols: List[str], target_column: str = "recommendation_score"
    ):
        """モデルを訓練する"""
        from config.settings import get_settings

        settings = get_settings()
        logger.info("Preparing dataset...")
        features, targets_reg, targets_cls = self.prepare_dataset(symbols)
        # ターゲットが存在するかチェック
        if (
            target_column not in targets_reg.columns
            and target_column not in targets_cls.columns
        ):
            raise ValueError(f"Target column {target_column} not found")
        # ターゲットデータを選択
        if target_column in targets_reg.columns:
            targets = targets_reg[target_column]
            task_type = "regression"
        else:
            targets = targets_cls[target_column]
            task_type = "classification"
        # 欠損値を除去
        valid_indices = ~(targets.isna() | features.isna().any(axis=1))
        features_clean = features[valid_indices]
        targets_clean = targets[valid_indices]
        if len(features_clean) < settings.model.min_training_data:
            raise ValueError(
                f"Insufficient training data: {len(features_clean)} < {settings.model.min_training_data}"
            )
        # 特徴量名を保存
        self.feature_names = features_clean.columns.tolist()
        # 時系列分割で訓練・テスト分割
        train_size = int(len(features_clean) * settings.model.train_test_split)
        X_train = features_clean.iloc[:train_size]
        X_test = features_clean.iloc[train_size:]
        y_train = targets_clean.iloc[:train_size]
        y_test = targets_clean.iloc[train_size:]
        # 特徴量スケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        logger.info(f"Training {self.model_type} model...")
        logger.info(f"Training data shape: {X_train_scaled.shape}")
        # モデル訓練
        if self.model_type == "xgboost":
            if task_type == "regression":
                self.model = xgb.XGBRegressor(
                    n_estimators=settings.model.xgb_n_estimators,
                    max_depth=settings.model.xgb_max_depth,
                    learning_rate=settings.model.xgb_learning_rate,
                    random_state=settings.model.xgb_random_state,
                    n_jobs=-1,
                )
            else:
                self.model = xgb.XGBClassifier(
                    n_estimators=settings.model.xgb_n_estimators,
                    max_depth=settings.model.xgb_max_depth,
                    learning_rate=settings.model.xgb_learning_rate,
                    random_state=settings.model.xgb_random_state,
                    n_jobs=-1,
                )
        elif self.model_type == "lightgbm":
            if task_type == "regression":
                self.model = lgb.LGBMRegressor(
                    n_estimators=settings.model.lgb_n_estimators,
                    max_depth=settings.model.lgb_max_depth,
                    learning_rate=settings.model.lgb_learning_rate,
                    random_state=settings.model.lgb_random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
            else:
                self.model = lgb.LGBMClassifier(
                    n_estimators=settings.model.lgb_n_estimators,
                    max_depth=settings.model.lgb_max_depth,
                    learning_rate=settings.model.lgb_learning_rate,
                    random_state=settings.model.lgb_random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
        # モデル訓練
        self.model.fit(X_train_scaled, y_train)
        # 予測と評価
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        if task_type == "regression":
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            logger.info(f"Training MSE: {train_mse:.4f}")
            logger.info(f"Test MSE: {test_mse:.4f}")
        else:
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            logger.info(f"Training Accuracy: {train_acc:.4f}")
            logger.info(f"Test Accuracy: {test_acc:.4f}")
        self.is_trained = True
        logger.info("Model training completed")
        # モデルを保存
        self.save_model()

    def predict_score(self, symbol: str) -> float:
        """単一銘柄のスコアを予測"""
        if not self.is_trained:
            logger.warning("Model not trained, loading from file...")
            if not self.load_model():
                logger.error("No trained model available")
                return 50.0
        try:
            # データ取得
            data = self.data_provider.get_stock_data(symbol, "1y")
            if data.empty:
                return 0.0
            # 特徴量準備
            features = self.prepare_features(data)
            if features.empty:
                return 0.0
            # 最新データの特徴量を取得
            latest_features = features.iloc[-1:].copy()
            # 銘柄ワンホットエンコーディング
            for feature_name in self.feature_names:
                if feature_name.startswith("symbol_"):
                    latest_features[feature_name] = (
                        1 if feature_name == f"symbol_{symbol}" else 0
                    )
                elif feature_name not in latest_features.columns:
                    latest_features[feature_name] = 0
            # 特徴量順序を合わせる
            latest_features = latest_features.reindex(
                columns=self.feature_names, fill_value=0
            )
            # スケーリング
            features_scaled = self.scaler.transform(latest_features)
            # 予測
            score = self.model.predict(features_scaled)[0]
            # スコアを0-100に正規化
            score = max(0, min(100, float(score)))
            return score
        except Exception as e:
            logger.error(f"Error predicting score for {symbol}: {str(e)}")
            return 50.0

    def predict_return_rate(self, symbol: str, days: int = 5) -> float:
        """リターン率を直接予測（改善されたMAPE対応）"""
        if not self.is_trained:
            logger.warning("Model not trained for return prediction")
            return 0.0
        try:
            # データ取得
            data = self.data_provider.get_stock_data(symbol, "1y")
            if data.empty or len(data) < 50:
                return 0.0
            # 特徴量準備
            features = self.prepare_features_for_return_prediction(data)
            if features.empty:
                return 0.0
            # 最新データの特徴量を取得
            latest_features = features.iloc[-1:].copy()
            # 銘柄ワンホットエンコーディング
            for feature_name in self.feature_names:
                if feature_name.startswith("symbol_"):
                    latest_features[feature_name] = (
                        1 if feature_name == f"symbol_{symbol}" else 0
                    )
                elif feature_name not in latest_features.columns:
                    latest_features[feature_name] = 0
            # 特徴量順序を合わせる
            latest_features = latest_features.reindex(
                columns=self.feature_names, fill_value=0
            )
            # スケーリング
            features_scaled = self.scaler.transform(latest_features)
            # 予測（リターン率）
            predicted_return = self.model.predict(features_scaled)[0]
            # 現実的な範囲に制限（日数に応じて調整）
            max_return = 0.006 * days  # 1日あたり最大0.6%
            predicted_return = max(
                -max_return, min(max_return, float(predicted_return))
            )
            return predicted_return
        except Exception as e:
            logger.error(f"Error predicting return rate for {symbol}: {str(e)}")
            return 0.0

    def prepare_features_for_return_prediction(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """リターン率予測用の特徴量準備"""
        features = self.prepare_features(data)
        # リターン率関連の特徴量を追加
        if len(data) >= 10:
            returns = data["Close"].pct_change()
            # 過去のリターン率統計
            features["return_mean_5d"] = returns.rolling(5).mean()
            features["return_std_5d"] = returns.rolling(5).std()
            features["return_mean_20d"] = returns.rolling(20).mean()
            features["return_std_20d"] = returns.rolling(20).std()
            # シャープレシオ風指標
            features["return_sharpe_5d"] = features["return_mean_5d"] / (
                features["return_std_5d"] + 1e-8
            )
            features["return_sharpe_20d"] = features["return_mean_20d"] / (
                features["return_std_20d"] + 1e-8
            )
            # 連続上昇/下降日数
            returns_sign = np.sign(returns)
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
        return features.fillna(0)

    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度を取得"""
        if not self.is_trained or self.model is None:
            return {}
        try:
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                return dict(zip(self.feature_names, importances))
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
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
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self) -> bool:
        """モデルを読み込み"""
        try:
            model_file = (
                self.model_path / f"ml_stock_predictor_{self.model_type}.joblib"
            )
            scaler_file = self.model_path / f"scaler_{self.model_type}.joblib"
            features_file = self.model_path / f"features_{self.model_type}.joblib"
            if not all(
                [model_file.exists(), scaler_file.exists(), features_file.exists()]
            ):
                return False
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            self.feature_names = joblib.load(features_file)
            self.is_trained = True
            logger.info(f"Model loaded from {model_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

class HyperparameterOptimizer:
    """ハイパーパラメータ自動調整"""

    def __init__(self):
        self.best_params = {}
        self.optimization_history = []

    def optimize_xgboost(self, X, y, cv_folds=5, n_trials=100):
        """XGBoostパラメータ最適化"""
        import optuna

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "random_state": 42,
                "n_jobs": -1,
            }
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(
                model, X, y, cv=cv_folds, scoring="neg_mean_squared_error"
            )
            return -scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        self.best_params["xgboost"] = study.best_params
        logger.info(f"Best XGBoost params: {study.best_params}")
        logger.info(f"Best XGBoost score: {study.best_value}")
        return study.best_params

    def optimize_lightgbm(self, X, y, cv_folds=5, n_trials=100):
        """LightGBMパラメータ最適化"""
        import optuna

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(
                model, X, y, cv=cv_folds, scoring="neg_mean_squared_error"
            )
            return -scores.mean()

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        self.best_params["lightgbm"] = study.best_params
        logger.info(f"Best LightGBM params: {study.best_params}")
        logger.info(f"Best LightGBM score: {study.best_value}")
        return study.best_params

    def save_best_params(self):
        """最適パラメータを保存"""
        try:
            params_file = Path("models/saved_models/best_hyperparams.json")
            with open(params_file, "w") as f:
                import json

                json.dump(self.best_params, f, indent=2)
            logger.info(f"Best parameters saved to {params_file}")
        except Exception as e:
            logger.error(f"Error saving hyperparameters: {str(e)}")

    def load_best_params(self):
        """最適パラメータを読み込み"""
        try:
            params_file = Path("models/saved_models/best_hyperparams.json")
            if params_file.exists():
                with open(params_file, "r") as f:
                    import json

                    self.best_params = json.load(f)
                logger.info("Best parameters loaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading hyperparameters: {str(e)}")
            return False

class ModelPerformanceMonitor:
    """モデル性能監視・評価システム"""

    def __init__(self):
        self.performance_history = []
        self.alerts = []

    def evaluate_model_performance(self, model, X_test, y_test, model_name="Unknown"):
        """詳細なモデル性能評価"""
        predictions = model.predict(X_test)
        # 基本メトリクス
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        # 方向精度（上昇/下降予測の正確性）
        y_direction = (y_test > 50).astype(int)
        pred_direction = (predictions > 50).astype(int)
        direction_accuracy = accuracy_score(y_direction, pred_direction)
        # 分位点ごとの性能
        quantiles = np.quantile(y_test, [0.25, 0.5, 0.75])
        quantile_performance = {}
        for i, q in enumerate([0.25, 0.5, 0.75]):
            mask = (
                y_test <= quantiles[i]
                if i == 0
                else (y_test > quantiles[i - 1]) & (y_test <= quantiles[i])
            )
            if mask.sum() > 0:
                quantile_mse = mean_squared_error(y_test[mask], predictions[mask])
                quantile_performance[f"Q{i+1}"] = quantile_mse
        # 性能記録
        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "direction_accuracy": direction_accuracy,
            "quantile_performance": quantile_performance,
            "sample_size": len(y_test),
        }
        self.performance_history.append(performance_record)
        # アラート判定
        self._check_performance_alerts(performance_record)
        logger.info(f"Model {model_name} Performance:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  Direction Accuracy: {direction_accuracy:.4f}")
        return performance_record

    def _check_performance_alerts(self, performance_record):
        """性能低下アラートをチェック"""
        rmse_threshold = 15.0
        r2_threshold = 0.1
        direction_threshold = 0.55
        alerts = []
        if performance_record["rmse"] > rmse_threshold:
            alerts.append(
                f"High RMSE: {performance_record['rmse']:.4f} > {rmse_threshold}"
            )
        if performance_record["r2_score"] < r2_threshold:
            alerts.append(
                f"Low R²: {performance_record['r2_score']:.4f} < {r2_threshold}"
            )
        if performance_record["direction_accuracy"] < direction_threshold:
            alerts.append(
                f"Low Direction Accuracy: {performance_record['direction_accuracy']:.4f} < {direction_threshold}"
            )
        if alerts:
            alert_record = {
                "timestamp": performance_record["timestamp"],
                "model_name": performance_record["model_name"],
                "alerts": alerts,
            }
            self.alerts.append(alert_record)
            logger.warning(
                f"Performance alerts for {performance_record['model_name']}: {alerts}"
            )

    def get_performance_summary(self, last_n_records=10):
        """性能サマリーを取得"""
        if not self.performance_history:
            return "No performance data available"
        recent_records = self.performance_history[-last_n_records:]
        avg_rmse = np.mean([r["rmse"] for r in recent_records])
        avg_r2 = np.mean([r["r2_score"] for r in recent_records])
        avg_direction = np.mean([r["direction_accuracy"] for r in recent_records])
        summary = f"""
Performance Summary (Last {len(recent_records)} records):
  Average RMSE: {avg_rmse:.4f}
  Average R²: {avg_r2:.4f}
  Average Direction Accuracy: {avg_direction:.4f}
  Total Alerts: {len(self.alerts)}
        """
        return summary

    def save_performance_data(self):
        """性能データを保存"""
        try:
            perf_file = Path("models/saved_models/performance_history.json")
            data = {
                "performance_history": self.performance_history,
                "alerts": self.alerts,
            }
            with open(perf_file, "w") as f:
                import json

                json.dump(data, f, indent=2)
            logger.info(f"Performance data saved to {perf_file}")
        except Exception as e:
            logger.error(f"Error saving performance data: {str(e)}")

    def load_performance_data(self):
        """性能データを読み込み"""
        try:
            perf_file = Path("models/saved_models/performance_history.json")
            if perf_file.exists():
                with open(perf_file, "r") as f:
                    import json

                    data = json.load(f)
                    self.performance_history = data.get("performance_history", [])
                    self.alerts = data.get("alerts", [])
                logger.info("Performance data loaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")
            return False
