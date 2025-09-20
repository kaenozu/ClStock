import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import joblib
from pathlib import Path

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, mean_absolute_error, r2_score

from data.stock_data import StockDataProvider

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
        features['price_change'] = data['Close'].pct_change()
        features['volume_change'] = data['Volume'].pct_change()
        features['high_low_ratio'] = (data['High'] - data['Low']) / data['Close']
        features['close_open_ratio'] = (data['Close'] - data['Open']) / data['Open']

        # === 移動平均関連（強化） ===
        features['sma_20_ratio'] = data['Close'] / data['SMA_20']
        features['sma_50_ratio'] = data['Close'] / data['SMA_50']
        features['sma_cross'] = (data['SMA_20'] > data['SMA_50']).astype(int)
        features['sma_distance'] = (data['SMA_20'] - data['SMA_50']) / data['Close']
        
        # 指数移動平均
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        features['ema_12_ratio'] = data['Close'] / ema_12
        features['ema_26_ratio'] = data['Close'] / ema_26
        features['ema_cross'] = (ema_12 > ema_26).astype(int)

        # === 高度な技術指標 ===
        features['rsi'] = data['RSI']
        features['rsi_normalized'] = (data['RSI'] - 50) / 50  # -1 to 1
        features['rsi_overbought'] = (data['RSI'] > 70).astype(int)
        features['rsi_oversold'] = (data['RSI'] < 30).astype(int)
        
        features['macd'] = data['MACD']
        features['macd_signal'] = data['MACD_Signal']
        features['macd_histogram'] = data['MACD'] - data['MACD_Signal']
        features['macd_bullish'] = (features['macd_histogram'] > 0).astype(int)
        
        features['atr_ratio'] = data['ATR'] / data['Close']

        # === ボリンジャーバンド ===
        bb_middle = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        features['bb_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
        features['bb_breakout_up'] = (data['Close'] > bb_upper).astype(int)
        features['bb_breakout_down'] = (data['Close'] < bb_lower).astype(int)

        # === ストキャスティクス ===
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        features['stoch_k'] = 100 * (data['Close'] - low_14) / (high_14 - low_14)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()

        # === ウィリアムズ%R ===
        features['williams_r'] = -100 * (high_14 - data['Close']) / (high_14 - low_14)

        # === CCIとROC ===
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: abs(x - x.mean()).mean())
        features['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        features['roc_5'] = (data['Close'] / data['Close'].shift(5) - 1) * 100
        features['roc_10'] = (data['Close'] / data['Close'].shift(10) - 1) * 100

        # === ラグ特徴量（強化） ===
        for lag in [1, 2, 3, 5, 10, 20]:
            features[f'close_lag_{lag}'] = data['Close'].shift(lag)
            features[f'volume_lag_{lag}'] = data['Volume'].shift(lag)
            features[f'rsi_lag_{lag}'] = data['RSI'].shift(lag)
            features[f'price_change_lag_{lag}'] = data['Close'].pct_change().shift(lag)

        # === ローリング統計（拡張） ===
        for window in [3, 5, 10, 20, 50]:
            features[f'close_mean_{window}'] = data['Close'].rolling(window).mean()
            features[f'close_std_{window}'] = data['Close'].rolling(window).std()
            features[f'close_min_{window}'] = data['Close'].rolling(window).min()
            features[f'close_max_{window}'] = data['Close'].rolling(window).max()
            features[f'volume_mean_{window}'] = data['Volume'].rolling(window).mean()
            features[f'volume_std_{window}'] = data['Volume'].rolling(window).std()

        # === ボラティリティ特徴量 ===
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}d'] = data['Close'].rolling(window).std()
            features[f'volatility_ratio_{window}d'] = features[f'volatility_{window}d'] / data['Close']

        # === トレンド強度とモメンタム ===
        for window in [5, 10, 20, 50]:
            returns = data['Close'].pct_change()
            features[f'trend_strength_{window}'] = returns.rolling(window).mean()
            features[f'momentum_{window}'] = data['Close'] / data['Close'].shift(window) - 1

        # === リターン特徴量 ===
        for period in [1, 2, 3, 5, 10, 15, 20]:
            features[f'return_{period}d'] = data['Close'].pct_change(period)

        # === 高度な統計特徴量 ===
        for window in [10, 20]:
            returns = data['Close'].pct_change()
            features[f'skewness_{window}'] = returns.rolling(window).skew()
            features[f'kurtosis_{window}'] = returns.rolling(window).kurt()

        # === ボリューム特徴量（強化） ===
        features['volume_price_trend'] = ((data['Close'] - data['Close'].shift()) / data['Close'].shift()) * data['Volume']
        features['on_balance_volume'] = (data['Volume'] * ((data['Close'] > data['Close'].shift()).astype(int) * 2 - 1)).cumsum()
        features['volume_rate_change'] = data['Volume'].pct_change()
        
        # ボリューム移動平均
        for window in [5, 10, 20]:
            vol_ma = data['Volume'].rolling(window).mean()
            features[f'volume_ma_ratio_{window}'] = data['Volume'] / vol_ma

        # === 季節性・曜日効果 ===
        features['day_of_week'] = data.index.dayofweek
        features['day_of_month'] = data.index.day
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter

        # === サポート・レジスタンス指標 ===
        for window in [10, 20, 50]:
            rolling_max = data['High'].rolling(window).max()
            rolling_min = data['Low'].rolling(window).min()
            features[f'resistance_distance_{window}'] = (rolling_max - data['Close']) / data['Close']
            features[f'support_distance_{window}'] = (data['Close'] - rolling_min) / data['Close']

        # === 価格パターン指標 ===
        # ドージ判定
        body_size = abs(data['Close'] - data['Open'])
        candle_range = data['High'] - data['Low']
        features['doji'] = (body_size / candle_range < 0.1).astype(int)
        
        # ハンマー・倒立ハンマー
        lower_shadow = data['Close'].combine(data['Open'], min) - data['Low']
        upper_shadow = data['High'] - data['Close'].combine(data['Open'], max)
        features['hammer'] = ((lower_shadow > 2 * body_size) & (upper_shadow < body_size)).astype(int)
        features['hanging_man'] = ((upper_shadow > 2 * body_size) & (lower_shadow < body_size)).astype(int)

        # === 相対パフォーマンス ===
        # 市場全体との相関を後で計算するプレースホルダー
        features['market_relative_strength'] = 0  # 後で実装

        # === ギャップ検出 ===
        prev_close = data['Close'].shift(1)
        features['gap_up'] = ((data['Open'] > prev_close * 1.02)).astype(int)
        features['gap_down'] = ((data['Open'] < prev_close * 0.98)).astype(int)
        features['gap_size'] = (data['Open'] - prev_close) / prev_close

        # 欠損値処理
        features = features.ffill().fillna(0)

        return features

    def create_targets(self, data: pd.DataFrame, prediction_days: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """予測ターゲットを作成（分類と回帰の両方）"""
        targets_regression = pd.DataFrame(index=data.index)
        targets_classification = pd.DataFrame(index=data.index)

        # 回帰ターゲット: 将来の価格変化率
        for days in [1, 3, 5, 10]:
            future_return = data['Close'].shift(-days) / data['Close'] - 1
            targets_regression[f'return_{days}d'] = future_return

        # 分類ターゲット: 価格上昇/下降
        for days in [1, 3, 5, 10]:
            future_return = data['Close'].shift(-days) / data['Close'] - 1
            targets_classification[f'direction_{days}d'] = (future_return > 0).astype(int)

        # 推奨スコアターゲット（0-100）
        targets_regression['recommendation_score'] = self._calculate_future_performance_score(data)

        return targets_regression, targets_classification

    def _calculate_future_performance_score(self, data: pd.DataFrame) -> pd.Series:
        """将来のパフォーマンスに基づくスコアを計算"""
        scores = pd.Series(index=data.index, dtype=float)

        for i in range(len(data) - 10):
            current_price = data['Close'].iloc[i]
            future_prices = data['Close'].iloc[i+1:i+11]

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

    def prepare_dataset(self, symbols: List[str], start_date: str = "2020-01-01") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
                features['symbol_' + symbol] = 1

                all_features.append(features)
                all_targets_reg.append(targets_reg)
                all_targets_cls.append(targets_cls)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue

        if not all_features:
            raise ValueError("No valid data available")

        # データを結合
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets_reg = pd.concat(all_targets_reg, ignore_index=True)
        combined_targets_cls = pd.concat(all_targets_cls, ignore_index=True)

        # ワンホットエンコーディング用の銘柄列を調整
        symbol_columns = [col for col in combined_features.columns if col.startswith('symbol_')]
        for symbol in symbols:
            col_name = f'symbol_{symbol}'
            if col_name not in combined_features.columns:
                combined_features[col_name] = 0

        combined_features = combined_features.fillna(0)

        return combined_features, combined_targets_reg, combined_targets_cls

    def train_model(self, symbols: List[str], target_column: str = 'recommendation_score'):
        """モデルを訓練する"""
        from config.settings import get_settings
        settings = get_settings()
        
        logger.info("Preparing dataset...")
        features, targets_reg, targets_cls = self.prepare_dataset(symbols)

        # ターゲットが存在するかチェック
        if target_column not in targets_reg.columns and target_column not in targets_cls.columns:
            raise ValueError(f"Target column {target_column} not found")

        # ターゲットデータを選択
        if target_column in targets_reg.columns:
            targets = targets_reg[target_column]
            task_type = 'regression'
        else:
            targets = targets_cls[target_column]
            task_type = 'classification'

        # 欠損値を除去
        valid_indices = ~(targets.isna() | features.isna().any(axis=1))
        features_clean = features[valid_indices]
        targets_clean = targets[valid_indices]

        if len(features_clean) < settings.model.min_training_data:
            raise ValueError(f"Insufficient training data: {len(features_clean)} < {settings.model.min_training_data}")

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
            if task_type == 'regression':
                self.model = xgb.XGBRegressor(
                    n_estimators=settings.model.xgb_n_estimators,
                    max_depth=settings.model.xgb_max_depth,
                    learning_rate=settings.model.xgb_learning_rate,
                    random_state=settings.model.xgb_random_state,
                    n_jobs=-1
                )
            else:
                self.model = xgb.XGBClassifier(
                    n_estimators=settings.model.xgb_n_estimators,
                    max_depth=settings.model.xgb_max_depth,
                    learning_rate=settings.model.xgb_learning_rate,
                    random_state=settings.model.xgb_random_state,
                    n_jobs=-1
                )

        elif self.model_type == "lightgbm":
            if task_type == 'regression':
                self.model = lgb.LGBMRegressor(
                    n_estimators=settings.model.lgb_n_estimators,
                    max_depth=settings.model.lgb_max_depth,
                    learning_rate=settings.model.lgb_learning_rate,
                    random_state=settings.model.lgb_random_state,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                self.model = lgb.LGBMClassifier(
                    n_estimators=settings.model.lgb_n_estimators,
                    max_depth=settings.model.lgb_max_depth,
                    learning_rate=settings.model.lgb_learning_rate,
                    random_state=settings.model.lgb_random_state,
                    n_jobs=-1,
                    verbose=-1
                )

        # モデル訓練
        self.model.fit(X_train_scaled, y_train)

        # 予測と評価
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        if task_type == 'regression':
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
                if feature_name.startswith('symbol_'):
                    latest_features[feature_name] = 1 if feature_name == f'symbol_{symbol}' else 0
                elif feature_name not in latest_features.columns:
                    latest_features[feature_name] = 0

            # 特徴量順序を合わせる
            latest_features = latest_features.reindex(columns=self.feature_names, fill_value=0)

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
                if feature_name.startswith('symbol_'):
                    latest_features[feature_name] = 1 if feature_name == f'symbol_{symbol}' else 0
                elif feature_name not in latest_features.columns:
                    latest_features[feature_name] = 0

            # 特徴量順序を合わせる
            latest_features = latest_features.reindex(columns=self.feature_names, fill_value=0)

            # スケーリング
            features_scaled = self.scaler.transform(latest_features)

            # 予測（リターン率）
            predicted_return = self.model.predict(features_scaled)[0]

            # 現実的な範囲に制限（日数に応じて調整）
            max_return = 0.006 * days  # 1日あたり最大0.6%
            predicted_return = max(-max_return, min(max_return, float(predicted_return)))

            return predicted_return

        except Exception as e:
            logger.error(f"Error predicting return rate for {symbol}: {str(e)}")
            return 0.0

    def prepare_features_for_return_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """リターン率予測用の特徴量準備"""
        features = self.prepare_features(data)
        
        # リターン率関連の特徴量を追加
        if len(data) >= 10:
            returns = data['Close'].pct_change()
            
            # 過去のリターン率統計
            features['return_mean_5d'] = returns.rolling(5).mean()
            features['return_std_5d'] = returns.rolling(5).std()
            features['return_mean_20d'] = returns.rolling(20).mean()
            features['return_std_20d'] = returns.rolling(20).std()
            
            # シャープレシオ風指標
            features['return_sharpe_5d'] = features['return_mean_5d'] / (features['return_std_5d'] + 1e-8)
            features['return_sharpe_20d'] = features['return_mean_20d'] / (features['return_std_20d'] + 1e-8)
            
            # 連続上昇/下降日数
            returns_sign = np.sign(returns)
            features['consecutive_up'] = (returns_sign.groupby((returns_sign != returns_sign.shift()).cumsum()).cumcount() + 1) * (returns_sign > 0)
            features['consecutive_down'] = (returns_sign.groupby((returns_sign != returns_sign.shift()).cumsum()).cumcount() + 1) * (returns_sign < 0)
        
        return features.fillna(0)

    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度を取得"""
        if not self.is_trained or self.model is None:
            return {}

        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                return dict(zip(self.feature_names, importances))
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")

        return {}

    def save_model(self):
        """モデルを保存"""
        try:
            model_file = self.model_path / f"ml_stock_predictor_{self.model_type}.joblib"
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
            model_file = self.model_path / f"ml_stock_predictor_{self.model_type}.joblib"
            scaler_file = self.model_path / f"scaler_{self.model_type}.joblib"
            features_file = self.model_path / f"features_{self.model_type}.joblib"

            if not all([model_file.exists(), scaler_file.exists(), features_file.exists()]):
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


class EnsembleStockPredictor:
    """複数モデルのアンサンブル予測器"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.data_provider = StockDataProvider()
        self.is_trained = False
        self.feature_names = []
        self.model_path = Path("models/saved_models")
        self.model_path.mkdir(exist_ok=True)
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """アンサンブルにモデルを追加"""
        self.models[name] = model
        self.weights[name] = weight
        
    def prepare_ensemble_models(self):
        """複数のモデルタイプを準備"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.svm import SVR
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Neural Network
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        )
        
        # アンサンブルに追加（重み付け）
        self.add_model("xgboost", xgb_model, weight=0.3)
        self.add_model("lightgbm", lgb_model, weight=0.3)
        self.add_model("random_forest", rf_model, weight=0.2)
        self.add_model("gradient_boost", gb_model, weight=0.15)
        self.add_model("neural_network", nn_model, weight=0.05)
        
    def train_ensemble(self, symbols: List[str], target_column: str = 'recommendation_score'):
        """アンサンブルモデルを訓練"""
        from config.settings import get_settings
        settings = get_settings()
        
        # モデル準備
        self.prepare_ensemble_models()
        
        # 単一モデルインスタンスでデータ準備
        ml_predictor = MLStockPredictor()
        logger.info("Preparing dataset for ensemble...")
        features, targets_reg, targets_cls = ml_predictor.prepare_dataset(symbols)
        
        if target_column not in targets_reg.columns:
            raise ValueError(f"Target column {target_column} not found")
            
        targets = targets_reg[target_column]
        
        # 欠損値除去
        valid_indices = ~(targets.isna() | features.isna().any(axis=1))
        features_clean = features[valid_indices]
        targets_clean = targets[valid_indices]
        
        if len(features_clean) < settings.model.min_training_data:
            raise ValueError(f"Insufficient training data: {len(features_clean)}")
            
        self.feature_names = features_clean.columns.tolist()
        
        # 時系列分割
        train_size = int(len(features_clean) * settings.model.train_test_split)
        X_train = features_clean.iloc[:train_size]
        X_test = features_clean.iloc[train_size:]
        y_train = targets_clean.iloc[:train_size]
        y_test = targets_clean.iloc[train_size:]
        
        # 特徴量スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 各モデルを訓練
        model_predictions = {}
        model_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                model.fit(X_train_scaled, y_train)
                
                # 予測と評価
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                model_predictions[name] = test_pred
                model_scores[name] = test_mse
                
                logger.info(f"{name} - Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                # 失敗したモデルは除外
                del self.models[name]
                del self.weights[name]
        
        # 動的重み調整（性能に基づく）
        self._adjust_weights_based_on_performance(model_scores)
        
        # アンサンブル予測の評価
        ensemble_pred = self._ensemble_predict_from_predictions(model_predictions)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        
        logger.info(f"Ensemble MSE: {ensemble_mse:.4f}")
        logger.info(f"Final model weights: {self.weights}")
        
        self.scaler = scaler
        self.is_trained = True
        
        # モデル保存
        self.save_ensemble()
        
    def _adjust_weights_based_on_performance(self, model_scores: Dict[str, float]):
        """性能に基づいて重みを動的調整"""
        # MSEが低いほど良いので、逆数を取って重み計算
        inverse_scores = {name: 1.0 / (score + 1e-6) for name, score in model_scores.items()}
        total_inverse = sum(inverse_scores.values())
        
        # 正規化して新しい重みを設定
        for name in self.weights:
            if name in inverse_scores:
                self.weights[name] = inverse_scores[name] / total_inverse
                
    def _ensemble_predict_from_predictions(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """複数モデルの予測を重み付き平均"""
        weighted_sum = np.zeros_like(list(model_predictions.values())[0])
        total_weight = 0
        
        for name, predictions in model_predictions.items():
            if name in self.weights:
                weighted_sum += predictions * self.weights[name]
                total_weight += self.weights[name]
                
        return weighted_sum / total_weight if total_weight > 0 else weighted_sum
        
    def predict_score(self, symbol: str) -> float:
        """アンサンブル予測"""
        if not self.is_trained:
            if not self.load_ensemble():
                logger.error("No trained ensemble model available")
                return 50.0
                
        try:
            # データ取得と特徴量準備
            data = self.data_provider.get_stock_data(symbol, "1y")
            if data.empty:
                return 0.0
                
            ml_predictor = MLStockPredictor()
            features = ml_predictor.prepare_features(data)
            if features.empty:
                return 0.0
                
            # 最新データの特徴量
            latest_features = features.iloc[-1:].copy()
            
            # 特徴量を訓練時と同じ順序に調整
            latest_features = latest_features.reindex(columns=self.feature_names, fill_value=0)
            
            # スケーリング
            features_scaled = self.scaler.transform(latest_features)
            
            # 各モデルの予測を収集
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(features_scaled)[0]
                    predictions[name] = pred
                except Exception as e:
                    logger.warning(f"Error with {name} prediction: {str(e)}")
                    
            # アンサンブル予測
            if predictions:
                ensemble_score = self._ensemble_predict_from_predictions(
                    {name: np.array([pred]) for name, pred in predictions.items()}
                )[0]
                return max(0, min(100, float(ensemble_score)))
            else:
                return 50.0
                
        except Exception as e:
            logger.error(f"Error in ensemble prediction for {symbol}: {str(e)}")
            return 50.0
            
    def save_ensemble(self):
        """アンサンブルモデルを保存"""
        try:
            ensemble_file = self.model_path / "ensemble_models.joblib"
            ensemble_data = {
                'models': self.models,
                'weights': self.weights,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained
            }
            joblib.dump(ensemble_data, ensemble_file)
            logger.info(f"Ensemble saved to {ensemble_file}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble: {str(e)}")
            
    def load_ensemble(self) -> bool:
        """アンサンブルモデルを読み込み"""
        try:
            ensemble_file = self.model_path / "ensemble_models.joblib"
            if not ensemble_file.exists():
                return False
                
            ensemble_data = joblib.load(ensemble_file)
            self.models = ensemble_data['models']
            self.weights = ensemble_data['weights']
            self.scaler = ensemble_data['scaler']
            self.feature_names = ensemble_data['feature_names']
            self.is_trained = ensemble_data['is_trained']
            
            logger.info(f"Ensemble loaded from {ensemble_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ensemble: {str(e)}")
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
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
            return -scores.mean()
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['xgboost'] = study.best_params
        logger.info(f"Best XGBoost params: {study.best_params}")
        logger.info(f"Best XGBoost score: {study.best_value}")
        
        return study.best_params
        
    def optimize_lightgbm(self, X, y, cv_folds=5, n_trials=100):
        """LightGBMパラメータ最適化"""
        import optuna
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
            return -scores.mean()
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['lightgbm'] = study.best_params
        logger.info(f"Best LightGBM params: {study.best_params}")
        logger.info(f"Best LightGBM score: {study.best_value}")
        
        return study.best_params
        
    def save_best_params(self):
        """最適パラメータを保存"""
        try:
            params_file = Path("models/saved_models/best_hyperparams.json")
            with open(params_file, 'w') as f:
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
                with open(params_file, 'r') as f:
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
            mask = y_test <= quantiles[i] if i == 0 else (y_test > quantiles[i-1]) & (y_test <= quantiles[i])
            if mask.sum() > 0:
                quantile_mse = mean_squared_error(y_test[mask], predictions[mask])
                quantile_performance[f'Q{i+1}'] = quantile_mse
                
        # 性能記録
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'direction_accuracy': direction_accuracy,
            'quantile_performance': quantile_performance,
            'sample_size': len(y_test)
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
        
        if performance_record['rmse'] > rmse_threshold:
            alerts.append(f"High RMSE: {performance_record['rmse']:.4f} > {rmse_threshold}")
            
        if performance_record['r2_score'] < r2_threshold:
            alerts.append(f"Low R²: {performance_record['r2_score']:.4f} < {r2_threshold}")
            
        if performance_record['direction_accuracy'] < direction_threshold:
            alerts.append(f"Low Direction Accuracy: {performance_record['direction_accuracy']:.4f} < {direction_threshold}")
            
        if alerts:
            alert_record = {
                'timestamp': performance_record['timestamp'],
                'model_name': performance_record['model_name'],
                'alerts': alerts
            }
            self.alerts.append(alert_record)
            logger.warning(f"Performance alerts for {performance_record['model_name']}: {alerts}")
            
    def get_performance_summary(self, last_n_records=10):
        """性能サマリーを取得"""
        if not self.performance_history:
            return "No performance data available"
            
        recent_records = self.performance_history[-last_n_records:]
        
        avg_rmse = np.mean([r['rmse'] for r in recent_records])
        avg_r2 = np.mean([r['r2_score'] for r in recent_records])
        avg_direction = np.mean([r['direction_accuracy'] for r in recent_records])
        
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
                'performance_history': self.performance_history,
                'alerts': self.alerts
            }
            with open(perf_file, 'w') as f:
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
                with open(perf_file, 'r') as f:
                    import json
                    data = json.load(f)
                    self.performance_history = data.get('performance_history', [])
                    self.alerts = data.get('alerts', [])
                logger.info("Performance data loaded")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error loading performance data: {str(e)}")
            return False

class ParallelStockPredictor:
    """並列処理対応の高速株価予測器"""
    
    def __init__(self, ensemble_predictor: EnsembleStockPredictor, n_jobs: int = -1):
        self.ensemble_predictor = ensemble_predictor
        self.n_jobs = n_jobs if n_jobs != -1 else os.cpu_count()
        self.batch_cache = {}
        
    def predict_multiple_stocks_parallel(self, symbols: List[str]) -> Dict[str, float]:
        """複数銘柄の並列予測"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        # キャッシュチェック
        uncached_symbols = []
        for symbol in symbols:
            if symbol in self.batch_cache:
                results[symbol] = self.batch_cache[symbol]
            else:
                uncached_symbols.append(symbol)
        
        if not uncached_symbols:
            return results
            
        logger.info(f"Predicting {len(uncached_symbols)} stocks in parallel...")
        
        # 並列実行
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_symbol = {
                executor.submit(self.ensemble_predictor.predict_score, symbol): symbol 
                for symbol in uncached_symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    score = future.result()
                    results[symbol] = score
                    self.batch_cache[symbol] = score
                except Exception as e:
                    logger.error(f"Error predicting {symbol}: {str(e)}")
                    results[symbol] = 50.0
                    
        return results
        
    def batch_data_preparation(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """バッチデータ準備（並列）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        data_results = {}
        
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_symbol = {
                executor.submit(self._get_stock_data_safe, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        data_results[symbol] = data
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {str(e)}")
                    
        return data_results
        
    def _get_stock_data_safe(self, symbol: str) -> pd.DataFrame:
        """安全なデータ取得"""
        try:
            return self.ensemble_predictor.data_provider.get_stock_data(symbol, "1y")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
            
    def clear_batch_cache(self):
        """バッチキャッシュをクリア"""
        self.batch_cache.clear()
        logger.info("Batch cache cleared")


class AdvancedCacheManager:
    """高度なキャッシュ管理システム"""
    
    def __init__(self):
        self.feature_cache = {}
        self.prediction_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'feature_cache_size': 0,
            'prediction_cache_size': 0
        }
        
    def get_cached_features(self, symbol: str, data_hash: str) -> Optional[pd.DataFrame]:
        """特徴量キャッシュから取得"""
        cache_key = f"{symbol}_{data_hash}"
        
        if cache_key in self.feature_cache:
            self.cache_stats['hits'] += 1
            return self.feature_cache[cache_key]
        else:
            self.cache_stats['misses'] += 1
            return None
            
    def cache_features(self, symbol: str, data_hash: str, features: pd.DataFrame):
        """特徴量をキャッシュ"""
        cache_key = f"{symbol}_{data_hash}"
        self.feature_cache[cache_key] = features
        self.cache_stats['feature_cache_size'] = len(self.feature_cache)
        
    def get_cached_prediction(self, symbol: str, features_hash: str) -> Optional[float]:
        """予測結果キャッシュから取得"""
        cache_key = f"{symbol}_{features_hash}"
        
        if cache_key in self.prediction_cache:
            self.cache_stats['hits'] += 1
            return self.prediction_cache[cache_key]
        else:
            self.cache_stats['misses'] += 1
            return None
            
    def cache_prediction(self, symbol: str, features_hash: str, prediction: float):
        """予測結果をキャッシュ"""
        cache_key = f"{symbol}_{features_hash}"
        self.prediction_cache[cache_key] = prediction
        self.cache_stats['prediction_cache_size'] = len(self.prediction_cache)
        
    def cleanup_old_cache(self, max_size: int = 1000):
        """古いキャッシュをクリーンアップ"""
        if len(self.feature_cache) > max_size:
            # 最も古いエントリを削除（簡単なLRU実装）
            keys_to_remove = list(self.feature_cache.keys())[:-max_size]
            for key in keys_to_remove:
                del self.feature_cache[key]
                
        if len(self.prediction_cache) > max_size:
            keys_to_remove = list(self.prediction_cache.keys())[:-max_size]
            for key in keys_to_remove:
                del self.prediction_cache[key]
                
        self.cache_stats['feature_cache_size'] = len(self.feature_cache)
        self.cache_stats['prediction_cache_size'] = len(self.prediction_cache)
        
    def get_cache_stats(self) -> Dict:
        """キャッシュ統計を取得"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

class DeepLearningPredictor:
    """LSTM/Transformer深層学習予測器"""
    
    def __init__(self, model_type="lstm"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = 60
        self.is_trained = False
        self.feature_columns = []
        
    def prepare_sequences(self, data: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """時系列データをシーケンスに変換"""
        # 特徴量とターゲット分離
        feature_data = data.drop(['Close'], axis=1).values
        target_data = data[target_col].values
        
        # 正規化
        feature_data = self.scaler.fit_transform(feature_data)
        
        X, y = [], []
        for i in range(self.sequence_length, len(feature_data)):
            X.append(feature_data[i-self.sequence_length:i])
            y.append(target_data[i])
            
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """LSTM モデル構築"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_transformer_model(self, input_shape):
        """Transformer モデル構築"""
        import tensorflow as tf
        from tensorflow.keras import layers
        
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Multi-head self-attention
            x = layers.MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
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
            x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.3)
        
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation="linear")(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )
        
        return model
    
    def train_deep_model(self, symbols: List[str]):
        """深層学習モデル訓練"""
        # データ準備
        all_data = []
        ml_predictor = MLStockPredictor()
        
        for symbol in symbols:
            data = ml_predictor.data_provider.get_stock_data(symbol, "3y")
            if len(data) < 200:
                continue
                
            features = ml_predictor.prepare_features(data)
            # Closeカラムを追加
            features['Close'] = data['Close']
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
        
        # 訓練
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=7, factor=0.5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        self.save_deep_model()
        
        # 評価
        test_pred = self.model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_pred)
        logger.info(f"Deep Learning {self.model_type} Test MSE: {test_mse:.4f}")
        
        return history
    
    def predict_deep(self, symbol: str) -> float:
        """深層学習予測"""
        if not self.is_trained:
            return 50.0
            
        try:
            ml_predictor = MLStockPredictor()
            data = ml_predictor.data_provider.get_stock_data(symbol, "1y")
            features = ml_predictor.prepare_features(data)
            features['Close'] = data['Close']
            
            if len(features) < self.sequence_length:
                return 50.0
                
            # 最新シーケンス準備
            recent_data = features.tail(self.sequence_length).drop(['Close'], axis=1).values
            recent_data = self.scaler.transform(recent_data)
            sequence = recent_data.reshape(1, self.sequence_length, -1)
            
            # 予測
            pred = self.model.predict(sequence)[0][0]
            
            # スコア変換 (価格予測→0-100スコア)
            current_price = data['Close'].iloc[-1]
            score = 50 + (pred - current_price) / current_price * 100
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Deep learning prediction error for {symbol}: {str(e)}")
            return 50.0
    
    def save_deep_model(self):
        """深層学習モデル保存"""
        try:
            model_path = Path("models/saved_models")
            self.model.save(model_path / f"deep_{self.model_type}_model.h5")
            joblib.dump(self.scaler, model_path / f"deep_{self.model_type}_scaler.joblib")
            logger.info(f"Deep {self.model_type} model saved")
        except Exception as e:
            logger.error(f"Error saving deep model: {str(e)}")


class SentimentAnalyzer:
    """センチメント分析器"""
    
    def __init__(self):
        self.sentiment_cache = {}
        
    def get_news_sentiment(self, symbol: str) -> Dict[str, float]:
        """ニュースセンチメント取得（模擬実装）"""
        # 実際の実装では Yahoo Finance APIやNews APIを使用
        import random
        
        cache_key = f"sentiment_{symbol}_{datetime.now().strftime('%Y-%m-%d')}"
        
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # 模擬センチメントデータ
        sentiment_data = {
            'positive_ratio': random.uniform(0.2, 0.8),
            'negative_ratio': random.uniform(0.1, 0.4),
            'neutral_ratio': random.uniform(0.2, 0.5),
            'news_volume': random.randint(5, 50),
            'sentiment_trend': random.uniform(-0.3, 0.3),
            'social_media_buzz': random.uniform(0.1, 0.9)
        }
        
        # 正規化
        total = sentiment_data['positive_ratio'] + sentiment_data['negative_ratio'] + sentiment_data['neutral_ratio']
        for key in ['positive_ratio', 'negative_ratio', 'neutral_ratio']:
            sentiment_data[key] /= total
            
        self.sentiment_cache[cache_key] = sentiment_data
        return sentiment_data
    
    def get_macro_economic_features(self) -> Dict[str, float]:
        """マクロ経済指標取得（模擬実装）"""
        # 実際の実装では FRED API や日本銀行 API を使用
        import random
        
        return {
            'interest_rate': random.uniform(0.001, 0.05),
            'inflation_rate': random.uniform(-0.01, 0.03),
            'gdp_growth': random.uniform(-0.02, 0.04),
            'unemployment_rate': random.uniform(0.02, 0.06),
            'exchange_rate_usd_jpy': random.uniform(140, 160),
            'oil_price': random.uniform(70, 120),
            'gold_price': random.uniform(1800, 2200),
            'vix_index': random.uniform(10, 40),
            'nikkei_momentum': random.uniform(-0.05, 0.05)
        }


class RedisCache:
    """Redis高速キャッシュシステム"""
    
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            import redis
            self.redis_client = redis.Redis(
                host=host, port=port, db=db, 
                decode_responses=True, socket_timeout=5
            )
            # 接続テスト
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"Redis not available, falling back to memory cache: {str(e)}")
            self.redis_available = False
            self.memory_cache = {}
    
    def get(self, key: str) -> Optional[str]:
        """キャッシュ取得"""
        try:
            if self.redis_available:
                return self.redis_client.get(key)
            else:
                return self.memory_cache.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None
    
    def set(self, key: str, value: str, ttl: int = 3600):
        """キャッシュ設定"""
        try:
            if self.redis_available:
                self.redis_client.setex(key, ttl, value)
            else:
                self.memory_cache[key] = value
                # メモリキャッシュのTTL管理は簡略化
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
    
    def get_json(self, key: str) -> Optional[Dict]:
        """JSON形式でキャッシュ取得"""
        try:
            data = self.get(key)
            if data:
                import json
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Cache JSON get error: {str(e)}")
            return None
    
    def set_json(self, key: str, value: Dict, ttl: int = 3600):
        """JSON形式でキャッシュ設定"""
        try:
            import json
            self.set(key, json.dumps(value), ttl)
        except Exception as e:
            logger.error(f"Cache JSON set error: {str(e)}")


class MetaLearningOptimizer:
    """メタラーニング・自動モデル選択システム"""
    
    def __init__(self):
        self.model_performance_history = {}
        self.meta_features = {}
        self.best_model_for_symbol = {}
        
    def extract_meta_features(self, symbol: str, data: pd.DataFrame) -> Dict[str, float]:
        """メタ特徴量抽出"""
        if data.empty:
            return {}
            
        meta_features = {
            # データ特性
            'data_length': len(data),
            'missing_ratio': data.isnull().sum().sum() / (data.shape[0] * data.shape[1]),
            
            # 価格特性
            'price_volatility': data['Close'].std() / data['Close'].mean(),
            'price_trend': (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0],
            'price_skewness': data['Close'].skew(),
            'price_kurtosis': data['Close'].kurtosis(),
            
            # ボリューム特性
            'volume_volatility': data['Volume'].std() / data['Volume'].mean(),
            'volume_trend': (data['Volume'].iloc[-20:].mean() - data['Volume'].iloc[:20].mean()) / data['Volume'].iloc[:20].mean(),
            
            # 技術指標特性
            'rsi_avg': data.get('RSI', pd.Series([50])).mean(),
            'macd_trend': data.get('MACD', pd.Series([0])).tail(10).mean(),
            
            # 市場特性
            'sector_correlation': 0.5,  # 業界相関（後で実装）
            'market_cap_category': 1.0,  # 時価総額カテゴリ（後で実装）
        }
        
        return meta_features
    
    def select_best_model(self, symbol: str, data: pd.DataFrame) -> str:
        """銘柄に最適なモデルを選択"""
        meta_features = self.extract_meta_features(symbol, data)
        
        # 過去の性能履歴から最適モデル選択
        if symbol in self.best_model_for_symbol:
            return self.best_model_for_symbol[symbol]
        
        # メタ特徴量に基づく推奨
        if meta_features.get('price_volatility', 0) > 0.05:
            return 'ensemble'  # 高ボラティリティ → アンサンブル
        elif meta_features.get('data_length', 0) > 500:
            return 'deep_learning'  # 長期データ → 深層学習
        else:
            return 'xgboost'  # デフォルト → XGBoost
    
    def update_model_performance(self, symbol: str, model_name: str, performance: float):
        """モデル性能を更新"""
        if symbol not in self.model_performance_history:
            self.model_performance_history[symbol] = {}
            
        if model_name not in self.model_performance_history[symbol]:
            self.model_performance_history[symbol][model_name] = []
            
        self.model_performance_history[symbol][model_name].append({
            'timestamp': datetime.now(),
            'performance': performance
        })
        
        # 最新10回の平均性能で最適モデル更新
        recent_performances = {}
        for model, history in self.model_performance_history[symbol].items():
            recent = [h['performance'] for h in history[-10:]]
            recent_performances[model] = np.mean(recent) if recent else 0
            
        if recent_performances:
            self.best_model_for_symbol[symbol] = max(recent_performances.keys(), 
                                                   key=lambda k: recent_performances[k])


class UltraHighPerformancePredictor:
    """超高性能予測システム統合"""
    
    def __init__(self):
        # コンポーネント初期化
        self.ensemble_predictor = EnsembleStockPredictor()
        self.deep_lstm = DeepLearningPredictor("lstm")
        self.deep_transformer = DeepLearningPredictor("transformer")
        self.sentiment_analyzer = SentimentAnalyzer()
        self.redis_cache = RedisCache()
        self.meta_optimizer = MetaLearningOptimizer()
        self.parallel_predictor = None
        
        # 性能監視
        self.performance_monitor = ModelPerformanceMonitor()
        
        # モデル重み
        self.model_weights = {
            'ensemble': 0.4,
            'deep_lstm': 0.25,
            'deep_transformer': 0.25,
            'sentiment': 0.1
        }
        
    def train_all_models(self, symbols: List[str]):
        """全モデルを並列訓練"""
        from concurrent.futures import ThreadPoolExecutor
        
        logger.info("Training ultra-high performance prediction system...")
        
        def train_ensemble():
            self.ensemble_predictor.train_ensemble(symbols)
            
        def train_lstm():
            self.deep_lstm.train_deep_model(symbols)
            
        def train_transformer():
            self.deep_transformer.train_deep_model(symbols)
        
        # 並列訓練
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(train_ensemble),
                executor.submit(train_lstm),
                executor.submit(train_transformer)
            ]
            
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Training error: {str(e)}")
        
        # 並列予測器初期化
        self.parallel_predictor = ParallelStockPredictor(self.ensemble_predictor)
        
        logger.info("Ultra-high performance system training completed!")
    
    def ultra_predict(self, symbol: str) -> float:
        """超高精度予測"""
        # キャッシュチェック
        cache_key = f"ultra_pred_{symbol}_{datetime.now().strftime('%Y-%m-%d-%H')}"
        cached_result = self.redis_cache.get(cache_key)
        
        if cached_result:
            return float(cached_result)
        
        try:
            # 最適モデル選択
            data = self.ensemble_predictor.data_provider.get_stock_data(symbol, "1y")
            best_model = self.meta_optimizer.select_best_model(symbol, data)
            
            # 各モデルの予測取得
            predictions = {}
            
            # アンサンブル予測
            try:
                predictions['ensemble'] = self.ensemble_predictor.predict_score(symbol)
            except:
                predictions['ensemble'] = 50.0
            
            # 深層学習予測
            try:
                predictions['deep_lstm'] = self.deep_lstm.predict_deep(symbol)
            except:
                predictions['deep_lstm'] = 50.0
                
            try:
                predictions['deep_transformer'] = self.deep_transformer.predict_deep(symbol)
            except:
                predictions['deep_transformer'] = 50.0
            
            # センチメント分析
            try:
                sentiment_data = self.sentiment_analyzer.get_news_sentiment(symbol)
                macro_data = self.sentiment_analyzer.get_macro_economic_features()
                
                # センチメントスコア計算
                sentiment_score = 50 + (sentiment_data['positive_ratio'] - sentiment_data['negative_ratio']) * 50
                sentiment_score += macro_data['gdp_growth'] * 100  # マクロ経済要因
                sentiment_score += (140 - macro_data['exchange_rate_usd_jpy']) * 0.5  # 為替影響
                
                predictions['sentiment'] = max(0, min(100, sentiment_score))
            except:
                predictions['sentiment'] = 50.0
            
            # 最適モデル強調
            if best_model in predictions:
                self.model_weights[best_model] *= 1.2
                # 重み正規化
                total_weight = sum(self.model_weights.values())
                self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
            
            # 重み付き最終予測
            final_score = sum(predictions[model] * weight 
                            for model, weight in self.model_weights.items() 
                            if model in predictions)
            
            # キャッシュ保存
            self.redis_cache.set(cache_key, str(final_score), ttl=3600)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Ultra prediction error for {symbol}: {str(e)}")
            return 50.0
