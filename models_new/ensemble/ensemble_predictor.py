#!/usr/bin/env python3
"""
アンサンブル予測器モジュール
複数のML技法を統合した高精度予測システム
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# サードパーティ ML ライブラリ
try:
    import xgboost as xgb
    import lightgbm as lgb
except ImportError:
    xgb = None
    lgb = None

from ..base.interfaces import StockPredictor, PredictionResult
from data.stock_data import StockDataProvider
from datetime import datetime


class EnsembleStockPredictor:
    """複数モデルのアンサンブル予測器"""

    def __init__(self, data_provider=None):
        self.models = {}
        self.weights = {}
        self.data_provider = data_provider or StockDataProvider()
        self.is_trained = False
        self.feature_names = []
        self.model_path = Path("models/saved_models")
        self.model_path.mkdir(exist_ok=True)
        self.scaler = None
        self.logger = logging.getLogger(__name__)

    def add_model(self, name: str, model, weight: float = 1.0):
        """アンサンブルにモデルを追加"""
        self.models[name] = model
        self.weights[name] = weight

    def prepare_ensemble_models(self):
        """複数のモデルタイプを準備"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.svm import SVR

        models_to_add = []

        # XGBoost
        if xgb is not None:
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            models_to_add.append(("xgboost", xgb_model, 0.3))

        # LightGBM
        if lgb is not None:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            models_to_add.append(("lightgbm", lgb_model, 0.3))

        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        models_to_add.append(("random_forest", rf_model, 0.2))

        # Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        models_to_add.append(("gradient_boost", gb_model, 0.15))

        # Neural Network
        nn_model = MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
        )
        models_to_add.append(("neural_network", nn_model, 0.05))

        # アンサンブルに追加（重み付け）
        for name, model, weight in models_to_add:
            self.add_model(name, model, weight=weight)

    def train_ensemble(
        self, symbols: List[str], target_column: str = "recommendation_score"
    ):
        """アンサンブルモデルを訓練"""
        from config.settings import get_settings
        settings = get_settings()

        # モデル準備
        self.prepare_ensemble_models()

        # 単一モデルインスタンスでデータ準備
        # MLStockPredictor は元のml_modelsにまだ存在するためそのまま維持
        from models.ml_models import MLStockPredictor
        ml_predictor = MLStockPredictor()

        self.logger.info("Preparing dataset for ensemble...")
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
            self.logger.info(f"Training {name}...")
            try:
                model.fit(X_train_scaled, y_train)

                # 予測と評価
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)

                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)

                model_predictions[name] = test_pred
                model_scores[name] = test_mse

                self.logger.info(
                    f"{name} - Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}"
                )

            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                # 失敗したモデルは除外
                del self.models[name]
                del self.weights[name]

        # 動的重み調整（性能に基づく）
        self._adjust_weights_based_on_performance(model_scores)

        # アンサンブル予測の評価
        ensemble_pred = self._ensemble_predict_from_predictions(model_predictions)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)

        self.logger.info(f"Ensemble MSE: {ensemble_mse:.4f}")
        self.logger.info(f"Final model weights: {self.weights}")

        self.scaler = scaler
        self.is_trained = True

        # モデル保存
        self.save_ensemble()

    def _adjust_weights_based_on_performance(self, model_scores: Dict[str, float]):
        """性能に基づいて重みを動的調整"""
        # MSEが低いほど良いので、逆数を取って重み計算
        inverse_scores = {
            name: 1.0 / (score + 1e-6) for name, score in model_scores.items()
        }
        total_inverse = sum(inverse_scores.values())

        # 正規化して新しい重みを設定
        for name in self.weights:
            if name in inverse_scores:
                self.weights[name] = inverse_scores[name] / total_inverse

    def _ensemble_predict_from_predictions(
        self, model_predictions: Dict[str, np.ndarray]
    ) -> np.ndarray:
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
                self.logger.error("No trained ensemble model available")
                return 50.0

        try:
            # データ取得と特徴量準備
            data = self.data_provider.get_stock_data(symbol, "1y")
            if data.empty:
                return 0.0

            # MLStockPredictor は元のml_modelsにまだ存在するためそのまま維持
        from models.ml_models import MLStockPredictor
            ml_predictor = MLStockPredictor()
            features = ml_predictor.prepare_features(data)

            if features.empty:
                return 0.0

            # 最新データの特徴量
            latest_features = features.iloc[-1:].copy()

            # 特徴量を訓練時と同じ順序に調整
            latest_features = latest_features.reindex(
                columns=self.feature_names, fill_value=0
            )

            # スケーリング
            features_scaled = self.scaler.transform(latest_features)

            # 各モデルの予測を収集
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(features_scaled)[0]
                    predictions[name] = pred
                except Exception as e:
                    self.logger.warning(f"Error with {name} prediction: {str(e)}")

            # アンサンブル予測
            if predictions:
                ensemble_score = self._ensemble_predict_from_predictions(
                    {name: np.array([pred]) for name, pred in predictions.items()}
                )[0]
                return max(0, min(100, float(ensemble_score)))
            else:
                return 50.0

        except Exception as e:
            self.logger.error(f"Error in ensemble prediction for {symbol}: {str(e)}")
            return 50.0

    def save_ensemble(self):
        """アンサンブルモデルを保存"""
        try:
            ensemble_file = self.model_path / "ensemble_models.joblib"
            ensemble_data = {
                "models": self.models,
                "weights": self.weights,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "is_trained": self.is_trained,
            }
            joblib.dump(ensemble_data, ensemble_file)
            self.logger.info(f"Ensemble saved to {ensemble_file}")
        except Exception as e:
            self.logger.error(f"Error saving ensemble: {str(e)}")

    def load_ensemble(self) -> bool:
        """アンサンブルモデルを読み込み"""
        try:
            ensemble_file = self.model_path / "ensemble_models.joblib"
            if not ensemble_file.exists():
                return False

            ensemble_data = joblib.load(ensemble_file)
            self.models = ensemble_data["models"]
            self.weights = ensemble_data["weights"]
            self.scaler = ensemble_data["scaler"]
            self.feature_names = ensemble_data["feature_names"]
            self.is_trained = ensemble_data["is_trained"]

            self.logger.info(f"Ensemble loaded from {ensemble_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading ensemble: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得"""
        return {
            'name': 'EnsembleStockPredictor',
            'version': '1.0.0',
            'models': list(self.models.keys()),
            'weights': self.weights,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names)
        }