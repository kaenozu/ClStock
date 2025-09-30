#!/usr/bin/env python3
"""
アンサンブル予測器モジュール
複数のML技法を統合した高精度予測システム
"""

import logging
import os
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


# 分離されたクラスのインポート
from .parallel_feature_calculator import ParallelFeatureCalculator
from .memory_efficient_cache import MemoryEfficientCache
from .multi_timeframe_integrator import MultiTimeframeIntegrator


class EnsembleStockPredictor(StockPredictor):
    """複数モデルのアンサンブル予測器"""

    # 予測重みの定数
    BASE_PREDICTION_WEIGHT = 0.7
    TIMEFRAME_PREDICTION_WEIGHT = 0.3

    # 信頼度調整の定数
    MAX_CONFIDENCE_RATIO = 0.3
    CONFIDENCE_MULTIPLIER = 10

    # 中性予測調整の定数
    NEUTRAL_PREDICTION_VALUE = 50.0
    HIGH_CONFIDENCE_NEUTRAL_WEIGHT = 0.1
    LOW_CONFIDENCE_NEUTRAL_WEIGHT = 0.3
    HIGH_CONFIDENCE_PREDICTION_WEIGHT = 0.9
    LOW_CONFIDENCE_PREDICTION_WEIGHT = 0.7

    # キャッシュサイズの定数
    FEATURE_CACHE_SIZE = 500
    PREDICTION_CACHE_SIZE = 200

    # フォールバック予測の定数
    TREND_THRESHOLD = 0.02  # 2%のトレンド閾値
    BULLISH_PREDICTION = 65.0
    BEARISH_PREDICTION = 35.0
    FALLBACK_ACCURACY = 45.0
    DEFAULT_CONFIDENCE = 0.1

    # 信頼度レベルの閾値
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    MEDIUM_CONFIDENCE_THRESHOLD = 0.4

    def __init__(self, data_provider=None):
        self.models = {}
        self.weights = {}
        self.data_provider = data_provider or StockDataProvider()
        self.is_trained = False
        self.feature_names = []
        self.model_path = Path("models/saved_models")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.scaler = None
        self.logger = logging.getLogger(__name__)

        # 並列特徴量計算システム
        self.parallel_calculator = ParallelFeatureCalculator()

        # メモリ効率キャッシュシステム（定数使用）
        self.feature_cache = MemoryEfficientCache(max_size=self.FEATURE_CACHE_SIZE)
        self.prediction_cache = MemoryEfficientCache(
            max_size=self.PREDICTION_CACHE_SIZE
        )

        # マルチタイムフレーム統合システム
        self.timeframe_integrator = MultiTimeframeIntegrator()

    def _validate_symbol(self, symbol: str) -> bool:
        """銘柄コードの検証"""
        if not symbol or not isinstance(symbol, str):
            return False
        # 日本株の銘柄コード検証（4桁数字 + オプションでT）
        import re

        pattern = r"^\d{4}(\.T)?$"
        return bool(re.match(pattern, symbol.strip()))

    def _validate_symbols_list(self, symbols: List[str]) -> List[str]:
        """銘柄リストの検証とクリーニング"""
        if not symbols or not isinstance(symbols, list):
            raise ValueError("Symbols must be a non-empty list")

        valid_symbols = []
        for symbol in symbols:
            if self._validate_symbol(symbol):
                valid_symbols.append(symbol.strip())
            else:
                self.logger.warning(f"Invalid symbol format: {symbol}")

        if not valid_symbols:
            raise ValueError("No valid symbols provided")

        return valid_symbols

    def _safe_model_operation(
        self, operation_name: str, operation_func, fallback_value=None
    ):
        """モデル操作の安全な実行"""
        try:
            return operation_func()
        except Exception as e:
            self.logger.error(f"Error in {operation_name}: {str(e)}")
            return fallback_value

    def _check_dependencies(self) -> Dict[str, bool]:
        """依存関係のチェック"""
        dependencies = {
            "sklearn": True,
            "numpy": True,
            "pandas": True,
            "xgboost": xgb is not None,
            "lightgbm": lgb is not None,
        }

        try:
            import sklearn
            import numpy
            import pandas
        except ImportError as e:
            self.logger.error(f"Critical dependency missing: {e}")
            dependencies.update({"sklearn": False, "numpy": False, "pandas": False})

        return dependencies

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
        from models.ml_stock_predictor import MLStockPredictor

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

    # ===== StockPredictor インターフェース実装 =====

    def predict(self, symbol: str) -> PredictionResult:
        """単一銘柄の予測を実行（インターフェース準拠）"""
        # 入力検証
        if not self._validate_symbol(symbol):
            self.logger.error(f"Invalid symbol format: {symbol}")
            return PredictionResult(
                prediction=self.NEUTRAL_PREDICTION_VALUE,
                confidence=0.0,
                accuracy=0.0,
                timestamp=datetime.now(),
                symbol=symbol,
                metadata={"error": "Invalid symbol format"},
            )

        # 依存関係チェック
        deps = self._check_dependencies()
        if not all([deps["sklearn"], deps["numpy"], deps["pandas"]]):
            self.logger.error("Critical dependencies missing")
            return PredictionResult(
                prediction=self.NEUTRAL_PREDICTION_VALUE,
                confidence=0.0,
                accuracy=0.0,
                timestamp=datetime.now(),
                symbol=symbol,
                metadata={"error": "Missing dependencies"},
            )

        try:
            # モデル準備状態チェック
            if not self.is_trained:
                if not self.load_ensemble():
                    self.logger.warning(
                        f"No trained model available for {symbol}, using fallback"
                    )
                    return self._fallback_prediction(symbol)

            prediction_value = self._safe_model_operation(
                "predict_score",
                lambda: self.predict_score(symbol),
                fallback_value=self.NEUTRAL_PREDICTION_VALUE,
            )

            confidence = self._safe_model_operation(
                "get_confidence",
                lambda: self.get_confidence(symbol),
                fallback_value=0.5,
            )

            # 予測値の範囲チェック
            prediction_value = max(0.0, min(100.0, float(prediction_value)))
            confidence = max(0.0, min(1.0, float(confidence)))

            return PredictionResult(
                prediction=prediction_value,
                confidence=confidence,
                accuracy=85.0,  # アンサンブルの推定精度
                timestamp=datetime.now(),
                symbol=symbol,
                metadata={
                    "model_type": "ensemble",
                    "models_used": list(self.models.keys()),
                    "weights": self.weights,
                    "validated": True,
                },
            )
        except ValueError as e:
            self.logger.error(f"Data validation error for {symbol}: {str(e)}")
            return self._fallback_prediction(
                symbol, error=f"データ検証エラー: {str(e)}"
            )
        except KeyError as e:
            self.logger.error(f"Missing data key for {symbol}: {str(e)}")
            return self._fallback_prediction(
                symbol, error=f"データキーエラー: {str(e)}"
            )
        except RuntimeError as e:
            self.logger.error(f"Model execution error for {symbol}: {str(e)}")
            return self._fallback_prediction(
                symbol, error=f"モデル実行エラー: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in predict for {symbol}: {str(e)}")
            return self._fallback_prediction(
                symbol, error=f"予期しないエラー: {str(e)}"
            )

    def _fallback_prediction(self, symbol: str, error: str = None) -> PredictionResult:
        """フォールバック予測（モデル利用不可時）"""
        try:
            # 簡単な移動平均ベースの予測
            data = self.data_provider.get_stock_data(symbol, "30d")
            if not data.empty and len(data) >= 5:
                recent_prices = data["Close"].tail(5)
                trend = (
                    recent_prices.iloc[-1] - recent_prices.iloc[0]
                ) / recent_prices.iloc[0]

                if trend > self.TREND_THRESHOLD:  # 2%以上上昇
                    prediction = self.BULLISH_PREDICTION
                elif trend < -self.TREND_THRESHOLD:  # 2%以上下落
                    prediction = self.BEARISH_PREDICTION
                else:
                    prediction = self.NEUTRAL_PREDICTION_VALUE

                confidence = min(
                    self.MAX_CONFIDENCE_RATIO, abs(trend) * self.CONFIDENCE_MULTIPLIER
                )
            else:
                prediction = self.NEUTRAL_PREDICTION_VALUE
                confidence = self.DEFAULT_CONFIDENCE

        except Exception:
            prediction = self.NEUTRAL_PREDICTION_VALUE
            confidence = self.DEFAULT_CONFIDENCE

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            accuracy=self.FALLBACK_ACCURACY,  # フォールバック精度は低め
            timestamp=datetime.now(),
            symbol=symbol,
            metadata={
                "model_type": "fallback",
                "method": "moving_average_trend",
                "error": error,
            },
        )

    def predict_batch(self, symbols: List[str]) -> List[PredictionResult]:
        """複数銘柄の一括予測（インターフェース準拠）"""
        try:
            # 入力検証
            valid_symbols = self._validate_symbols_list(symbols)
            self.logger.info(
                f"Processing batch prediction for {len(valid_symbols)} symbols"
            )

            results = []
            failed_count = 0

            for i, symbol in enumerate(valid_symbols):
                try:
                    result = self.predict(symbol)
                    results.append(result)

                    # 進捗ログ（大量データ処理時用）
                    if (i + 1) % 10 == 0:
                        self.logger.info(
                            f"Processed {i + 1}/{len(valid_symbols)} symbols"
                        )

                except Exception as e:
                    self.logger.error(f"Error predicting {symbol}: {str(e)}")
                    failed_count += 1
                    # エラーでも結果は返す（フォールバック）
                    results.append(self._fallback_prediction(symbol, error=str(e)))

            if failed_count > 0:
                self.logger.warning(
                    f"Batch prediction completed with {failed_count} failures"
                )

            return results

        except ValueError as e:
            self.logger.error(f"Batch prediction validation error: {str(e)}")
            # 無効な入力の場合、空のリストを返す
            return []
        except Exception as e:
            self.logger.error(f"Batch prediction error: {str(e)}")
            # 重大なエラーの場合、全て中性予測で返す
            return [
                self._fallback_prediction(sym, error=str(e))
                for sym in symbols
                if isinstance(sym, str)
            ]

    def get_confidence(self, symbol: str) -> float:
        """予測信頼度を取得（インターフェース準拠）"""
        if not self.is_trained:
            return 0.0

        try:
            # モデル数と重みの一貫性から信頼度を計算
            active_models = len([m for m in self.models.values() if m is not None])
            weight_consistency = 1.0 - np.std(list(self.weights.values()))

            # 基本信頼度：モデル数が多いほど高い
            base_confidence = min(0.9, active_models / 10.0)

            # 重みの一貫性を考慮
            confidence = base_confidence * weight_consistency

            return max(0.1, min(0.95, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating confidence for {symbol}: {str(e)}")
            return 0.1

    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報取得（インターフェース準拠）"""
        return {
            "name": "EnsembleStockPredictor",
            "version": "1.0.0",
            "models": list(self.models.keys()),
            "weights": self.weights,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names),
            "interface_compliant": True,
        }

    # ===== 既存メソッド（下位互換性のため維持） =====

    def predict_score(self, symbol: str) -> float:
        """アンサンブル予測（並列特徴量計算＋マルチタイムフレーム統合対応）"""
        if not self.is_trained:
            if not self.load_ensemble():
                self.logger.error("No trained ensemble model available")
                return self.NEUTRAL_PREDICTION_VALUE

        try:
            # キャッシュチェック
            cache_key = f"prediction_{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
            cached_result = self.prediction_cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Using cached prediction for {symbol}")
                return cached_result

            # マルチタイムフレーム統合分析
            timeframe_analysis = self.timeframe_integrator.integrate_predictions(
                symbol, self.data_provider
            )

            # 基本の特徴量ベース予測
            base_prediction = self._get_base_ensemble_prediction(symbol)

            # タイムフレーム統合予測
            integrated_prediction = timeframe_analysis["integrated_prediction"]
            confidence_adjustment = timeframe_analysis["confidence_adjustment"]

            # 最終予測の計算（基本予測 + タイムフレーム統合）
            final_prediction = (
                base_prediction * self.BASE_PREDICTION_WEIGHT
                + integrated_prediction * self.TIMEFRAME_PREDICTION_WEIGHT
            )

            # 信頼度による調整（定数使用）
            if confidence_adjustment > self.HIGH_CONFIDENCE_THRESHOLD:  # 高信頼度
                final_prediction = final_prediction  # そのまま
            elif confidence_adjustment > self.MEDIUM_CONFIDENCE_THRESHOLD:  # 中信頼度
                final_prediction = (
                    final_prediction * self.HIGH_CONFIDENCE_PREDICTION_WEIGHT
                    + self.NEUTRAL_PREDICTION_VALUE
                    * self.HIGH_CONFIDENCE_NEUTRAL_WEIGHT
                )
            else:  # 低信頼度
                final_prediction = (
                    final_prediction * self.LOW_CONFIDENCE_PREDICTION_WEIGHT
                    + self.NEUTRAL_PREDICTION_VALUE * self.LOW_CONFIDENCE_NEUTRAL_WEIGHT
                )

            result = max(0, min(100, float(final_prediction)))

            # 結果をキャッシュ
            self.prediction_cache.put(cache_key, result)

            # デバッグ情報をログ出力
            self.logger.debug(
                f"Enhanced prediction for {symbol}: "
                f"base={base_prediction:.1f}, integrated={integrated_prediction:.1f}, "
                f"confidence={confidence_adjustment:.2f}, final={result:.1f}"
            )

            return result

        except ConnectionError as e:
            self.logger.error(f"Data connection error for {symbol}: {str(e)}")
            return self.NEUTRAL_PREDICTION_VALUE
        except KeyError as e:
            self.logger.error(f"Missing data key for {symbol}: {str(e)}")
            return self.NEUTRAL_PREDICTION_VALUE
        except ValueError as e:
            self.logger.error(f"Data validation error for {symbol}: {str(e)}")
            return self.NEUTRAL_PREDICTION_VALUE
        except RuntimeError as e:
            self.logger.error(f"Model execution error for {symbol}: {str(e)}")
            return self.NEUTRAL_PREDICTION_VALUE
        except Exception as e:
            self.logger.error(
                f"Unexpected error in enhanced ensemble prediction for {symbol}: {str(e)}"
            )
            return self.NEUTRAL_PREDICTION_VALUE

    def _get_base_ensemble_prediction(self, symbol: str) -> float:
        """基本のアンサンブル予測（従来の特徴量ベース）"""
        try:
            # データ取得
            data = self.data_provider.get_stock_data(symbol, "1y")
            if data.empty:
                return 50.0

            # 特徴量計算（並列処理対応）
            features = self._calculate_features_optimized(symbol, data)
            if features.empty:
                return 50.0

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
            self.logger.error(
                f"Error in base ensemble prediction for {symbol}: {str(e)}"
            )
            return 50.0

    def _calculate_features_optimized(
        self, symbol: str, data: pd.DataFrame
    ) -> pd.DataFrame:
        """最適化された特徴量計算（キャッシュ対応）"""
        # キャッシュキー生成（データのハッシュベース）
        data_hash = pd.util.hash_pandas_object(data.tail(100)).sum()
        cache_key = f"features_{symbol}_{data_hash}"

        # キャッシュチェック
        cached_features = self.feature_cache.get(cache_key)
        if cached_features is not None:
            self.logger.debug(f"Using cached features for {symbol}")
            return cached_features

        try:
            # 並列特徴量計算システムを使用
            features = self.parallel_calculator._calculate_single_symbol_features(
                symbol, self.data_provider
            )

            if not features.empty:
                # キャッシュに保存
                self.feature_cache.put(cache_key, features)
                self.logger.debug(f"Cached features for {symbol}")

            return features

        except Exception as e:
            self.logger.error(
                f"Error calculating optimized features for {symbol}: {str(e)}"
            )
            # フォールバック：従来の特徴量計算
            return self._calculate_features_fallback(data)

    def _calculate_features_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """フォールバック特徴量計算（従来方式）"""
        try:
            from models.ml_stock_predictor import MLStockPredictor

            ml_predictor = MLStockPredictor()
            return ml_predictor.prepare_features(data)
        except Exception as e:
            self.logger.error(f"Fallback feature calculation failed: {str(e)}")
            return pd.DataFrame()

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
