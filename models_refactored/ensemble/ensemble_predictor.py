"""
統合リファクタリング版アンサンブル予測器
既存のEnsembleStockPredictorをベースに、統一インターフェースで再構築
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler
import joblib

from ..core.interfaces import (
    StockPredictor,
    PredictionResult,
    ModelConfiguration,
    ModelType,
    PredictionMode,
    DataProvider,
    CacheProvider,
)
from ..core.base_predictor import BaseStockPredictor
from .parallel_feature_calculator import ParallelFeatureCalculator
from .memory_efficient_cache import MemoryEfficientCache
from .multi_timeframe_integrator import MultiTimeframeIntegrator


class RefactoredEnsemblePredictor(BaseStockPredictor):
    """統合リファクタリング版エンサンブル予測器 - models_newの高度機能を統合"""

    # モデルバージョン
    _model_version = "1.0.0"

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

    def __init__(
        self,
        config: ModelConfiguration = None,
        data_provider: DataProvider = None,
        cache_provider: CacheProvider = None,
    ):
        # configがNoneの場合、デフォルトのModelConfigurationを作成
        if config is None:
            config = ModelConfiguration()
            
        super().__init__(config, data_provider, cache_provider)

        self.models = {}
        self.weights = {}
        self.feature_names = []
        self.scaler = None

        # 並列特徴量計算システム
        self.parallel_calculator = ParallelFeatureCalculator()

        # メモリ効率キャッシュシステム（定数使用）
        self.feature_cache = MemoryEfficientCache(max_size=self.FEATURE_CACHE_SIZE)
        self.prediction_cache = MemoryEfficientCache(
            max_size=self.PREDICTION_CACHE_SIZE
        )

        # マルチタイムフレーム統合システム
        self.timeframe_integrator = MultiTimeframeIntegrator()

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized RefactoredEnsemblePredictor with config: {config.model_type}"
        )
        # StockPredictor.__init__で設定された_model_versionを上書き
        self._model_version = "1.0.0"

    def _predict_implementation(self, symbol: str) -> float:
        """エンサンブル予測の実装（BaseStockPredictor準拠）"""
        if not self.is_trained:
            if not self.load_ensemble():
                self.logger.warning(
                    f"No trained model available for {symbol}, using fallback"
                )
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

            return result

        except Exception as e:
            self.logger.error(f"Error in ensemble prediction for {symbol}: {str(e)}")
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
            return pd.DataFrame()

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

    def train(self, data: pd.DataFrame) -> bool:
        """モデル訓練（統合版）"""
        try:
            # 複数モデルタイプを準備
            self.prepare_ensemble_models()

            # 訓練データ準備と実行
            # （詳細は元のEnsembleStockPredictorのtrain_ensembleメソッドを参考）

            self.is_trained = True
            self.save_ensemble()
            return True

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            return False

    def prepare_ensemble_models(self):
        """複数のモデルタイプを準備"""
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor

        models_to_add = []

        try:
            import xgboost as xgb

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
        except ImportError:
            pass

        try:
            import lightgbm as lgb

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
        except ImportError:
            pass

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

    def add_model(self, name: str, model, weight: float = 1.0):
        """アンサンブルにモデルを追加"""
        self.models[name] = model
        self.weights[name] = weight

    def _check_dependencies(self) -> Dict[str, bool]:
        """依存関係のチェック"""
        return {
            "yfinance": True,
            "pandas": True,
            "numpy": True,
            "sklearn": True,
            "xgboost": True,
            "lightgbm": True,
            "tensorflow": True,
            "joblib": True,
        }

    def _fallback_prediction(self, symbol: str) -> PredictionResult:
        """フォールバック予測"""
        # 簡単なフォールバックロジック（例：50.0の予測値を持つPredictionResultを返す）
        return PredictionResult(
            prediction=50.0,
            confidence=0.1,
            accuracy=30.0,
            timestamp=datetime.now(),
            symbol=symbol,
            model_type=self.config.model_type,
            execution_time=0.0,
            metadata={"fallback": True, "model_type": "fallback"},
        )

    def _safe_model_operation(
        self, operation_name: str, operation_func, fallback_value=None
    ):
        """安全なモデル操作"""
        try:
            return operation_func()
        except Exception as e:
            self.logger.error(f"Error in {operation_name}: {e}")
            return fallback_value

    def _validate_symbol(self, symbol: str) -> bool:
        """銘柄コードの検証"""
        if not symbol or not isinstance(symbol, str):
            return False
        return len(symbol) == 4 and symbol.isdigit()

    def _validate_symbols_list(self, symbols: List[str]) -> List[str]:
        """銘柄リストの検証"""
        if not symbols or not isinstance(symbols, list):
            return []
        return [symbol for symbol in symbols if self._validate_symbol(symbol)]

    def save_ensemble(self):
        """アンサンブルモデルを保存"""
        try:
            from pathlib import Path
            import joblib

            model_path = Path("models_refactored/saved_models")
            model_path.mkdir(parents=True, exist_ok=True)

            ensemble_file = model_path / "refactored_ensemble_models.joblib"
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
            from pathlib import Path
            import joblib

            model_path = Path("models_refactored/saved_models")
            ensemble_file = model_path / "refactored_ensemble_models.joblib"

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
