"""
予測モデルファクトリ - 統一された予測器生成システム
複数の予測器を統一的に管理・生成するファクトリパターン実装
"""

import logging
from typing import Dict, Optional, Type, Any, List
from enum import Enum

from .interfaces import (
    StockPredictor,
    ModelConfiguration,
    ModelType,
    PredictionMode,
    DataProvider,
    CacheProvider,
)


class PredictorFactory:
    """
    予測器ファクトリクラス
    全ての予測モデルを統一的に生成・管理
    """

    _registered_predictors: Dict[ModelType, Type[StockPredictor]] = {}
    _instances: Dict[str, StockPredictor] = {}

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._register_default_predictors()

    @classmethod
    def register_predictor(
        cls, model_type: ModelType, predictor_class: Type[StockPredictor]
    ):
        """予測器クラスの登録"""
        cls._registered_predictors[model_type] = predictor_class
        logging.getLogger(__name__).info(
            f"Registered predictor: {model_type.value} -> {predictor_class.__name__}"
        )

    @classmethod
    def create_predictor(
        cls,
        model_type: ModelType,
        config: Optional[ModelConfiguration] = None,
        data_provider: Optional[DataProvider] = None,
        cache_provider: Optional[CacheProvider] = None,
        **kwargs,
    ) -> StockPredictor:
        """
        予測器の生成

        Args:
            model_type: モデルタイプ
            config: モデル設定
            data_provider: データプロバイダー
            cache_provider: キャッシュプロバイダー
            **kwargs: その他のパラメータ

        Returns:
            StockPredictor: 生成された予測器
        """
        if model_type not in cls._registered_predictors:
            available_types = list(cls._registered_predictors.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. Available: {available_types}"
            )

        # デフォルト設定の作成
        if config is None:
            config = ModelConfiguration(
                model_type=model_type, prediction_mode=PredictionMode.BALANCED
            )
        else:
            config.model_type = model_type

        predictor_class = cls._registered_predictors[model_type]

        try:
            # 予測器の生成
            predictor = predictor_class(
                config=config,
                data_provider=data_provider,
                cache_provider=cache_provider,
                **kwargs,
            )

            logging.getLogger(__name__).info(
                f"Created predictor: {predictor_class.__name__} with mode {config.prediction_mode.value}"
            )

            return predictor

        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to create predictor {model_type.value}: {str(e)}"
            )
            raise

    @classmethod
    def get_or_create_predictor(
        cls,
        model_type: ModelType,
        instance_name: Optional[str] = None,
        config: Optional[ModelConfiguration] = None,
        data_provider: Optional[DataProvider] = None,
        cache_provider: Optional[CacheProvider] = None,
        **kwargs,
    ) -> StockPredictor:
        """
        予測器の取得または生成（シングルトンパターン）

        Args:
            model_type: モデルタイプ
            instance_name: インスタンス名（Noneの場合はmodel_type.valueを使用）
            config: モデル設定
            data_provider: データプロバイダー
            cache_provider: キャッシュプロバイダー
            **kwargs: その他のパラメータ

        Returns:
            StockPredictor: 既存または新規作成された予測器
        """
        if instance_name is None:
            instance_name = model_type.value

        if instance_name in cls._instances:
            return cls._instances[instance_name]

        predictor = cls.create_predictor(
            model_type=model_type,
            config=config,
            data_provider=data_provider,
            cache_provider=cache_provider,
            **kwargs,
        )

        cls._instances[instance_name] = predictor
        return predictor

    @classmethod
    def create_ensemble_predictors(
        cls,
        model_types: List[ModelType],
        config: Optional[ModelConfiguration] = None,
        data_provider: Optional[DataProvider] = None,
        cache_provider: Optional[CacheProvider] = None,
    ) -> List[StockPredictor]:
        """
        複数の予測器をアンサンブル用に生成

        Args:
            model_types: 生成するモデルタイプのリスト
            config: 共通設定
            data_provider: データプロバイダー
            cache_provider: キャッシュプロバイダー

        Returns:
            List[StockPredictor]: 生成された予測器のリスト
        """
        predictors = []

        for model_type in model_types:
            try:
                predictor = cls.create_predictor(
                    model_type=model_type,
                    config=config,
                    data_provider=data_provider,
                    cache_provider=cache_provider,
                )
                predictors.append(predictor)
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Failed to create predictor for ensemble: {model_type.value} - {str(e)}"
                )

        logging.getLogger(__name__).info(
            f"Created ensemble with {len(predictors)} predictors"
        )
        return predictors

    @classmethod
    def list_available_types(cls) -> List[ModelType]:
        """利用可能なモデルタイプの取得"""
        return list(cls._registered_predictors.keys())

    @classmethod
    def get_predictor_class(
        cls, model_type: ModelType
    ) -> Optional[Type[StockPredictor]]:
        """モデルタイプに対応する予測器クラスの取得"""
        return cls._registered_predictors.get(model_type)

    @classmethod
    def clear_instances(cls):
        """インスタンスキャッシュのクリア"""
        cls._instances.clear()
        logging.getLogger(__name__).info("Cleared all predictor instances")

    @classmethod
    def get_instance_info(cls) -> Dict[str, Dict[str, Any]]:
        """インスタンス情報の取得"""
        info = {}
        for name, predictor in cls._instances.items():
            info[name] = predictor.get_model_info()
        return info

    def _register_default_predictors(self):
        """デフォルト予測器の登録"""
        try:
            # 遅延インポートで循環参照を回避
            from ..ensemble.ensemble_predictor import EnsemblePredictor
            from ..hybrid.hybrid_predictor import RefactoredHybridPredictor
            from ..deep_learning.deep_predictor import RefactoredDeepLearningPredictor

            self.register_predictor(ModelType.ENSEMBLE, EnsemblePredictor)
            self.register_predictor(ModelType.HYBRID, RefactoredHybridPredictor)
            self.register_predictor(
                ModelType.DEEP_LEARNING, RefactoredDeepLearningPredictor
            )

            self.logger.info("Default predictors registered successfully")

        except ImportError as e:
            self.logger.warning(f"Some default predictors could not be registered: {e}")

    @staticmethod
    def create_default_config(
        model_type: ModelType,
        prediction_mode: PredictionMode = PredictionMode.BALANCED,
        cache_enabled: bool = True,
        parallel_enabled: bool = True,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> ModelConfiguration:
        """デフォルト設定の作成ヘルパー"""
        return ModelConfiguration(
            model_type=model_type,
            prediction_mode=prediction_mode,
            cache_enabled=cache_enabled,
            parallel_enabled=parallel_enabled,
            custom_params=custom_params or {},
        )


# グローバルファクトリインスタンス
_global_factory = PredictorFactory()


# 便利関数
def create_predictor(model_type: ModelType, **kwargs) -> StockPredictor:
    """グローバルファクトリを使用した予測器生成"""
    return _global_factory.create_predictor(model_type, **kwargs)


def get_or_create_predictor(model_type: ModelType, **kwargs) -> StockPredictor:
    """グローバルファクトリを使用した予測器取得/生成"""
    return _global_factory.get_or_create_predictor(model_type, **kwargs)


def register_predictor(model_type: ModelType, predictor_class: Type[StockPredictor]):
    """グローバルファクトリへの予測器登録"""
    return _global_factory.register_predictor(model_type, predictor_class)


def list_available_types() -> List[ModelType]:
    """利用可能なモデルタイプの取得"""
    return _global_factory.list_available_types()


def clear_instances():
    """インスタンスキャッシュのクリア"""
    return _global_factory.clear_instances()
