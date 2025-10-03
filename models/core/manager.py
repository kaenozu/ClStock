"""
モデル管理システム - 予測器のライフサイクル管理
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .interfaces import StockPredictor, ModelType, PerformanceMetrics


class ModelManager:
    """予測器のライフサイクルを管理するマネージャー"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, StockPredictor] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}

    def register_model(self, name: str, predictor: StockPredictor) -> bool:
        """モデルを登録"""
        try:
            self.models[name] = predictor
            self.model_metadata[name] = {
                "registered_at": datetime.now(),
                "model_info": predictor.get_model_info(),
            }
            self.logger.info(f"Model registered: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register model {name}: {e}")
            return False

    def get_model(self, name: str) -> Optional[StockPredictor]:
        """登録されたモデルを取得"""
        return self.models.get(name)

    def list_models(self) -> List[str]:
        """登録されているモデル名のリストを取得"""
        return list(self.models.keys())

    def remove_model(self, name: str) -> bool:
        """モデルを削除"""
        if name in self.models:
            del self.models[name]
            del self.model_metadata[name]
            self.logger.info(f"Model removed: {name}")
            return True
        return False
