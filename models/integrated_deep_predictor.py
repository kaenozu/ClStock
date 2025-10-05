"""新モデルのClStockシステムへの統合"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime

from models.custom_deep_predictor import CustomDeepPredictor
from analysis.multimodal_integration import MultimodalIntegrator
from models.attention_mechanism import AdaptiveTemporalAttention
from systems.reinforcement_trading_system import ReinforcementTradingSystem
from models.self_supervised_learning import SelfSupervisedModel
from analysis.model_interpretability import ModelInterpretability

logger = logging.getLogger(__name__)

class IntegratedDeepPredictor:
    """ClStockシステムに統合された新しい深層学習モデル"""
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 attention_units: int = 64,
                 encoding_dim: int = 32,
                 learning_rate: float = 0.001):
        self.input_shape = input_shape
        self.attention_units = attention_units
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        
        # 各コンポーネントの初期化
        self.custom_model = CustomDeepPredictor(
            input_shape=input_shape, 
            learning_rate=learning_rate
        )
        self.multimodal_integrator = MultimodalIntegrator()
        self.adaptive_attention = AdaptiveTemporalAttention(attention_units)
        self.self_supervised_model = SelfSupervisedModel(
            input_shape=(input_shape[0],), 
            encoding_dim=encoding_dim,
            learning_rate=learning_rate
        )
        self.reinforcement_trader = ReinforcementTradingSystem()
        self.interpretability = None
        
        # トレーニング状態
        self.is_trained = False
        self.training_history = {}
        
    def integrate_multimodal_data(self, 
                                  price_data: pd.DataFrame,
                                  fundamentals_data: Optional[pd.DataFrame] = None,
                                  sentiment_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        マルチモーダルデータの統合
        
        Args:
            price_data: 株価データ
            fundamentals_data: ファンダメンタルズデータ
            sentiment_data: センチメントデータ
            
        Returns:
            integrated_features: 統合された特徴量
        """
        integrated_features = self.multimodal_integrator.integrate_data(
            price_data, fundamentals_data, sentiment_data
        )
        return integrated_features
        
    def extract_self_supervised_features(self, X: np.ndarray) -> np.ndarray:
        """
        自己教師あり学習による特徴量抽出
        
        Args:
            X: 入力データ
            
        Returns:
            extracted_features: 抽出された特徴量
        """
        return self.self_supervised_model.extract_features(X)
        
    def apply_attention(self, X: np.ndarray) -> np.ndarray:
        """
        アテンション機構の適用
        
        Args:
            X: 入力データ
            
        Returns:
            attended_output: アテンション適用後の出力
        """
        return self.adaptive_attention(X)
        
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32) -> Dict:
        """
        モデルの訓練
        
        Args:
            X_train: 訓練データ
            y_train: 訓練ラベル
            X_val: 検証データ
            y_val: 検証ラベル
            epochs: エポック数
            batch_size: バッチサイズ
            
        Returns:
            history: 訓練履歴
        """
        logger.info("統合深層学習モデルの訓練開始")
        
        # 自己教師あり学習による特徴量抽出
        logger.info("自己教師あり学習による特徴量抽出中...")
        self_supervised_features = self.extract_self_supervised_features(X_train)
        
        # 訓練データの前処理（アテンション適用）
        logger.info("アテンション機構適用中...")
        attended_features = self.apply_attention(X_train)
        
        # 統合特徴量の作成
        integrated_features = np.concatenate([
            X_train.reshape(X_train.shape[0], -1),  # 元の特徴量
            self_supervised_features,                # 自己教師あり特徴量
            np.expand_dims(attended_features, axis=1).repeat(X_train.shape[0], axis=0)  # アテンション特徴量
        ], axis=1)
        
        # 独自モデルの訓練
        logger.info("独自深層学習モデル訓練中...")
        history = self.custom_model.train(
            integrated_features, 
            y_train, 
            X_val, 
            y_val, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        self.is_trained = True
        self.training_history = history.history if hasattr(history, 'history') else {}
        
        logger.info("統合深層学習モデルの訓練完了")
        return self.training_history
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測の実行
        
        Args:
            X: 入力データ
            
        Returns:
            predictions: 予測結果
        """
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません。train()を呼び出してください。")
            
        # 自己教師あり学習による特徴量抽出
        self_supervised_features = self.extract_self_supervised_features(X)
        
        # アテンション適用
        attended_features = self.apply_attention(X)
        
        # 統合特徴量の作成
        integrated_features = np.concatenate([
            X.reshape(X.shape[0], -1),
            self_supervised_features,
            np.expand_dims(attended_features, axis=1).repeat(X.shape[0], axis=0)
        ], axis=1)
        
        # 予測
        predictions = self.custom_model.predict(integrated_features)
        return predictions
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        モデルの評価
        
        Args:
            X_test: テストデータ
            y_test: テストラベル
            
        Returns:
            metrics: 評価指標
        """
        predictions = self.predict(X_test)
        
        mse = np.mean((y_test - predictions) ** 2)
        mae = np.mean(np.abs(y_test - predictions))
        r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"モデル評価結果: {metrics}")
        return metrics
        
    def setup_interpretability(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        解釈性分析のセットアップ
        
        Args:
            X_train: 訓練データ
            feature_names: 特徴量名
        """
        self.interpretability = ModelInterpretability(
            self.custom_model.model,  # Kerasモデルを渡す
            X_train,
            feature_names=feature_names
        )
        self.interpretability.setup_explainer(method='kernel')  # 一般的に使用される方法
        
    def explain_prediction(self, X_sample: np.ndarray, sample_index: int = 0) -> np.ndarray:
        """
        予測の解釈
        
        Args:
            X_sample: サンプルデータ
            sample_index: 解釈するサンプルのインデックス
            
        Returns:
            shap_values: SHAP値
        """
        if self.interpretability is None:
            raise ValueError("解釈性分析がセットアップされていません。setup_interpretability()を呼び出してください。")
            
        shap_values = self.interpretability.calculate_shap_values(X_sample)
        return shap_values
        
    def optimize_trading_strategy(self, symbol: str, period: str = '1y', timesteps: int = 10000):
        """
        強化学習による取引戦略の最適化
        
        Args:
            symbol: 株式銘柄
            period: 期間
            timesteps: 訓練ステップ数
        """
        logger.info(f"{symbol}の取引戦略最適化開始")
        self.reinforcement_trader.train_model(symbol, period, timesteps)
        logger.info(f"{symbol}の取引戦略最適化完了")
        
    def get_trading_recommendation(self, symbol: str, period: str = '6m') -> float:
        """
        取引推奨スコアの取得
        
        Args:
            symbol: 株式銘柄
            period: 期間
            
        Returns:
            recommendation_score: 推奨スコア（-1.0～1.0）
        """
        if self.reinforcement_trader.model is None:
            raise ValueError("取引戦略が最適化されていません。optimize_trading_strategy()を呼び出してください。")
            
        reward = self.reinforcement_trader.evaluate_strategy(symbol, period)
        # 報酬を-1.0～1.0の範囲に正規化
        recommendation_score = np.tanh(reward)  # tanhで-1.0～1.0の範囲に収束
        return recommendation_score

# 例として、統合モデルを使用するためのユーティリティ関数
def create_integrated_predictor_from_config(config: Dict) -> IntegratedDeepPredictor:
    """
    設定から統合予測器を作成
    
    Args:
        config: 設定辞書
        
    Returns:
        IntegratedDeepPredictor: 統合予測器
    """
    input_shape = tuple(config.get('input_shape', (50, 10)))
    attention_units = config.get('attention_units', 64)
    encoding_dim = config.get('encoding_dim', 32)
    learning_rate = config.get('learning_rate', 0.001)
    
    return IntegratedDeepPredictor(
        input_shape=input_shape,
        attention_units=attention_units,
        encoding_dim=encoding_dim,
        learning_rate=learning_rate
    )