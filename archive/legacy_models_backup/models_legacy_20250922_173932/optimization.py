"""Model optimization and hyperparameter tuning."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .base import StockPredictor, PredictionResult

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """ハイパーパラメータ最適化器"""

    def __init__(self):
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = 0.0

    def optimize_xgboost_params(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """XGBoostパラメータ最適化"""
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 0.9, 1.0],
        }

        best_params = {}
        best_score = float("-inf")

        try:
            import xgboost as xgb
            from sklearn.metrics import mean_squared_error

            for n_est in param_grid["n_estimators"]:
                for max_d in param_grid["max_depth"]:
                    for lr in param_grid["learning_rate"]:
                        for subsample in param_grid["subsample"]:
                            params = {
                                "n_estimators": n_est,
                                "max_depth": max_d,
                                "learning_rate": lr,
                                "subsample": subsample,
                                "random_state": 42,
                            }

                            model = xgb.XGBRegressor(**params)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_val)
                            score = -mean_squared_error(
                                y_val, y_pred
                            )  # Negative MSE for maximization

                            self.optimization_history.append(
                                {
                                    "params": params.copy(),
                                    "score": score,
                                    "timestamp": datetime.now(),
                                }
                            )

                            if score > best_score:
                                best_score = score
                                best_params = params.copy()

            self.best_params = best_params
            self.best_score = best_score

            logger.info(f"Best XGBoost params: {best_params}, Score: {best_score}")
            return best_params

        except ImportError:
            logger.warning("XGBoost not available for optimization")
            return {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            }

    def optimize_lightgbm_params(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """LightGBMパラメータ最適化"""
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
            "num_leaves": [31, 50, 100],
        }

        best_params = {}
        best_score = float("-inf")

        try:
            import lightgbm as lgb
            from sklearn.metrics import mean_squared_error

            for n_est in param_grid["n_estimators"]:
                for max_d in param_grid["max_depth"]:
                    for lr in param_grid["learning_rate"]:
                        for num_leaves in param_grid["num_leaves"]:
                            params = {
                                "n_estimators": n_est,
                                "max_depth": max_d,
                                "learning_rate": lr,
                                "num_leaves": num_leaves,
                                "random_state": 42,
                                "verbose": -1,
                            }

                            model = lgb.LGBMRegressor(**params)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_val)
                            score = -mean_squared_error(y_val, y_pred)

                            self.optimization_history.append(
                                {
                                    "params": params.copy(),
                                    "score": score,
                                    "timestamp": datetime.now(),
                                }
                            )

                            if score > best_score:
                                best_score = score
                                best_params = params.copy()

            logger.info(f"Best LightGBM params: {best_params}, Score: {best_score}")
            return best_params

        except ImportError:
            logger.warning("LightGBM not available for optimization")
            return {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            }

    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化結果のサマリーを取得"""
        if not self.optimization_history:
            return {}

        scores = [entry["score"] for entry in self.optimization_history]
        return {
            "total_trials": len(self.optimization_history),
            "best_score": self.best_score,
            "best_params": self.best_params,
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "score_improvement": self.best_score - min(scores) if scores else 0,
        }


class MetaLearningOptimizer:
    """メタ学習最適化器"""

    def __init__(self):
        self.model_performances: Dict[str, List[float]] = {}
        self.best_model_combinations: List[Dict[str, Any]] = []

    def record_model_performance(self, model_name: str, performance: float):
        """モデルパフォーマンスを記録"""
        if model_name not in self.model_performances:
            self.model_performances[model_name] = []

        self.model_performances[model_name].append(performance)

        # 最近のパフォーマンスのみ保持（最大100件）
        if len(self.model_performances[model_name]) > 100:
            self.model_performances[model_name] = self.model_performances[model_name][
                -100:
            ]

    def analyze_model_performance_patterns(self) -> Dict[str, Dict[str, float]]:
        """モデルパフォーマンスのパターン分析"""
        analysis = {}

        for model_name, performances in self.model_performances.items():
            if len(performances) < 5:  # 最低5回の記録が必要
                continue

            recent_perfs = performances[-10:]  # 最近10回
            older_perfs = performances[:-10] if len(performances) > 10 else []

            analysis[model_name] = {
                "mean_performance": np.mean(performances),
                "std_performance": np.std(performances),
                "recent_mean": np.mean(recent_perfs),
                "trend": self._calculate_trend(performances),
                "consistency": 1.0 / (1.0 + np.std(performances)),  # 一貫性指標
                "recent_improvement": (
                    np.mean(recent_perfs) - np.mean(older_perfs) if older_perfs else 0
                ),
            }

        return analysis

    def _calculate_trend(self, performances: List[float]) -> float:
        """パフォーマンストレンドを計算"""
        if len(performances) < 3:
            return 0.0

        x = np.arange(len(performances))
        y = np.array(performances)

        # 線形回帰の傾きを計算
        slope = np.polyfit(x, y, 1)[0]
        return slope

    def recommend_best_models(self, top_k: int = 3) -> List[str]:
        """最適なモデルを推奨"""
        analysis = self.analyze_model_performance_patterns()

        if not analysis:
            return []

        # 総合スコアを計算（パフォーマンス + 一貫性 + トレンド）
        model_scores = {}
        for model_name, stats in analysis.items():
            score = (
                stats["mean_performance"] * 0.4
                + stats["consistency"] * 0.3
                + stats["recent_improvement"] * 0.2
                + max(0, stats["trend"]) * 0.1  # 正のトレンドのみボーナス
            )
            model_scores[model_name] = score

        # スコア順にソート
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        return [model_name for model_name, _ in sorted_models[:top_k]]

    def optimize_ensemble_weights(
        self, model_predictions: Dict[str, List[float]], true_values: List[float]
    ) -> Dict[str, float]:
        """アンサンブル重みの最適化"""
        from sklearn.linear_model import LinearRegression

        # 予測値を行列に変換
        model_names = list(model_predictions.keys())
        X = np.array([model_predictions[name] for name in model_names]).T
        y = np.array(true_values)

        # 線形回帰で最適重みを計算
        reg = LinearRegression(positive=True, fit_intercept=False)
        reg.fit(X, y)

        weights = reg.coef_
        weights = weights / np.sum(weights)  # 正規化

        return dict(zip(model_names, weights))

    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """メタ学習の洞察を取得"""
        analysis = self.analyze_model_performance_patterns()
        best_models = self.recommend_best_models()

        return {
            "total_models_analyzed": len(self.model_performances),
            "models_with_sufficient_data": len(analysis),
            "best_models": best_models,
            "performance_analysis": analysis,
            "recommendations": {
                "most_consistent": max(
                    analysis.items(),
                    key=lambda x: x[1]["consistency"],
                    default=(None, {}),
                )[0],
                "most_improving": max(
                    analysis.items(),
                    key=lambda x: x[1]["recent_improvement"],
                    default=(None, {}),
                )[0],
                "highest_performance": max(
                    analysis.items(),
                    key=lambda x: x[1]["mean_performance"],
                    default=(None, {}),
                )[0],
            },
        }

    def update_best_combinations(self, combination: Dict[str, Any], performance: float):
        """最良の組み合わせを更新"""
        self.best_model_combinations.append(
            {
                "combination": combination.copy(),
                "performance": performance,
                "timestamp": datetime.now(),
            }
        )

        # パフォーマンス順にソート
        self.best_model_combinations.sort(key=lambda x: x["performance"], reverse=True)

        # 上位10個のみ保持
        self.best_model_combinations = self.best_model_combinations[:10]
