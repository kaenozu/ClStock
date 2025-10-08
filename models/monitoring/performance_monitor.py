#!/usr/bin/env python3
"""パフォーマンス監視モジュール
モデル性能の監視・評価・アラートシステム
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ..core.interfaces import PredictionResult

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """モデル性能監視・評価システム"""

    def __init__(self, data_file: Optional[str] = None):
        self.performance_history = []
        self.alerts = []
        self.prediction_records = []
        self.data_file = Path(
            data_file or "models/saved_models/performance_history.json",
        )
        self.data_file.parent.mkdir(exist_ok=True)

        # パフォーマンス閾値
        self.thresholds = {
            "rmse": 15.0,
            "r2": 0.1,
            "direction_accuracy": 0.55,
            "mae": 10.0,
        }

    def record_prediction(self, result: PredictionResult) -> None:
        """予測結果記録"""
        prediction_record = {
            "timestamp": result.timestamp.isoformat(),
            "symbol": result.symbol,
            "prediction": result.prediction,
            "confidence": result.confidence,
            "accuracy": result.accuracy,
            "metadata": result.metadata,
        }
        self.prediction_records.append(prediction_record)

        # 定期的にファイルに保存
        if len(self.prediction_records) % 100 == 0:
            self.save_performance_data()

    def get_accuracy_metrics(self, period_days: int = 30) -> Dict[str, float]:
        """精度指標取得"""
        cutoff_date = datetime.now().timestamp() - (period_days * 24 * 3600)

        recent_records = [
            r
            for r in self.prediction_records
            if datetime.fromisoformat(r["timestamp"]).timestamp() > cutoff_date
        ]

        if not recent_records:
            return {"count": 0}

        accuracies = [r["accuracy"] for r in recent_records]
        confidences = [r["confidence"] for r in recent_records]
        predictions = [r["prediction"] for r in recent_records]

        return {
            "count": len(recent_records),
            "avg_accuracy": np.mean(accuracies),
            "avg_confidence": np.mean(confidences),
            "avg_prediction": np.mean(predictions),
            "std_accuracy": np.std(accuracies),
            "min_accuracy": np.min(accuracies),
            "max_accuracy": np.max(accuracies),
            "accuracy_trend": self._calculate_trend(accuracies),
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """パフォーマンスレポート生成"""
        if not self.performance_history:
            return {
                "status": "No performance data available",
                "summary": {},
                "alerts": self.alerts,
                "recommendations": [],
            }

        recent_performance = self.performance_history[-10:]  # 最新10件

        avg_metrics = {
            "rmse": np.mean([p["rmse"] for p in recent_performance]),
            "mae": np.mean([p["mae"] for p in recent_performance]),
            "r2_score": np.mean([p["r2_score"] for p in recent_performance]),
            "direction_accuracy": np.mean(
                [p["direction_accuracy"] for p in recent_performance],
            ),
        }

        # トレンド分析
        rmse_trend = self._calculate_trend([p["rmse"] for p in recent_performance])
        accuracy_trend = self._calculate_trend(
            [p["direction_accuracy"] for p in recent_performance],
        )

        # 全体的なパフォーマンス評価
        performance_grade = self._calculate_performance_grade(avg_metrics)

        # 推奨事項生成
        recommendations = self._generate_recommendations(
            avg_metrics,
            rmse_trend,
            accuracy_trend,
        )

        return {
            "status": "active",
            "summary": {
                "total_evaluations": len(self.performance_history),
                "avg_metrics": avg_metrics,
                "rmse_trend": rmse_trend,
                "accuracy_trend": accuracy_trend,
                "performance_grade": performance_grade,
                "active_alerts": len(
                    [a for a in self.alerts if self._is_recent_alert(a)],
                ),
            },
            "recent_performance": recent_performance[-5:],  # 最新5件の詳細
            "alerts": self.alerts[-10:],  # 最新10件のアラート
            "recommendations": recommendations,
            "prediction_stats": self.get_accuracy_metrics(),
        }

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
            if i == 0:
                mask = y_test <= quantiles[i]
            elif i == len(quantiles) - 1:
                mask = y_test > quantiles[i - 1]
            else:
                mask = (y_test > quantiles[i - 1]) & (y_test <= quantiles[i])

            if mask.sum() > 0:
                quantile_mse = mean_squared_error(y_test[mask], predictions[mask])
                quantile_performance[f"Q{i + 1}"] = quantile_mse

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
        alerts = []

        if performance_record["rmse"] > self.thresholds["rmse"]:
            alerts.append(
                f"High RMSE: {performance_record['rmse']:.4f} > {self.thresholds['rmse']}",
            )

        if performance_record["r2_score"] < self.thresholds["r2"]:
            alerts.append(
                f"Low R²: {performance_record['r2_score']:.4f} < {self.thresholds['r2']}",
            )

        if (
            performance_record["direction_accuracy"]
            < self.thresholds["direction_accuracy"]
        ):
            alerts.append(
                f"Low Direction Accuracy: {performance_record['direction_accuracy']:.4f} < {self.thresholds['direction_accuracy']}",
            )

        if performance_record["mae"] > self.thresholds["mae"]:
            alerts.append(
                f"High MAE: {performance_record['mae']:.4f} > {self.thresholds['mae']}",
            )

        if alerts:
            alert_record = {
                "timestamp": performance_record["timestamp"],
                "model_name": performance_record["model_name"],
                "alerts": alerts,
                "severity": self._calculate_alert_severity(alerts),
            }
            self.alerts.append(alert_record)

            logger.warning(
                f"Performance alerts for {performance_record['model_name']}: {alerts}",
            )

    def _calculate_trend(self, values: List[float]) -> str:
        """トレンド計算"""
        if len(values) < 2:
            return "insufficient_data"

        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        older_avg = np.mean(values[:-3]) if len(values) >= 6 else values[0]

        change_ratio = (recent_avg - older_avg) / older_avg if older_avg != 0 else 0

        if change_ratio > 0.05:
            return "improving"
        if change_ratio < -0.05:
            return "declining"
        return "stable"

    def _calculate_performance_grade(self, metrics: Dict[str, float]) -> str:
        """性能グレード計算"""
        score = 0

        # RMSE スコア (低いほど良い)
        if metrics["rmse"] < 5:
            score += 3
        elif metrics["rmse"] < 10:
            score += 2
        elif metrics["rmse"] < 15:
            score += 1

        # R² スコア (高いほど良い)
        if metrics["r2_score"] > 0.5:
            score += 3
        elif metrics["r2_score"] > 0.3:
            score += 2
        elif metrics["r2_score"] > 0.1:
            score += 1

        # 方向精度スコア (高いほど良い)
        if metrics["direction_accuracy"] > 0.7:
            score += 3
        elif metrics["direction_accuracy"] > 0.6:
            score += 2
        elif metrics["direction_accuracy"] > 0.55:
            score += 1

        # グレード判定
        if score >= 8:
            return "A+"
        if score >= 6:
            return "A"
        if score >= 4:
            return "B"
        if score >= 2:
            return "C"
        return "D"

    def _generate_recommendations(
        self,
        metrics: Dict[str, float],
        rmse_trend: str,
        accuracy_trend: str,
    ) -> List[str]:
        """推奨事項生成"""
        recommendations = []

        if metrics["rmse"] > self.thresholds["rmse"]:
            recommendations.append("モデルの正則化パラメータを調整してください")

        if metrics["r2_score"] < self.thresholds["r2"]:
            recommendations.append("特徴量エンジニアリングの改善を検討してください")

        if metrics["direction_accuracy"] < self.thresholds["direction_accuracy"]:
            recommendations.append("分類閾値の調整を検討してください")

        if rmse_trend == "declining":
            recommendations.append("モデルの再訓練が必要です")

        if accuracy_trend == "declining":
            recommendations.append("データ品質の確認を実施してください")

        if not recommendations:
            recommendations.append(
                "現在の性能は良好です。定期的な監視を継続してください",
            )

        return recommendations

    def _calculate_alert_severity(self, alerts: List[str]) -> str:
        """アラート重要度計算"""
        if len(alerts) >= 3:
            return "critical"
        if len(alerts) == 2:
            return "high"
        return "medium"

    def _is_recent_alert(self, alert: Dict, hours: int = 24) -> bool:
        """最近のアラートかチェック"""
        try:
            alert_time = datetime.fromisoformat(alert["timestamp"])
            return (datetime.now() - alert_time).total_seconds() < hours * 3600
        except (KeyError, ValueError):
            return False

    def save_performance_data(self):
        """性能データを保存"""
        try:
            data = {
                "performance_history": self.performance_history,
                "alerts": self.alerts,
                "prediction_records": self.prediction_records[-1000:],  # 最新1000件のみ
                "thresholds": self.thresholds,
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Performance data saved to {self.data_file}")

        except Exception as e:
            logger.error(f"Error saving performance data: {e!s}")

    def load_performance_data(self):
        """性能データを読み込み"""
        try:
            if self.data_file.exists():
                with open(self.data_file, encoding="utf-8") as f:
                    data = json.load(f)

                self.performance_history = data.get("performance_history", [])
                self.alerts = data.get("alerts", [])
                self.prediction_records = data.get("prediction_records", [])
                self.thresholds.update(data.get("thresholds", {}))

                logger.info("Performance data loaded successfully")
                return True

            return False

        except Exception as e:
            logger.error(f"Error loading performance data: {e!s}")
            return False

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
