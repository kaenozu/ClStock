"""Model performance monitoring and tracking."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """パフォーマンス指標のデータクラス"""

    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]


class ModelPerformanceMonitor:
    """モデルパフォーマンス監視システム"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.performance_history: Dict[str, List[PerformanceMetric]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}

    def record_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction: float,
        actual: Optional[float] = None,
        confidence: float = 0.5,
    ):
        """予測結果を記録"""
        metric = PerformanceMetric(
            name=f"{model_name}_prediction",
            value=prediction,
            timestamp=datetime.now(),
            metadata={
                "model": model_name,
                "symbol": symbol,
                "confidence": confidence,
                "actual": actual,
            },
        )

        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        self.performance_history[model_name].append(metric)

        # 履歴サイズ制限
        if len(self.performance_history[model_name]) > self.max_history:
            self.performance_history[model_name] = self.performance_history[model_name][
                -self.max_history :
            ]

        # 実際の値がある場合、精度を計算
        if actual is not None:
            self._calculate_and_record_accuracy(model_name, prediction, actual)

    def _calculate_and_record_accuracy(
        self,
        model_name: str,
        prediction: float,
        actual: float,
    ):
        """精度を計算して記録"""
        # 方向精度（上昇/下降の予測精度）
        pred_direction = 1 if prediction > 50 else 0
        actual_direction = 1 if actual > 50 else 0
        direction_accuracy = 1.0 if pred_direction == actual_direction else 0.0

        # 絶対誤差
        absolute_error = abs(prediction - actual)

        # MAPE (Mean Absolute Percentage Error)
        mape = abs(prediction - actual) / max(abs(actual), 1e-8) * 100

        # 精度指標を記録
        accuracy_metrics = [
            PerformanceMetric(
                name=f"{model_name}_direction_accuracy",
                value=direction_accuracy,
                timestamp=datetime.now(),
                metadata={"prediction": prediction, "actual": actual},
            ),
            PerformanceMetric(
                name=f"{model_name}_absolute_error",
                value=absolute_error,
                timestamp=datetime.now(),
                metadata={"prediction": prediction, "actual": actual},
            ),
            PerformanceMetric(
                name=f"{model_name}_mape",
                value=mape,
                timestamp=datetime.now(),
                metadata={"prediction": prediction, "actual": actual},
            ),
        ]

        for metric in accuracy_metrics:
            if metric.name not in self.performance_history:
                self.performance_history[metric.name] = []

            self.performance_history[metric.name].append(metric)

            # 履歴サイズ制限
            if len(self.performance_history[metric.name]) > self.max_history:
                self.performance_history[metric.name] = self.performance_history[
                    metric.name
                ][-self.max_history :]

    def set_performance_threshold(
        self,
        metric_name: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """パフォーマンス閾値を設定"""
        self.thresholds[metric_name] = {}
        if min_value is not None:
            self.thresholds[metric_name]["min"] = min_value
        if max_value is not None:
            self.thresholds[metric_name]["max"] = max_value

    def check_performance_thresholds(self) -> List[Dict[str, Any]]:
        """パフォーマンス閾値をチェックしてアラートを生成"""
        new_alerts = []

        for metric_name, thresholds in self.thresholds.items():
            if metric_name not in self.performance_history:
                continue

            recent_metrics = self.performance_history[metric_name][-10:]  # 最近10件
            if not recent_metrics:
                continue

            recent_avg = np.mean([m.value for m in recent_metrics])

            # 最小値チェック
            if "min" in thresholds and recent_avg < thresholds["min"]:
                alert = {
                    "type": "threshold_violation",
                    "metric": metric_name,
                    "current_value": recent_avg,
                    "threshold": thresholds["min"],
                    "violation_type": "below_minimum",
                    "timestamp": datetime.now(),
                    "severity": "warning",
                }
                new_alerts.append(alert)

            # 最大値チェック
            if "max" in thresholds and recent_avg > thresholds["max"]:
                alert = {
                    "type": "threshold_violation",
                    "metric": metric_name,
                    "current_value": recent_avg,
                    "threshold": thresholds["max"],
                    "violation_type": "above_maximum",
                    "timestamp": datetime.now(),
                    "severity": "warning",
                }
                new_alerts.append(alert)

        self.alerts.extend(new_alerts)
        return new_alerts

    def get_performance_summary(self, model_name: str, days: int = 7) -> Dict[str, Any]:
        """指定期間のパフォーマンスサマリーを取得"""
        cutoff_time = datetime.now() - timedelta(days=days)

        summary = {"model_name": model_name, "period_days": days, "metrics": {}}

        for metric_name, metrics in self.performance_history.items():
            if not metric_name.startswith(model_name):
                continue

            # 期間内のメトリクスをフィルタ
            period_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            if not period_metrics:
                continue

            values = [m.value for m in period_metrics]

            metric_summary = {
                "count": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "latest": values[-1] if values else None,
                "trend": self._calculate_trend(values),
            }

            # 方向精度の場合は特別な処理
            if "direction_accuracy" in metric_name:
                metric_summary["accuracy_percentage"] = np.mean(values) * 100

            summary["metrics"][metric_name] = metric_summary

        return summary

    def _calculate_trend(self, values: List[float]) -> str:
        """トレンドを計算"""
        if len(values) < 3:
            return "insufficient_data"

        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return "improving"
        if slope < -0.01:
            return "declining"
        return "stable"

    def detect_performance_anomalies(
        self,
        model_name: str,
        window_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """パフォーマンス異常を検出"""
        anomalies = []

        for metric_name, metrics in self.performance_history.items():
            if not metric_name.startswith(model_name):
                continue

            if len(metrics) < window_size:
                continue

            values = [m.value for m in metrics[-window_size:]]
            mean_val = np.mean(values)
            std_val = np.std(values)

            # 最新値が2標準偏差を超えている場合は異常
            latest_value = values[-1]
            z_score = abs(latest_value - mean_val) / (std_val + 1e-8)

            if z_score > 2.0:
                anomaly = {
                    "metric": metric_name,
                    "latest_value": latest_value,
                    "mean_value": mean_val,
                    "z_score": z_score,
                    "timestamp": metrics[-1].timestamp,
                    "severity": "high" if z_score > 3.0 else "medium",
                }
                anomalies.append(anomaly)

        return anomalies

    def get_model_comparison(
        self,
        models: List[str],
        metric_type: str = "direction_accuracy",
    ) -> Dict[str, Any]:
        """モデル間のパフォーマンス比較"""
        comparison = {"metric_type": metric_type, "models": {}}

        for model in models:
            metric_name = f"{model}_{metric_type}"
            if metric_name in self.performance_history:
                recent_metrics = self.performance_history[metric_name][-30:]  # 最近30件
                if recent_metrics:
                    values = [m.value for m in recent_metrics]
                    comparison["models"][model] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "count": len(values),
                        "latest": values[-1],
                    }

        # ランキング作成
        if comparison["models"]:
            ranking = sorted(
                comparison["models"].items(),
                key=lambda x: x[1]["mean"],
                reverse=True,
            )
            comparison["ranking"] = [model for model, _ in ranking]

        return comparison

    def export_performance_data(self, model_name: str, format: str = "dict") -> Any:
        """パフォーマンスデータをエクスポート"""
        model_data = {}

        for metric_name, metrics in self.performance_history.items():
            if metric_name.startswith(model_name):
                metric_data = []
                for metric in metrics:
                    metric_data.append(
                        {
                            "timestamp": metric.timestamp.isoformat(),
                            "value": metric.value,
                            "metadata": metric.metadata,
                        },
                    )
                model_data[metric_name] = metric_data

        if format == "dataframe":
            try:
                import pandas as pd

                # DataFrameに変換する処理
                all_data = []
                for metric_name, metric_list in model_data.items():
                    for item in metric_list:
                        row = {
                            "metric_name": metric_name,
                            "timestamp": item["timestamp"],
                            "value": item["value"],
                        }
                        row.update(item["metadata"])
                        all_data.append(row)

                return pd.DataFrame(all_data)
            except ImportError:
                logger.warning("Pandas not available, returning dict format")

        return model_data

    def clear_old_data(self, days: int = 30):
        """古いデータをクリア"""
        cutoff_time = datetime.now() - timedelta(days=days)

        for metric_name in self.performance_history:
            self.performance_history[metric_name] = [
                m
                for m in self.performance_history[metric_name]
                if m.timestamp >= cutoff_time
            ]

        # 古いアラートもクリア
        self.alerts = [
            alert for alert in self.alerts if alert["timestamp"] >= cutoff_time
        ]

    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """監視ダッシュボード用のデータを取得"""
        dashboard = {
            "summary": {
                "total_metrics": len(self.performance_history),
                "total_alerts": len(self.alerts),
                "active_models": len(
                    set(
                        metric_name.split("_")[0]
                        for metric_name in self.performance_history.keys()
                    ),
                ),
            },
            "recent_alerts": sorted(
                self.alerts[-10:],
                key=lambda x: x["timestamp"],
                reverse=True,
            ),
            "model_status": {},
        }

        # 各モデルの状態
        models = set(
            metric_name.split("_")[0] for metric_name in self.performance_history.keys()
        )

        for model in models:
            summary = self.get_performance_summary(model, days=1)
            anomalies = self.detect_performance_anomalies(model)

            dashboard["model_status"][model] = {
                "status": "warning" if anomalies else "healthy",
                "recent_accuracy": (
                    summary["metrics"]
                    .get(f"{model}_direction_accuracy", {})
                    .get("accuracy_percentage", 0)
                ),
                "prediction_count": (
                    summary["metrics"].get(f"{model}_prediction", {}).get("count", 0)
                ),
                "anomalies_count": len(anomalies),
            }

        return dashboard
