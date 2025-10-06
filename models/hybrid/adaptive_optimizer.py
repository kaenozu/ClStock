#!/usr/bin/env python3
"""学習型パフォーマンス最適化システム
システムの使用パターンを学習して自動最適化するAIエンジン
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np

# PredictionModeは文字列として処理（循環インポート回避）


@dataclass
class UsagePattern:
    """使用パターンデータクラス"""

    frequent_symbols: List[str]
    time_based_preferences: Dict[str, str]
    mode_performance: Dict[str, Dict[str, float]]
    symbol_mode_correlation: Dict[str, str]
    peak_usage_hours: List[int]
    average_session_length: float


class UsagePatternAnalyzer:
    """使用パターン分析器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze(self, prediction_history: List[Dict]) -> UsagePattern:
        """使用パターン分析"""
        if not prediction_history:
            return self._default_pattern()

        # 頻繁に使用される銘柄
        frequent_symbols = self._analyze_frequent_symbols(prediction_history)

        # 時間帯別の好み
        time_based_preferences = self._analyze_time_preferences(prediction_history)

        # モード別パフォーマンス
        mode_performance = self._analyze_mode_performance(prediction_history)

        # 銘柄とモードの相関
        symbol_mode_correlation = self._analyze_symbol_mode_correlation(
            prediction_history,
        )

        # ピーク使用時間
        peak_usage_hours = self._analyze_peak_hours(prediction_history)

        # 平均セッション長
        average_session_length = self._analyze_session_length(prediction_history)

        return UsagePattern(
            frequent_symbols=frequent_symbols,
            time_based_preferences=time_based_preferences,
            mode_performance=mode_performance,
            symbol_mode_correlation=symbol_mode_correlation,
            peak_usage_hours=peak_usage_hours,
            average_session_length=average_session_length,
        )

    def _analyze_frequent_symbols(
        self, history: List[Dict], top_n: int = 10,
    ) -> List[str]:
        """頻繁使用銘柄分析"""
        symbol_counts = Counter([h["symbol"] for h in history])
        return [symbol for symbol, count in symbol_counts.most_common(top_n)]

    def _analyze_time_preferences(self, history: List[Dict]) -> Dict[str, str]:
        """時間帯別モード好み分析"""
        time_mode_counts = defaultdict(Counter)

        for h in history:
            hour = (
                h["timestamp"].hour
                if isinstance(h["timestamp"], datetime)
                else datetime.now().hour
            )
            time_slot = self._get_time_slot(hour)
            mode = h["mode"]
            time_mode_counts[time_slot][mode] += 1

        preferences = {}
        for time_slot, mode_counts in time_mode_counts.items():
            if mode_counts:
                most_used_mode = mode_counts.most_common(1)[0][0]
                try:
                    preferences[time_slot] = most_used_mode
                except ValueError:
                    preferences[time_slot] = "auto"

        return preferences

    def _get_time_slot(self, hour: int) -> str:
        """時間帯分類"""
        if 9 <= hour <= 11:
            return "morning_session"
        if 12 <= hour <= 14:
            return "afternoon_session"
        if 15 <= hour <= 17:
            return "closing_session"
        if 18 <= hour <= 22:
            return "evening_analysis"
        return "off_hours"

    def _analyze_mode_performance(
        self, history: List[Dict],
    ) -> Dict[str, Dict[str, float]]:
        """モード別パフォーマンス分析"""
        mode_stats = defaultdict(list)

        for h in history:
            mode = h["mode"]
            mode_stats[mode].append(
                {
                    "prediction_time": h["prediction_time"],
                    "confidence": h["confidence"],
                    "accuracy": h["accuracy"],
                },
            )

        performance = {}
        for mode, stats in mode_stats.items():
            if stats:
                performance[mode] = {
                    "avg_prediction_time": np.mean(
                        [s["prediction_time"] for s in stats],
                    ),
                    "avg_confidence": np.mean([s["confidence"] for s in stats]),
                    "avg_accuracy": np.mean([s["accuracy"] for s in stats]),
                    "consistency": (
                        1.0
                        - np.std([s["confidence"] for s in stats])
                        / np.mean([s["confidence"] for s in stats])
                        if np.mean([s["confidence"] for s in stats]) > 0
                        else 0.5
                    ),
                }

        return performance

    def _analyze_symbol_mode_correlation(self, history: List[Dict]) -> Dict[str, str]:
        """銘柄とモードの相関分析"""
        symbol_mode_counts = defaultdict(Counter)

        for h in history:
            symbol = h["symbol"]
            mode = h["mode"]
            symbol_mode_counts[symbol][mode] += 1

        correlations = {}
        for symbol, mode_counts in symbol_mode_counts.items():
            if mode_counts:
                best_mode = mode_counts.most_common(1)[0][0]
                try:
                    correlations[symbol] = best_mode
                except ValueError:
                    correlations[symbol] = "auto"

        return correlations

    def _analyze_peak_hours(self, history: List[Dict]) -> List[int]:
        """ピーク使用時間分析"""
        hour_counts = Counter()

        for h in history:
            hour = (
                h["timestamp"].hour
                if isinstance(h["timestamp"], datetime)
                else datetime.now().hour
            )
            hour_counts[hour] += 1

        # 上位3時間帯
        peak_hours = [hour for hour, count in hour_counts.most_common(3)]
        return sorted(peak_hours)

    def _analyze_session_length(self, history: List[Dict]) -> float:
        """平均セッション長分析"""
        if len(history) < 2:
            return 30.0  # デフォルト30分

        timestamps = [
            h["timestamp"] for h in history if isinstance(h["timestamp"], datetime)
        ]
        if len(timestamps) < 2:
            return 30.0

        timestamps.sort()
        sessions = []
        current_session_start = timestamps[0]

        for i in range(1, len(timestamps)):
            time_gap = (
                timestamps[i] - timestamps[i - 1]
            ).total_seconds() / 60  # 分単位

            if time_gap > 30:  # 30分以上の間隔でセッション区切り
                session_length = (
                    timestamps[i - 1] - current_session_start
                ).total_seconds() / 60
                sessions.append(session_length)
                current_session_start = timestamps[i]

        # 最後のセッション
        if timestamps:
            session_length = (
                timestamps[-1] - current_session_start
            ).total_seconds() / 60
            sessions.append(session_length)

        return np.mean(sessions) if sessions else 30.0

    def _default_pattern(self) -> UsagePattern:
        """デフォルト使用パターン"""
        return UsagePattern(
            frequent_symbols=["6758.T", "7203.T", "8306.T"],
            time_based_preferences={
                "morning_session": "speed",
                "afternoon_session": "balanced",
                "closing_session": "accuracy",
                "evening_analysis": "research",
                "off_hours": "auto",
            },
            mode_performance={},
            symbol_mode_correlation={},
            peak_usage_hours=[9, 13, 15],
            average_session_length=45.0,
        )


class PerformanceMonitor:
    """パフォーマンス監視器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = []
        self.logger = logging.getLogger(__name__)

    def record_performance(self, metrics: Dict[str, float]):
        """パフォーマンス記録"""
        timestamp = datetime.now()
        metrics["timestamp"] = timestamp
        self.metrics_history.append(metrics)

        # ウィンドウサイズ制限
        if len(self.metrics_history) > self.window_size:
            self.metrics_history = self.metrics_history[-self.window_size :]

    def get_performance_trends(self) -> Dict[str, Any]:
        """パフォーマンストレンド取得"""
        if len(self.metrics_history) < 10:
            return {"insufficient_data": True}

        recent_metrics = self.metrics_history[-50:]  # 直近50件

        trends = {}
        for key in ["prediction_time", "confidence", "cache_hit_rate"]:
            if key in recent_metrics[0]:
                values = [m[key] for m in recent_metrics if key in m]
                if values:
                    trends[f"{key}_trend"] = self._calculate_trend(values)
                    trends[f"{key}_current"] = np.mean(values[-10:])  # 直近10件の平均

        return trends

    def _calculate_trend(self, values: List[float]) -> float:
        """トレンド計算"""
        if len(values) < 5:
            return 0.0

        x = np.arange(len(values))
        try:
            slope, _ = np.polyfit(x, values, 1)
            return float(slope)
        except Exception:
            return 0.0


class OptimizationEngine:
    """最適化エンジン"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimization_history = []

    def generate_optimizations(
        self, usage_pattern: UsagePattern, performance_trends: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """最適化提案生成"""
        optimizations = []

        # 頻繁使用銘柄の事前ロード提案
        if usage_pattern.frequent_symbols:
            optimizations.append(
                {
                    "type": "preload_models",
                    "priority": "high",
                    "symbols": usage_pattern.frequent_symbols[:5],
                    "description": "Preload models for frequently used symbols",
                    "expected_improvement": 0.3,  # 30%高速化期待
                },
            )

        # 時間帯別デフォルトモード設定提案
        if usage_pattern.time_based_preferences:
            optimizations.append(
                {
                    "type": "time_based_defaults",
                    "priority": "medium",
                    "preferences": usage_pattern.time_based_preferences,
                    "description": "Set time-based default prediction modes",
                    "expected_improvement": 0.15,  # 15%効率化期待
                },
            )

        # パフォーマンス低下検出と対策提案
        if "prediction_time_trend" in performance_trends:
            if performance_trends["prediction_time_trend"] > 0.001:  # 予測時間増加傾向
                optimizations.append(
                    {
                        "type": "performance_tuning",
                        "priority": "high",
                        "action": "cache_optimization",
                        "description": "Optimize cache settings due to performance degradation",
                        "expected_improvement": 0.25,
                    },
                )

        # キャッシュヒット率改善提案
        if "cache_hit_rate_current" in performance_trends:
            if performance_trends["cache_hit_rate_current"] < 0.7:
                optimizations.append(
                    {
                        "type": "cache_tuning",
                        "priority": "medium",
                        "action": "extend_ttl",
                        "description": "Extend cache TTL to improve hit rate",
                        "expected_improvement": 0.2,
                    },
                )

        return optimizations


class AdaptivePerformanceOptimizer:
    """学習型パフォーマンス最適化システム

    特徴:
    - 使用パターンの自動学習
    - パフォーマンス監視と傾向分析
    - AI駆動の最適化提案
    - 継続的な改善サイクル
    """

    def __init__(self):
        self.usage_analyzer = UsagePatternAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        self.optimization_engine = OptimizationEngine()
        self.logger = logging.getLogger(__name__)

        # 最適化状態
        self.active_optimizations = {}
        self.optimization_effects = {}

        self.logger.info("AdaptivePerformanceOptimizer initialized")

    def learn_and_optimize(self, prediction_history: List[Dict]) -> Dict[str, Any]:
        """学習と最適化実行"""
        # 使用パターン学習
        usage_pattern = self.usage_analyzer.analyze(prediction_history)

        # パフォーマンストレンド分析
        performance_trends = self.performance_monitor.get_performance_trends()

        # 最適化提案生成
        optimizations = self.optimization_engine.generate_optimizations(
            usage_pattern, performance_trends,
        )

        # 最適化適用
        applied_optimizations = self._apply_optimizations(optimizations)

        # 結果レポート
        optimization_report = {
            "usage_pattern": {
                "frequent_symbols": usage_pattern.frequent_symbols,
                "peak_hours": usage_pattern.peak_usage_hours,
                "session_length": usage_pattern.average_session_length,
            },
            "performance_trends": performance_trends,
            "optimizations_applied": applied_optimizations,
            "expected_improvements": sum(
                [opt.get("expected_improvement", 0) for opt in applied_optimizations],
            ),
            "timestamp": datetime.now(),
        }

        self.logger.info(
            f"Optimization cycle completed: {len(applied_optimizations)} optimizations applied",
        )
        return optimization_report

    def _apply_optimizations(
        self, optimizations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """最適化適用"""
        applied = []

        for opt in optimizations:
            try:
                if self._should_apply_optimization(opt):
                    success = self._execute_optimization(opt)
                    if success:
                        applied.append(opt)
                        self.active_optimizations[opt["type"]] = opt
                        self.logger.info(f"Applied optimization: {opt['type']}")
            except Exception as e:
                self.logger.error(
                    f"Failed to apply optimization {opt['type']}: {e!s}",
                )

        return applied

    def _should_apply_optimization(self, optimization: Dict[str, Any]) -> bool:
        """最適化適用判定"""
        # 既に同じタイプの最適化が適用されているかチェック
        if optimization["type"] in self.active_optimizations:
            return False

        # 優先度チェック
        if optimization.get("priority") == "low":
            return False

        # 期待改善効果チェック
        if optimization.get("expected_improvement", 0) < 0.1:
            return False

        return True

    def _execute_optimization(self, optimization: Dict[str, Any]) -> bool:
        """最適化実行"""
        opt_type = optimization["type"]

        try:
            if opt_type == "preload_models":
                return self._preload_models(optimization["symbols"])
            if opt_type == "time_based_defaults":
                return self._configure_time_defaults(optimization["preferences"])
            if opt_type == "performance_tuning":
                return self._tune_performance(optimization["action"])
            if opt_type == "cache_tuning":
                return self._tune_cache(optimization["action"])
            self.logger.warning(f"Unknown optimization type: {opt_type}")
            return False

        except Exception as e:
            self.logger.error(f"Optimization execution failed: {e!s}")
            return False

    def _preload_models(self, symbols: List[str]) -> bool:
        """モデル事前ロード"""
        # 実際の実装では、頻繁使用銘柄のモデルをメモリに事前ロード
        self.logger.info(f"Preloading models for symbols: {symbols}")
        return True

    def _configure_time_defaults(self, preferences: Dict[str, str]) -> bool:
        """時間帯別デフォルト設定"""
        # 実際の実装では、システム設定を更新
        self.logger.info(f"Configuring time-based defaults: {preferences}")
        return True

    def _tune_performance(self, action: str) -> bool:
        """パフォーマンスチューニング"""
        # 実際の実装では、システムパラメータを調整
        self.logger.info(f"Performance tuning action: {action}")
        return True

    def _tune_cache(self, action: str) -> bool:
        """キャッシュチューニング"""
        # 実際の実装では、キャッシュ設定を調整
        self.logger.info(f"Cache tuning action: {action}")
        return True

    def record_performance_metrics(
        self, prediction_time: float, confidence: float, cache_hit: bool = False,
    ):
        """パフォーマンスメトリクス記録"""
        metrics = {
            "prediction_time": prediction_time,
            "confidence": confidence,
            "cache_hit_rate": 1.0 if cache_hit else 0.0,
        }

        self.performance_monitor.record_performance(metrics)

    def get_optimization_status(self) -> Dict[str, Any]:
        """最適化状況取得"""
        return {
            "active_optimizations": list(self.active_optimizations.keys()),
            "total_applied": len(self.active_optimizations),
            "performance_trends": self.performance_monitor.get_performance_trends(),
            "next_optimization_due": self._calculate_next_optimization_time(),
        }

    def _calculate_next_optimization_time(self) -> datetime:
        """次回最適化時間計算"""
        # 通常は1時間ごとに最適化チェック
        return datetime.now() + timedelta(hours=1)
