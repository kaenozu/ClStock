"""
自動再学習システム
84.6%精度維持のための継続的モデル更新
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
import threading
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score
import joblib
import math

from data.stock_data import StockDataProvider
from models.stock_specific_predictor import StockSpecificPredictor
from models.predictor import StockPredictor
from config.settings import get_settings
from utils.exceptions import ModelTrainingError, DataFetchError

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """モデル性能監視システム"""

    def __init__(self):
        self.settings = get_settings()
        self.data_provider = StockDataProvider()
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.performance_thresholds = {
            "accuracy_decline": 0.05,  # 5%以上の精度低下
            "min_accuracy": 0.75,  # 最低75%精度
            "prediction_count": 50,  # 最低50回の予測が必要
        }

    def track_prediction(
        self,
        symbol: str,
        prediction: Dict[str, Any],
        actual_result: Optional[bool] = None,
    ) -> None:
        """予測結果を追跡"""
        if symbol not in self.performance_history:
            self.performance_history[symbol] = []

        tracking_record = {
            "timestamp": datetime.now(),
            "prediction": prediction,
            "actual_result": actual_result,
            "verified": actual_result is not None,
        }

        self.performance_history[symbol].append(tracking_record)

        # 古いレコードを削除（90日以上前）
        cutoff_date = datetime.now() - timedelta(days=90)
        self.performance_history[symbol] = [
            record
            for record in self.performance_history[symbol]
            if record["timestamp"] > cutoff_date
        ]

    def calculate_recent_accuracy(self, symbol: str, days: int = 30) -> Optional[float]:
        """最近の精度を計算"""
        if symbol not in self.performance_history:
            return None

        cutoff_date = datetime.now() - timedelta(days=days)
        recent_records = [
            record
            for record in self.performance_history[symbol]
            if record["timestamp"] > cutoff_date and record["verified"]
        ]

        if len(recent_records) < 10:  # 最低10回の検証データが必要
            return None

        correct_predictions = sum(
            1
            for record in recent_records
            if record["prediction"].get("signal", 0)
            == (1 if record["actual_result"] else 0)
        )

        return correct_predictions / len(recent_records)

    def detect_performance_decline(self, symbol: str) -> Dict[str, Any]:
        """性能低下を検出"""
        recent_accuracy = self.calculate_recent_accuracy(symbol, days=30)

        if recent_accuracy is None:
            return {
                "needs_retraining": False,
                "reason": "insufficient_data",
                "recent_accuracy": None,
            }

        # 基準精度（84.6%または過去の最高精度）
        baseline_accuracy = max(0.846, self._get_historical_best_accuracy(symbol))

        accuracy_decline = baseline_accuracy - recent_accuracy
        needs_retraining = (
            accuracy_decline > self.performance_thresholds["accuracy_decline"]
            or recent_accuracy < self.performance_thresholds["min_accuracy"]
        )

        return {
            "needs_retraining": needs_retraining,
            "reason": "performance_decline" if needs_retraining else "performance_ok",
            "recent_accuracy": recent_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_decline": accuracy_decline,
            "prediction_count": len(
                [
                    r
                    for r in self.performance_history.get(symbol, [])
                    if r["verified"]
                    and r["timestamp"] > datetime.now() - timedelta(days=30)
                ]
            ),
        }

    def _get_historical_best_accuracy(self, symbol: str) -> float:
        """過去の最高精度を取得"""
        if symbol not in self.performance_history:
            return 0.846

        # 30日間隔で精度を計算し、最高値を取得
        best_accuracy = 0.0
        for days_back in range(30, 365, 30):  # 30日から365日まで30日間隔
            accuracy = self.calculate_recent_accuracy(symbol, days_back)
            if accuracy and accuracy > best_accuracy:
                best_accuracy = accuracy

        return max(best_accuracy, 0.846)

    def get_retraining_candidates(self) -> List[Dict[str, Any]]:
        """再学習候補を取得"""
        candidates = []

        for symbol in self.settings.target_stocks.keys():
            decline_info = self.detect_performance_decline(symbol)
            if decline_info["needs_retraining"]:
                candidates.append(
                    {
                        "symbol": symbol,
                        "priority": self._calculate_retraining_priority(
                            symbol, decline_info
                        ),
                        **decline_info,
                    }
                )

        # 優先度順にソート
        candidates.sort(key=lambda x: x["priority"], reverse=True)
        return candidates

    def _calculate_retraining_priority(
        self, symbol: str, decline_info: Dict[str, Any]
    ) -> float:
        """再学習優先度を計算"""
        accuracy_factor = max(0, 1.0 - decline_info.get("recent_accuracy", 0.5))
        decline_factor = decline_info.get("accuracy_decline", 0) * 2
        data_factor = min(1.0, decline_info.get("prediction_count", 0) / 100)

        return accuracy_factor + decline_factor + data_factor


class DataDriftDetector:
    """データドリフト検出システム"""

    def __init__(self):
        self.data_provider = StockDataProvider()
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

    def establish_baseline(self, symbol: str, period: str = "6mo") -> Dict[str, float]:
        """ベースライン統計を確立"""
        try:
            data = self.data_provider.get_stock_data(symbol, period)

            price_series = data["Close"].pct_change(fill_method=None)
            volume_rolling = data["Volume"].rolling(20).mean()

            start_index = 20 if len(volume_rolling) > 20 else -1
            baseline_volume = volume_rolling.iloc[start_index]
            latest_volume = volume_rolling.iloc[-1]

            stats = {
                "volatility": self._to_float(price_series.std(skipna=True)),
                "avg_volume": self._to_float(data["Volume"].mean(skipna=True)),
                "price_trend": self._calculate_price_trend(data["Close"]),
                "volume_trend": self._calculate_ratio_change(
                    baseline_volume, latest_volume
                ),
            }

            self.baseline_stats[symbol] = stats
            logger.info(f"ベースライン確立: {symbol}")
            return stats

        except Exception as e:
            logger.error(f"ベースライン確立エラー {symbol}: {e}")
            return {}

    def detect_drift(self, symbol: str, current_period: str = "1mo") -> Dict[str, Any]:
        """データドリフトを検出"""
        try:
            if symbol not in self.baseline_stats:
                self.establish_baseline(symbol)

            current_data = self.data_provider.get_stock_data(symbol, current_period)

            current_stats = {
                "volatility": self._to_float(
                    current_data["Close"].pct_change(fill_method=None).std(skipna=True)
                ),
                "avg_volume": self._to_float(current_data["Volume"].mean(skipna=True)),
                "price_trend": self._calculate_price_trend(current_data["Close"]),
            }

            baseline = self.baseline_stats[symbol]

            # ドリフト率を計算
            volatility_drift = self._safe_relative_change(
                baseline.get("volatility", 0.0), current_stats["volatility"]
            )
            volume_drift = self._safe_relative_change(
                baseline.get("avg_volume", 0.0), current_stats["avg_volume"]
            )

            # ドリフト判定（30%以上の変化）
            significant_drift = volatility_drift > 0.3 or volume_drift > 0.3

            return {
                "symbol": symbol,
                "has_drift": bool(significant_drift),
                "volatility_drift": volatility_drift,
                "volume_drift": volume_drift,
                "current_stats": current_stats,
                "baseline_stats": baseline,
                "recommendation": "retrain" if significant_drift else "continue",
            }

        except Exception as e:
            logger.error(f"ドリフト検出エラー {symbol}: {e}")
            return {"symbol": symbol, "has_drift": False, "error": str(e)}

    @staticmethod
    def _safe_relative_change(baseline_value: float, current_value: float) -> float:
        """相対変化量を安全に計算（0除算を防止）"""

        if baseline_value is None or (isinstance(baseline_value, float) and math.isnan(baseline_value)):
            baseline_value = 0.0

        if current_value is None or (isinstance(current_value, float) and math.isnan(current_value)):
            current_value = 0.0

        if baseline_value == 0:
            return float("inf") if current_value != 0 else 0.0

        return abs(current_value - baseline_value) / abs(baseline_value)

    @staticmethod
    def _calculate_price_trend(close_series: pd.Series) -> float:
        """価格トレンドを安全に計算"""

        if close_series.empty:
            return 0.0

        start_price = close_series.iloc[0]
        end_price = close_series.iloc[-1]

        if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
            return 0.0

        return float((end_price / start_price) - 1)

    @staticmethod
    def _calculate_ratio_change(baseline_value: float, current_value: float) -> float:
        """比率変化を安全に計算"""

        if baseline_value is None or (isinstance(baseline_value, float) and math.isnan(baseline_value)):
            return 0.0

        if current_value is None or (isinstance(current_value, float) and math.isnan(current_value)):
            return 0.0

        if baseline_value == 0:
            return 0.0

        return float((current_value / baseline_value) - 1)

    @staticmethod
    def _to_float(value: Optional[float]) -> float:
        """NaN安全なfloat変換"""

        if value is None:
            return 0.0

        if isinstance(value, float) and math.isnan(value):
            return 0.0

        return float(value)


class AutoRetrainingScheduler:
    """自動再学習スケジューラー"""

    def __init__(self):
        self.settings = get_settings()
        self.performance_monitor = ModelPerformanceMonitor()
        self.drift_detector = DataDriftDetector()
        self.stock_predictor = StockSpecificPredictor()
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None

        # 再学習設定
        self.retraining_config = {
            "check_interval_hours": 24,  # 24時間ごとにチェック
            "max_concurrent_retraining": 3,  # 同時に3銘柄まで再学習
            "retraining_data_period": "2y",  # 2年分のデータで再学習
            "backup_models": True,  # 既存モデルをバックアップ
        }

    def start_scheduler(self) -> None:
        """スケジューラー開始"""
        if self.is_running:
            logger.warning("スケジューラーは既に実行中です")
            return

        self.is_running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop, daemon=True
        )
        self.scheduler_thread.start()
        logger.info("自動再学習スケジューラー開始")

    def stop_scheduler(self) -> None:
        """スケジューラー停止"""
        self.is_running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=30)
        logger.info("自動再学習スケジューラー停止")

    def _scheduler_loop(self) -> None:
        """スケジューラーメインループ"""
        while self.is_running:
            try:
                logger.info("定期チェック開始")

                # 再学習候補を取得
                candidates = self.performance_monitor.get_retraining_candidates()

                # データドリフトもチェック
                drift_candidates = self._check_data_drift()

                # 候補を統合
                all_candidates = self._merge_candidates(candidates, drift_candidates)

                if all_candidates:
                    logger.info(f"再学習候補: {len(all_candidates)}銘柄")
                    self._execute_retraining(all_candidates)
                else:
                    logger.info("再学習が必要な銘柄はありません")

                # 次のチェックまで待機
                time.sleep(self.retraining_config["check_interval_hours"] * 3600)

            except Exception as e:
                logger.error(f"スケジューラーエラー: {e}")
                time.sleep(3600)  # エラー時は1時間後に再試行

    def _check_data_drift(self) -> List[Dict[str, Any]]:
        """全銘柄のデータドリフトをチェック"""
        drift_candidates = []

        for symbol in list(self.settings.target_stocks.keys())[
            :10
        ]:  # 最初の10銘柄でテスト
            drift_result = self.drift_detector.detect_drift(symbol)
            if drift_result.get("has_drift", False):
                drift_candidates.append(
                    {
                        "symbol": symbol,
                        "reason": "data_drift",
                        "priority": 0.8,  # ドリフトは高優先度
                        "drift_info": drift_result,
                    }
                )

        return drift_candidates

    def _merge_candidates(
        self, performance_candidates: List[Dict], drift_candidates: List[Dict]
    ) -> List[Dict]:
        """候補リストを統合"""
        all_candidates = {}

        # 性能低下候補
        for candidate in performance_candidates:
            symbol = candidate["symbol"]
            all_candidates[symbol] = candidate

        # ドリフト候補（既存がある場合は優先度を加算）
        for candidate in drift_candidates:
            symbol = candidate["symbol"]
            if symbol in all_candidates:
                all_candidates[symbol]["priority"] += candidate["priority"]
                all_candidates[symbol]["drift_info"] = candidate["drift_info"]
            else:
                all_candidates[symbol] = candidate

        # 優先度順にソート
        sorted_candidates = sorted(
            all_candidates.values(), key=lambda x: x["priority"], reverse=True
        )

        # 最大同時再学習数に制限
        max_concurrent = self.retraining_config["max_concurrent_retraining"]
        return sorted_candidates[:max_concurrent]

    def _execute_retraining(self, candidates: List[Dict[str, Any]]) -> None:
        """再学習を実行"""
        logger.info(f"再学習実行開始: {len(candidates)}銘柄")

        # 並列実行
        max_workers = min(
            len(candidates), self.retraining_config["max_concurrent_retraining"]
        )

        if max_workers <= 0:
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._retrain_single_model, candidate): candidate[
                    "symbol"
                ]
                for candidate in candidates
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=3600)  # 1時間タイムアウト
                    if result["success"]:
                        logger.info(
                            f"✅ {symbol} 再学習成功: 精度 {result['new_accuracy']:.3f}"
                        )
                    else:
                        logger.error(f"❌ {symbol} 再学習失敗: {result['error']}")
                except Exception as e:
                    logger.error(f"❌ {symbol} 再学習例外: {e}")

    def _retrain_single_model(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """単一モデルの再学習"""
        symbol = candidate["symbol"]

        try:
            logger.info(f"再学習開始: {symbol} (理由: {candidate['reason']})")

            # 既存モデルのバックアップ
            if self.retraining_config["backup_models"]:
                self._backup_existing_model(symbol)

            # 新しいモデルを訓練
            training_result = self.stock_predictor.train_symbol_model(
                symbol, self.retraining_config["retraining_data_period"]
            )

            new_accuracy = training_result["accuracy"]

            # 改善確認
            old_accuracy = candidate.get("recent_accuracy", 0.5)
            improvement = new_accuracy - old_accuracy

            return {
                "symbol": symbol,
                "success": True,
                "new_accuracy": new_accuracy,
                "old_accuracy": old_accuracy,
                "improvement": improvement,
                "training_result": training_result,
            }

        except Exception as e:
            logger.error(f"再学習エラー {symbol}: {e}")
            return {"symbol": symbol, "success": False, "error": str(e)}

    def _backup_existing_model(self, symbol: str) -> None:
        """既存モデルをバックアップ"""
        try:
            model_dir = "models/stock_specific"
            backup_dir = "models/stock_specific/backup"
            os.makedirs(backup_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            source_path = os.path.join(model_dir, f"{symbol}_model.pkl")
            backup_path = os.path.join(backup_dir, f"{symbol}_model_{timestamp}.pkl")

            if os.path.exists(source_path):
                import shutil

                shutil.copy2(source_path, backup_path)
                logger.info(f"モデルバックアップ: {symbol}")

        except Exception as e:
            logger.warning(f"バックアップエラー {symbol}: {e}")

    def manual_retrain(
        self, symbols: List[str], reason: str = "manual"
    ) -> Dict[str, Any]:
        """手動再学習"""
        logger.info(f"手動再学習開始: {symbols}")

        candidates = [
            {"symbol": symbol, "reason": reason, "priority": 1.0} for symbol in symbols
        ]

        self._execute_retraining(candidates)

        return {
            "status": "completed",
            "symbols": symbols,
            "reason": reason,
            "timestamp": datetime.now(),
        }

    def get_system_status(self) -> Dict[str, Any]:
        """システム状態を取得"""
        return {
            "scheduler_running": self.is_running,
            "config": self.retraining_config,
            "performance_tracking": {
                "symbols_tracked": len(self.performance_monitor.performance_history),
                "total_predictions": sum(
                    len(history)
                    for history in self.performance_monitor.performance_history.values()
                ),
            },
            "drift_baselines": len(self.drift_detector.baseline_stats),
            "last_check": datetime.now(),  # 実際の実装では最後のチェック時刻を保存
        }


class RetrainingOrchestrator:
    """再学習統合管理システム"""

    def __init__(self):
        self.scheduler = AutoRetrainingScheduler()
        self.monitor = ModelPerformanceMonitor()
        self.drift_detector = DataDriftDetector()

    def initialize_system(self) -> None:
        """システム初期化"""
        logger.info("自動再学習システム初期化開始")

        # 全銘柄のベースライン確立
        symbols = list(get_settings().target_stocks.keys())

        logger.info(f"ベースライン確立: {len(symbols)}銘柄")
        for symbol in symbols[:5]:  # 最初の5銘柄でテスト
            try:
                self.drift_detector.establish_baseline(symbol)
            except Exception as e:
                logger.error(f"ベースライン確立エラー {symbol}: {e}")

        logger.info("自動再学習システム初期化完了")

    def start_automatic_retraining(self) -> None:
        """自動再学習開始"""
        self.initialize_system()
        self.scheduler.start_scheduler()

    def stop_automatic_retraining(self) -> None:
        """自動再学習停止"""
        self.scheduler.stop_scheduler()

    def force_full_retraining(self) -> Dict[str, Any]:
        """全銘柄強制再学習"""
        symbols = list(get_settings().target_stocks.keys())
        return self.scheduler.manual_retrain(symbols, "force_full_retrain")

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """包括的なシステム状態"""
        return {
            "auto_retraining": self.scheduler.get_system_status(),
            "retraining_candidates": self.monitor.get_retraining_candidates(),
            "system_health": "operational",
            "recommendations": [
                "84.6%精度維持のため定期的な再学習を推奨",
                "データドリフト検出により適応的なモデル更新",
                "性能低下時の自動再学習で持続的な高精度",
            ],
        }
