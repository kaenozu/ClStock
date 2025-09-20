#!/usr/bin/env python3
"""
最適化履歴管理システム（改善版）
- ロギングの適切な使用
- セキュリティの向上（SHA256）
- 設定の一元管理
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib
import logging

# ロガー設定
logger = logging.getLogger(__name__)

# 定数定義
DEFAULT_KEEP_RECORDS = 30
CONFIG_HASH_LENGTH = 8
UUID_SUFFIX_LENGTH = 8


@dataclass
class OptimizationRecord:
    """最適化記録"""
    id: str
    timestamp: datetime
    stocks: List[str]
    performance_metrics: Dict[str, float]
    config_hash: str
    is_active: bool
    description: str
    rollback_available: bool = True


class OptimizationHistoryManager:
    """最適化履歴管理クラス（改善版）"""

    def __init__(self, history_dir: str = "optimization_history", logger_instance: Optional[logging.Logger] = None):
        # ロガー設定
        self.logger = logger_instance or logging.getLogger(__name__)

        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)

        self.history_file = self.history_dir / "history.json"
        self.backup_dir = self.history_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)

        # 設定ファイルもhistory_dir内に配置
        config_dir = self.history_dir / "config"
        config_dir.mkdir(exist_ok=True)
        self.current_config_file = config_dir / "optimal_stocks.json"
        self.history: List[OptimizationRecord] = self._load_history()

        self.logger.info(f"OptimizationHistoryManager initialized with {len(self.history)} existing records")

    def _load_history(self) -> List[OptimizationRecord]:
        """履歴を読み込む"""
        if not self.history_file.exists():
            self.logger.info("No existing history file found. Starting with empty history.")
            return []

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                records = []
                for item in data:
                    try:
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        records.append(OptimizationRecord(**item))
                    except (KeyError, ValueError, TypeError) as e:
                        self.logger.warning(f"Skipping invalid record: {e}")
                        continue

                self.logger.info(f"Successfully loaded {len(records)} records from history")
                return records

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in history file: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error loading history: {e}")
            return []

    def _save_history(self):
        """履歴を保存"""
        data = []
        for record in self.history:
            item = asdict(record)
            item['timestamp'] = record.timestamp.isoformat()
            data.append(item)

        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _generate_id(self) -> str:
        """一意のIDを生成"""
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = str(uuid.uuid4())[:UUID_SUFFIX_LENGTH]
        return f"OPT_{timestamp}_{unique_suffix}"

    def _calculate_config_hash(self, stocks: List[str]) -> str:
        """設定のハッシュ値を計算（SHA256を使用）"""
        content = json.dumps(sorted(stocks))
        # MD5からSHA256に変更（セキュリティ向上）
        return hashlib.sha256(content.encode()).hexdigest()[:CONFIG_HASH_LENGTH]

    def save_optimization_result(
        self,
        stocks: List[str],
        performance_metrics: Dict[str, float],
        description: str = "",
        auto_apply: bool = False
    ) -> str:
        """最適化結果を保存"""

        # 現在のアクティブ設定をバックアップ
        if auto_apply:
            self._backup_current_config()

        # 新しい記録を作成
        record_id = self._generate_id()
        config_hash = self._calculate_config_hash(stocks)

        record = OptimizationRecord(
            id=record_id,
            timestamp=datetime.now(),
            stocks=stocks,
            performance_metrics=performance_metrics,
            config_hash=config_hash,
            is_active=auto_apply,
            description=description or f"最適化実行 {datetime.now().strftime('%Y/%m/%d %H:%M')}"
        )

        # 自動適用の場合、他の記録を非アクティブに
        if auto_apply:
            for r in self.history:
                r.is_active = False

        self.history.append(record)
        self._save_history()

        # 自動適用の場合、設定を更新
        if auto_apply:
            self._apply_config(stocks)
            self.logger.info(f"最適化結果を自動適用しました (ID: {record_id})")
        else:
            self.logger.info(f"最適化結果を保存しました (ID: {record_id})")

        return record_id

    def _backup_current_config(self):
        """現在の設定をバックアップ"""
        if self.current_config_file.exists():
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_path = self.backup_dir / backup_name
            shutil.copy2(self.current_config_file, backup_path)
            self.logger.info(f"現在の設定をバックアップ: {backup_name}")

    def _apply_config(self, stocks: List[str]):
        """設定を適用"""
        config = {
            "optimal_stocks": stocks,
            "updated_at": datetime.now().isoformat(),
            "auto_applied": True
        }

        with open(self.current_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        self.logger.info(f"設定を適用しました: {len(stocks)}銘柄")

    def rollback_to(self, record_id: str) -> bool:
        """指定IDの設定にロールバック"""
        record = self.get_record(record_id)
        if not record:
            self.logger.error(f"ID {record_id} の記録が見つかりません")
            return False

        if not record.rollback_available:
            self.logger.error(f"この記録はロールバック不可です")
            return False

        # バックアップしてから適用
        self._backup_current_config()
        self._apply_config(record.stocks)

        # アクティブ状態を更新
        for r in self.history:
            r.is_active = (r.id == record_id)
        self._save_history()

        self.logger.info(f"ID {record_id} にロールバックしました")
        self.logger.info(f"   時刻: {record.timestamp}")
        self.logger.info(f"   説明: {record.description}")

        return True

    def get_record(self, record_id: str) -> Optional[OptimizationRecord]:
        """指定IDの記録を取得"""
        for record in self.history:
            if record.id == record_id:
                return record
        return None

    def get_active_record(self) -> Optional[OptimizationRecord]:
        """現在アクティブな記録を取得"""
        for record in self.history:
            if record.is_active:
                return record
        return None

    def list_history(self, limit: int = 10) -> List[OptimizationRecord]:
        """履歴をリスト表示（新しい順）"""
        return sorted(self.history, key=lambda x: x.timestamp, reverse=True)[:limit]

    def compare_records(self, id1: str, id2: str) -> Dict[str, Any]:
        """2つの記録を比較"""
        record1 = self.get_record(id1)
        record2 = self.get_record(id2)

        if not record1 or not record2:
            return {"error": "指定された記録が見つかりません"}

        # 共通銘柄
        common_stocks = set(record1.stocks) & set(record2.stocks)

        # 差分
        only_in_1 = set(record1.stocks) - set(record2.stocks)
        only_in_2 = set(record2.stocks) - set(record1.stocks)

        # パフォーマンス比較
        perf_diff = {}
        for key in record1.performance_metrics:
            if key in record2.performance_metrics:
                perf_diff[key] = {
                    "record1": record1.performance_metrics[key],
                    "record2": record2.performance_metrics[key],
                    "diff": record2.performance_metrics[key] - record1.performance_metrics[key]
                }

        return {
            "record1": {"id": id1, "timestamp": record1.timestamp},
            "record2": {"id": id2, "timestamp": record2.timestamp},
            "common_stocks": list(common_stocks),
            "only_in_1": list(only_in_1),
            "only_in_2": list(only_in_2),
            "performance_diff": perf_diff
        }

    def cleanup_old_records(self, keep_count: int = DEFAULT_KEEP_RECORDS):
        """古い記録をクリーンアップ"""
        if len(self.history) <= keep_count:
            self.logger.info(f"No cleanup needed. Current records: {len(self.history)}, Keep count: {keep_count}")
            return

        # タイムスタンプでソート（新しい順）
        sorted_history = sorted(self.history, key=lambda x: x.timestamp, reverse=True)

        # アクティブな記録は保持
        active_records = [r for r in sorted_history if r.is_active]
        inactive_records = [r for r in sorted_history if not r.is_active]

        # 保持する記録を選択
        keep_records = active_records + inactive_records[:keep_count - len(active_records)]

        removed_count = len(self.history) - len(keep_records)

        self.history = keep_records
        self._save_history()

        self.logger.info(f"Cleaned up {removed_count} old records. Remaining: {len(self.history)}")

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        if not self.history:
            return {"total_records": 0}

        performances = [r.performance_metrics.get('return_rate', 0) for r in self.history]

        # 最新記録を正しく取得（ソート後の最初の要素）
        sorted_history = sorted(self.history, key=lambda x: x.timestamp, reverse=True)
        latest_record = sorted_history[0] if sorted_history else None

        return {
            "total_records": len(self.history),
            "active_record": self.get_active_record().id if self.get_active_record() else None,
            "average_return": sum(performances) / len(performances) if performances else 0,
            "best_return": max(performances) if performances else 0,
            "worst_return": min(performances) if performances else 0,
            "latest_optimization": latest_record.timestamp if latest_record else None
        }

    def get_optimal_stocks_from_config(self) -> List[str]:
        """設定ファイルから最適銘柄リストを取得"""
        try:
            # 設定ファイルから読み込み
            from config.settings import get_settings
            settings = get_settings()

            # デフォルトの最適銘柄を返す
            default_optimal_stocks = list(settings.target_stocks.keys())[:10]

            # カスタム設定がある場合はそれを使用
            if self.current_config_file.exists():
                with open(self.current_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('optimal_stocks', default_optimal_stocks)

            return default_optimal_stocks

        except Exception as e:
            self.logger.error(f"設定読み込みエラー: {e}")
            return []


# グローバルインスタンス
history_manager = None


def get_history_manager() -> OptimizationHistoryManager:
    """履歴管理インスタンスを取得"""
    global history_manager
    if history_manager is None:
        history_manager = OptimizationHistoryManager()
    return history_manager


if __name__ == "__main__":
    # デモ実行
    import logging
    logging.basicConfig(level=logging.INFO)

    manager = get_history_manager()

    logger.info("=== 最適化履歴管理システム（改善版） ===")

    # サンプルデータで履歴作成
    sample_stocks = manager.get_optimal_stocks_from_config()[:5]
    sample_metrics = {"return_rate": 17.32, "sharpe_ratio": 1.85, "max_drawdown": -8.2}

    # 結果を保存（自動適用）
    record_id = manager.save_optimization_result(
        stocks=sample_stocks,
        performance_metrics=sample_metrics,
        description="サンプル最適化結果",
        auto_apply=True
    )

    # 履歴表示
    logger.info("\n📊 最適化履歴:")
    for record in manager.list_history(5):
        status = "✅" if record.is_active else "  "
        logger.info(f"{status} {record.id}: {record.description}")
        logger.info(f"   収益率: {record.performance_metrics.get('return_rate', 0):.2f}%")

    # 統計表示
    stats = manager.get_statistics()
    logger.info(f"\n📈 統計情報:")
    logger.info(f"総記録数: {stats['total_records']}")
    logger.info(f"平均収益率: {stats['average_return']:.2f}%")