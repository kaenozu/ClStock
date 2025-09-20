#!/usr/bin/env python3
"""
最適化履歴管理システムのテスト
TDD: Red-Green-Refactorサイクル
"""

import pytest
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.optimization_history import OptimizationHistoryManager, OptimizationRecord


class TestOptimizationHistoryManager:
    """最適化履歴管理のテスト"""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリのフィクスチャ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_dir):
        """テスト用マネージャーのフィクスチャ"""
        return OptimizationHistoryManager(history_dir=temp_dir)

    @pytest.fixture
    def sample_data(self):
        """サンプルデータのフィクスチャ"""
        return {
            "stocks": ["7203", "6758", "9432", "8306", "6861"],
            "metrics": {
                "return_rate": 17.32,
                "sharpe_ratio": 1.85,
                "max_drawdown": -8.2,
                "win_rate": 73.4
            },
            "description": "テスト最適化結果"
        }

    # === RED: 失敗するテストを先に書く ===

    def test_初期化時に履歴が空であること(self, manager):
        """初期状態で履歴が空であることを確認"""
        assert len(manager.history) == 0
        assert manager.get_active_record() is None

    def test_最適化結果を保存できること(self, manager, sample_data):
        """最適化結果が正しく保存されることを確認"""
        # Act
        record_id = manager.save_optimization_result(
            stocks=sample_data["stocks"],
            performance_metrics=sample_data["metrics"],
            description=sample_data["description"],
            auto_apply=False
        )

        # Assert
        assert record_id is not None
        assert record_id.startswith("OPT_")
        assert len(manager.history) == 1

        record = manager.history[0]
        assert record.stocks == sample_data["stocks"]
        assert record.performance_metrics == sample_data["metrics"]
        assert record.description == sample_data["description"]
        assert record.is_active is False

    def test_自動適用で設定が更新されること(self, manager, sample_data, temp_dir):
        """auto_apply=Trueで設定ファイルが作成されることを確認"""
        # Arrange
        config_file = Path(temp_dir) / "config" / "optimal_stocks.json"

        # Act
        record_id = manager.save_optimization_result(
            stocks=sample_data["stocks"],
            performance_metrics=sample_data["metrics"],
            description=sample_data["description"],
            auto_apply=True
        )

        # Assert
        assert config_file.exists()

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        assert config["optimal_stocks"] == sample_data["stocks"]
        assert config["auto_applied"] is True
        assert "updated_at" in config

        # アクティブ記録の確認
        active_record = manager.get_active_record()
        assert active_record is not None
        assert active_record.id == record_id
        assert active_record.is_active is True

    def test_複数の記録で最新のみアクティブになること(self, manager, sample_data):
        """複数の自動適用で最新のみアクティブになることを確認"""
        # Act
        id1 = manager.save_optimization_result(
            stocks=sample_data["stocks"],
            performance_metrics=sample_data["metrics"],
            auto_apply=True
        )

        id2 = manager.save_optimization_result(
            stocks=["9984", "4519", "6098"],
            performance_metrics={"return_rate": 15.0},
            auto_apply=True
        )

        # Assert
        assert len(manager.history) == 2

        record1 = manager.get_record(id1)
        record2 = manager.get_record(id2)

        assert record1.is_active is False
        assert record2.is_active is True
        assert manager.get_active_record().id == id2

    def test_指定IDにロールバックできること(self, manager, sample_data):
        """過去の設定にロールバックできることを確認"""
        # Arrange
        id1 = manager.save_optimization_result(
            stocks=sample_data["stocks"],
            performance_metrics=sample_data["metrics"],
            auto_apply=True
        )

        id2 = manager.save_optimization_result(
            stocks=["9984", "4519", "6098"],
            performance_metrics={"return_rate": 15.0},
            auto_apply=True
        )

        # Act
        success = manager.rollback_to(id1)

        # Assert
        assert success is True
        assert manager.get_active_record().id == id1

        record1 = manager.get_record(id1)
        record2 = manager.get_record(id2)
        assert record1.is_active is True
        assert record2.is_active is False

    def test_存在しないIDへのロールバックが失敗すること(self, manager):
        """存在しないIDへのロールバックが失敗することを確認"""
        success = manager.rollback_to("INVALID_ID")
        assert success is False

    def test_履歴の永続化と読み込みができること(self, temp_dir, sample_data):
        """履歴が永続化され、再読み込みできることを確認"""
        # Arrange
        manager1 = OptimizationHistoryManager(history_dir=temp_dir)

        # Act - 保存
        record_id = manager1.save_optimization_result(
            stocks=sample_data["stocks"],
            performance_metrics=sample_data["metrics"],
            description=sample_data["description"],
            auto_apply=True
        )

        # Act - 新しいインスタンスで読み込み
        manager2 = OptimizationHistoryManager(history_dir=temp_dir)

        # Assert
        assert len(manager2.history) == 1
        loaded_record = manager2.history[0]
        assert loaded_record.id == record_id
        assert loaded_record.stocks == sample_data["stocks"]
        assert loaded_record.is_active is True

    def test_記録の比較ができること(self, manager):
        """2つの記録を比較できることを確認"""
        # Arrange
        id1 = manager.save_optimization_result(
            stocks=["7203", "6758", "9432"],
            performance_metrics={"return_rate": 15.0, "sharpe_ratio": 1.5}
        )

        id2 = manager.save_optimization_result(
            stocks=["7203", "8306", "6861"],
            performance_metrics={"return_rate": 18.0, "sharpe_ratio": 1.8}
        )

        # Act
        comparison = manager.compare_records(id1, id2)

        # Assert
        assert comparison is not None
        assert "7203" in comparison["common_stocks"]
        assert "6758" in comparison["only_in_1"]
        assert "8306" in comparison["only_in_2"]
        assert comparison["performance_diff"]["return_rate"]["diff"] == 3.0

    def test_統計情報を取得できること(self, manager):
        """統計情報が正しく計算されることを確認"""
        # Arrange
        manager.save_optimization_result(
            stocks=["7203"],
            performance_metrics={"return_rate": 10.0}
        )
        manager.save_optimization_result(
            stocks=["6758"],
            performance_metrics={"return_rate": 20.0}
        )
        manager.save_optimization_result(
            stocks=["9432"],
            performance_metrics={"return_rate": 15.0},
            auto_apply=True
        )

        # Act
        stats = manager.get_statistics()

        # Assert
        assert stats["total_records"] == 3
        assert stats["average_return"] == 15.0
        assert stats["best_return"] == 20.0
        assert stats["worst_return"] == 10.0
        assert stats["active_record"] is not None

    def test_古い記録のクリーンアップができること(self, manager):
        """古い記録が適切にクリーンアップされることを確認"""
        # Arrange - 10件の記録を作成
        for i in range(10):
            manager.save_optimization_result(
                stocks=[f"stock_{i}"],
                performance_metrics={"return_rate": float(i)}
            )

        # Act
        manager.cleanup_old_records(keep_count=5)

        # Assert
        assert len(manager.history) == 5

    def test_バックアップが作成されること(self, manager, sample_data, temp_dir):
        """設定変更時にバックアップが作成されることを確認"""
        # Arrange
        backup_dir = Path(temp_dir) / "backups"

        # Act - 2回自動適用して2つのバックアップが作成される
        manager.save_optimization_result(
            stocks=sample_data["stocks"],
            performance_metrics=sample_data["metrics"],
            auto_apply=True
        )

        manager.save_optimization_result(
            stocks=["9984"],
            performance_metrics={"return_rate": 10.0},
            auto_apply=True
        )

        # Assert
        backup_files = list(backup_dir.glob("backup_*.json"))
        assert len(backup_files) >= 1  # 最低1つのバックアップ

    def test_ハッシュ値が同じ銘柄で一致すること(self, manager):
        """同じ銘柄セットのハッシュ値が一致することを確認"""
        # Arrange
        stocks1 = ["7203", "6758", "9432"]
        stocks2 = ["9432", "7203", "6758"]  # 順序違い

        # Act
        hash1 = manager._calculate_config_hash(stocks1)
        hash2 = manager._calculate_config_hash(stocks2)

        # Assert
        assert hash1 == hash2

    def test_ID生成が一意であること(self, manager):
        """生成されるIDが一意であることを確認"""
        # Act
        ids = [manager._generate_id() for _ in range(100)]

        # Assert
        assert len(ids) == len(set(ids))  # 重複なし

    def test_ロールバック不可フラグが機能すること(self, manager, sample_data):
        """rollback_available=Falseの記録にロールバックできないことを確認"""
        # Arrange
        record_id = manager.save_optimization_result(
            stocks=sample_data["stocks"],
            performance_metrics=sample_data["metrics"]
        )

        # ロールバック不可に変更
        record = manager.get_record(record_id)
        record.rollback_available = False
        manager._save_history()

        # Act
        success = manager.rollback_to(record_id)

        # Assert
        assert success is False

    def test_空の履歴でget_active_recordがNoneを返すこと(self, manager):
        """空の履歴でget_active_recordがNoneを返すことを確認"""
        assert manager.get_active_record() is None

    def test_list_historyが新しい順で返すこと(self, manager):
        """履歴リストが新しい順で返されることを確認"""
        # Arrange
        with patch('systems.optimization_history.datetime') as mock_datetime:
            base_time = datetime.now()

            # 時間をずらして3つの記録を作成
            for i in range(3):
                mock_datetime.now.return_value = base_time + timedelta(hours=i)
                manager.save_optimization_result(
                    stocks=[f"stock_{i}"],
                    performance_metrics={"return_rate": float(i)}
                )

        # Act
        history_list = manager.list_history(limit=2)

        # Assert
        assert len(history_list) == 2
        # 最新の記録が最初に来ること
        assert "stock_2" in history_list[0].stocks
        assert "stock_1" in history_list[1].stocks


class TestOptimizationRecord:
    """OptimizationRecordデータクラスのテスト"""

    def test_レコードの作成ができること(self):
        """OptimizationRecordが正しく作成されることを確認"""
        # Arrange & Act
        record = OptimizationRecord(
            id="TEST_001",
            timestamp=datetime.now(),
            stocks=["7203", "6758"],
            performance_metrics={"return_rate": 15.0},
            config_hash="abcd1234",
            is_active=True,
            description="テストレコード",
            rollback_available=True
        )

        # Assert
        assert record.id == "TEST_001"
        assert len(record.stocks) == 2
        assert record.performance_metrics["return_rate"] == 15.0
        assert record.is_active is True
        assert record.rollback_available is True


# === Integration Tests ===

class TestIntegration:
    """統合テスト"""

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリのフィクスチャ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_完全なワークフロー(self, temp_dir):
        """最適化→適用→ロールバック→比較の完全なワークフローをテスト"""
        # Arrange
        manager = OptimizationHistoryManager(history_dir=temp_dir)

        # Act 1: 初回最適化
        id1 = manager.save_optimization_result(
            stocks=["7203", "6758", "9432", "8306", "6861"],
            performance_metrics={"return_rate": 15.0, "sharpe_ratio": 1.5},
            description="初回最適化",
            auto_apply=True
        )

        # Act 2: 2回目の最適化
        id2 = manager.save_optimization_result(
            stocks=["9984", "4519", "6098", "7203", "6758"],
            performance_metrics={"return_rate": 18.0, "sharpe_ratio": 1.8},
            description="改善版",
            auto_apply=True
        )

        # Act 3: ロールバック
        manager.rollback_to(id1)

        # Act 4: 比較
        comparison = manager.compare_records(id1, id2)

        # Act 5: 統計取得
        stats = manager.get_statistics()

        # Assert
        assert manager.get_active_record().id == id1
        assert len(comparison["common_stocks"]) == 2  # 7203, 6758
        assert stats["total_records"] == 2
        assert stats["average_return"] == 16.5
        assert stats["best_return"] == 18.0