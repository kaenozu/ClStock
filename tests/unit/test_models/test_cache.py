"""
キャッシュ機能のテスト
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

from utils.cache import (
    DataCache,
    cached,
    cache_dataframe,
    get_cache,
    clear_cache,
    cleanup_cache,
)


class TestDataCache:
    """DataCacheクラスのテスト"""

    def setup_method(self):
        """テスト前の設定"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = DataCache(self.temp_dir, default_ttl=10)

    def teardown_method(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir)

    def test_cache_set_and_get(self):
        """キャッシュの保存と取得のテスト"""
        key = "test_key"
        value = {"data": "test_value"}

        # キャッシュに保存
        self.cache.set(key, value)

        # キャッシュから取得
        cached_value = self.cache.get(key)

        assert cached_value == value

    def test_cache_miss(self):
        """キャッシュミスのテスト"""
        result = self.cache.get("non_existent_key")
        assert result is None

    def test_cache_expiration(self):
        """キャッシュの期限切れテスト"""
        key = "expiring_key"
        value = "expiring_value"

        # 短いTTLでキャッシュ
        self.cache.set(key, value, ttl=1)

        # すぐに取得できることを確認
        assert self.cache.get(key) == value

        # 期限切れまで待機
        time.sleep(1.1)

        # 期限切れで取得できないことを確認
        assert self.cache.get(key) is None

    def test_cache_delete(self):
        """キャッシュの削除テスト"""
        key = "delete_test"
        value = "delete_value"

        self.cache.set(key, value)
        assert self.cache.get(key) == value

        self.cache.delete(key)
        assert self.cache.get(key) is None

    def test_cache_clear(self):
        """全キャッシュ削除のテスト"""
        # 複数のキャッシュを設定
        for i in range(3):
            self.cache.set(f"key_{i}", f"value_{i}")

        # すべて取得できることを確認
        for i in range(3):
            assert self.cache.get(f"key_{i}") == f"value_{i}"

        # 全削除
        self.cache.clear()

        # すべて取得できないことを確認
        for i in range(3):
            assert self.cache.get(f"key_{i}") is None

    def test_cache_cleanup_expired(self):
        """期限切れキャッシュのクリーンアップテスト"""
        # 通常のキャッシュと期限切れキャッシュを設定
        self.cache.set("normal", "value", ttl=10)
        self.cache.set("expiring", "value", ttl=1)

        time.sleep(1.1)

        # クリーンアップ実行
        deleted_count = self.cache.cleanup_expired()

        assert deleted_count == 1
        assert self.cache.get("normal") == "value"
        assert self.cache.get("expiring") is None

    def test_cache_key_generation(self):
        """キャッシュキー生成のテスト"""
        key1 = self.cache._get_cache_key("func", "arg1", "arg2", kwarg1="value1")
        key2 = self.cache._get_cache_key("func", "arg1", "arg2", kwarg1="value1")
        key3 = self.cache._get_cache_key("func", "arg1", "arg2", kwarg1="value2")

        # 同じ引数なら同じキー
        assert key1 == key2

        # 違う引数なら違うキー
        assert key1 != key3

    def test_corrupted_cache_handling(self):
        """破損したキャッシュファイルの処理テスト"""
        key = "corrupted_key"
        cache_path = self.cache._get_cache_path(key)

        # 破損したファイルを作成
        cache_path.write_text("corrupted data")

        # 破損したキャッシュからの取得はNoneを返す
        result = self.cache.get(key)
        assert result is None

        # 破損したファイルは削除される
        assert not cache_path.exists()


class TestCachedDecorator:
    """cachedデコレータのテスト"""

    def setup_method(self):
        """テスト前の設定"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = DataCache(self.temp_dir, default_ttl=10)

    def teardown_method(self):
        """テスト後のクリーンアップ"""
        shutil.rmtree(self.temp_dir)

    def test_cached_decorator_basic(self):
        """基本的なcachedデコレータのテスト"""
        call_count = 0

        @cached(ttl=10, cache_instance=self.cache)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # 初回呼び出し
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1

        # 2回目呼び出し（キャッシュから）
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # 関数は呼ばれない

        # 異なる引数での呼び出し
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2  # 関数が呼ばれる

    def test_cached_decorator_with_kwargs(self):
        """キーワード引数を含むcachedデコレータのテスト"""
        call_count = 0

        @cached(ttl=10, cache_instance=self.cache)
        def function_with_kwargs(x, y=10, z=20):
            nonlocal call_count
            call_count += 1
            return x + y + z

        # 初回呼び出し
        result1 = function_with_kwargs(1, y=5, z=15)
        assert result1 == 21
        assert call_count == 1

        # 同じ引数での呼び出し（キャッシュから）
        result2 = function_with_kwargs(1, y=5, z=15)
        assert result2 == 21
        assert call_count == 1

        # 順序を変えた同じ引数での呼び出し（キャッシュから）
        result3 = function_with_kwargs(1, z=15, y=5)
        assert result3 == 21
        assert call_count == 1

    def test_cached_decorator_expiration(self):
        """cachedデコレータの期限切れテスト"""
        call_count = 0

        @cached(ttl=1, cache_instance=self.cache)
        def expiring_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # 初回呼び出し
        result1 = expiring_function(5)
        assert result1 == 10
        assert call_count == 1

        # 期限切れまで待機
        time.sleep(1.1)

        # 期限切れ後の呼び出し
        result2 = expiring_function(5)
        assert result2 == 10
        assert call_count == 2  # 関数が再度呼ばれる


class TestCacheDataFrameDecorator:
    """cache_dataframeデコレータのテスト"""

    def test_cache_dataframe_decorator(self):
        """cache_dataframeデコレータのテスト"""
        import pandas as pd

        call_count = 0

        @cache_dataframe(ttl=10)
        def get_dataframe(rows):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame({"value": range(rows)})

        # 初回呼び出し
        df1 = get_dataframe(5)
        assert len(df1) == 5
        assert call_count == 1

        # 2回目呼び出し（キャッシュから）
        df2 = get_dataframe(5)
        assert len(df2) == 5
        assert call_count == 1

        # DataFrameの内容が同じことを確認
        pd.testing.assert_frame_equal(df1, df2)


class TestGlobalCacheFunctions:
    """グローバルキャッシュ関数のテスト"""

    def test_get_cache(self):
        """get_cache関数のテスト"""
        cache = get_cache()
        assert isinstance(cache, DataCache)

    def test_clear_cache(self):
        """clear_cache関数のテスト"""
        cache = get_cache()
        cache.set("test", "value")

        assert cache.get("test") == "value"

        clear_cache()

        assert cache.get("test") is None

    def test_cleanup_cache(self):
        """cleanup_cache関数のテスト"""
        cache = get_cache()

        # 期限切れキャッシュを作成
        cache.set("expiring", "value", ttl=1)
        time.sleep(1.1)

        deleted_count = cleanup_cache()
        assert deleted_count >= 0  # 削除された数


class TestCacheIntegration:
    """キャッシュ統合テスト"""

    def test_cache_with_exception(self):
        """例外が発生する関数でのキャッシュテスト"""
        call_count = 0

        @cached(ttl=10)
        def failing_function(should_fail):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Test exception")
            return "success"

        # 成功ケース
        result = failing_function(False)
        assert result == "success"
        assert call_count == 1

        # キャッシュから取得
        result = failing_function(False)
        assert result == "success"
        assert call_count == 1

        # 例外ケース（キャッシュされない）
        with pytest.raises(ValueError):
            failing_function(True)
        assert call_count == 2

        # 再度例外ケース（キャッシュされていないので再実行）
        with pytest.raises(ValueError):
            failing_function(True)
        assert call_count == 3
