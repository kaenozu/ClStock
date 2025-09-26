"""
メモリ効率的なキャッシュシステム - LRU + 圧縮
"""

import logging
import pandas as pd
import pickle
import gzip


class MemoryEfficientCache:
    """メモリ効率的なキャッシュシステム - LRU + 圧縮"""

    def __init__(self, max_size: int = 1000, compression_enabled: bool = True):
        self.max_size = max_size
        self.compression_enabled = compression_enabled
        self.cache = {}
        self.access_order = []
        self.logger = logging.getLogger(__name__)

    def get(self, key: str):
        """キャッシュからデータ取得"""
        if key in self.cache:
            # LRU更新
            self.access_order.remove(key)
            self.access_order.append(key)

            data = self.cache[key]
            if self.compression_enabled and isinstance(data, bytes):
                return self._decompress(data)
            return data
        return None

    def put(self, key: str, value):
        """キャッシュにデータ保存"""
        # サイズ制限チェック
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()

        # データ圧縮（DataFrameの場合）
        if self.compression_enabled and isinstance(value, pd.DataFrame):
            value = self._compress(value)

        self.cache[key] = value

        # アクセス順序更新
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def _evict_lru(self):
        """最も古いエントリを削除"""
        if self.access_order:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

    def _compress(self, df: pd.DataFrame) -> bytes:
        """DataFrameを圧縮"""
        return gzip.compress(pickle.dumps(df))

    def _decompress(self, data: bytes) -> pd.DataFrame:
        """圧縮データを展開"""
        return pickle.loads(gzip.decompress(data))

    def clear(self):
        """キャッシュクリア"""
        self.cache.clear()
        self.access_order.clear()

    def size(self) -> int:
        """キャッシュサイズ取得"""
        return len(self.cache)
