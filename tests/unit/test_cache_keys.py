import math

from utils.cache import DataCache


def test_cache_sanitizes_windows_unsafe_keys(tmp_path):
    cache = DataCache(cache_dir=str(tmp_path), auto_cleanup=False)
    try:
        key = "stock::7203.T::2y"
        assert cache.set(key, {"value": 42}, ttl=60)
        files = list(tmp_path.glob("*.cache"))
        assert len(files) == 1
        assert "::" not in files[0].name
        assert cache.get(key) == {"value": 42}
    finally:
        cache.shutdown()
