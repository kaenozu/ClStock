import asyncio
import importlib.util
import os
import sys
from datetime import datetime, timedelta
from types import ModuleType
from typing import List

import pytest

from utils.exceptions import DataFetchError
import utils.cache as cache_module

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

# Provide lightweight aiohttp and pandas stubs so the module under test can be imported
aiohttp_stub = ModuleType("aiohttp")


class _DummySession:  # pragma: no cover - stub for import
    closed = False

    async def close(self):  # pragma: no cover - stub
        self.closed = True


class _DummyConnector:  # pragma: no cover - stub
    def __init__(self, *_, **__):
        pass


class _DummyTimeout:  # pragma: no cover - stub
    def __init__(self, *_, **__):
        pass


aiohttp_stub.ClientSession = _DummySession
aiohttp_stub.TCPConnector = _DummyConnector
aiohttp_stub.ClientTimeout = _DummyTimeout

sys.modules["aiohttp"] = aiohttp_stub

pandas_stub = ModuleType("pandas")


class _DummyDataFrame:
    def __init__(self, data=None, index=None, **_):
        self._data = list(data or [])
        if index is None:
            self.index = list(range(len(self._data)))
        else:
            self.index = list(index)
        self._columns = {}

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key, value):  # pragma: no cover - trivial helper
        self._columns[key] = value

    def copy(self):  # pragma: no cover - used in helpers
        dup = _DummyDataFrame(self._data.copy(), index=self.index.copy())
        dup._columns = self._columns.copy()
        return dup


def _date_range(start: datetime, periods: int):
    return [start + timedelta(days=i) for i in range(periods)]


pandas_stub.DataFrame = _DummyDataFrame
pandas_stub.date_range = _date_range
pandas_stub.Timestamp = datetime

sys.modules["pandas"] = pandas_stub

MODULE_PATH = os.path.join(PROJECT_ROOT, "data", "async_stock_data.py")
MODULE_SPEC = importlib.util.spec_from_file_location("data.async_stock_data", MODULE_PATH)
async_stock_data = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(async_stock_data)
async_stock_data.pd = pandas_stub

AsyncStockDataProvider = async_stock_data.AsyncStockDataProvider


class FakeLoop:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    async def run_in_executor(self, executor, func, *args):
        self.call_count += 1
        if not self._responses:
            raise AssertionError("No more responses configured for FakeLoop")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def sample_dataframe():
    index = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(2)]
    return pandas_stub.DataFrame([100.0, 101.0], index=index)


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    class _DummyCache:
        def __init__(self):
            self._store = {}

        def get(self, key):
            return self._store.get(key)

        def set(self, key, value, ttl=None):  # pragma: no cover - trivial cache stub
            self._store[key] = value

    cache = _DummyCache()
    monkeypatch.setattr(cache_module, "get_cache", lambda: cache)
    return cache


def test_fetch_with_yfinance_async_awaits_sleep_before_success(monkeypatch, sample_dataframe):
    provider = AsyncStockDataProvider()
    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    fake_loop = FakeLoop([sample_dataframe])

    monkeypatch.setattr(async_stock_data.asyncio, "get_event_loop", lambda: fake_loop)
    monkeypatch.setattr(async_stock_data.asyncio, "sleep", fake_sleep)

    result = asyncio.run(provider._fetch_with_yfinance_async("TEST", "1d"))

    assert not result.empty
    assert sleep_calls == [0.1]
    assert fake_loop.call_count == 1


def test_fetch_with_yfinance_async_uses_async_sleep_on_retry(monkeypatch, sample_dataframe):
    provider = AsyncStockDataProvider()
    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    fake_loop = FakeLoop([pandas_stub.DataFrame(), sample_dataframe])

    monkeypatch.setattr(async_stock_data.asyncio, "get_event_loop", lambda: fake_loop)
    monkeypatch.setattr(async_stock_data.asyncio, "sleep", fake_sleep)

    result = asyncio.run(provider._fetch_with_yfinance_async("TEST", "1d"))

    assert not result.empty
    assert sleep_calls == [0.1, 0.5, 0.1]
    assert fake_loop.call_count == 2


def test_fetch_with_yfinance_async_retries_with_async_sleep_on_exception(monkeypatch):
    provider = AsyncStockDataProvider()
    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    fake_loop = FakeLoop(
        [
            Exception("HTTP 429 error"),
            Exception("HTTP 500 transient"),
            Exception("HTTP 429 error"),
        ]
    )

    monkeypatch.setattr(async_stock_data.asyncio, "get_event_loop", lambda: fake_loop)
    monkeypatch.setattr(async_stock_data.asyncio, "sleep", fake_sleep)

    result = asyncio.run(provider._fetch_with_yfinance_async("TEST", "1d"))

    assert result.empty
    assert sleep_calls == [0.1, 1.0, 0.1, 2.0, 0.1, 3.0]
    assert fake_loop.call_count == 3


class _StubSyncProvider:
    def __init__(self, local_result, http_result, market_provider="hybrid"):
        self.local_calls = 0
        self.http_calls = 0
        self.local_result = local_result
        self.http_result = http_result
        self.market_data_config = ModuleType("config")
        self.market_data_config.provider = market_provider
        self.jp_stock_codes = {"TEST": "Test Corp"}

    def _ticker_formats(self, symbol):
        return [symbol, f"{symbol}.T"]

    def _fetch_from_local_csv(self, symbol, period, start_ts, end_ts):
        self.local_calls += 1
        return self.local_result.copy(), "LOCAL"

    def _fetch_from_http_api(
        self,
        symbol,
        ticker_candidates,
        period,
        start,
        end,
        start_ts,
        end_ts,
    ):
        self.http_calls += 1
        return self.http_result.copy(), "HTTP"

    def _prepare_history_frame(self, data, symbol, actual_ticker):
        prepared = data.copy()
        prepared["Symbol"] = symbol
        prepared["ActualTicker"] = actual_ticker
        prepared["CompanyName"] = self.jp_stock_codes.get(symbol, symbol)
        return prepared

    def _should_use_local_first(self, symbol):
        return True


@pytest.mark.asyncio
async def test_get_stock_data_uses_local_before_yfinance(monkeypatch, sample_dataframe):
    provider = AsyncStockDataProvider()
    stub = _StubSyncProvider(sample_dataframe, pandas_stub.DataFrame())

    monkeypatch.setattr(provider, "_get_sync_provider", lambda: stub)

    async def fail_yfinance(*_args, **_kwargs):
        raise AssertionError("yfinance should not be called when local data exists")

    monkeypatch.setattr(provider, "_fetch_with_yfinance_async", fail_yfinance)

    result = await provider.get_stock_data("TEST", "1d_local")

    assert not result.empty
    assert stub.local_calls == 1
    assert stub.http_calls == 0
    assert getattr(result, "_columns", {}).get("ActualTicker") == "LOCAL"


@pytest.mark.asyncio
async def test_get_stock_data_attempts_http_before_yfinance(monkeypatch, sample_dataframe):
    provider = AsyncStockDataProvider()
    empty = pandas_stub.DataFrame()
    order: List[str] = []

    class OrderedStub(_StubSyncProvider):
        def __init__(self):
            super().__init__(empty, empty)

        def _fetch_from_local_csv(self, symbol, period, start_ts, end_ts):
            order.append("local")
            return empty.copy(), "LOCAL"

        def _fetch_from_http_api(
            self,
            symbol,
            ticker_candidates,
            period,
            start,
            end,
            start_ts,
            end_ts,
        ):
            order.append("http")
            return empty.copy(), "HTTP"

    stub = OrderedStub()

    monkeypatch.setattr(provider, "_get_sync_provider", lambda: stub)

    async def fake_yfinance(ticker, period):
        order.append("yfinance")
        return sample_dataframe.copy()

    monkeypatch.setattr(provider, "_fetch_with_yfinance_async", fake_yfinance)

    result = await provider.get_stock_data("TEST", "fallback")

    assert not result.empty
    assert order == ["local", "http", "yfinance"]


@pytest.mark.asyncio
async def test_get_stock_data_raises_error_with_yfinance_reason(monkeypatch):
    provider = AsyncStockDataProvider()
    empty = pandas_stub.DataFrame()
    stub = _StubSyncProvider(empty, empty)

    monkeypatch.setattr(provider, "_get_sync_provider", lambda: stub)

    async def failing_yfinance(ticker, period):
        provider._last_yfinance_error = "timeout"
        return pandas_stub.DataFrame()

    monkeypatch.setattr(provider, "_fetch_with_yfinance_async", failing_yfinance)

    with pytest.raises(DataFetchError) as exc_info:
        await provider.get_stock_data("TEST", "error")

    assert "timeout" in str(exc_info.value)
