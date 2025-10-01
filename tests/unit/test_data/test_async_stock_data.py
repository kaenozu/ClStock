import asyncio
import importlib.util
import os
import sys
from datetime import datetime, timedelta
from types import ModuleType

import pytest

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

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)


def _date_range(start: datetime, periods: int):
    return [start + timedelta(days=i) for i in range(periods)]


pandas_stub.DataFrame = _DummyDataFrame
pandas_stub.date_range = _date_range

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
