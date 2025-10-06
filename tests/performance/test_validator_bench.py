import random
import string

import pytest

from utils.validators import sanitize_string, validate_email, validate_stock_symbol

pytest.importorskip("pytest_benchmark")


_ALPHABET = string.ascii_letters + string.digits + " -_.@"
_RANDOM = random.Random(42)


def _generate_payload(size: int) -> str:
    return "".join(_RANDOM.choice(_ALPHABET) for _ in range(size))


@pytest.mark.benchmark(group="validators")
def test_sanitize_string_benchmark(benchmark):
    payload = _generate_payload(5000)
    benchmark(lambda: sanitize_string(payload, max_length=256))


@pytest.mark.benchmark(group="validators")
def test_validate_email_benchmark(benchmark):
    payload = _generate_payload(64) + "@example.com"
    benchmark(lambda: validate_email(payload))


@pytest.mark.benchmark(group="validators")
def test_validate_stock_symbol_benchmark(benchmark):
    payload = "AAPL" * 10
    benchmark(lambda: validate_stock_symbol(payload))
