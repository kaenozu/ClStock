"""Tests for the central target universe helper."""

import pytest

from config.target_universe import get_target_universe

EXPECTED_CORE_CODES = {
    "6758",
    "7203",
    "8306",
    "9984",
    "6861",
    "4502",
    "6503",
    "7201",
    "8001",
    "9022",
}

EXPECTED_CORE_ORDER = [
    "6758",
    "7203",
    "8306",
    "9984",
    "6861",
    "4502",
    "6503",
    "7201",
    "8001",
    "9022",
]


def test_target_universe_provides_base_codes():
    """The helper should expose canonical base ticker codes without suffixes."""
    universe = get_target_universe()

    assert EXPECTED_CORE_CODES.issubset(set(universe.base_codes))
    assert all("." not in code for code in universe.base_codes)


@pytest.mark.parametrize("suffix", [".T", ".JP"])
def test_target_universe_generates_variants(suffix: str):
    """Variant lookups should include suffix formatted tickers."""
    universe = get_target_universe()

    formatted = universe.format_codes(universe.default_codes, suffix=suffix)

    assert formatted == [f"{code}{suffix}" for code in universe.default_codes]
    for code in EXPECTED_CORE_CODES:
        variants = universe.variants_for(code, suffixes=[suffix])
        assert f"{code}{suffix}" in variants
        assert code in variants  # base code always included


def test_target_universe_default_formatted_symbols():
    """Default formatted list should match the configured core codes."""
    universe = get_target_universe()

    assert universe.default_codes == EXPECTED_CORE_ORDER
    assert universe.default_formatted() == [f"{code}.T" for code in EXPECTED_CORE_ORDER]
