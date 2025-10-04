import numpy as np
import pandas as pd
import pytest


from utils.technical_indicators import calculate_keltner_channels


pytest.importorskip("pandas")

try:  # Ensure the real pandas Series implementation is available
    pd.Series([0])
except RuntimeError as exc:  # pragma: no cover - guard for stubbed pandas
    pytest.skip(str(exc), allow_module_level=True)


def test_calculate_keltner_channels_values():
    high = pd.Series([10.0, 11.0, 12.0, 13.0])
    low = pd.Series([9.0, 9.5, 10.0, 11.0])
    close = pd.Series([9.5, 10.5, 11.5, 12.5])

    channels = calculate_keltner_channels(
        high=high, low=low, close=close, window=2, multiplier=1.5
    )

    expected_middle = pd.Series(
        [9.5, 10.0555556, 10.7962964, 11.7098766]
    )
    expected_upper = pd.Series(
        [np.nan, 11.9305556, 13.4212964, 14.7098766]
    )
    expected_lower = pd.Series(
        [np.nan, 8.1805556, 8.1712964, 8.7098766]
    )

    pd.testing.assert_series_equal(
        channels["Middle"],
        expected_middle,
        check_names=False,
        check_exact=False,
        atol=1e-6,
    )
    pd.testing.assert_series_equal(
        channels["Upper"],
        expected_upper,
        check_names=False,
        check_exact=False,
        atol=1e-6,
    )
    pd.testing.assert_series_equal(
        channels["Lower"],
        expected_lower,
        check_names=False,
        check_exact=False,
        atol=1e-6,
    )


def test_calculate_keltner_channels_error_fallback():
    high = pd.Series(["a", "b"])
    low = pd.Series(["c", "d"])
    close = pd.Series([1.0, 2.0])

    channels = calculate_keltner_channels(high, low, close)

    expected = pd.DataFrame(
        {
            "Upper": close,
            "Middle": close,
            "Lower": close,
        }
    )

    pd.testing.assert_frame_equal(channels, expected)
