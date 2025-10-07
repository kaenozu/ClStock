"""Minimal sklearn.model_selection stub for test environment."""

from __future__ import annotations

import numpy as np


def train_test_split(
    *arrays,
    test_size: float | int = 0.25,
    random_state: int | None = None,
    shuffle: bool = True,
):  # pragma: no cover - lightweight stub
    if not arrays:
        raise ValueError("At least one array required")

    n_samples = len(arrays[0])
    indices = np.arange(n_samples)
    rng = np.random.default_rng(random_state)

    if shuffle:
        rng.shuffle(indices)

    if isinstance(test_size, float):
        test_count = max(1, int(round(n_samples * test_size)))
    else:
        test_count = int(test_size)

    test_count = min(max(test_count, 1), n_samples - 1)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    result = []
    for array in arrays:
        arr = np.asarray(array)
        result.append(arr[train_indices])
        result.append(arr[test_indices])

    return result


def cross_val_score(*_, **__):  # pragma: no cover - placeholder
    raise RuntimeError(
        "sklearn.model_selection is unavailable in this test environment",
    )


class TimeSeriesSplit:  # pragma: no cover - lightweight stub
    def __init__(self, *_, **__):
        pass

    def split(self, *_args, **_kwargs):
        return []
