"""A lightweight fallback stub for pandas used during testing."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
_search_paths = [p for p in sys.path if p != _repo_root]
_spec = importlib.machinery.PathFinder.find_spec("pandas", _search_paths)

if _spec and _spec.loader:
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
else:
    from copy import deepcopy
    from datetime import datetime, timedelta
    import json
    from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


    class Series:
        """Very small subset of :class:`pandas.Series`."""

        def __init__(self, data: Optional[Iterable[Any]] = None) -> None:
            self._data = list(data) if data is not None else []

        def copy(self) -> "Series":
            return Series(self._data)

        def tolist(self) -> List[Any]:
            return list(self._data)

        def replace(self, old: Any, new: Any) -> "Series":
            return Series([new if value == old else value for value in self._data])

        def mean(self) -> float:
            return sum(self._data) / len(self._data) if self._data else 0.0

        def __len__(self) -> int:  # pragma: no cover - trivial
            return len(self._data)

        @property
        def empty(self) -> bool:
            return len(self._data) == 0

        def __iter__(self) -> Iterator[Any]:  # pragma: no cover - trivial
            return iter(self._data)

        def __getitem__(self, index: int) -> Any:  # pragma: no cover - trivial
            return self._data[index]


    class DataFrame:
        """Minimal DataFrame implementation tailored for the unit tests."""

        def __init__(
            self,
            data: Optional[Dict[str, Sequence[Any]]] = None,
            index: Optional[Sequence[Any]] = None,
        ) -> None:
            if data is None:
                self._data: Dict[str, List[Any]] = {}
                self._index: List[Any] = [] if index is None else list(index)
            else:
                self._data = {key: list(values) for key, values in data.items()}
                length = len(next(iter(self._data.values()))) if self._data else 0
                for values in self._data.values():
                    if len(values) != length:
                        raise ValueError("All columns must be the same length")
                self._index = list(index) if index is not None else list(range(length))

        # ------------------------------------------------------------------
        @property
        def columns(self) -> List[str]:
            return list(self._data.keys())

        @property
        def empty(self) -> bool:
            return len(self) == 0

        def __len__(self) -> int:
            if not self._data:
                return 0
            return len(next(iter(self._data.values())))

        def __getitem__(self, key: str) -> Series:
            if key not in self._data:
                raise KeyError(key)
            return Series(self._data[key])

        def get(self, key: str, default: Optional[Series] = None) -> Series:
            if key in self._data:
                return Series(self._data[key])
            return default if default is not None else Series()

        def copy(self) -> "DataFrame":
            new_df = DataFrame()
            new_df._data = deepcopy(self._data)
            new_df._index = list(self._index)
            return new_df

        def tail(self, n: int) -> "DataFrame":
            if n <= 0:
                return DataFrame()
            new_data = {key: values[-n:] for key, values in self._data.items()}
            new_index = self._index[-n:]
            return DataFrame(new_data, new_index)

        def to_json(self, *_, **__) -> str:
            data = {
                "columns": self.columns,
                "index": list(self._index),
                "data": [
                    [self._data[col][i] for col in self.columns]
                    for i in range(len(self))
                ],
            }
            return json.dumps(data, separators=(",", ":"))

        def to_dict(self) -> Dict[str, List[Any]]:
            return {key: list(values) for key, values in self._data.items()}

        def round(self, decimals: int = 0) -> "DataFrame":
            def _round(value: Any) -> Any:
                try:
                    return round(value, decimals)
                except TypeError:
                    return value

            new_data = {key: [_round(v) for v in values] for key, values in self._data.items()}
            return DataFrame(new_data, self._index)


    def date_range(
        start: Any, end: Any = None, periods: int = None, freq: str = "D"
    ) -> List[datetime]:
        if freq != "D":
            raise ValueError("Stub only supports daily frequency")

        def _to_datetime(value: Any) -> datetime:
            if isinstance(value, datetime):
                return value
            if hasattr(value, "to_pydatetime"):
                return value.to_pydatetime()
            if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
                return datetime(value.year, value.month, value.day)
            return datetime.fromisoformat(str(value))

        if periods is not None and end is not None:
            raise ValueError("Specify either end or periods, not both")

        start_dt = _to_datetime(start)

        if periods is not None:
            return [start_dt + timedelta(days=i) for i in range(periods)]

        if end is None:
            raise ValueError("end must be provided when periods is None")

        end_dt = _to_datetime(end)
        days = (end_dt - start_dt).days
        return [start_dt + timedelta(days=i) for i in range(days + 1)]


    __all__ = ["DataFrame", "Series", "date_range"]
