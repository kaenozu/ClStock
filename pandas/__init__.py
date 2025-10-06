"""A lightweight fallback stub for pandas used during testing."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]


def _is_repo_path(entry: str) -> bool:
    try:
        return Path(entry).resolve() == _repo_root
    except Exception:
        return False


_search_paths = [p for p in sys.path if not _is_repo_path(p)]
_spec = importlib.machinery.PathFinder.find_spec("pandas", _search_paths)

if _spec and _spec.loader:
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
else:
    # Provide a very small fallback stub so targeted tests can run without the
    # heavy pandas dependency. This is intentionally minimal and only supports
    # functionality required by the tests that rely on read_sql_query and the
    # to_dict("records") helper.
    import sqlite3
    from typing import Iterable, List, Sequence

    stub_module = types.ModuleType(__name__)

    class Series:  # pragma: no cover - minimal placeholder
        def __init__(self, *_, **__):
            raise RuntimeError("pandas Series is unavailable in this test environment")

    class DataFrame:
        """Minimal subset of the pandas DataFrame API used in tests."""

        def __init__(self, rows: Sequence[Sequence[object]], columns: Iterable[str]):
            self._rows: List[Sequence[object]] = list(rows)
            self.columns = list(columns)

        @property
        def empty(self) -> bool:
            return not self._rows

        def to_dict(self, orient: str = "records"):
            if orient != "records":
                raise NotImplementedError(
                    "Only the 'records' orient is supported in the stub",
                )
            return [dict(zip(self.columns, row)) for row in self._rows]

    def read_sql_query(query: str, conn: sqlite3.Connection, params=None):
        cursor = conn.cursor()
        if params is None:
            cursor.execute(query)
        else:
            cursor.execute(query, params)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return DataFrame(rows, columns)

    stub_module.Series = Series
    stub_module.DataFrame = DataFrame
    stub_module.read_sql_query = read_sql_query

    sys.modules[__name__] = stub_module
    globals().update(stub_module.__dict__)
