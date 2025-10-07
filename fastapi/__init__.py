"""Simplified FastAPI compatibility layer for offline testing.

このモジュールは、実際の FastAPI が利用できない環境でテストを
実行するために必要最低限の機能のみを提供するスタブ実装です。
本番環境での使用は想定していません。
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class Request:
    def __init__(self, headers: Optional[Dict[str, str]] = None, url: str = "") -> None:
        self.headers = headers or {}
        self.url = type("URL", (), {"path": url})
        self.client = type("Client", (), {"host": "testclient"})()


class Query:
    def __init__(self, default: Any = None, **_: Any) -> Any:
        self.default = default


class Depends:
    def __init__(self, dependency: Callable[..., Any]) -> None:
        self.dependency = dependency


@dataclass
class _Route:
    methods: Iterable[str]
    path: str
    endpoint: Callable[..., Awaitable[Any] | Any]
    path_regex: Any
    param_names: List[str]


def _compile_path(path: str) -> tuple[Any, List[str]]:
    import re

    param_names: List[str] = []
    pattern = ""
    i = 0
    while i < len(path):
        if path[i] == "{":
            j = path.find("}", i)
            if j == -1:
                raise ValueError(f"Invalid path template: {path}")
            name = path[i + 1 : j]
            param_names.append(name)
            pattern += rf"(?P<{name}>[^/]+)"
            i = j + 1
        else:
            pattern += re.escape(path[i])
            i += 1
    return re.compile(f"^{pattern}$"), param_names


class APIRouter:
    def __init__(self) -> None:
        self.routes: List[_Route] = []

    def add_api_route(
        self, path: str, endpoint: Callable[..., Any], methods: Optional[Iterable[str]] = None
    ) -> None:
        methods = tuple(m.upper() for m in (methods or ["GET"]))
        regex, params = _compile_path(path)
        self.routes.append(_Route(methods, path, endpoint, regex, params))

    def get(self, path: str, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_api_route(path, func, methods=["GET"])
            return func

        return decorator

    def post(self, path: str, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_api_route(path, func, methods=["POST"])
            return func

        return decorator


class FastAPI:
    def __init__(self) -> None:
        self.routes: List[_Route] = []

    def include_router(self, router: APIRouter) -> None:
        self.routes.extend(router.routes)

    def add_api_route(
        self, path: str, endpoint: Callable[..., Any], methods: Optional[Iterable[str]] = None
    ) -> None:
        router = APIRouter()
        router.add_api_route(path, endpoint, methods)
        self.include_router(router)

    def middleware(self, _type: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            # ミドルウェアはテストでは使用しないため、そのまま返す
            return func

        return decorator


class _JSONResponse:
    def __init__(self, data: Any, status_code: int = 200, headers: Optional[Dict[str, str]] = None):
        self._data = data
        self.status_code = status_code
        self.headers: Dict[str, str] = headers or {}

    def json(self) -> Any:
        return self._data


class TestClient:
    def __init__(self, app: FastAPI):
        self.app = app

    def get(self, url: str, headers: Optional[Dict[str, str]] = None):
        return self._request("GET", url, headers=headers)

    def post(self, url: str, json: Any = None, headers: Optional[Dict[str, str]] = None):
        return self._request("POST", url, json=json, headers=headers)

    def _request(
        self,
        method: str,
        url: str,
        json: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> _JSONResponse:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url)
        path = parsed.path
        query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        for route in self.app.routes:
            if method.upper() not in route.methods:
                continue
            match = route.path_regex.match(path)
            if match:
                path_params = match.groupdict()
                return self._call_route(route, path_params, query_params, json, headers)

        raise HTTPException(status_code=404, detail="Not Found")

    def _call_route(
        self,
        route: _Route,
        path_params: Dict[str, Any],
        query_params: Dict[str, Any],
        body: Any,
        headers: Optional[Dict[str, str]],
    ) -> _JSONResponse:
        import inspect

        func = route.endpoint
        sig = inspect.signature(func)
        kwargs: Dict[str, Any] = {}

        for name, param in sig.parameters.items():
            if name in path_params:
                kwargs[name] = path_params[name]
            elif name in query_params:
                kwargs[name] = query_params[name]
            elif body is not None and name == "body":
                kwargs[name] = body
            elif isinstance(param.default, Query):
                kwargs[name] = query_params.get(name, param.default.default)
            else:
                if param.default is not inspect._empty:
                    kwargs[name] = param.default

        try:
            result = func(**kwargs)
            if inspect.iscoroutine(result):
                result = asyncio.run(result)  # type: ignore[arg-type]
            status_code = 200
            if isinstance(result, tuple):
                data, status_code = result
            return _JSONResponse(result, status_code=status_code)
        except HTTPException as exc:  # pragma: no cover - behavior mirrors real FastAPI
            return _JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


__all__ = [
    "APIRouter",
    "Depends",
    "FastAPI",
    "HTTPException",
    "Query",
    "Request",
    "TestClient",
]
