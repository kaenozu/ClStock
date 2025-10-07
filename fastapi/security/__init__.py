from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HTTPAuthorizationCredentials:
    scheme: str = "Bearer"
    credentials: Optional[str] = None


class HTTPBearer:
    def __call__(self, request) -> HTTPAuthorizationCredentials:
        authorization = request.headers.get("Authorization") if request else None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization.split(" ", 1)[1]
            return HTTPAuthorizationCredentials(credentials=token)
        return HTTPAuthorizationCredentials(credentials=None)


__all__ = ["HTTPAuthorizationCredentials", "HTTPBearer"]
