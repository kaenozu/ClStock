"""
Security middleware for the ClStock API
"""

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Callable, Optional, Set
import time

import os

from functools import wraps
from utils.logger_config import get_logger

logger = get_logger(__name__)

_env_cache: Dict[str, Optional[str]] = {}
_logged_missing_env_vars: Set[str] = set()
_env_tokens_cache: Dict[str, str] = {}
_env_tokens_cache_sources: Dict[str, Optional[str]] = {}
_TEST_TOKEN_FLAG = "API_ENABLE_TEST_TOKENS"
_TEST_TOKENS = {
    "admin_token_secure_2024": "administrator",
    "user_token_basic_2024": "user",
}

def reset_env_token_cache() -> None:
    """Reset cached environment token values and warning tracking."""

    _env_cache.clear()
    _logged_missing_env_vars.clear()
    _env_tokens_cache.clear()
    _env_tokens_cache_sources.clear()

def _load_required_env_var(var_name: str) -> str:
    """Fetch an environment variable or raise an explicit error."""

    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {var_name}. "
            "Set this variable before starting the application."
        )
    return value


def _initialize_api_keys() -> Dict[str, str]:
    """Initialize API keys from config or environment variables."""

    try:
        from config.secrets import API_KEYS as secrets_api_keys  # type: ignore

        if not secrets_api_keys:
            raise RuntimeError("config.secrets.API_KEYS is empty")

        return dict(secrets_api_keys)
    except ImportError:
        dev_key = _load_required_env_var("CLSTOCK_DEV_KEY")
        admin_key = _load_required_env_var("CLSTOCK_ADMIN_KEY")
        return {
            dev_key: "developer",
            admin_key: "administrator",
        }


def _get_env_with_warning(var_name: str) -> Optional[str]:
    """
    環境変数を取得し、存在しない場合は警告をログに出力します。

    Args:
        var_name: 取得する環境変数名

    Returns:
        環境変数の値。存在しない場合は None。
    """
    if var_name in _env_cache:
        return _env_cache[var_name]

    value = os.getenv(var_name)
    if not value and var_name not in _logged_missing_env_vars:
        logger.warning(f"{var_name} environment variable not set")
        _logged_missing_env_vars.add(var_name)

    _env_cache[var_name] = value
    return value


def _get_env_tokens_from_cache() -> Dict[str, str]:
    """Fetch cached environment-based tokens, refreshing if necessary."""

    admin_token = _get_env_with_warning("API_ADMIN_TOKEN")
    user_token = _get_env_with_warning("API_USER_TOKEN")

    current_sources = {
        "API_ADMIN_TOKEN": admin_token,
        "API_USER_TOKEN": user_token,
    }

    if current_sources != _env_tokens_cache_sources:
        _env_tokens_cache.clear()

        if admin_token:
            _env_tokens_cache[admin_token] = "administrator"

        if user_token:
            _env_tokens_cache[user_token] = "user"

        _env_tokens_cache_sources.clear()
        _env_tokens_cache_sources.update(current_sources)

    return _env_tokens_cache

# Simple in-memory storage for rate limiting
# In production, you would use Redis or similar
rate_limit_storage: Dict[str, Dict[str, int]] = {}

API_KEYS: Dict[str, str] = _initialize_api_keys()
ALLOW_TEST_TOKENS = False
TEST_TOKENS: Dict[str, str] = {}


def configure_security(
    api_keys: Optional[Dict[str, str]] = None,
    test_tokens: Optional[Dict[str, str]] = None,
    enable_test_tokens: Optional[bool] = None,
) -> None:
    """Configure security settings, primarily for testing purposes."""

    global API_KEYS, TEST_TOKENS, ALLOW_TEST_TOKENS

    if api_keys is not None:
        if not api_keys:
            raise ValueError("api_keys cannot be empty")
        API_KEYS = dict(api_keys)

    if test_tokens is not None:
        TEST_TOKENS = dict(test_tokens)

    if enable_test_tokens is not None:
        ALLOW_TEST_TOKENS = enable_test_tokens

security = HTTPBearer()


class RateLimiter:
    """Simple rate limiter for API endpoints"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, client_id: str, endpoint: str) -> bool:
        """Check if client is allowed to make request"""
        current_time = int(time.time())
        key = f"{client_id}:{endpoint}"

        if key not in rate_limit_storage:
            rate_limit_storage[key] = {
                "count": 1,
                "reset_time": current_time + self.window_seconds,
            }
            return True

        client_data = rate_limit_storage[key]

        # Check if window has expired
        if current_time > client_data["reset_time"]:
            # Reset window
            client_data["count"] = 1
            client_data["reset_time"] = current_time + self.window_seconds
            return True

        # Check if limit exceeded
        if client_data["count"] >= self.max_requests:
            return False

        # Increment count
        client_data["count"] += 1
        return True

    def get_remaining_requests(self, client_id: str, endpoint: str) -> int:
        """Get remaining requests for client"""
        key = f"{client_id}:{endpoint}"

        if key not in rate_limit_storage:
            return self.max_requests

        client_data = rate_limit_storage[key]
        current_time = int(time.time())

        if current_time > client_data["reset_time"]:
            return self.max_requests

        return max(0, self.max_requests - client_data["count"])

    def get_reset_time(self, client_id: str, endpoint: str) -> int:
        """Get reset time for client"""
        key = f"{client_id}:{endpoint}"

        if key not in rate_limit_storage:
            return int(time.time()) + self.window_seconds

        return rate_limit_storage[key]["reset_time"]


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Verify API key and return user type"""
    token = credentials.credentials

    if token not in API_KEYS:
        logger.warning(f"Invalid API key attempt: {token}")
        raise HTTPException(status_code=401, detail="Invalid API key")

    user_type = API_KEYS[token]
    logger.info(f"API key verified for user type: {user_type}")
    return user_type


def _should_include_test_tokens() -> bool:
    """テスト用トークンを許可するかを判定"""

    if ALLOW_TEST_TOKENS:
        return True

    flag_value = os.getenv(_TEST_TOKEN_FLAG, "").strip().lower()
    return flag_value in {"1", "true", "yes", "on"}


def _build_allowed_tokens() -> Dict[str, str]:
    """許可されたトークンの一覧を構築"""

    tokens = dict(API_KEYS)

    env_tokens = _get_env_tokens_from_cache()
    tokens.update(env_tokens)

    if _should_include_test_tokens():
        logger.warning(
            "Test tokens enabled via environment flag. Do not use in production."
        )
        tokens.update(_TEST_TOKENS)

    return tokens


def verify_token(token: str) -> str:
    """トークン検証関数（互換性のため）"""
    if not token:
        raise HTTPException(status_code=401, detail="Invalid token")

    allowed_tokens = _build_allowed_tokens()

    if token not in allowed_tokens:
        logger.warning(f"Invalid token attempt: {token}")
        raise HTTPException(status_code=401, detail="Invalid token")

    user_type = allowed_tokens[token]
    logger.info(f"Token verified for user type: {user_type}")
    return user_type


def require_role(required_role: str):
    """Dependency to require specific role"""

    def role_checker(user_type: str = Depends(verify_api_key)) -> str:
        if required_role == "admin" and user_type != "administrator":
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user_type

    return role_checker


def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """Rate limiting decorator"""
    limiter = RateLimiter(max_requests, window_seconds)

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host if request.client else "unknown"
            endpoint = request.url.path

            if not limiter.is_allowed(client_ip, endpoint):
                reset_time = limiter.get_reset_time(client_ip, endpoint)
                remaining = limiter.get_remaining_requests(client_ip, endpoint)

                logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}")

                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "remaining": remaining,
                        "reset_time": reset_time,
                    },
                )

            # Add rate limit headers to response
            response = await func(request, *args, **kwargs)

            remaining = limiter.get_remaining_requests(client_ip, endpoint)
            reset_time = limiter.get_reset_time(client_ip, endpoint)

            if hasattr(response, "headers"):
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                response.headers["X-RateLimit-Reset"] = str(reset_time)
                response.headers["X-RateLimit-Limit"] = str(max_requests)

            return response

        return wrapper

    return decorator


def add_security_middleware(app: FastAPI):
    """Add security middleware to the FastAPI app"""

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Add security headers to all responses"""
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

        return response

    logger.info("Security middleware added to application")