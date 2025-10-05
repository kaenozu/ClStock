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
_API_KEYS_CACHE: Optional[Dict[str, str]] = None

DUMMY_DEV_API_KEY = "clstock-dev-placeholder"
DUMMY_ADMIN_API_KEY = "clstock-admin-placeholder"
UNCONFIGURED_ROLE = "unconfigured"
_TEST_TOKEN_FLAG = "API_ENABLE_TEST_TOKENS"
_TEST_TOKENS = {
    "admin_token_secure_2024": "administrator",
    "user_token_basic_2024": "user",
}


def _redact_secret(value: Optional[str], visible: int = 4) -> str:
    """Create a redacted representation for sensitive values."""

    if value is None:
        return "<none>"

    if value == "":
        return "<empty>"

    visible = max(1, visible)

    if len(value) <= visible:
        return "***"

    if len(value) <= visible * 2:
        return f"{value[:visible]}***"

    return f"{value[:visible]}***{value[-visible:]}"


def reset_env_token_cache() -> None:
    """Reset cached environment token values and warning tracking."""

    _env_cache.clear()
    _logged_missing_env_vars.clear()
    _env_tokens_cache.clear()
    _env_tokens_cache_sources.clear()
    global _API_KEYS_CACHE
    _API_KEYS_CACHE = None


def _set_api_keys_cache(new_keys: Dict[str, str]) -> Dict[str, str]:
    """Store API keys in the local cache."""

    global _API_KEYS_CACHE
    _API_KEYS_CACHE = dict(new_keys)
    return _API_KEYS_CACHE


def _initialize_api_keys() -> Dict[str, str]:
    """Initialize API keys from config or environment variables lazily."""

    try:
        from config.secrets import API_KEYS as secrets_api_keys  # type: ignore

        if not secrets_api_keys:
            raise RuntimeError("config.secrets.API_KEYS is empty")

        return dict(secrets_api_keys)
    except ImportError:
        dev_key = os.getenv("CLSTOCK_DEV_KEY")
        admin_key = os.getenv("CLSTOCK_ADMIN_KEY")

        missing_vars = [
            name
            for name, value in (
                ("CLSTOCK_DEV_KEY", dev_key),
                ("CLSTOCK_ADMIN_KEY", admin_key),
            )
            if not value
        ]

        if missing_vars:
            logger.error(
                "Missing environment variables for API keys: %s. "
                "Using temporary placeholder keys; configure secrets before production use.",
                ", ".join(missing_vars),
            )
            return {
                DUMMY_DEV_API_KEY: UNCONFIGURED_ROLE,
                DUMMY_ADMIN_API_KEY: UNCONFIGURED_ROLE,
            }

        return {
            dev_key: "developer",
            admin_key: "administrator",
        }


def _get_api_keys(force_refresh: bool = False) -> Dict[str, str]:
    """Return the configured API keys, loading them lazily when required."""

    global _API_KEYS_CACHE

    if force_refresh or _API_KEYS_CACHE is None:
        _set_api_keys_cache(_initialize_api_keys())

    return _API_KEYS_CACHE if _API_KEYS_CACHE is not None else {}


def _get_env_with_warning(var_name: str) -> Optional[str]:
    """環境変数を取得し、存在しない場合は警告をログに出力します。

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

ALLOW_TEST_TOKENS = False
TEST_TOKENS: Dict[str, str] = dict(_TEST_TOKENS)


def configure_security(
    api_keys: Optional[Dict[str, str]] = None,
    test_tokens: Optional[Dict[str, str]] = None,
    enable_test_tokens: Optional[bool] = None,
) -> None:
    """Configure security settings, primarily for testing purposes."""

    global TEST_TOKENS, ALLOW_TEST_TOKENS

    if api_keys is not None:
        if not api_keys:
            raise ValueError("api_keys cannot be empty")
        _set_api_keys_cache(api_keys)
    else:
        _get_api_keys(force_refresh=True)

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

    api_keys = _get_api_keys()

    if token not in api_keys:
        logger.warning(
            "Invalid API key attempt: %s",
            _redact_secret(token),
        )
        raise HTTPException(status_code=401, detail="Invalid API key")

    user_type = api_keys[token]
    if user_type == UNCONFIGURED_ROLE:
        logger.error(
            "Placeholder API key used while security configuration is incomplete."
        )
        raise HTTPException(
            status_code=503,
            detail="API keys are not configured. Contact the system administrator.",
        )
    logger.info(f"API key verified for user type: {user_type}")
    return user_type


def _is_test_token_flag_enabled() -> bool:
    """Check if the environment flag for test tokens is enabled."""

    flag_value = os.getenv(_TEST_TOKEN_FLAG, "").strip().lower()
    return flag_value in {"1", "true", "yes", "on"}


def _should_include_test_tokens() -> bool:
    """テスト用トークンを許可するかを判定"""

    return ALLOW_TEST_TOKENS or _is_test_token_flag_enabled()


def _build_allowed_tokens() -> Dict[str, str]:
    """許可されたトークンの一覧を構築"""

    tokens = dict(_get_api_keys())

    env_tokens = _get_env_tokens_from_cache()
    tokens.update(env_tokens)

    if _should_include_test_tokens():
        if _is_test_token_flag_enabled():
            logger.warning(
                "Test tokens enabled via environment flag. Do not use in production."
            )
        elif ALLOW_TEST_TOKENS:
            logger.warning(
                "Test tokens enabled via configuration. Do not use in production."
            )
        tokens.update(TEST_TOKENS)

    return tokens


def verify_token(token: str) -> str:
    """トークン検証関数（互換性のため）"""
    if not token:
        raise HTTPException(status_code=401, detail="Invalid token")

    allowed_tokens = _build_allowed_tokens()

    if token not in allowed_tokens:
        logger.warning(
            "Invalid token attempt: %s",
            _redact_secret(token),
        )
        raise HTTPException(status_code=401, detail="Invalid token")

    user_type = allowed_tokens[token]
    if user_type == UNCONFIGURED_ROLE:
        logger.error(
            "Placeholder API token used while security configuration is incomplete."
        )
        raise HTTPException(
            status_code=503,
            detail="API tokens are not configured. Contact the system administrator.",
        )
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

