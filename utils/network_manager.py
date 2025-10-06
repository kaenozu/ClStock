"""Network connection management utilities
"""

from contextlib import contextmanager
from typing import Any, Optional

from utils.connection_pool import HTTPConnectionPool, get_http_pool
from utils.logger_config import get_logger

logger = get_logger(__name__)


class NetworkConnectionManager:
    """Manages network connections with proper cleanup"""

    def __init__(self):
        self.max_retries = 3
        self.timeout = 30
        self._http_pool: Optional[HTTPConnectionPool] = None

    def _get_http_pool(self) -> HTTPConnectionPool:
        """Get or create HTTP connection pool"""
        if self._http_pool is None:
            self._http_pool = get_http_pool(
                max_connections=10, connection_timeout=self.timeout,
            )
        return self._http_pool

    @contextmanager
    def managed_session(self, **kwargs):
        """Context manager for HTTP sessions with automatic cleanup using connection pool

        Args:
            **kwargs: Arguments to pass to requests.Session()

        """
        session = None
        try:
            # Get connection from pool
            pool = self._get_http_pool()
            session = pool.get_connection()

            # Apply any provided configuration
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)

            yield session

        except Exception as e:
            logger.error(f"Error in managed session: {e}")
            raise
        finally:
            # Return connection to pool
            if session and self._http_pool:
                self._http_pool.return_connection(session)

    def close_all_sessions(self):
        """Close all active sessions and connection pools"""
        if self._http_pool:
            self._http_pool.close_all_connections()
            self._http_pool = None
        logger.info("Closed all network sessions and connection pools")

    def make_request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Make HTTP request with retry logic using connection pool

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            max_retries: Number of retries (defaults to self.max_retries)
            timeout: Request timeout (defaults to self.timeout)
            **kwargs: Additional arguments to pass to requests.request()

        Returns:
            requests.Response object

        Raises:
            requests.RequestException: If all retries fail

        """
        if max_retries is None:
            max_retries = self.max_retries
        if timeout is None:
            timeout = self.timeout

        last_exception = None

        for attempt in range(max_retries + 1):
            session = None
            try:
                # Get connection from pool
                pool = self._get_http_pool()
                session = pool.get_connection()

                response = session.request(method, url, timeout=timeout, **kwargs)
                response.raise_for_status()  # Raise exception for bad status codes
                return response
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}",
                    )
                    # Exponential backoff
                    import time

                    time.sleep(2**attempt)
                else:
                    logger.error(
                        f"Request failed after {max_retries + 1} attempts: {e}",
                    )
            finally:
                # Return connection to pool
                if session and self._http_pool:
                    self._http_pool.return_connection(session)

        raise last_exception


# Global instance
network_manager = NetworkConnectionManager()


def get_network_manager() -> NetworkConnectionManager:
    """Get the global network connection manager instance"""
    return network_manager


@contextmanager
def managed_http_session(**kwargs):
    """Context manager for HTTP sessions

    Args:
        **kwargs: Arguments to pass to requests.Session()

    """
    with network_manager.managed_session(**kwargs) as session:
        yield session
