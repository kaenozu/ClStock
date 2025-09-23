"""
Connection pooling utilities for external services
"""

import logging
import threading
from typing import Dict, Any, Optional
from queue import Queue, Empty
import time

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Generic connection pool for external services"""

    def __init__(self, max_connections: int = 10, connection_timeout: int = 30):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.pool: Queue = Queue(maxsize=max_connections)
        self.active_connections = 0
        self.lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the connection pool"""
        logger.info(
            f"Initializing connection pool with {self.max_connections} connections"
        )

    def _create_connection(self) -> Any:
        """Create a new connection - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _create_connection method")

    def _validate_connection(self, connection: Any) -> bool:
        """Validate if a connection is still alive - to be implemented by subclasses"""
        raise NotImplementedError(
            "Subclasses must implement _validate_connection method"
        )

    def _close_connection(self, connection: Any) -> None:
        """Close a connection - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _close_connection method")

    def get_connection(self) -> Any:
        """Get a connection from the pool"""
        try:
            # Try to get an existing connection
            connection = self.pool.get_nowait()
            if self._validate_connection(connection):
                logger.debug("Reusing existing connection from pool")
                return connection
            else:
                # Connection is invalid, create a new one
                logger.debug("Invalid connection found, creating new one")
                self._close_connection(connection)
                return self._create_connection()
        except Empty:
            # No available connections, create a new one if under limit
            with self.lock:
                if self.active_connections < self.max_connections:
                    self.active_connections += 1
                    logger.debug("Creating new connection")
                    return self._create_connection()
                else:
                    # Wait for a connection to become available
                    logger.debug("Waiting for connection to become available")
                    connection = self.pool.get(timeout=self.connection_timeout)
                    if self._validate_connection(connection):
                        return connection
                    else:
                        self._close_connection(connection)
                        return self._create_connection()

    def return_connection(self, connection: Any) -> None:
        """Return a connection to the pool"""
        if self._validate_connection(connection):
            try:
                self.pool.put_nowait(connection)
                logger.debug("Connection returned to pool")
            except:
                # Pool is full, close the connection
                logger.debug("Pool is full, closing connection")
                self._close_connection(connection)
                with self.lock:
                    self.active_connections = max(0, self.active_connections - 1)
        else:
            # Connection is invalid, close it
            logger.debug("Invalid connection, closing")
            self._close_connection(connection)
            with self.lock:
                self.active_connections = max(0, self.active_connections - 1)

    def close_all_connections(self) -> None:
        """Close all connections in the pool"""
        logger.info("Closing all connections in pool")
        with self.lock:
            while not self.pool.empty():
                try:
                    connection = self.pool.get_nowait()
                    self._close_connection(connection)
                except Empty:
                    break
            self.active_connections = 0


class HTTPConnectionPool(ConnectionPool):
    """Connection pool for HTTP connections"""

    def __init__(
        self,
        max_connections: int = 10,
        connection_timeout: int = 30,
        base_url: str = "",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url
        self.headers = headers or {}
        super().__init__(max_connections, connection_timeout)

    def _create_connection(self) -> Any:
        """Create a new HTTP session"""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            session = requests.Session()

            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )

            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # Set default headers
            session.headers.update(self.headers)

            return session
        except ImportError:
            logger.error("requests library not available for HTTP connection pool")
            raise

    def _validate_connection(self, connection: Any) -> bool:
        """Validate if an HTTP session is still valid"""
        # For HTTP sessions, we consider them always valid
        # In a real implementation, you might want to do a lightweight check
        return connection is not None

    def _close_connection(self, connection: Any) -> None:
        """Close an HTTP session"""
        if connection:
            connection.close()


# Global connection pool instances
_http_pool: Optional[HTTPConnectionPool] = None
_pool_lock = threading.Lock()


def get_http_pool(
    max_connections: int = 10,
    connection_timeout: int = 30,
    base_url: str = "",
    headers: Optional[Dict[str, str]] = None,
) -> HTTPConnectionPool:
    """Get or create a global HTTP connection pool"""
    global _http_pool

    with _pool_lock:
        if _http_pool is None:
            _http_pool = HTTPConnectionPool(
                max_connections=max_connections,
                connection_timeout=connection_timeout,
                base_url=base_url,
                headers=headers,
            )
        return _http_pool


def close_all_pools() -> None:
    """Close all connection pools"""
    global _http_pool

    with _pool_lock:
        if _http_pool:
            _http_pool.close_all_connections()
            _http_pool = None
