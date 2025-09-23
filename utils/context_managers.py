"""
Context managers for proper resource cleanup
"""

import signal
import atexit
import threading
import logging
import os
import sys
from typing import Optional, Callable, Any
from contextlib import contextmanager
from utils.logger_config import get_logger

logger = get_logger(__name__)


class GracefulShutdownManager:
    """Manages graceful shutdown of the application"""

    def __init__(self):
        self.shutdown_handlers = []
        self.is_shutting_down = False
        self._setup_signal_handlers()
        # Register the shutdown method to be called on Python interpreter exit
        atexit.register(self.shutdown)

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, AttributeError):
            # Not available on all platforms (e.g., Windows)
            logger.warning("Signal handlers not available on this platform")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.shutdown()
        # Exit the process after shutdown
        sys.exit(0)

    def register_shutdown_handler(self, handler: Callable[[], None]):
        """Register a function to be called during shutdown"""
        self.shutdown_handlers.append(handler)

    def shutdown(self):
        """Perform graceful shutdown"""
        if self.is_shutting_down:
            return

        self.is_shutting_down = True
        logger.info("Initiating graceful shutdown...")

        # Call all registered shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error during shutdown handler execution: {e}")

        # Perform final cleanup
        self._final_cleanup()

    def _final_cleanup(self):
        """Final cleanup operations"""
        # Shutdown cache
        try:
            from utils.cache import shutdown_cache

            shutdown_cache()
            logger.info("Cache shutdown completed")
        except Exception as e:
            logger.error(f"Error during cache shutdown: {e}")

        # Close logging
        try:
            logging.shutdown()
            logger.info("Logging shutdown completed")
        except Exception as e:
            logger.error(f"Error during logging shutdown: {e}")

        logger.info("Graceful shutdown completed")


# Global instance
shutdown_manager = GracefulShutdownManager()


@contextmanager
def managed_resource(resource_factory: Callable, cleanup_func: Callable):
    """
    Context manager for resources that need explicit cleanup

    Args:
        resource_factory: Function that creates the resource
        cleanup_func: Function that cleans up the resource
    """
    resource = None
    try:
        resource = resource_factory()
        yield resource
    finally:
        if resource is not None:
            try:
                cleanup_func(resource)
            except Exception as e:
                logger.error(f"Error cleaning up resource: {e}")


@contextmanager
def network_connection(connection_factory: Callable):
    """
    Context manager for network connections

    Args:
        connection_factory: Function that creates the network connection
    """
    connection = None
    try:
        connection = connection_factory()
        yield connection
    except Exception as e:
        logger.error(f"Network connection error: {e}")
        raise
    finally:
        if connection is not None:
            try:
                if hasattr(connection, "close"):
                    connection.close()
                elif hasattr(connection, "disconnect"):
                    connection.disconnect()
            except Exception as e:
                logger.error(f"Error closing network connection: {e}")


@contextmanager
def file_lock(lock_file_path: str, timeout: int = 30):
    """
    Context manager for file-based locking

    Args:
        lock_file_path: Path to the lock file
        timeout: Timeout in seconds
    """
    import time

    lock_acquired = False
    start_time = time.time()

    try:
        # Try to acquire lock
        while not lock_acquired and (time.time() - start_time) < timeout:
            try:
                # Try to create lock file
                fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                lock_acquired = True
            except FileExistsError:
                # Lock file exists, wait and retry
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error creating lock file: {e}")
                raise

        if not lock_acquired:
            raise TimeoutError(f"Could not acquire lock within {timeout} seconds")

        yield lock_file_path

    finally:
        # Release lock
        if lock_acquired and os.path.exists(lock_file_path):
            try:
                os.remove(lock_file_path)
            except Exception as e:
                logger.error(f"Error removing lock file: {e}")


def register_shutdown_handler(handler: Callable[[], None]):
    """Register a function to be called during shutdown"""
    shutdown_manager.register_shutdown_handler(handler)


def get_shutdown_manager() -> GracefulShutdownManager:
    """Get the global shutdown manager instance"""
    return shutdown_manager
