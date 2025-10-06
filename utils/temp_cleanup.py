"""Temporary file cleanup utilities
"""

import glob
import os
import shutil
import tempfile
from typing import Callable, Iterable, List, Set

from utils.context_managers import get_shutdown_manager
from utils.logger_config import get_logger

logger = get_logger(__name__)


class TempFileCleanup:
    """Manages cleanup of temporary files"""

    def __init__(self):
        self.temp_files: List[str] = []
        self.temp_dirs: List[str] = []

        # Register cleanup handler with shutdown manager
        shutdown_manager = get_shutdown_manager()
        shutdown_manager.register_shutdown_handler(self.cleanup_all)

    def register_temp_file(self, file_path: str):
        """Register a temporary file for cleanup"""
        self.temp_files.append(file_path)
        logger.debug(f"Registered temp file for cleanup: {file_path}")

    def register_temp_dir(self, dir_path: str):
        """Register a temporary directory for cleanup"""
        self.temp_dirs.append(dir_path)
        logger.debug(f"Registered temp directory for cleanup: {dir_path}")

    def create_temp_file(self, suffix: str = "", prefix: str = "tmp") -> str:
        """Create a temporary file and register it for cleanup"""
        fd, temp_file = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        os.close(fd)
        self.register_temp_file(temp_file)
        return temp_file

    def create_temp_dir(self, suffix: str = "", prefix: str = "tmp") -> str:
        """Create a temporary directory and register it for cleanup"""
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
        self.register_temp_dir(temp_dir)
        return temp_dir

    def _cleanup_path(
        self, path: str, remove_func: Callable[[str], None], description: str,
    ) -> bool:
        """Clean up a path using the provided removal function"""
        try:
            if not os.path.exists(path):
                logger.debug(f"{description} already removed: {path}")
                return True

            remove_func(path)
            logger.debug(f"Cleaned up {description}: {path}")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up {description} {path}: {e}")
            return False

    def cleanup_file(self, file_path: str) -> bool:
        """Clean up a specific temporary file"""
        return self._cleanup_path(file_path, os.remove, "temp file")

    def cleanup_dir(self, dir_path: str) -> bool:
        """Clean up a specific temporary directory"""
        return self._cleanup_path(dir_path, shutil.rmtree, "temp directory")

    def cleanup_all(self):
        """Clean up all registered temporary files and directories"""
        logger.info("Cleaning up temporary files and directories...")

        # Clean up files
        failed_files = []
        for file_path in self.temp_files:
            if not self.cleanup_file(file_path):
                failed_files.append(file_path)

        # Clean up directories
        failed_dirs = []
        for dir_path in self.temp_dirs:
            if not self.cleanup_dir(dir_path):
                failed_dirs.append(dir_path)

        # Remove successfully cleaned items from lists
        self.temp_files = [f for f in self.temp_files if f in failed_files]
        self.temp_dirs = [d for d in self.temp_dirs if d in failed_dirs]

        if failed_files or failed_dirs:
            logger.warning(
                f"Failed to clean up {len(failed_files)} files and {len(failed_dirs)} directories",
            )
        else:
            logger.info("All temporary files and directories cleaned up successfully")

    def cleanup_old_files(self, pattern: str, days_old: int = 1):
        """Clean up old temporary files matching a pattern"""
        try:
            import time

            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 60 * 60)

            files = glob.glob(pattern)
            cleaned_count = 0

            for file_path in files:
                try:
                    if os.path.getctime(file_path) < cutoff_time:
                        os.remove(file_path)
                        cleaned_count += 1
                        logger.debug(f"Cleaned up old temp file: {file_path}")
                except Exception as e:
                    logger.warning(
                        f"Error checking/cleaning old temp file {file_path}: {e}",
                    )

            if cleaned_count > 0:
                logger.info(
                    f"Cleaned up {cleaned_count} old temporary files matching pattern: {pattern}",
                )

        except Exception as e:
            logger.error(f"Error during old files cleanup: {e}")

    def cleanup_unnecessary_files(
        self, base_dir: str, required_entries: Iterable[str],
    ) -> List[str]:
        """Remove files or directories in base_dir that are not required."""
        if not os.path.isdir(base_dir):
            logger.debug("Skipped cleanup for non-existent directory: %s", base_dir)
            return []

        required_set: Set[str] = set(required_entries)
        removed: List[str] = []

        for entry in os.listdir(base_dir):
            if entry in required_set:
                continue

            full_path = os.path.join(base_dir, entry)

            if os.path.isdir(full_path):
                if self.cleanup_dir(full_path):
                    removed.append(full_path)
            elif self.cleanup_file(full_path):
                removed.append(full_path)

        removed.sort()
        if removed:
            logger.info(
                "Removed %d unnecessary entries from %s", len(removed), base_dir,
            )
        else:
            logger.debug("No unnecessary entries found in %s", base_dir)

        return removed

    def cleanup_by_pattern(self, pattern: str) -> int:
        """Remove files or directories that match the provided glob pattern."""
        removed_count = 0
        paths = glob.glob(pattern)

        for path in paths:
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed_count += 1
                logger.debug(f"Removed path via pattern cleanup: {path}")
            except FileNotFoundError:
                logger.debug(f"Path already removed before pattern cleanup: {path}")
            except Exception as exc:
                logger.warning(
                    f"Failed to remove path {path} during pattern cleanup: {exc}",
                )

        if removed_count > 0:
            logger.info(
                "Removed %d items matching pattern '%s' during temp cleanup",
                removed_count,
                pattern,
            )

        return removed_count


# Global instance
temp_cleanup = TempFileCleanup()


def get_temp_cleanup() -> TempFileCleanup:
    """Get the global temporary file cleanup instance"""
    return temp_cleanup


def register_temp_file(file_path: str):
    """Register a temporary file for cleanup"""
    temp_cleanup.register_temp_file(file_path)


def register_temp_dir(dir_path: str):
    """Register a temporary directory for cleanup"""
    temp_cleanup.register_temp_dir(dir_path)


def create_temp_file(suffix: str = "", prefix: str = "tmp") -> str:
    """Create a temporary file and register it for cleanup"""
    return temp_cleanup.create_temp_file(suffix, prefix)


def create_temp_dir(suffix: str = "", prefix: str = "tmp") -> str:
    """Create a temporary directory and register it for cleanup"""
    return temp_cleanup.create_temp_dir(suffix, prefix)
