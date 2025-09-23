"""
Temporary file cleanup utilities
"""

import os
import tempfile
import glob
import logging
import shutil
from pathlib import Path
from typing import List, Optional
from utils.logger_config import get_logger
from utils.context_managers import get_shutdown_manager

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
        temp_file = tempfile.mktemp(suffix=suffix, prefix=prefix)
        self.register_temp_file(temp_file)
        return temp_file

    def create_temp_dir(self, suffix: str = "", prefix: str = "tmp") -> str:
        """Create a temporary directory and register it for cleanup"""
        temp_dir = tempfile.mkdtemp(suffix=suffix, prefix=prefix)
        self.register_temp_dir(temp_dir)
        return temp_dir

    def cleanup_file(self, file_path: str) -> bool:
        """Clean up a specific temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
                return True
        except Exception as e:
            logger.error(f"Error cleaning up temp file {file_path}: {e}")
        return False

    def cleanup_dir(self, dir_path: str) -> bool:
        """Clean up a specific temporary directory"""
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                logger.debug(f"Cleaned up temp directory: {dir_path}")
                return True
        except Exception as e:
            logger.error(f"Error cleaning up temp directory {dir_path}: {e}")
        return False

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
                f"Failed to clean up {len(failed_files)} files and {len(failed_dirs)} directories"
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
                        f"Error checking/cleaning old temp file {file_path}: {e}"
                    )

            if cleaned_count > 0:
                logger.info(
                    f"Cleaned up {cleaned_count} old temporary files matching pattern: {pattern}"
                )

        except Exception as e:
            logger.error(f"Error during old files cleanup: {e}")


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
