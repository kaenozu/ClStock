"""Shared resource monitoring utilities for process management."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass(frozen=True)
class SystemUsage:
    """Snapshot of aggregated system resource usage."""

    cpu_percent: float
    memory_percent: float
    memory_total: float


class ResourceMonitor:
    """Caches system resource usage for short intervals to avoid blocking calls."""

    def __init__(self, cache_ttl: float = 0.5) -> None:
        self._cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._cached_usage: Optional[tuple[float, SystemUsage]] = None

    def get_system_usage(self) -> SystemUsage:
        """Return the current CPU and memory usage, cached for ``cache_ttl`` seconds."""

        now = time.monotonic()

        with self._lock:
            if self._cached_usage is not None:
                cached_at, usage = self._cached_usage
                if now - cached_at < self._cache_ttl:
                    return usage

            cpu_percent = psutil.cpu_percent(interval=None)
            virtual_memory = psutil.virtual_memory()
            usage = SystemUsage(
                cpu_percent=cpu_percent,
                memory_percent=virtual_memory.percent,
                memory_total=float(virtual_memory.total),
            )
            self._cached_usage = (now, usage)
            return usage

    def invalidate(self) -> None:
        """Clear any cached system usage snapshot."""

        with self._lock:
            self._cached_usage = None
