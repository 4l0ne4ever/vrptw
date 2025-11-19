"""
Lightweight profiling helper to instrument pipeline stages without external deps.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional


class PipelineProfiler:
    """Collects timing information for named pipeline stages."""

    def __init__(self):
        self._lock = threading.Lock()
        self._local = threading.local()
        self.reset()

    def reset(self):
        with self._lock:
            self._records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            self._context: Dict[str, Any] = {}

    def set_context(self, **kwargs):
        """Attach shared context metadata (e.g., dataset, n_customers)."""
        with self._lock:
            for key, value in kwargs.items():
                if value is not None:
                    self._context[key] = value

    def clear_context(self, *keys: str):
        with self._lock:
            for key in keys:
                self._context.pop(key, None)

    def profile(self, stage: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to measure a code block."""
        return _ProfileBlock(self, stage, metadata or {})

    def record(self, stage: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        entry = {
            "duration": duration,
            "metadata": metadata or {},
            "context": self._context.copy(),
            "timestamp": time.time(),
            "stage": stage,
        }
        with self._lock:
            self._records[stage].append(entry)
        self._notify_listeners(entry)

    def get_summary(self) -> List[Dict[str, Any]]:
        """Return aggregated stats per stage sorted by total time desc."""
        summary = []
        with self._lock:
            for stage, entries in self._records.items():
                durations = [e["duration"] for e in entries]
                total = sum(durations)
                count = len(durations)
                summary.append(
                    {
                        "stage": stage,
                        "count": count,
                        "total_seconds": total,
                        "avg_seconds": total / count if count else 0.0,
                        "max_seconds": max(durations) if durations else 0.0,
                        "min_seconds": min(durations) if durations else 0.0,
                    }
                )
        summary.sort(key=lambda item: item["total_seconds"], reverse=True)
        return summary

    def format_summary(self, top_n: Optional[int] = None) -> str:
        summary = self.get_summary()
        if top_n is not None:
            summary = summary[:top_n]
        lines = ["=== Pipeline Profiling Summary ==="]
        for item in summary:
            lines.append(
                f"{item['stage']:<30s} count={item['count']:>4} "
                f"total={item['total_seconds']:.4f}s "
                f"avg={item['avg_seconds']:.4f}s "
                f"max={item['max_seconds']:.4f}s"
            )
        return "\n".join(lines)

    @contextmanager
    def listen(self, callback: Callable[[Dict[str, Any]], None]):
        """Temporarily forward stage completions to a callback (thread-local)."""
        stack = self._get_listener_stack()
        stack.append(callback)
        try:
            yield
        finally:
            stack.pop()

    def _get_listener_stack(self) -> List[Callable[[Dict[str, Any]], None]]:
        if not hasattr(self._local, "listeners"):
            self._local.listeners = []
        return self._local.listeners

    def _notify_listeners(self, entry: Dict[str, Any]):
        listeners = getattr(self._local, "listeners", None)
        if not listeners:
            return
        for callback in reversed(listeners):
            try:
                callback(entry)
            except Exception:
                # Listener failures should not break profiling
                continue


class _ProfileBlock:
    def __init__(self, profiler: PipelineProfiler, stage: str, metadata: Dict[str, Any]):
        self.profiler = profiler
        self.stage = stage
        self.metadata = metadata
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start is None:
            return False
        duration = time.perf_counter() - self._start
        self.profiler.record(self.stage, duration, self.metadata)
        return False


pipeline_profiler = PipelineProfiler()

