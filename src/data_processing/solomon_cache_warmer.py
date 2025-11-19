"""
Utility to warm Solomon distance matrix caches proactively.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.config.settings import (
    SOLOMON_WARM_CACHE_ENABLED,
    SOLOMON_WARM_CACHE_ASYNC,
    SOLOMON_WARM_CACHE_LIMIT,
)
from src.data_processing.solomon_fast_path import SolomonFastPath, SolomonFastPathEntry


class SolomonCacheWarmer:
    """Background warmer that pre-builds distance caches for Solomon instances."""

    _lock = threading.Lock()
    _thread: Optional[threading.Thread] = None
    _running = False
    _completed = False
    _last_summary: Dict[str, object] = {
        "warmed": [],
        "skipped": [],
        "failed": [],
        "duration_sec": 0.0,
        "limit": None,
        "dataset_names": None,
    }

    @classmethod
    def ensure_started(
        cls,
        *,
        async_mode: Optional[bool] = None,
        force: bool = False,
        limit: Optional[int] = None,
        dataset_names: Optional[Sequence[str]] = None,
        cache_dir_override: Optional[str] = None,
    ) -> bool:
        """
        Ensure the warmer is running (at most once per process).

        Returns True if a warm-up run was started.
        """
        if not SOLOMON_WARM_CACHE_ENABLED and not force:
            return False

        if async_mode is None:
            async_mode = SOLOMON_WARM_CACHE_ASYNC

        with cls._lock:
            if cls._running:
                return False
            if cls._completed and not force:
                return False
            cls._running = True
            cls._completed = False

        def _target():
            cls._execute(limit=limit, dataset_names=dataset_names, cache_dir_override=cache_dir_override, forced=force)

        if async_mode:
            thread = threading.Thread(
                target=_target,
                name="SolomonCacheWarmer",
                daemon=True,
            )
            cls._thread = thread
            thread.start()
            return True

        _target()
        return True

    @classmethod
    def warm_cache_sync(
        cls,
        *,
        limit: Optional[int] = None,
        dataset_names: Optional[Sequence[str]] = None,
        cache_dir_override: Optional[str] = None,
        force: bool = True,
    ) -> Dict[str, object]:
        """
        Run warm-up synchronously (primarily for tests and CLI usage).
        """
        cls.ensure_started(
            async_mode=False,
            force=force,
            limit=limit,
            dataset_names=dataset_names,
            cache_dir_override=cache_dir_override,
        )
        return cls.get_status()

    @classmethod
    def get_status(cls) -> Dict[str, object]:
        """Return the last warm-up summary and current status."""
        with cls._lock:
            status = {
                "running": cls._running,
                "completed": cls._completed,
                "thread_alive": bool(cls._thread and cls._thread.is_alive()),
            }
            return {**cls._last_summary, **status}

    @classmethod
    def _execute(
        cls,
        *,
        limit: Optional[int],
        dataset_names: Optional[Sequence[str]],
        cache_dir_override: Optional[str],
        forced: bool,
    ) -> None:
        logger = logging.getLogger(__name__)
        start = time.time()
        warmed: List[str] = []
        skipped: List[str] = []
        failed: List[str] = []

        try:
            resolved_limit = limit if limit is not None else SOLOMON_WARM_CACHE_LIMIT
            entries = cls._resolve_entries(dataset_names)
            if resolved_limit is not None:
                entries = entries[: max(resolved_limit, 0)]

            logger.info(
                "Starting Solomon cache warm-up for %d instances (limit=%s)...",
                len(entries),
                resolved_limit if resolved_limit is not None else "∞",
            )

            for entry in entries:
                dataset_name = entry["name"]
                try:
                    coordinates = cls._load_coordinates(entry)
                    if not coordinates:
                        skipped.append(dataset_name)
                        continue

                    # Import lazily to avoid circular dependency at load time
                    from src.data_processing.distance import DistanceCalculator

                    calculator = DistanceCalculator(
                        dataset_type="solomon",
                        dataset_name=dataset_name,
                        use_real_routes=False,
                    )
                    if cache_dir_override:
                        Path(cache_dir_override).mkdir(parents=True, exist_ok=True)
                        calculator.cache_dir = cache_dir_override

                    calculator.calculate_distance_matrix(coordinates, use_cache=True)
                    warmed.append(dataset_name)
                    logger.info("Warm cache ✅ %s", dataset_name)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Warm cache failed for %s: %s", dataset_name, exc)
                    failed.append(dataset_name)

            duration = time.time() - start
            logger.info(
                "Solomon cache warm-up completed in %.2fs (warmed=%d, skipped=%d, failed=%d)",
                duration,
                len(warmed),
                len(skipped),
                len(failed),
            )
        finally:
            duration = time.time() - start
            with cls._lock:
                cls._running = False
                cls._completed = True
                cls._last_summary = {
                    "warmed": warmed,
                    "skipped": skipped,
                    "failed": failed,
                    "duration_sec": round(duration, 3),
                    "limit": limit if limit is not None else SOLOMON_WARM_CACHE_LIMIT,
                    "dataset_names": list(dataset_names) if dataset_names else None,
                    "forced": forced,
                }

    @classmethod
    def _resolve_entries(cls, dataset_names: Optional[Sequence[str]]) -> List[SolomonFastPathEntry]:
        """Resolve which Solomon entries should be warmed."""
        target_names = None
        if dataset_names:
            target_names = {name.strip().upper() for name in dataset_names}

        entries: List[SolomonFastPathEntry] = []
        for name in SolomonFastPath.get_known_instances():
            if target_names and name not in target_names:
                continue
            entry = SolomonFastPath.match(name)
            if entry:
                entries.append(entry)

        return entries

    @staticmethod
    def _load_coordinates(entry: SolomonFastPathEntry):
        """Load coordinates for a Solomon dataset."""
        source_path = Path(entry["source_path"])
        if not source_path.exists():
            return None

        try:
            with source_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

        coords = []
        depot = data.get("depot")
        if depot and "x" in depot and "y" in depot:
            coords.append((float(depot["x"]), float(depot["y"])))

        for customer in data.get("customers", []):
            if "x" in customer and "y" in customer:
                coords.append((float(customer["x"]), float(customer["y"])))

        return coords if coords else None


