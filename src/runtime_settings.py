"""
Runtime settings for fullstack backend/solver integration.
"""

from __future__ import annotations

import os

OSRM_BASE_URL = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org")
OSRM_TIMEOUT = float(os.getenv("OSRM_TIMEOUT", "10"))
USE_REAL_ROUTES = os.getenv("USE_REAL_ROUTES", "true").lower() == "true"
OSRM_MAX_BATCH_SIZE = int(os.getenv("OSRM_MAX_BATCH_SIZE", "15"))
OSRM_MAX_CONCURRENCY = int(os.getenv("OSRM_MAX_CONCURRENCY", "1"))
REPLAY_DWELL_SPEEDUP = float(os.getenv("REPLAY_DWELL_SPEEDUP", "20"))
REPLAY_DWELL_STEP_MIN = float(os.getenv("REPLAY_DWELL_STEP_MIN", "0.2"))
REPLAY_WAIT_TARGET_EVENTS = int(os.getenv("REPLAY_WAIT_TARGET_EVENTS", "120"))

SOLOMON_WARM_CACHE_ENABLED = os.getenv("SOLOMON_WARM_CACHE_ENABLED", "true").lower() == "true"
SOLOMON_WARM_CACHE_ASYNC = os.getenv("SOLOMON_WARM_CACHE_ASYNC", "true").lower() == "true"

_limit_raw = os.getenv("SOLOMON_WARM_CACHE_LIMIT")
if _limit_raw is None:
    SOLOMON_WARM_CACHE_LIMIT = None
else:
    try:
        SOLOMON_WARM_CACHE_LIMIT = int(_limit_raw)
    except ValueError:
        SOLOMON_WARM_CACHE_LIMIT = None

