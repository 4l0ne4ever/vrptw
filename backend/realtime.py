"""
Realtime bridge primitives shared across backend modules.
"""

from __future__ import annotations

import queue
from typing import Any, Dict

OutboundMessage = Dict[str, Any]

# Single in-process bridge: producers from threads put messages here,
# FastAPI async loop consumes and broadcasts to websocket clients.
outbound_queue: "queue.Queue[OutboundMessage]" = queue.Queue()

