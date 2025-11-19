"""
Streamlit helpers for displaying pipeline progress in real time.
"""

from __future__ import annotations

from contextlib import ContextDecorator
from typing import Dict, List, Optional

import streamlit as st

from src.core.pipeline_profiler import pipeline_profiler


def _format_stage(entry: Dict[str, object]) -> str:
    """Generate a concise, human-friendly line for a profiling entry."""
    stage = str(entry.get("stage", ""))
    duration_ms = float(entry.get("duration", 0.0)) * 1000.0
    metadata = entry.get("metadata") or {}
    context = entry.get("context") or {}

    friendly = _STAGE_LABELS.get(stage, stage.replace("_", " ").replace(".", " ").title())

    details: List[str] = []
    n_points = metadata.get("n_points") or metadata.get("batch_size")
    if n_points:
        details.append(f"{int(n_points)} pts")

    dataset = context.get("distance_dataset")
    if dataset:
        details.append(str(dataset).upper())

    if metadata.get("fast_path"):
        details.append(f"fast-path: {metadata['fast_path']}")

    detail_str = f" [{' | '.join(details)}]" if details else ""
    return f"- {friendly}{detail_str} ({duration_ms:.1f} ms)"


_STAGE_LABELS = {
    "data.create_vrp_problem": "Prepare VRP problem",
    "distance.matrix.build": "Build distance matrix",
    "distance.calculate": "Distance calculation",
    "distance.cache_lookup": "Check caches",
    "distance.cache_save": "Store caches",
    "distance.osrm_table_service": "OSRM table service",
    "distance.osrm_batch": "OSRM batch",
    "distance.osrm_request": "OSRM request",
    "distance.osrm_fallback": "OSRM fallback",
    "distance.haversine_vectorized": "Vectorized haversine",
    "distance.euclidean_vectorized": "Vectorized euclidean",
}


class PipelineProgress(ContextDecorator):
    """
    Context manager that displays stage-by-stage updates using Streamlit status blocks.
    """

    def __init__(
        self,
        label: str = "Processing...",
        *,
        success_label: str = "Completed",
        failure_label: str = "Failed",
        expanded: bool = False,
    ):
        self.label = label
        self.success_label = success_label
        self.failure_label = failure_label
        self.expanded = expanded
        self._status_ctx = None
        self._status = None
        self._listener_ctx = None

    def __enter__(self):
        self._status_ctx = st.status(self.label, expanded=self.expanded)
        self._status = self._status_ctx.__enter__()
        self._listener_ctx = pipeline_profiler.listen(self._handle_stage)
        self._listener_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._listener_ctx:
            self._listener_ctx.__exit__(exc_type, exc_val, exc_tb)

        if self._status:
            if exc_type is None:
                self._status.update(label=self.success_label, state="complete", expanded=True)
            else:
                self._status.update(label=self.failure_label, state="error", expanded=True)

        if self._status_ctx:
            self._status_ctx.__exit__(exc_type, exc_val, exc_tb)

    def write(self, message: str):
        """Allow callers to add custom lines."""
        if self._status:
            self._status.write(message)

    def _handle_stage(self, entry: Dict[str, object]):
        if self._status:
            self._status.write(_format_stage(entry))


