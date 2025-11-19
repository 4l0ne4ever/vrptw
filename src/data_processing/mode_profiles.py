"""
Mode-specific configuration profiles for distance calculation and GA defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModeProfile:
    """Describes runtime preferences for a dataset mode."""

    key: str
    label: str
    description: str
    default_ga_preset: str
    use_real_routes: bool
    warm_cache: bool
    penalty_weight: int


_MODE_PROFILES: Dict[str, ModeProfile] = {
    "hanoi": ModeProfile(
        key="hanoi",
        label="Hanoi Mode",
        description="Real-road routing with OSRM, adaptive traffic, and city-specific penalties.",
        default_ga_preset="standard",
        use_real_routes=True,
        warm_cache=False,
        penalty_weight=1200,
    ),
    "solomon": ModeProfile(
        key="solomon",
        label="Solomon Mode",
        description="Deterministic Euclidean benchmarks with warmed caches and strict penalties.",
        default_ga_preset="benchmark",
        use_real_routes=False,
        warm_cache=True,
        penalty_weight=5000,
    ),
}


def _normalize_key(dataset_type: str) -> str:
    """Normalize dataset_type strings to canonical profile keys."""
    key = (dataset_type or "").strip().lower()
    if key.startswith("hanoi"):
        return "hanoi"
    if key.startswith("solomon"):
        return "solomon"
    return key


def get_mode_profile(dataset_type: str) -> ModeProfile:
    """Return the mode profile for the provided dataset type."""
    key = _normalize_key(dataset_type)
    return _MODE_PROFILES.get(key, _MODE_PROFILES["hanoi"])


def list_mode_profiles() -> List[ModeProfile]:
    """Return all available mode profiles."""
    return list(_MODE_PROFILES.values())


