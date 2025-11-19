"""
Utilities for detecting standard Solomon instances and providing fast-path cache keys.
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SolomonFastPathEntry(Dict[str, object]):
    """
    Typed dictionary describing a Solomon fast-path entry.
    Keys:
        name: Instance name (e.g., C101)
        cache_key_hash: Deterministic cache key hash for this instance
        signature: Coordinate signature (MD5 of rounded/sorted coordinates)
        num_points: Number of points (including depot)
        source_path: Path to the dataset file
    """


class SolomonFastPath:
    """Detects Solomon benchmark instances and provides deterministic cache metadata."""

    _dataset_lookup: Dict[str, SolomonFastPathEntry] = {}
    _lookup_loaded: bool = False

    _INSTANCE_PATTERN = re.compile(r"^(C|R|RC)\d{3}$", re.IGNORECASE)

    _BASE_DIR = Path(__file__).resolve().parents[2]
    _DATASET_DIR = _BASE_DIR / "data" / "datasets" / "solomon"
    _CATALOG_PATH = _BASE_DIR / "data" / "datasets" / "catalog.json"

    @classmethod
    def match(
        cls,
        dataset_name: Optional[str],
        coordinates: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[SolomonFastPathEntry]:
        """
        Returns a fast-path entry if the dataset name matches a standard Solomon instance
        and the provided coordinates align with the canonical signature.
        """
        if not dataset_name:
            return None

        cls._ensure_lookup()
        normalized_name = dataset_name.strip().upper()
        entry = cls._dataset_lookup.get(normalized_name)
        if entry is None:
            return None

        # Optional sanity check to prevent mismatches: ensure coordinate signature matches
        if coordinates:
            signature = cls._compute_signature(coordinates)
            if signature != entry["signature"]:
                return None

        return entry

    @classmethod
    def get_known_instances(cls) -> List[str]:
        """Return the list of known Solomon instance names."""
        cls._ensure_lookup()
        return sorted(cls._dataset_lookup.keys())

    @classmethod
    def _ensure_lookup(cls) -> None:
        if cls._lookup_loaded:
            return

        cls._dataset_lookup = {}
        dataset_entries = cls._load_catalog_entries()

        for name, file_path in dataset_entries.items():
            coords = cls._extract_coordinates(file_path)
            if not coords:
                continue

            signature = cls._compute_signature(coords)
            cache_key_hash = cls._compute_fast_cache_key(name)

            cls._dataset_lookup[name] = SolomonFastPathEntry(
                name=name,
                cache_key_hash=cache_key_hash,
                signature=signature,
                num_points=len(coords),
                source_path=str(file_path),
            )

        cls._lookup_loaded = True

    @classmethod
    def _load_catalog_entries(cls) -> Dict[str, Path]:
        """
        Load dataset entries from catalog.json (preferred) or fallback to directory listing.
        Returns mapping of NAME -> file path.
        """
        entries: Dict[str, Path] = {}

        if cls._CATALOG_PATH.exists():
            try:
                with cls._CATALOG_PATH.open("r", encoding="utf-8") as f:
                    catalog = json.load(f)
                for item in catalog.get("datasets", []):
                    if str(item.get("type")).lower() != "solomon":
                        continue
                    name = str(item.get("name", "")).strip().upper()
                    if not cls._INSTANCE_PATTERN.match(name):
                        continue
                    filepath = item.get("filepath")
                    if not filepath:
                        continue
                    path_obj = cls._resolve_path(filepath)
                    if path_obj.exists():
                        entries[name] = path_obj
            except Exception:
                # Fallback to directory scan if catalog parsing fails
                entries = {}

        if entries:
            return entries

        # Fallback: scan dataset directory
        if cls._DATASET_DIR.exists():
            for file in cls._DATASET_DIR.glob("*.json"):
                name = file.stem.upper()
                if cls._INSTANCE_PATTERN.match(name):
                    entries[name] = file

        return entries

    @classmethod
    def _resolve_path(cls, filepath: str) -> Path:
        path_obj = Path(filepath)
        if not path_obj.is_absolute():
            path_obj = cls._BASE_DIR / filepath
        return path_obj

    @staticmethod
    def _extract_coordinates(file_path: Path) -> Optional[List[Tuple[float, float]]]:
        """Extract depot + customer coordinates from a dataset file."""
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        coords: List[Tuple[float, float]] = []
        depot = data.get("depot")
        customers = data.get("customers", [])

        if depot and "x" in depot and "y" in depot:
            coords.append((float(depot["x"]), float(depot["y"])))

        for customer in customers:
            if "x" in customer and "y" in customer:
                coords.append((float(customer["x"]), float(customer["y"])))

        return coords if coords else None

    @staticmethod
    def _compute_signature(coordinates: List[Tuple[float, float]]) -> str:
        """Compute deterministic signature for a list of coordinates."""
        rounded = sorted((round(float(x), 6), round(float(y), 6)) for x, y in coordinates)
        coords_str = str(rounded)
        return hashlib.md5(coords_str.encode()).hexdigest()

    @staticmethod
    def _compute_fast_cache_key(dataset_name: str) -> str:
        """Compute deterministic cache key hash for a Solomon instance."""
        normalized = dataset_name.strip().upper()
        raw_key = f"solomon::{normalized}"
        return hashlib.md5(raw_key.encode()).hexdigest()[:16]


