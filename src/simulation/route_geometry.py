"""
Route geometry helper for replay/map polyline.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict

import requests

from src.runtime_settings import OSRM_BASE_URL, OSRM_TIMEOUT, USE_REAL_ROUTES

class RouteSegment(TypedDict):
    polyline: List[Dict[str, float]]
    duration_s: float
    distance_m: float


_cache: Dict[Tuple[float, float, float, float], RouteSegment] = {}


def _cache_key(origin: Tuple[float, float], destination: Tuple[float, float]) -> Tuple[float, float, float, float]:
    return (round(origin[0], 6), round(origin[1], 6), round(destination[0], 6), round(destination[1], 6))


def get_route_polyline(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    use_real_routes: bool = USE_REAL_ROUTES,
) -> List[Dict[str, float]]:
    return get_route_segment(origin, destination, use_real_routes=use_real_routes)["polyline"]


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in meters (WGS84 lon/lat)."""
    return _haversine_m(lon1, lat1, lon2, lat2)


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    # Great-circle distance; used only as a fallback when OSRM fails/unavailable.
    import math

    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def get_route_segment(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    use_real_routes: bool = USE_REAL_ROUTES,
) -> RouteSegment:
    key = _cache_key(origin, destination)
    if key in _cache:
        return _cache[key]

    polyline_fallback = [{"lat": origin[1], "lon": origin[0]}, {"lat": destination[1], "lon": destination[0]}]

    if not use_real_routes:
        dist_m = _haversine_m(origin[0], origin[1], destination[0], destination[1])
        speed_m_s = 8.33  # ~30 km/h
        seg: RouteSegment = {"polyline": polyline_fallback, "duration_s": max(0.1, dist_m / speed_m_s), "distance_m": dist_m}
        _cache[key] = seg
        return seg

    coordinates = f"{origin[0]},{origin[1]};{destination[0]},{destination[1]}"
    url = f"{OSRM_BASE_URL.rstrip('/')}/route/v1/driving/{coordinates}"
    params = {"overview": "full", "geometries": "geojson"}
    headers = {"User-Agent": "VRPTW/1.0 (routing; +https://github.com/Project-OSRM/osrm-backend)"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=OSRM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        routes = data.get("routes") or []
        if not routes:
            dist_m = _haversine_m(origin[0], origin[1], destination[0], destination[1])
            speed_m_s = 8.33
            seg = {"polyline": polyline_fallback, "duration_s": max(0.1, dist_m / speed_m_s), "distance_m": dist_m}
            _cache[key] = seg
            return seg

        r0 = routes[0]
        duration_s = float(r0.get("duration", 0.0) or 0.0)
        distance_m = float(r0.get("distance", 0.0) or 0.0)
        coords = r0.get("geometry", {}).get("coordinates", [])
        polyline = [{"lat": float(lat), "lon": float(lon)} for lon, lat in coords]
        if not polyline:
            polyline = polyline_fallback
        if duration_s <= 0.0:
            dist_m = distance_m if distance_m > 0 else _haversine_m(origin[0], origin[1], destination[0], destination[1])
            speed_m_s = 8.33
            duration_s = max(0.1, dist_m / speed_m_s)
        if distance_m <= 0.0:
            distance_m = _haversine_m(origin[0], origin[1], destination[0], destination[1])

        seg: RouteSegment = {"polyline": polyline, "duration_s": duration_s, "distance_m": distance_m}
        _cache[key] = seg
        return seg
    except Exception:
        try:
            resp2 = requests.get(url, params=params, headers=headers, timeout=min(OSRM_TIMEOUT * 2, 30.0))
            resp2.raise_for_status()
            data = resp2.json()
            routes = data.get("routes") or []
            if routes:
                r0 = routes[0]
                duration_s = float(r0.get("duration", 0.0) or 0.0)
                distance_m = float(r0.get("distance", 0.0) or 0.0)
                coords = r0.get("geometry", {}).get("coordinates", [])
                polyline = [{"lat": float(lat), "lon": float(lon)} for lon, lat in coords]
                if not polyline:
                    polyline = polyline_fallback
                if duration_s <= 0.0:
                    dist_m = distance_m if distance_m > 0 else _haversine_m(origin[0], origin[1], destination[0], destination[1])
                    duration_s = max(0.1, dist_m / 8.33)
                if distance_m <= 0.0:
                    distance_m = _haversine_m(origin[0], origin[1], destination[0], destination[1])
                seg: RouteSegment = {"polyline": polyline, "duration_s": duration_s, "distance_m": distance_m}
                _cache[key] = seg
                return seg
        except Exception:
            pass
        dist_m = _haversine_m(origin[0], origin[1], destination[0], destination[1])
        speed_m_s = 8.33
        seg = {"polyline": polyline_fallback, "duration_s": max(0.1, dist_m / speed_m_s), "distance_m": dist_m}
        _cache[key] = seg
        return seg

