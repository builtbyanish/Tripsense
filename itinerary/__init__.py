"""
itinerary package
=================
Core itinerary planner services and scheduling logic for TripSense.
Exposes both services and the scheduling engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure member2/src and member3/src are importable
_REPO_ROOT = Path(__file__).resolve().parent
_M2_SRC = _REPO_ROOT / "member2" / "src"
_M3_SRC = _REPO_ROOT / "member3" / "src"

for _p in [_M2_SRC, _M3_SRC]:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Re-export key scheduling engine classes and functions from member3/src/itinerary.py
try:
    from member3.src.itinerary import (
        build_itinerary,
        Itinerary,
        ItineraryDay,
        ItineraryStop,
        _haversine_km,
        _travel_time_hours,
        _sort_by_proximity,
        _PACE_MAX_HOURS,
        _PACE_MAX_PLACES,
        _PACE_TARGETS,
    )
except ImportError:
    try:
        # Fallback if member3 is in sys.path directly
        import importlib.util
        _itin_path = _M3_SRC / "itinerary.py"
        if _itin_path.exists():
            _spec = importlib.util.spec_from_file_location("member3_itinerary", _itin_path)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            build_itinerary = _mod.build_itinerary
            Itinerary = _mod.Itinerary
            ItineraryDay = _mod.ItineraryDay
            ItineraryStop = _mod.ItineraryStop
            _haversine_km = _mod._haversine_km
            _travel_time_hours = _mod._travel_time_hours
            _sort_by_proximity = _mod._sort_by_proximity
            _PACE_MAX_HOURS = _mod._PACE_MAX_HOURS
            _PACE_MAX_PLACES = _mod._PACE_MAX_PLACES
            _PACE_TARGETS = getattr(_mod, '_PACE_TARGETS', {})
    except Exception:
        pass
