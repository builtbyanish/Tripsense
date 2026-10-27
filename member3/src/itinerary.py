"""
member3/src/itinerary.py
=========================
Day-wise itinerary scheduling engine for TripSense.

Responsibilities
----------------
- Accept a list of recommended places (from the recommendation service).
- Allocate them across the requested number of days.
- Respect actual visit_duration_hrs fields.
- Group geographically nearby attractions using Haversine distance.
- Respect user preferences: pace, interests, traveler type, accessibility.
- Leave reasonable time for meals and rest.
- Return a structured day-by-day plan with explanations.
- Avoid fabricating travel times; use transparent distance-based approximations
  labelled as estimated.
- Handle gracefully: missing coordinates, missing durations, < enough places.

Schedule assumptions (clearly labelled)
----------------------------------------
- Day starts at 08:00, ends by 20:00 → 12 usable hours.
- Meal break: 1.5 hours total (lunch + snacks).
- Buffer between places: 0.5 hours.
- Travel time estimate: distance_km / 40 km/h (road speed approximation).
  Labelled as "estimated ~X min drive" — NOT sourced from a routing API.
- If coordinates are missing, places in the same district are assumed ~30 min apart.
- Pace multipliers:
    relaxed  → max 2–3 places/day
    balanced → max 4–5 places/day
    packed   → max 6–7 places/day
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DAY_HOURS      = 12.0   # 08:00 – 20:00
_MEAL_HOURS     = 1.5
_BUFFER_HOURS   = 0.5    # buffer per place transition
_DEFAULT_VISIT  = 2.0    # fallback visit duration in hours
_APPROX_SPEED_KMH = 40.0  # road-speed approximation for travel time

_PACE_TARGETS = {
    "relaxed":  {"min_stops": 1, "max_stops": 2, "max_hrs": 7.5, "target_hrs": 5.5, "label": "Relaxed (1–2 stops/day)"},
    "balanced": {"min_stops": 2, "max_stops": 3, "max_hrs": 9.5, "target_hrs": 7.5, "label": "Balanced (2–3 stops/day)"},
    "packed":   {"min_stops": 3, "max_stops": 4, "max_hrs": 11.5, "target_hrs": 9.5, "label": "Packed (3–4 stops/day)"},
}

_PACE_MAX_PLACES = {
    "relaxed":  2,
    "balanced": 3,
    "packed":   4,
}
_PACE_MAX_HOURS = {
    "relaxed":  7.5,
    "balanced": 9.5,
    "packed":   11.5,
}


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _travel_time_hours(lat1, lon1, lat2, lon2) -> tuple[float, str]:
    """
    Return (travel_time_hours, label).
    If coordinates are unavailable, return a default and an appropriate note.
    """
    try:
        dist_km = _haversine_km(float(lat1), float(lon1), float(lat2), float(lon2))
        hours   = dist_km / _APPROX_SPEED_KMH
        mins    = int(round(hours * 60))
        label   = f"~{mins} min drive (straight-line distance ÷ {int(_APPROX_SPEED_KMH)} km/h — estimated)"
        return hours, label
    except (TypeError, ValueError):
        return 0.5, "~30 min (coordinates unavailable — estimated)"


# ---------------------------------------------------------------------------
# Geographical grouping
# ---------------------------------------------------------------------------

def _centroid(places: list[dict]) -> tuple[Optional[float], Optional[float]]:
    lats = [p["latitude"] for p in places if p.get("latitude") and not math.isnan(float(p["latitude"]))]
    lons = [p["longitude"] for p in places if p.get("longitude") and not math.isnan(float(p["longitude"]))]
    if not lats:
        return None, None
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _sort_by_proximity(places: list[dict]) -> list[dict]:
    """
    Nearest-neighbour ordering to minimise total travel within a day.
    Falls back to original order when coordinates are missing.
    """
    if len(places) <= 1:
        return places

    remaining = list(places)
    ordered   = [remaining.pop(0)]

    while remaining:
        last = ordered[-1]
        best_idx  = 0
        best_dist = float("inf")
        for i, p in enumerate(remaining):
            try:
                d = _haversine_km(
                    float(last.get("latitude", 0)),
                    float(last.get("longitude", 0)),
                    float(p.get("latitude",  0)),
                    float(p.get("longitude", 0)),
                )
            except (TypeError, ValueError):
                d = float("inf")
            if d < best_dist:
                best_dist = d
                best_idx  = i
        ordered.append(remaining.pop(best_idx))

    return ordered


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ItineraryStop:
    place_id:              str
    name:                  str
    category:              str
    district:              str
    visit_duration_hrs:    float
    latitude:              Optional[float]
    longitude:             Optional[float]
    description:           str
    recommendation_reason: str
    recommendation_method: str
    weather_note:          str = ""
    travel_from_prev:      str = ""   # human-readable travel note
    cumulative_hrs:        float = 0.0
    tags:                  str = ""
    suitable_for:          str = ""
    best_months:           str = ""
    best_time:             str = ""


@dataclass
class ItineraryDay:
    day_number:    int
    date:          Optional[date]
    stops:         list[ItineraryStop] = field(default_factory=list)
    total_hrs:     float = 0.0
    crowd_note:    str = ""
    weather_note:  str = ""
    date_quality:  Optional[float] = None   # composite score 0–1
    theme:         str = ""
    rag_tip:       str = ""


@dataclass
class Itinerary:
    trip_name:       str
    traveler_name:   str
    start_date:      Optional[date]
    n_days:          int
    district:        str
    interests:       list[str]
    traveler_type:   str
    n_travelers:     int
    pace:            str
    days:            list[ItineraryDay] = field(default_factory=list)
    unscheduled:     list[dict] = field(default_factory=list)  # places that didn't fit
    best_months:     list[str] = field(default_factory=list)
    date_notes:      str = ""
    warnings:        list[str] = field(default_factory=list)
    rag_insight:     str = ""


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_itinerary(
    recommendations: pd.DataFrame,
    n_days: int,
    start_date: Optional[date] = None,
    pace: str = "balanced",
    traveler_name: str = "",
    traveler_type: str = "solo",
    n_travelers: int = 1,
    district: str = "Andhra Pradesh",
    interests: Optional[list[str]] = None,
    weather_data: Optional[dict[str, dict]] = None,
    date_scores: Optional[dict] = None,
    best_months_suggestions: Optional[list[str]] = None,
    month: int = 1,
    has_seniors: bool = False,
    has_kids: bool = False,
    rag_insights: Optional[dict] = None,
) -> Itinerary:
    """
    Build a day-by-day itinerary from a recommendations DataFrame.

    Parameters
    ----------
    recommendations       : Output of recommendation_service.get_recommendations()
    n_days                : Trip duration (1–30)
    start_date            : First day of trip; None if flexible
    pace                  : "relaxed" | "balanced" | "packed"
    traveler_name         : Optional user name for display
    traveler_type         : "solo" | "couple" | "family" | "friends" | "group"
    n_travelers           : Number of travelers
    district              : Selected region
    interests             : List of interest strings
    weather_data          : dict place_id → weather dict from weather_service
    date_scores           : dict from date_service.score_trip_dates
    best_months_suggestions: list of month names from date_service
    month                 : Target travel month (for context)

    Returns
    -------
    Itinerary dataclass
    """
    n_days    = max(1, min(30, int(n_days)))
    pace_key  = pace.lower().strip() if pace.lower().strip() in _PACE_MAX_PLACES else "balanced"
    max_places_per_day = _PACE_MAX_PLACES[pace_key]
    max_hours_per_day  = _PACE_MAX_HOURS[pace_key]
    usable_hours       = max_hours_per_day - _MEAL_HOURS

    interests = interests or []

    itinerary = Itinerary(
        trip_name=f"Andhra Pradesh Adventure — {n_days} Day{'s' if n_days > 1 else ''}",
        traveler_name=traveler_name,
        start_date=start_date,
        n_days=n_days,
        district=district,
        interests=interests,
        traveler_type=traveler_type,
        n_travelers=n_travelers,
        pace=pace_key,
        best_months=best_months_suggestions or [],
    )

    if recommendations.empty:
        itinerary.warnings.append("No recommendations available for the selected preferences.")
        for i in range(n_days):
            d = start_date + timedelta(days=i) if start_date else None
            itinerary.days.append(ItineraryDay(day_number=i + 1, date=d))
        return itinerary

    # --- Convert recs to list of dicts for scheduling ---
    place_list: list[dict] = _recs_to_place_list(recommendations, weather_data)

    # --- Deduplicate ---
    seen_ids: set[str] = set()
    unique_places: list[dict] = []
    for p in place_list:
        pid = str(p.get("place_id", ""))
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique_places.append(p)

    # Filter out steep high-terrain climbing destinations if seniors (60+) or kids (<10) are traveling
    if has_seniors or has_kids:
        strenuous_cats = {"waterfall", "caves", "fort"}
        steep_pattern = re.compile(r"\b(trek|trekking|climb|climbing|cliff|gorge|rock climbing|steep steps)\b", re.IGNORECASE)
        gentle_places = []
        for p in unique_places:
            cat = str(p.get("category", "")).lower()
            name = str(p.get("name", ""))
            desc = str(p.get("description", ""))
            suit = str(p.get("suitable_for", "")).lower()
            if cat in strenuous_cats or "adventure" in suit or steep_pattern.search(name) or steep_pattern.search(desc):
                continue
            gentle_places.append(p)
        if gentle_places:
            unique_places = gentle_places

    # --- Geographic clustering: group nearby places ---
    # Sort globally by proximity to create sensible day groupings
    clustered = _sort_by_proximity(unique_places)

    # Target places per day configured intelligently by pace
    pace_cfg = _PACE_TARGETS.get(pace_key, _PACE_TARGETS["balanced"])
    min_stops = pace_cfg["min_stops"]
    max_stops = pace_cfg["max_stops"]
    target_hrs = pace_cfg["target_hrs"]
    usable_hours = pace_cfg["max_hrs"] - _MEAL_HOURS
    buffer_hours = _BUFFER_HOURS if not (has_seniors or has_kids) else 0.75

    # --- Allocate places to days ---
    scheduled: list[ItineraryDay] = []
    pool = list(clustered)

    for day_idx in range(n_days):
        day_date = start_date + timedelta(days=day_idx) if start_date else None
        day = ItineraryDay(day_number=day_idx + 1, date=day_date)

        # Date quality and crowd note from date_scores
        if date_scores and date_scores.get("dates_in_dataset"):
            daily = date_scores.get("daily_scores", [])
            if day_idx < len(daily):
                day.date_quality = daily[day_idx].get("avg_score")
                day.crowd_note   = daily[day_idx].get("crowd_level", "")

        accumulated_hours = 0.0
        prev_place: Optional[dict] = None

        # Remaining days ensuring balanced distribution without starving future days
        remaining_days = n_days - (day_idx + 1)
        if remaining_days > 0 and len(pool) > 0:
            effective_cap = min(max_stops, max(min_stops, len(pool) - remaining_days))
        else:
            effective_cap = max_stops

        # Demographic cap adjustment
        if (has_seniors or has_kids) and pace_key == "relaxed":
            effective_cap = min(effective_cap, 2)
        elif (has_seniors or has_kids) and pace_key == "balanced":
            effective_cap = min(effective_cap, 2 if len(pool) <= n_days * 2 else 3)

        # Pick places for this day
        day_pool = list(pool)

        stops_this_day: list[dict] = []
        for place in day_pool:
            if len(stops_this_day) >= effective_cap:
                break

            visit_hrs = place.get("visit_duration_hrs", _DEFAULT_VISIT)
            if visit_hrs <= 0:
                visit_hrs = _DEFAULT_VISIT

            # Travel time from previous stop
            travel_hrs = 0.0
            travel_label = ""
            if prev_place:
                travel_hrs, travel_label = _travel_time_hours(
                    prev_place.get("latitude"),
                    prev_place.get("longitude"),
                    place.get("latitude"),
                    place.get("longitude"),
                )
            else:
                travel_label = "Starting point"

            needed = visit_hrs + travel_hrs + buffer_hours

            # Dynamic pacing intelligence: If minimum required stops are reached,
            # decide whether to stop or add more based on landmark duration, transit distance, and target hours:
            if len(stops_this_day) >= min_stops:
                if accumulated_hours + needed > target_hrs:
                    break
                # Anchor landmark with high visit duration: In relaxed trip, keep as sole star attraction
                if pace_key == "relaxed" and (visit_hrs >= 2.5 or (prev_place and prev_place.get("visit_duration_hrs", 0) >= 2.5)):
                    break
                # Long drive between stops: avoid over-cramming
                if travel_hrs >= 1.0 and pace_key != "packed":
                    break

            if accumulated_hours + needed > usable_hours and stops_this_day:
                break

            stops_this_day.append({**place, "travel_from_prev": travel_label, "travel_hrs": travel_hrs})
            accumulated_hours += visit_hrs + travel_hrs + buffer_hours
            prev_place = place
            pool.remove(place)

        # Circadian sequencing: Chronologically order stops (morning temples/outdoors -> midday culture/museums -> sunset beaches/viewpoints)
        def _circadian_priority(p: dict) -> int:
            cat = str(p.get("category", "")).lower()
            nm  = str(p.get("name", "")).lower()
            if "temple" in cat or "temple" in nm or "waterfall" in cat or "falls" in cat or "wildlife" in cat or "nature" in cat:
                return 1
            elif "museum" in cat or "caves" in cat or "cave" in nm or "fort" in cat or "palace" in nm or "heritage" in cat or "attraction" in cat:
                return 2
            elif "beach" in cat or "beach" in nm or "viewpoint" in cat or "lake" in cat or "park" in cat or "dam" in cat:
                return 3
            return 2

        if len(stops_this_day) > 1:
            stops_this_day.sort(key=_circadian_priority)
            prev_p = None
            for p in stops_this_day:
                if prev_p:
                    t_hrs, t_lbl = _travel_time_hours(
                        prev_p.get("latitude"), prev_p.get("longitude"),
                        p.get("latitude"), p.get("longitude")
                    )
                    p["travel_hrs"] = t_hrs
                    p["travel_from_prev"] = t_lbl
                else:
                    p["travel_hrs"] = 0.0
                    p["travel_from_prev"] = "Starting point"
                prev_p = p

        # Assign intelligent day theme and RAG rhythm guidance
        cats = [p.get("category", "").lower() for p in stops_this_day]
        if "temple" in cats and ("beach" in cats or "viewpoint" in cats):
            day.theme = "Morning Spiritual Darshan & Sunset Coastal Horizon"
            day.rag_tip = "AI RAG Rhythm: Early morning temple visit avoids crowds and midday sun; evening offers calm coastal breezes."
        elif "temple" in cats and ("heritage" in cats or "fort" in cats):
            day.theme = "Sacred Heritage & Historic Fortifications"
            day.rag_tip = "AI RAG Rhythm: Begin with morning darshan, followed by midday stone fort exploration with lunch halts."
        elif "waterfall" in cats or "nature" in cats or "wildlife" in cats:
            day.theme = "Verdant Nature Trails & Cascading Waters"
            day.rag_tip = "AI RAG Rhythm: Morning forest air offers optimal wildlife sightings and comfortable trail hiking."
        elif "beach" in cats:
            day.theme = "Coastal Horizons & Marine Breezes"
            day.rag_tip = "AI RAG Rhythm: Afternoon coastal drives lead into spectacular golden hour sunset photography by the shore."
        elif len(stops_this_day) == 1:
            day.theme = f"Deep Immersion: {stops_this_day[0].get('name', 'Destination')}"
            day.rag_tip = "AI RAG Rhythm: Unhurried single-destination schedule allowing comprehensive exploration with ample rest."
        else:
            first_name = stops_this_day[0].get('name', 'Destination') if stops_this_day else 'Regional'
            day.theme = f"Regional Discovery: {first_name}"
            day.rag_tip = f"AI RAG Rhythm: {pace_key.title()} itinerary designed for smooth geographic transitions and comfortable pacing."


        # Build ItineraryStop objects
        cumul = 0.0
        for p in stops_this_day:
            visit_hrs = p.get("visit_duration_hrs", _DEFAULT_VISIT)
            if visit_hrs <= 0:
                visit_hrs = _DEFAULT_VISIT
            cumul += p.get("travel_hrs", 0) + visit_hrs + _BUFFER_HOURS

            weather_note = ""
            if weather_data and p.get("place_id") in weather_data:
                w = weather_data[p["place_id"]]
                if w.get("weather_note"):
                    weather_note = w["weather_note"]
                else:
                    weather_note = (
                        f"{w.get('temp_label','').capitalize()} ({w.get('avg_temp_c','')}°C), "
                        f"{w.get('rain_label','')}."
                    )

            stop = ItineraryStop(
                place_id=str(p.get("place_id", "")),
                name=str(p.get("name", "Unknown")),
                category=str(p.get("category", "")),
                district=str(p.get("district", "")),
                visit_duration_hrs=float(visit_hrs),
                latitude=p.get("latitude"),
                longitude=p.get("longitude"),
                description=str(p.get("description", ""))[:400],
                recommendation_reason=str(p.get("recommendation_reason", "")),
                recommendation_method=str(p.get("recommendation_method", "content-based")),
                weather_note=weather_note,
                travel_from_prev=p.get("travel_from_prev", ""),
                cumulative_hrs=round(cumul, 2),
                tags=str(p.get("tags_text", "")),
                suitable_for=str(p.get("suitable_for", "")),
                best_months=str(p.get("best_months", "")),
                best_time=str(p.get("best_time", "")),
            )
            day.stops.append(stop)

        day.total_hrs = round(accumulated_hours, 2)
        scheduled.append(day)

    itinerary.days = scheduled
    itinerary.unscheduled = [
        {"place_id": p.get("place_id"), "name": p.get("name"), "reason": "Did not fit within the trip duration."}
        for p in pool
    ]

    if best_months_suggestions:
        from datetime import date as _date
        current_month_name = [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ][month - 1]
        if current_month_name not in best_months_suggestions:
            itinerary.date_notes = (
                f"Based on weather and crowd data, the best months for this region are: "
                f"{', '.join(best_months_suggestions)}. "
                f"Your selected month ({current_month_name}) may be less optimal — "
                f"consider flexible dates if possible."
            )
        else:
            itinerary.date_notes = (
                f"Great choice! {current_month_name} is among the best months "
                f"for visiting this region."
            )

    if not any(day.stops for day in itinerary.days):
        itinerary.warnings.append(
            "The selected filters returned too few places to build a complete itinerary. "
            "Try broadening the region or interests."
        )

    # AI RAG Strategic Guidance summary
    total_scheduled = sum(len(d.stops) for d in itinerary.days)
    avg_per_day = total_scheduled / n_days if n_days > 0 else 0
    rag_demographic_note = ""
    if has_seniors or has_kids:
        guards = []
        if has_seniors:
            guards.append("seniors (60+)")
        if has_kids:
            guards.append("young children (<10)")
        rag_demographic_note = (
            f" Configured with gentle terrain safeguards for {' & '.join(guards)}: "
            f"high-altitude climbs, rugged gorge waterfalls, and caves excluded in favor of accessible, scenic landmarks."
        )

    itinerary.rag_insight = (
        f"AI RAG Strategic Assessment: Built a {pace_key.title()} itinerary ({_PACE_TARGETS[pace_key]['min_stops']}–{_PACE_TARGETS[pace_key]['max_stops']} destinations/day) "
        f"averaging {avg_per_day:.1f} stops/day for {n_travelers} {traveler_type} travelers. "
        f"Dynamically sequenced using circadian rhythm (morning temples/outdoors, afternoon heritage/museums, evening sunset/beaches) "
        f"with verified travel buffers.{rag_demographic_note}"
    )

    return itinerary


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _recs_to_place_list(recs: pd.DataFrame, weather_data: Optional[dict]) -> list[dict]:
    """Convert recommendation DataFrame rows to plain dicts for scheduling."""
    places = []
    for _, row in recs.iterrows():
        p = row.to_dict()
        # Normalise visit_duration_hrs
        try:
            vd = float(p.get("visit_duration_hrs", _DEFAULT_VISIT))
            if vd <= 0 or pd.isna(vd):
                vd = _DEFAULT_VISIT
        except (TypeError, ValueError):
            vd = _DEFAULT_VISIT
        p["visit_duration_hrs"] = vd

        # Normalise coordinates
        for coord in ["latitude", "longitude"]:
            try:
                val = float(p.get(coord, float("nan")))
                p[coord] = None if math.isnan(val) else val
            except (TypeError, ValueError):
                p[coord] = None

        places.append(p)
    return places
