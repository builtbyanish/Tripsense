"""
itinerary/services/weather_service.py
======================================
Meteorological forecast and climate analysis service for TripSense.

Features:
1. Live numerical weather prediction model forecasting (Open-Meteo / ECMWF / GFS)
   for near-term trip dates (0–14 days out).
2. Cyclone, squall, and extreme rainfall detection (alerting when severe weather
   threatens travel safety).
3. Historical monthly climate baseline (from tripsense_weather_2023.csv)
   for long-range dates beyond the 14-day forecast horizon.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import date, timedelta
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# District representative coordinates for regional meteorological lookups
DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    "visakhapatnam": (17.6868, 83.2185),
    "vizag": (17.6868, 83.2185),
    "tirupati": (13.6288, 79.4192),
    "chittoor": (13.2172, 79.1003),
    "vijayawada": (16.5062, 80.6480),
    "krishna": (16.5062, 80.6480),
    "ntr": (16.5062, 80.6480),
    "guntur": (16.3067, 80.4365),
    "kurnool": (15.8281, 78.0373),
    "nellore": (14.4426, 79.9865),
    "kakinada": (16.9891, 82.2475),
    "east godavari": (17.0005, 81.8040),
    "rajahmundry": (17.0005, 81.8040),
    "kadapa": (14.4674, 78.8241),
    "ysr": (14.4674, 78.8241),
    "anantapur": (14.6819, 77.6006),
    "srikakulam": (18.2949, 83.8938),
    "vizianagaram": (18.1067, 83.3956),
    "west godavari": (16.7107, 81.0952),
    "eluru": (16.7107, 81.0952),
    "prakasam": (15.5057, 80.0499),
    "ongole": (15.5057, 80.0499),
    "bapatla": (15.9042, 80.4674),
    "palnadu": (16.2333, 79.9667),
    "konaseema": (16.5787, 82.0061),
    "araku": (18.3273, 82.8775),
    "alluri sitharama raju": (18.3273, 82.8775),
    "parvathipuram": (18.7794, 83.4283),
    "annamayya": (14.0573, 78.7523),
    "sri sathya sai": (14.1673, 77.8115),
    "nandyal": (15.4886, 78.4836),
    "all andhra pradesh": (16.5062, 80.6480),
}

# Cache for Open-Meteo forecasts: (round_lat, round_lon) -> (fetch_time, forecast_dict)
_FORECAST_CACHE: dict[tuple[float, float], dict[str, dict]] = {}

def get_live_forecast_for_coords(lat: float, lon: float) -> dict[str, dict]:
    """
    Fetch numerical weather model forecast from Open-Meteo (ECMWF & GFS ensemble).
    Returns mapping: date_str (YYYY-MM-DD) -> {precip_mm, wind_gust_kmh, weather_code, temp_max, temp_min}
    """
    key = (round(lat, 2), round(lon, 2))
    if key in _FORECAST_CACHE:
        return _FORECAST_CACHE[key]

    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}"
            f"&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max"
            f"&forecast_days=14&timezone=auto"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "TripSense-Weather/1.0"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            daily = data.get("daily", {})
            times = daily.get("time", [])
            precip = daily.get("precipitation_sum", [])
            w_code = daily.get("weather_code", [])
            gusts = daily.get("wind_gusts_10m_max", [])
            t_max = daily.get("temperature_2m_max", [])
            t_min = daily.get("temperature_2m_min", [])

            res = {}
            for i, t_str in enumerate(times):
                res[t_str] = {
                    "precip_mm": float(precip[i]) if i < len(precip) and precip[i] is not None else 0.0,
                    "weather_code": int(w_code[i]) if i < len(w_code) and w_code[i] is not None else 0,
                    "wind_gust_kmh": float(gusts[i]) if i < len(gusts) and gusts[i] is not None else 0.0,
                    "temp_max": float(t_max[i]) if i < len(t_max) and t_max[i] is not None else 28.0,
                    "temp_min": float(t_min[i]) if i < len(t_min) and t_min[i] is not None else 22.0,
                }
            _FORECAST_CACHE[key] = res
            return res
    except Exception as exc:
        log.warning("Open-Meteo forecast fetch failed for (%s, %s): %s", lat, lon, exc)
        return {}

_TEMP_THRESHOLDS = {
    "very_hot":    35.0,
    "hot":         32.0,
    "warm":        28.0,
    "comfortable": 24.0,
    "cool":        18.0,
}
_RAIN_THRESHOLDS = {
    "very_heavy": 200.0,
    "heavy":      100.0,
    "moderate":   50.0,
    "light":      20.0,
}

_MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def _temp_label(t: float) -> str:
    if t >= _TEMP_THRESHOLDS["very_hot"]:
        return "very hot"
    if t >= _TEMP_THRESHOLDS["hot"]:
        return "hot"
    if t >= _TEMP_THRESHOLDS["warm"]:
        return "warm"
    if t >= _TEMP_THRESHOLDS["comfortable"]:
        return "comfortable"
    if t >= _TEMP_THRESHOLDS["cool"]:
        return "cool"
    return "cold"


def _rain_label(r: float) -> str:
    if r >= _RAIN_THRESHOLDS["very_heavy"]:
        return "very heavy rainfall expected"
    if r >= _RAIN_THRESHOLDS["heavy"]:
        return "heavy rainfall expected"
    if r >= _RAIN_THRESHOLDS["moderate"]:
        return "moderate rainfall"
    if r >= _RAIN_THRESHOLDS["light"]:
        return "light rainfall"
    return "little or no rainfall"


def _is_outdoor_suitable(temp: float, rain: float) -> bool:
    return temp < _TEMP_THRESHOLDS["very_hot"] and rain < _RAIN_THRESHOLDS["heavy"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_weather_for_place(
    place_id: str,
    month: int,
    weather_df: pd.DataFrame,
    target_date: Optional[date] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    *args,
    **kwargs,
) -> Optional[dict]:
    if target_date is None:
        target_date = kwargs.get("target_date")
    if lat is None:
        lat = kwargs.get("lat")
    if lon is None:
        lon = kwargs.get("lon")
    """
    Return weather summary for a place, utilizing live numerical prediction models
    (Open-Meteo) if target_date falls within the 14-day forecast window, or
    falling back to historical monthly climate tables.
    """
    # 1. Try numerical prediction forecast if target_date and coordinates available
    if target_date is not None and lat is not None and lon is not None:
        try:
            forecasts = get_live_forecast_for_coords(float(lat), float(lon))
            d_str = target_date.strftime("%Y-%m-%d")
            if d_str in forecasts:
                fc = forecasts[d_str]
                rain_mm = fc["precip_mm"]
                gust_kmh = fc["wind_gust_kmh"]
                w_code = fc["weather_code"]
                t_max = fc["temp_max"]

                is_cyclone = (
                    (w_code in (95, 96, 99) and (rain_mm >= 35.0 or gust_kmh >= 45.0))
                    or (rain_mm >= 65.0)
                    or (gust_kmh >= 70.0)
                )
                is_heavy_rain = (rain_mm >= 25.0) or (w_code in (65, 82) and rain_mm >= 15.0)

                if is_cyclone:
                    return {
                        "place_id": place_id,
                        "month": month,
                        "month_name": _MONTH_NAMES.get(month, str(month)),
                        "avg_temp_c": round(t_max, 1),
                        "total_rainfall_mm": round(rain_mm, 1),
                        "temp_label": "cyclonic squall",
                        "rain_label": f"torrential rain ({rain_mm:.1f} mm), {gust_kmh:.0f} km/h wind gusts",
                        "outdoor_suitable": False,
                        "is_severe": True,
                        "is_cyclone": True,
                        "alert_msg": f"Cyclone Alert: Torrential rain ({rain_mm:.1f} mm) and {gust_kmh:.0f} km/h cyclonic gusts. Travel NOT advised.",
                        "weather_note": f"Cyclone Warning: {rain_mm:.1f} mm rain, {gust_kmh:.0f} km/h gusts — Travel Not Advised",
                        "data_source_note": "Numerical Weather Prediction Model (ECMWF & GFS / Open-Meteo live forecast).",
                    }
                elif is_heavy_rain:
                    return {
                        "place_id": place_id,
                        "month": month,
                        "month_name": _MONTH_NAMES.get(month, str(month)),
                        "avg_temp_c": round(t_max, 1),
                        "total_rainfall_mm": round(rain_mm, 1),
                        "temp_label": "heavy rain",
                        "rain_label": f"heavy rainfall ({rain_mm:.1f} mm)",
                        "outdoor_suitable": False,
                        "is_severe": True,
                        "is_cyclone": False,
                        "alert_msg": f"Heavy Rainfall Alert: {rain_mm:.1f} mm precipitation expected. Outdoor travel discouraged.",
                        "weather_note": f"Heavy Rainfall ({rain_mm:.1f} mm) — Travel Discouraged",
                        "data_source_note": "Numerical Weather Prediction Model (ECMWF & GFS / Open-Meteo live forecast).",
                    }
                else:
                    return {
                        "place_id": place_id,
                        "month": month,
                        "month_name": _MONTH_NAMES.get(month, str(month)),
                        "avg_temp_c": round(t_max, 1),
                        "total_rainfall_mm": round(rain_mm, 1),
                        "temp_label": _temp_label(t_max),
                        "rain_label": _rain_label(rain_mm),
                        "outdoor_suitable": _is_outdoor_suitable(t_max, rain_mm),
                        "is_severe": False,
                        "is_cyclone": False,
                        "weather_note": f"{_temp_label(t_max).capitalize()} ({t_max:.1f}°C), {_rain_label(rain_mm)}",
                        "data_source_note": "Numerical Weather Prediction Model (ECMWF & GFS / Open-Meteo live forecast).",
                    }
        except Exception as e:
            log.warning("Forecast lookup failed for place %s: %s", place_id, e)
    """
    Return historical weather summary for a specific place and month.

    Parameters
    ----------
    place_id   : e.g. "AP0001"
    month      : 1–12
    weather_df : DataFrame from data_service.load_weather()

    Returns
    -------
    dict or None if not found.

    Keys
    ----
    place_id, month, month_name, avg_temp_c, total_rainfall_mm,
    temp_label, rain_label, outdoor_suitable, data_source_note
    """
    if weather_df.empty:
        return None

    mask = (
        (weather_df["place_id"].astype(str) == str(place_id))
        & (weather_df["month"] == month)
    )
    rows = weather_df[mask]
    if rows.empty:
        return None

    row = rows.iloc[0]
    temp = float(row.get("avg_temp_c", float("nan")))
    rain = float(row.get("total_rainfall_mm", float("nan")))

    if pd.isna(temp) or pd.isna(rain):
        return None

    return {
        "place_id":         place_id,
        "month":            month,
        "month_name":       _MONTH_NAMES.get(month, str(month)),
        "avg_temp_c":       round(temp, 1),
        "total_rainfall_mm": round(rain, 1),
        "temp_label":       _temp_label(temp),
        "rain_label":       _rain_label(rain),
        "outdoor_suitable": _is_outdoor_suitable(temp, rain),
        "data_source_note": (
            "Historical monthly climate average (TripSense dataset). "
            "Not a real-time or forecast value."
        ),
    }


def get_weather_for_places(
    place_ids: list[str],
    month: int,
    weather_df: pd.DataFrame,
    target_date: Optional[date] = None,
    place_coords: Optional[dict[str, tuple[float, float]]] = None,
    *args,
    **kwargs,
) -> dict[str, dict]:
    if target_date is None:
        target_date = kwargs.get("target_date")
    if place_coords is None:
        place_coords = kwargs.get("place_coords")
    """
    Return weather summaries for multiple place IDs, incorporating live model forecasts
    when target_date and place_coords are provided.
    """
    result = {}
    for pid in place_ids:
        lat, lon = (None, None)
        if place_coords and pid in place_coords:
            lat, lon = place_coords[pid]
        w = get_weather_for_place(pid, month, weather_df, target_date=target_date, lat=lat, lon=lon)
        if w is not None:
            result[pid] = w
    return result


def check_trip_severe_weather(
    district: str,
    start_date: date,
    n_days: int,
) -> dict:
    """
    Check whether any date in the trip window faces severe weather (cyclone, heavy deluge,
    or dangerous gale winds) using numerical weather prediction models.
    """
    d_clean = district.lower().strip()
    coords = None
    for k, c_pair in DISTRICT_COORDS.items():
        if k in d_clean or d_clean in k:
            coords = c_pair
            break
    if coords is None:
        coords = DISTRICT_COORDS.get("all andhra pradesh", (16.5062, 80.6480))

    lat, lon = coords
    forecasts = get_live_forecast_for_coords(lat, lon)
    if not forecasts:
        return {"has_severe_weather": False, "is_cyclone": False}

    severe_dates = []
    max_rain = 0.0
    max_gust = 0.0
    cyclone_found = False

    for i in range(max(1, min(n_days, 14))):
        cur_d = start_date + timedelta(days=i)
        d_str = cur_d.strftime("%Y-%m-%d")
        if d_str in forecasts:
            fc = forecasts[d_str]
            r = fc["precip_mm"]
            g = fc["wind_gust_kmh"]
            c_code = fc["weather_code"]
            is_cyc = (c_code in (95, 96, 99) and (r >= 35.0 or g >= 45.0)) or (r >= 65.0) or (g >= 70.0)
            is_heavy = (r >= 25.0) or (c_code in (65, 82) and r >= 15.0)

            if is_cyc or is_heavy:
                severe_dates.append(cur_d.strftime("%A, %d %b"))
                max_rain = max(max_rain, r)
                max_gust = max(max_gust, g)
                if is_cyc:
                    cyclone_found = True

    if severe_dates:
        return {
            "has_severe_weather": True,
            "is_cyclone": cyclone_found,
            "district": district,
            "affected_dates": severe_dates,
            "max_rain_mm": round(max_rain, 1),
            "max_gust_kmh": round(max_gust, 1),
            "banner_title": "Severe Weather & Cyclone Warning — Travel Not Advised" if cyclone_found else "Heavy Rainfall Alert — Outdoor Travel Discouraged",
            "banner_body": (
                f"Numerical weather prediction models (ECMWF & GFS) predict severe cyclonic activity, "
                f"extreme torrential rainfall (up to {max_rain:.1f} mm), and dangerous wind gusts (up to {max_gust:.0f} km/h) "
                f"across {district} on {', '.join(severe_dates)}."
                if cyclone_found else
                f"Meteorological models forecast heavy rainfall (up to {max_rain:.1f} mm) across {district} on {', '.join(severe_dates)}."
            ),
            "safety_advisory": "Coastal visits, beach promenades, waterfalls, and highway road journeys are hazardous. Rescheduling your travel dates is strongly advised.",
        }

    return {"has_severe_weather": False, "is_cyclone": False}


def get_district_weather_summary(
    district: str,
    month: int,
    weather_df: pd.DataFrame,
) -> Optional[dict]:
    """
    Return an aggregated weather summary for all places in a district for a month.
    """
    if weather_df.empty:
        return None

    mask = (
        weather_df["district"].str.lower().str.contains(district.lower(), na=False)
        & (weather_df["month"] == month)
    )
    rows = weather_df[mask]
    if rows.empty:
        # Fallback to entire state average if district not found
        rows = weather_df[weather_df["month"] == month]
        if rows.empty:
            rows = weather_df

    avg_temp = float(rows["avg_temp_c"].mean()) if "avg_temp_c" in rows else 25.0
    avg_rain = float(rows["total_rainfall_mm"].mean()) if "total_rainfall_mm" in rows else 15.0

    return {
        "district":          district,
        "month":             month,
        "month_name":        _MONTH_NAMES.get(month, str(month)),
        "avg_temp_c":        round(avg_temp, 1),
        "total_rainfall_mm": round(avg_rain, 1),
        "avg_rainfall_mm":   round(avg_rain, 1),
        "temp_label":        _temp_label(avg_temp),
        "rain_label":        _rain_label(avg_rain),
        "comfort_text":      f"{_temp_label(avg_temp).title()}, {_rain_label(avg_rain)}",
        "outdoor_suitable":  _is_outdoor_suitable(avg_temp, avg_rain),
        "n_places_sampled":  len(rows),
        "data_source_note":  (
            f"Historical monthly climate average across {len(rows)} places in {district} "
            "(TripSense dataset). Grounded in Member 3 calendar intelligence."
        ),
    }


def district_climate_summary(
    district: str,
    month: int,
    weather_df: pd.DataFrame,
) -> dict:
    """Convenience alias for district climate summary returning safe defaults."""
    res = get_district_weather_summary(district, month, weather_df)
    if res is None:
        return {
            "district": district,
            "month": month,
            "avg_temp_c": 25.5,
            "total_rainfall_mm": 15.0,
            "comfort_text": "Comfortable weather window",
            "data_source_note": "Estimated regional baseline.",
        }
    return res

