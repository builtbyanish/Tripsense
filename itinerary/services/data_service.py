"""
itinerary/services/data_service.py
====================================
Centralised dataset loading and validation layer for the TripSense itinerary
planner.  All downstream modules import from here so path resolution and
column-sanity checks happen in exactly one place.

Path strategy
-------------
All paths are resolved relative to the repository root (the directory that
contains this file's grandparent: itinerary/ → Tripsense/).  The module
never relies on the current working directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository root resolution
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()          # …/itinerary/services/data_service.py
_REPO_ROOT = _HERE.parents[2]            # …/Tripsense/

# ---------------------------------------------------------------------------
# Default file paths  (relative to repo root)
# ---------------------------------------------------------------------------
PLACES_CSV         = _REPO_ROOT / "data" / "places"      / "tripsense_places_v2.csv"
INTERACTIONS_CSV   = _REPO_ROOT / "data" / "interactions" / "tripsense_reviews_dataset.csv"
WEATHER_CSV        = _REPO_ROOT / "data" / "weather"      / "tripsense_weather_2023.csv"
HOTELS_CSV         = _REPO_ROOT / "data" / "hotels"       / "tripsense_hotels.csv"
HOTEL_REVIEWS_CSV  = _REPO_ROOT / "data" / "hotels"       / "tripsense_hotel_reviews_dataset.csv"
REVIEW_INSIGHTS_CSV= _REPO_ROOT / "data" / "hotels"       / "review_insights.csv"
TOPIC_SUMMARY_CSV  = _REPO_ROOT / "data" / "hotels"       / "review_topic_summary.csv"
CALENDAR_CSV       = _REPO_ROOT / "data" / "calendars"    / "calendar_india.csv"
HOLIDAY_CROWD_CSV  = _REPO_ROOT / "data" / "calendars"    / "holiday_crowd.csv"
BEST_MONTHS_CSV    = _REPO_ROOT / "data" / "calendars"    / "best_months_results.csv"
BEST_DATES_CSV     = _REPO_ROOT / "data" / "calendars"    / "best_dates_results.csv"
MEMBER1_FORECAST_CSV = _REPO_ROOT / "member1" / "future_forecast_7d.csv"
MEMBER1_METRICS_CSV  = _REPO_ROOT / "member1" / "model_metrics.csv"
MEMBER1_PRICING_CSV  = _REPO_ROOT / "member1" / "hotel_pricing.csv"

# ---------------------------------------------------------------------------
# Minimum required columns per dataset
# ---------------------------------------------------------------------------
_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "places":          ["place_id", "name", "category", "district", "latitude", "longitude"],
    "interactions":    ["reviewer_name", "place_id", "rating"],
    "weather":         ["place_id", "month", "avg_temp_c", "total_rainfall_mm"],
    "hotels":          ["hotel_id", "hotel_name", "district"],
    "hotel_reviews":   ["hotel_id", "nearby_place_id"],
    "review_insights": ["hotel_id", "mean_star_rating", "mean_vader_compound"],
    "holiday_crowd":   ["date", "crowd_proxy_normalized", "base_visit_score"],
    "best_months":     ["place_id", "month_name", "rank_within_place", "composite_score"],
}

# ---------------------------------------------------------------------------
# Generic safe reader
# ---------------------------------------------------------------------------

def _read_csv(path: Path, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Read a CSV safely; return empty DataFrame on any error."""
    if not path.exists():
        log.warning("File not found: %s", path)
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates, low_memory=False)
    except Exception as exc:
        log.error("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def _validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Warn about missing required columns; never raise."""
    required = _REQUIRED_COLUMNS.get(name, [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning("Dataset '%s' is missing columns: %s", name, missing)
    return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_places(path: Optional[Path] = None) -> pd.DataFrame:
    df = _read_csv(path or PLACES_CSV)
    df = _validate(df, "places")
    if not df.empty:
        # Ensure numeric lat/lon
        for col in ["latitude", "longitude", "visit_duration_hrs",
                    "avg_temp_c", "annual_rain_mm"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Ensure text columns are strings
        for col in ["name", "category", "district", "tags_text",
                    "suitable_for", "description", "best_months"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
    return df


def load_interactions(path: Optional[Path] = None) -> pd.DataFrame:
    df = _read_csv(path or INTERACTIONS_CSV)
    df = _validate(df, "interactions")
    if not df.empty and "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["reviewer_name", "place_id", "rating"])
    return df


def load_weather(path: Optional[Path] = None) -> pd.DataFrame:
    df = _read_csv(path or WEATHER_CSV)
    df = _validate(df, "weather")
    if not df.empty:
        for col in ["avg_temp_c", "total_rainfall_mm"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "month" in df.columns:
            df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    return df


def load_hotels(path: Optional[Path] = None) -> pd.DataFrame:
    df = _read_csv(path or HOTELS_CSV)
    return _validate(df, "hotels")


def load_hotel_reviews(path: Optional[Path] = None) -> pd.DataFrame:
    df = _read_csv(path or HOTEL_REVIEWS_CSV)
    return _validate(df, "hotel_reviews")


def load_review_insights(path: Optional[Path] = None) -> pd.DataFrame:
    df = _read_csv(path or REVIEW_INSIGHTS_CSV)
    return _validate(df, "review_insights")


def load_topic_summary(path: Optional[Path] = None) -> pd.DataFrame:
    return _read_csv(path or TOPIC_SUMMARY_CSV)


def load_holiday_crowd(path: Optional[Path] = None) -> pd.DataFrame:
    df = _read_csv(path or HOLIDAY_CROWD_CSV, parse_dates=["date"])
    return _validate(df, "holiday_crowd")


def load_best_months(path: Optional[Path] = None) -> pd.DataFrame:
    df = _read_csv(path or BEST_MONTHS_CSV)
    return _validate(df, "best_months")


def load_best_dates(path: Optional[Path] = None) -> pd.DataFrame:
    """Large file (~25 MB).  Load lazily; callers should cache the result."""
    return _read_csv(path or BEST_DATES_CSV)


def load_all() -> dict[str, pd.DataFrame]:
    """
    Load every dataset and return as a named dict.
    Always returns a dict even when files are missing (values will be empty DataFrames).
    """
    return {
        "places":          load_places(),
        "interactions":    load_interactions(),
        "weather":         load_weather(),
        "hotels":          load_hotels(),
        "hotel_reviews":   load_hotel_reviews(),
        "review_insights": load_review_insights(),
        "topic_summary":   load_topic_summary(),
        "holiday_crowd":   load_holiday_crowd(),
        "best_months":     load_best_months(),
    }


def load_demand_forecast(path: Optional[Path] = None) -> pd.DataFrame:
    return _read_csv(path or MEMBER1_FORECAST_CSV)


def load_model_metrics(path: Optional[Path] = None) -> pd.DataFrame:
    return _read_csv(path or MEMBER1_METRICS_CSV)


def load_hotel_pricing(path: Optional[Path] = None) -> pd.DataFrame:
    return _read_csv(path or MEMBER1_PRICING_CSV)
