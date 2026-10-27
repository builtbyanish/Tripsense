"""
itinerary/services/recommendation_service.py
=============================================
Adapter between the TripSense itinerary planner and Member 2's recommendation
pipeline.  Adds the member2/src directory to sys.path so imports work regardless
of the working directory, then delegates to the real recommendation logic.

Design decisions
----------------
- Member 2's modules use bare imports (e.g. `from content_recommender import ...`).
  We insert member2/src at the front of sys.path before importing, then restore it.
- We expose a single public function `get_recommendations(...)` that returns a
  normalised DataFrame with consistent column names.
- We never re-implement TF-IDF or collaborative filtering here.
- Column mismatches between the places dataset and the pipeline output are bridged
  through a merge against the full places DataFrame.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup for Member 2 imports
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_M2_SRC = _REPO_ROOT / "member2" / "src"


def _add_m2_to_path() -> None:
    p = str(_M2_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


_add_m2_to_path()

try:
    from recommendation_pipeline import RecommendationPipeline  # type: ignore
    _PIPELINE_AVAILABLE = True
except ImportError as _e:
    log.warning("Member2 RecommendationPipeline unavailable: %s", _e)
    _PIPELINE_AVAILABLE = False

try:
    from content_recommender import ContentBasedRecommender  # type: ignore
    _CONTENT_AVAILABLE = True
except ImportError:
    _CONTENT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Full columns we want to carry through from the places dataset
# ---------------------------------------------------------------------------
_PLACE_COLS = [
    "place_id", "name", "category", "district",
    "latitude", "longitude", "description", "tags_text",
    "suitable_for", "visit_duration_hrs", "best_months",
    "avg_temp_c", "annual_rain_mm",
]


def _enrich_with_places(recs: pd.DataFrame, places_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join recommendation results with full place metadata."""
    if places_df.empty or recs.empty:
        return recs
    extra_cols = [c for c in _PLACE_COLS if c not in recs.columns and c in places_df.columns]
    if not extra_cols:
        return recs
    merge_cols = ["place_id"] + extra_cols
    return recs.merge(places_df[merge_cols], on="place_id", how="left")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _filter_by_district(
    df: pd.DataFrame,
    district: Optional[str],
    min_required: int = 15,
) -> pd.DataFrame:
    """
    Filter places by the chosen district.
    If the chosen district has fewer than `min_required` matching places (e.g. smaller
    districts like Chittoor, Konaseema, Alluri Sitharama Raju) for a multi-day trip,
    gracefully augment with the closest attractions from adjacent neighboring districts
    so that multi-day itineraries (e.g. 4-7 days) have realistic and enjoyable stops.
    """
    if not district or district.lower() in ("all", "all andhra pradesh", ""):
        return df
    if "district" not in df.columns or df.empty:
        return df

    dist_lower = district.lower().strip()
    primary = df[df["district"].str.lower().str.contains(dist_lower, na=False)].copy()

    # If we already have enough places for the requested trip duration, return them
    if len(primary) >= min_required:
        return primary

    # Find the geographic center of the selected district
    valid_primary = primary.dropna(subset=["latitude", "longitude"])
    if valid_primary.empty:
        valid_all_dist = df[df["district"].str.lower().str.contains(dist_lower, na=False)].dropna(subset=["latitude", "longitude"])
        if valid_all_dist.empty:
            return primary
        c_lat = float(valid_all_dist["latitude"].mean())
        c_lon = float(valid_all_dist["longitude"].mean())
    else:
        c_lat = float(valid_primary["latitude"].mean())
        c_lon = float(valid_primary["longitude"].mean())

    # Get candidate places from outside the district
    others = df[~df["district"].str.lower().str.contains(dist_lower, na=False)].copy()
    others_valid = others.dropna(subset=["latitude", "longitude"]).copy()

    if others_valid.empty:
        return primary

    # Calculate distance from primary district center
    others_valid["dist_to_target_km"] = others_valid.apply(
        lambda r: _haversine_km(c_lat, c_lon, float(r["latitude"]), float(r["longitude"])),
        axis=1,
    )

    # Consider accessible nearby places (within ~110 km driving radius)
    nearby = others_valid[others_valid["dist_to_target_km"] <= 110.0].copy()
    if nearby.empty:
        # Take the closest 15 places regardless of radius
        nearby = others_valid.sort_values("dist_to_target_km").head(min_required)

    # Sort nearby places by recommendation content/hybrid score first, then proximity
    sort_cols = []
    asc_flags = []
    for score_col in ["hybrid_score", "content_score"]:
        if score_col in nearby.columns:
            sort_cols.append(score_col)
            asc_flags.append(False)
            break
    sort_cols.append("dist_to_target_km")
    asc_flags.append(True)

    nearby = nearby.sort_values(sort_cols, ascending=asc_flags)

    needed = max(0, min_required - len(primary))
    augmented = pd.concat([primary, nearby.head(needed)], ignore_index=True)
    return augmented.drop_duplicates("place_id").reset_index(drop=True)


def _filter_by_suitable(
    df: pd.DataFrame,
    traveler_type: Optional[str] = None,
    has_seniors: bool = False,
    has_kids: bool = False,
) -> pd.DataFrame:
    """
    Filter places based on traveler group and age composition.
    - If seniors/elderly are in the group: avoid strenuous climbing/terrain
      (e.g., steep waterfalls, caves, trekking/adventure, forts with extensive climbing)
      and prioritize places suitable for seniors, temples, gardens, museums, and relaxed visits.
    - If young kids are in the group: prioritize places suitable for kids and families,
      avoiding high-risk waterfalls or extreme adventure terrain.
    """
    if df.empty:
        return df

    filtered = df.copy()

    # 1. Traveler type soft match
    if traveler_type and "suitable_for" in filtered.columns:
        t = traveler_type.lower().strip()
        mask = (
            filtered["suitable_for"].str.lower().str.contains(t, na=False)
            | (filtered["suitable_for"].str.strip() == "")
        )
        sub = filtered[mask]
        if not sub.empty:
            filtered = sub.copy()

    # 2. Senior (60+) & Children (<10) High-Terrain / Climbing Safeguards
    if has_seniors or has_kids:
        # Exclude high-terrain, steep climbing, and strenuous physical categories
        # (rocky gorge waterfalls, subterranean steep caves, hilltop forts with long ascents)
        strenuous_categories = {"waterfall", "caves", "fort"}
        if "category" in filtered.columns:
            non_strenuous = filtered[~filtered["category"].str.lower().isin(strenuous_categories)]
            if not non_strenuous.empty:
                filtered = non_strenuous.copy()

        # Filter out places explicitly marked for adventure or trekking
        if "suitable_for" in filtered.columns:
            non_adv = filtered[~filtered["suitable_for"].str.lower().str.contains("adventure|trekking|extreme", na=False)]
            if not non_adv.empty:
                filtered = non_adv.copy()

        # Exclude steep climbing, high terrain, or trekking in name or description
        steep_regex = r"(?i)(trek|trekking|climb|climbing|cliff|gorge|rock climbing|steep steps)"
        name_mask = filtered["name"].str.contains(steep_regex, regex=True, na=False)
        desc_mask = (
            filtered["description"].str.contains(steep_regex, regex=True, na=False)
            if "description" in filtered.columns
            else pd.Series(False, index=filtered.index)
        )
        gentle_mask = ~(name_mask | desc_mask)
        if gentle_mask.any():
            filtered = filtered[gentle_mask].copy()

    return filtered if not filtered.empty else df


# ---------------------------------------------------------------------------
def _filter_non_requested_temples(recs: pd.DataFrame, interests: str) -> pd.DataFrame:
    """
    If the traveler did not specifically request spiritual/temple themes,
    deprioritize/remove pure temples so they do not crowd out hill stations,
    beaches, waterfalls, forts, or nature reserves.
    """
    if recs.empty:
        return recs

    wants_temple = any(
        w in str(interests).lower()
        for w in ["temple", "spiritual", "pilgrimage", "religious", "darshan", "sacred", "mandir", "shrine", "devasthanam"]
    )
    if wants_temple:
        return recs

    cat_col = recs["category"].astype(str).str.lower() if "category" in recs.columns else pd.Series("", index=recs.index)
    name_col = recs["name"].astype(str).str.lower() if "name" in recs.columns else pd.Series("", index=recs.index)
    is_temple = (cat_col == "temple") | name_col.str.contains(r"\btemple\b|\bmandir\b|\bshrine\b|darshan|swamy vaari", regex=True)

    non_temples = recs[~is_temple]
    if not non_temples.empty:
        return non_temples
    return recs


def _filter_positive_scores(recs: pd.DataFrame) -> pd.DataFrame:
    if recs.empty:
        return recs
    for col in ["final_score", "content_score"]:
        if col in recs.columns:
            pos = recs[recs[col] > 0.0001]
            if not pos.empty:
                return pos
    return recs

# Public API
# ---------------------------------------------------------------------------

def get_recommendations(
    interests: str,
    places_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    district: Optional[str] = None,
    traveler_type: Optional[str] = None,
    user_id: Optional[str] = None,
    top_n: int = 30,
    content_weight: float = 0.7,
    collaborative_weight: float = 0.3,
    has_seniors: bool = False,
    has_kids: bool = False,
) -> pd.DataFrame:
    """
    Return recommended places using Member 2's pipeline.

    Parameters
    ----------
    interests        : Space-separated interest keywords (e.g. "temples nature waterfalls")
    places_df        : Full places DataFrame from data_service.load_places()
    interactions_df  : Interactions DataFrame from data_service.load_interactions()
    district         : Optional district filter string (empty = all AP)
    traveler_type    : e.g. "family", "solo", "couple"
    user_id          : Existing reviewer_name for hybrid recommendations; None for new users
    top_n            : Max results to return
    content_weight   : Weight for content-based component (hybrid only)
    collaborative_weight : Weight for collaborative component (hybrid only)

    Returns
    -------
    DataFrame with at minimum: place_id, name, category, district,
    recommendation_method, recommendation_reason, content_score
    """
    if places_df.empty:
        log.warning("Places dataframe is empty; no recommendations possible.")
        return pd.DataFrame()

    if not interests or not interests.strip():
        interests = "travel nature culture"

    # Determine how many places to retrieve initially from recommendation pipeline
    # If a district filter is applied, request all places so district-specific attractions
    # are never starved by global top_n, then filter and take top_n
    pipeline_top_n = len(places_df)

    # ---- Use full pipeline if available ----
    if _PIPELINE_AVAILABLE:
        try:
            pipeline = RecommendationPipeline(places_df, interactions_df)
            recs = pipeline.recommend_places(
                interests=interests,
                user_id=user_id,
                top_n=pipeline_top_n,
                content_weight=content_weight,
                collaborative_weight=collaborative_weight,
            )
            recs = _enrich_with_places(recs, places_df)
            recs = _filter_by_district(recs, district, min_required=min(top_n, 20))
            recs = _filter_by_suitable(
                recs,
                traveler_type=traveler_type,
                has_seniors=has_seniors,
                has_kids=has_kids,
            )
            recs = _filter_non_requested_temples(recs, interests)
            recs = _filter_positive_scores(recs)
            if recs.empty:
                log.info("Pipeline returned no results after filtering; falling back to content-based.")
            else:
                return recs.drop_duplicates("place_id").head(top_n).reset_index(drop=True)
        except Exception as exc:
            log.warning("RecommendationPipeline failed (%s); falling back to ContentBasedRecommender.", exc)

    # ---- Fallback: content-based only ----
    if _CONTENT_AVAILABLE:
        try:
            cbr = ContentBasedRecommender(places_df)
            recs = cbr.recommend(
                interests=interests,
                district=district if district and district.lower() not in ("all", "") else None,
                suitable_for=traveler_type,
                top_n=top_n,
            )
            recs = _enrich_with_places(recs, places_df)
            recs = _filter_by_suitable(
                recs,
                traveler_type=traveler_type,
                has_seniors=has_seniors,
                has_kids=has_kids,
            )
            recs = _filter_non_requested_temples(recs, interests)
            recs = _filter_positive_scores(recs)
            if "recommendation_method" not in recs.columns:
                recs["recommendation_method"] = "content-based"
            if "recommendation_reason" not in recs.columns:
                recs["recommendation_reason"] = (
                    recs["name"].fillna("This place")
                    + " matches your interest in "
                    + interests[:40]
                    + "."
                )
            return recs.drop_duplicates("place_id").reset_index(drop=True)
        except Exception as exc:
            log.error("ContentBasedRecommender also failed: %s", exc)

    # ---- Last resort: return all places in district ----
    log.warning("All recommendation engines failed; returning unranked places.")
    candidates = places_df.copy()
    candidates = _filter_by_district(candidates, district)
    candidates = _filter_by_suitable(
        candidates,
        traveler_type=traveler_type,
        has_seniors=has_seniors,
        has_kids=has_kids,
    )
    candidates["recommendation_method"] = "unranked"
    candidates["recommendation_reason"] = "Included based on location match."
    candidates["content_score"] = 0.0
    return candidates.drop_duplicates("place_id").head(top_n).reset_index(drop=True)
