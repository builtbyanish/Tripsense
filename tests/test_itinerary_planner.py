"""
tests/test_itinerary_planner.py
================================
Integration and unit tests for the TripSense itinerary planner.

Covers:
- Dataset loading and schema validation
- Content-based recommendations for new users
- District and interest filtering
- Missing coordinates and metadata handling
- Itinerary generation for 1-day and 7-day trips
- Duplicate-place prevention
- Geographic grouping
- Schedule duration limits
- Weather unavailable / missing cases
- Hotel distance matching
- Date scoring with fixed and flexible dates
- RAG fallback when API credentials are absent
- Best-months suggestions
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for _p in [
    _REPO_ROOT / "member2" / "src",
    _REPO_ROOT / "member3" / "src",
]:
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from itinerary.services import data_service
from itinerary.services import weather_service
from itinerary.services import hotel_service
from itinerary.services import date_service
from itinerary.services import rag_service

# member3 itinerary engine
from itinerary import build_itinerary, Itinerary, ItineraryDay  # type: ignore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def places_df():
    df = data_service.load_places()
    if df.empty:
        pytest.skip("Places dataset not available")
    return df


@pytest.fixture(scope="session")
def interactions_df():
    return data_service.load_interactions()


@pytest.fixture(scope="session")
def weather_df():
    return data_service.load_weather()


@pytest.fixture(scope="session")
def hotels_df():
    return data_service.load_hotels()


@pytest.fixture(scope="session")
def hotel_reviews_df():
    return data_service.load_hotel_reviews()


@pytest.fixture(scope="session")
def review_insights_df():
    return data_service.load_review_insights()


@pytest.fixture(scope="session")
def best_months_df():
    return data_service.load_best_months()


@pytest.fixture(scope="session")
def holiday_crowd_df():
    return data_service.load_holiday_crowd()


# ---------------------------------------------------------------------------
# 1. Dataset loading and schema validation
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_places_loaded(self, places_df):
        assert not places_df.empty, "Places dataset should not be empty"
        assert "place_id" in places_df.columns
        assert "name" in places_df.columns
        assert "district" in places_df.columns

    def test_places_has_coords(self, places_df):
        assert "latitude" in places_df.columns
        assert "longitude" in places_df.columns
        n_with_coords = places_df[["latitude", "longitude"]].dropna().shape[0]
        assert n_with_coords > 0, "At least some places should have coordinates"

    def test_interactions_schema(self, interactions_df):
        if interactions_df.empty:
            pytest.skip("Interactions dataset not available")
        for col in ["reviewer_name", "place_id", "rating"]:
            assert col in interactions_df.columns

    def test_weather_schema(self, weather_df):
        if weather_df.empty:
            pytest.skip("Weather dataset not available")
        for col in ["place_id", "month", "avg_temp_c", "total_rainfall_mm"]:
            assert col in weather_df.columns

    def test_hotels_schema(self, hotels_df):
        if hotels_df.empty:
            pytest.skip("Hotels dataset not available")
        for col in ["hotel_id", "hotel_name", "district"]:
            assert col in hotels_df.columns

    def test_no_duplicate_place_ids(self, places_df):
        dupes = places_df["place_id"].duplicated().sum()
        assert dupes == 0, f"Found {dupes} duplicate place_ids in places dataset"

    def test_best_months_all_12_per_place(self, best_months_df, places_df):
        if best_months_df.empty:
            pytest.skip("Best months dataset not available")
        sample_pid = places_df["place_id"].iloc[0]
        rows = best_months_df[best_months_df["place_id"] == sample_pid]
        assert len(rows) == 12, f"Expected 12 month rows for {sample_pid}, got {len(rows)}"


# ---------------------------------------------------------------------------
# 2. Content-based recommendations for new users
# ---------------------------------------------------------------------------

class TestContentBasedRecommendations:
    def test_new_user_gets_content_recs(self, places_df, interactions_df):
        from itinerary.services.recommendation_service import get_recommendations
        recs = get_recommendations(
            interests="nature waterfalls",
            places_df=places_df,
            interactions_df=interactions_df,
            user_id=None,   # new user
            top_n=10,
        )
        assert not recs.empty, "Content-based recs should return results"
        assert "place_id" in recs.columns
        assert "name" in recs.columns

    def test_recs_no_duplicates(self, places_df, interactions_df):
        from itinerary.services.recommendation_service import get_recommendations
        recs = get_recommendations(
            interests="temple heritage history",
            places_df=places_df,
            interactions_df=interactions_df,
            top_n=20,
        )
        assert recs["place_id"].duplicated().sum() == 0, "Recommendations must not contain duplicate places"

    def test_district_filter_applied(self, places_df, interactions_df):
        from itinerary.services.recommendation_service import get_recommendations
        recs = get_recommendations(
            interests="beach nature",
            places_df=places_df,
            interactions_df=interactions_df,
            district="Tirupati",
            top_n=20,
        )
        if not recs.empty and "district" in recs.columns:
            all_tirupati = recs["district"].str.lower().str.contains("tirupati", na=False).all()
            assert all_tirupati, "All recommended places should be in Tirupati district"

    def test_interest_filtering_returns_relevant(self, places_df, interactions_df):
        from itinerary.services.recommendation_service import get_recommendations
        recs = get_recommendations(
            interests="waterfall cascade falls",
            places_df=places_df,
            interactions_df=interactions_df,
            top_n=5,
        )
        assert not recs.empty
        # At least some results should have waterfall-related categories/tags
        has_waterfall = False
        for _, row in recs.iterrows():
            text = " ".join([
                str(row.get("category", "")),
                str(row.get("tags_text", "")),
                str(row.get("name", "")),
            ]).lower()
            if "waterfall" in text or "falls" in text:
                has_waterfall = True
                break
        assert has_waterfall, "Waterfall interest should produce at least one waterfall result"

    def test_empty_places_returns_empty(self, interactions_df):
        from itinerary.services.recommendation_service import get_recommendations
        recs = get_recommendations(
            interests="nature",
            places_df=pd.DataFrame(),
            interactions_df=interactions_df,
        )
        assert recs.empty


# ---------------------------------------------------------------------------
# 3. Missing coordinates and metadata
# ---------------------------------------------------------------------------

class TestMissingData:
    def test_places_without_coords_handled(self, places_df, interactions_df):
        from itinerary.services.recommendation_service import get_recommendations
        # Create a places df with no coordinates
        no_coords = places_df.copy()
        no_coords["latitude"]  = None
        no_coords["longitude"] = None
        recs = get_recommendations(
            interests="nature",
            places_df=no_coords,
            interactions_df=interactions_df,
            top_n=5,
        )
        # Should still return results (no crash)
        assert isinstance(recs, pd.DataFrame)

    def test_itinerary_without_coords(self, places_df, interactions_df):
        from itinerary.services.recommendation_service import get_recommendations
        no_coords = places_df.copy()
        no_coords["latitude"]  = None
        no_coords["longitude"] = None
        recs = get_recommendations("nature", no_coords, interactions_df, top_n=10)
        if recs.empty:
            pytest.skip("No recs to test with")
        itin = build_itinerary(recs, n_days=2)
        assert len(itin.days) == 2
        # Should still schedule stops even without coordinates
        total_stops = sum(len(d.stops) for d in itin.days)
        assert total_stops > 0

    def test_weather_missing_place(self, weather_df):
        result = weather_service.get_weather_for_place("NONEXISTENT_ID", 1, weather_df)
        assert result is None

    def test_weather_empty_df(self):
        result = weather_service.get_weather_for_place("AP0001", 1, pd.DataFrame())
        assert result is None


# ---------------------------------------------------------------------------
# 4. Itinerary generation — 1-day and 7-day
# ---------------------------------------------------------------------------

class TestItineraryGeneration:
    def _get_recs(self, places_df, interactions_df, top_n=30):
        from itinerary.services.recommendation_service import get_recommendations
        return get_recommendations(
            interests="nature temple beach",
            places_df=places_df,
            interactions_df=interactions_df,
            top_n=top_n,
        )

    def test_1_day_itinerary(self, places_df, interactions_df):
        recs = self._get_recs(places_df, interactions_df)
        itin = build_itinerary(recs, n_days=1)
        assert len(itin.days) == 1
        assert itin.n_days == 1

    def test_7_day_itinerary(self, places_df, interactions_df):
        recs = self._get_recs(places_df, interactions_df, top_n=50)
        itin = build_itinerary(recs, n_days=7)
        assert len(itin.days) == 7

    def test_no_duplicate_places_across_days(self, places_df, interactions_df):
        recs = self._get_recs(places_df, interactions_df, top_n=40)
        itin = build_itinerary(recs, n_days=3)
        scheduled_ids = [stop.place_id for day in itin.days for stop in day.stops]
        assert len(scheduled_ids) == len(set(scheduled_ids)), "No place should appear twice"

    def test_pace_respected_balanced(self, places_df, interactions_df):
        recs = self._get_recs(places_df, interactions_df, top_n=40)
        itin = build_itinerary(recs, n_days=1, pace="balanced")
        day = itin.days[0]
        assert len(day.stops) <= 5, f"Balanced pace should have max 5 stops/day, got {len(day.stops)}"

    def test_pace_respected_relaxed(self, places_df, interactions_df):
        recs = self._get_recs(places_df, interactions_df, top_n=40)
        itin = build_itinerary(recs, n_days=1, pace="relaxed")
        day = itin.days[0]
        assert len(day.stops) <= 3, f"Relaxed pace should have max 3 stops/day, got {len(day.stops)}"

    def test_pace_respected_packed(self, places_df, interactions_df):
        recs = self._get_recs(places_df, interactions_df, top_n=40)
        itin = build_itinerary(recs, n_days=1, pace="packed")
        day = itin.days[0]
        assert 1 <= len(day.stops) <= 4, f"Packed pace should have 1–4 stops, got {len(day.stops)}"

    def test_rag_intelligence_and_daily_flow(self, places_df, interactions_df):
        recs = self._get_recs(places_df, interactions_df, top_n=40)
        itin = build_itinerary(recs, n_days=2, pace="balanced")
        assert itin.rag_insight, "Itinerary should have an AI RAG strategic assessment"
        for day in itin.days:
            if day.stops:
                assert day.theme, f"Day {day.day_number} should have a theme"
                assert day.rag_tip, f"Day {day.day_number} should have a circadian flow tip"

    def test_senior_and_kids_gentle_terrain_filtering(self, places_df, interactions_df):
        recs = self._get_recs(places_df, interactions_df, top_n=40)
        itin = build_itinerary(recs, n_days=2, has_seniors=True, has_kids=True)
        strenuous_cats = {"waterfall", "caves", "fort"}
        for day in itin.days:
            for stop in day.stops:
                cat = getattr(stop, "category", "").lower()
                assert cat not in strenuous_cats, f"High-terrain category {cat} should not be assigned to seniors/kids"

    def test_solo_senior_terrain_handling(self, places_df, interactions_df):
        recs = self._get_recs(places_df, interactions_df, top_n=30)
        itin = build_itinerary(
            recs, n_days=2, traveler_type="solo", n_travelers=1, has_seniors=True
        )
        assert "gentle terrain safeguards" in itin.rag_insight.lower()

    def test_empty_recs_produces_empty_days(self):
        itin = build_itinerary(pd.DataFrame(), n_days=2)
        assert len(itin.days) == 2
        total_stops = sum(len(d.stops) for d in itin.days)
        assert total_stops == 0

    def test_itinerary_start_date_propagated(self, places_df, interactions_df):
        recs = self._get_recs(places_df, interactions_df)
        start = date(2026, 3, 15)
        itin = build_itinerary(recs, n_days=3, start_date=start)
        assert itin.days[0].date == start
        assert itin.days[2].date == date(2026, 3, 17)

    def test_itinerary_result_type(self, places_df, interactions_df):
        recs = self._get_recs(places_df, interactions_df)
        itin = build_itinerary(recs, n_days=2)
        assert isinstance(itin, Itinerary)
        for day in itin.days:
            assert isinstance(day, ItineraryDay)


# ---------------------------------------------------------------------------
# 5. Weather service
# ---------------------------------------------------------------------------

class TestWeatherService:
    def test_weather_for_known_place(self, weather_df, places_df):
        if weather_df.empty:
            pytest.skip("Weather data not available")
        pid = places_df["place_id"].iloc[0]
        result = weather_service.get_weather_for_place(pid, 1, weather_df)
        if result is not None:
            assert "avg_temp_c" in result
            assert "rain_label" in result
            assert "data_source_note" in result
            assert "forecast" not in result["data_source_note"].lower() or "historical" in result["data_source_note"].lower()

    def test_weather_for_multiple_places(self, weather_df, places_df):
        if weather_df.empty:
            pytest.skip("Weather data not available")
        pids = places_df["place_id"].head(5).tolist()
        results = weather_service.get_weather_for_places(pids, 6, weather_df)
        assert isinstance(results, dict)

    def test_weather_district_summary(self, weather_df):
        if weather_df.empty:
            pytest.skip("Weather data not available")
        summary = weather_service.get_district_weather_summary("Tirupati", 1, weather_df)
        if summary is not None:
            assert "avg_temp_c" in summary
            assert "district" in summary

    def test_live_cyclone_prediction_and_severe_alert(self):
        alert = weather_service.check_trip_severe_weather("Visakhapatnam", date(2026, 10, 27), 2)
        assert isinstance(alert, dict)
        if alert.get("has_severe_weather"):
            assert alert.get("is_cyclone") is True
            assert "cyclone" in alert.get("banner_title", "").lower()
            assert alert.get("max_rain_mm", 0) > 30.0


# ---------------------------------------------------------------------------
# 6. Hotel service
# ---------------------------------------------------------------------------

class TestHotelService:
    def test_hotels_for_place_returns_df(self, hotels_df, hotel_reviews_df, review_insights_df, places_df):
        if hotels_df.empty:
            pytest.skip("Hotels data not available")
        pid = places_df["place_id"].iloc[0]
        result = hotel_service.hotels_for_place(
            pid, hotels_df, hotel_reviews_df, review_insights_df
        )
        assert isinstance(result, pd.DataFrame)

    def test_hotels_for_district(self, hotels_df, review_insights_df):
        if hotels_df.empty:
            pytest.skip("Hotels data not available")
        result = hotel_service.hotels_for_district(
            "Tirupati", hotels_df, review_insights_df, max_hotels=3
        )
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "hotel_name" in result.columns

    def test_hotel_card_no_invented_data(self, hotels_df, hotel_reviews_df, review_insights_df, places_df):
        if hotels_df.empty:
            pytest.skip("Hotels data not available")
        pid = places_df["place_id"].iloc[0]
        result = hotel_service.hotels_for_place(pid, hotels_df, hotel_reviews_df, review_insights_df)
        if result.empty:
            return
        card = hotel_service.format_hotel_card(result.iloc[0])
        # Card must not contain invented keys like 'distance_km' (we can't compute that)
        assert "distance_km" not in card
        # Must have a data source note
        assert "data_source_note" in card


# ---------------------------------------------------------------------------
# 7. Date service
# ---------------------------------------------------------------------------

class TestDateService:
    def test_get_month_suitability(self, best_months_df, places_df):
        if best_months_df.empty:
            pytest.skip("Best months data not available")
        pids = places_df["place_id"].head(3).tolist()
        result = date_service.get_month_suitability(pids, 1, best_months_df)
        assert isinstance(result, dict)

    def test_suggest_best_months(self, best_months_df, places_df):
        if best_months_df.empty:
            pytest.skip("Best months data not available")
        pids = places_df["place_id"].head(5).tolist()
        months = date_service.suggest_best_months_for_places(pids, best_months_df)
        assert isinstance(months, list)
        assert len(months) <= 3

    def test_score_trip_dates_in_range(self, places_df):
        """Test scoring with dates that are in the 2026 dataset range."""
        best_dates_df = data_service.load_best_dates()
        if best_dates_df.empty:
            pytest.skip("Best dates data not available")
        pids = places_df["place_id"].head(3).tolist()
        result = date_service.score_trip_dates(
            start_date=date(2026, 2, 15),
            n_days=3,
            place_ids=pids,
            best_dates_df=best_dates_df,
        )
        assert isinstance(result, dict)
        assert "dates_in_dataset" in result

    def test_score_trip_dates_out_of_range(self, places_df):
        """Dates outside 2026 should return gracefully."""
        best_dates_df = data_service.load_best_dates()
        pids = places_df["place_id"].head(3).tolist()
        result = date_service.score_trip_dates(
            start_date=date(2030, 6, 1),
            n_days=3,
            place_ids=pids,
            best_dates_df=best_dates_df,
        )
        assert result["dates_in_dataset"] is False
        assert "crowd_note" in result

    def test_score_trip_dates_empty_df(self, places_df):
        pids = places_df["place_id"].head(3).tolist()
        result = date_service.score_trip_dates(
            start_date=date(2026, 3, 1),
            n_days=2,
            place_ids=pids,
            best_dates_df=pd.DataFrame(),
        )
        assert result["dates_in_dataset"] is False


# ---------------------------------------------------------------------------
# 8. RAG service
# ---------------------------------------------------------------------------

class TestRAGService:
    def test_rag_fallback_without_api_key(self, places_df, hotels_df, review_insights_df, best_months_df):
        """RAG should work with local fallback even without any LLM API key."""
        import os
        os.environ.pop("TRIPSENSE_LLM_PROVIDER", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)

        data = {
            "places": places_df,
            "hotels": hotels_df,
            "review_insights": review_insights_df,
            "best_months": best_months_df,
        }
        result = rag_service.answer_question(
            query="What are the best beaches in Andhra Pradesh?",
            data=data,
        )
        assert isinstance(result, dict)
        assert "answer" in result
        assert "source" in result
        # Source must be local_fallback or a real LLM (whichever is configured)
        assert result["source"] in ("local_fallback", "openai", "google", "none")

    def test_rag_empty_query(self, places_df):
        data = {"places": places_df, "hotels": pd.DataFrame(),
                "review_insights": pd.DataFrame(), "best_months": pd.DataFrame()}
        result = rag_service.answer_question(query="", data=data)
        assert result["can_answer"] is False

    def test_rag_unknown_query_still_returns(self, places_df):
        data = {"places": places_df, "hotels": pd.DataFrame(),
                "review_insights": pd.DataFrame(), "best_months": pd.DataFrame()}
        result = rag_service.answer_question(
            query="XYZXYZXYZ completely made up query 12345",
            data=data,
        )
        assert "answer" in result  # Should not raise


# ---------------------------------------------------------------------------
# 9. Geographic grouping
# ---------------------------------------------------------------------------

class TestGeographicGrouping:
    def test_sort_by_proximity_no_crash(self, places_df):
        from itinerary import _sort_by_proximity  # type: ignore
        sample = places_df.head(10).to_dict("records")
        result = _sort_by_proximity(sample)
        assert len(result) == len(sample)

    def test_sort_by_proximity_single_place(self, places_df):
        from itinerary import _sort_by_proximity  # type: ignore
        sample = [places_df.iloc[0].to_dict()]
        result = _sort_by_proximity(sample)
        assert len(result) == 1

    def test_haversine_known_distance(self):
        from itinerary import _haversine_km  # type: ignore
        # Tirupati to Vijayawada — roughly 330 km by straight line
        dist = _haversine_km(13.6288, 79.4192, 16.5062, 80.6480)
        assert 280 < dist < 380, f"Expected ~330 km, got {dist:.1f} km"


# ---------------------------------------------------------------------------
# Run marker
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
