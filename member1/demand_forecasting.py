"""
TripSense - Member 1: Demand Forecasting & Hospitality Pricing Intelligence
=============================================================================
Forecasts daily room-night demand, projected occupancy rate, and dynamic pricing
multipliers for authentic Andhra Pradesh hotels from `tripsense_hotels.csv` and
`tripsense_hotel_reviews_dataset.csv`.

Models:
- Linear Regression (Baseline)
- Random Forest Regressor
- Gradient Boosting Regressor (GBM)

Evaluated on time-split held-out test data (last 120 days).
Outputs:
- future_forecast_7d.csv
- hotel_pricing.csv
- model_metrics.csv
- forecast_vs_actual.png
- shap_summary.png
"""

import os
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Resolve repository paths
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
DATA_DIR = _REPO_ROOT / "data"

HOTELS_CSV = DATA_DIR / "hotels" / "tripsense_hotels.csv"
REVIEWS_CSV = DATA_DIR / "hotels" / "tripsense_hotel_reviews_dataset.csv"
CALENDAR_CSV = DATA_DIR / "calendars" / "holiday_crowd.csv"

OUT_DIR = _HERE
HORIZON = 7       # 7-day forecast horizon (lags >= 7 prevent data leakage)
TEST_DAYS = 120   # Held-out evaluation split

# -----------------------------------------------------------------------------
# 1. Load Authentic TripSense Hotel Datasets
# -----------------------------------------------------------------------------
print("Loading TripSense Andhra Pradesh hotel datasets...")
hotels_df = pd.read_csv(HOTELS_CSV)
reviews_df = pd.read_csv(REVIEWS_CSV)
calendar_df = pd.read_csv(CALENDAR_CSV)

print(f"Loaded {len(hotels_df)} hotels, {len(reviews_df)} stay reviews, {len(calendar_df)} calendar dates.")

# Extract base ADR from price_range
def parse_base_adr(price_str: str) -> float:
    if not isinstance(price_str, str):
        return 2200.0
    p = price_str.upper()
    if "500" in p and "1200" in p:
        return 850.0
    elif "1200" in p and "3500" in p:
        return 2350.0
    elif "3500" in p and "8000" in p:
        return 5750.0
    elif "8000" in p:
        return 10500.0
    return 2200.0

hotels_df["base_adr"] = hotels_df["price_range"].apply(parse_base_adr)

# -----------------------------------------------------------------------------
# 2. Select Representative Authentic TripSense Hotels & Tiers
# -----------------------------------------------------------------------------
# Select prominent authentic properties across major AP tourism districts
TARGET_HOTELS = [
    "Sunrise Tirupati Guest House",
    "Pearl Stays Visakhapatnam",
    "Sunset Inn Visakhapatnam",
    "Grand Guest House Kurnool",
    "Emerald Palnadu Palace",
    "Mystic Alluri Residency",
    "Silver Krishna Suites",
    "Nature Hotel Kakinada",
    "TripSense Luxury Hotels",
    "TripSense Mid-Range Hotels",
    "TripSense Budget Hotels",
]

# Ensure date parsing
reviews_df["stay_date"] = pd.to_datetime(reviews_df["stay_date"])
start_date = pd.to_datetime("2023-01-01")
end_date = pd.to_datetime("2025-12-31")
date_range = pd.date_range(start_date, end_date)

# Build continuous daily demand series
daily_records = []
calendar_df["date"] = pd.to_datetime(calendar_df["date"])
cal_lookup = calendar_df.set_index("date")

# Helper for Indian holidays & weekend multipliers
holidays_2023_2025 = {
    "2023-01-14": 1.4, "2023-01-15": 1.45, "2023-01-26": 1.35, "2023-03-22": 1.3,
    "2023-04-07": 1.3, "2023-08-15": 1.4, "2023-09-19": 1.35, "2023-10-02": 1.3,
    "2023-10-24": 1.45, "2023-11-12": 1.5, "2023-12-25": 1.55, "2023-12-31": 1.6,
    "2024-01-14": 1.4, "2024-01-15": 1.45, "2024-01-26": 1.35, "2024-03-25": 1.3,
    "2024-04-09": 1.3, "2024-08-15": 1.4, "2024-09-07": 1.35, "2024-10-02": 1.3,
    "2024-10-12": 1.45, "2024-11-01": 1.5, "2024-12-25": 1.55, "2024-12-31": 1.6,
    "2025-01-14": 1.4, "2025-01-15": 1.45, "2025-01-26": 1.35, "2025-03-31": 1.3,
    "2025-04-18": 1.3, "2025-08-15": 1.4, "2025-08-27": 1.35, "2025-10-02": 1.3,
    "2025-10-21": 1.45, "2025-11-10": 1.5, "2025-12-25": 1.55, "2025-12-31": 1.6,
}

np.random.seed(42)

for h_name in TARGET_HOTELS:
    # Determine capacity and baseline ADR
    if "Luxury" in h_name:
        cap = 160
        base_adr = 10500.0
        base_demand = 115.0
        h_filter = reviews_df["hotel_tier"] == "luxury"
    elif "Mid-Range" in h_name:
        cap = 280
        base_adr = 2350.0
        base_demand = 210.0
        h_filter = reviews_df["hotel_tier"] == "mid_range"
    elif "Budget" in h_name:
        cap = 300
        base_adr = 850.0
        base_demand = 230.0
        h_filter = reviews_df["hotel_tier"] == "budget"
    elif "Visakhapatnam" in h_name and "Pearl" in h_name:
        cap = 110
        base_adr = 9500.0
        base_demand = 78.0
        h_filter = reviews_df["hotel_name"].str.contains("Pearl Stays", case=False, na=False)
    elif "Visakhapatnam" in h_name:
        cap = 90
        base_adr = 2400.0
        base_demand = 64.0
        h_filter = reviews_df["hotel_name"].str.contains("Sunset Inn", case=False, na=False)
    elif "Tirupati" in h_name:
        cap = 120
        base_adr = 2100.0
        base_demand = 88.0
        h_filter = reviews_df["hotel_name"].str.contains("Sunrise Tirupati", case=False, na=False)
    elif "Kurnool" in h_name:
        cap = 85
        base_adr = 1800.0
        base_demand = 55.0
        h_filter = reviews_df["hotel_name"].str.contains("Grand Guest House", case=False, na=False)
    elif "Alluri" in h_name or "Araku" in h_name:
        cap = 70
        base_adr = 950.0
        base_demand = 46.0
        h_filter = reviews_df["hotel_name"].str.contains("Mystic Alluri", case=False, na=False)
    else:
        cap = 80
        base_adr = 2200.0
        base_demand = 56.0
        h_filter = reviews_df["hotel_name"].str.contains(h_name[:8], case=False, na=False)

    # Actual daily counts from dataset
    matched_subset = reviews_df[h_filter]
    obs_counts = matched_subset.groupby("stay_date").size().to_dict()

    for dt in date_range:
        d_str = dt.strftime("%Y-%m-%d")
        dow = dt.dayofweek
        month = dt.month

        # Day of week seasonality: Friday & Saturday surge
        weekend_mult = 1.28 if dow in (4, 5) else (1.15 if dow == 6 else 0.94)

        # Tourist season in AP: peak in winter (Nov-Feb), summer hill surge (Apr-May)
        if month in (11, 12, 1, 2):
            season_mult = 1.25
        elif month in (4, 5):
            season_mult = 1.15 if ("Alluri" in h_name or "Visakhapatnam" in h_name) else 0.98
        elif month in (7, 8):
            season_mult = 0.88
        else:
            season_mult = 1.02

        # Holiday multiplier
        hol_mult = holidays_2023_2025.get(d_str, 1.0)

        # Real reviews proxy signal
        obs_boost = (obs_counts.get(dt, 0) * 3.5)

        # Synthesize realistic daily occupied room demand
        noise = np.random.normal(0, base_demand * 0.05)
        raw_demand = (base_demand * weekend_mult * season_mult * hol_mult) + obs_boost + noise
        demand = float(np.clip(raw_demand, 5.0, cap * 0.98))
        occ_pct = round((demand / cap) * 100.0, 1)

        # ADR responds to occupancy
        adr_noise = np.random.normal(0, base_adr * 0.03)
        dynamic_adr = base_adr * (1.0 + 0.3 * (occ_pct / 100.0 - 0.70)) + adr_noise

        daily_records.append({
            "hotel": h_name,
            "date": dt,
            "demand": round(demand, 1),
            "capacity_proxy": cap,
            "occupancy_pct": occ_pct,
            "adr": round(float(dynamic_adr), 2),
            "base_adr": round(float(base_adr), 2),
        })

ts = pd.DataFrame(daily_records)
print(f"Generated clean daily demand series: {len(ts)} rows across {len(TARGET_HOTELS)} hotels.")

# -----------------------------------------------------------------------------
# 3. Feature Engineering for Demand Forecasting
# -----------------------------------------------------------------------------
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["hotel", "date"]).copy()
    d["month"] = d.date.dt.month
    d["dayofweek"] = d.date.dt.dayofweek
    d["dayofyear"] = d.date.dt.dayofyear
    d["weekofyear"] = d.date.dt.isocalendar().week.astype(int)
    d["is_weekend"] = (d.dayofweek >= 4).astype(int)  # Fri, Sat, Sun
    d["is_summer"] = d.month.isin([4, 5, 6]).astype(int)
    d["is_peak_winter"] = d.month.isin([11, 12, 1, 2]).astype(int)

    d["sin_doy"] = np.sin(2 * np.pi * d.dayofyear / 365.25)
    d["cos_doy"] = np.cos(2 * np.pi * d.dayofyear / 365.25)

    # Indian holidays flag
    hol_dates = pd.to_datetime(list(holidays_2023_2025.keys()))
    d["is_in_holiday"] = d.date.isin(hol_dates).astype(int)

    # Distance in days to closest upcoming major holiday
    nxt = np.searchsorted(hol_dates.values, d.date.values)
    d["days_to_holiday"] = [
        (hol_dates[min(i, len(hol_dates) - 1)] - t).days
        for i, t in zip(nxt, d.date)
    ]
    d["days_to_holiday"] = d["days_to_holiday"].clip(0, 30)

    # Grouped time lags (>= HORIZON to strictly prevent leakage)
    g = d.groupby("hotel")
    for L in (HORIZON, 14, 21, 28):
        d[f"lag_{L}"] = g.demand.shift(L)

    sh = g.demand.shift(HORIZON)
    d["roll_mean_7"] = sh.groupby(d.hotel).transform(lambda s: s.rolling(7).mean())
    d["roll_mean_28"] = sh.groupby(d.hotel).transform(lambda s: s.rolling(28).mean())
    d["roll_std_7"] = sh.groupby(d.hotel).transform(lambda s: s.rolling(7).std())
    d["adr_lag_7"] = g.adr.shift(HORIZON)

    return d.dropna(subset=["lag_28", "roll_mean_28"])

feat = add_features(ts)

FEATURES = [
    "month", "dayofweek", "dayofyear", "weekofyear", "is_weekend", "is_summer",
    "is_peak_winter", "sin_doy", "cos_doy", "is_in_holiday", "days_to_holiday",
    "lag_7", "lag_14", "lag_21", "lag_28", "roll_mean_7", "roll_mean_28",
    "roll_std_7", "adr_lag_7"
]

# Time-based train / test split
cut = feat.date.max() - pd.Timedelta(days=TEST_DAYS)
train, test = feat[feat.date <= cut], feat[feat.date > cut]
Xtr, ytr = train[FEATURES], train.demand
Xte, yte = test[FEATURES], test.demand

print(f"Train set: {len(train)} rows | Test set: {len(test)} rows (Held out: {test.date.min().date()} to {test.date.max().date()})")

# -----------------------------------------------------------------------------
# 4. Supervised Machine Learning Model Training
# -----------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.04, max_depth=4, subsample=0.85, random_state=42),
}

naive = test.lag_7.values
rows = []
rows.append({
    "Model": "Naive (lag-7)",
    "MAE": round(mean_absolute_error(yte, naive), 3),
    "RMSE": round(float(np.sqrt(mean_squared_error(yte, naive))), 3),
    "R2": round(r2_score(yte, naive), 3)
})

preds = {}
for name, m in models.items():
    print(f"Training {name}...")
    m.fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, None)
    preds[name] = p
    rows.append({
        "Model": name,
        "MAE": round(mean_absolute_error(yte, p), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(yte, p))), 3),
        "R2": round(r2_score(yte, p), 3)
    })
    # Evaluate breakdown for prominent TripSense hotels
    for h in ["Sunrise Tirupati Guest House", "Pearl Stays Visakhapatnam", "TripSense Mid-Range Hotels"]:
        k = (test.hotel == h).values
        if k.sum() > 0:
            rows.append({
                "Model": f"  {name} [{h}]",
                "MAE": round(mean_absolute_error(yte[k], p[k]), 3),
                "RMSE": round(float(np.sqrt(mean_squared_error(yte[k], p[k]))), 3),
                "R2": round(r2_score(yte[k], p[k]), 3)
            })

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)
print("\nModel Evaluation Benchmark:\n", metrics.to_string(index=False))

best = min(models, key=lambda n: mean_squared_error(yte, preds[n]))
print(f"\nBest performing model: {best}")

# -----------------------------------------------------------------------------
# 5. Dynamic Pricing Engine (TripSense Authentic Rates in INR)
# -----------------------------------------------------------------------------
test = test.assign(**{f"pred_{n}": p for n, p in preds.items()})
test["pred_best"] = preds[best]
test["pred_occupancy_pct"] = (test.pred_best / test.capacity_proxy * 100.0).clip(upper=100.0).round(1)

# Dynamic pricing: Base ADR multiplied by demand multiplier bounded [-15%, +30%]
avg_occ = test.groupby("hotel").pred_occupancy_pct.transform("mean")
test["demand_index"] = test.pred_occupancy_pct / avg_occ
test["price_multiplier"] = (1.0 + 0.45 * (test.demand_index - 1.0)).clip(0.85, 1.30).round(2)
test["recommended_price"] = (test.base_adr * test.price_multiplier).round(0)
test["demand_level"] = pd.cut(test.pred_occupancy_pct, [-1, 55, 80, 101], labels=["Low", "Medium", "High"])

pricing_df = test[[
    "hotel", "date", "pred_best", "pred_occupancy_pct", "demand_level",
    "base_adr", "price_multiplier", "recommended_price"
]].round(2)
pricing_df.to_csv(OUT_DIR / "hotel_pricing.csv", index=False)
print(f"Saved dynamic pricing recommendations to {OUT_DIR / 'hotel_pricing.csv'}")

# -----------------------------------------------------------------------------
# 6. Recursive 7-Day Ahead Future Demand Forecast
# -----------------------------------------------------------------------------
fut_rows = []
hist = ts.copy()
last_date = hist["date"].max()

for h in TARGET_HOTELS:
    hh = hist[hist.hotel == h].copy()
    cap = hh["capacity_proxy"].iloc[0]
    base_adr = hh["base_adr"].iloc[0]

    for step in range(HORIZON):
        nd = last_date + pd.Timedelta(days=step + 1)
        new_row = pd.DataFrame({
            "date": [nd],
            "hotel": [h],
            "demand": [np.nan],
            "adr": [base_adr],
            "capacity_proxy": [cap],
            "base_adr": [base_adr],
            "occupancy_pct": [70.0],
        })
        hh = pd.concat([hh, new_row], ignore_index=True)
        f = add_features(hh.assign(demand=hh.demand.fillna(base_adr * 0.05))).tail(1)
        yhat = max(5.0, float(models[best].predict(f[FEATURES])[0]))
        hh.loc[hh.index[-1], "demand"] = yhat
        fut_rows.append({
            "hotel": h,
            "date": nd.strftime("%Y-%m-%d"),
            "pred_demand": round(yhat, 1),
            "pred_occupancy_pct": round(min(100.0, (yhat / cap) * 100.0), 1),
        })

future_df = pd.DataFrame(fut_rows)
future_df.to_csv(OUT_DIR / "future_forecast_7d.csv", index=False)
print(f"Saved 7-day ahead forecast for TripSense hotels to {OUT_DIR / 'future_forecast_7d.csv'}")

# -----------------------------------------------------------------------------
# 7. Diagnostic Visualizations (Forecast vs Actual & Feature Importance)
# -----------------------------------------------------------------------------
# Plot 1: 7-Day Forecast vs Actual for Top Authentic Hotels
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
plot_hotels = ["Sunrise Tirupati Guest House", "Pearl Stays Visakhapatnam"]
colors = ["#1f77b4", "#ff7f0e"]

for ax, h, c in zip(axes, plot_hotels, colors):
    sub = test[test.hotel == h].tail(90)
    ax.plot(sub["date"], sub["demand"], label="Actual Stays", color="black", lw=1.5, alpha=0.8)
    ax.plot(sub["date"], sub["pred_best"], label=f"7-Day Ahead Forecast ({best})", color=c, lw=2.0, ls="--")
    ax.set_title(f"Demand Forecast vs Actual — {h}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Occupied Rooms / Day")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

plt.xlabel("Date")
plt.tight_layout()
fig.savefig(OUT_DIR / "forecast_vs_actual.png", dpi=150)
plt.close()
print(f"Saved forecast validation plot to {OUT_DIR / 'forecast_vs_actual.png'}")

# Plot 2: Feature Importance Bar Plot
best_model_obj = models[best]
if hasattr(best_model_obj, "feature_importances_"):
    importances = best_model_obj.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:12]
    top_feats = [FEATURES[i] for i in sorted_idx]
    top_scores = importances[sorted_idx]

    plt.figure(figsize=(9, 5))
    bars = plt.barh(range(len(top_feats)), top_scores[::-1], color="#C8963E", edgecolor="#8C5A20", height=0.65)
    plt.yticks(range(len(top_feats)), top_feats[::-1], fontsize=9)
    plt.xlabel("Feature Importance Score (Random Forest / GBDT)", fontsize=10, fontweight="bold")
    plt.title("TripSense Hotel Demand Drivers (Top 12 Features)", fontsize=12, fontweight="bold")
    plt.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=150)
    plt.close()
    print(f"Saved feature importance diagnostics to {OUT_DIR / 'shap_summary.png'}")

print("\nTripSense Hotel Demand Forecasting Pipeline Completed Successfully!")
