# ══════════════════════════════════════════════════════════════════════════════
# Vegetable_Final_v3.py
# Changes over v2 — synced with Cereal_Final_v3 for feature parity:
#   1. Arrivals    : added arrivals_lag_7, arrivals_pct_change_7,
#                    supply_stress_index
#   2. arrival_shock redefined: now uses rolling_7 base (was lag_3 base)
#   3. supply_tightness redefined: now price_pct_change_7 / supply ratio
#                    (was simple arrival deficit)
#   4. Weather     : added rainfall_30d, rainfall_60d, rainfall_shock_30d
#                    (union with cereal pipeline)
#   5. Volatility  : price_volatility_14 already present — kept
#   6. price_vs_30d_mean, supply_shock_7v30, arrival_rolling_30 — kept
#   7. No MSP features — government does not provide MSP for vegetables
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("  VEGETABLE FEATURE ENGINEERING PIPELINE v3")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1 — LOAD PRICE & ARRIVAL DATA
# ─────────────────────────────────────────────────────────────
input_file = "Agri-Hotri_Price_Prediction/Data/Vegetable_Price.xlsx"
sheets = pd.read_excel(input_file, sheet_name=None)
print(" Excel Loaded")

product_sheets = {}
for sheet_name, df in sheets.items():
    product, sheet_type = sheet_name.rsplit(" ", 1)
    product_sheets.setdefault(product, {})[sheet_type] = df

final_data = []

for product, data in product_sheets.items():
    if "Price" not in data or "Arrival" not in data:
        continue

    p_df       = data["Price"].copy()
    arrival_df = data["Arrival"].copy()

    p_df["Date"]       = pd.to_datetime(p_df["Date"])
    arrival_df["Date"] = pd.to_datetime(arrival_df["Date"])

    price_col   = [c for c in p_df.columns       if c != "Date"][0]
    arrival_col = [c for c in arrival_df.columns if c != "Date"][0]

    p_df.rename(columns={price_col: "modal_price"},   inplace=True)
    arrival_df.rename(columns={arrival_col: "arrivals"}, inplace=True)

    min_date   = min(p_df["Date"].min(), arrival_df["Date"].min())
    max_date   = max(p_df["Date"].max(), arrival_df["Date"].max())
    full_dates = pd.DataFrame({"Date": pd.date_range(min_date, max_date, freq="D")})

    merged = (
        full_dates
        .merge(p_df,       on="Date", how="left")
        .merge(arrival_df, on="Date", how="left")
    )

    merged["commodity"] = product
    final_data.append(merged)

veg_df = pd.concat(final_data, ignore_index=True)
veg_df["date"]      = pd.to_datetime(veg_df["Date"])
veg_df["commodity"] = veg_df["commodity"].str.strip().str.lower()
print(" Price & Arrival merged")

# ─────────────────────────────────────────────────────────────
# SECTION 2 — WEATHER MERGE
# ─────────────────────────────────────────────────────────────
weather_df = pd.read_csv("Agri-Hotri_Price_Prediction/Data/maharashtra_daily_weather_2015_2025.csv")
weather_df["date"] = pd.to_datetime(weather_df["date"])

weather_daily = weather_df.groupby("date").agg({
    "temperature":     "mean",
    "rainfall":        "sum",
    "solar_radiation": "mean",
    "wind_speed":      "mean"
}).reset_index()

# Short-window rolling (vegetable-relevant — shorter spoilage cycle)
weather_daily["rainfall_7d"]  = weather_daily["rainfall"].rolling(7).sum()
weather_daily["rainfall_15d"] = weather_daily["rainfall"].rolling(15).sum()
weather_daily["temp_7d_avg"]  = weather_daily["temperature"].rolling(7).mean()

# Long-window rolling (aligned with Cereal pipeline)
weather_daily["rainfall_30d"] = weather_daily["rainfall"].rolling(30).sum()
weather_daily["rainfall_60d"] = weather_daily["rainfall"].rolling(60).sum()
weather_daily["temp_14d_avg"] = weather_daily["temperature"].rolling(14).mean()

# Rainfall shock — 7-day (shared) and 30-day (cereal-aligned)
weather_daily["rainfall_shock_7d"] = (
    (weather_daily["rainfall"] - weather_daily["rainfall"].rolling(7).mean()) /
    (weather_daily["rainfall"].rolling(7).mean() + 1e-6)
)
weather_daily["rainfall_shock_30d"] = (
    (weather_daily["rainfall"] - weather_daily["rainfall"].rolling(30).mean()) /
    (weather_daily["rainfall"].rolling(30).mean() + 1e-6)
)

# Temperature deviation from 14-day average
weather_daily["temp_deviation_14d"] = (
    weather_daily["temperature"] - weather_daily["temp_14d_avg"]
)

veg_df = veg_df.merge(weather_daily, on="date", how="left")
print(" Weather merged")

# ─────────────────────────────────────────────────────────────
# SECTION 3 — FUEL MERGE
# ─────────────────────────────────────────────────────────────
fuel_df = pd.read_csv("Agri-Hotri_Price_Prediction/Data/Fuel_prices.csv")

fuel_df["Petrol_price"] = (
    fuel_df["Petrol_price"].astype(str).str.replace("*", "", regex=False).astype(float)
)
fuel_df["Diesel_price"] = (
    fuel_df["Diesel_price"].astype(str).str.replace("*", "", regex=False).astype(float)
)

fuel_df["Month"]     = fuel_df["Month"].astype(str).str.strip()
month_map = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
    "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
}
fuel_df["Month_Num"] = fuel_df["Month"].map(month_map)
fuel_df              = fuel_df[fuel_df["Month_Num"].notna()]
fuel_df["Year"]      = pd.to_numeric(fuel_df["Year"], errors="coerce")
fuel_df              = fuel_df[fuel_df["Year"].notna()]
fuel_df["date"]      = pd.to_datetime(dict(
    year=fuel_df["Year"].astype(int),
    month=fuel_df["Month_Num"].astype(int),
    day=1
))
fuel_df = fuel_df[fuel_df["date"].notna()].sort_values("date")

fuel_daily = pd.DataFrame({
    "date": pd.date_range(fuel_df["date"].min(), fuel_df["date"].max(), freq="D")
})
fuel_daily = pd.merge_asof(
    fuel_daily.sort_values("date"),
    fuel_df.sort_values("date"),
    on="date",
    direction="backward"
)

fuel_daily["diesel_lag_7"]  = fuel_daily["Diesel_price"].shift(7)
fuel_daily["diesel_lag_30"] = fuel_daily["Diesel_price"].shift(30)
fuel_daily["petrol_lag_7"]  = fuel_daily["Petrol_price"].shift(7)
fuel_daily["petrol_lag_30"] = fuel_daily["Petrol_price"].shift(30)

fuel_daily["diesel_pct_change_30"] = (
    (fuel_daily["Diesel_price"] - fuel_daily["diesel_lag_30"]) /
    (fuel_daily["diesel_lag_30"] + 1e-6)
)
fuel_daily["diesel_pct_change_7"] = (
    (fuel_daily["Diesel_price"] - fuel_daily["diesel_lag_7"]) /
    (fuel_daily["diesel_lag_7"] + 1e-6)
)
fuel_daily["petrol_pct_change_7"] = (
    (fuel_daily["Petrol_price"] - fuel_daily["petrol_lag_7"]) /
    (fuel_daily["petrol_lag_7"] + 1e-6)
)

veg_df = veg_df.merge(fuel_daily, on="date", how="left")

fuel_cols = [
    "Petrol_price","Diesel_price",
    "diesel_lag_7","diesel_lag_30",
    "petrol_lag_7","petrol_lag_30",
    "diesel_pct_change_30",
    "diesel_pct_change_7","petrol_pct_change_7"
]
veg_df[fuel_cols] = veg_df[fuel_cols].ffill()
print(" Fuel data merged")

# ─────────────────────────────────────────────────────────────
# SECTION 4 — DATA CLEANING & OUTLIER HANDLING
# ─────────────────────────────────────────────────────────────
veg_df = veg_df[
    (veg_df["modal_price"] > 0) &
    (veg_df["arrivals"] >= 0)
]

veg_df = veg_df.sort_values(["commodity", "date"])

# Outlier smoothing via rolling median (per commodity)
veg_df["rolling_median_5"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.rolling(5, min_periods=2).median())
)

outlier_mask = veg_df["modal_price"] > 3 * veg_df["rolling_median_5"]
veg_df.loc[outlier_mask, "modal_price"] = veg_df["rolling_median_5"]

# Interpolate missing prices when arrivals exist
mask = veg_df["modal_price"].isna() & (veg_df["arrivals"] > 0)
veg_df.loc[mask, "modal_price"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.interpolate())
)

veg_df["market_closed_flag"] = (
    veg_df["modal_price"].isna() &
    (veg_df["arrivals"] == 0)
).astype(int)

veg_df["zero_arrival_flag"] = (veg_df["arrivals"] == 0).astype(int)

# ─────────────────────────────────────────────────────────────
# SECTION 5 — TIME FEATURES
# ─────────────────────────────────────────────────────────────
print(" Engineering time features")
veg_df["Year"]       = veg_df["date"].dt.year
veg_df["Month_Num"]  = veg_df["date"].dt.month
veg_df["DayOfYear"]  = veg_df["date"].dt.dayofyear
veg_df["WeekOfYear"] = veg_df["date"].dt.isocalendar().week.astype(int)

# Season for vegetables (Kharif = Jun–Nov, Rabi = Dec–May)
veg_df["season"] = veg_df["Month_Num"].apply(
    lambda m: "kharif" if 6 <= m <= 11 else "rabi"
)
le = LabelEncoder()
veg_df["season_enc"] = le.fit_transform(veg_df["season"])

# ─────────────────────────────────────────────────────────────
# SECTION 6 — PRICE LAG FEATURES (grouped by commodity)
# ─────────────────────────────────────────────────────────────
print(" Engineering price lag features (per commodity — no contamination)")
veg_df = veg_df.sort_values(["commodity", "date"])

for lag in [1, 3, 7, 14, 30]:
    veg_df[f"price_lag_{lag}"] = (
        veg_df.groupby("commodity")["modal_price"].shift(lag)
    )

# Contamination verification
veg_df["_prev_commodity"] = veg_df["commodity"].shift(1)
boundary_rows = veg_df[veg_df["commodity"] != veg_df["_prev_commodity"]]
contaminated  = boundary_rows["price_lag_1"].notna().sum()
if contaminated == 0:
    print("   Lag verification passed — no cross-commodity contamination")
else:
    print(f"  ⚠️  WARNING: {contaminated} boundary rows have non-NaN lag_1")
veg_df.drop(columns=["_prev_commodity"], inplace=True)

# ─────────────────────────────────────────────────────────────
# SECTION 7 — ROLLING PRICE STATISTICS (per commodity)
# ─────────────────────────────────────────────────────────────
print(" Engineering rolling price statistics")

veg_df["price_rolling_median_7"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.shift(1).rolling(7).median())
)
veg_df["price_rolling_mean_14"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.shift(1).rolling(14).mean())
)
veg_df["price_rolling_mean_30"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.shift(1).rolling(30).mean())
)
veg_df["price_volatility_7"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.shift(1).rolling(7).std())
)
veg_df["price_volatility_14"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.shift(1).rolling(14).std())
)
veg_df["price_volatility_30"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.shift(1).rolling(30).std())
)

# Price regime indicator — how far is today's price from the 30-day average
veg_df["price_vs_30d_mean"] = (
    (veg_df["modal_price"] - veg_df["price_rolling_mean_30"]) /
    (veg_df["price_rolling_mean_30"] + 1e-6)
)

# ─────────────────────────────────────────────────────────────
# SECTION 8 — PRICE MOMENTUM FEATURES
# ─────────────────────────────────────────────────────────────
print(" Engineering price momentum features")

veg_df["price_pct_change_3"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.pct_change(3))
)
veg_df["price_pct_change_7"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.pct_change(7))
)
veg_df["price_pct_change_30"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.pct_change(30))
)

# ─────────────────────────────────────────────────────────────
# SECTION 9 — ARRIVAL FEATURES
# ─────────────────────────────────────────────────────────────
print(" Engineering arrival features")
veg_df = veg_df.sort_values(["commodity", "date"])

veg_df["arrivals_lag_7"] = (
    veg_df.groupby("commodity")["arrivals"].shift(7)
)
veg_df["arrivals_pct_change_7"] = (
    (veg_df["arrivals"] - veg_df["arrivals_lag_7"]) /
    (veg_df["arrivals_lag_7"] + 1e-6)
)

veg_df["arrival_lag_3"] = (
    veg_df.groupby("commodity")["arrivals"].shift(3)
)

veg_df["arrival_rolling_7"] = (
    veg_df.groupby("commodity")["arrivals"]
    .transform(lambda x: x.rolling(7).mean())
)
veg_df["arrival_rolling_14"] = (
    veg_df.groupby("commodity")["arrivals"]
    .transform(lambda x: x.rolling(14).mean())
)
veg_df["arrival_rolling_30"] = (
    veg_df.groupby("commodity")["arrivals"]
    .transform(lambda x: x.rolling(30).mean())
)

# Arrival shock — sudden deviation from 7-day rolling average
veg_df["arrival_shock"] = (
    (veg_df["arrivals"] - veg_df["arrival_rolling_7"]) /
    (veg_df["arrival_rolling_7"] + 1e-6)
)

# Supply stress — how far current arrivals are below the 14-day average
veg_df["supply_stress_index"] = (
    (veg_df["arrival_rolling_14"] - veg_df["arrivals"]) /
    (veg_df["arrival_rolling_14"] + 1e-6)
)

# Supply shock — 7-day vs 30-day arrivals (short vs long baseline)
veg_df["supply_shock_7v30"] = (
    (veg_df["arrival_rolling_7"] - veg_df["arrival_rolling_30"]) /
    (veg_df["arrival_rolling_30"] + 1e-6)
)

# Supply tightness — price momentum relative to supply level
veg_df["supply_tightness"] = (
    veg_df["price_pct_change_7"] /
    (veg_df["arrivals"] / (veg_df["arrival_rolling_7"] + 1e-6) + 1e-6)
)

# ─────────────────────────────────────────────────────────────
# SECTION 10 — NaN FILLING FOR LAG & ROLLING FEATURES
# ─────────────────────────────────────────────────────────────
print(" Filling NaN values in lag & rolling features")

lag_cols = [
    "price_lag_1", "price_lag_3", "price_lag_7", "price_lag_14", "price_lag_30",
    "price_rolling_median_7", "price_rolling_mean_14", "price_rolling_mean_30",
    "price_volatility_7", "price_volatility_14", "price_volatility_30",
    "price_pct_change_3", "price_pct_change_7", "price_pct_change_30",
    "price_vs_30d_mean",
    "arrivals_lag_7", "arrivals_pct_change_7",
    "arrival_lag_3", "arrival_rolling_7", "arrival_rolling_14", "arrival_rolling_30",
    "arrival_shock", "supply_stress_index", "supply_shock_7v30", "supply_tightness",
]

lag_cols = [c for c in lag_cols if c in veg_df.columns]

before_nan = veg_df[lag_cols].isna().sum().sum()
print(f"  Total NaNs before filling : {before_nan}")

veg_df = veg_df.sort_values(["commodity", "date"])

# Step 1: Drop warmup rows (longest lag = price_lag_30)
rows_before = len(veg_df)
veg_df = veg_df.dropna(subset=["price_lag_30"])
rows_dropped = rows_before - len(veg_df)
print(f"  Warmup rows dropped       : {rows_dropped} ({rows_dropped/rows_before*100:.2f}%)")

# Step 2: Backfill remaining NaNs per commodity
veg_df[lag_cols] = (
    veg_df.groupby("commodity")[lag_cols]
    .transform(lambda x: x.bfill())
)

# Step 3: Per-commodity median fill as last resort
veg_df[lag_cols] = (
    veg_df.groupby("commodity")[lag_cols]
    .transform(lambda x: x.fillna(x.median()))
)

after_nan = veg_df[lag_cols].isna().sum().sum()
print(f"  Total NaNs after filling  : {after_nan}")

print("\n  NaN % per feature after filling:")
for c in lag_cols:
    pct = veg_df[c].isna().mean() * 100
    status = "" if pct == 0 else "⚠️ "
    print(f"    {status} {c:<35} {pct:.2f}%")

# ─────────────────────────────────────────────────────────────
# SECTION 11 — FINAL CLEANUP
# ─────────────────────────────────────────────────────────────
print(" Final cleanup")

drop_cols = ["Date", "rolling_median_5"]
veg_df.drop(columns=drop_cols, errors="ignore", inplace=True)

veg_df = veg_df.sort_values(["commodity", "date"]).reset_index(drop=True)

print(f"\n  Final dataset shape : {veg_df.shape}")
print(f"  Total features      : {veg_df.shape[1]}")
print(f"  Date range          : {veg_df['date'].min().date()} -> {veg_df['date'].max().date()}")
print(f"  Commodities         : {veg_df['commodity'].unique().tolist()}")

veg_df.to_csv("Agri-Hotri_Price_Prediction/Data/Vegetable_Final_v3.csv", index=False)
print("\n Saved -> Agri-Hotri_Price_Prediction/Data/Vegetable_Final_v3.csv")