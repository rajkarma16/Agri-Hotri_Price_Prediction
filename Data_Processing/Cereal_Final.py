# ══════════════════════════════════════════════════════════════════════════════
# Cereal_Final_v2.py
# Upgrades over v1:
#   1. Price lag features  : lag_1, 3, 7, 14, 30  (grouped by commodity )
#   2. Rolling price stats : median_7, mean_14, mean_30
#   3. Price momentum      : pct_change_3, 7, 30
#   4. Price volatility    : volatility_7, volatility_30
#   5. MSP distance        : price_above_msp, msp_pct_above_msp, msp_yearly_growth
#   6. Arrival features    : arrival_rolling_7, arrival_shock, supply_tightness
#   7. Fuel momentum       : diesel_pct_change_7, petrol_pct_change_7
#   8. Weather interaction : rainfall_shock_30d
#   9. Time features       : DayOfYear, WeekOfYear  (Year & Month_Num already exist)
#  10. Season encoding     : season_enc (label encoded)
#  11. Bug fixes           : .ffill() replaces deprecated fillna(method='ffill')
#                            rolling_median_7 no longer dropped
#                            boundary verification after every groupby lag
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────
# SECTION 1 — LOAD PRICE & ARRIVAL DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  CEREAL FEATURE ENGINEERING PIPELINE v2")
print("=" * 60)

input_file = "Agri-Hotri_Price_Prediction/Data/Cereal_Price.xlsx"
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

    p_df.rename(columns={price_col: "Price"},     inplace=True)
    arrival_df.rename(columns={arrival_col: "Arrival"}, inplace=True)

    min_date   = min(p_df["Date"].min(), arrival_df["Date"].min())
    max_date   = max(p_df["Date"].max(), arrival_df["Date"].max())
    full_dates = pd.DataFrame({"Date": pd.date_range(min_date, max_date, freq="D")})

    merged = (
        full_dates
        .merge(p_df,       on="Date", how="left")
        .merge(arrival_df, on="Date", how="left")
    )

    merged["commodity"] = product
    merged["Date"]      = merged["Date"].dt.strftime("%d-%m-%Y")
    final_data.append(merged)

price_df = pd.concat(final_data, ignore_index=True)
print(" Price & Arrival merged")

# ─────────────────────────────────────────────────────────────
# SECTION 2 — MSP MERGE
# ─────────────────────────────────────────────────────────────
msp_df     = pd.read_excel("Agri-Hotri_Price_Prediction/Data/MSP.xlsx")
weather_df = pd.read_csv("Agri-Hotri_Price_Prediction/Data/maharashtra_daily_weather_2015_2025.csv")
print(" MSP & Weather loaded")

price_df["date"]      = pd.to_datetime(price_df["Date"])
weather_df["date"]    = pd.to_datetime(weather_df["date"])

price_df["commodity"] = price_df["commodity"].str.strip().str.lower()
msp_df["Commodity"]   = msp_df["Commodity"].str.strip().str.lower()

column_map = {
    "Price":         "modal_price",
    "Min Price (₹)": "min_price",
    "Arrival":       "arrivals",
    "Max Price (₹)": "max_price"
}
price_df = price_df.rename(columns=column_map)

# Basic validity filter
price_df = price_df[
    (price_df["modal_price"] > 0) &
    (price_df["min_price"] <= price_df["modal_price"]) &
    (price_df["modal_price"] <= price_df["max_price"]) &
    (price_df["arrivals"] >= 0)
]

price_df = price_df.sort_values(["commodity", "date"])

# ── Rolling median outlier smoothing (computed per commodity)
price_df["rolling_median_7"] = (
    price_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.rolling(7, min_periods=3).median())
)

outlier_mask = price_df["modal_price"] > 3 * price_df["rolling_median_7"]
price_df.loc[
    outlier_mask & (price_df["arrivals"] > 0), "modal_price"
] = price_df.loc[
    outlier_mask & (price_df["arrivals"] > 0), "rolling_median_7"
]

# Zero Arrival Flag
price_df["zero_arrival_flag"] = (price_df["arrivals"] == 0).astype(int)

# Season mapping
season_mapping = {
    "rice":           "kharif",
    "arhar (tur dal)": "kharif",
    "wheat":          "rabi"
}
price_df["season"]        = price_df["commodity"].map(season_mapping)
price_df["msp_commodity"] = price_df["commodity"]

# MSP merge
msp_df.rename(columns={"Commodity": "msp_commodity"}, inplace=True)

msp_long = msp_df.melt(
    id_vars=["msp_commodity"],
    var_name="year",
    value_name="msp"
)

def get_msp_start_date(row):
    start_year = int(str(row["year"]).split("-")[0])
    if row["msp_commodity"] in ["paddy (common)", "tur (arhar)"]:
        return pd.Timestamp(f"{start_year}-10-01")
    elif row["msp_commodity"] == "wheat":
        return pd.Timestamp(f"{start_year + 1}-04-01")
    return pd.Timestamp(f"{start_year}-04-01")

msp_long["start_date"] = msp_long.apply(get_msp_start_date, axis=1)
msp_long = msp_long.sort_values(["msp_commodity", "start_date"])

full_dates = pd.date_range(
    start=price_df["date"].min(),
    end=price_df["date"].max(),
    freq="D"
)
commodities = price_df["msp_commodity"].dropna().unique()

daily_index = pd.MultiIndex.from_product(
    [commodities, full_dates],
    names=["msp_commodity", "date"]
)
msp_daily = pd.DataFrame(index=daily_index).reset_index()

msp_daily = pd.merge_asof(
    msp_daily.sort_values("date"),
    msp_long.sort_values("start_date"),
    left_on="date",
    right_on="start_date",
    by="msp_commodity",
    direction="backward"
)
msp_daily = msp_daily[["msp_commodity", "date", "msp"]]

final_df = price_df.merge(msp_daily, on=["msp_commodity", "date"], how="left")
print(" MSP merged")

# ─────────────────────────────────────────────────────────────
# SECTION 3 — WEATHER MERGE
# ─────────────────────────────────────────────────────────────
weather_daily = weather_df.groupby("date").agg({
    "temperature":    "mean",
    "rainfall":       "sum",
    "solar_radiation":"mean",
    "wind_speed":     "mean"
}).reset_index().sort_values("date")

weather_daily["rainfall_30d"]  = weather_daily["rainfall"].rolling(30).sum()
weather_daily["rainfall_60d"]  = weather_daily["rainfall"].rolling(60).sum()
weather_daily["temp_14d_avg"]  = weather_daily["temperature"].rolling(14).mean()

# NEW: Rainfall shock — how different is today from 30d baseline
weather_daily["rainfall_shock_30d"] = (
    (weather_daily["rainfall"] - weather_daily["rainfall"].rolling(30).mean()) /
    (weather_daily["rainfall"].rolling(30).mean() + 1e-6)
)

final_df = final_df.merge(weather_daily, on="date", how="left")
print(" Weather merged")

# ─────────────────────────────────────────────────────────────
# SECTION 4 — FUEL MERGE
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

# Existing fuel lags
fuel_daily["diesel_lag_7"]  = fuel_daily["Diesel_price"].shift(7)
fuel_daily["diesel_lag_30"] = fuel_daily["Diesel_price"].shift(30)
fuel_daily["petrol_lag_7"]  = fuel_daily["Petrol_price"].shift(7)
fuel_daily["petrol_lag_30"] = fuel_daily["Petrol_price"].shift(30)

fuel_daily["diesel_pct_change_30"] = (
    (fuel_daily["Diesel_price"] - fuel_daily["diesel_lag_30"]) /
    (fuel_daily["diesel_lag_30"] + 1e-6)
)

# NEW: 7-day fuel momentum
fuel_daily["diesel_pct_change_7"] = (
    (fuel_daily["Diesel_price"] - fuel_daily["diesel_lag_7"]) /
    (fuel_daily["diesel_lag_7"] + 1e-6)
)
fuel_daily["petrol_pct_change_7"] = (
    (fuel_daily["Petrol_price"] - fuel_daily["petrol_lag_7"]) /
    (fuel_daily["petrol_lag_7"] + 1e-6)
)

final_df = final_df.merge(fuel_daily, on="date", how="left")

fuel_cols = [
    "Petrol_price","Diesel_price",
    "diesel_lag_7","diesel_lag_30",
    "petrol_lag_7","petrol_lag_30",
    "diesel_pct_change_30",
    "diesel_pct_change_7","petrol_pct_change_7"
]
# FIX: .ffill() replaces deprecated fillna(method='ffill')
final_df[fuel_cols] = final_df[fuel_cols].ffill()
print(" Fuel data merged")

# ─────────────────────────────────────────────────────────────
# SECTION 5 — MISSING VALUE HANDLING
# ─────────────────────────────────────────────────────────────
mask = final_df["modal_price"].isna() & (final_df["arrivals"] > 0)
final_df.loc[mask, "modal_price"] = (
    final_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.interpolate())
)

final_df["market_closed_flag"] = (
    final_df["modal_price"].isna() &
    (final_df["arrivals"] == 0)
).astype(int)

# ─────────────────────────────────────────────────────────────
# SECTION 6 — TIME FEATURES
# ─────────────────────────────────────────────────────────────
print(" Engineering time features")
final_df["Year"]      = final_df["date"].dt.year
final_df["Month_Num"] = final_df["date"].dt.month
final_df["DayOfYear"] = final_df["date"].dt.dayofyear
final_df["WeekOfYear"] = final_df["date"].dt.isocalendar().week.astype(int)

# Season label encoding
le = LabelEncoder()
final_df["season_enc"] = le.fit_transform(final_df["season"].fillna("unknown"))

# ─────────────────────────────────────────────────────────────
# SECTION 7 — MSP-BASED FEATURES
# ─────────────────────────────────────────────────────────────
print(" Engineering MSP features")
final_df["price_to_msp_ratio"] = final_df["modal_price"] / final_df["msp"]
final_df["below_msp_flag"]     = (final_df["modal_price"] < final_df["msp"]).astype(int)

# NEW: absolute distance from MSP
final_df["price_above_msp"] = final_df["modal_price"] - final_df["msp"]

# NEW: how much MSP grew year-on-year (per commodity)
final_df["msp_yearly_growth"] = (
    final_df.groupby("commodity")["msp"]
    .transform(lambda x: x.pct_change(365))
    .fillna(0)
)

# ─────────────────────────────────────────────────────────────
# SECTION 8 — ARRIVAL FEATURES
# ─────────────────────────────────────────────────────────────
print(" Engineering arrival features")
final_df = final_df.sort_values(["commodity", "date"])

final_df["arrivals_lag_7"] = (
    final_df.groupby("commodity")["arrivals"].shift(7)
)
final_df["arrivals_pct_change_7"] = (
    (final_df["arrivals"] - final_df["arrivals_lag_7"]) /
    (final_df["arrivals_lag_7"] + 1e-6)
)

final_df["arrival_rolling_14"] = (
    final_df.groupby("commodity")["arrivals"]
    .transform(lambda x: x.rolling(14).mean())
)

# NEW: 7-day arrival rolling average
final_df["arrival_rolling_7"] = (
    final_df.groupby("commodity")["arrivals"]
    .transform(lambda x: x.rolling(7).mean())
)

final_df["supply_stress_index"] = (
    (final_df["arrival_rolling_14"] - final_df["arrivals"]) /
    (final_df["arrival_rolling_14"] + 1e-6)
)

# NEW: Arrival shock — sudden deviation from 7-day average
final_df["arrival_shock"] = (
    (final_df["arrivals"] - final_df["arrival_rolling_7"]) /
    (final_df["arrival_rolling_7"] + 1e-6)
)

# NEW: Supply tightness — price momentum vs arrival ratio
final_df["arrival_lag_3"] = (
    final_df.groupby("commodity")["arrivals"].shift(3)
)

# ─────────────────────────────────────────────────────────────
# SECTION 9 — PRICE LAG FEATURES (grouped by commodity )
# ─────────────────────────────────────────────────────────────
print(" Engineering price lag features (per commodity — no contamination)")
final_df = final_df.sort_values(["commodity", "date"])

for lag in [1, 3, 7, 14, 30]:
    final_df[f"price_lag_{lag}"] = (
        final_df.groupby("commodity")["modal_price"].shift(lag)
    )

# ── Verification: boundary rows must be NaN
final_df["_prev_commodity"] = final_df["commodity"].shift(1)
boundary_rows = final_df[final_df["commodity"] != final_df["_prev_commodity"]]
contaminated  = boundary_rows["price_lag_1"].notna().sum()
if contaminated == 0:
    print("   Lag verification passed — no cross-commodity contamination")
else:
    print(f"  ⚠️  WARNING: {contaminated} boundary rows have non-NaN lag_1")
final_df.drop(columns=["_prev_commodity"], inplace=True)

# ─────────────────────────────────────────────────────────────
# SECTION 10 — ROLLING PRICE STATISTICS (grouped by commodity )
# ─────────────────────────────────────────────────────────────
print(" Engineering rolling price statistics")

# Rolling median — robust to price spikes (shift(1) prevents leakage)
final_df["price_rolling_median_7"] = (
    final_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.shift(1).rolling(7).median())
)

# Rolling means at different horizons
final_df["price_rolling_mean_14"] = (
    final_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.shift(1).rolling(14).mean())
)
final_df["price_rolling_mean_30"] = (
    final_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.shift(1).rolling(30).mean())
)

# Rolling volatility — std dev of past prices
final_df["price_volatility_7"] = (
    final_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.shift(1).rolling(7).std())
)
final_df["price_volatility_30"] = (
    final_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.shift(1).rolling(30).std())
)

# ─────────────────────────────────────────────────────────────
# SECTION 11 — PRICE MOMENTUM FEATURES
# ─────────────────────────────────────────────────────────────
print(" Engineering price momentum features")

final_df["price_pct_change_3"] = (
    final_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.pct_change(3))
)
final_df["price_pct_change_7"] = (
    final_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.pct_change(7))
)
final_df["price_pct_change_30"] = (
    final_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.pct_change(30))
)

# Supply tightness: price momentum when arrivals are low
final_df["supply_tightness"] = (
    final_df["price_pct_change_7"] /
    (final_df["arrivals"] / (final_df["arrival_rolling_7"] + 1e-6) + 1e-6)
)

# ─────────────────────────────────────────────────────────────
# SECTION 12 — NaN FILLING FOR LAG & ROLLING FEATURES
# ─────────────────────────────────────────────────────────────
print(" Filling NaN values in lag & rolling features")

# All lag and rolling feature columns that may have warmup NaNs
lag_cols = [
    "price_lag_1", "price_lag_3", "price_lag_7", "price_lag_14", "price_lag_30",
    "price_rolling_median_7", "price_rolling_mean_14", "price_rolling_mean_30",
    "price_volatility_7", "price_volatility_30",
    "price_pct_change_3", "price_pct_change_7", "price_pct_change_30",
    "arrivals_lag_7", "arrivals_pct_change_7",
    "arrival_rolling_7", "arrival_rolling_14",
    "arrival_shock", "arrival_lag_3", "supply_tightness", "supply_stress_index",
]

# Only include columns that actually exist in the dataframe
lag_cols = [c for c in lag_cols if c in final_df.columns]

before_nan = final_df[lag_cols].isna().sum().sum()
print(f"  Total NaNs before filling : {before_nan}")

# ── STRATEGY: Per commodity
#   Step 1 — Drop warmup rows where the LONGEST lag (price_lag_30) is NaN.
#             This removes only the first ~30 rows per commodity (~90 rows total)
#             which is negligible vs 9,000+ rows.
#   Step 2 — Backfill any remaining shorter-lag NaNs within each commodity.
#             These are rows where lag_30 is valid but lag_1..lag_14 still
#             had NaN due to rolling window requirements (e.g. rolling(7).std).

final_df = final_df.sort_values(["commodity", "date"])

# Step 1: Drop rows where price_lag_30 is NaN (the longest warmup period)
rows_before = len(final_df)
final_df = final_df.dropna(subset=["price_lag_30"])
rows_dropped = rows_before - len(final_df)
print(f"  Warmup rows dropped       : {rows_dropped} ({rows_dropped/rows_before*100:.2f}%)")

# Step 2: Backfill remaining NaNs per commodity
#   Why bfill and not ffill?
#   At the start of a commodity's data, there's nothing before to ffill from.
#   bfill uses the NEXT valid value going backward — so early NaNs get the
#   first real computed value, which is the best available approximation.
final_df[lag_cols] = (
    final_df.groupby("commodity")[lag_cols]
    .transform(lambda x: x.bfill())
)

# Step 3: Any remaining NaNs (extremely rare edge cases) -> fill with
#          per-commodity median as a last resort
final_df[lag_cols] = (
    final_df.groupby("commodity")[lag_cols]
    .transform(lambda x: x.fillna(x.median()))
)

after_nan = final_df[lag_cols].isna().sum().sum()
print(f"  Total NaNs after filling  : {after_nan}")

# Report NaN % per feature after filling
print("\n  NaN % per feature after filling:")
for c in lag_cols:
    pct = final_df[c].isna().mean() * 100
    status = "" if pct == 0 else "⚠️ "
    print(f"    {status} {c:<35} {pct:.2f}%")

# ─────────────────────────────────────────────────────────────
# SECTION 13 — FINAL CLEANUP
# ─────────────────────────────────────────────────────────────
print(" Final cleanup")

# Drop helper/duplicate columns — keep rolling_median_7 this time 
drop_cols = ["msp_commodity"]
final_df.drop(columns=drop_cols, errors="ignore", inplace=True)

# Sort chronologically for clean output
final_df = final_df.sort_values(["commodity", "date"]).reset_index(drop=True)

# Report NaN counts for key new features
key_features = [
    "price_lag_1","price_lag_7","price_lag_30",
    "price_rolling_median_7","price_volatility_7",
    "price_pct_change_7","arrival_shock","supply_tightness"
]
print("\n  NaN counts for new features:")
for f in key_features:
    if f in final_df.columns:
        pct = final_df[f].isna().mean() * 100
        print(f"    {f:<30} {pct:.1f}%")

print(f"\n  Final dataset shape : {final_df.shape}")
print(f"  Total features      : {final_df.shape[1]}")
print(f"  Date range          : {final_df['date'].min().date()} -> {final_df['date'].max().date()}")
print(f"  Commodities         : {final_df['commodity'].unique().tolist()}")

final_df.to_csv("Agri-Hotri_Price_Prediction/Data/Cereal_Final_v2.csv", index=False)
print("\n Saved -> Agri-Hotri_Price_Prediction/Data/Cereal_Final_v2.csv")