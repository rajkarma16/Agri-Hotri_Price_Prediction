
import pandas as pd
import numpy as np

input_file = "Data/Vegetable_Price.xlsx"


sheets = pd.read_excel(input_file, sheet_name=None)
print("Excel Loaded")

product_sheets = {}

for sheet_name, df in sheets.items():
    product, sheet_type = sheet_name.rsplit(" ", 1)
    product_sheets.setdefault(product, {})[sheet_type] = df

final_data = []

for product, data in product_sheets.items():
    if "Price" not in data or "Arrival" not in data:
        continue

    p_df = data["Price"].copy()
    arrival_df = data["Arrival"].copy()

    p_df["Date"] = pd.to_datetime(p_df["Date"])
    arrival_df["Date"] = pd.to_datetime(arrival_df["Date"])

    price_col = [c for c in p_df.columns if c != "Date"][0]
    arrival_col = [c for c in arrival_df.columns if c != "Date"][0]

    p_df.rename(columns={price_col: "modal_price"}, inplace=True)
    arrival_df.rename(columns={arrival_col: "arrivals"}, inplace=True)

    min_date = min(p_df["Date"].min(), arrival_df["Date"].min())
    max_date = max(p_df["Date"].max(), arrival_df["Date"].max())

    full_dates = pd.DataFrame({
        "Date": pd.date_range(min_date, max_date, freq="D")
    })

    merged = (
        full_dates
        .merge(p_df, on="Date", how="left")
        .merge(arrival_df, on="Date", how="left")
    )

    merged["commodity"] = product
    final_data.append(merged)

veg_df = pd.concat(final_data, ignore_index=True)

veg_df["date"] = pd.to_datetime(veg_df["Date"])
veg_df["commodity"] = veg_df["commodity"].str.strip().str.lower()


# Load Weather Data


weather_df = pd.read_csv("Data/maharashtra_daily_weather_2015_2025.csv")
weather_df["date"] = pd.to_datetime(weather_df["date"])

weather_daily = weather_df.groupby("date").agg({
    "temperature": "mean",
    "rainfall": "sum",
    "solar_radiation": "mean",
    "wind_speed": "mean"
}).reset_index()

# Stronger weather signals for vegetables
weather_daily["rainfall_7d"] = weather_daily["rainfall"].rolling(7).sum()
weather_daily["rainfall_15d"] = weather_daily["rainfall"].rolling(15).sum()
weather_daily["temp_7d_avg"] = weather_daily["temperature"].rolling(7).mean()

veg_df = veg_df.merge(weather_daily, on="date", how="left")

#Load Fuel Data
fuel_df = pd.read_csv("Data/Fuel_prices.csv")

# Remove '*' if present
fuel_df["Petrol_price"] = (
    fuel_df["Petrol_price"].astype(str).str.replace("*", "", regex=False).astype(float)
)

fuel_df["Diesel_price"] = (
    fuel_df["Diesel_price"].astype(str).str.replace("*", "", regex=False).astype(float)
)

fuel_df["Month"] = fuel_df["Month"].astype(str).str.strip()

month_map = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
    "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
}

fuel_df["Month_Num"] = fuel_df["Month"].map(month_map)
fuel_df = fuel_df[fuel_df["Month_Num"].notna()]
fuel_df["Year"] = pd.to_numeric(fuel_df["Year"], errors="coerce")

fuel_df = fuel_df[fuel_df["Year"].notna()]
fuel_df["date"] = pd.to_datetime(
    dict(
        year=fuel_df["Year"].astype(int),
        month=fuel_df["Month_Num"].astype(int),
        day=1
    )
)

fuel_df = fuel_df[fuel_df["date"].notna()]

fuel_df = fuel_df.sort_values("date")

full_dates = pd.date_range(
    start=fuel_df["date"].min(),
    end=fuel_df["date"].max(),
    freq="D"
)

fuel_daily = pd.DataFrame({"date": full_dates})

fuel_daily = pd.merge_asof(
    fuel_daily.sort_values("date"),
    fuel_df.sort_values("date"),
    on="date",
    direction="backward"
)

fuel_daily["diesel_lag_7"] = fuel_daily["Diesel_price"].shift(7)
fuel_daily["diesel_lag_30"] = fuel_daily["Diesel_price"].shift(30)

fuel_daily["petrol_lag_7"] = fuel_daily["Petrol_price"].shift(7)
fuel_daily["petrol_lag_30"] = fuel_daily["Petrol_price"].shift(30)

fuel_daily["diesel_pct_change_30"] = (
    (fuel_daily["Diesel_price"] - fuel_daily["diesel_lag_30"]) /
    fuel_daily["diesel_lag_30"]
)

veg_df = veg_df.merge(fuel_daily, on="date", how="left")

veg_df[[
        "Petrol_price","Diesel_price",
        "diesel_lag_7","diesel_lag_30",
        "petrol_lag_7","petrol_lag_30",
        "diesel_pct_change_30"
    ]] = veg_df[[
        "Petrol_price","Diesel_price",
        "diesel_lag_7","diesel_lag_30",
        "petrol_lag_7","petrol_lag_30",
        "diesel_pct_change_30"
    ]].fillna(method="ffill")

#Data Cleaning
veg_df = veg_df[
    (veg_df["modal_price"] > 0) &
    (veg_df["arrivals"] >= 0)
]

veg_df = veg_df.sort_values(["commodity", "date"])

# Outlier smoothing using rolling median (vegetables fluctuate heavily)
veg_df["rolling_median_5"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.rolling(5, min_periods=2).median())
)

outlier_mask = veg_df["modal_price"] > 3 * veg_df["rolling_median_5"]

veg_df.loc[outlier_mask, "modal_price"] = veg_df["rolling_median_5"]

# Missing Value Handling

# Interpolate prices when arrivals exist
mask = veg_df["modal_price"].isna() & (veg_df["arrivals"] > 0)

veg_df.loc[mask, "modal_price"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.interpolate())
)

veg_df["market_closed_flag"] = (
    veg_df["modal_price"].isna() &
    (veg_df["arrivals"] == 0)
).astype(int)

# Core Vegetable Features

# Short-term momentum (vegetables move fast)
veg_df["price_lag_1"] = veg_df.groupby("commodity")["modal_price"].shift(1)
veg_df["price_lag_3"] = veg_df.groupby("commodity")["modal_price"].shift(3)
veg_df["price_lag_7"] = veg_df.groupby("commodity")["modal_price"].shift(7)

veg_df["price_pct_change_3"] = (
    (veg_df["modal_price"] - veg_df["price_lag_3"]) /
    veg_df["price_lag_3"]
)

# Arrival shock
veg_df["arrival_lag_3"] = veg_df.groupby("commodity")["arrivals"].shift(3)

veg_df["arrival_shock"] = (
    (veg_df["arrivals"] - veg_df["arrival_lag_3"]) /
    veg_df["arrival_lag_3"]
)

# Supply tightness index
veg_df["arrival_rolling_7"] = (
    veg_df.groupby("commodity")["arrivals"]
    .transform(lambda x: x.rolling(7).mean())
)

veg_df["supply_tightness"] = (
    (veg_df["arrival_rolling_7"] - veg_df["arrivals"]) /
    veg_df["arrival_rolling_7"]
)

# Price volatility (important for onion/tomato)
veg_df["price_volatility_7"] = (
    veg_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.rolling(7).std())
)

veg_df = veg_df.drop(columns=["Date", "rolling_median_5"], errors="ignore")

print("Vegetable dataset shape:", veg_df.shape)
print(veg_df.head())

veg_df.to_csv("Data/Vegetable_Final.csv", index=False)