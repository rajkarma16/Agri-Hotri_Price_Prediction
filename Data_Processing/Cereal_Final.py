# merges Cereal sorted with weather and msp data and fills missing values
# After this process analysis and modeltraining can be performed

import pandas as pd
import numpy as np

input_file = "Data\Cereal_Price.xlsx" 

# Read all sheets
sheets = pd.read_excel(input_file, sheet_name=None)

print("Excel Loaded")
# Group sheets by product name
product_sheets = {}

for sheet_name, df in sheets.items():
    product, sheet_type = sheet_name.rsplit(" ", 1)
    product_sheets.setdefault(product, {})[sheet_type] = df

final_data = []

for product, data in product_sheets.items():
    if "Price" not in data or "Arrival" not in data:
        continue  # skip incomplete pairs

    p_df = data["Price"].copy()
    arrival_df = data["Arrival"].copy()

    # Convert Date column
    p_df["Date"] = pd.to_datetime(p_df["Date"])
    arrival_df["Date"] = pd.to_datetime(arrival_df["Date"])

    # Rename value columns
    price_col = [c for c in p_df.columns if c != "Date"][0]
    arrival_col = [c for c in arrival_df.columns if c != "Date"][0]

    p_df.rename(columns={price_col: "Price"}, inplace=True)
    arrival_df.rename(columns={arrival_col: "Arrival"}, inplace=True)

    # Create full date range
    min_date = min(p_df["Date"].min(), arrival_df["Date"].min())
    max_date = max(p_df["Date"].max(), arrival_df["Date"].max())

    full_dates = pd.DataFrame({
        "Date": pd.date_range(min_date, max_date, freq="D")
    })

    # Merge
    merged = (
        full_dates
        .merge(p_df, on="Date", how="left")
        .merge(arrival_df, on="Date", how="left")
    )

    # Add product name
    merged["commodity"] = product

    # Format date
    merged["Date"] = merged["Date"].dt.strftime("%d-%m-%Y")

    final_data.append(merged)

# Combine all products into one dataframe
price_df = pd.concat(final_data, ignore_index=True)

print("Arrival And Prices are merged in single file")


msp_df = pd.read_excel("Data/MSP.xlsx")
weather_df = pd.read_csv("Data/maharashtra_daily_weather_2015_2025.csv")

print(" MSP and Weather Data Loaded")

# Convert dates
price_df["date"] = pd.to_datetime(price_df["Date"])
weather_df["date"] = pd.to_datetime(weather_df["date"])


# Standardize commodity names
price_df["commodity"] = price_df["commodity"].str.strip().str.lower()
msp_df["Commodity"] = msp_df["Commodity"].str.strip().str.lower()

Column_map = {
    "Price" : "modal_price",
    "Min Price (₹)" : "min_price",
    "Arrival" : "arrivals",
    "Max Price (₹)" : "max_price"
}

price_df = price_df.rename(columns=Column_map)


price_df = price_df[
    (price_df["modal_price"] > 0) &
    (price_df["min_price"] <= price_df["modal_price"]) &
    (price_df["modal_price"] <= price_df["max_price"]) &
    (price_df["arrivals"] >= 0)
]

price_df = price_df.sort_values(["commodity", "date"])

# 7-day rolling median for outlier detection
price_df["rolling_median_7"] = (
    price_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.rolling(7, min_periods=3).median())
)

outlier_mask = price_df["modal_price"] > 3 * price_df["rolling_median_7"]

price_df.loc[outlier_mask & (price_df["arrivals"] > 0), "modal_price"] = \
    price_df.loc[outlier_mask & (price_df["arrivals"] > 0), "rolling_median_7"]

# Zero Arrival Flag
price_df["zero_arrival_flag"] = (price_df["arrivals"] == 0).astype(int)

# Merging MSP AND CEREAL SORTED


season_mapping = {
    "rice": "kharif",
    "arhar (tur dal)": "kharif",
    "wheat": "rabi"
}

price_df["season"] = price_df["commodity"].map(season_mapping)
price_df["msp_commodity"] = price_df["commodity"]

msp_df.rename(columns={"Commodity": "msp_commodity"}, inplace=True)

msp_long = msp_df.melt(
    id_vars=["msp_commodity"],
    var_name="year",
    value_name="msp"
)

def get_msp_start_date(row):
    year_str = row["year"]
    commodity = row["msp_commodity"]

    start_year = int(year_str.split("-")[0])

    # Identify season
    if commodity in ["paddy (common)", "tur (arhar)"]:
        # Kharif → starts Oct 1
        return pd.Timestamp(f"{start_year}-10-01")
    elif commodity == "wheat":
        # Rabi → starts April 1 of next calendar year
        return pd.Timestamp(f"{start_year + 1}-04-01")
    else:
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

final_df = price_df.merge(
    msp_daily,
    on=["msp_commodity", "date"],
    how="left"
)

print("MSP Merged")
# MERGE WEATHER DATA WITH MSP AND PRICE

weather_daily = weather_df.groupby("date").agg({
    "temperature": "mean",
    "rainfall": "sum",
    "solar_radiation": "mean",
    "wind_speed": "mean"
}).reset_index()

weather_daily = weather_daily.sort_values("date")

# Create cereal-appropriate lag features
weather_daily["rainfall_30d"] = weather_daily["rainfall"].rolling(30).sum()
weather_daily["rainfall_60d"] = weather_daily["rainfall"].rolling(60).sum()
weather_daily["temp_14d_avg"] = weather_daily["temperature"].rolling(14).mean()

final_df = final_df.merge(weather_daily, on="date", how="left")

print("Weather Data Merged")

# Handling Missing Values

mask = final_df["modal_price"].isna() & (final_df["arrivals"] > 0)

final_df.loc[mask, "modal_price"] = (
    final_df.groupby("commodity")["modal_price"]
    .transform(lambda x: x.interpolate())
)

# Market closed flag
final_df["market_closed_flag"] = (
    final_df["modal_price"].isna() &
    (final_df["arrivals"] == 0)
).astype(int)

# CORE FEATURES

print("Creating Core Features")
# msp 
final_df["price_to_msp_ratio"] = final_df["modal_price"] / final_df["msp"]
final_df["below_msp_flag"] = (final_df["modal_price"] < final_df["msp"]).astype(int)

# Arrival lags
final_df = final_df.sort_values(["commodity", "date"])

final_df["arrivals_lag_7"] = (
    final_df.groupby("commodity")["arrivals"].shift(7)
)

final_df["arrivals_pct_change_7"] = (
    (final_df["arrivals"] - final_df["arrivals_lag_7"]) /
    final_df["arrivals_lag_7"]
)

# Supply Stress Index
final_df["arrival_rolling_14"] = (
    final_df.groupby("commodity")["arrivals"]
    .transform(lambda x: x.rolling(14).mean())
)

final_df["supply_stress_index"] = (
    (final_df["arrival_rolling_14"] - final_df["arrivals"]) /
    final_df["arrival_rolling_14"]
)

#Adding Fuel Prices
print("Adding Fuel Prices to data")
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

final_df = final_df.merge(fuel_daily, on="date", how="left")

final_df[[
        "Petrol_price","Diesel_price",
        "diesel_lag_7","diesel_lag_30",
        "petrol_lag_7","petrol_lag_30",
        "diesel_pct_change_30"
    ]] = final_df[[
        "Petrol_price","Diesel_price",
        "diesel_lag_7","diesel_lag_30",
        "petrol_lag_7","petrol_lag_30",
        "diesel_pct_change_30"
    ]].fillna(method="ffill")

# Drop helper column

print("Creating Final Dataset")
final_df = final_df.sort_values(["commodity", "date"])

final_df.drop(columns=[["date","msp_commodity","rolling_median_7" ]], inplace=True, errors="ignore")

print("Final dataset shape:", final_df.shape)
print(final_df.head())

# Optionally save
final_df.to_csv("Data\Cereal_Final.csv", index=False)
print("Data Saved as CSV file")