import requests
import pandas as pd

# Maharashtra (Mumbai as representative point)
LATITUDE = 19.0760
LONGITUDE = 72.8777

START_DATE = "2015-01-01"
END_DATE = "2025-10-30"

URL = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": [
        "temperature_2m_mean",
        "precipitation_sum",
        "shortwave_radiation_sum",
        "wind_speed_10m_max"
    ],
    "timezone": "Asia/Kolkata"
}

response = requests.get(URL, params=params)
response.raise_for_status()
data = response.json()

df = pd.DataFrame({
    "date": pd.to_datetime(data["daily"]["time"]),
    "temperature": data["daily"]["temperature_2m_mean"],
    "rainfall": data["daily"]["precipitation_sum"],
    "solar_radiation": data["daily"]["shortwave_radiation_sum"],
    "wind_speed": data["daily"]["wind_speed_10m_max"]
})

df["state"] = "Maharashtra"

df.to_csv("maharashtra_daily_weather_2015_2025.csv", index=False)

print("Data saved: maharashtra_daily_weather_2015_2025.csv")
print(df.head())
print(df.tail())
