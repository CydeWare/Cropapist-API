import streamlit as st
import joblib
import numpy as np
import requests
import pandas as pd
import pickle
import re

from datetime import date, timedelta

stats_df = pd.read_csv("yield_stats_per_crop.csv", index_col="Item")

with open("yield_percentiles_per_crop.pkl", "rb") as f:
    percentile_data = pickle.load(f)

model = joblib.load("yield_model.pkl")
label_encoder = joblib.load("item_label_encoder.pkl")


def classify_yield_per_crop(crop, predicted_yield):
    if crop not in stats_df.index or crop not in percentile_data:
        return "unknown_crop", None

    row = stats_df.loc[crop]

    q1 = row["q1"]
    q3 = row["q3"]

    if predicted_yield < q1:
        category = "bad"
    elif predicted_yield > q3:
        category = "good"
    else:
        category = "medium"

    values = percentile_data[crop]["values"]
    n = percentile_data[crop]["n"]

    pos = np.searchsorted(values, predicted_yield, side="right")
    percentile = (pos / n) * 100.0

    return category, percentile


def fetch_seasonal_with_auto_limit(base_url, lat, lon, start_date, end_date):
    def build_url(e_date):
        return (
            base_url
            + f"?latitude={lat}&longitude={lon}"
            + f"&start_date={start_date}&end_date={e_date}"
            + "&daily=rain_sum,temperature_2m_mean&timezone=auto"
        )

    try:
        url = build_url(end_date)

        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            return response.json(), None

        error_json = response.json()
        reason = error_json.get("reason", "")

        matches = re.findall(r"\d{4}-\d{2}-\d{2}", reason)

        if len(matches) >= 2:
            max_date = matches[-1]

            retry_url = build_url(max_date)

            retry_response = requests.get(
                retry_url,
                timeout=30
            )

            if retry_response.status_code == 200:
                return retry_response.json(), None

    except Exception as e:
        return None, str(e)

    return None, "Seasonal forecast unavailable"


st.title("Crop Yield Prediction")

crop = st.selectbox(
    "Select Crop",
    [
        "Cassava",
        "Maize",
        "Plantains and others",
        "Potatoes",
        "Rice, paddy",
        "Sorghum",
        "Soybeans",
        "Sweet potatoes",
        "Wheat",
        "Yams",
    ],
)

lat = st.number_input(
    "Latitude",
    value=0.0,
    min_value=-90.0,
    max_value=90.0,
)

lon = st.number_input(
    "Longitude",
    value=0.0,
    min_value=-180.0,
    max_value=180.0,
)

if st.button("Predict Yield"):
    with st.spinner("Fetching weather data..."):

        today = date.today()
        year = today.year

        first_day = date(year, 1, 1)
        today_minus_4 = today - timedelta(days=4)
        last_day = date(year, 12, 31)

        archive_url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={first_day.isoformat()}"
            f"&end_date={today_minus_4.isoformat()}"
            "&daily=rain_sum,temperature_2m_mean"
            "&timezone=auto"
        )

        try:
            archive_response = requests.get(
                archive_url,
                timeout=30
            )

            archive_response.raise_for_status()

            archive_json = archive_response.json()

            seasonal_json, warning = (
                fetch_seasonal_with_auto_limit(
                    "https://seasonal-api.open-meteo.com/v1/seasonal",
                    lat,
                    lon,
                    today_minus_4.isoformat(),
                    last_day.isoformat(),
                )
            )

            archive_rain = archive_json["daily"]["rain_sum"]
            archive_temp = archive_json["daily"]["temperature_2m_mean"]

            seasonal_rain = []
            seasonal_temp = []

            if seasonal_json:
                seasonal_rain = seasonal_json["daily"]["rain_sum"]
                seasonal_temp = seasonal_json["daily"]["temperature_2m_mean"]

            all_rain = [
                x for x in archive_rain + seasonal_rain
                if x is not None
            ]

            all_temp = [
                x for x in archive_temp + seasonal_temp
                if x is not None
            ]

            rainfall = float(sum(all_rain))

            avg_temp = (
                float(sum(all_temp) / len(all_temp))
                if all_temp
                else 0.0
            )

            item_encoded = label_encoder.transform([crop])[0]

            features = np.array(
                [[
                    rainfall,
                    avg_temp,
                    year,
                    lat,
                    lon,
                    item_encoded
                ]]
            )

            prediction = model.predict(features)[0]

            category, percentile = (
                classify_yield_per_crop(
                    crop,
                    prediction
                )
            )

            st.success("Prediction completed")

            st.write(
                f"Predicted Yield: "
                f"{prediction:.2f} hg/ha"
            )

            st.write(
                f"Category: {category}"
            )

            if percentile is not None:
                st.write(
                    f"Percentile: "
                    f"{percentile:.2f}%"
                )

            st.write(
                f"Average Rainfall: "
                f"{rainfall:.2f} mm/year"
            )

            st.write(
                f"Average Temperature: "
                f"{avg_temp:.2f} °C"
            )

            if warning:
                st.warning(warning)

        except Exception as e:
            st.error(str(e))
