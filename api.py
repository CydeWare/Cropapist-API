from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import date, timedelta
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
from typing import Optional
import re
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time


stats_df = pd.read_csv("yield_stats_per_crop.csv", index_col="Item")

with open("yield_percentiles_per_crop.pkl", "rb") as f:
    percentile_data = pickle.load(f)

model = joblib.load("yield_model.pkl")
label_encoder = joblib.load("item_label_encoder.pkl")


def make_session():
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

SESSION = make_session()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    latitude: float
    longitude: float
    item: str


class PredictResponse(BaseModel):
    predicted_yield: float
    yield_category: str
    yield_percentile: Optional[float]
    average_rain_fall_mm_per_year: float
    avg_temp: float
    year: int
    warning: Optional[str]


def classify_yield_per_crop(crop: str, predicted_yield: float):
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


@lru_cache(maxsize=256)
def fetch_archive_cached(lat: float, lon: float, first_day_str: str, today_minus_4_str: str):
    archive_url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={first_day_str}&end_date={today_minus_4_str}"
        "&daily=rain_sum,temperature_2m_mean&timezone=auto"
    )
    start = time.time()
    print("Archive URL:", archive_url)
    response = SESSION.get(archive_url, timeout=30)
    print(f"Archive API took {time.time() - start:.2f}s")
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=256)
def fetch_seasonal_cached(lat: float, lon: float, start_date: str, end_date: str):
    def build_url(s_date, e_date):
        return (
            "https://seasonal-api.open-meteo.com/v1/seasonal"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={s_date}&end_date={e_date}"
            "&daily=rain_sum,temperature_2m_mean&timezone=auto"
        )

    try:
        start = time.time()
        response = SESSION.get(build_url(start_date, end_date), timeout=30)
        print(f"Seasonal API took {time.time() - start:.2f}s")
    except requests.exceptions.RequestException:
        return None, None, "Seasonal forecast unavailable."

    if response.status_code == 200:
        return response.json(), end_date, None

    try:
        error_json = response.json()
        reason = error_json.get("reason", "")
        matches = re.findall(r"\d{4}-\d{2}-\d{2}", reason)

        if len(matches) >= 2:
            min_date = matches[0]
            max_date = matches[1]

            clamped_start = max(start_date, min_date)
            clamped_end = min(end_date, max_date)

            if clamped_start >= clamped_end:
                return None, None, "Seasonal forecast not available for this period."

            retry_response = SESSION.get(build_url(clamped_start, clamped_end), timeout=30)
            if retry_response.status_code == 200:
                return retry_response.json(), clamped_end, None
    except Exception:
        pass

    return None, None, "Seasonal forecast unavailable for this location or date."


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/health")
def health_check():
    return {"message": "API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict_yield(payload: PredictRequest):
    today = date.today()
    year = today.year

    if payload.longitude < -180 or payload.longitude > 180 or payload.latitude < -90 or payload.latitude > 90:
        return PredictResponse(
            predicted_yield=0.0,
            yield_category="invalid_location",
            yield_percentile=None,
            average_rain_fall_mm_per_year=0.0,
            avg_temp=0.0,
            year=year,
            warning="Invalid coordinates."
        )

    lat = round(payload.latitude, 2)
    lon = round(payload.longitude, 2)

    first_day_str = date(year, 1, 1).isoformat()
    today_minus_4_str = (today - timedelta(days=4)).isoformat()
    last_day_str = date(year, 12, 31).isoformat()

    with ThreadPoolExecutor(max_workers=2) as executor:
        archive_future = executor.submit(
            fetch_archive_cached, lat, lon, first_day_str, today_minus_4_str
        )
        seasonal_future = executor.submit(
            fetch_seasonal_cached, lat, lon, today_minus_4_str, last_day_str
        )

        try:
            archive_json = archive_future.result()
        except requests.exceptions.RequestException:
            return PredictResponse(
                predicted_yield=0.0,
                yield_category="weather_unavailable",
                yield_percentile=None,
                average_rain_fall_mm_per_year=0.0,
                avg_temp=0.0,
                year=year,
                warning="Archive weather service unavailable."
            )

        seasonal_json, seasonal_used_end, seasonal_warning = seasonal_future.result()

    archive_rain = archive_json["daily"]["rain_sum"]
    archive_temp = archive_json["daily"]["temperature_2m_mean"]

    seasonal_rain = []
    seasonal_temp = []

    if seasonal_json:
        seasonal_rain = seasonal_json["daily"]["rain_sum"]
        seasonal_temp = seasonal_json["daily"]["temperature_2m_mean"]

    all_rain = [x for x in (archive_rain + seasonal_rain) if x is not None]
    all_temp = [x for x in (archive_temp + seasonal_temp) if x is not None]

    average_rain_fall_mm_per_year = float(sum(all_rain))
    avg_temp_year = float(sum(all_temp) / len(all_temp)) if all_temp else 0.0

    warning_msg = seasonal_warning

    if avg_temp_year < 0:
        warning_msg = "Average yearly temperature below 0°C. Prediction may be unreliable."
    elif average_rain_fall_mm_per_year < 50:
        warning_msg = "Yearly rainfall extremely low. Prediction may be unreliable."
    elif avg_temp_year > 40:
        warning_msg = "Yearly temperature extremely high. Prediction may be unreliable."

    try:
        item_encoded = label_encoder.transform([payload.item])[0]
    except ValueError:
        return PredictResponse(
            predicted_yield=0.0,
            yield_category="unknown_item",
            yield_percentile=None,
            average_rain_fall_mm_per_year=average_rain_fall_mm_per_year,
            avg_temp=avg_temp_year,
            year=year,
            warning=warning_msg
        )

    features = np.array([[
        average_rain_fall_mm_per_year,
        avg_temp_year,
        year,
        lat,
        lon,
        item_encoded
    ]])

    pred = model.predict(features)[0]
    category, percentile = classify_yield_per_crop(payload.item, pred)

    return PredictResponse(
        predicted_yield=float(pred),
        yield_category=category,
        yield_percentile=percentile,
        average_rain_fall_mm_per_year=average_rain_fall_mm_per_year,
        avg_temp=avg_temp_year,
        year=year,
        warning=warning_msg,
    )
