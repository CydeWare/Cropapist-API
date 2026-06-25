# Cropapist-API

---

# Cropapist API

Cropapist API is the backend service powering the Cropapist crop yield prediction application. Built with FastAPI and deployed on Azure App Service, it combines machine learning with real-time weather data to predict how well a crop will grow at any location in the world for the current year.

## Concept

Agriculture is heavily influenced by climate. Two of the most critical factors determining crop yield are rainfall and temperature — too little or too much of either can significantly impact harvest outcomes. Cropapist leverages this relationship by:

1. Fetching real weather data for a given location for the current year (combining historical archive data with seasonal forecasts for the remaining months)
2. Feeding that weather data alongside geographic and crop information into a trained machine learning model
3. Returning a predicted yield along with context — how good or bad that yield is relative to historical data for that specific crop

This makes Cropapist useful for farmers, agricultural researchers, or anyone wanting a data-driven estimate of crop performance at a particular location.

## How the API Works

### Weather Data Pipeline

For any given location (latitude/longitude), the API constructs a full-year weather picture by combining two sources:

- **Open-Meteo Archive API** — provides actual historical daily rainfall and temperature data from January 1st of the current year up to 4 days ago
- **Open-Meteo Seasonal Forecast API** — provides forecast data from 4 days ago through December 31st of the current year

These are combined to estimate the total annual rainfall (mm/year) and average annual temperature (°C) for the location.

### Machine Learning Model

The prediction model is a **Random Forest Regressor** trained on a global agricultural dataset. It takes the following features as input:

| Feature | Description |
|---|---|
| `average_rain_fall_mm_per_year` | Total estimated annual rainfall (mm) |
| `avg_temp` | Estimated average annual temperature (°C) |
| `Year` | Current year |
| `latitude` | Location latitude |
| `longitude` | Location longitude |
| `Item_encoded` | Crop type (label-encoded) |

The model outputs a predicted yield in **hg/ha** (hectograms per hectare).

### Yield Classification

The predicted yield is classified relative to historical yield data for that specific crop using Q1/Q3 thresholds:

| Category | Condition |
|---|---|
| **Good** | Predicted yield is above the 75th percentile (Q3) for that crop |
| **Medium** | Predicted yield is between Q1 and Q3 |
| **Bad** | Predicted yield is below the 25th percentile (Q1) for that crop |

A **percentile rank** is also returned, showing exactly where the predicted yield falls within the full historical distribution for that crop.

## Endpoints

### `GET /health`
Returns API status.

### `POST /predict`
Main prediction endpoint.

**Request body:**
```json
{
  "latitude": -6.2,
  "longitude": 106.8,
  "item": "Maize"
}
```

**Response:**
```json
{
  "predicted_yield": 55432.10,
  "yield_category": "good",
  "yield_percentile": 82.4,
  "average_rain_fall_mm_per_year": 2450.3,
  "avg_temp": 27.1,
  "year": 2026,
  "warning": null
}
```

## Supported Crops

Cassava, Maize, Plantains and others, Potatoes, Rice (paddy), Sorghum, Soybeans, Sweet potatoes, Wheat, Yams

## Tech Stack

- **Framework** — FastAPI
- **ML** — scikit-learn (Random Forest Regressor)
- **Weather** — Open-Meteo Archive API + Seasonal Forecast API
- **Deployment** — Azure App Service (Indonesia Central) and Vercel (for frontend)
- **Performance** — In-memory caching (`lru_cache`) and parallel weather fetching for low-latency responses
