from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import logging

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("housing_api")

# --- Load model and artifacts at startup ---
try:
    model = joblib.load("lgbm_housing_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    ocean_proximity_categories = joblib.load("ocean_proximity_categories.pkl")
    logger.info("Model and artifacts loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    raise

app = FastAPI(
    title="California Housing Price Prediction API",
    description="Predicts median house value using a tuned LightGBM model.",
    version="1.0.0"
)


# --- Input schema (raw features, before engineering) ---
class HousingInput(BaseModel):
    longitude: float = Field(..., example=-122.23)
    latitude: float = Field(..., example=37.88)
    housing_median_age: float = Field(..., example=41.0)
    total_rooms: float = Field(..., example=880.0)
    total_bedrooms: float = Field(..., example=129.0)
    population: float = Field(..., example=322.0)
    households: float = Field(..., example=126.0)
    median_income: float = Field(..., example=8.3252)
    ocean_proximity: str = Field(..., example="NEAR BAY")


class PredictionResponse(BaseModel):
    predicted_house_value: float
    confidence_note: str


def engineer_features(data: HousingInput) -> pd.DataFrame:
    """Recreate the same feature engineering used during training."""
    df = pd.DataFrame([data.model_dump()])

    # Engineered ratio features
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    # Drop raw collinear columns (matches training pipeline)
    df = df.drop(columns=["population", "total_bedrooms", "households"])

    # One-hot encode ocean_proximity, aligned to training categories
    for category in ocean_proximity_categories:
        col_name = f"ocean_proximity_{category}"
        df[col_name] = (df["ocean_proximity"] == category).astype(int)
    df = df.drop(columns=["ocean_proximity"])

    # Align column order exactly to training data; fill any missing with 0
    df = df.reindex(columns=model_columns, fill_value=0)

    return df


@app.get("/")
async def root():
    return {"message": "California Housing Price Prediction API is running."}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: HousingInput):
    try:
        features = engineer_features(data)
        prediction = model.predict(features)[0]

        logger.info(f"Prediction made: {prediction:.2f}")

        return PredictionResponse(
            predicted_house_value=round(float(prediction), 2),
            confidence_note="Model RMSE ~$40,755 on test set; treat as an estimate."
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")