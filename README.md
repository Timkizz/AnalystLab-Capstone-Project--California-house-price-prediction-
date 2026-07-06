HEAD
# AnalystLabAfrica Capstone Project — California Housing Price Prediction

## Overview

This project builds a regression model to predict median house values in California using the California Housing dataset. It covers the full ML pipeline: exploratory data analysis, feature engineering, preprocessing, model training, evaluation, and deployment via a FastAPI backend and Streamlit frontend.

## Table of Contents

- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [API Reference](#api-reference)
- [Tools & Libraries](#tools--libraries)
- [Author](#author)

## Dataset

The dataset contains housing information across California districts, including:

- `longitude`, `latitude` — geographic coordinates
- `housing_median_age`, `total_rooms`, `total_bedrooms`
- `population`, `households`, `median_income`
- `ocean_proximity` — categorical distance to the ocean
- `median_house_value` — target variable

## Exploratory Data Analysis

- Analyzed feature distributions via histograms; identified skew in `total_rooms`, `total_bedrooms`, `population`, and `households`.
- Examined `ocean_proximity` category counts — `ISLAND` is a rare category (~5 rows).
- Visualized relationships between features and `median_house_value` using scatter plots; `median_income` showed the strongest positive correlation.
- Used boxplots to compare `median_house_value` across `ocean_proximity` categories — `INLAND` homes had the lowest median value, while coastal categories were higher.
- Noted a capping effect at $500,000 in `median_house_value`.

## Feature Engineering

Engineered the following features to improve model performance:

- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`

## Preprocessing

- One-hot encoded the `ocean_proximity` categorical feature.
- Performed train-test split for model evaluation.

## Modeling

Trained and compared the following models:

- Random Forest
- XGBoost
- LightGBM

## Results

**Best Model: LightGBM**

|Metric  |Value    |
|--------|---------|
|R² Score|0.8334   |
|RMSE    |40,755.00|

LightGBM outperformed the other models, explaining ~83% of the variance in median house values.

## Deployment

- Serialized the trained LightGBM model using `joblib`.
- Built a **FastAPI** backend to serve predictions, with Pydantic models for input validation.
- Built a **Streamlit** frontend that connects to the FastAPI backend, allowing users to input housing features and get real-time median house value predictions.

## Project Structure

```
AnalystLabAfrica_Capstone_Project/
├── housing.ipynb           # EDA, feature engineering, modeling notebook
├── model.pkl                # Serialized LightGBM model (joblib)
├── api/
│   └── main.py               # FastAPI backend
├── app/
│   └── app.py                 # Streamlit frontend
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/AnalystLabAfrica_Capstone_Project.git
cd AnalystLabAfrica_Capstone_Project
```

### 2. Set up a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI backend

```bash
cd api
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

### 5. Run the Streamlit frontend

In a new terminal:

```bash
cd app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## API Reference

|Method|Endpoint  |Description                                                            |
|------|----------|-----------------------------------------------------------------------|
|GET   |`/health` |Health check — confirms the API is running and model is loaded         |
|POST  |`/predict`|Accepts housing feature inputs and returns predicted median house value|

**Sample request to `/predict`:**

```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
```

**Sample response:**

```json
{
  "predicted_price": 452600.00,
  "confidence_tier": "high"
}
```

> Note: adjust field names/values above to match your actual Pydantic schema if it differs.

## Tools & Libraries

- Python, pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost, LightGBM
- FastAPI, Pydantic, joblib
- Streamlit

## Author

Michael — ML intern at Analyst Lab Africa.

# AnalystLab-Capstone-Project--California-house-price-prediction ad0d5f73e2ec46366c028981c81d3c1a7c1b6bf2
