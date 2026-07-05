import streamlit as st
import requests

# --- Page config ---
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("🏠 California Housing Price Predictor")
st.markdown(
    "Enter the district details below to estimate the median house value, "
    "powered by a tuned LightGBM model."
)

st.divider()

# --- Input form ---
with st.form("housing_form"):
    col1, col2 = st.columns(2)

    with col1:
        longitude = st.number_input("Longitude", value=-122.23, format="%.4f")
        latitude = st.number_input("Latitude", value=37.88, format="%.4f")
        housing_median_age = st.number_input("Housing Median Age", value=41.0, min_value=0.0)
        total_rooms = st.number_input("Total Rooms", value=880.0, min_value=1.0)
        total_bedrooms = st.number_input("Total Bedrooms", value=129.0, min_value=1.0)

    with col2:
        population = st.number_input("Population", value=322.0, min_value=1.0)
        households = st.number_input("Households", value=126.0, min_value=1.0)
        median_income = st.number_input(
            "Median Income (in tens of thousands, e.g. 8.3 = $83k)",
            value=8.3252, min_value=0.0, format="%.4f"
        )
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
        )

    submitted = st.form_submit_button("Predict House Value", use_container_width=True)

# --- Handle prediction request ---
if submitted:
    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }

    with st.spinner("Getting prediction..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()
                predicted_value = result["predicted_house_value"]
                confidence_note = result["confidence_note"]

                st.success("Prediction complete!")
                st.metric(
                    label="Predicted Median House Value",
                    value=f"${predicted_value:,.2f}"
                )
                st.caption(confidence_note)
            else:
                st.error(f"API returned an error: {response.status_code} — {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API. Make sure the FastAPI server is running "
                "at `http://127.0.0.1:8000` (run `uvicorn main:app --reload`)."
            )
        except requests.exceptions.Timeout:
            st.error("The request timed out. Please try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.divider()
st.caption(
    "Model: LightGBM (tuned) | R² ≈ 0.833 | RMSE ≈ $40,755 on held-out test data."
)