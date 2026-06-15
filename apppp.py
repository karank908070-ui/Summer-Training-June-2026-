"""
Automotive Market Intelligence System - Live Demo
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Automotive Market Intelligence System", layout="centered")

# ---------------- Load Saved Models & Objects ----------------
@st.cache_resource
def load_objects():
    lin_model = joblib.load("linear_model.pkl")
    scaler_linear = joblib.load("scaler_linear.pkl")
    log_model = joblib.load("logistic_model.pkl")
    scaler_logistic = joblib.load("scaler_logistic.pkl")
    encoders = joblib.load("encoders.pkl")
    median_price = joblib.load("median_price.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return lin_model, scaler_linear, log_model, scaler_logistic, encoders, median_price, feature_cols

lin_model, scaler_linear, log_model, scaler_logistic, encoders, median_price, feature_cols = load_objects()

st.title("🚗 Automotive Market Intelligence System")
st.write("Enter car details below to get predictions from **two ML models**:")
st.markdown("- **Linear Regression** → Predicted Selling Price\n- **Logistic Regression** → Market Value Category (High / Low)")

st.divider()

# ---------------- User Inputs ----------------
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", sorted(encoders["brand"].classes_))
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)
    km_driven = st.number_input("KM Driven", min_value=0, max_value=500000, value=50000, step=1000)
    seller_type = st.selectbox("Seller Type", sorted(encoders["seller_type"].classes_))
    fuel_type = st.selectbox("Fuel Type", sorted(encoders["fuel_type"].classes_))

with col2:
    transmission_type = st.selectbox("Transmission Type", sorted(encoders["transmission_type"].classes_))
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=18.0, step=0.1)
    engine = st.number_input("Engine (cc)", min_value=500, max_value=5000, value=1200, step=50)
    max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=400.0, value=80.0, step=1.0)
    seats = st.number_input("Seats", min_value=2, max_value=10, value=5)

# ---------------- Predict ----------------
if st.button("🔍 Predict", use_container_width=True):

    # Encode categorical inputs
    brand_enc = encoders["brand"].transform([brand])[0]
    seller_enc = encoders["seller_type"].transform([seller_type])[0]
    fuel_enc = encoders["fuel_type"].transform([fuel_type])[0]
    trans_enc = encoders["transmission_type"].transform([transmission_type])[0]

    # Build feature row in same order as training
    input_data = pd.DataFrame([[
        brand_enc, vehicle_age, km_driven, seller_enc,
        fuel_enc, trans_enc, mileage, engine, max_power, seats
    ]], columns=feature_cols)

    # ---- Linear Regression: Price Prediction (model predicts log price) ----
    input_scaled_lin = scaler_linear.transform(input_data)
    predicted_log_price = lin_model.predict(input_scaled_lin)[0]
    predicted_price = np.exp(predicted_log_price)
    predicted_price = max(predicted_price, 0)  # avoid negative price

    # ---- Logistic Regression: High/Low Value Classification ----
    input_scaled_log = scaler_logistic.transform(input_data)
    pred_class = log_model.predict(input_scaled_log)[0]
    pred_proba = log_model.predict_proba(input_scaled_log)[0]

    st.divider()
    st.subheader("📈 Prediction Results")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Predicted Selling Price", f"₹ {predicted_price:,.0f}")

    with c2:
        label = "High Value 🟢" if pred_class == 1 else "Low Value 🔴"
        confidence = pred_proba[pred_class] * 100
        st.metric("Market Value Category", label, f"{confidence:.1f}% confidence")

    st.caption(f"Classification threshold (median selling price): ₹ {median_price:,.0f}")

    # Probability bar
    st.write("**Class Probabilities (Logistic Regression):**")
    prob_df = pd.DataFrame({
        "Category": ["Low Value", "High Value"],
        "Probability": pred_proba
    })
    st.bar_chart(prob_df.set_index("Category"))

st.divider()
st.caption("Models: Linear Regression (price prediction) & Logistic Regression (value classification) trained on CarDekho dataset.")