# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

st.set_page_config(page_title="Medical Insurance Predictor", page_icon="🏥", layout="centered")

# Load trained assets
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('insurance_model.keras')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    return model, scaler, columns

try:
    model, scaler, feature_columns = load_assets()
except Exception as e:
    st.error("Model or preprocessor files not found. Please run 'python train.py' first.")
    st.stop()

st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("Enter individual health metrics to estimate annual medical insurance charges using a Deep Learning ANN.")
st.write("---")

# User Inputs Form
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=30, step=1)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    children = st.selectbox("Number of Children/Dependents", options=[0, 1, 2, 3, 4, 5], index=0)

with col2:
    sex = st.selectbox("Sex", options=["Female", "Male"])
    smoker = st.selectbox("Smoker Status", options=["No", "Yes"])
    region = st.selectbox("US Region", options=["Northeast", "Northwest", "Southeast", "Southwest"])

# Prediction Logic
if st.button("Calculate Predicted Charges", type="primary", use_container_width=True):
    # Construct base dictionary matching training schema
    input_data = {
        'age': age,
        'bmi': bmi,
        'children': children,
        'sex_male': 1 if sex == "Male" else 0,
        'smoker_yes': 1 if smoker == "Yes" else 0,
        'region_northwest': 1 if region == "Northwest" else 0,
        'region_southeast': 1 if region == "Southeast" else 0,
        'region_southwest': 1 if region == "Southwest" else 0,
    }
    
    # Ensure DataFrame columns match exact order of train.py
    input_df = pd.DataFrame([input_data])[feature_columns]
    
    # Scale input features
    scaled_features = scaler.transform(input_df)
    
    # Predict with ANN
    prediction = model.predict(scaled_features, verbose=0)[0][0]
    final_cost = max(0.0, float(prediction))  # Prevent negative cost values
    
    st.success(f"### Estimated Annual Charge: **${final_cost:,.2f}**")
    
    # Contextual Callout
    if smoker == "Yes":
        st.warning("⚠️ Smoking status significantly increases the projected cost risk.")