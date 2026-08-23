import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Indian Medical Insurance Premium Predictor",
    page_icon="🇮🇳",
    layout="centered"
)

# Load artifacts
@st.cache_resource
def load_assets():
    model = joblib.load('insurance_model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    return model, scaler, columns

try:
    model, scaler, feature_columns = load_assets()
except Exception:
    st.error("Model files not found! Ensure insurance_model.pkl, scaler.pkl, and columns.pkl are uploaded.")
    st.stop()

st.title("🇮🇳 Indian Medical Insurance Premium Predictor")
st.markdown("Predict estimated annual health insurance premiums based on Indian health and demographic indicators.")
st.write("---")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (Years)", 18, 85, 30)
    height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=160, value=70)
    surgeries = st.selectbox("Number of Major Surgeries", [0, 1, 2, 3])
    transplants = st.selectbox("Any Organ Transplants?", ["No", "Yes"])

with col2:
    diabetes = st.selectbox("Diabetes History?", ["No", "Yes"])
    bp = st.selectbox("Blood Pressure Issues?", ["No", "Yes"])
    chronic = st.selectbox("Any Chronic Diseases?", ["No", "Yes"])
    allergies = st.selectbox("Known Allergies?", ["No", "Yes"])
    cancer_history = st.selectbox("Family History of Cancer?", ["No", "Yes"])

st.write("")

if st.button("Calculate Annual Premium (INR)", type="primary", use_container_width=True):
    input_data = {
        'Age': age,
        'Diabetes': 1 if diabetes == "Yes" else 0,
        'BloodPressureProblems': 1 if bp == "Yes" else 0,
        'AnyTransplants': 1 if transplants == "Yes" else 0,
        'AnyChronicDiseases': 1 if chronic == "Yes" else 0,
        'Height': height,
        'Weight': weight,
        'KnownAllergies': 1 if allergies == "Yes" else 0,
        'HistoryOfCancerInFamily': 1 if cancer_history == "Yes" else 0,
        'NumberOfMajorSurgeries': surgeries
    }
    
    input_df = pd.DataFrame([input_data])[feature_columns]
    scaled_input = scaler.transform(input_df)
    predicted_premium = max(0.0, float(model.predict(scaled_input)[0]))
    
    st.success(f"### Estimated Yearly Premium: **₹{predicted_premium:,.2f}**")