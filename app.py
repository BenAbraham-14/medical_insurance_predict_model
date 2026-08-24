import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="ArogyaPay | Health Insurance Predictor",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for Modern Card UI
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .metric-title { font-size: 1.1rem; opacity: 0.9; margin-bottom: 8px; }
    .metric-value { font-size: 2.3rem; font-weight: 700; }
    .metric-sub { font-size: 0.85rem; opacity: 0.8; margin-top: 6px; }
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_assets():
    model = joblib.load('insurance_model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    return model, scaler, columns

try:
    model, scaler, feature_columns = load_assets()
except Exception:
    st.error("⚠️ Model artifacts missing. Ensure insurance_model.pkl, scaler.pkl, and columns.pkl are uploaded to GitHub.")
    st.stop()

# Sidebar: Context & Health Guidelines
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/shield.png", width=110)
    st.title("About ArogyaPay")
    st.caption("AI-Powered Health Insurance Premium Estimator for India")
    st.markdown("---")
    st.markdown("""
    **How It Works:**
    * 🧠 **Engine:** Multi-Layer Perceptron (ANN)
    * 📊 **Dataset:** Kaggle Indian Medical Premium Registry
    * ⚡ **Output:** Estimated Annual Premium (₹ INR)
    """)
    st.info("💡 **Tip:** Regular check-ups and maintaining a normal BMI (18.5–24.9) reduce baseline actuarial risk.")

# Header
st.title("🛡️ Health Insurance Premium Estimator")
st.markdown("Enter demographic indicators and clinical history below to evaluate actuarial health risk and projected annual premiums.")
st.write("")

# Form Section in Two Cards
tab1, tab2 = st.tabs(["📋 Patient Profile & Clinical Assessment", "📈 How Premium is Calculated"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("👤 Demographic & Vitals")
        age = st.slider("Age (Years)", 18, 85, 28)
        
        c_h, c_w = st.columns(2)
        with c_h:
            height = st.number_input("Height (cm)", min_value=120, max_value=220, value=172, step=1)
        with c_w:
            weight = st.number_input("Weight (kg)", min_value=30, max_value=160, value=68, step=1)
            
        # Computed BMI Indicator
        bmi = weight / ((height / 100) ** 2)
        if bmi < 18.5:
            bmi_status = "Underweight"
            bmi_color = "normal"
        elif 18.5 <= bmi <= 24.9:
            bmi_status = "Normal Weight"
            bmi_color = "normal"
        elif 25.0 <= bmi <= 29.9:
            bmi_status = "Overweight"
            bmi_color = "off"
        else:
            bmi_status = "Obese"
            bmi_color = "inverse"
            
        st.metric(label="Calculated BMI", value=f"{bmi:.1f} kg/m²", delta=bmi_status, delta_color=bmi_color)

    with col2:
        st.subheader("🩺 Medical History & Risk Factors")
        
        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            diabetes = st.toggle("Diabetes History", value=False)
            bp = st.toggle("Blood Pressure Issues", value=False)
            chronic = st.toggle("Any Chronic Diseases", value=False)
        with c_diag2:
            transplants = st.toggle("Any Organ Transplants", value=False)
            allergies = st.toggle("Known Allergies", value=False)
            cancer_history = st.toggle("Family History of Cancer", value=False)
            
        surgeries = st.select_slider(
            "Major Surgeries Undergone",
            options=[0, 1, 2, 3],
            value=0,
            help="Select the total count of invasive/major surgeries."
        )

    st.write("")
    calculate_btn = st.button("Calculate Estimated Premium ⚡", type="primary", use_container_width=True)

    if calculate_btn:
        input_data = {
            'Age': age,
            'Diabetes': 1 if diabetes else 0,
            'BloodPressureProblems': 1 if bp else 0,
            'AnyTransplants': 1 if transplants else 0,
            'AnyChronicDiseases': 1 if chronic else 0,
            'Height': height,
            'Weight': weight,
            'KnownAllergies': 1 if allergies else 0,
            'HistoryOfCancerInFamily': 1 if cancer_history else 0,
            'NumberOfMajorSurgeries': surgeries
        }
        
        input_df = pd.DataFrame([input_data])[feature_columns]
        scaled_input = scaler.transform(input_df)
        predicted_annual = max(0.0, float(model.predict(scaled_input)[0]))
        monthly_approx = predicted_annual / 12

        st.markdown("---")
        
        # Results Cards Layout
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ESTIMATED ANNUAL PREMIUM</div>
                <div class="metric-value">₹ {predicted_annual:,.0f}</div>
                <div class="metric-sub">Billed yearly (excl. 18% GST)</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                <div class="metric-title">ESTIMATED MONTHLY EQUIVALENT</div>
                <div class="metric-value">₹ {monthly_approx:,.0f}</div>
                <div class="metric-sub">EMI / Monthly auto-debit plan</div>
            </div>
            """, unsafe_allow_html=True)

        # Risk Factor Highlighting
        risk_flags = []
        if transplants: risk_flags.append("Major Organ Transplant History")
        if chronic: risk_flags.append("Pre-existing Chronic Conditions")
        if surgeries > 1: risk_flags.append(f"{surgeries} Previous Major Surgeries")
        if cancer_history: risk_flags.append("Genetic Cancer Risk")
        
        if risk_flags:
            st.warning(f"**Key Cost Drivers Identified:** {', '.join(risk_flags)}.")
        else:
            st.success("✅ **Standard Risk Profile:** Baseline rates applied with minimal hazard loading.")

with tab2:
    st.subheader("Model & Mathematical Architecture")
    st.markdown("""
    The estimate is produced through a **Deep Feedforward Neural Network (Multi-Layer Perceptron)**:
    1. **Input Normalization:** Vitals and binary clinical flags are standardized using $z$-score transformations.
    2. **Hidden Feature Representations:** 2 dense hidden layers ($64 \\rightarrow 32$ neurons) model high-order risk interactions (e.g., how Age compounds with Chronic Illness).
    3. **Activation Function:** Non-linear ReLU (Rectified Linear Unit) activations:
    """)
    st.latex(r"\text{ReLU}(x) = \max(0, x)")
    st.markdown("4. **Output Unit:** Continuous regression layer computing expected risk-adjusted cost.")