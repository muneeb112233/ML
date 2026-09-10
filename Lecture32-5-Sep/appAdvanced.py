import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ============================== PAGE CONFIG ==============================
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================== CUSTOM CSS ==============================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Poppins', sans-serif;
        }

        /* Soft medical gradient background */
        .stApp {
            background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 50%, #e8dff5 100%);
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f2027 0%, #203a43 55%, #2c5364 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #f2f7ff !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.25) !important;
        }

        /* ---------- Header ---------- */
        .main-title {
            text-align: center;
            font-size: 2.9rem;
            font-weight: 800;
            line-height: 1.2;
            background: linear-gradient(90deg, #4b6cb7 0%, #7f4b9b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #4a5568;
            font-size: 1.05rem;
            font-weight: 400;
            margin-bottom: 1.8rem;
        }

        /* ---------- Input card (form) ---------- */
        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(10px);
            border-radius: 22px;
            border: 1px solid rgba(255, 255, 255, 0.6);
            box-shadow: 0 10px 34px rgba(31, 38, 135, 0.16);
            padding: 1.8rem 2rem 1.4rem 2rem;
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.9rem;
        }

        /* ---------- Number inputs ---------- */
        div[data-testid="stNumberInput"] label p {
            font-weight: 600;
            color: #34495e;
            font-size: 0.92rem;
        }
        div[data-testid="stNumberInput"] input {
            border-radius: 12px;
            border: 1.5px solid #d5dce8;
            font-weight: 500;
            color: #2c3e50;
            transition: all 0.2s ease;
        }
        div[data-testid="stNumberInput"]:focus-within input {
            border-color: #4b6cb7 !important;
            box-shadow: 0 0 0 3px rgba(75, 108, 183, 0.18);
        }

        /* ---------- Buttons ---------- */
        div[data-testid="stForm"] button,
        div[data-testid="stForm"] button[kind="primaryFormSubmit"],
        .stButton > button {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            padding: 0.65rem 0 !important;
            box-shadow: 0 4px 14px rgba(24, 40, 72, 0.35);
            transition: transform 0.18s ease, box-shadow 0.18s ease !important;
        }
        div[data-testid="stForm"] button:hover,
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 22px rgba(24, 40, 72, 0.45) !important;
        }

        /* ---------- Result cards ---------- */
        .result-card {
            border-radius: 22px;
            padding: 2.4rem 2rem;
            text-align: center;
            animation: fadeUp 0.5s ease both;
            box-shadow: 0 12px 36px rgba(31, 38, 135, 0.18);
        }
        .result-danger {
            background: linear-gradient(135deg, #ffe3e3 0%, #ffcaca 60%, #ffd8e4 100%);
            border: 2px solid #e74c3c;
        }
        .result-safe {
            background: linear-gradient(135deg, #d4fcdf 0%, #c2f0e8 60%, #d7ecfb 100%);
            border: 2px solid #27ae60;
        }
        .result-icon { font-size: 4.2rem; line-height: 1; margin-bottom: 0.6rem; }
        .result-title {
            font-size: 1.75rem;
            font-weight: 800;
            margin: 0.2rem 0 0.4rem 0;
            color: #1a2a3a;
        }
        .result-sub { color: #40525f; font-size: 1rem; margin: 0; }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ---------- Progress bar ---------- */
        div[data-testid="stProgress"] > div {
            border-radius: 10px;
        }

        /* ---------- Metrics ---------- */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.7);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: 0 6px 18px rgba(31, 38, 135, 0.10);
        }

        /* Hide Streamlit chrome */
        #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================== MODEL LOADING ==============================
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


# Model path anchored to this file's folder — works no matter where the app runs from
MODEL_PATH = Path(__file__).parent / "diabetes_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Could not load `diabetes_model.pkl` — {e}")
    st.stop()

# ============================== SIDEBAR ==============================
with st.sidebar:
    st.markdown("## 🩺 Diabetes Risk Predictor")
    st.markdown("---")
    st.markdown("### 📖 About this app")
    st.markdown(
        "A machine-learning web app that estimates diabetes risk from "
        "**8 clinical parameters** using a **Logistic Regression** model "
        "trained on the Pima Indians Diabetes dataset (768 records)."
    )
    st.markdown("### 🤖 Model details")
    st.markdown(
        """
        - **Algorithm:** Logistic Regression
        - **Input features:** 8
        - **Output:** Prediction + risk probability
        """
    )
    st.markdown("### 📌 How to use")
    st.markdown(
        """
        1. Fill in the patient's details
        2. Click **Predict Risk**
        3. View the prediction & risk score
        """
    )
    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** Educational ML demo — not a medical diagnosis.", icon="⚠️")

# ============================== HEADER ==============================
st.markdown('<h1 class="main-title">🩺 Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Enter the patient\'s clinical details below to assess diabetes risk instantly</p>',
    unsafe_allow_html=True,
)

# ============================== INPUT FORM ==============================
with st.form("prediction_form"):
    st.markdown('<div class="section-title">🧾 Patient Details</div>', unsafe_allow_html=True)

    row1, row2 = st.columns(2)

    with row1:
        pregnancies = st.number_input(
            "🤰 Pregnancies", min_value=0, max_value=20, value=1, step=1,
            help="Number of times pregnant", key="pregnancies",
        )
        glucose = st.number_input(
            "🩸 Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1,
            help="Plasma glucose concentration (2h oral glucose tolerance test)", key="glucose",
        )
        blood_pressure = st.number_input(
            "❤️ Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1,
            help="Diastolic blood pressure", key="blood_pressure",
        )
        skin_thickness = st.number_input(
            "📏 Skin Thickness (mm)", min_value=0, max_value=110, value=20, step=1,
            help="Triceps skin fold thickness", key="skin_thickness",
        )

    with row2:
        insulin = st.number_input(
            "💉 Insulin (mu U/ml)", min_value=0, max_value=900, value=80, step=1,
            help="2-hour serum insulin", key="insulin",
        )
        bmi = st.number_input(
            "⚖️ BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
            help="Body Mass Index = weight(kg) / height(m)²", key="bmi",
        )
        dpf = st.number_input(
            "🧬 Diabetes Pedigree Function", min_value=0.000, max_value=2.500, value=0.400,
            step=0.001, format="%.3f",
            help="Genetic likelihood based on family history", key="dpf",
        )
        age = st.number_input(
            "🎂 Age (years)", min_value=1, max_value=120, value=33, step=1,
            help="Patient's age", key="age",
        )

    st.markdown("")
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        submitted = st.form_submit_button("🔍  Predict Risk", width="stretch")

# Quick-fill sample (same values used in the training notebook)
with st.expander("🧪 Try a sample patient (from the training notebook)"):
    st.markdown(
        "Sample input: `Pregnancies=0, Glucose=100, BloodPressure=115, SkinThickness=35, "
        "Insulin=0, BMI=29.1, DPF=0.221, Age=30` → model prediction: **No Diabetes**"
    )
    if st.button("↩️ Load sample values"):
        st.session_state.update(
            pregnancies=0, glucose=100, blood_pressure=115, skin_thickness=35,
            insulin=0, bmi=29.1, dpf=0.221, age=30,
        )
        st.rerun()

# ============================== PREDICTION ==============================
if submitted:
    # Column order must exactly match training features
    input_df = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=FEATURES,
    )

    prediction = int(model.predict(input_df)[0])
    prob_diabetes = float(model.predict_proba(input_df)[0][1])  # P(Outcome == 1)

    # ---- Result banner ----
    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-card result-danger">
                <div class="result-icon">⚠️</div>
                <div class="result-title">High Risk — Diabetes Likely</div>
                <p class="result-sub">The model predicts this patient is <b>likely to have diabetes</b>.
                A medical consultation is recommended.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-safe">
                <div class="result-icon">✅</div>
                <div class="result-title">Low Risk — No Diabetes Detected</div>
                <p class="result-sub">The model predicts this patient is <b>unlikely to have diabetes</b>.
                Maintain a healthy lifestyle!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Risk gauge ----
    st.markdown("")
    st.markdown('<div class="section-title">📈 Estimated Risk Score</div>', unsafe_allow_html=True)
    st.progress(
        min(max(prob_diabetes, 0.0), 1.0),
        text=f"**{prob_diabetes * 100:.1f}%** probability of diabetes",
    )

    # ---- Summary metrics ----
    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "🧾 Prediction",
        "Diabetic" if prediction == 1 else "Non-Diabetic",
    )
    m2.metric("⚠️ Risk Score", f"{prob_diabetes * 100:.1f}%")
    m3.metric("🩸 Glucose Level", f"{int(glucose)} mg/dL")
    m4.metric("⚖️ BMI", f"{bmi:.1f}")

    with st.expander("🔍 See the exact input sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), width="stretch")

st.markdown("<br>", unsafe_allow_html=True)
st.caption("🩺 Built with Streamlit · Logistic Regression · Educational demo only — not medical advice.")