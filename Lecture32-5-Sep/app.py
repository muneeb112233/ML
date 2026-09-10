import streamlit as st
# ── 1. Page setup ──────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
st.title("🩺 Diabetes Prediction")
st.write("Fill in the details and click **Predict**.")
# ── 2. Load the model ──────────────────────────────────────
import pickle
from pathlib import Path
import pandas as pd
@st.cache_resource
def load_model():
    with open(Path(__file__).parent / "diabetes_model.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()
# ── 3. Collect user input ─────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    pregnancies      = st.number_input("Pregnancies",              0,    20,  1)
    glucose          = st.number_input("Glucose (mg/dL)",          0,   300, 120)
    blood_pressure   = st.number_input("Blood Pressure (mm Hg)",   0,   200,  70)
    skin_thickness   = st.number_input("Skin Thickness (mm)",      0,   110,  20)
with col2:
    insulin          = st.number_input("Insulin (mu U/ml)",        0,   900,  80)
    bmi              = st.number_input("BMI",                      0.0, 70.0, 25.0, step=0.1)
    dpf              = st.number_input("Diabetes Pedigree Func.", 0.0,  2.5,  0.4, step=0.001, format="%.3f")
    age              = st.number_input("Age",                       1,   120,  33)
# ── 4. Predict ──────────────────────────────────────────
if st.button("Predict", type="primary"):
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    )
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **Diabetes likely** — risk score: {probability:.0%}")
    else:
        st.success(f"✅ **No diabetes detected** — risk score: {probability:.0%}")
