# app.py
# Streamlit UI for pre-trained Iris ANN models
# Run:  streamlit run app.py

import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# ===================== CONFIG: point to your files =====================
MODEL_PATHS = {
    "Scaled (with scaler.pkl)":   "stable_iris_model.keras",       
    "Unscaled (raw features)":    "iris_model_unscaled.pkl",      
}
SCALER_PATH = "scaler.pkl"
# =======================================================================

st.set_page_config(page_title="Iris ANN Classifier", page_icon="🌸", layout="centered")

# ----------------------------- Styling -----------------------------
st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
        .title-text {
            background: linear-gradient(90deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        .result-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(96, 165, 250, 0.3);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .species-name { font-size: 28px; font-weight: 700; }
        .prob-bar {
            height: 28px; border-radius: 8px;
            display: flex; align-items: center; padding-left: 12px;
            color: white; font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Load artifacts -----------------------------
@st.cache_resource
def load_scaler(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    if path.endswith('.pkl'):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return tf.keras.models.load_model(path)

scaler = load_scaler(SCALER_PATH)
models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        try:
            models[name] = load_model(path)
        except Exception as e:
            st.error(f"Could not load `{path}` ({name}): {e}")

class_names = np.array(["setosa", "versicolor", "virginica"])

# ----------------------------- Header -----------------------------
st.markdown('<h1 class="title-text">🌸 Iris ANN Classifier</h1>', unsafe_allow_html=True)
st.markdown("A neural network trained on the Iris dataset. Pick a model, adjust the "
            "sliders or pick a sample flower, and see the prediction with confidence.")

# ----------------------------- Model selector -----------------------------
if not models:
    st.error("No models found. Check MODEL_PATHS at the top of app.py.")
    st.stop()

model_choice = st.radio(
    "🧠 Choose model",
    list(models.keys()),
    horizontal=True,
)
use_scaler = "Scaled" in model_choice
model = models[model_choice]

if use_scaler and scaler is None:
    st.warning("You selected the scaled model but `scaler.pkl` wasn't found.")

# ----------------------------- Sample table -----------------------------
sample_records = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4},
    {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5},
    {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
    {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 5.1, "petal_width": 1.8},
    {"sepal_length": 5.5, "sepal_width": 2.4, "petal_length": 3.8, "petal_width": 1.1},
    {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 5.6, "petal_width": 2.4},
    {"sepal_length": 5.0, "sepal_width": 3.4, "petal_length": 1.5, "petal_width": 0.2},
    {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 4.9, "petal_width": 1.5},
]
sample_df = pd.DataFrame(sample_records)

# ----------------------------- Layout -----------------------------
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader("🌱 Try a sample flower")
    st.caption("Click a record to load it into the sliders.")

    defaults = sample_records[0].copy()
    for idx, row in sample_df.iterrows():
        c0, c1, c2 = st.columns([0.4, 4, 1.3])
        with c0:
            st.markdown(f"**{idx+1}**")
        with c1:
            st.markdown(
                f"SL `{row.sepal_length}` · SW `{row.sepal_width}` · "
                f"PL `{row.petal_length}` · PW `{row.petal_width}`"
            )
        with c2:
            if st.button("Try this", key=f"sample_{idx}"):
                defaults = row.to_dict()
                st.rerun()

with col_right:
    st.subheader("🎚️ Flower measurements")
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, float(defaults["sepal_length"]), 0.1)
    sepal_width  = st.slider("Sepal width (cm)",  2.0, 4.5, float(defaults["sepal_width"]),  0.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, float(defaults["petal_length"]), 0.1)
    petal_width  = st.slider("Petal width (cm)",  0.1, 2.5, float(defaults["petal_width"]),  0.1)

# ----------------------------- Prediction -----------------------------
input_vec = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=float)

if use_scaler and scaler is not None:
    X_in = scaler.transform(input_vec)
else:
    X_in = input_vec

probs = model.predict(X_in, verbose=0)[0]
pred_idx = int(np.argmax(probs))
pred_class = class_names[pred_idx]
confidence = probs[pred_idx]

species_colors = {
    "setosa":     "#34d399",
    "versicolor": "#60a5fa",
    "virginica":  "#f472b6",
}
pred_color = species_colors[pred_class]

# ----------------------------- Result Card -----------------------------
st.markdown("### 🎯 Prediction")
st.markdown(
    f"""
    <div class="result-card">
        <div style="display:flex; align-items:center; gap:16px;">
            <div style="font-size:48px;">🌸</div>
            <div>
                <div style="color:#94a3b8; font-size:13px; letter-spacing:1px;">PREDICTED SPECIES</div>
                <div class="species-name" style="color:{pred_color};">{pred_class.capitalize()}</div>
            </div>
        </div>
        <div style="margin-top:14px; color:#cbd5e1; font-size:15px;">
            Confidence: <b style="color:{pred_color};">{confidence*100:.2f}%</b>
            &nbsp;·&nbsp; <span style="color:#94a3b8; font-size:13px;">{model_choice}</span>
        </div>
        <div style="margin-top:10px; background:rgba(15,23,42,0.6); border-radius:10px; overflow:hidden;">
            <div style="height:14px; width:{confidence*100:.2f}%;
                        background:{pred_color}; transition:width 0.5s;"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Probability Bars -----------------------------
st.markdown("### 📊 Class probabilities")
prob_df = pd.DataFrame({
    "Species": [c.capitalize() for c in class_names],
    "Probability": probs,
    "Color": [species_colors[c] for c in class_names],
}).sort_values("Probability", ascending=False)

for _, r in prob_df.iterrows():
    pct = r["Probability"] * 100
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <div style="width:90px; color:#e2e8f0; font-weight:600;">{r['Species']}</div>
            <div style="flex:1; background:rgba(15,23,42,0.6); border-radius:8px; overflow:hidden;">
                <div class="prob-bar" style="width:{pct:.2f}%; background:{r['Color']};">
                    {pct:.2f}%
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    f"Model: **{model_choice}** · Scaler: "
    f"{'✅ applied' if (use_scaler and scaler is not None) else '⛔ raw features'}"
)