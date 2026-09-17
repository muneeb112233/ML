import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import hog

# ---------- CONFIG ----------
MODEL_PATH = "vehicle_model.pkl"
IMG_SIZE = 128

# ---------- CACHED LOAD ----------
@st.cache_data(show_spinner="Loading vehicle classifier...")
def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # data has: decision_tree, random_forest, best_model, classes, img_size
    return data

# ---------- FEATURE EXTRACTION (matches notebook) ----------
def extract_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv = cv2.resize(hsv, (IMG_SIZE, IMG_SIZE))
    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    feat = np.concatenate([hog_feat,
                           h_hist / max(h_hist.sum(), 1e-9),
                           s_hist / max(s_hist.sum(), 1e-9)]).astype(np.float32)
    return feat.reshape(1, -1)

# ---------- APP ----------
st.set_page_config(page_title="Vehicle Classifier", page_icon="🚗", layout="wide")
st.title("🚗 Vehicle Image Classifier")
st.caption("Compare Decision Tree vs Random Forest using your saved vehicle_model.pkl — upload any vehicle photo.")

try:
    model_data = load_model()
except Exception as e:
    st.error(f"Failed to load {MODEL_PATH}: {e}")
    st.stop()

# Unpack
classes = model_data.get("classes", [])
dt_model = model_data.get("decision_tree")
rf_model = model_data.get("random_forest")
best_model = model_data.get("best_model")
if best_model is None:
    best_model = dt_model  # graceful fallback

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("- Uses **HOG + HSV** features (same pipeline as notebook)\n- Model file: `vehicle_model.pkl`\n- Deploy: push to GitHub → Streamlit Community Cloud")
    st.markdown("**Classes:**")
    for c in classes:
        st.write(f"- {c}")

# Upload
uploaded = st.file_uploader("Upload a vehicle image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    img_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read image.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded", use_container_width=True)
        with col2:
            feat = extract_features(img)
            # Predictions
            dt_pred = dt_model.predict(feat)[0] if dt_model else None
            rf_pred = rf_model.predict(feat)[0] if rf_model else None
            best_pred = best_model.predict(feat)[0] if best_model else None

            # Show results
            st.subheader("Predictions")
            if dt_pred is not None:
                st.metric("DecisionTree", classes[int(dt_pred)])
            if rf_pred is not None:
                st.metric("RandomForest", classes[int(rf_pred)])
            if best_pred is not None:
                st.success(f"Best model prediction: **{classes[int(best_pred)]}**")

            # Confidence-like display (class probability if available; else just label)
            if hasattr(best_model, "predict_proba"):
                proba = best_model.predict_proba(feat)[0]
                top_idx = int(np.argmax(proba))
                st.progress(float(proba[top_idx]), text=f"Best confidence: {proba[top_idx]:.2%}")

