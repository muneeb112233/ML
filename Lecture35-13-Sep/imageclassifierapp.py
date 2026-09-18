import streamlit as st
import cv2
import numpy as np
import pickle
from skimage.feature import hog

# ---------- CONFIG ----------

MODEL_PATH = "Lecture35-13-Sep/vehicle_model.pkl"
# MODEL_PATH = "vehicle_model.pkl"
IMG_SIZE = 128


# ---------- CACHED LOAD ----------

@st.cache_data(show_spinner="Loading vehicle classifier...")
def load_model(path=MODEL_PATH):

    with open(path, "rb") as f:
        data = pickle.load(f)

    # data has:
    # decision_tree, random_forest, best_model, classes, img_size

    return data


# ---------- FEATURE EXTRACTION (matches notebook) ----------

def extract_features(img_bgr):

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    hsv = cv2.resize(hsv, (IMG_SIZE, IMG_SIZE))

    h_hist = cv2.calcHist(
        [hsv],
        [0],
        None,
        [32],
        [0, 180]
    ).flatten()

    s_hist = cv2.calcHist(
        [hsv],
        [1],
        None,
        [32],
        [0, 256]
    ).flatten()

    feat = np.concatenate([
        hog_feat,
        h_hist / max(h_hist.sum(), 1e-9),
        s_hist / max(s_hist.sum(), 1e-9)
    ]).astype(np.float32)

    return feat.reshape(1, -1)


# ---------- APP ----------

st.set_page_config(
    page_title="Vehicle Classifier",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Vehicle Image Classifier")

st.caption(
    "Compare Decision Tree vs Random Forest using your saved "
    "vehicle_model.pkl — upload one or multiple vehicle photos."
)


# ---------- LOAD MODEL ----------

try:

    model_data = load_model()

except Exception as e:

    st.error(f"Failed to load {MODEL_PATH}: {e}")
    st.stop()


# ---------- UNPACK MODELS ----------

classes = model_data.get("classes", [])

dt_model = model_data.get("decision_tree")

rf_model = model_data.get("random_forest")

best_model = model_data.get("best_model")


# Graceful fallback
if best_model is None:
    best_model = dt_model


# ---------- SIDEBAR ----------

with st.sidebar:

    st.header("About")

    st.markdown(
        "- Uses **HOG (Histogram of Oriented Gradients) + "
        "HSV (Hue, Saturation) features**\n"
        "- Model file: `vehicle_model.pkl`\n"
    )

    st.markdown("**Classes:**")

    for c in classes:
        st.write(f"- {c}")


# ---------- MULTIPLE IMAGE UPLOAD ----------

uploaded_files = st.file_uploader(
    "Upload vehicle images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# ---------- PROCESS IMAGES ----------

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} image(s) uploaded successfully."
    )

    for uploaded_file in uploaded_files:

        st.markdown("---")

        st.subheader(uploaded_file.name)

        # ---------- READ IMAGE ----------

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        img = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if img is None:

            st.error(
                f"Could not read {uploaded_file.name}"
            )

            continue


        # ---------- SHOW IMAGE ----------

        st.image(
            cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            ),
            caption=uploaded_file.name,
            width=400
        )


        # ---------- FEATURE EXTRACTION ----------

        feat = extract_features(img)


        # ---------- PREDICTIONS ----------

        dt_pred = dt_model.predict(feat)[0]

        rf_pred = rf_model.predict(feat)[0]

        best_pred = best_model.predict(feat)[0]


        # ---------- CONVERT LABELS ----------

        dt_class = classes[int(dt_pred)]

        rf_class = classes[int(rf_pred)]

        best_class = classes[int(best_pred)]


        # ---------- PROBABILITIES ----------

        dt_prob = (
            dt_model.predict_proba(feat)[0].max()
            if hasattr(dt_model, "predict_proba")
            else None
        )

        rf_prob = (
            rf_model.predict_proba(feat)[0].max()
            if hasattr(rf_model, "predict_proba")
            else None
        )


        # ---------- RESULTS ----------

        col1, col2, col3 = st.columns(3)


        # Decision Tree

        with col1:

            st.metric(
                "Decision Tree",
                dt_class
            )

            if dt_prob is not None:

                st.progress(float(dt_prob))

                st.caption(
                    f"Probability: {dt_prob:.1%}"
                )


        # Random Forest

        with col2:

            st.metric(
                "Random Forest",
                rf_class
            )

            if rf_prob is not None:

                st.progress(float(rf_prob))

                st.caption(
                    f"Probability: {rf_prob:.1%}"
                )


        # Best Model

        with col3:

            st.metric(
                "Best Model",
                best_class
            )

            if hasattr(best_model, "predict_proba"):

                proba = best_model.predict_proba(feat)[0]

                top_idx = int(np.argmax(proba))

                st.progress(
                    float(proba[top_idx])
                )

                st.caption(
                    f"Best confidence: {proba[top_idx]:.2%}"
                )

