import streamlit as st
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
import json

# st.set_page_config must be the FIRST Streamlit command in the script.
# It sets the browser tab title, favicon, and overall layout (wide = full-width).
st.set_page_config(
    page_title="YOLO Object Detection Hub",
    page_icon="🎯",
    layout="wide"
)

# st.title renders the large H1 header at the top of the main panel
st.title("🎯 YOLO Object Detection App")

# st.markdown lets you write formatted descriptive text using Markdown syntax
st.markdown("""
This app uses the **Ultralytics YOLO** model to detect objects in uploaded images.
Adjust the parameters on the sidebar, upload your image, and view results instantly!
""")


# --- Sidebar: Model Configuration ------------------------------------------

st.sidebar.header("⚙️ Model Configuration")

# Pick a use case first — this determines which model list shows below
task = st.sidebar.selectbox(
    "Use Case",
    ["Detection", "Segmentation", "Pose"],
    help="Detection = boxes, Segmentation = masks, Pose = keypoints"
)

# Curated set per use case — includes YOLO26 n/m/l/x for Detection
MODELS = {
    "Detection": [
        "yolo26n.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",   # newest
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",                  # previous gen
        "yolov8n.pt",                                              # legacy fallback
    ],
    "Segmentation": [
        "yolo26n-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt",
        "yolo11n-seg.pt", "yolo11s-seg.pt",
        "yolov8n-seg.pt",
    ],
    "Pose": [
        "yolo26n-pose.pt", "yolo26m-pose.pt", "yolo26l-pose.pt",
        "yolo11n-pose.pt", "yolo11s-pose.pt",
        "yolov8n-pose.pt",
    ],
}

# This selectbox depends on `task` — when task changes, the model list
# updates automatically because MODELS[task] resolves to a different list.
model_name = st.sidebar.selectbox(
    "YOLO Model",
    MODELS[task],
    help="YOLO26 = newest. n = fastest/Nano, m = balanced/Medium, "
         "l = accurate/Large, x = most accurate/Xtreme (GPU recommended)."
)

st.sidebar.caption(f"Selected: `{model_name}`")


# --- Sidebar: Inference Parameters ------------------------------------------

# Confidence threshold — only detections with score above this value are kept
conf_threshold = st.sidebar.slider(
    label="Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
    help="Only detections with confidence above this value will be shown"
)

# Image size — discrete steps only (must match YOLO's accepted imgsz values)
img_size = st.sidebar.select_slider(
    label="Inference Image Size (imgsz)",
    options=[320, 416, 512, 640, 800, 1024],
    value=640,
    help="Image is resized to this square before inference. Larger = more accurate but slower."
)


# --- Cached Functions -------------------------------------------------------

@st.cache_resource
def load_model(model_name: str):
    """Load a YOLO model and cache the model OBJECT itself.

    Why @st.cache_resource (not @st.cache_data)?
      - YOLO model objects hold tensors and internal state that are NOT picklable.
      - @st.cache_resource keeps the object in memory as-is (no pickling).
      - First call downloads the weights automatically; later calls return
        the same cached instance.
    """
    return YOLO(model_name)


@st.cache_data
def run_inference(image_bytes: bytes, model_name: str, img_size: int, conf_threshold: float):
    """Run YOLO inference and return PICKLABLE results (numpy + list of dicts).

    Why we don't return the raw `Results` object:
      - @st.cache_data pickles the return value to disk/memory.
      - Ultralytics `Results` contains tensors + model references that don't
        pickle reliably — caching it can crash or return stale data.
      - So we extract everything we need HERE: the annotated image (as a plain
        numpy array) and the detections (as plain dicts). Both pickle cleanly.

    Cache key = (image_bytes, model_name, img_size, conf_threshold).
    Dragging a slider changes one of these → cache miss → re-run. ✅
    Re-uploading the same image with same params → cache hit → instant. ✅
    """
    # Load the (cached) model — no reload if weights already in memory
    model = load_model(model_name)

    # Decode raw bytes into an OpenCV BGR numpy array
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
    
    # Run YOLO prediction. save=False means we don't write files to disk.
    results_list = list(model.predict(source=img, imgsz=img_size, conf=conf_threshold, save=False))
    res = results_list[0]

    # res.plot() draws boxes/masks/keypoints onto a BGR copy of the image.
    # Streamlit's st.image() assumes RGB, so we MUST convert before returning.
    # (This is the bug we fixed earlier — without this conversion, red and
    # blue channels are swapped in the rendered output.)
    annotated_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)

    # Extract structured detections as plain dicts.
    # res.boxes is populated for Detection, Segmentation, AND Pose tasks
    # (seg/pose models also produce bounding boxes alongside masks/keypoints).
    detections = []
    if res.boxes is not None:
        for box in res.boxes:
            xyxy = box.xyxy.cpu().numpy()[0]              # [x1, y1, x2, y2]
            cls_idx = int(box.cls.cpu().numpy()[0])       # class index
            conf = float(box.conf.cpu().numpy()[0])       # confidence score
            detections.append({
                "label": model.names.get(cls_idx, str(cls_idx)),
                "class_id": cls_idx,
                "confidence": round(conf, 4),
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
            })

    return annotated_rgb, detections


# --- Main Panel: Upload + Inference -----------------------------------------

# File uploader accepts JPG, JPEG, PNG
uploaded_file = st.file_uploader(
    label="Upload an Image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image file (JPG, JPEG, PNG) for object detection"
)

if uploaded_file is not None:
    # Grab raw bytes ONCE — reused for both display and inference.
    # (Avoids decoding the same image twice, which the earlier version did.)
    image_bytes = uploaded_file.getvalue()

    # Decode bytes → OpenCV BGR → RGB for display
    opencv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), 1)
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # Two-column layout: original on left, predictions on right
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(rgb_image, use_container_width=True)

    with st.spinner("Running detection..."):
        # Cached call — instant on cache hit, runs inference on cache miss
        annotated_rgb, detections = run_inference(
            image_bytes, model_name, img_size, conf_threshold
        )

    with col2:
        st.subheader("Processed Image")
        st.image(annotated_rgb, use_container_width=True)

    # For Segmentation / Pose tasks, the boxes table alone doesn't tell the
    # full story — masks and keypoints are rendered on the image above.
    # These captions make that clear to the user.
    if task == "Segmentation":
        st.caption("🎭 Segmentation masks are rendered directly on the processed image above.")
    elif task == "Pose":
        st.caption("🦴 Pose keypoints are rendered directly on the processed image above.")

    # --- Results table + JSON download ---
    st.subheader("📊 Detection Results")

    if len(detections) > 0:
        # Interactive table — sortable, scrollable
        st.dataframe(pd.DataFrame(detections), use_container_width=True)

        # Serialize detections to a pretty-printed JSON string for download
        json_string = json.dumps(detections, indent=2)

        st.download_button(
            label="💾 Download Detections as JSON",
            data=json_string,
            file_name="detections.json",
            mime="application/json"
        )
    else:
        st.info("No targets detected above the selected confidence threshold.")

else:
    # No image uploaded yet — show a friendly prompt instead of an empty screen
    st.info("Please upload an image to begin object detection.")