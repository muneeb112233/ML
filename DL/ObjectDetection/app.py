import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import json

# st.set_page_config configures the page title, icon, and layout
st.set_page_config(
    page_title="YOLO Object Detection Hub",
    page_icon="🎯",
    layout="wide"
)

# st.title displays a large header title for the application
st.title("🎯 YOLO Object Detection App")

# st.markdown allows us to write formatted descriptive text using Markdown syntax
st.markdown("""
This app uses the **Ultralytics YOLO** model to detect objects in uploaded images.
Adjust the parameters on the sidebar, upload your image, and view results instantly!
""")

# st.sidebar creates a side panel where we can group configuration inputs
st.sidebar.header("⚙️ Model Configuration")

# Pick a use case first
task = st.sidebar.selectbox(
    "Use Case",
    ["Detection", "Segmentation", "Pose"],
    help="Detection = boxes, Segmentation = masks, Pose = keypoints"
)
# Curated set per use case — includes YOLO26 n/m/l/x for Detection #Dictionary 
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

model_name = st.sidebar.selectbox(
    "YOLO Model",
    MODELS[task],
    help="YOLO26 = newest. n = fastest/Nano, m = balanced/Medium, l = accurate/Large, x = most accurate/Xtreme (GPU recommended)."
)

st.sidebar.caption(f"Selected: `{model_name}`")


# st.sidebar.text_input lets the user type in the specific YOLO model weight name , lets do selector 
# model_name = st.sidebar.text_input(
#     label="YOLO Model Name",
#     value="yolo26n.pt",
#     help="Specify any YOLOv8/YOLO11 weights file (e.g., yolo26n.pt, yolov8n.pt)"
# )

# st.sidebar.slider provides a visual slider to set the detection confidence threshold
conf_threshold = st.sidebar.slider(
    label="Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
    help="Only detections with confidence above this value will be shown"
)
# st.sidebar.select_slider lets the user select from specific predefined image size steps
img_size = st.sidebar.select_slider(
    label="Inference Image Size (imgsz)",
    options=[320, 416, 512, 640, 800, 1024],
    value=640,
    help="Image size to resize to before running inference"
)

# Cache the model load — runs once per weight, not on every rerun
@st.cache_resource
def load_model(model_name: str):
    from ultralytics import YOLO
    return YOLO(model_name)  # auto-downloads weights on first use

#Load the model using the selected model name
model = load_model(model_name)

#File Upload 
uploaded_file = st.file_uploader(
    label="Upload an Image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image file (JPG, JPEG, PNG) for object detection"
)

if uploaded_file is not None and model is not None:
    # Read the file uploaded by the user into a format compatible with OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1) # BGR

    # Convert BGR (OpenCV) to RGB for Streamlit/Matplotlib compatibility
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # st.columns splits the screen layout into side-by-side columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        # st.image renders images directly onto the web interface
        st.image(rgb_image, use_container_width=True)

    with st.spinner("Running detection..."): # st.spinner shows a loading animation while code executes
        # Run prediction on the input image
        results = model.predict(source=opencv_image, imgsz=img_size, conf=conf_threshold, save=False)
        res = results[0]

        # Render annotated detections onto a new RGB image
        annotated_rgb = res.plot()

        # Collect detections in structured format
        detections = []
        for box in res.boxes:
            xyxy = box.xyxy.cpu().numpy().tolist()[0]
            conf = float(box.conf.cpu().numpy()[0]) if hasattr(box, "conf") else float(box.conf)
            cls_idx = int(box.cls.cpu().numpy()[0]) if hasattr(box, "cls") else int(box.cls)
            label = model.names.get(cls_idx, str(cls_idx))
            detections.append({
                "label": label,
                "class_id": cls_idx,
                "confidence": round(conf, 4),
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
            })

    with col2:
        st.subheader("Processed Image")
        st.image(annotated_rgb, use_container_width=True)

    st.subheader("📊 Detection Results")

    if len(detections) > 0:
        # st.dataframe displays interactive tables (Pandas DataFrames)
        import pandas as pd
        st.dataframe(pd.DataFrame(detections), use_container_width=True)

        # Convert detections data into JSON string format for downloading
        json_string = json.dumps(detections, indent=2)

        # st.download_button allows the user to save output files locally
        st.download_button(
            label="💾 Download Detections as JSON",
            data=json_string,
            file_name="detections.json",
            mime="application/json"
        )
    else:
        # st.info is used to present useful status information to the user
        st.info("No targets detected above the selected confidence threshold.")
else:
    if model is None:
        st.warning("Waiting for a valid YOLO model to load...")
    else:
        st.info("Please upload an image to begin object detection.")