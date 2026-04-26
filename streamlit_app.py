from pathlib import Path
from time import perf_counter
from datetime import datetime
import re

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "runs"
STREAMLIT_RESULTS_ROOT = ROOT / "outputs" / "streamlit_results"
MODEL_SPECS = [
    {
        "name": "yolo11n",
        "weights": RUNS_ROOT / "drone_object_detection_yolo11n_img640_e50" / "weights" / "best.pt",
    },
    {
        "name": "yolo11s",
        "weights": RUNS_ROOT / "drone_object_detection_yolo11s_img640_e50" / "weights" / "best.pt",
    },
    {
        "name": "yolov8n",
        "weights": RUNS_ROOT / "drone_object_detection_yolov8n_img640_e50" / "weights" / "best.pt",
    },
    {
        "name": "yolov8s",
        "weights": RUNS_ROOT / "drone_object_detection_yolov8s_img640_e50" / "weights" / "best.pt",
    },
]


def available_model_specs():
    return [spec for spec in MODEL_SPECS if spec["weights"].exists()]


def make_result_dir(uploaded_filename):
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(uploaded_filename).stem).strip("._") or "uploaded_image"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = STREAMLIT_RESULTS_ROOT / f"{safe_stem}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir


def save_annotated_image(image_rgb, output_path):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)


def decode_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return image_bgr


def infer_with_model(weights_path, image_bgr, conf_threshold, image_size, device):
    model = YOLO(str(weights_path))
    start_time = perf_counter()
    results = model.predict(
        source=image_bgr,
        conf=conf_threshold,
        imgsz=image_size,
        device=device,
        verbose=False,
    )
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()
    elapsed_ms = (perf_counter() - start_time) * 1000.0

    result = results[0]
    plotted_bgr = result.plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)

    detection_count = 0 if result.boxes is None else len(result.boxes)
    max_confidence = 0.0
    if detection_count:
        max_confidence = float(result.boxes.conf.max().item())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "annotated_image": plotted_rgb,
        "elapsed_ms": elapsed_ms,
        "detection_count": detection_count,
        "max_confidence": max_confidence,
    }


def main():
    st.set_page_config(page_title="Drone Detection Model Comparison", layout="wide")
    st.title("Drone Detection Model Comparison")
    st.write("Upload an image to compare the four trained YOLO models side by side.")

    specs = available_model_specs()
    if not specs:
        st.error("No trained model weights were found under the runs directory.")
        return

    missing_models = [spec["name"] for spec in MODEL_SPECS if not spec["weights"].exists()]
    if missing_models:
        st.warning(f"Missing saved weights for: {', '.join(missing_models)}")

    default_device = 0 if torch.cuda.is_available() else "cpu"
    with st.sidebar:
        st.header("Inference Settings")
        device = st.selectbox(
            "Device",
            options=[default_device, "cpu"] if torch.cuda.is_available() else ["cpu"],
            format_func=lambda value: "GPU (CUDA:0)" if value == 0 else "CPU",
        )
        conf_threshold = st.slider("Confidence Threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
        image_size = st.select_slider("Image Size", options=[416, 512, 640, 768, 960], value=512)
        st.caption(f"PyTorch: {torch.__version__}")
        st.caption(f"CUDA available: {torch.cuda.is_available()}")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file is None:
        st.info("Upload an image to run model comparison.")
        return

    try:
        image_bgr = decode_uploaded_image(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        return

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.subheader("Uploaded Image")
    st.image(original_rgb, use_container_width=True)

    if st.button("Run Comparison", type="primary"):
        comparison_rows = []
        result_columns = st.columns(2)
        result_dir = make_result_dir(uploaded_file.name)

        for index, spec in enumerate(specs):
            with st.spinner(f"Running {spec['name']}..."):
                result = infer_with_model(
                    weights_path=spec["weights"],
                    image_bgr=image_bgr,
                    conf_threshold=conf_threshold,
                    image_size=image_size,
                    device=device,
                )

            save_annotated_image(result["annotated_image"], result_dir / f"{spec['name']}.jpg")

            comparison_rows.append(
                {
                    "model_name": spec["name"],
                    "device": "cuda:0" if device == 0 else "cpu",
                    "processing_time_ms": round(result["elapsed_ms"], 2),
                    "detections": result["detection_count"],
                    "max_confidence": round(result["max_confidence"], 4),
                }
            )

            with result_columns[index % 2]:
                st.markdown(f"### {spec['name']}")
                st.image(result["annotated_image"], use_container_width=True)
                st.write(f"Processing time: {result['elapsed_ms']:.2f} ms")
                st.write(f"Detections: {result['detection_count']}")
                st.write(f"Max confidence: {result['max_confidence']:.4f}")

        summary_df = pd.DataFrame(comparison_rows).sort_values(by="processing_time_ms").reset_index(drop=True)
        st.subheader("Comparison Summary")
        st.dataframe(summary_df, use_container_width=True)
        st.success(f"Saved annotated model outputs to: {result_dir}")


if __name__ == "__main__":
    main()