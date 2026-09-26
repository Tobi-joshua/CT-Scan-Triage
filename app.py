from __future__ import annotations
from pathlib import Path
import io
import json
import time

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.model import build_model, mc_predict
from src.gradcam import GradCAM, overlay_cam
from src.data import IMAGENET_MEAN, IMAGENET_STD

st.set_page_config(page_title="Chest Imaging Triage", page_icon="🫁", layout="wide")
st.title("Chest Imaging Triage — Research Prototype")
st.warning(
    "Research prototype only. Not a medical device and not validated for diagnosis "
    "or clinical decision-making."
)

MODEL_PATH = Path("artifacts/cxr_mobilenetv3.pt")
METRICS_PATH = Path("artifacts/test_metrics.json")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = build_model(
        len(ckpt["classes"]),
        dropout=float(ckpt.get("dropout", 0.30)),
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt

model, ckpt = load_model()

with st.sidebar:
    st.header("Model status")
    if model is None:
        st.error("No trained CXR checkpoint found.")
    else:
        st.success("Trained CXR baseline loaded.")
        st.caption(f"Dataset: {ckpt.get('dataset', 'unknown')}")
        st.caption(f"Input size: {ckpt.get('input_size', 224)} px")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        st.metric("Held-out test AUC", f"{metrics.get('auc', float('nan')):.3f}")
        st.metric("Sensitivity", f"{metrics.get('recall_sensitivity', float('nan')):.3f}")
        st.metric("Specificity", f"{metrics.get('specificity', float('nan')):.3f}")

modality = st.radio("Modality", ["Chest X-ray", "CT (preview only)"], horizontal=True)
file = st.file_uploader("Upload a de-identified PNG/JPEG image", type=["png", "jpg", "jpeg"])

if modality.startswith("CT"):
    st.info(
        "CT is intentionally a separate model track. The current checkpoint is a chest-X-ray "
        "pneumonia baseline and will not be applied to CT."
    )

if file:
    image = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
    left, right = st.columns(2)
    left.image(image, caption="Input", use_container_width=True)

    if model is None:
        right.info(
            "Training has not produced artifacts/cxr_mobilenetv3.pt yet. "
            "The app will not fabricate a medical prediction."
        )
    elif modality.startswith("Chest"):
        size = int(ckpt.get("input_size", 224))
        tfm = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        x = tfm(image).unsqueeze(0)

        t0 = time.perf_counter()
        mean, var, entropy, mi = mc_predict(model, x, passes=20)
        latency_ms = (time.perf_counter() - t0) * 1000

        idx = int(mean.argmax(1).item())
        classes = ckpt["classes"]
        cam_engine = GradCAM(model, model.features[-1])
        cam, _ = cam_engine(x, idx)
        cam_engine.close()
        overlay = overlay_cam(image, cam)
        right.image(overlay, caption="Grad-CAM explanation", use_container_width=True)

        st.subheader("Research output")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted class", classes[idx])
        c2.metric("Mean probability", f"{mean[0, idx].item():.3f}")
        c3.metric("Predictive entropy", f"{entropy.item():.3f}")
        c4.metric("20-pass latency", f"{latency_ms:.0f} ms")
        st.caption(
            f"Epistemic uncertainty (mutual information): {mi.item():.4f}. "
            "Uncertainty values are descriptive research outputs, not clinical confidence guarantees."
        )

st.divider()
st.caption(
    "Current scope: pediatric chest-X-ray pneumonia baseline. "
    "External validation, adult emergency-imaging data, multi-condition triage, calibration refinement, "
    "and CT-specific modeling remain required before the broader preprint concept is validated."
)
