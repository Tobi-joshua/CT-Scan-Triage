from __future__ import annotations

from pathlib import Path
import io
import json
import time

import numpy as np
import pydicom
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.model import build_model, mc_predict, mc_predict_multilabel
from src.ensemble import ensemble_predict
from src.gradcam import GradCAM, overlay_cam
from src.data import IMAGENET_MEAN, IMAGENET_STD

st.set_page_config(page_title="Chest Imaging Triage", page_icon="🫁", layout="wide")
st.title("Chest Imaging Triage — Research Prototype")
st.warning(
    "Research prototype only. Not a medical device and not validated for diagnosis "
    "or clinical decision-making."
)

PED_PATH = Path("artifacts/cxr_mobilenetv3.pt")
PED_METRICS = Path("artifacts/test_metrics.json")
NIH_PATH = Path("artifacts/cxr_nih_multilabel.pt")
NIH_METRICS = Path("artifacts/nih_multilabel_metrics.json")
ENSEMBLE_PATHS = [Path("artifacts/cxr_mobilenetv3.pt"), Path("artifacts/cxr_seed_2027.pt"), Path("artifacts/cxr_seed_2028.pt")]
ENSEMBLE_METRICS = Path("artifacts/ensemble_metrics.json")

@st.cache_resource
def load_checkpoint(path_string: str):
    path = Path(path_string)
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(
        len(ckpt["classes"]),
        dropout=float(ckpt.get("dropout", 0.30)),
        pretrained=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt

def uploaded_to_pil(upload, modality: str) -> Image.Image:
    name = upload.name.lower()
    raw = upload.getvalue()
    if name.endswith((".dcm", ".dicom")):
        ds = pydicom.dcmread(io.BytesIO(raw))
        arr = ds.pixel_array.astype(np.float32)
        if modality.startswith("CT"):
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            hu = arr * slope + intercept
            center, width = -600.0, 1500.0
            lo, hi = center - width / 2, center + width / 2
            arr = np.clip(hu, lo, hi)
        else:
            lo, hi = np.percentile(arr, [1, 99])
            arr = np.clip(arr, lo, hi)
        arr -= arr.min()
        arr /= max(float(arr.max()), 1e-6)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 1.0 - arr
        return Image.fromarray(np.uint8(arr * 255)).convert("RGB")
    return Image.open(io.BytesIO(raw)).convert("RGB")

def model_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

ped_model, ped_ckpt = load_checkpoint(str(PED_PATH))
nih_model, nih_ckpt = load_checkpoint(str(NIH_PATH))
ensemble_models = []
ensemble_ckpts = []
if all(p.exists() for p in ENSEMBLE_PATHS):
    for p in ENSEMBLE_PATHS:
        m, c = load_checkpoint(str(p))
        ensemble_models.append(m)
        ensemble_ckpts.append(c)

with st.sidebar:
    st.header("System status")
    if ped_model is not None:
        st.success("Pediatric pneumonia baseline loaded")
        if PED_METRICS.exists():
            m = json.loads(PED_METRICS.read_text())
            st.metric("Pediatric test AUC", f"{m['auc']:.3f}")
            st.caption(f"Sensitivity {m['recall_sensitivity']:.3f} • Specificity {m['specificity']:.3f}")
    else:
        st.error("Pediatric checkpoint missing")

    if ensemble_models:
        st.success("Three-member pediatric ensemble loaded")
        if ENSEMBLE_METRICS.exists():
            em = json.loads(ENSEMBLE_METRICS.read_text())
            st.metric("Ensemble test AUC", f"{em['auc']:.3f}")

    if nih_model is not None:
        st.success("Adult NIH multi-label baseline loaded")
        if NIH_METRICS.exists():
            n = json.loads(NIH_METRICS.read_text())
            auc = n.get("metrics", {}).get("macro_auc_available_labels")
            if auc is not None:
                st.metric("NIH subset macro AUC", f"{auc:.3f}")
    else:
        st.info("Adult NIH multi-label checkpoint not yet available")

modality = st.radio("Modality", ["Chest X-ray", "CT (preview only)"], horizontal=True)

available_models = []
if ped_model is not None:
    available_models.append("Pediatric pneumonia baseline")
if ensemble_models:
    available_models.append("Pediatric pneumonia deep ensemble")
if nih_model is not None:
    available_models.append("Adult NIH 14-finding baseline")

selected_model = None
if modality == "Chest X-ray" and available_models:
    selected_model = st.selectbox("Research model", available_models)

if modality.startswith("CT"):
    st.info(
        "CT is a separate research track. DICOM CT slices can be previewed with lung-window "
        "normalization, but no CXR model will be applied to them."
    )

upload = st.file_uploader(
    "Upload a de-identified chest image",
    type=["png", "jpg", "jpeg", "dcm", "dicom"],
    help="Do not upload identifiable patient data to this public research demo.",
)

if upload:
    try:
        image = uploaded_to_pil(upload, modality)
    except Exception as exc:
        st.error(f"Could not decode the uploaded image: {exc}")
        st.stop()

    left, right = st.columns(2)
    left.image(image, caption="De-identified input preview", use_container_width=True)

    if modality.startswith("CT"):
        right.info("CT preview only — no diagnostic or triage prediction is produced.")
    elif not available_models:
        right.error("No trained CXR checkpoint is available.")
    elif selected_model == "Pediatric pneumonia baseline":
        model, ckpt = ped_model, ped_ckpt
        size = int(ckpt.get("input_size", 160))
        x = model_transform(size)(image).unsqueeze(0)

        t0 = time.perf_counter()
        mean, var, entropy, mi = mc_predict(model, x, passes=20)
        latency_ms = (time.perf_counter() - t0) * 1000
        idx = int(mean.argmax(1).item())

        cam_engine = GradCAM(model, model.features[-1])
        cam, _ = cam_engine(x, idx)
        cam_engine.close()
        right.image(overlay_cam(image, cam), caption="Grad-CAM explanation", use_container_width=True)

        st.subheader("Research output")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted class", ckpt["classes"][idx])
        c2.metric("Mean probability", f"{mean[0, idx].item():.3f}")
        c3.metric("Predictive entropy", f"{entropy.item():.3f}")
        c4.metric("20-pass latency", f"{latency_ms:.0f} ms")
        st.caption(
            f"Epistemic uncertainty (mutual information): {mi.item():.4f}. "
            "This is a pediatric pneumonia research baseline, not an emergency triage decision."
        )

    elif selected_model == "Pediatric pneumonia deep ensemble":
        size = int(ensemble_ckpts[0].get("input_size", 160))
        x = model_transform(size)(image).unsqueeze(0)

        t0 = time.perf_counter()
        mean, var, entropy, mi = ensemble_predict(ensemble_models, x)
        latency_ms = (time.perf_counter() - t0) * 1000
        idx = int(mean.argmax(1).item())

        cam_engine = GradCAM(ensemble_models[0], ensemble_models[0].features[-1])
        cam, _ = cam_engine(x, idx)
        cam_engine.close()
        right.image(
            overlay_cam(image, cam),
            caption="Grad-CAM from ensemble member 1",
            use_container_width=True,
        )

        st.subheader("Deep ensemble research output")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted class", ensemble_ckpts[0]["classes"][idx])
        c2.metric("Ensemble mean", f"{mean[0, idx].item():.3f}")
        c3.metric("Between-model variance", f"{var[0, idx].item():.6f}")
        c4.metric("3-member latency", f"{latency_ms:.0f} ms")
        st.caption(
            f"Ensemble mutual information: {mi.item():.4f}. "
            "Member disagreement is an epistemic-uncertainty indicator, not a clinical guarantee."
        )

    elif selected_model == "Adult NIH 14-finding baseline":
        model, ckpt = nih_model, nih_ckpt
        size = int(ckpt.get("input_size", 128))
        x = model_transform(size)(image).unsqueeze(0)

        t0 = time.perf_counter()
        mean, var, entropy, mi = mc_predict_multilabel(model, x, passes=20)
        latency_ms = (time.perf_counter() - t0) * 1000
        probs = mean[0]
        order = torch.argsort(probs, descending=True).tolist()
        top = order[:5]
        target_idx = top[0]

        cam_engine = GradCAM(model, model.features[-1])
        cam, _ = cam_engine(x, target_idx)
        cam_engine.close()
        right.image(
            overlay_cam(image, cam),
            caption=f"Grad-CAM for {ckpt['classes'][target_idx]}",
            use_container_width=True,
        )

        st.subheader("Multi-label research output")
        rows = []
        for i in top:
            rows.append({
                "Finding": ckpt["classes"][i],
                "Probability": round(float(mean[0, i]), 4),
                "MC variance": round(float(var[0, i]), 6),
                "Predictive entropy": round(float(entropy[0, i]), 4),
                "Mutual information": round(float(mi[0, i]), 6),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            f"20-pass inference latency: {latency_ms:.0f} ms. Probabilities are not clinically "
            "thresholded or calibrated for patient-care decisions."
        )

st.divider()
st.markdown("### Scope")
st.caption(
    "The deployed system is a reproducible research prototype. The pediatric binary model is "
    "validated only on its held-out source split. The NIH model, when available, is trained on "
    "text-mined multi-label radiographs. External validation, prospective evaluation, domain-shift "
    "testing, clinician studies, and a dedicated CT model remain required."
)
