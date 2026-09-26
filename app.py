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

from src.model import build_model, mc_predict, mc_predict_multilabel, gradcam_target_layer
from src.ensemble import ensemble_predict
from src.gradcam import GradCAM, overlay_cam
from src.data import IMAGENET_MEAN, IMAGENET_STD

# ---------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Chest Imaging Triage",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fb; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f33 0%, #102a43 100%);
        color: white;
    }
    [data-testid="stSidebar"] * { color: #eef6ff; }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0b1f33 0%, #174a6e 55%, #1d6f8a 100%);
        color: white;
        box-shadow: 0 12px 32px rgba(11,31,51,.18);
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .45rem 0 0 0; color: #d9edf7; }
    .card {
        background: white;
        border: 1px solid #e6edf4;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 16px rgba(15,23,42,.05);
        margin-bottom: .75rem;
    }
    .status-ok {
        display:inline-block; padding:.22rem .55rem; border-radius:999px;
        background:#e9f9ef; color:#147a3d; font-weight:700; font-size:.78rem;
    }
    .status-warn {
        display:inline-block; padding:.22rem .55rem; border-radius:999px;
        background:#fff5df; color:#9a6200; font-weight:700; font-size:.78rem;
    }
    .metric-label { color:#62748a; font-size:.8rem; margin-bottom:.2rem; }
    .metric-value { color:#0b1f33; font-size:1.45rem; font-weight:800; }
    .muted { color:#66788a; font-size:.9rem; }
    .queue-card {
        background:white; border-radius:14px; padding:.8rem 1rem;
        border:1px solid #e4ebf2; margin:.45rem 0;
    }
    .risk-high { border-left:5px solid #c5392f; }
    .risk-med  { border-left:5px solid #d79b14; }
    .risk-low  { border-left:5px solid #2d8a57; }
    .smallcaps { font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color:#718096; }
    .footer-note { color:#718096; font-size:.82rem; text-align:center; padding:1rem 0 0.5rem; }
    div[data-testid="stMetric"] {
        background:white; border:1px solid #e6edf4; border-radius:14px; padding:.7rem .85rem;
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PED_PATH = Path("artifacts/cxr_mobilenetv3.pt")
PED_METRICS = Path("artifacts/test_metrics.json")
NIH_PATH = Path("artifacts/cxr_nih_multilabel.pt")
NIH_METRICS = Path("artifacts/nih_multilabel_metrics.json")
ENSEMBLE_PATHS = [
    Path("artifacts/cxr_mobilenetv3.pt"),
    Path("artifacts/cxr_seed_2027.pt"),
    Path("artifacts/cxr_seed_2028.pt"),
]
ENSEMBLE_METRICS = Path("artifacts/ensemble_metrics.json")


# ---------------------------------------------------------------------
# Data / model helpers
# ---------------------------------------------------------------------
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
        architecture=ckpt.get("architecture", "mobilenet_v3_small"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@st.cache_data
def load_json(path_string: str):
    p = Path(path_string)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def uploaded_to_pil(upload, modality: str) -> tuple[Image.Image, dict]:
    name = upload.name.lower()
    raw = upload.getvalue()
    meta = {"filename": upload.name, "format": "Image"}
    if name.endswith((".dcm", ".dicom")):
        ds = pydicom.dcmread(io.BytesIO(raw))
        arr = ds.pixel_array.astype(np.float32)
        meta["format"] = "DICOM"
        meta["rows"] = int(getattr(ds, "Rows", arr.shape[-2]))
        meta["columns"] = int(getattr(ds, "Columns", arr.shape[-1]))
        meta["modality"] = str(getattr(ds, "Modality", "Unknown"))
        if modality.startswith("CT"):
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            hu = arr * slope + intercept
            center, width = -600.0, 1500.0
            lo, hi = center - width / 2, center + width / 2
            arr = np.clip(hu, lo, hi)
            meta["window"] = "Lung (-600 / 1500 HU)"
        else:
            lo, hi = np.percentile(arr, [1, 99])
            arr = np.clip(arr, lo, hi)
        arr -= arr.min()
        arr /= max(float(arr.max()), 1e-6)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = 1.0 - arr
        return Image.fromarray(np.uint8(arr * 255)).convert("RGB"), meta

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    meta["rows"] = image.height
    meta["columns"] = image.width
    return image, meta


def model_transform(size: int, architecture: str = "mobilenet_v3_small"):
    if architecture == "tiny_cxr":
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def render_probability_bar(label: str, value: float):
    st.markdown(f"**{label}**")
    st.progress(float(max(0.0, min(1.0, value))))
    st.caption(f"{value * 100:.1f}%")


def uncertainty_band(entropy: float, mi: float) -> str:
    # Interface-only heuristic for visually summarizing uncertainty.
    if entropy > 0.62 or mi > 0.06:
        return "High"
    if entropy > 0.42 or mi > 0.025:
        return "Moderate"
    return "Low"


def demo_triage_tier(p_positive: float, uncertainty: str, threshold: float) -> str:
    # NOT a clinical rule. Used only to demonstrate a queue interface.
    if uncertainty == "High":
        return "Review"
    if p_positive >= max(threshold, 0.80):
        return "Priority"
    if p_positive >= threshold:
        return "Elevated"
    return "Routine"


ped_model, ped_ckpt = load_checkpoint(str(PED_PATH))
nih_model, nih_ckpt = load_checkpoint(str(NIH_PATH))

ensemble_models, ensemble_ckpts = [], []
if all(p.exists() for p in ENSEMBLE_PATHS):
    for p in ENSEMBLE_PATHS:
        m, c = load_checkpoint(str(p))
        ensemble_models.append(m)
        ensemble_ckpts.append(c)

ped_metrics = load_json(str(PED_METRICS))
ensemble_metrics = load_json(str(ENSEMBLE_METRICS))
nih_metrics = load_json(str(NIH_METRICS))


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🫁 Triage Lab")
    st.caption("Chest imaging AI research workspace")
    st.markdown("---")

    page = st.radio(
        "Workspace",
        ["Overview", "Analyze Study", "Model Performance", "Research Notes"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### System status")
    if ped_model is not None:
        st.markdown('<span class="status-ok">● Primary CXR model loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">● Primary model missing</span>', unsafe_allow_html=True)

    if ensemble_models:
        st.markdown('<span class="status-ok">● 3-model ensemble loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">● Ensemble unavailable</span>', unsafe_allow_html=True)

    if nih_model is not None:
        st.markdown('<span class="status-ok">● NIH multi-label loaded</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">● NIH model pending</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Research prototype only. No patient-care decisions should be based on this interface.")


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="smallcaps" style="color:#9fd7ea">AI-assisted chest imaging research</div>
      <h1>Chest Imaging Triage Dashboard</h1>
      <p>Uncertainty-aware inference, Grad-CAM explanations, model comparison, and deployment diagnostics in one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------
if page == "Overview":
    a, b, c, d = st.columns(4)
    a.metric("Primary test AUC", f"{ped_metrics.get('auc', float('nan')):.3f}" if ped_metrics else "—")
    b.metric("Sensitivity", f"{ped_metrics.get('recall_sensitivity', float('nan')):.1%}" if ped_metrics else "—")
    c.metric("Specificity", f"{ped_metrics.get('specificity', float('nan')):.1%}" if ped_metrics else "—")
    d.metric("Calibration ECE", f"{ped_metrics.get('ece_10', float('nan')):.3f}" if ped_metrics else "—")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### Workflow")
        st.markdown(
            """
            <div class="card">
            <b>1. Ingest</b><br><span class="muted">PNG, JPEG, or de-identified DICOM chest images</span><br><br>
            <b>2. Infer</b><br><span class="muted">Single-model or deep-ensemble research prediction</span><br><br>
            <b>3. Quantify uncertainty</b><br><span class="muted">MC Dropout entropy, mutual information, or ensemble disagreement</span><br><br>
            <b>4. Explain</b><br><span class="muted">Grad-CAM localization for visual model inspection</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### Demonstration queue")
        demo_rows = [
            ("ST-1042", "Chest X-ray", "Priority", "Model demo"),
            ("ST-1047", "Chest X-ray", "Elevated", "Model demo"),
            ("ST-1051", "CT", "Review", "Preview only"),
            ("ST-1054", "Chest X-ray", "Routine", "Model demo"),
        ]
        for study, modality, tier, note in demo_rows:
            cls = "risk-high" if tier == "Priority" else "risk-med" if tier in {"Elevated", "Review"} else "risk-low"
            st.markdown(
                f"""
                <div class="queue-card {cls}">
                  <b>{study}</b> &nbsp; <span class="smallcaps">{modality}</span><br>
                  <span style="font-weight:700">{tier}</span>
                  <span class="muted"> · {note}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption("Queue entries above are interface simulations, not real patients or model-evaluation cases.")

    st.markdown("### What is currently implemented")
    cols = st.columns(4)
    cols[0].success("CXR inference")
    cols[1].success("MC Dropout + ensemble")
    cols[2].success("Grad-CAM")
    cols[3].info("CT preview only")


# ---------------------------------------------------------------------
# Analyze Study
# ---------------------------------------------------------------------
elif page == "Analyze Study":
    top1, top2 = st.columns([1, 1])
    with top1:
        modality = st.segmented_control(
            "Modality",
            options=["Chest X-ray", "CT (preview only)"],
            default="Chest X-ray",
        )
    with top2:
        source = st.segmented_control(
            "Input",
            options=["Upload", "Synthetic example"],
            default="Synthetic example",
        )

    available_models = []
    if ped_model is not None:
        available_models.append("Single CXR model")
    if ensemble_models:
        available_models.append("3-model ensemble")
    if nih_model is not None:
        available_models.append("NIH 14-finding model")

    control1, control2, control3 = st.columns(3)
    with control1:
        selected_model = st.selectbox(
            "Research model",
            available_models if modality == "Chest X-ray" and available_models else ["No predictive model for CT"],
            disabled=modality != "Chest X-ray",
        )
    with control2:
        mc_passes = st.slider("MC Dropout passes", 5, 50, 20, 5, disabled=selected_model != "Single CXR model")
    with control3:
        threshold = st.slider("Demo alert threshold", 0.30, 0.90, 0.50, 0.05)

    advanced = st.expander("Visualization controls")
    with advanced:
        cam_alpha = st.slider("Grad-CAM overlay opacity", 0.15, 0.75, 0.38, 0.05)
        show_raw_metrics = st.checkbox("Show raw uncertainty values", value=True)

    image = None
    meta = {}
    is_synthetic = False

    if source == "Upload":
        upload = st.file_uploader(
            "Upload a de-identified chest image",
            type=["png", "jpg", "jpeg", "dcm", "dicom"],
            help="Do not upload identifiable patient data to this public research demo.",
        )
        if upload:
            try:
                image, meta = uploaded_to_pil(upload, modality)
            except Exception as exc:
                st.error(f"Could not decode image: {exc}")
                st.stop()
    else:
        demo_path = Path("images/cxr_synthetic.png" if modality == "Chest X-ray" else "images/ct_synthetic.png")
        if demo_path.exists():
            image = Image.open(demo_path).convert("RGB")
            meta = {"filename": demo_path.name, "format": "Synthetic", "rows": image.height, "columns": image.width}
            is_synthetic = True
        else:
            st.info("Synthetic example image is not available in this deployment.")

    if image is not None:
        st.markdown("---")
        study_col, details_col = st.columns([1.45, 1])

        with study_col:
            tabs = st.tabs(["Original", "Explanation"])
            with tabs[0]:
                st.image(image, use_container_width=True)
                if is_synthetic:
                    st.caption("Synthetic, non-clinical image used only to demonstrate the interface.")
                else:
                    st.caption("Input preview. Ensure all uploaded medical data are de-identified.")

        with details_col:
            st.markdown("### Study information")
            st.markdown(
                f"""
                <div class="card">
                  <div class="smallcaps">Source</div><b>{meta.get('filename', 'Unknown')}</b><br><br>
                  <div class="smallcaps">Format</div><b>{meta.get('format', 'Unknown')}</b><br><br>
                  <div class="smallcaps">Dimensions</div><b>{meta.get('columns', '—')} × {meta.get('rows', '—')}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if modality.startswith("CT"):
            with tabs[1]:
                st.image(image, use_container_width=True)
                st.info("CT is currently a preprocessing/visualization track only. No CXR classifier is applied to CT images.")
            st.markdown(
                '<div class="card"><b>CT status:</b> Dedicated CT training and validation remain a separate research milestone.</div>',
                unsafe_allow_html=True,
            )

        elif selected_model == "Single CXR model":
            model, ckpt = ped_model, ped_ckpt
            size = int(ckpt.get("input_size", 160))
            arch = ckpt.get("architecture", "mobilenet_v3_small")
            x = model_transform(size, arch)(image).unsqueeze(0)

            with st.spinner("Running uncertainty-aware inference..."):
                t0 = time.perf_counter()
                mean, var, entropy, mi = mc_predict(model, x, passes=mc_passes)
                latency_ms = (time.perf_counter() - t0) * 1000
                idx = int(mean.argmax(1).item())

                cam_engine = GradCAM(model, gradcam_target_layer(model, arch))
                cam, _ = cam_engine(x, idx)
                cam_engine.close()
                overlay = overlay_cam(image, cam, alpha=cam_alpha)

            with tabs[1]:
                st.image(overlay, use_container_width=True)
                st.caption(f"Grad-CAM for predicted class: {ckpt['classes'][idx]}")

            p_pos = float(mean[0, 1].item()) if len(ckpt["classes"]) > 1 else float(mean[0, 0].item())
            ent = float(entropy.item())
            mi_v = float(mi.item())
            unc = uncertainty_band(ent, mi_v)
            tier = demo_triage_tier(p_pos, unc, threshold)

            st.markdown("### Analysis summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted class", ckpt["classes"][idx])
            m2.metric("Pneumonia probability", f"{p_pos:.1%}")
            m3.metric("Uncertainty", unc)
            m4.metric("Demo triage tier", tier)

            render_probability_bar("Positive-class probability", p_pos)

            if show_raw_metrics:
                r1, r2, r3 = st.columns(3)
                r1.metric("Predictive entropy", f"{ent:.4f}")
                r2.metric("Mutual information", f"{mi_v:.4f}")
                r3.metric("Inference latency", f"{latency_ms:.0f} ms")

            st.warning("The triage tier is an interface heuristic for demonstration only and is not a clinical operating threshold.")

        elif selected_model == "3-model ensemble":
            ckpt = ensemble_ckpts[0]
            size = int(ckpt.get("input_size", 160))
            arch = ckpt.get("architecture", "mobilenet_v3_small")
            x = model_transform(size, arch)(image).unsqueeze(0)

            with st.spinner("Running deep ensemble..."):
                t0 = time.perf_counter()
                mean, var, entropy, mi = ensemble_predict(ensemble_models, x)
                latency_ms = (time.perf_counter() - t0) * 1000
                idx = int(mean.argmax(1).item())
                cam_engine = GradCAM(
                    ensemble_models[0],
                    gradcam_target_layer(ensemble_models[0], arch),
                )
                cam, _ = cam_engine(x, idx)
                cam_engine.close()
                overlay = overlay_cam(image, cam, alpha=cam_alpha)

            with tabs[1]:
                st.image(overlay, use_container_width=True)
                st.caption("Grad-CAM shown from ensemble member 1.")

            p_pos = float(mean[0, 1].item())
            ent = float(entropy.item())
            mi_v = float(mi.item())
            disagreement = float(var[0, 1].item())
            unc = uncertainty_band(ent, mi_v)
            tier = demo_triage_tier(p_pos, unc, threshold)

            st.markdown("### Ensemble summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Predicted class", ckpt["classes"][idx])
            m2.metric("Ensemble probability", f"{p_pos:.1%}")
            m3.metric("Model disagreement", f"{disagreement:.5f}")
            m4.metric("Demo triage tier", tier)
            render_probability_bar("Positive-class ensemble probability", p_pos)

            if show_raw_metrics:
                r1, r2, r3 = st.columns(3)
                r1.metric("Predictive entropy", f"{ent:.4f}")
                r2.metric("Mutual information", f"{mi_v:.4f}")
                r3.metric("Inference latency", f"{latency_ms:.0f} ms")

            st.warning("The triage tier is an interface heuristic for demonstration only and is not a clinical operating threshold.")

        elif selected_model == "NIH 14-finding model":
            model, ckpt = nih_model, nih_ckpt
            size = int(ckpt.get("input_size", 128))
            arch = ckpt.get("architecture", "mobilenet_v3_small")
            x = model_transform(size, arch)(image).unsqueeze(0)

            with st.spinner("Running multi-label model..."):
                t0 = time.perf_counter()
                mean, var, entropy, mi = mc_predict_multilabel(model, x, passes=mc_passes)
                latency_ms = (time.perf_counter() - t0) * 1000
                probs = mean[0]
                order = torch.argsort(probs, descending=True).tolist()
                top = order[:5]
                target_idx = top[0]
                cam_engine = GradCAM(model, gradcam_target_layer(model, arch))
                cam, _ = cam_engine(x, target_idx)
                cam_engine.close()
                overlay = overlay_cam(image, cam, alpha=cam_alpha)

            with tabs[1]:
                st.image(overlay, use_container_width=True)
                st.caption(f"Grad-CAM for highest-scoring finding: {ckpt['classes'][target_idx]}")

            st.markdown("### Top findings")
            rows = []
            for i in top:
                rows.append({
                    "Finding": ckpt["classes"][i],
                    "Probability": round(float(mean[0, i]), 4),
                    "MC variance": round(float(var[0, i]), 6),
                    "Entropy": round(float(entropy[0, i]), 4),
                    "Mutual information": round(float(mi[0, i]), 6),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption(f"20-pass inference latency: {latency_ms:.0f} ms")


# ---------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------
elif page == "Model Performance":
    st.markdown("### Held-out performance")
    st.caption("Metrics shown below are source-dataset research results, not evidence of clinical effectiveness.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Primary CXR model")
        if ped_metrics:
            c1, c2 = st.columns(2)
            c1.metric("ROC-AUC", f"{ped_metrics.get('auc', 0):.4f}")
            c2.metric("Accuracy", f"{ped_metrics.get('accuracy', 0):.2%}")
            c3, c4 = st.columns(2)
            c3.metric("Sensitivity", f"{ped_metrics.get('recall_sensitivity', 0):.2%}")
            c4.metric("Specificity", f"{ped_metrics.get('specificity', 0):.2%}")
            c5, c6 = st.columns(2)
            c5.metric("Brier score", f"{ped_metrics.get('brier', 0):.4f}")
            c6.metric("ECE (10 bins)", f"{ped_metrics.get('ece_10', 0):.4f}")
        else:
            st.info("Metrics file not available.")

    with right:
        st.markdown("#### Three-model ensemble")
        if ensemble_metrics:
            c1, c2 = st.columns(2)
            c1.metric("ROC-AUC", f"{ensemble_metrics.get('auc', 0):.4f}")
            c2.metric("Accuracy", f"{ensemble_metrics.get('accuracy', 0):.2%}")
            c3, c4 = st.columns(2)
            c3.metric("Sensitivity", f"{ensemble_metrics.get('sensitivity', 0):.2%}")
            c4.metric("Specificity", f"{ensemble_metrics.get('specificity', 0):.2%}")
            st.metric(
                "Mean between-model variance",
                f"{ensemble_metrics.get('mean_between_model_variance', 0):.6f}",
            )
        else:
            st.info("Ensemble metrics file not available.")

    st.markdown("### Confusion matrix — primary model")
    cm = ped_metrics.get("confusion_matrix") if ped_metrics else None
    if cm:
        st.dataframe(
            {
                "": ["Actual normal", "Actual pneumonia"],
                "Predicted normal": [cm[0][0], cm[1][0]],
                "Predicted pneumonia": [cm[0][1], cm[1][1]],
            },
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Research interpretation")
    st.info(
        "The current trained baseline is a pediatric binary pneumonia classifier. "
        "It does not yet represent the full preprint scope of multi-condition emergency triage across adult CXR and CT."
    )


# ---------------------------------------------------------------------
# Research Notes
# ---------------------------------------------------------------------
else:
    st.markdown("### Research scope and safeguards")
    st.markdown(
        """
        <div class="card">
        <b>Implemented</b><br>
        • Chest X-ray classification<br>
        • Deep ensemble inference<br>
        • MC Dropout uncertainty<br>
        • Predictive entropy and mutual information<br>
        • Grad-CAM visualization<br>
        • PNG/JPEG/DICOM ingestion<br>
        • CT lung-window preview<br><br>

        <b>Not yet clinically validated</b><br>
        • Adult emergency-department triage<br>
        • Multi-condition urgent finding detection<br>
        • Dedicated CT pathology model<br>
        • External / multi-center validation<br>
        • Prospective clinician workflow evaluation
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Intended role")
    st.write(
        "This application is a research and interface prototype designed to test the engineering "
        "workflow described in the preprint. It is not a diagnostic device and should not be used "
        "to make patient-care decisions."
    )

    st.markdown("### Data handling")
    st.write(
        "Only de-identified images should be uploaded. Synthetic images bundled with the repository "
        "are used for interface demonstrations and are not clinical examples."
    )


st.markdown(
    '<div class="footer-note">Chest Imaging Triage Research Prototype · Tobi Joshua Samuel · Not for clinical use</div>',
    unsafe_allow_html=True,
)
