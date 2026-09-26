from __future__ import annotations
from pathlib import Path
import io, time
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from src.model import build_model, mc_predict
from src.gradcam import GradCAM, overlay_cam
from src.data import IMAGENET_MEAN, IMAGENET_STD

st.set_page_config(page_title="Chest Imaging Triage", page_icon="🫁", layout="wide")
st.title("Chest Imaging Triage — Research Prototype")
st.warning("Research prototype only. Not validated for diagnosis or clinical decision-making.")

MODEL_PATH=Path("artifacts/cxr_mobilenetv3.pt")
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists(): return None,None
    ckpt=torch.load(MODEL_PATH,map_location="cpu")
    model=build_model(len(ckpt["classes"]),pretrained=False)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    return model,ckpt

model,ckpt=load_model()
modality=st.radio("Modality",["Chest X-ray","CT (preview only)"],horizontal=True)
file=st.file_uploader("Upload a de-identified PNG/JPEG image",type=["png","jpg","jpeg"])
if modality.startswith("CT"):
    st.info("CT training/inference is intentionally separated from the CXR model. This preview does not make a CT prediction.")
if file:
    image=Image.open(io.BytesIO(file.getvalue())).convert("RGB")
    left,right=st.columns(2); left.image(image,caption="Input",use_container_width=True)
    if model is None:
        right.info("No trained checkpoint is committed. Train the CXR model first, then place the checkpoint at artifacts/cxr_mobilenetv3.pt.")
    elif modality.startswith("Chest"):
        tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
        x=tfm(image).unsqueeze(0)
        t=time.perf_counter(); mean,var,entropy,mi=mc_predict(model,x,20); ms=(time.perf_counter()-t)*1000
        idx=int(mean.argmax(1).item()); classes=ckpt["classes"]
        cam_engine=GradCAM(model,model.features[-1]); cam,_=cam_engine(x,idx); cam_engine.close()
        overlay=overlay_cam(image,cam)
        right.image(overlay,caption="Grad-CAM explanation",use_container_width=True)
        st.subheader("Research output")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Predicted class",classes[idx]); c2.metric("Mean probability",f"{mean[0,idx].item():.3f}")
        c3.metric("Predictive entropy",f"{entropy.item():.3f}"); c4.metric("MC latency",f"{ms:.0f} ms")
        st.caption(f"Epistemic uncertainty (mutual information): {mi.item():.4f}. 20 stochastic dropout passes.")
