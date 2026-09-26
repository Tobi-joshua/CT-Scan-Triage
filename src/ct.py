from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pydicom
from PIL import Image

def _z_position(ds) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return float(ipp[2])
    return float(getattr(ds, "InstanceNumber", 0))

def load_dicom_series(folder: str | Path) -> tuple[np.ndarray, list]:
    """Load an axial CT DICOM series and return HU volume [z,y,x] plus datasets."""
    folder=Path(folder)
    datasets=[]
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        try:
            ds=pydicom.dcmread(p, stop_before_pixels=False)
            if hasattr(ds,"PixelData"):
                datasets.append(ds)
        except Exception:
            continue
    if not datasets:
        raise ValueError(f"No readable DICOM slices with pixel data found in {folder}")
    datasets.sort(key=_z_position)
    slices=[]
    for ds in datasets:
        a=ds.pixel_array.astype(np.float32)
        slope=float(getattr(ds,"RescaleSlope",1.0))
        intercept=float(getattr(ds,"RescaleIntercept",0.0))
        slices.append(a*slope+intercept)
    shape=slices[0].shape
    if any(s.shape!=shape for s in slices):
        raise ValueError("Inconsistent slice dimensions in CT series")
    return np.stack(slices,axis=0),datasets

def window_hu(volume: np.ndarray, center: float=-600.0, width: float=1500.0) -> np.ndarray:
    """Window Hounsfield units and normalize to [0,1]."""
    lo=center-width/2.0; hi=center+width/2.0
    out=np.clip(volume.astype(np.float32),lo,hi)
    out=(out-lo)/max(hi-lo,1e-6)
    return out

def sample_indices(depth: int, n: int=32) -> np.ndarray:
    if depth<=0: raise ValueError("depth must be positive")
    n=max(1,min(int(n),depth))
    return np.unique(np.rint(np.linspace(0,depth-1,n)).astype(int))

def three_slice_rgb(windowed: np.ndarray, index: int) -> Image.Image:
    """Create a pseudo-RGB image from previous/current/next axial slices."""
    z=windowed.shape[0]
    idx=[max(0,index-1),index,min(z-1,index+1)]
    a=np.stack([windowed[i] for i in idx],axis=-1)
    return Image.fromarray(np.uint8(np.clip(a,0,1)*255),"RGB")
