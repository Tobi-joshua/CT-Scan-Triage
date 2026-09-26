from __future__ import annotations
import argparse, hashlib, shutil
from pathlib import Path
import pandas as pd
import pydicom
import numpy as np
from PIL import Image

def split_for(patient_id, train=.70, val=.15):
    r=int(hashlib.sha256(patient_id.encode()).hexdigest()[:8],16)/0xffffffff
    return "train" if r<train else ("val" if r<train+val else "test")

def dicom_to_png(src, dst):
    ds=pydicom.dcmread(src)
    a=ds.pixel_array.astype(np.float32)
    a-=a.min(); a/=max(float(a.max()),1e-6)
    if getattr(ds,"PhotometricInterpretation","")=="MONOCHROME1": a=1-a
    Image.fromarray(np.uint8(a*255)).convert("RGB").save(dst)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dicom-dir",required=True)
    ap.add_argument("--class-csv",required=True,help="RSNA detailed class info CSV with patientId and class")
    ap.add_argument("--out",default="data/cxr")
    args=ap.parse_args()
    df=pd.read_csv(args.class_csv).drop_duplicates("patientId")
    out=Path(args.out); src=Path(args.dicom_dir)
    kept=0
    for row in df.itertuples():
        label="pneumonia" if getattr(row,"_2",getattr(row,"class",""))=="Lung Opacity" else "normal"
        # Exclude 'No Lung Opacity / Not Normal' from this binary baseline.
        cls=getattr(row,"_2",getattr(row,"class",""))
        if cls not in {"Lung Opacity","Normal"}: continue
        pid=str(row.patientId); dcm=src/f"{pid}.dcm"
        if not dcm.exists(): continue
        dst=out/split_for(pid)/label/f"{pid}.png"; dst.parent.mkdir(parents=True,exist_ok=True)
        dicom_to_png(dcm,dst); kept+=1
    print(f"Prepared {kept} patient-level images under {out}")

if __name__=="__main__": main()
