from __future__ import annotations
import argparse, json
from pathlib import Path
from src.ct import load_dicom_series, window_hu, sample_indices, three_slice_rgb

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dicom-dir",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--num-slices",type=int,default=32)
    ap.add_argument("--center",type=float,default=-600.0)
    ap.add_argument("--width",type=float,default=1500.0)
    args=ap.parse_args()

    volume,datasets=load_dicom_series(args.dicom_dir)
    win=window_hu(volume,args.center,args.width)
    indices=sample_indices(len(win),args.num_slices)
    out=Path(args.out); imgdir=out/"slices"; imgdir.mkdir(parents=True,exist_ok=True)
    for j,i in enumerate(indices):
        three_slice_rgb(win,int(i)).save(imgdir/f"slice_{j:03d}_z{i:04d}.png")

    first=datasets[0]
    spacing=getattr(first,"PixelSpacing",None)
    thickness=getattr(first,"SliceThickness",None)
    meta={
        "num_input_slices":int(volume.shape[0]),
        "height":int(volume.shape[1]),
        "width":int(volume.shape[2]),
        "exported_indices":[int(i) for i in indices],
        "window_center":args.center,
        "window_width":args.width,
        "pixel_spacing":[float(x) for x in spacing] if spacing is not None else None,
        "slice_thickness":float(thickness) if thickness is not None else None,
        "note":"No patient-identifying DICOM metadata is exported."
    }
    (out/"series_metadata.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))

if __name__=="__main__":
    main()
