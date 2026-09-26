from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss
from src.data import imagefolder_loaders
from src.model import build_model

def ece(y, p, bins=10):
    conf=np.maximum(p,1-p); pred=(p>=.5).astype(int); out=0.
    edges=np.linspace(0,1,bins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(conf>lo)&(conf<=hi)
        if m.any(): out += m.mean()*abs((pred[m]==y[m]).mean()-conf[m].mean())
    return float(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--checkpoint",required=True); ap.add_argument("--out",default="artifacts/test_metrics.json")
    args=ap.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _,_,test_ds,_,_,loader=imagefolder_loaders(args.data,batch_size=64)
    if test_ds is None: raise ValueError("A test split is required.")
    ckpt=torch.load(args.checkpoint,map_location=device); model=build_model(len(ckpt["classes"]),pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"]); model.eval(); ys=[]; ps=[]
    with torch.inference_mode():
        for x,y in loader:
            p=torch.softmax(model(x.to(device)),1)[:,1].cpu().numpy()
            ys.extend(y.numpy()); ps.extend(p)
    y=np.asarray(ys); p=np.asarray(ps); pred=(p>=.5).astype(int)
    metrics={"auc":float(roc_auc_score(y,p)),"brier":float(brier_score_loss(y,p)),"ece":ece(y,p),
             "confusion_matrix":confusion_matrix(y,pred).tolist(),
             "classification_report":classification_report(y,pred,target_names=ckpt["classes"],output_dict=True)}
    Path(args.out).write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
