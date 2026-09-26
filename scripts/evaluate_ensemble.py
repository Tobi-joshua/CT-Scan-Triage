from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from src.data import IMAGENET_MEAN,IMAGENET_STD
from src.model import build_model
from src.ensemble import ensemble_predict

class HFDataset(Dataset):
    def __init__(self,split,size):
        self.split=split
        self.t=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),
                                   transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    def __len__(self): return len(self.split)
    def __getitem__(self,i):
        r=self.split[i]
        return self.t(r["image"].convert("RGB")),int(r["label"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoints",nargs="+",required=True)
    ap.add_argument("--dataset",default="mmenendezg/pneumonia_x_ray")
    ap.add_argument("--out",default="artifacts/ensemble_metrics.json")
    args=ap.parse_args()
    ckpts=[torch.load(p,map_location="cpu") for p in args.checkpoints]
    sizes={int(c.get("input_size",160)) for c in ckpts}
    if len(sizes)!=1: raise ValueError(f"ensemble input sizes differ: {sizes}")
    size=sizes.pop(); models=[]
    for c in ckpts:
        m=build_model(len(c["classes"]),dropout=float(c.get("dropout",.30)),pretrained=False)
        m.load_state_dict(c["state_dict"]);m.eval();models.append(m)
    ds=load_dataset(args.dataset,split="test")
    loader=DataLoader(HFDataset(ds,size),batch_size=64,shuffle=False,num_workers=2)
    ys=[];ps=[];dis=[]
    for x,y in loader:
        mean,var,ent,mi=ensemble_predict(models,x)
        ys.extend(y.numpy());ps.extend(mean[:,1].numpy());dis.extend(var[:,1].numpy())
    y=np.asarray(ys);p=np.asarray(ps);pred=(p>=.5).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    metrics={
      "members":args.checkpoints,
      "n":int(len(y)),
      "auc":float(roc_auc_score(y,p)),
      "accuracy":float(accuracy_score(y,pred)),
      "sensitivity":float(tp/max(tp+fn,1)),
      "specificity":float(tn/max(tn+fp,1)),
      "f1":float(f1_score(y,pred)),
      "mean_between_model_variance":float(np.mean(dis)),
      "confusion_matrix":cm.tolist()
    }
    Path(args.out).write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
