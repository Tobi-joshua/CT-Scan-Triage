from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from src.data import imagefolder_loaders
from src.model import build_model

def run_epoch(model, loader, loss_fn, optimizer, device):
    training=optimizer is not None
    model.train(training)
    total=correct=0; loss_sum=0.; ys=[]; ps=[]
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        logits=model(x); loss=loss_fn(logits,y)
        if training: loss.backward(); optimizer.step()
        p=torch.softmax(logits,1)[:,1]
        loss_sum += loss.item()*len(y); total += len(y)
        correct += (logits.argmax(1)==y).sum().item()
        ys.extend(y.detach().cpu().tolist()); ps.extend(p.detach().cpu().tolist())
    try: auc=roc_auc_score(ys,ps)
    except ValueError: auc=float("nan")
    return {"loss":loss_sum/max(total,1),"accuracy":correct/max(total,1),"auc":auc}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True,help="ImageFolder root with train/val[/test]/class_name")
    ap.add_argument("--epochs",type=int,default=10); ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--lr",type=float,default=3e-4); ap.add_argument("--out",default="artifacts/cxr_mobilenetv3.pt")
    args=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds,val_ds,_,tr,va,_=imagefolder_loaders(args.data,args.batch_size)
    if len(train_ds.classes)!=2: raise ValueError("Current baseline expects exactly two classes.")
    model=build_model(2).to(device); loss_fn=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=-1.; history=[]
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        a=run_epoch(model,tr,loss_fn,opt,device)
        with torch.no_grad(): b=run_epoch(model,va,loss_fn,None,device)
        history.append({"epoch":epoch,"train":a,"val":b}); print(history[-1])
        score=b["auc"] if b["auc"]==b["auc"] else b["accuracy"]
        if score>best:
            best=score
            torch.save({"state_dict":model.state_dict(),"classes":train_ds.classes,"input_size":224},out)
    out.with_suffix(".metrics.json").write_text(json.dumps(history,indent=2))

if __name__=="__main__": main()
