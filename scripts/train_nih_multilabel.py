from __future__ import annotations

import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset, Image as HFImage
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from src.model import build_model

CLASSES = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion","Emphysema",
    "Fibrosis","Hernia","Infiltration","Mass","Nodule","Pleural_Thickening",
    "Pneumonia","Pneumothorax"
]
TARGETS = ["Pneumothorax","Edema","Effusion","Consolidation","Pneumonia"]
MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)

def make_stream(dataset, split, n, seed, decode=True):
    ds=load_dataset(dataset, split=split, streaming=True)
    if not decode:
        ds=ds.cast_column("image", HFImage(decode=False))
    ds=ds.shuffle(seed=seed, buffer_size=min(5000,n))
    return ds.take(n)

def count_labels(dataset, n, seed):
    counts=np.zeros(len(CLASSES),dtype=np.int64)
    total=0
    for row in make_stream(dataset,"train",n,seed,decode=False):
        names=set(row.get("label_names") or [])
        for i,c in enumerate(CLASSES):
            counts[i]+=int(c in names)
        total+=1
    return total,counts

class HFStream(IterableDataset):
    def __init__(self, dataset, split, n, seed, transform):
        self.dataset=dataset; self.split=split; self.n=n; self.seed=seed; self.transform=transform
    def __iter__(self):
        for row in make_stream(self.dataset,self.split,self.n,self.seed,decode=True):
            img=row["image"].convert("RGB")
            names=set(row.get("label_names") or [])
            y=torch.tensor([float(c in names) for c in CLASSES],dtype=torch.float32)
            yield self.transform(img),y

@torch.inference_mode()
def evaluate(model,loader,device):
    model.eval(); ys=[]; ps=[]
    for x,y in loader:
        p=torch.sigmoid(model(x.to(device))).cpu().numpy()
        ys.append(y.numpy()); ps.append(p)
    y=np.concatenate(ys); p=np.concatenate(ps)
    per={}
    aucs=[]
    for i,c in enumerate(CLASSES):
        pos=int(y[:,i].sum()); neg=len(y)-pos
        auc=None
        if pos>0 and neg>0:
            auc=float(roc_auc_score(y[:,i],p[:,i])); aucs.append(auc)
        per[c]={"auc":auc,"positives":pos,"negatives":neg}
    return {
        "n":int(len(y)),
        "macro_auc_available_labels":float(np.mean(aucs)) if aucs else None,
        "per_class":per,
        "target_findings":{c:per[c] for c in TARGETS},
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="timm/nih-chest-xray-14")
    ap.add_argument("--train-n",type=int,default=16000)
    ap.add_argument("--test-n",type=int,default=4000)
    ap.add_argument("--epochs",type=int,default=2)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--size",type=int,default=128)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--out",default="artifacts/cxr_nih_multilabel.pt")
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device",device)

    total,counts=count_labels(args.dataset,args.train_n,args.seed)
    pos_weight=(total-counts)/np.maximum(counts,1)
    pos_weight=np.clip(pos_weight,1.0,25.0)
    print("class counts",dict(zip(CLASSES,counts.tolist())))

    train_tf=transforms.Compose([
        transforms.Resize((args.size,args.size)),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)
    ])
    eval_tf=transforms.Compose([
        transforms.Resize((args.size,args.size)),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)
    ])

    model=build_model(len(CLASSES),dropout=.30,pretrained=True).to(device)
    for p in model.features.parameters(): p.requires_grad=False
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(model.classifier.parameters(),lr=1e-3,weight_decay=1e-4)
    history=[]; out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)

    for epoch in range(1,args.epochs+1):
        loader=DataLoader(HFStream(args.dataset,"train",args.train_n,args.seed+epoch,train_tf),
                          batch_size=args.batch_size,num_workers=0)
        model.train(); model.features.eval(); loss_sum=0.; seen=0
        for step,(x,y) in enumerate(loader,1):
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True); logits=model(x); loss=loss_fn(logits,y)
            loss.backward(); opt.step()
            loss_sum+=loss.item()*len(y); seen+=len(y)
            if step%50==0: print("epoch",epoch,"step",step,"loss",loss_sum/max(seen,1))
        history.append({"epoch":epoch,"train_loss":loss_sum/max(seen,1)})
        torch.save({"state_dict":model.state_dict(),"classes":CLASSES,"input_size":args.size,
                    "dropout":.30,"dataset":args.dataset,"seed":args.seed,
                    "train_n":args.train_n,"task":"multilabel"},out)

    test_loader=DataLoader(HFStream(args.dataset,"test",args.test_n,args.seed+99,eval_tf),
                           batch_size=args.batch_size,num_workers=0)
    metrics=evaluate(model,test_loader,device)
    sha=hashlib.sha256(out.read_bytes()).hexdigest()
    summary={"dataset":args.dataset,"train_n":args.train_n,"test_n":args.test_n,
             "epochs":args.epochs,"input_size":args.size,"classes":CLASSES,
             "targets":TARGETS,"metrics":metrics,"history":history,
             "checkpoint_sha256":sha,
             "limitations":[
               "Training uses a reproducible subset of NIH ChestX-ray14 rather than the full collection.",
               "NIH labels are text-mined from reports and are noisy; they are not equivalent to adjudicated ground truth.",
               "Thresholds are not clinically tuned; per-class ROC AUC is the primary reported metric.",
               "Research use only; no prospective or external clinical validation."
             ]}
    out.with_name("nih_multilabel_metrics.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
