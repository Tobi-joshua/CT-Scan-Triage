from __future__ import annotations
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib, json, urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from src.model import build_model

URL="https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
MD5="28209eda62fecd6e6a2d98b1501bb15f"
OUT=Path("artifacts/cxr_mobilenetv3.pt")
DATA=Path(".cache/pneumoniamnist.npz")

def fetch():
    DATA.parent.mkdir(parents=True,exist_ok=True)
    if not DATA.exists() or hashlib.md5(DATA.read_bytes()).hexdigest()!=MD5:
        print("Downloading official PneumoniaMNIST from Zenodo...")
        urllib.request.urlretrieve(URL,DATA)
    digest=hashlib.md5(DATA.read_bytes()).hexdigest()
    if digest!=MD5:
        raise RuntimeError(f"Dataset checksum mismatch: {digest}")

def loader(images, labels, batch, shuffle):
    x=torch.from_numpy(images).float().unsqueeze(1)/255.0
    x=x.repeat(1,3,1,1)
    x=(x-0.5)/0.5
    y=torch.from_numpy(labels.reshape(-1)).long()
    return DataLoader(TensorDataset(x,y),batch_size=batch,shuffle=shuffle)

@torch.inference_mode()
def evaluate(model, dl):
    ys=[]; ps=[]; model.eval()
    for x,y in dl:
        p=torch.softmax(model(x),1)[:,1]
        ys.extend(y.tolist()); ps.extend(p.tolist())
    y=np.asarray(ys); p=np.asarray(ps)
    return float(roc_auc_score(y,p)), float(accuracy_score(y,(p>=0.5).astype(int)))

def main():
    if OUT.exists():
        print("Checkpoint already exists; skipping bootstrap training.")
        return
    torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42)
    fetch(); d=np.load(DATA)
    tr=loader(d["train_images"],d["train_labels"],128,True)
    va=loader(d["val_images"],d["val_labels"],256,False)
    te=loader(d["test_images"],d["test_labels"],256,False)
    model=build_model(2,dropout=.30,pretrained=False,architecture="tiny_cxr")
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    loss_fn=nn.CrossEntropyLoss(); best=-1.; best_state=None
    for epoch in range(1,13):
        model.train(); total=0
        for x,y in tr:
            opt.zero_grad(set_to_none=True)
            logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step()
            total += len(y)
        auc,acc=evaluate(model,va)
        print(f"epoch={epoch} val_auc={auc:.4f} val_acc={acc:.4f}")
        if auc>best:
            best=auc
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_auc,test_acc=evaluate(model,te)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    torch.save({
        "state_dict":model.state_dict(),"classes":["normal","pneumonia"],"input_size":28,
        "architecture":"tiny_cxr","dropout":0.30,
        "dataset":"PneumoniaMNIST v3 / MedMNIST+",
        "dataset_doi":"10.5281/zenodo.10519652",
        "test_auc":test_auc,"test_accuracy":test_acc
    },OUT)
    Path("artifacts/test_metrics.json").write_text(json.dumps({
        "auc":test_auc,"accuracy":test_acc,"recall_sensitivity":None,"specificity":None,
        "source":"PneumoniaMNIST v3 / MedMNIST+ proof-of-concept"
    },indent=2))
    print(f"saved {OUT}; test_auc={test_auc:.4f} test_acc={test_acc:.4f}")

if __name__=="__main__":
    main()
