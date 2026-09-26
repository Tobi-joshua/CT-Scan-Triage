from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.model import build_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class HFXrayDataset(Dataset):
    def __init__(self, split, transform):
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        row = self.split[idx]
        image = row["image"].convert("RGB")
        return self.transform(image), int(row["label"])

def seed_everything(seed: int = 2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def ece_score(y_true, prob_pos, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    prob_pos = np.asarray(prob_pos)
    conf = np.maximum(prob_pos, 1.0 - prob_pos)
    pred = (prob_pos >= 0.5).astype(int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            acc = (pred[mask] == y_true[mask]).mean()
            ece += mask.mean() * abs(acc - conf[mask].mean())
    return float(ece)

@torch.inference_mode()
def evaluate(model, loader, device, classes):
    model.eval()
    y_true, p_pos = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_true.extend(y.numpy().tolist())
        p_pos.extend(probs.cpu().numpy().tolist())
    y_true = np.asarray(y_true, dtype=int)
    p_pos = np.asarray(p_pos, dtype=float)
    y_pred = (p_pos >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, p_pos)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, p_pos)),
        "ece_10": ece_score(y_true, p_pos, bins=10),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=classes, output_dict=True, zero_division=0
        ),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mmenendezg/pneumonia_x_ray")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="artifacts/cxr_mobilenetv3.pt")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    ds = load_dataset(args.dataset)
    classes = ["normal", "pneumonia"]

    train_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = HFXrayDataset(ds["train"], train_tf)
    val_ds = HFXrayDataset(ds["validation"], eval_tf)
    test_ds = HFXrayDataset(ds["test"], eval_tf)

    kwargs = dict(batch_size=args.batch_size, num_workers=2, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    train_labels = np.asarray(ds["train"]["label"], dtype=int)
    counts = np.bincount(train_labels, minlength=2)
    weights = len(train_labels) / (2.0 * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("train class counts:", counts.tolist(), "weights:", weights.tolist())

    model = build_model(num_classes=2, dropout=0.30, pretrained=True).to(device)
    for p in model.features.parameters():
        p.requires_grad = False

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.features.eval()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y)
            seen += len(y)

        val_metrics = evaluate(model, val_loader, device, classes)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record))

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "input_size": args.size,
                    "dropout": 0.30,
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "frozen_features": True,
                },
                out,
            )

    checkpoint = torch.load(out, map_location=device)
    model = build_model(num_classes=2, dropout=float(checkpoint["dropout"]), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate(model, test_loader, device, classes)

    sha256 = hashlib.sha256(out.read_bytes()).hexdigest()
    summary = {
        "dataset": args.dataset,
        "dataset_counts": {
            "train": len(train_ds),
            "validation": len(val_ds),
            "test": len(test_ds),
        },
        "classes": classes,
        "input_size": args.size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "best_validation_auc": best_auc,
        "test": test_metrics,
        "checkpoint_sha256": sha256,
        "history": history,
        "limitations": [
            "Single pediatric source dataset; not representative of adult emergency imaging.",
            "Binary pneumonia classification only; this is not yet the full multi-condition triage task.",
            "Transfer-learning baseline with frozen convolutional features.",
            "Research use only; not clinically validated.",
        ],
    }
    (out.parent / "training_summary.json").write_text(json.dumps(summary, indent=2))
    (out.parent / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
