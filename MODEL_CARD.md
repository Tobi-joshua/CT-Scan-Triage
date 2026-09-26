# Model Card — CXR Pneumonia Baseline

## Intended use

Research prototype for testing the chest-imaging triage software path. It classifies a chest X-ray as **normal** or **pneumonia** and supports Monte Carlo Dropout uncertainty plus Grad-CAM visualization.

**It is not a medical device and must not be used for diagnosis or patient-care decisions.**

## Model

- Architecture: MobileNetV3-Small
- Initialization: ImageNet transfer learning
- Convolutional feature extractor: frozen for this baseline
- Classification head: 2 classes with dropout 0.30
- Input: RGB-converted chest X-ray resized to 160 x 160
- Seed: 2026
- Training epochs: 4
- Checkpoint SHA-256: `a77edac62327a48f607b56e7e5aa2fb9272f690ba178e56ada80a6dc7acdccfa`

## Dataset

Public processed version of the Kermany et al. pediatric chest-X-ray pneumonia dataset:

- Train: 4,187
- Validation: 1,045
- Held-out test: 624
- Labels: normal / pneumonia
- Population described by the source dataset: pediatric patients, approximately ages 1–5

Source used by the reproducible workflow: `mmenendezg/pneumonia_x_ray` (Hugging Face), derived from Kermany et al., Mendeley Data DOI 10.17632/rscbjbr9sj.3.

## Held-out test results

| Metric | Value |
|---|---:|
| ROC AUC | 0.9813 |
| Accuracy | 0.9359 |
| Pneumonia precision | 0.9605 |
| Sensitivity / recall | 0.9359 |
| Specificity | 0.9359 |
| F1 | 0.9481 |
| Brier score | 0.0490 |
| ECE (10 bins) | 0.0147 |

Confusion matrix (rows=true, columns=predicted; normal then pneumonia):

```
[[219, 15],
 [ 25,365]]
```

## Limitations

These results validate only the software baseline on one pediatric source dataset. They do **not** validate the broader preprint claim for adult emergency-department triage, multiple pathologies, CT, domain shift, clinical workflow benefit, or clinical safety. External and prospective validation are required.

## Reproducibility

The exact workflow is `.github/workflows/train-cxr.yml`; detailed epoch history is stored in `artifacts/training_summary.json`.
