# Chest Imaging Triage

[![Train real CXR baseline](https://github.com/Tobi-joshua/CT-Scan-Triage/actions/workflows/train-cxr.yml/badge.svg)](https://github.com/Tobi-joshua/CT-Scan-Triage/actions/workflows/train-cxr.yml)
[![Inference smoke test](https://github.com/Tobi-joshua/CT-Scan-Triage/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/Tobi-joshua/CT-Scan-Triage/actions/workflows/smoke-test.yml)
[![Streamlit runtime smoke test](https://github.com/Tobi-joshua/CT-Scan-Triage/actions/workflows/runtime-smoke.yml/badge.svg)](https://github.com/Tobi-joshua/CT-Scan-Triage/actions/workflows/runtime-smoke.yml)

Research implementation accompanying **Development of a Lightweight AI-Assisted Triage Pipeline for Chest Imaging**.

> **Research prototype only — not a medical device and not validated for diagnosis or clinical decision-making.**

## Working baseline

The repository contains a genuinely trained chest-X-ray baseline rather than placeholder/random weights.

- **Task:** normal vs pneumonia CXR classification
- **Backbone:** MobileNetV3-Small, ImageNet transfer learning
- **Explainability:** Grad-CAM
- **Uncertainty:** Monte Carlo Dropout, predictive entropy, mutual information
- **Interface:** Streamlit upload + prediction + uncertainty + Grad-CAM
- **Training:** reproducible GitHub Actions workflow
- **CT:** maintained as a separate track; the CXR checkpoint is never applied to CT

### Held-out test results

The current baseline was trained on the public processed Kermany pediatric pneumonia CXR dataset and evaluated on its held-out test split (624 images).

| Metric | Result |
|---|---:|
| ROC AUC | **0.9813** |
| Accuracy | **0.9359** |
| Pneumonia precision | **0.9605** |
| Sensitivity | **0.9359** |
| Specificity | **0.9359** |
| F1 | **0.9481** |
| Brier score | **0.0490** |
| ECE (10 bins) | **0.0147** |

Confusion matrix (true rows, predicted columns; normal then pneumonia):

```text
[[219, 15],
 [ 25,365]]
```

These results apply only to this pediatric source dataset. They do **not** establish performance for adult emergency-department populations, CT, other pathologies, other hospitals/scanners, or prospective clinical use. See [MODEL_CARD.md](MODEL_CARD.md).
### Three-member deep ensemble

The same held-out pediatric test split was also evaluated with three independently seeded MobileNetV3-Small members (2026, 2027, 2028):

| Metric | Result |
|---|---:|
| ROC AUC | **0.9827** |
| Accuracy | **0.9327** |
| Sensitivity | **0.9538** |
| Specificity | **0.8974** |
| F1 | **0.9466** |
| Mean between-model variance | **0.00265** |

The ensemble is included to demonstrate epistemic uncertainty through model disagreement; it is not evidence of clinical safety or generalization.


## Run the app

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install "torch>=2.3,<3" "torchvision>=0.18,<1" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-app.txt
streamlit run app.py
```

The committed checkpoint is:

```text
artifacts/cxr_mobilenetv3.pt
SHA-256: a77edac62327a48f607b56e7e5aa2fb9272f690ba178e56ada80a6dc7acdccfa
```

## Deploy

A Render Blueprint is included at [render.yaml](render.yaml).

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Tobi-joshua/CT-Scan-Triage)

The Blueprint installs CPU-only PyTorch, installs the slim inference dependency set, and runs:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
```

## Reproduce training

The default reproducible baseline uses `mmenendezg/pneumonia_x_ray`, a processed version of the Kermany et al. dataset.

```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/train_hf_baseline.py \
  --epochs 4 \
  --batch-size 64 \
  --size 160 \
  --out artifacts/cxr_mobilenetv3.pt
```

GitHub Actions performs the same training with [.github/workflows/train-cxr.yml](.github/workflows/train-cxr.yml) and records `training_summary.json`, test metrics, and the checkpoint hash.

## RSNA track

The repository also contains patient-level preparation for the RSNA Pneumonia Detection Challenge DICOM dataset:

```bash
python scripts/prepare_rsna.py \
  --dicom-dir /path/to/stage_2_train_images \
  --class-csv /path/to/stage_2_detailed_class_info.csv \
  --out data/cxr
```

Raw medical data is never committed to Git.

## Repository layout

```text
app.py                         Streamlit research demo
train.py                       ImageFolder CXR training path
scripts/train_hf_baseline.py   Reproducible public baseline training
scripts/prepare_rsna.py        RSNA patient-level DICOM preparation
scripts/smoke_inference.py     Checkpoint + uncertainty + Grad-CAM smoke test
src/data.py                    transforms/loaders
src/model.py                   MobileNetV3 + MC Dropout
src/gradcam.py                 Grad-CAM
artifacts/                     trained baseline + recorded metrics
MODEL_CARD.md                  exact scope/results/limitations
docs/DATASETS.md               dataset provenance plan
render.yaml                    Render web-service Blueprint
```

## Research roadmap

1. External validation on an adult chest-X-ray cohort.
2. Multi-label triage for pneumothorax, effusion, edema and consolidation.
3. Threshold selection around triage sensitivity rather than generic 0.5 classification.
4. Temperature scaling / calibration on an untouched validation set.
5. Dedicated CT series preprocessing and CT-specific model; never cross-apply the CXR model.
6. Domain-shift and subgroup analysis.
7. Clinician-in-the-loop usability testing.
8. Quantization/ONNX benchmarking for edge deployment.

## Citation

Preprint DOI: **10.13140/RG.2.2.19454.24647**

Tobi Joshua Samuel. *Development of a Lightweight AI-Assisted Triage Pipeline for Chest Imaging*. Preprint / working draft, 2026.
