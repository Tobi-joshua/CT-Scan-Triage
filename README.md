# CT-Scan-Triage

Research implementation accompanying **Development of a Lightweight AI-Assisted Triage Pipeline for Chest Imaging**.

> **Research prototype only — not a medical device and not validated for diagnosis or clinical decision-making.**

## Current architecture

The repository now separates the two imaging modalities rather than mixing them into one classifier:

- **Chest X-ray track:** patient-level preparation of the RSNA/NIH pneumonia data, MobileNetV3-Small transfer learning, calibration-ready probabilities, Monte Carlo Dropout uncertainty, and Grad-CAM.
- **CT track:** data provenance and preparation plan for TCIA/LIDC-IDRI. CT inference remains disabled in the demo until a dedicated CT model is trained and evaluated.
- **Demo:** Streamlit upload interface for a de-identified CXR, predicted class/probability, predictive entropy, mutual information, latency, and Grad-CAM.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare RSNA chest X-rays

Download the RSNA Pneumonia Detection Challenge images and detailed class CSV under the dataset's terms. Keep raw data outside Git.

```bash
python scripts/prepare_rsna.py \
  --dicom-dir /path/to/stage_2_train_images \
  --class-csv /path/to/stage_2_detailed_class_info.csv \
  --out data/cxr
```

The script creates deterministic patient-level train/validation/test folders and excludes the ambiguous `No Lung Opacity / Not Normal` class from the first binary baseline.

## Train

```bash
python train.py --data data/cxr --epochs 10 --out artifacts/cxr_mobilenetv3.pt
```

## Run Streamlit

```bash
streamlit run app.py
```

The app expects `artifacts/cxr_mobilenetv3.pt`. Model weights are intentionally not fabricated or committed before training.

## Repository layout

```
app.py                  Streamlit research demo
train.py                CXR training entry point
scripts/prepare_rsna.py RSNA DICOM -> patient-level PNG split
src/data.py             transforms/loaders
src/model.py            MobileNetV3 + MC Dropout
src/gradcam.py          Grad-CAM implementation
docs/DATASETS.md        dataset provenance/licensing plan
artifacts/              local checkpoints (do not commit large weights)
```

## Reproducibility / safety

Do not commit patient data. Keep dataset versions, licenses, split seed/rule, preprocessing, model checkpoint hashes, and evaluation metrics with each experiment. A clinically meaningful release still requires held-out and external validation, calibration analysis, subgroup analysis where metadata permits, and clinician evaluation.
