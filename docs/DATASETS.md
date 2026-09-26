# Dataset and provenance plan

Raw medical images are **not committed to Git**. The repository stores code, small trained research checkpoints, metrics, dataset identifiers, and preprocessing rules.

## 1. Kermany pediatric chest X-ray — trained baseline

Current binary baseline source: `mmenendezg/pneumonia_x_ray`, a processed Hugging Face representation of the Kermany et al. pediatric chest-X-ray dataset.

- 5,856 images
- Train: 4,187
- Validation: 1,045
- Test: 624
- Labels: normal / pneumonia
- Original dataset DOI: 10.17632/rscbjbr9sj.3
- Original Mendeley dataset license: CC BY 4.0
- Population: pediatric chest radiographs, approximately ages 1–5

The committed checkpoint and metrics correspond to this source and are documented in `MODEL_CARD.md`.

## 2. NIH ChestX-ray14 — adult multi-label track

Source used by the automated track: `timm/nih-chest-xray-14`.

The original NIH ChestX-ray14 collection contains 112,120 frontal radiographs from 30,805 unique patients and 14 text-mined findings:

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax.

The Hugging Face representation preserves the official patient-separated train/test split. The lightweight automated baseline intentionally trains on a reproducible subset for compute efficiency. NIH labels are report-derived/noisy; they are not equivalent to adjudicated diagnostic ground truth.

## 3. RSNA Pneumonia Detection Challenge — DICOM track

The repository provides `scripts/prepare_rsna.py` for DICOM conversion and deterministic patient-level train/validation/test splitting.

Expected layout after preparation:

```text
data/cxr/
  train/normal
  train/pneumonia
  val/normal
  val/pneumonia
  test/normal
  test/pneumonia
```

Keep the competition/source data outside Git and follow the source terms and attribution requirements.

## 4. LIDC-IDRI / TCIA — CT track

For thoracic CT ingestion, the project targets LIDC-IDRI from The Cancer Imaging Archive (TCIA).

- 1,010 subjects
- Thoracic CT DICOM studies
- Radiologist nodule annotations
- CC BY 3.0
- Full collection is large, so raw studies are not vendored into this repository

The operational CT preprocessing path is:

```bash
PYTHONPATH=. python scripts/prepare_ct_series.py \
  --dicom-dir /path/to/deidentified/series \
  --out data/ct/example_series \
  --num-slices 32
```

It sorts axial slices, converts stored values to Hounsfield units, applies a configurable lung window (default center -600 HU, width 1500 HU), samples the volume uniformly, creates three-slice pseudo-RGB images, and exports only non-identifying derived metadata.

**Important:** LIDC-IDRI is suitable for developing CT ingestion and nodule-oriented experiments, but its annotations do not directly provide all emergency triage targets claimed in the broader preprint. A dedicated, appropriately labelled CT cohort is still required for pneumothorax/effusion/edema/consolidation triage validation.

## Reproducibility requirements

For every experiment record:

- source dataset and version/DOI
- license / terms
- population and modality
- patient-level split rule
- inclusion/exclusion criteria
- preprocessing parameters
- training seed
- checkpoint SHA-256
- held-out metrics
- known label-quality limitations

Synthetic images in `images/` are for visualization/UI demonstration only and are never used as evidence of model performance.
