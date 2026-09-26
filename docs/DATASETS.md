# Dataset plan

Raw medical images are **not committed to Git**.

## Chest X-ray baseline
Use the RSNA Pneumonia Detection Challenge / NIH-derived chest radiographs and annotations. RSNA permits dataset use subject to its terms and attribution requirements. Convert DICOM images into a patient-level split:

data/cxr/
  train/normal
  train/pneumonia
  val/normal
  val/pneumonia
  test/normal
  test/pneumonia

Avoid image-level leakage: all images from one patient must remain in exactly one split.

## CT track
CT is a separate modality and model track. LIDC-IDRI on TCIA contains thoracic CT DICOM studies and radiologist annotations and is licensed CC BY 3.0. Do not feed arbitrary axial CT slices into the CXR classifier.

The first CT milestone is a slice/volume preparation pipeline followed by a dedicated CT model. The Streamlit UI exposes CT only as a non-predictive preview until that checkpoint exists.

## Provenance
Record dataset version, source URL/DOI, license, patient-level split seed, exclusion criteria, and preprocessing parameters for every experiment.
