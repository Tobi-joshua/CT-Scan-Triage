# CT-Scan-Triage

CT-Scan-Triage is a lightweight, CPU-based Grad-CAM visualization pipeline for CT scan images.
It is designed for model interpretability, medical imaging explainability, and research demos.

The project produces:
- Heatmap overlays on CT images using Grad-CAM
- A standalone horizontal colorbar image
- Fully reproducible outputs using PyTorch and matplotlib

---

## Key Features

- Grad-CAM implementation from scratch (no third-party explainability libs)
- Works entirely on CPU (no CUDA required)
- Compatible with pretrained or custom-trained models
- Saves publication-ready overlay images (300 DPI)
- Generates a separate colorbar for UI / paper figures
- Clean, minimal, research-friendly structure

---

## Dependencies

- Python 3.10+
- torch
- torchvision
- pillow
- numpy
- matplotlib

---

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies (CPU-only PyTorch):

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow matplotlib numpy
```

---

## Running the Script

Activate the virtual environment:

```bash
source venv/bin/activate
```

Run the Grad-CAM pipeline:

```bash
python gradcam_overlay.py
```

---

## What the Script Does

1. Loads a CT image from disk
2. Preprocesses the image using ImageNet normalization
3. Passes the image through a pretrained ResNet18 model
4. Computes Grad-CAM using gradients from the final convolutional layer
5. Resizes the heatmap to match the original image
6. Overlays the heatmap on the original CT image
7. Saves:
   - Heatmap overlay image
   - Standalone horizontal colorbar image

---

## Configuration

Edit these variables in `gradcam_overlay.py`:

```python
image_path = 'images/ct_synthetic.png'
overlay_out = 'images/ui_mockup_overlay.png'
colorbar_out = 'images/heatmap_colorbar.png'
input_size = (224, 224)
```

To change the Grad-CAM target layer:

```python
target_layer = model.layer4[-1].conv2
```

---

## Project Structure

```
CT-Scan-Triage/
├── gradcam_overlay.py
├── images/
│   ├── ct_synthetic.png
│   ├── ui_mockup_overlay.png
│   └── heatmap_colorbar.png
├── README.md
├── requirements.txt
└── venv/
```

---

## Project Status

This repository represents an **early-stage research scaffold**, not a finished medical AI system.
The current implementation focuses on demonstrating feasibility, visualization capability, and
overall pipeline direction.

The code is intentionally lightweight and modular to allow rapid iteration.

---

## What This Is (Right Now)

- A **proof-of-concept** CT-image triage pipeline
- A **baseline CNN + Grad-CAM explainability demo**
- A starting point for research, not a clinical tool

---

## What This Is NOT (Yet)

- A clinically validated diagnostic system
- A deployable hospital-grade AI
- A final research contribution

---

## Main TODO Roadmap

### Phase 1 — Data & Problem Definition (Highest Priority)
- Define triage labels (binary vs multi-class)
- Select and document CT datasets
- Establish clinical motivation and decision boundaries
- Address class imbalance and data bias

### Phase 2 — Model Training & Evaluation
- Replace ImageNet-only baselines with task-trained models
- Perform cross-validation
- Evaluate calibration and uncertainty
- Compare multiple architectures

### Phase 3 — Explainability Validation
- Quantitatively assess Grad-CAM heatmaps
- Compare explainability methods
- Analyze failure and misleading explanations

### Phase 4 — Triage Pipeline Design
- Implement multi-stage triage logic
- Confidence-based case routing
- Latency and efficiency analysis
- Risk-aware decision thresholds

---

## Research Direction

The long-term goal is to develop a **lightweight, explainable, and risk-aware AI-assisted triage
pipeline** for medical imaging that prioritizes safety, transparency, and clinical relevance.

Future iterations will expand the pipeline with stronger datasets, validation, and domain expertise.
"""


## License

MIT License

Copyright (c) 2026 Tobi Joshua
