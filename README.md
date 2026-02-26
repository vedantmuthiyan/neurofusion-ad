# NeuroFusion-AD

**Multimodal Graph Neural Network for Alzheimer's Disease Progression Prediction**

> ⚠️ **REGULATORY NOTICE**: This is an investigational Software as a Medical Device (SaMD) in development. Not approved for clinical use. All model outputs are for research purposes only.
>
> **Mandatory disclaimer**: *"This tool is intended to support, not replace, clinical judgment."*

---

## Overview

NeuroFusion-AD combines four multimodal data streams with cross-modal attention and graph-based patient similarity to predict Alzheimer's disease progression risk in patients aged 50–90 with Mild Cognitive Impairment.

- **Regulatory class**: SaMD — FDA De Novo + EU MDR Class IIa (in development)
- **Software lifecycle**: IEC 62304 Class B
- **Risk management**: ISO 14971
- **Current phase**: Phase 1 — Foundation, Requirements & Architecture

## Architecture

```
[Fluid Biomarkers (6)]  →  FluidBiomarkerEncoder   ─┐
[Acoustic Features (12)] →  DigitalAcousticEncoder  ─┤
[Motor Features (8)]    →  DigitalMotorEncoder      ─┤→ CrossModalAttention → NeuroFusionGNN → Output Heads
[Clinical/Demo (10)]    →  ClinicalDemographicEncoder─┘                         (GraphSAGE)
                                                                                     │
                                                              ┌──────────────────────┤
                                                              ▼                      ▼
                                                    Amyloid Logit           MMSE Slope + Cox Hazard
                                                    (Classification)        (Regression + Survival)
```

- **Encoders**: 4 modality-specific MLPs, all output 768-dim
- **Fusion**: CrossModalAttention (embed_dim=768, num_heads=8), fluid as query anchor
- **GNN**: 3-layer GraphSAGE (cosine similarity graph, threshold=0.7, hidden_dim=768)
- **Output heads**: BCEWithLogits (amyloid), MSE (MMSE slope), Cox (survival)
- **Parameters**: ~12.8M trainable

## Project Structure

```
neurofusion-ad/
├── src/
│   ├── models/
│   │   ├── encoders.py               # 4 modality encoders
│   │   ├── cross_modal_attention.py  # CrossModalAttention fusion
│   │   ├── gnn.py                    # NeuroFusionGNN + graph construction
│   │   └── neurofusion_model.py      # Full integrated model
│   └── data/
│       ├── validators.py             # InputValidator (clinical range checks)
│       ├── adni_preprocessing.py     # ADNIPreprocessor
│       ├── digital_biomarker_synthesis.py  # Synthetic data generation
│       └── dataset.py               # NeuroFusionDataset + DataLoaders
├── tests/unit/                       # 89 unit tests (all passing)
├── configs/model_config.yaml         # Architecture hyperparameters
├── scripts/sanity_check_e2e.py       # End-to-end forward pass check
└── docs/
    ├── regulatory/                   # SRS, SAD, RMF, SDP, REG, DRD
    ├── dhf/                          # Design History File
    └── agent_handoffs/               # Inter-session agent memory
```

## Quick Start

### Prerequisites

- Python 3.10
- CUDA 11.8 (optional — CPU inference supported)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd neurofusion-ad

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
# IMPORTANT: Pin numpy to <2.0 for torch 2.1.2 compatibility
pip install "numpy<2.0" torch==2.1.2 torch-geometric==2.5.0
pip install -r requirements.txt
```

### Run End-to-End Sanity Check

```bash
python scripts/sanity_check_e2e.py
```

Expected output:
```
=== NeuroFusion-AD End-to-End Sanity Check ===
Model instantiated. Parameters: 12,760,835
Dataset: 64 samples. Train batches: 3
✓ Shape assertions: PASSED
✓ NaN checks: PASSED
✓ Disclaimer present: PASSED
=== SANITY CHECK PASSED ===
```

### Run Unit Tests

```bash
pytest tests/unit/ -v
# Expected: 89 passed, 0 failed
```

### Run Tests with Coverage

```bash
pytest tests/unit/ --cov=src --cov-report=term-missing
```

## Model Usage

```python
import torch
from src.models.neurofusion_model import NeuroFusionAD

model = NeuroFusionAD()
model.eval()

# Prepare input batch (all values must be within validated ranges)
batch = {
    'fluid':    torch.tensor([[5.0, 0.10, 50.0, 150.0, 250.0, 800.0]]),  # [batch, 6]
    'acoustic': torch.tensor([[0.005, 0.05, 15.0, 130.0, 25.0, 0., 0., 0., 0., 0., 0., 0.]]),  # [batch, 12]
    'motor':    torch.randn(1, 8),   # [batch, 8]
    'clinical': torch.tensor([[72.0, 14.0, 0.0, 26.0, 1.5, 5.0, 27.0, 130.0, 0.0, 2.0]]),  # [batch, 10]
}

with torch.no_grad():
    outputs = model(batch)

print(f"Amyloid logit: {outputs['amyloid_logit'].item():.3f}")
print(f"MMSE slope:    {outputs['mmse_slope'].item():.3f} pts/year")
print(f"Cox log-hazard:{outputs['cox_log_hazard'].item():.3f}")
print(f"Disclaimer:    {outputs['disclaimer']}")
```

## Validated Input Ranges

| Feature | Range | Units |
|---------|-------|-------|
| pTau-217 | 0.1 – 100 | pg/mL |
| Abeta42/40 ratio | 0.01 – 0.30 | dimensionless |
| NfL | 5 – 200 | pg/mL |
| Acoustic jitter | 0.0001 – 0.05 | dimensionless |
| Acoustic shimmer | 0.001 – 0.30 | dimensionless |
| MMSE | 0 – 30 | score |

Values outside these ranges raise `ValueError` at inference time.

## Performance Targets (Phase 2 validation)

| Metric | Target |
|--------|--------|
| Classification AUC | ≥ 0.85 |
| Regression RMSE | ≤ 3.0 pts/year |
| Survival C-index | ≥ 0.75 |
| Inference latency p95 | < 2.0 seconds |

## Data Access

This repository contains **no real patient data**. Tests use synthetic data only.

Real data sources require institutional access:
- **ADNI**: Data Use Agreement required — [adni.loni.usc.edu](https://adni.loni.usc.edu)
- **Bio-Hermes**: Access via [globalalzplatform.org](https://globalalzplatform.org)
- **DementiaBank**: Contact [dementia.talkbank.org](https://dementia.talkbank.org)

All raw data belongs in `data/raw/` (git-ignored).

## Security & Privacy

- Patient IDs are SHA-256 hashed before any logging
- No PHI in any log, file, or commit
- Encryption at rest: AES-256 (deployment requirement)
- Encryption in transit: TLS 1.3 minimum
- Audit trail: every prediction logged to PostgreSQL with hashed patient ID

## Known Issues (Phase 1)

1. **NumPy ABI incompatibility**: `torch==2.1.2` was compiled against NumPy 1.x. Running with NumPy 2.x prints a `UserWarning` but all code uses `.tolist()` workaround. **Pin `numpy<2.0` in production.**
2. **No real data**: All validation uses synthetic data. Real ADNI validation is Phase 2.
3. **CPU only in CI**: GPU inference not tested in Phase 1.

## Phase 1 Status

See `docs/PHASE1_CHECKLIST.md` for the full exit checklist.

**Completed**: All model implementation + data pipeline + regulatory documentation (draft)
**Pending human review**: SRS peer review, SAD technical review, risk acceptance decisions

## License

Proprietary. All rights reserved. For research and regulatory development purposes only.

## Contact

For regulatory questions, contact the Regulatory Affairs Lead.
For technical questions, contact the ML Architecture Lead.
