# NeuroFusion-AD — Technical Specification

**Version**: 1.0.0
**Status**: Production Ready
**Last Updated**: March 2026

## Project Overview

NeuroFusion-AD is a multimodal Graph Neural Network (GNN) for Alzheimer's Disease
Progression Prediction. It combines fluid biomarkers, digital speech/gait features,
and clinical demographics through a cross-modal attention fusion architecture to
simultaneously predict amyloid positivity, MMSE trajectory, and survival risk.

## Architecture

### Model Architecture (v1.0)
| Component | Specification |
|-----------|--------------|
| Fluid encoder | MLP (input_dim=2: pTau181, NfL) → 256-dim |
| Acoustic encoder | MLP (input_dim=15) → 256-dim |
| Motor encoder | MLP (input_dim=20) → 256-dim |
| Clinical encoder | Embedding + MLP → 256-dim |
| Fusion | 4-head Cross-Modal Attention |
| Graph | 2-layer GraphSAGE GNN |
| Total parameters | 2,244,611 |
| Dropout | 0.40 |
| Temperature (calibration) | 0.756 |

### Multi-Task Outputs
1. **Classification**: Amyloid positivity probability (BCEWithLogitsLoss)
2. **Regression**: MMSE slope in points/year (MSELoss, masked for NaN)
3. **Survival**: Cox risk score for progression to dementia (Cox partial likelihood)

## Datasets

| Dataset | N | Split | Used for |
|---------|---|-------|---------|
| ADNI | 494 MCI patients | 345/74/75 train/val/test | All 3 tasks |
| Bio-Hermes-001 | 945 participants | 662/141/142 | Classification only |

### Key Data Notes
- ADNI fluid features: CSF pTau181 + NfL (proxy for plasma pTau217)
- ABETA42_CSF is NOT a model input (it defines the label — removed to prevent data leakage)
- ADNI acoustic/motor features: synthesized from clinical distributions
- Bio-Hermes-001: real plasma pTau217 (Roche Elecsys assay) + real NfL

## Validated Performance (v1.0, March 2026)

### ADNI Internal Test Set (N=75, N_labeled=44)
| Metric | Value | 95% CI |
|--------|-------|--------|
| AUC (ROC) | 0.890 | 0.790–0.990 |
| Sensitivity | 79.3% | — |
| Specificity | 93.3% | — |
| PPV | 95.8% | — |
| NPV | 70.0% | — |
| F1 | 0.868 | — |
| MMSE RMSE | 1.804 pts/yr | — |
| Survival C-index | 0.651 | — |
| ECE (calibrated) | 0.083 | — |

### Bio-Hermes-001 External Test Set (N=142)
| Metric | Value | 95% CI |
|--------|-------|--------|
| AUC (ROC) | 0.907 | 0.860–0.950 |
| Sensitivity | 90.2% | — |
| Specificity | 87.9% | — |

## API

- Endpoint: `POST /fhir/RiskAssessment/$process`
- Input: FHIR R4 Parameters bundle
- Output: FHIR R4 RiskAssessment resource
- Latency: p95 = 125ms (measured on RTX 3090)
- Authentication: OAuth 2.0 (client credentials)

## Training Configuration (final)

```yaml
embed_dim: 256
num_gnn_layers: 2
num_attention_heads: 4
dropout: 0.4
weight_decay: 1.0e-3
batch_size: 32
gradient_accumulation_steps: 4
early_stopping_patience: 30
```

## Regulatory Status

- Regulatory pathway: FDA De Novo + EU MDR Class IIa
- Software class: IEC 62304 Class B
- Risk management: ISO 14971
- Development lifecycle: IEC 62304 compliant
