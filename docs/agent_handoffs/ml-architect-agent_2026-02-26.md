# ml-architect-agent Handoff — 2026-02-26

## Completed This Session

- `src/models/encoders.py` — FluidBiomarkerEncoder, DigitalAcousticEncoder, DigitalMotorEncoder, ClinicalDemographicEncoder; all output [batch, 768]; input validation per SRS-001 §4.2; Google-style docstrings; structlog logging; no PHI
- `tests/unit/test_encoders.py` — 35 unit tests across 5 test classes; all 35 pass; pure torch synthetic data (no numpy, avoids NumPy 2.x/torch ABI issue on this machine)
- `configs/model_config.yaml` — model architecture config (embed_dim, num_heads, gnn_layers, dropout, similarity_threshold), encoder dims, training hyperparameters, performance targets
- `src/models/__init__.py` — exports all 4 encoders from src.models
- `src/__init__.py` — package root init
- `tests/__init__.py` — test package init
- `tests/unit/__init__.py` — unit test package init

## Decisions Made (with rationale)

- **All encoders use Linear→LayerNorm→GELU→Dropout(0.1) blocks**: LayerNorm at each projection block stabilises gradient flow; GELU is preferred over ReLU for transformer-adjacent architectures (consistent with BERT-style embeddings)
- **Output dim 768 matches BERT-base embedding size**: enables potential future cross-architecture fine-tuning (e.g., ClinicalBERT pre-training on discharge summaries)
- **Input validation at forward() time, not __init__()**: model serialisation (torch.save/load) must succeed before any data is available; deferring validation to inference time is the correct IEC 62304-compliant approach
- **Shared `_build_encoder_layers()` factory**: identical MLP topology across all 4 encoders enforces architectural uniformity per SAD-001 §5.1; single point of change if topology changes in Phase 2
- **Shared `_validate_features()` helper**: consolidates range-checking logic; reduces duplication and ensures consistent error messages referencing feature names and offending values
- **Pure torch in test data factories**: the machine has NumPy 2.x installed while torch was compiled against NumPy 1.x; `torch.from_numpy()` raises RuntimeError; using `torch.tensor()`/`torch.zeros()` avoids the incompatibility without requiring environment changes
- **`structlog` installed as a runtime dependency**: not present in the environment by default; `pip install structlog` was run; the requirements.txt should be updated to pin `structlog>=23.0.0`

## Current State

- Working: All 4 encoders (FluidBiomarkerEncoder, DigitalAcousticEncoder, DigitalMotorEncoder, ClinicalDemographicEncoder), 35/35 unit tests passing
- Blocked: CrossModalAttention and NeuroFusionGNN are not yet implemented

## Next Session Must Start With

1. Implement `src/models/cross_modal_attention.py` — CrossModalAttention(embed_dim=768, num_heads=8) using `torch.nn.MultiheadAttention`; fluid biomarker embedding is the query anchor; unit tests in `tests/unit/test_cross_modal_attention.py`
2. Implement `src/models/gnn.py` — NeuroFusionGNN using PyTorch Geometric GraphSAGE (3 layers, hidden_dim=768, cosine similarity graph, threshold=0.7); unit tests in `tests/unit/test_gnn.py`
3. Implement `src/models/neurofusion_model.py` — top-level NeuroFusionAD model wiring all 4 encoders → CrossModalAttention → GNN → 3 output heads (BCEWithLogits, MSE, Cox); end-to-end forward pass test

## Open Questions for Human Review

- `structlog` is not in `requirements.txt` — should it be added with a pinned version? Recommend `structlog>=23.0.0`
- NumPy 2.x is installed but torch was compiled against NumPy 1.x — this prints a UserWarning on every pytest run. Consider pinning `numpy<2` or upgrading torch to a NumPy 2.x-compatible build
- `tests/integration/` directory exists but is empty — should integration test stubs be created alongside unit tests, or left for the data-engineer-agent and api-agent to populate?

---

## Session 2 — Integration

### Completed
- `src/models/neurofusion_model.py` — NeuroFusionAD full model (12,760,835 params; all 4 encoders + CrossModalAttention + 3-layer GraphSAGE GNN + 3 output heads)
- `tests/unit/test_neurofusion_model.py` — 12 unit tests (all passing): instantiation, output keys, amyloid shape, mmse slope shape, cox hazard shape, fused embedding shape, no-NaN, disclaimer exact match, parameter count > 10M, eval mode, missing key KeyError, batch sizes 1/4/16
- `scripts/sanity_check_e2e.py` — e2e sanity check: batch_size=16 synthetic data, all shape assertions pass, no NaN, disclaimer verified
- `src/models/__init__.py` — updated to export NeuroFusionAD and CLINICAL_DISCLAIMER
- `docs/PHASE1_CHECKLIST.md` — all model implementation + data pipeline items ticked [x]; status updated to MODEL IMPLEMENTATION COMPLETE

### Decisions Made (with rationale)

- **torch_geometric 2.5.0 installed** (not 2.7.0): 2.7.0 introduced a dependency on pandas which triggers a NumPy ABI incompatibility (dtype size mismatch) in this environment. 2.5.0 avoids the pandas import path and is the version specified in CLAUDE.md
- **graph_threshold stored as instance attribute**: allows the threshold to be configurable at test time (e.g., for edge-case testing with batch_size=1 where all nodes may be isolated)
- **KeyError raised directly for missing batch keys**: more informative than letting the downstream encoder raise an AttributeError; IEC 62304 compliant error message names the missing key

### All ML implementation tests passing

Run: `pytest tests/unit/ -v` — 89 passed, 0 failed

### Remaining Phase 1 work (NOT this agent's ownership)

- Regulatory docs: SRS, SAD, RMF, SDP, regulatory strategy, data requirements (`regulatory-agent`)
- Gate review checklist (`regulatory-agent`)
- Traceability matrix (`regulatory-agent`)
- DHF compilation (`regulatory-agent`)
- README.md update (shared — coordinate before editing)
- requirements.txt update (shared — coordinate before editing)
