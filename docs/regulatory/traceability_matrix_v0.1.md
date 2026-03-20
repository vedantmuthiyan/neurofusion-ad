# Traceability Matrix v0.1

**Document ID**: SRS-001 / TM-001
**Status**: Final
**Date**: 2026-02-26
**IEC 62304 Clause**: 5.2.6 (Software System Testing)

This matrix traces each software requirement from SRS-001 to its architectural design element (SAD-001),
source code implementation, and verification test.

---

## 1. Purpose

This Traceability Matrix (TM-001) provides bidirectional traceability between:

- Software requirements defined in SRS-001
- Architectural design decisions documented in SAD-001
- Source code implementation in `src/`
- Verification tests in `tests/`

Bidirectional traceability is a mandatory IEC 62304 § 5.2.6 artifact and supports FDA De Novo submission
under 21 CFR Part 820 (Design Controls). Every requirement must be traceable from its origin (user need)
through design, implementation, and verification.

---

## 2. Scope

This matrix covers Phase 1 requirements only. It will be updated in subsequent phases as new requirements
are identified during detailed design (Phase 2) and integration testing (Phase 3).

**Requirement categories covered:**

| Category | Code | Description |
|----------|------|-------------|
| Functional — Input | FRI | Requirements governing system inputs and input validation |
| Functional — Processing | FRP | Requirements governing internal data processing and model inference |
| Functional — Output | FRO | Requirements governing system outputs and output formatting |
| Non-Functional | NFR | Performance, security, privacy, and operational requirements |

---

## 3. Traceability Matrix

### 3.1 Functional Requirements — Input (FRI)

| Req ID | Requirement Summary | Priority | Design Element | Implementation | Test | Status |
|--------|---------------------|----------|----------------|----------------|------|--------|
| SRS-001-FRI-001 | System shall accept fluid biomarker input: pTau-217, Abeta42/40 ratio, NfL, GFAP, total-tau, Abeta42 as a 6-element floating-point vector | Essential | SAD-001 § 5.1.1 — FluidBiomarkerEncoder input specification | `src/models/encoders.py` — `FluidBiomarkerEncoder.__init__`, `FluidBiomarkerEncoder.forward` | `tests/unit/test_encoders.py` — `test_fluid_encoder_accepts_6_features`, `test_fluid_encoder_forward_shape` | Implemented |
| SRS-001-FRI-002 | System shall accept acoustic digital biomarker input: jitter, shimmer, HNR, F0 mean/std, MFCCs (12 features total) as a 12-element floating-point vector | Essential | SAD-001 § 5.1.2 — DigitalAcousticEncoder input specification | `src/models/encoders.py` — `DigitalAcousticEncoder.__init__`, `DigitalAcousticEncoder.forward` | `tests/unit/test_encoders.py` — `test_acoustic_encoder_accepts_12_features`, `test_acoustic_encoder_forward_shape` | Implemented |
| SRS-001-FRI-003 | System shall accept motor digital biomarker input: tremor frequency, bradykinesia index, spiral irregularity, tapping rate, tapping variability, grip force mean, grip force variability, gait cadence (8 features total) | Essential | SAD-001 § 5.1.3 — DigitalMotorEncoder input specification | `src/models/encoders.py` — `DigitalMotorEncoder.__init__`, `DigitalMotorEncoder.forward` | `tests/unit/test_encoders.py` — `test_motor_encoder_accepts_8_features`, `test_motor_encoder_forward_shape` | Implemented |
| SRS-001-FRI-004 | System shall accept clinical/demographic input: age, education years, sex, MMSE, CDR global, CDR sum-of-boxes, APOE4 allele count, years since symptom onset, comorbidity index, medication count (10 features total) | Essential | SAD-001 § 5.1.4 — ClinicalDemographicEncoder input specification | `src/models/encoders.py` — `ClinicalDemographicEncoder.__init__`, `ClinicalDemographicEncoder.forward` | `tests/unit/test_encoders.py` — `test_clinical_encoder_accepts_10_features`, `test_clinical_encoder_forward_shape` | Implemented |
| SRS-001-FRI-005 | System shall validate pTau-217 input is within clinically valid range [0.1, 100] pg/mL; inputs outside this range shall be rejected with a structured error before inference | Essential | SAD-001 § 5.5 — InputValidator specification | `src/models/encoders.py` — `InputValidator.validate_fluid_biomarkers` | `tests/unit/test_encoders.py` — `test_validator_rejects_ptau217_below_min`, `test_validator_rejects_ptau217_above_max`, `test_validator_accepts_valid_ptau217` | Implemented |
| SRS-001-FRI-006 | System shall validate Abeta42/40 ratio input is within clinically valid range [0.01, 0.30]; inputs outside this range shall be rejected with a structured error before inference | Essential | SAD-001 § 5.5 — InputValidator specification | `src/models/encoders.py` — `InputValidator.validate_fluid_biomarkers` | `tests/unit/test_encoders.py` — `test_validator_rejects_abeta_ratio_below_min`, `test_validator_rejects_abeta_ratio_above_max`, `test_validator_accepts_valid_abeta_ratio` | Implemented |
| SRS-001-FRI-007 | System shall validate NfL input is within clinically valid range [5, 200] pg/mL; inputs outside this range shall be rejected with a structured error before inference | Essential | SAD-001 § 5.5 — InputValidator specification | `src/models/encoders.py` — `InputValidator.validate_fluid_biomarkers` | `tests/unit/test_encoders.py` — `test_validator_rejects_nfl_below_min`, `test_validator_rejects_nfl_above_max`, `test_validator_accepts_valid_nfl` | Implemented |
| SRS-001-FRI-008 | System shall validate acoustic jitter input is within valid range [0.0001, 0.05]; inputs outside this range shall be rejected with a structured error before inference | Essential | SAD-001 § 5.5 — InputValidator specification | `src/models/encoders.py` — `InputValidator.validate_acoustic_features` | `tests/unit/test_encoders.py` — `test_validator_rejects_jitter_below_min`, `test_validator_rejects_jitter_above_max`, `test_validator_accepts_valid_jitter` | Implemented |
| SRS-001-FRI-009 | System shall validate acoustic shimmer input is within valid range [0.001, 0.3]; inputs outside this range shall be rejected with a structured error before inference | Essential | SAD-001 § 5.5 — InputValidator specification | `src/models/encoders.py` — `InputValidator.validate_acoustic_features` | `tests/unit/test_encoders.py` — `test_validator_rejects_shimmer_below_min`, `test_validator_rejects_shimmer_above_max`, `test_validator_accepts_valid_shimmer` | Implemented |
| SRS-001-FRI-010 | System shall validate MMSE score input is within valid range [0, 30] (integer or float); inputs outside this range shall be rejected with a structured error before inference | Essential | SAD-001 § 5.5 — InputValidator specification | `src/models/encoders.py` — `InputValidator.validate_clinical_features` | `tests/unit/test_encoders.py` — `test_validator_rejects_mmse_below_min`, `test_validator_rejects_mmse_above_max`, `test_validator_accepts_valid_mmse` | Implemented |

### 3.2 Functional Requirements — Processing (FRP)

| Req ID | Requirement Summary | Priority | Design Element | Implementation | Test | Status |
|--------|---------------------|----------|----------------|----------------|------|--------|
| SRS-001-FRP-001 | System shall encode fluid biomarker input vectors to a 768-dimensional embedding space using a learned neural encoder; output shape shall be (batch_size, 768) | Essential | SAD-001 § 5.2.1 — FluidBiomarkerEncoder architecture (MLP: 6 → 128 → 256 → 512 → 768, LayerNorm, GELU) | `src/models/encoders.py` — `FluidBiomarkerEncoder.forward` | `tests/unit/test_encoders.py` — `test_fluid_encoder_output_dim_768`, `test_fluid_encoder_output_batch_dim` | Implemented |
| SRS-001-FRP-002 | System shall encode acoustic digital biomarker feature vectors to a 768-dimensional embedding space using a learned neural encoder; output shape shall be (batch_size, 768) | Essential | SAD-001 § 5.2.2 — DigitalAcousticEncoder architecture (MLP: 12 → 128 → 256 → 512 → 768, LayerNorm, GELU) | `src/models/encoders.py` — `DigitalAcousticEncoder.forward` | `tests/unit/test_encoders.py` — `test_acoustic_encoder_output_dim_768`, `test_acoustic_encoder_output_batch_dim` | Implemented |
| SRS-001-FRP-003 | System shall encode motor digital biomarker feature vectors to a 768-dimensional embedding space using a learned neural encoder; output shape shall be (batch_size, 768) | Essential | SAD-001 § 5.2.3 — DigitalMotorEncoder architecture (MLP: 8 → 128 → 256 → 512 → 768, LayerNorm, GELU) | `src/models/encoders.py` — `DigitalMotorEncoder.forward` | `tests/unit/test_encoders.py` — `test_motor_encoder_output_dim_768`, `test_motor_encoder_output_batch_dim` | Implemented |
| SRS-001-FRP-004 | System shall encode clinical/demographic feature vectors to a 768-dimensional embedding space using a learned neural encoder; output shape shall be (batch_size, 768) | Essential | SAD-001 § 5.2.4 — ClinicalDemographicEncoder architecture (MLP: 10 → 128 → 256 → 512 → 768, LayerNorm, GELU) | `src/models/encoders.py` — `ClinicalDemographicEncoder.forward` | `tests/unit/test_encoders.py` — `test_clinical_encoder_output_dim_768`, `test_clinical_encoder_output_batch_dim` | Implemented |
| SRS-001-FRP-005 | System shall fuse the four 768-dimensional modality embeddings via cross-modal attention (embed_dim=768, num_heads=8) using fluid biomarker embedding as the query anchor; fused output shall be (batch_size, 768) | Essential | SAD-001 § 5.2 — CrossModalAttention module (fluid as query, acoustic/motor/clinical as keys/values, residual connections, LayerNorm) | `src/models/cross_modal_attention.py` — `CrossModalAttention.forward` | `tests/unit/test_cross_modal_attention.py` — `test_cross_modal_attention_output_shape`, `test_cross_modal_attention_fluid_as_query`, `test_cross_modal_attention_residual` | Implemented |
| SRS-001-FRP-006 | System shall construct a patient similarity graph by computing pairwise cosine similarity between fused embeddings and connecting patients whose cosine similarity exceeds threshold 0.7; graph shall be represented as edge_index tensor | Essential | SAD-001 § 5.3 — construct_patient_similarity_graph (cosine sim matrix, threshold=0.7, self-loops excluded) | `src/models/gnn.py` — `construct_patient_similarity_graph` | `tests/unit/test_gnn.py` — `test_graph_construction_threshold`, `test_graph_construction_no_self_loops`, `test_graph_construction_cosine_similarity` | Implemented |
| SRS-001-FRP-007 | System shall aggregate neighborhood context via a 3-layer GraphSAGE network (hidden_dim=768, mean aggregation) to produce graph-contextualized patient embeddings; output shape shall be (num_patients, 768) | Essential | SAD-001 § 5.3 — NeuroFusionGNN (3 SAGEConv layers, hidden_dim=768, ReLU activations, dropout=0.1) | `src/models/gnn.py` — `NeuroFusionGNN.forward` | `tests/unit/test_gnn.py` — `test_gnn_output_shape`, `test_gnn_three_layers`, `test_gnn_neighborhood_aggregation` | Implemented |

### 3.3 Functional Requirements — Output (FRO)

| Req ID | Requirement Summary | Priority | Design Element | Implementation | Test | Status |
|--------|---------------------|----------|----------------|----------------|------|--------|
| SRS-001-FRO-001 | System shall output an amyloid positivity logit (unbounded scalar) from a binary classification head; this logit shall be used with BCEWithLogitsLoss during training and converted to probability via sigmoid during inference | Essential | SAD-001 § 5.4 — NeuroFusionAD classification head (Linear 768→1, BCEWithLogits) | `src/models/neurofusion_model.py` — `NeuroFusionAD.forward` (classification_logit output key) | `tests/unit/test_neurofusion_model.py` — `test_model_output_classification_logit`, `test_model_classification_head_shape` | Implemented |
| SRS-001-FRO-002 | System shall output an MMSE slope prediction (points/year, unbounded scalar) from a regression head trained with MSE loss; output represents predicted rate of cognitive decline | Essential | SAD-001 § 5.4 — NeuroFusionAD regression head (Linear 768→1, MSE loss during training) | `src/models/neurofusion_model.py` — `NeuroFusionAD.forward` (mmse_slope output key) | `tests/unit/test_neurofusion_model.py` — `test_model_output_mmse_slope`, `test_model_regression_head_shape` | Implemented |
| SRS-001-FRO-003 | System shall output a Cox proportional hazards log-hazard score (unbounded scalar) from a survival analysis head; output is used to stratify patients by progression risk | Essential | SAD-001 § 5.4 — NeuroFusionAD survival head (Linear 768→1, Cox partial likelihood during training) | `src/models/neurofusion_model.py` — `NeuroFusionAD.forward` (cox_log_hazard output key) | `tests/unit/test_neurofusion_model.py` — `test_model_output_cox_log_hazard`, `test_model_survival_head_shape` | Implemented |
| SRS-001-FRO-004 | System shall include the mandatory disclaimer "This tool is intended to support, not replace, clinical judgment." on every output payload; outputs without this disclaimer shall be considered non-compliant | Essential | SAD-001 § 6.2 — API output schema (FHIR RiskAssessment resource, disclaimer field required) | `src/api/` — disclaimer field in FHIR RiskAssessment response schema (Phase 2 implementation) | `tests/unit/` — test_disclaimer_present_in_output (Phase 2) | Pending |

### 3.4 Non-Functional Requirements (NFR)

| Req ID | Requirement Summary | Priority | Design Element | Implementation | Test | Status |
|--------|---------------------|----------|----------------|----------------|------|--------|
| SRS-001-NFR-001 | System inference latency shall not exceed 2.0 seconds at the 95th percentile on standard clinical workstation CPU hardware (Intel Xeon or equivalent); this includes input validation, full forward pass, and output serialization | Essential | SAD-001 § 7.1 — Performance requirements; architecture choices (no conv layers > 3, batch inference) | `src/models/neurofusion_model.py` — `NeuroFusionAD.forward` (full pipeline) | `tests/unit/test_neurofusion_model.py` — `test_inference_latency_p95_cpu` | Partial |
| SRS-001-NFR-002 | No Protected Health Information (PHI) shall be written to any log, file, or external service; patient identifiers shall be SHA-256 hashed before any logging operation | Essential | SAD-001 § 7.3 — Privacy requirements; logging specification (structlog, JSON, hashed IDs) | `src/utils/` — logging utilities with hashlib.sha256 (Phase 2 implementation); `src/data/` — no PHI logging | `tests/unit/` — test_no_phi_in_logs (Phase 2) | Partial |
| SRS-001-NFR-003 | All input parameters shall be validated against defined clinical ranges before any model inference is executed; out-of-range inputs shall be rejected with HTTP 422 and a structured error response identifying the offending parameter | Essential | SAD-001 § 5.5 — InputValidator (pre-inference gate, ValueError with parameter name and range) | `src/models/encoders.py` — `InputValidator` class (all validate_* methods) | `tests/unit/test_encoders.py` — `test_validator_rejects_before_inference`, `test_validator_error_message_contains_parameter_name` | Implemented |
| SRS-001-NFR-004 | System shall support batch inference for up to 64 patients simultaneously without error; batch dimension shall be propagated correctly through all model components including graph construction | High | SAD-001 § 5.3 — Graph construction supports batch; § 5.4 — NeuroFusionAD supports variable batch_size | `src/models/neurofusion_model.py` — `NeuroFusionAD.forward` (batch_size parameter handling); `src/models/gnn.py` — `construct_patient_similarity_graph` (N×N cosine sim) | `tests/unit/test_neurofusion_model.py` — `test_batch_inference_size_1`, `test_batch_inference_size_16`, `test_batch_inference_size_64` | Implemented |

---

## 4. Coverage Summary

| Category | Total Requirements | Implemented | Partial | Pending | Coverage % |
|----------|--------------------|-------------|---------|---------|------------|
| FRI (Input) | 10 | 10 | 0 | 0 | 100% |
| FRP (Processing) | 7 | 7 | 0 | 0 | 100% |
| FRO (Output) | 4 | 3 | 0 | 1 | 75% |
| NFR (Non-Functional) | 4 | 2 | 2 | 0 | 50–100% |
| **Total** | **25** | **22** | **2** | **1** | **88%** |

**Notes on Partial/Pending items:**

- **SRS-001-FRO-004** (Disclaimer): The disclaimer requirement is architecturally specified in SAD-001 § 6.2. Full implementation is deferred to Phase 2 API development. The requirement is fully designed and traceable.
- **SRS-001-NFR-001** (Latency): The `test_inference_latency_p95_cpu` test exists as a benchmark test but is marked `@pytest.mark.slow` and excluded from the standard CI run. Full verification requires hardware-specific benchmarking on the target clinical workstation.
- **SRS-001-NFR-002** (No PHI in logs): PHI protection is partially implemented in `src/data/` (no patient IDs in preprocessing). Full logging utility with SHA-256 hashing is deferred to Phase 2 (`src/utils/`).

---

## 5. Bidirectional Traceability Index

This section provides the reverse mapping: for each implementation artifact, which requirements does it satisfy?

| Implementation Artifact | Satisfies Requirements |
|-------------------------|------------------------|
| `src/models/encoders.py` — `FluidBiomarkerEncoder` | SRS-001-FRI-001, SRS-001-FRP-001 |
| `src/models/encoders.py` — `DigitalAcousticEncoder` | SRS-001-FRI-002, SRS-001-FRP-002 |
| `src/models/encoders.py` — `DigitalMotorEncoder` | SRS-001-FRI-003, SRS-001-FRP-003 |
| `src/models/encoders.py` — `ClinicalDemographicEncoder` | SRS-001-FRI-004, SRS-001-FRP-004 |
| `src/models/encoders.py` — `InputValidator` | SRS-001-FRI-005, SRS-001-FRI-006, SRS-001-FRI-007, SRS-001-FRI-008, SRS-001-FRI-009, SRS-001-FRI-010, SRS-001-NFR-003 |
| `src/models/cross_modal_attention.py` — `CrossModalAttention` | SRS-001-FRP-005 |
| `src/models/gnn.py` — `construct_patient_similarity_graph` | SRS-001-FRP-006 |
| `src/models/gnn.py` — `NeuroFusionGNN` | SRS-001-FRP-007 |
| `src/models/neurofusion_model.py` — `NeuroFusionAD` | SRS-001-FRO-001, SRS-001-FRO-002, SRS-001-FRO-003, SRS-001-NFR-001, SRS-001-NFR-004 |

---

## 6. Open Items and Actions

| ID | Issue | Owner | Target Resolution |
|----|-------|-------|-------------------|
| TM-OI-001 | SRS-001-FRO-004 (Disclaimer) — implement in API layer | api-agent | Phase 2 Sprint 1 |
| TM-OI-002 | SRS-001-NFR-001 (Latency) — run p95 benchmark on target hardware | devops-agent | Phase 2 Sprint 2 |
| TM-OI-003 | SRS-001-NFR-002 (PHI logging) — implement SHA-256 logging utility | api-agent | Phase 2 Sprint 1 |
| TM-OI-004 | Update this matrix after Phase 2 implementation completes | regulatory-agent | Phase 2 completion |

---

## 7. Approvals

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Regulatory Affairs Lead | [To be completed] | | |
| Software Architect | [To be completed] | | |
| Quality Assurance | [To be completed] | | |

---

*Document ID: TM-001 | Version: 0.1 | Status: Final | IEC 62304 § 5.2.6*
