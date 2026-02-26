---
document_id: srs-section-5-8
generated: 2026-02-26T22:55:20.879361
batch_id: msgbatch_01DTMbBbcyvTviGxwBhePxKr
status: DRAFT — requires human review before approval
---

# NeuroFusion-AD Software Requirements Specification
## SRS-001 v1.0 — Sections 5–8
### IEC 62304 Compliant | FDA De Novo | EU MDR Class IIa

---

**Document Control**

| Field | Value |
|---|---|
| Document ID | SRS-001 |
| Version | 1.0 |
| Status | DRAFT — For Internal Review |
| Sections | 5–8 (Continuation) |
| Author | Regulatory Affairs Office |
| Classification | Confidential — Regulatory Submission |
| Applicable Standards | IEC 62304:2006+A1:2015, ISO 14971:2019, IEC 82304-1:2016 |
| Related Documents | SRS-001 Sections 1–4, RMF-001, SDD-001, VVP-001 |

---

> **Regulatory Note:** This document constitutes a continuation of SRS-001 v1.0. All requirement identifiers are unique, traceable, and shall be mapped to corresponding test cases in the System Verification and Validation Plan (VVP-001) and to hazards in the Risk Management File (RMF-001) per ISO 14971:2019. Requirements designated **[SAFETY-CRITICAL]** trigger mandatory hazard analysis entries. Requirements designated **[REGULATORY-ANCHOR]** directly support a specific FDA De Novo or EU MDR submission artifact.

---

## Section 5: Functional Requirements — Model Inference (FRM-001 to FRM-020)

### 5.1 Overview and Scope

This section specifies functional requirements governing the NeuroFusion-AD inference pipeline from validated input receipt (post-Section 4 input validation) through multi-task output generation. The inference pipeline comprises four sequential subsystems:

1. **Modality Encoders** — Four parallel encoding pathways (fluid biomarkers, acoustic, motor, clinical/demographic)
2. **Cross-Modal Attention Mechanism** — 768-dimensional, 8-head transformer-style fusion layer
3. **GraphSAGE GNN Forward Pass** — 3-layer graph neural network for patient-cohort relational reasoning
4. **Multi-Task Output Heads** — Parallel classification, regression, and survival analysis branches

All inference requirements apply to the software item identified as `NEUROFUSION-INFERENCE-ENGINE v1.x` under IEC 62304 software item decomposition.

> **IEC 62304 Traceability:** All FRM requirements trace to Software System Test cases (prefix: SST-FRM-xxx) and Unit Test cases (prefix: UT-FRM-xxx) in VVP-001. SOUP dependencies (PyTorch 2.1.2, PyTorch Geometric 2.5.0) are documented in SOUP-REG-001.

---

### 5.2 Modality Encoder Requirements

---

#### FRM-001 — Fluid Biomarker Encoder: Input Tensor Formation

**Requirement ID:** FRM-001
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B — Safety-Related
**Regulatory Anchor:** FDA De Novo Special Controls — Algorithm Input Specification [REGULATORY-ANCHOR]

**Statement:**
The fluid biomarker encoder SHALL accept as input a validated floating-point tensor of shape `[batch_size, 4]` containing the following features in the prescribed column order:

| Index | Feature | Unit | Validated Range |
|---|---|---|---|
| 0 | pTau-217 | pg/mL | [0.1, 100.0] |
| 1 | Abeta42/40 ratio | dimensionless | [0.01, 0.30] |
| 2 | NfL | pg/mL | [5.0, 200.0] |
| 3 | GFAP | pg/mL | [20.0, 500.0] |

**Acceptance Criteria:**
- AC-FRM-001-1: The encoder SHALL reject any input tensor that has not been validated by the Input Validation Module (IVM) per Section 4 requirements; a validated timestamp token SHALL accompany the tensor.
- AC-FRM-001-2: Column ordering SHALL be enforced programmatically via a named tensor schema; positional ambiguity SHALL be a detectable fault condition.
- AC-FRM-001-3: The encoder SHALL produce an intermediate embedding of dimension 256 with `dtype=torch.float32`.
- AC-FRM-001-4: Missing or NaN values that survive validation SHALL cause encoder termination with fault code `ENC-FLUID-002`, logged to the audit trail.

**Test Reference:** SST-FRM-001, UT-FRM-001-A through UT-FRM-001-D
**Hazard Reference:** RMF-H-007 (Incorrect biomarker weighting leading to erroneous risk prediction)

---

#### FRM-002 — Acoustic Encoder: Speech Feature Processing

**Requirement ID:** FRM-002
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B

**Statement:**
The acoustic encoder SHALL process pre-extracted speech feature vectors derived from standardized cognitive assessment recordings (e.g., Cookie Theft picture description, verbal fluency tasks). The encoder SHALL accept input tensors of shape `[batch_size, sequence_length, 40]` representing 40-dimensional Mel-Frequency Cepstral Coefficient (MFCC) features with a maximum sequence length of 512 frames.

**Acceptance Criteria:**
- AC-FRM-002-1: The encoder SHALL implement a 3-layer bidirectional LSTM with hidden dimension 128, producing a fixed-length embedding of dimension 256 via mean pooling over valid (non-padded) time steps.
- AC-FRM-002-2: Padding masks SHALL be applied such that padded frames contribute zero weight to the pooled embedding; failure to apply padding masks SHALL be a detectable software fault.
- AC-FRM-002-3: The encoder SHALL handle variable-length sequences within the range [1, 512] frames; sequences of length 0 SHALL be rejected with fault code `ENC-ACOUSTIC-001`.
- AC-FRM-002-4: When acoustic data is absent and the modality is designated optional for a given patient, the encoder SHALL return a zero-initialized embedding of dimension 256 and set a modality-absent flag `ACOUSTIC_ABSENT=TRUE` propagated to the attention mechanism.
- AC-FRM-002-5: The acoustic encoder output embedding SHALL maintain `dtype=torch.float32` and SHALL be bounded in the range [-50.0, 50.0]; values outside this range SHALL trigger fault code `ENC-ACOUSTIC-003`. **[SAFETY-CRITICAL]**

**Test Reference:** SST-FRM-002, UT-FRM-002-A through UT-FRM-002-E
**Hazard Reference:** RMF-H-012 (Acoustic feature extraction errors propagating to risk score)

---

#### FRM-003 — Motor Encoder: Gait and Dexterity Feature Processing

**Requirement ID:** FRM-003
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B

**Statement:**
The motor encoder SHALL process quantitative motor assessment features including gait parameters (stride length, cadence, gait variability, dual-task cost) and fine motor dexterity metrics (finger tapping frequency, inter-tap interval coefficient of variation). The encoder SHALL accept input tensors of shape `[batch_size, 12]` representing 12 validated motor features.

**Acceptance Criteria:**
- AC-FRM-003-1: The encoder SHALL implement a 3-layer fully connected network with dimensions [12 → 64 → 128 → 256] and ReLU activations, producing a 256-dimensional embedding.
- AC-FRM-003-2: Batch normalization SHALL be applied after each hidden layer during inference; the encoder SHALL operate in `model.eval()` mode with batch normalization running statistics frozen from the training configuration.
- AC-FRM-003-3: When motor data is absent, behavior SHALL comply with AC-FRM-002-4 modality-absent protocol, setting flag `MOTOR_ABSENT=TRUE`.
- AC-FRM-003-4: The encoder SHALL apply feature-level standardization using training-set derived mean and standard deviation values stored in the model artifact; these statistics SHALL be version-locked to the model version identifier.

**Test Reference:** SST-FRM-003, UT-FRM-003-A through UT-FRM-003-D
**Hazard Reference:** RMF-H-015 (Motor feature normalization errors)

---

#### FRM-004 — Clinical and Demographic Encoder

**Requirement ID:** FRM-004
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B

**Statement:**
The clinical/demographic encoder SHALL process structured clinical variables including MMSE score, CDR-SB, age, sex, years of education, APOE4 allele count, comorbidity burden score, and current medications class count. The encoder SHALL accept input tensors of shape `[batch_size, 16]` after one-hot encoding of categorical variables.

**Acceptance Criteria:**
- AC-FRM-004-1: Categorical variables (sex, APOE4 status) SHALL be encoded using pre-defined one-hot encoding schemes stored in the model artifact's feature schema; the encoding scheme SHALL not be dynamically inferred at inference time.
- AC-FRM-004-2: The encoder SHALL produce a 256-dimensional embedding.
- AC-FRM-004-3: MMSE score SHALL be treated as an ordinal continuous variable in the range [0, 30]; values outside this range SHALL have been rejected by the IVM prior to encoder invocation; the encoder SHALL assert this precondition and raise `ENC-CLINICAL-001` if violated. **[SAFETY-CRITICAL]**
- AC-FRM-004-4: Age SHALL be constrained to [50, 90] years per intended use specification; out-of-range ages that reach the encoder SHALL be treated as a critical fault condition `ENC-CLINICAL-002` triggering safe-state transition. **[SAFETY-CRITICAL]**

**Test Reference:** SST-FRM-004, UT-FRM-004-A through UT-FRM-004-D
**Hazard Reference:** RMF-H-003 (Use outside intended patient population)

---

#### FRM-005 — Encoder Parallelism and Synchronization

**Requirement ID:** FRM-005
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B

**Statement:**
The four modality encoders (FRM-001 through FRM-004) SHALL execute in parallel where hardware resources permit, with their outputs synchronized before input to the cross-modal attention mechanism.

**Acceptance Criteria:**
- AC-FRM-005-1: The inference orchestrator SHALL join all four encoder output embeddings into a stacked tensor of shape `[batch_size, 4, 256]` before attention processing.
- AC-FRM-005-2: Encoder execution SHALL be deterministic for a given input; non-determinism arising from parallel execution (e.g., race conditions in GPU thread scheduling) SHALL be eliminated by setting `torch.use_deterministic_algorithms(True)` for inference operations.
- AC-FRM-005-3: If any mandatory encoder (fluid biomarker, clinical/demographic) fails, the entire inference request SHALL fail with a structured error response; optional encoder failures SHALL be handled per modality-absent protocol.
- AC-FRM-005-4: Maximum combined encoder execution time SHALL not exceed 500 milliseconds at p95 to preserve the overall 2.0-second latency budget.

**Test Reference:** SST-FRM-005, UT-FRM-005-A through UT-FRM-005-C
**Hazard Reference:** RMF-H-021 (Inference timeout causing incomplete output)

---

### 5.3 Cross-Modal Attention Mechanism Requirements

---

#### FRM-006 — Attention Architecture Specification

**Requirement ID:** FRM-006
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B
**Regulatory Anchor:** FDA De Novo — Algorithm Architecture Documentation [REGULATORY-ANCHOR]

**Statement:**
The cross-modal attention mechanism SHALL implement a multi-head self-attention layer with the following fixed architectural parameters that are immutable at inference time:

| Parameter | Value |
|---|---|
| Embedding dimension | 768 |
| Number of attention heads | 8 |
| Head dimension | 96 (768 / 8) |
| Input projection | Linear [256 → 768] per modality |
| Output dimension | 768 |
| Dropout rate (inference) | 0.0 (disabled) |

**Acceptance Criteria:**
- AC-FRM-006-1: Each modality embedding SHALL be individually projected from 256 to 768 dimensions via learned linear projections stored in the model artifact; projection weights SHALL be frozen (grad=False) during inference.
- AC-FRM-006-2: The scaled dot-product attention formula SHALL be implemented as: `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V` where `d_k = 96`.
- AC-FRM-006-3: The attention mechanism SHALL produce per-head, per-modality attention weight matrices of shape `[batch_size, 8, 4, 4]` that SHALL be preserved for explainability output per FRO requirements.
- AC-FRM-006-4: Dropout SHALL be explicitly disabled during inference (`model.eval()` mode confirmed); any inference execution with dropout active SHALL be a critical fault condition.
- AC-FRM-006-5: The attention output SHALL be a tensor of shape `[batch_size, 768]` after concatenation and linear projection of multi-head outputs.

**Test Reference:** SST-FRM-006, UT-FRM-006-A through UT-FRM-006-E
**Hazard Reference:** RMF-H-018 (Attention collapse leading to modality suppression)

---

#### FRM-007 — Attention Weight Extraction and Preservation

**Requirement ID:** FRM-007
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B — Explainability-Critical
**Regulatory Anchor:** EU MDR Annex I §17 — Transparency and Explainability [REGULATORY-ANCHOR]

**Statement:**
The cross-modal attention mechanism SHALL extract and preserve attention weight matrices at inference time for downstream explainability processing. Attention weights SHALL not be discarded after forward pass completion.

**Acceptance Criteria:**
- AC-FRM-007-1: Attention weight matrices of shape `[batch_size, 8, 4, 4]` SHALL be stored in an inference context object (`InferenceContext`) that persists for the duration of the request lifecycle.
- AC-FRM-007-2: Averaged attention weights (mean across 8 heads) of shape `[batch_size, 4, 4]` SHALL be computed and stored, representing cross-modal attention strength between modality pairs.
- AC-FRM-007-3: Per-modality attention importance scores SHALL be derived as the mean attention received by each modality across all heads and source positions, producing a vector of shape `[batch_size, 4]` with values summing to 1.0 ± 1e-5.
- AC-FRM-007-4: Attention weight extraction SHALL add no more than 10 milliseconds to inference latency at p95.
- AC-FRM-007-5: If attention weight extraction fails, the inference SHALL continue but SHALL set flag `ATTENTION_WEIGHTS_UNAVAILABLE=TRUE`; the output SHALL include an explainability degradation notice per FRO-010.

**Test Reference:** SST-FRM-007, UT-FRM-007-A through UT-FRM-007-D
**Hazard Reference:** RMF-H-025 (Clinician reliance on AI without explainability context)

---

#### FRM-008 — Attention Numerical Stability

**Requirement ID:** FRM-008
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B — Safety-Related **[SAFETY-CRITICAL]**

**Statement:**
The attention mechanism SHALL implement numerical stability safeguards to prevent NaN or infinity propagation in the output tensor.

**Acceptance Criteria:**
- AC-FRM-008-1: The softmax operation in scaled dot-product attention SHALL be computed in float32 precision with numerical stabilization via subtraction of the maximum value per row before exponentiation.
- AC-FRM-008-2: The system SHALL check for NaN or infinity values in the attention output tensor `[batch_size, 768]` before passing to the GNN; detection of NaN/Inf SHALL trigger inference termination with fault code `ATTN-NAN-001` and safe-state output. **[SAFETY-CRITICAL]**
- AC-FRM-008-3: Attention logits SHALL be clamped to the range [-100.0, 100.0] before softmax application to prevent overflow; the clamping operation SHALL be logged when activated.
- AC-FRM-008-4: Zero-division guards SHALL be implemented in all normalization operations within the attention computation.

**Test Reference:** SST-FRM-008, UT-FRM-008-A through UT-FRM-008-C
**Hazard Reference:** RMF-H-019 (NaN propagation leading to invalid risk score output)

---

### 5.4 GraphSAGE GNN Forward Pass Requirements

---

#### FRM-009 — GNN Architecture and Layer Specification

**Requirement ID:** FRM-009
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B
**Regulatory Anchor:** FDA De Novo — Novel Algorithm Justification [REGULATORY-ANCHOR]

**Statement:**
The GraphSAGE GNN SHALL implement a 3-layer graph neural network that operates over a patient similarity graph, aggregating information from clinically similar patient nodes to enrich individual patient representations. The GNN architecture SHALL conform to the Hamilton et al. (2017) GraphSAGE inductive learning framework as implemented in PyTorch Geometric 2.5.0 (`torch_geometric.nn.SAGEConv`).

| Layer | Input Dim | Output Dim | Aggregator |
|---|---|---|---|
| GNN Layer 1 | 768 | 512 | Mean |
| GNN Layer 2 | 512 | 256 | Mean |
| GNN Layer 3 | 256 | 128 | Mean |

**Acceptance Criteria:**
- AC-FRM-009-1: Each SAGEConv layer SHALL apply mean aggregation over 1-hop neighbors in the patient similarity graph; aggregator type SHALL be `'mean'` and SHALL be immutable at inference time.
- AC-FRM-009-2: Residual connections SHALL be applied between GNN layers where dimensionality permits (Layer 2 → Layer 3 via learned projection); the residual connection architecture SHALL be specified in SDD-001.
- AC-FRM-009-3: Layer normalization SHALL be applied after each SAGEConv operation; batch normalization SHALL not be used in GNN layers due to graph mini-batch incompatibility.
- AC-FRM-009-4: The final GNN output embedding of dimension 128 per patient node SHALL be the input to the multi-task output heads.
- AC-FRM-009-5: The GNN SHALL operate in inductive inference mode; the patient similarity graph at inference time SHALL be constructed dynamically per FRM-011 without access to training graph node features.

**Test Reference:** SST-FRM-009, UT-FRM-009-A through UT-FRM-009-E
**Hazard Reference:** RMF-H-020 (Graph construction errors leading to incorrect neighbor aggregation)

---

#### FRM-010 — Patient Similarity Graph Construction

**Requirement ID:** FRM-010
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B — Safety-Related **[SAFETY-CRITICAL]**

**Statement:**
At inference time, the system SHALL construct a patient similarity graph connecting the query patient to a cohort of clinically validated reference patients retrieved from the Reference Patient Database (RPD). Graph construction SHALL follow a deterministic, clinically interpretable similarity metric.

**Acceptance Criteria:**
- AC-FRM-010-1: The patient similarity graph SHALL be constructed using a cosine similarity metric applied to clinical/demographic embeddings (output of FRM-004); edges SHALL be created between the query patient and reference patients where cosine similarity exceeds a configurable threshold (default: 0.75, range: [0.60, 0.90]).
- AC-FRM-010-2: The maximum number of connected reference patient neighbors SHALL be configurable (default: 50, maximum: 200); exceeding the maximum SHALL result in selection of the top-k most similar neighbors by cosine similarity score.
- AC-FRM-010-3: The query patient node SHALL always be included in the graph; a graph with only the query patient node (isolated node) is a valid degenerate case and SHALL be handled without error, with the GNN effectively performing identity mapping.
- AC-FRM-010-4: Graph construction SHALL complete within 200 milliseconds at p95 for a reference database of up to 10,000 patients.
- AC-FRM-010-5: The constructed graph edge index and edge weights SHALL be validated for structural correctness (no self-loops on query node, no duplicate edges, valid COO format); structural validation failures SHALL trigger fault code `GRAPH-STRUCT-001`. **[SAFETY-CRITICAL]**
- AC-FRM-010-6: The reference patient features used for similarity computation SHALL be retrieved exclusively from the RPD and SHALL not include the query patient's own future outcome data (temporal leakage prevention). **[SAFETY-CRITICAL]**

**Test Reference:** SST-FRM-010, UT-FRM-010-A through UT-FRM-010-F
**Hazard Reference:** RMF-H-020, RMF-H-022 (Data leakage affecting prognostic validity)

---

#### FRM-011 — GNN Inference Mode Enforcement

**Requirement ID:** FRM-011
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B **[SAFETY-CRITICAL]**

**Statement:**
The GNN module SHALL enforce strict inference-mode operation at all times within the deployed software system, preventing gradient computation and ensuring frozen model parameters.

**Acceptance Criteria:**
- AC-FRM-011-1: All GNN forward pass operations SHALL execute within a `torch.no_grad()` context; absence of this context SHALL be a critical software fault detectable at runtime via a context assertion check. **[SAFETY-CRITICAL]**
- AC-FRM-011-2: The GNN module SHALL be set to `model.eval()` mode at system initialization and SHALL assert `not model.training` before each forward pass; a training-mode assertion failure SHALL halt inference and trigger `SYS-FAULT-TRAIN-MODE`.
- AC-FRM-011-3: Model parameters SHALL be loaded as read-only from cryptographically verified model artifacts (SHA-256 checksum validation per FRM-019); runtime modification of parameters SHALL be impossible through the deployed API surface.
- AC-FRM-011-4: The model loading procedure SHALL verify that the loaded model's parameter count matches the expected value for the registered model version; parameter count mismatch SHALL trigger `MODEL-INTEGRITY-001`. **[SAFETY-CRITICAL]**

**Test Reference:** SST-FRM-011, UT-FRM-011-A through UT-FRM-011-D
**Hazard Reference:** RMF-H-030 (Model parameter corruption or unintended modification)

---

#### FRM-012 — GNN Numerical Stability and Output Validation

**Requirement ID:** FRM-012
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B — Safety-Related **[SAFETY-CRITICAL]**

**Statement:**
The GNN forward pass output SHALL be validated for numerical integrity before transmission to the multi-task output heads.

**Acceptance Criteria:**
- AC-FRM-012-1: The GNN output tensor of shape `[1, 128]` (single query patient at inference) SHALL be checked for NaN and infinity values; detection SHALL trigger fault code `GNN-NAN-001` and initiate safe-state output protocol. **[SAFETY-CRITICAL]**
- AC-FRM-012-2: GNN output embedding L2-norm SHALL be monitored; embeddings with L2-norm exceeding 3 standard deviations above the training distribution norm (stored in model artifact metadata) SHALL trigger an `out-of-distribution` warning flag propagated to the output response.
- AC-FRM-012-3: GNN computation SHALL be deterministic for identical inputs and graph structure; determinism SHALL be verified during system qualification testing with a tolerance of 0.0 (exact floating-point reproducibility on the same hardware).

**Test Reference:** SST-FRM-012, UT-FRM-012-A through UT-FRM-012-C
**Hazard Reference:** RMF-H-019, RMF-H-028 (Out-of-distribution inference without warning)

---

### 5.5 Multi-Task Output Requirements

---

#### FRM-013 — Classification Head: AD Progression Risk Category

**Requirement ID:** FRM-013
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B — Safety-Related **[SAFETY-CRITICAL]**
**Regulatory Anchor:** FDA De Novo Performance Standard — AUC ≥ 0.85 [REGULATORY-ANCHOR]

**Statement:**
The classification output head SHALL produce a probabilistic estimate of 36-month AD progression risk, categorized into three ordinal risk classes. The classification head SHALL meet or exceed the performance standard of AUC ≥ 0.85 on the validation dataset defined in the Clinical Evaluation Report (CER-001).

**Acceptance Criteria:**
- AC-FRM-013-1: The classification head SHALL implement a 2-layer fully connected network: `[128 → 64 → 3]` with ReLU activation after layer 1 and softmax activation at the output, producing a probability distribution over three classes: `{LOW (0), MODERATE (1), HIGH (2)}`.
- AC-FRM-013-2: Output probabilities SHALL sum to 1.0 ± 1e-6; violation of this constraint SHALL trigger fault code `CLASSIF-PROB-001`. **[SAFETY-CRITICAL]**
- AC-FRM-013-3: The predicted class SHALL be the argmax of the probability vector; ties SHALL be resolved by selecting the higher-risk class (conservative tie-breaking rule). **[SAFETY-CRITICAL]**
- AC-FRM-013-4: The classification output SHALL include:
  - Predicted risk class label (`LOW`, `MODERATE`, `HIGH`)
  - Probability for each class (three values summing to 1.0)
  - 95% confidence interval for the predicted class probability, computed via calibrated temperature scaling applied to logits
- AC-FRM-013-5: The system SHALL achieve AUC ≥ 0.85 as measured on the independent held-out test set (n ≥ 500 MCI patients) documented in CER-001; this performance standard SHALL be re-evaluated at each model version update.
- AC-FRM-013-6: The classification output SHALL be accompanied by an explicit label warning: *"For decision support only. Clinical judgment required."* in all output formats. **[REGULATORY-ANCHOR]**

**Test Reference:** SST-FRM-013, UT-FRM-013-A through UT-FRM-013-F
**Hazard Reference:** RMF-H-001 (Incorrect risk classification leading to delayed intervention), RMF-H-002 (Overconfident risk output)

---

#### FRM-014 — Regression Head: Cognitive Decline Rate Prediction

**Requirement ID:** FRM-014
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B
**Regulatory Anchor:** FDA De Novo Performance Standard — RMSE ≤ 3.0 MMSE points/year [REGULATORY-ANCHOR]

**Statement:**
The regression output head SHALL predict the expected rate of cognitive decline, expressed as annualized MMSE point change, over a 36-month prediction horizon. The regression head SHALL meet or exceed the performance standard of RMSE ≤ 3.0 MMSE points per year on the validation dataset.

**Acceptance Criteria:**
- AC-FRM-014-1: The regression head SHALL implement a 2-layer fully connected network: `[128 → 64 → 1]` with ReLU activation after layer 1 and a linear activation at the output, producing a scalar MMSE decline rate in units of MMSE points per year.
- AC-FRM-014-2: The regression output SHALL be clipped to the physiologically plausible range [-3.0, 30.0] MMSE points per year; values at clip boundaries SHALL set a `REGRESSION_CLIPPED` flag in the output. **[SAFETY-CRITICAL]**
- AC-FRM-014-3: The regression output SHALL include:
  - Point estimate: predicted annual MMSE decline rate
  - 80% prediction interval: lower and upper bounds derived from quantile regression auxiliary head
  - Standard error of estimate derived from Monte Carlo dropout (50 forward passes with dropout=0.1 on regression head only)
- AC-FRM-014-4: The system SHALL achieve RMSE ≤ 3.0 MMSE points/year on the held-out test set documented in CER-001.
- AC-FRM-014-5: Units SHALL be explicitly labeled as "MMSE points per year" in all output representations to prevent clinician misinterpretation.

**Test Reference:** SST-FRM-014, UT-FRM-014-A through UT-FRM-014-E
**Hazard Reference:** RMF-H-004 (MMSE unit misinterpretation by clinical user)

---

#### FRM-015 — Survival Analysis Head: Time-to-Event Prediction

**Requirement ID:** FRM-015
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B
**Regulatory Anchor:** FDA De Novo Performance Standard — C-index ≥ 0.75 [REGULATORY-ANCHOR]

**Statement:**
The survival analysis output head SHALL implement a deep survival network (DeepSurv-compatible architecture) producing a hazard function estimate for the time-to-MCI-to-AD-dementia conversion event. The survival head SHALL meet or exceed a concordance index (C-index) of ≥ 0.75 on the validation dataset.

**Acceptance Criteria:**
- AC-FRM-015-1: The survival head SHALL implement a Cox proportional hazard log-risk scoring network: `[128 → 64 → 1]` producing a log-risk score, implemented using the `pycox` library (SOUP dependency, version-controlled in SOUP-REG-001).
- AC-FRM-015-2: The survival output SHALL include:
  - Log-risk score (continuous scalar)
  - Estimated survival probabilities at time horizons: 12, 24, 36, 48 months
  - Predicted median time-to-conversion (months) with 95% confidence interval
  - Baseline hazard function (stored in model artifact, time-indexed)
- AC-FRM-015-3: Survival probability estimates SHALL be bounded in [0.0, 1.0]; values outside this range SHALL trigger fault code `SURVIVAL-BOUND-001`. **[SAFETY-CRITICAL]**
- AC-FRM-015-4: The system SHALL achieve C-index ≥ 0.75 as measured on the held-out test set documented in CER-001.
- AC-FRM-015-5: The output SHALL clearly state the event definition: *"Time to clinical diagnosis of Alzheimer's disease dementia from MCI baseline, per NIA-AA 2011 criteria."* [REGULATORY-ANCHOR]
- AC-FRM-015-6: Survival predictions SHALL account for right-censoring; the system SHALL not interpret censored training observations as event occurrences; censoring mechanism correctness SHALL be verified in VVP-001.

**Test Reference:** SST-FRM-015, UT-FRM-015-A through UT-FRM-015-F
**Hazard Reference:** RMF-H-005 (Incorrect survival estimate leading to inappropriate care planning)

---

#### FRM-016 — Multi-Task Output Consistency Validation

**Requirement ID:** FRM-016
**Priority:** Mandatory
**Safety Classification:** IEC 62304 Class B — Safety-Related **[SAFETY-CRITICAL]**

**Statement:**
The system SHALL validate internal consistency between the three multi-task outputs (classification, regression, survival) before generating the final response. Cross-output consistency checks SHALL detect implausible contradictions between concurrent predictions.

**Acceptance Criteria:**
- AC