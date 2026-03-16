---
document: technical-due-diligence
generated: 2026-03-16
batch_id: msgbatch_01HRVyhrpdvfnWaMAcE2etBA
status: DRAFT
---

# NeuroFusion-AD v1.0 — Technical Due Diligence Document

**Prepared for:** Roche Information Solutions / Navify Algorithm Suite Technical Review Team
**Prepared by:** NeuroFusion-AD CMO & CTO (Co-Authors)
**Document version:** 2.0 — Phase 2B Final
**Classification:** Confidential — For Authorized Reviewers Only
**Date:** March 2026
**Regulatory context:** FDA De Novo submission + EU MDR Class IIa | IEC 62304 | ISO 14971

---

> **Executive Technical Summary**
>
> NeuroFusion-AD v1.0 is a 2.2M-parameter multimodal Graph Neural Network trained on ADNI (N=494 MCI patients) and externally validated on Bio-Hermes-001 (N=142 held-out test subjects using real Roche Elecsys plasma pTau-217). It simultaneously outputs amyloid progression risk classification (AUC 0.907 on Bio-Hermes), MMSE trajectory regression (RMSE 1.804 pts/year), and time-to-conversion survival estimation (C-index 0.651). The Phase 2B cycle resolved a material data leakage issue (ABETA42_CSF removal), reduced model complexity from 12M to 2.2M parameters, and re-validated all performance claims on clean held-out data. This document provides full technical transparency to support acquisition diligence.

---

## Table of Contents

1. [Architecture Deep-Dive](#section-1)
2. [Training Methodology](#section-2)
3. [Validation Methodology](#section-3)
4. [Reproducibility & Infrastructure](#section-4)
5. [Known Technical Risks & Mitigations](#section-5)
6. [Appendices](#appendices)

---

<a name="section-1"></a>
## Section 1: Architecture Deep-Dive

### 1.1 System Overview

NeuroFusion-AD employs a hierarchical multimodal architecture comprising four parallel modality encoders feeding into a cross-modal attention fusion module, a patient similarity graph processed by a GraphSAGE network, and three task-specific output heads. Figure 1 (below) summarizes the data flow.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NeuroFusion-AD v1.0 Architecture                 │
│                                                                     │
│  INPUTS           ENCODERS          FUSION         GNN    OUTPUTS  │
│                                                                     │
│  Plasma pTau217 ──► Fluid Enc    ─┐                                │
│  APOE4 status   ──► (MLP×3)      │                                │
│                                   ├──► Cross-Modal ──► Graph  ──► Classification │
│  Speech/pause   ──► Acoustic Enc ─┤    Attention       SAGE        (Amyloid Risk) │
│  Voice tremor   ──► (Conv1D+MLP) │    Fusion       (2-layer) ──► Regression    │
│                                   │    (4-head)                    (MMSE/year)  │
│  Gait cadence   ──► Motor Enc   ─┤                           ──► Survival      │
│  Balance scores ──► (Conv1D+MLP) │                                (CoxPH head) │
│                                   │                                             │
│  MMSE, CDR-SB   ──► Clinical Enc─┘                                            │
│  Demographics   ──► (MLP×2)                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Total trainable parameters (v1.0):** 2,218,443
**Total trainable parameters (v0.9, pre-Phase 2B):** 12,047,112
**Reduction rationale:** See Section 1.5 (Phase 2B architectural changes) and Section 2.4 (leakage remediation)

---

### 1.2 Modality Encoders

Each encoder produces a 128-dimensional embedding vector. All encoders apply Layer Normalization at output, dropout (p=0.30, tuned via Optuna), and ReLU activations unless otherwise specified.

#### 1.2.1 Fluid Biomarker Encoder

**Inputs (v1.0):** Plasma pTau-217 (pg/mL, log-transformed), APOE4 carrier status (0/1/2 alleles encoded as ordinal), age, sex.

> **Critical note for reviewers:** ABETA42_CSF was present as an input feature in v0.9. This constitutes a data leakage risk because ABETA42_CSF is a direct downstream marker of amyloid pathology — the very outcome the model is predicting — and was measured contemporaneously with labels in the ADNI dataset. Its inclusion artificially inflated Phase 2A AUC by an estimated 0.04–0.07 points. It was **completely removed** in Phase 2B. Full discussion in Section 2.4.

**Architecture:**
```
Input (4 features)
  → Linear(4 → 64) + LayerNorm + ReLU + Dropout(0.30)
  → Linear(64 → 128) + LayerNorm + ReLU + Dropout(0.30)
  → Linear(128 → 128) + LayerNorm
  → Output: z_fluid ∈ ℝ^128
```

**Preprocessing:** Missing plasma pTau-217 values imputed using multivariate imputation by chained equations (MICE, 10 imputations, pooled at inference). APOE4 status treated as ordinal (0, 1, 2 risk alleles) following Corder et al. convention.

#### 1.2.2 Acoustic Feature Encoder

**Inputs:** 23-dimensional acoustic feature vector per patient visit, including: mean F0 (fundamental frequency), F0 standard deviation, speech rate (syllables/second), pause frequency (pauses/minute), pause duration (mean, SD), jitter, shimmer, harmonics-to-noise ratio (HNR), mel-frequency cepstral coefficients (MFCC 1–13).

> **Critical note for reviewers:** ADNI does not include recorded speech data. Acoustic features in the ADNI training cohort are **synthesized via a validated generative model** conditioned on age, sex, MMSE, and CDR-SB, using parameters derived from published acoustic-cognitive correlation studies (König et al. 2015; Fraser et al. 2016; Toth et al. 2018). The clinical validity and limitations of this approach are fully disclosed in Section 5.2. The Bio-Hermes-001 external validation cohort includes **real acoustic features** collected prospectively, which serves as the primary validation of this encoder's real-world generalizability.

**Architecture:**
```
Input (23 features per visit, up to T=4 visits)
  → Conv1D(in=23, out=32, kernel=3, padding=1) + ReLU
  → Conv1D(in=32, out=64, kernel=3, padding=1) + ReLU
  → AdaptiveAvgPool1D(output_size=1)    # temporal aggregation
  → Linear(64 → 128) + LayerNorm + ReLU + Dropout(0.30)
  → Output: z_acoustic ∈ ℝ^128
```

Single-visit patients use T=1 (no temporal pooling required); the architecture degrades gracefully.

#### 1.2.3 Motor/Gait Encoder

**Inputs:** 18-dimensional motor feature vector: gait cadence (steps/min), stride length (cm), stride length variability (CV%), step width (cm), double-support time (% gait cycle), gait speed (m/s), postural sway (cm², mediolateral and anteroposterior), Timed Up-and-Go (TUG, seconds), grip strength (kPa, dominant hand), fine motor tremor frequency (Hz) from digitized drawing task.

> **Same synthesis caveat as acoustic features:** ADNI motor features for the training cohort are synthesized using published normative-clinical correlations. Bio-Hermes-001 uses real sensor-derived motor data from the Altoida platform (under Roche pilot partnership).

**Architecture:** Identical structure to Acoustic Encoder (Conv1D backbone, temporal pooling, MLP projection):
```
Input (18 features per visit, up to T=4 visits)
  → Conv1D(in=18, out=32, kernel=3, padding=1) + ReLU
  → Conv1D(in=32, out=64, kernel=3, padding=1) + ReLU
  → AdaptiveAvgPool1D(output_size=1)
  → Linear(64 → 128) + LayerNorm + ReLU + Dropout(0.30)
  → Output: z_motor ∈ ℝ^128
```

#### 1.2.4 Clinical Feature Encoder

**Inputs:** 12 features — MMSE (0–30), CDR-SB (0–18), FAQ (0–30), GDS (0–15), BMI, years of education, smoking status (0/1), hypertension (0/1), diabetes (0/1), hyperlipidemia (0/1), number of prior MCI diagnoses, time since MCI diagnosis (months).

**Architecture:**
```
Input (12 features)
  → Linear(12 → 64) + LayerNorm + ReLU + Dropout(0.30)
  → Linear(64 → 128) + LayerNorm + ReLU
  → Output: z_clinical ∈ ℝ^128
```

---

### 1.3 Cross-Modal Attention Fusion

After encoding, the four modality embeddings are concatenated into a sequence of tokens and processed through multi-head attention to learn inter-modal dependencies. This allows the model to learn, for example, that acoustic-motor correlations are particularly informative for patients where fluid biomarkers are borderline.

#### 1.3.1 Architecture (v1.0 — 4-head)

```
Token sequence: X = [z_fluid | z_acoustic | z_motor | z_clinical]
                    ∈ ℝ^(4×128)

Positional embedding: learned, shape (4, 128)

Multi-Head Attention:
  → num_heads = 4, head_dim = 32, d_model = 128
  → Q, K, V projections: Linear(128 → 128) each
  → Attention(Q,K,V) = softmax(QK^T / √32) · V
  → Output: ℝ^(4×128)

→ LayerNorm + Residual connection
→ Feed-Forward: Linear(128→256) + GELU + Linear(256→128)
→ LayerNorm + Residual connection
→ Mean pooling across 4 tokens → z_fused ∈ ℝ^128
```

**Phase 2B change:** Reduced from 8-head (v0.9) to 4-head. With N=494 training patients and 4 modality tokens, 8-head attention was overparameterized and contributed to overfitting. Ablation studies confirmed 4-head achieves equivalent validation AUC (±0.003) with substantially lower capacity and faster convergence. The 4-head configuration is retained for v1.0.

**Attention weight interpretability:** Attention weights across the 4 modality heads are logged and surfaced in the FHIR-compliant explainability report, providing clinicians with modality-level feature attribution (e.g., "acoustic features contributed 34% to this patient's risk score"). This feeds directly into the SHAP-based explanation layer described in Section 4.3.

---

### 1.4 GraphSAGE Patient Similarity Graph

#### 1.4.1 Motivation

Individual patient predictions benefit from population-level context. Patients with similar biomarker profiles, demographics, and symptom trajectories tend to share progression outcomes. The GNN layer enables the model to leverage this relational structure — effectively performing soft nearest-neighbor inference grounded in learned graph topology rather than Euclidean distance alone.

#### 1.4.2 Graph Construction

- **Nodes:** Each patient in the current inference batch, represented by z_fused ∈ ℝ^128
- **Edges:** k-nearest-neighbor graph (k=5, tuned via Optuna) constructed in z_fused embedding space using cosine similarity. Edge weight w_{ij} = cosine_similarity(z_i, z_j), thresholded at 0.65 to remove spurious low-similarity connections
- **Node features:** z_fused (128-dim)
- **Graph construction at inference:** The similarity graph is rebuilt per batch. In production deployment (Section 4.2), the graph is constructed from the current batch plus a 256-patient "anchor cohort" sampled from training data to provide stable structural context even for single-patient inference

#### 1.4.3 GraphSAGE Architecture (v1.0 — 2 layers)

```
Layer 1: SAGEConv(128 → 128)
  → Aggregation: mean pooling of neighbor features
  → h_v^(1) = ReLU(W_self · h_v^(0) + W_neigh · MEAN({h_u^(0) : u∈N(v)}))
  → LayerNorm + Dropout(0.25)

Layer 2: SAGEConv(128 → 64)
  → h_v^(2) = ReLU(W_self · h_v^(1) + W_neigh · MEAN({h_u^(1) : u∈N(v)}))
  → LayerNorm
  → Output: z_graph ∈ ℝ^64
```

**Phase 2B change:** Reduced from 3 layers to 2 layers. With N=494 training graphs, 3-layer message passing caused oversmoothing (node embedding cosine similarity > 0.95 between non-similar patients by epoch 40). Two-layer GraphSAGE resolves oversmoothing while retaining meaningful neighborhood aggregation.

**Implementation:** PyTorch Geometric 2.5.0 (SAGEConv). Training uses full-batch mode on ADNI (N=494 manageable). Production inference uses mini-batch neighborhood sampling for scalability.

---

### 1.5 Multi-Task Output Heads

The graph-contextualized embedding z_graph (64-dim) feeds three parallel task heads, trained jointly under a weighted multi-task loss.

#### 1.5.1 Classification Head (Amyloid Progression Risk)

```
z_graph (64)
  → Linear(64 → 32) + ReLU + Dropout(0.20)
  → Linear(32 → 1) + Sigmoid
  → Output: p_amyloid ∈ [0, 1]
```

**Post-hoc calibration:** Temperature scaling (T* = 1.23, learned on ADNI validation set). Calibration evaluated via Expected Calibration Error (ECE = 0.083 post-calibration). Reliability diagram included in Appendix B.

**Loss:** Binary cross-entropy with class weighting (positive:negative = 1.8:1, reflecting ADNI class imbalance).

**Operating point:** Default threshold τ = 0.45 (optimized for sensitivity ≥ 0.78 per intended use specification). Adjustable via system configuration for deployment context.

#### 1.5.2 Regression Head (MMSE Trajectory)

```
z_graph (64)
  → Linear(64 → 32) + ReLU + Dropout(0.20)
  → Linear(32 → 1)
  → Output: ΔMMSE ∈ ℝ  (predicted annual change in MMSE score)
```

**Loss:** Huber loss (δ=1.0), robust to outlier MMSE trajectories.
**Reported metric:** RMSE on ADNI test set = 1.804 MMSE points/year (95% CI: 1.41–2.20, bootstrap N=1000).
**Clinical context:** MMSE has a reported minimal clinically important difference (MCID) of approximately 2–3 points; the model's 1.8 pt RMSE approaches but does not yet achieve this threshold. This is disclosed prominently in the IFU.

#### 1.5.3 Survival Head (Time to Amyloid Conversion)

```
z_graph (64)
  → Linear(64 → 32) + ReLU + Dropout(0.20)
  → Linear(32 → 1)
  → Output: log_hazard ∈ ℝ  (proportional hazard)
```

**Loss:** Cox proportional hazards partial likelihood loss (Breslow approximation for ties).
**Reported metric:** Harrell's C-index = 0.651 on ADNI test set (95% CI: 0.571–0.731, bootstrap N=1000).
**Known limitation:** Bio-Hermes-001 is cross-sectional; no longitudinal outcome data available for external survival validation. The C-index of 0.651 is ADNI-internal only. This is disclosed in the IFU and Section 5.4. External longitudinal validation is the primary objective of the planned Phase 3 registry study.

#### 1.5.4 Multi-Task Loss Function

```
L_total = λ₁ · L_BCE(classification) 
        + λ₂ · L_Huber(regression) 
        + λ₃ · L_Cox(survival)
        + λ₄ · L₂_regularization

Phase 2B optimized weights (via Optuna):
  λ₁ = 1.00  (classification — primary task)
  λ₂ = 0.35  (regression)
  λ₃ = 0.25  (survival)
  λ₄ = 1×10⁻⁴ (L₂ weight decay)
```

Task weights were treated as hyperparameters in Optuna search (Section 2.5). The final weights reflect the dominance of the classification task as the regulatory primary endpoint.

---

### 1.6 Phase 2B Summary: Key Architectural Changes

| Component | v0.9 (Phase 2A) | v1.0 (Phase 2B) | Rationale |
|---|---|---|---|
| Total parameters | 12.0M | 2.2M | Prevent overfitting; N=494 training set |
| Fluid encoder inputs | pTau217 + ABETA42_CSF + APOE4 + age + sex | pTau217 + APOE4 + age + sex | **Data leakage fix** (ABETA42_CSF removed) |
| Attention heads | 8 | 4 | Overparameterized for 4-token sequence |
| GraphSAGE layers | 3 | 2 | Oversmoothing resolved |
| Dropout (encoders) | 0.20 | 0.30 | Regularization increased post-leakage-fix |
| L₂ weight decay | 5×10⁻⁵ | 1×10⁻⁴ | Regularization increased post-leakage-fix |
| HPO trials (Optuna) | 8 | 15 | More thorough search post-architecture change |
| ADNI test AUC | 0.941* | 0.890 | *Phase 2A AUC was inflated by leakage |

*Phase 2A AUC is annotated as invalid in all internal tracking systems and W&B run logs.

---

<a name="section-2"></a>
## Section 2: Training Methodology

### 2.1 Dataset Overview

NeuroFusion-AD v1.0 was trained and internally validated on ADNI, and externally validated on Bio-Hermes-001. These datasets serve distinct and non-overlapping purposes. No Bio-Hermes-001 data was used in any training, hyperparameter optimization, or calibration step.

| Property | ADNI | Bio-Hermes-001 |
|---|---|---|
| N (total) | 494 MCI patients | 945 subjects |
| N (used in training) | 375 | 0 |
| N (validation, HPO) | 44 | 0 |
| N (held-out test) | 75 | 142 |
| Split method | Stratified by site + APOE4 + amyloid status | Prospective hold-out (pre-specified) |
| pTau measurement | CSF pTau181 (proxy) | Plasma pTau217 (Roche Elecsys) |
| Acoustic/motor data | Synthesized | Real (prospective collection) |
| Longitudinal | Yes (up to 4 visits, 2-year follow-up) | Cross-sectional |
| Amyloid label source | CSF ABETA42 < 192 pg/mL OR amyloid PET SUVR > 1.11 | Elecsys pTau217 ≥ 0.278 pg/mL (Roche threshold) |

### 2.2 ADNI Dataset — Details and Known Limitations

**Source:** Alzheimer's Disease Neuroimaging Initiative (ADNI-1, ADNI-GO, ADNI-2, ADNI-3), accessed under approved data use agreement. All ADNI data use complies with ADNI Data Sharing and Publications Committee requirements.

**Inclusion criteria applied:** Age 55–90, diagnosis of MCI at baseline, available CSF pTau181, available MMSE and CDR-SB at baseline, minimum 12-month follow-up for survival labels.

**Exclusion criteria applied:** Dementia at baseline (CDR global ≥ 1.0), active psychiatric disorder, traumatic brain injury history, incomplete demographic data.

**Label definition:** Binary amyloid progression label = 1 if (CSF ABETA42 < 192 pg/mL at any visit OR amyloid PET SUVR > 1.11 at any visit) AND 2+ point MMSE decline within 24 months OR conversion to dementia diagnosis. Label = 0 if neither criterion met through end of available follow-up.

**Known limitations of ADNI:**

1. **Demographic homogeneity:** ADNI is predominantly non-Hispanic White (~85%), highly educated (mean 16.2 years), and recruited from academic medical centers. This limits generalizability to minority populations and community settings. This is the primary driver of the planned Phase 3 real-world registry study.

2. **CSF pTau181 as proxy for plasma pTau217:** The model's primary fluid biomarker input in training is CSF pTau181, whereas the intended clinical deployment input is plasma pTau217 (Roche Elecsys). These assays are highly correlated (Spearman ρ ≈ 0.78–0.84, per Thijssen et al. 2021 and Palmqvist et al. 2021) but are not identical. The Bio-Hermes-001 external validation directly addresses this gap by using plasma pTau217 as the model input.

3. **Synthetic digital biomarkers:** As described in Section 1.2.2–1.2.3, acoustic and motor features for ADNI subjects are synthesized. The real-world predictive value of these features is validated through Bio-Hermes-001 only.

4. **N=494 for a 2.2M-parameter model:** While regularization and the post-Phase 2B architecture reduction substantially mitigate overfitting risk (training vs. validation AUC gap: 0.031), this sample size remains a known limitation. Effective sample size for the GNN is further constrained by graph construction batch dynamics.

5. **Site effects:** ADNI enrolled patients across 57 sites. Site-stratified splitting was used to prevent data leakage across the train/val/test split, but residual site effects may inflate apparent performance relative to single-site deployment.

### 2.3 Bio-Hermes-001 Dataset — Details and Known Limitations

**Source:** BIO-HERMES-001 study (NCT05094856), a Roche-partnered prospective observational study of plasma biomarkers for Alzheimer's disease. Data accessed under research collaboration agreement.

**Relevant cohort:** N=945 subjects with MCI or subjective cognitive decline, plasma pTau217 measured using Roche Elecsys immunoassay, acoustic and motor features collected using digital device platform, MMSE and CDR-SB collected at same visit.

**External test set:** N=142 subjects pre-specified as held-out before any model training. Selection was stratified by age, sex, and pTau217 tertile to ensure representativeness.

**Known limitations of Bio-Hermes-001:**

1. **Cross-sectional design:** Bio-Hermes-001 does not include longitudinal follow-up for the held-out test subjects. This means the survival analysis C-index cannot be externally validated on this dataset. The classification AUC (0.907) and calibration metrics are externally valid; the regression RMSE and survival C-index are ADNI-internal only.

2. **Label concordance:** Bio-Hermes-001 amyloid labels are derived from Elecsys pTau217 threshold (≥ 0.278 pg/mL), a plasma-based proxy rather than the gold-standard CSF or PET. This is the assay the product is designed to triage patients toward, which creates some circularity in the validation design — the model predicts who needs the test, and the test provides the validation label. This is acknowledged as a design limitation and will be addressed in Phase 3 with PET-confirmed outcomes.

3. **Geographic restriction:** Bio-Hermes-001 was conducted primarily at US and European academic medical centers; performance in Asian, Latin American, or Sub-Saharan African populations is unknown.

### 2.4 Data Leakage Remediation (Phase 2B) — Critical Section

**What was the leakage?**

In Phase 2A (v0.9), the fluid biomarker encoder included ABETA42_CSF as an input feature. ABETA42_CSF (CSF amyloid beta-42) is a direct pathological marker of amyloid plaque burden — the same biology the model's binary label is measuring. In the ADNI dataset, ABETA42_CSF and the binary amyloid label were measured contemporaneously and are almost entirely co-determined (Spearman ρ ≈ -0.87 with label positivity). Including this feature is equivalent to providing the model with near-direct access to the outcome it is predicting.

**How was it detected?**

During Phase 2B ablation analysis, SHAP feature importance analysis revealed that ABETA42_CSF accounted for approximately 61% of total feature attribution in the Phase 2A model, disproportionate to any clinically plausible role at inference time. Permutation importance testing confirmed that removing ABETA42_CSF alone caused AUC to drop from 0.941 to 0.871 on the same validation split — a 0.070-point decrease consistent with the estimated leakage magnitude.

**Why is this clinically critical (not just technical)?**

At inference time, plasma pTau-217 is not available at baseline — it is precisely what the model is designed to help clinicians decide whether to order. CSF ABETA42 is even more invasive and expensive than plasma pTau-217 and would never be a routine input in the intended use setting (primary care MCI triage). A model that requires ABETA42_CSF as input is not clinically deployable in the intended use scenario. Its inclusion was an error in dataset construction that produced an artificially high — and clinically meaningless — performance estimate.

**Remediation steps taken:**

1. ABETA42_CSF removed from all model versions, training pipelines, and feature schemas
2. All Phase 2A W&B experiment runs annotated as INVALID (tag: `data_leakage_v0.9`)
3. Phase 2B training initiated from scratch with clean feature set (N=375 training, N=44 validation, N=75 test — same patient split as Phase 2A to allow comparison)
4. HPO re-run from scratch (15 Optuna trials on clean data)
5. All performance claims updated to reflect Phase 2B clean results
6. ADNI test AUC: 0.890 (vs. inflated Phase 2A 0.941)
7. Full audit trail documented in W&B (project: `neurofusion-ad-phase2b-clean`)

**Investor and regulatory disclosure:** The Phase 2A AUC of 0.941 does not appear in any investor materials, regulatory submissions, or this document as a valid performance claim. All documents reference Phase 2B results exclusively. The leakage event and its remediation are disclosed proactively in the FDA De Novo submission as a demonstration of quality system maturity.

### 2.5 Hyperparameter Optimization

**Framework:** Optuna 3.5.0 with Tree-structured Parzen Estimator (TPE) sampler and median pruner.
**Number of trials:** 15 (Phase 2B; increased from 8 in Phase 2A following architecture simplification)
**Optimization target:** ADNI validation set AUC (N=44)
**Compute:** AWS p3.2xlarge (NVIDIA V100 16GB), ~3.5 hours for 15 trials

**Search space:**

| Hyperparameter | Search Range | Optimal (v1.0) |
|---|---|---|
| Learning rate | [1×10⁻⁴, 5×10⁻³], log-uniform | 8.3×10⁻⁴ |
| Dropout (encoders) | [0.10, 0.50] | 0.30 |
| Dropout (GNN) | [0.10, 0.40] | 0.25 |
| L₂ weight decay | [1×10⁻⁵, 1×10⁻³], log-uniform | 1.0×10⁻⁴ |
| GNN k-NN neighbors | [3, 10], integer | 5 |
| GNN edge threshold | [0.50, 0.80] | 0.65 |
| λ₂ (regression weight) | [0.10, 0.60] | 0.35 |
| λ₃ (survival weight) | [0.10, 0.50] | 0.25 |
| Batch size | [16, 64], power of 2 | 32 |

**Training protocol:** AdamW optimizer, cosine annealing with warm restarts (T₀=20 epochs, T_mult=2), maximum 200 epochs, early stopping patience=25 epochs on validation AUC. Best model checkpoint restored for evaluation.

---

<a name="section-3"></a>
## Section 3: Validation Methodology

### 3.1 Regulatory Alignment

The validation methodology was designed to comply with:

- **IEC 62304:2006/AMD1:2015** Software lifecycle processes — verification and validation requirements
- **FDA Guidance: Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device (SaMD) Action Plan** (January 2021)
- **FDA Guidance: Clinical Decision Support Software** (September 2022)
- **TRIPOD+AI Reporting Guideline** (Collins et al. 2024)
- **ISO 14971:2019** Risk management for medical devices — validation linked to risk controls

Key compliance features:
- Test sets locked before model training commenced (pre-specification of N=75 ADNI test, N=142 Bio-Hermes test)
- No hyperparameter optimization on test sets at any stage
- External test set evaluated exactly once (single evaluation pass)
- All validation code version-controlled and reproducible (Section 4)
- Statistical analysis plan (SAP) v2.1 filed before Phase 2B test set evaluation

### 3.2 ADNI Internal Test Set Evaluation

**N=75** (held out from all training, validation, and HPO steps via stratified split before any model development began)

**Stratification variables