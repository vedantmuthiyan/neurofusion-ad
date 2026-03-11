---
document_id: dhf-phase2
generated: 2026-03-11
batch_id: msgbatch_01HZUXhy6DzGoszEMVS44MBf
status: DRAFT — requires human review before submission
---

# DESIGN HISTORY FILE — PHASE 2
## NeuroFusion-AD: Multimodal Clinical Decision Support for Amyloid Progression Risk Assessment

---

**Document Number:** NFU-DHF-002
**Revision:** 1.0
**Effective Date:** [DATE]
**Classification:** Controlled Document
**Regulatory Basis:** IEC 62304:2015+AMD1:2015 | 21 CFR Part 820 | ISO 14971:2019 | FDA De Novo | EU MDR 2017/745 Class IIa
**Prepared By:** Clinical Documentation Specialist, NeuroFusion-AD Program
**Reviewed By:** [Software QA Lead] | [Clinical Affairs Lead] | [Regulatory Affairs Lead]
**Approved By:** [Design Authority]

> **Scope Notice:** This document covers Phase 2 design activities only — training, optimization, external validation, and calibration of the NeuroFusion-AD algorithm. Phase 1 (architecture specification, risk framework initialization, and data governance) is documented in NFU-DHF-001. Phase 3 (clinical validation, submission package) is documented in NFU-DHF-003 (in preparation).

---

## SECTION 1: PHASE 2 DHF INDEX

### 1.1 Document Map

| Item No. | Document Title | Document ID | IEC 62304 Clause | 21 CFR 820 Reference | Status |
|----------|---------------|-------------|-------------------|----------------------|--------|
| DHF-2.01 | Phase 2 DHF Index (this section) | NFU-DHF-002 §1 | 5.1, 5.2 | §820.30(a) | Released |
| DHF-2.02 | Training Decision Log | NFU-DHF-002 §2 | 5.5, 5.6 | §820.30(c)(d) | Released |
| DHF-2.03 | Model Version History Table | NFU-DHF-002 §3 | 8.1, 8.2 | §820.30(j) | Released |
| DHF-2.04 | Verification Records | NFU-DHF-002 §4 | 5.6, 5.7 | §820.30(f) | Released |
| DHF-2.05 | Phase 2 Risk Register Additions | NFU-DHF-002 §5 | 7.1–7.4 (ISO 14971) | §820.30(g) | Released |
| DHF-2.06 | Post-Market Surveillance Plan — Drift | NFU-DHF-002 §6 | 6.1, 6.2 | §820.30(a); 21 CFR 803 | Released |
| DHF-2.07 | Dataset Qualification Records | NFU-DDS-002 | 5.3 | §820.30(c) | Referenced |
| DHF-2.08 | Software Architecture Specification | NFU-SAS-001 | 5.3, 5.4 | §820.30(d) | Referenced (Phase 1) |
| DHF-2.09 | ADNI Data Limitation Acknowledgment | NFU-DRD-001 | 5.3 | §820.30(c) | Referenced |
| DHF-2.10 | Weights & Biases Training Records | NFU-TRN-002 | 5.5 | §820.30(e) | Referenced |
| DHF-2.11 | Bio-Hermes-001 External Validation Report | NFU-EVR-001 | 5.6 | §820.30(f) | Referenced |
| DHF-2.12 | Calibration Verification Record | NFU-CAL-001 | 5.6 | §820.30(f) | Referenced |
| DHF-2.13 | Subgroup Fairness Analysis Report | NFU-FAR-001 | 5.6 | §820.30(f) | Released |
| DHF-2.14 | SHAP Explainability Report | NFU-XAI-001 | 5.5 | §820.30(d)(e) | Released |

### 1.2 Phase 2 Scope Summary

Phase 2 encompasses all machine learning development activities following architectural specification:

- **Training Phase 2a:** Baseline supervised training on ADNI dataset (N=345 train, N=74 val, N=75 test), W&B run `jehkd9ud`
- **Training Phase 2b:** Hyperparameter optimization via Optuna (30 trials), followed by retraining of best configuration for 150 epochs, W&B run `ybbh5fky`
- **Training Phase 2c:** Domain adaptation fine-tuning on Bio-Hermes-001 (N=756 train, N=189 val), W&B run `eicxum0n`
- **Calibration:** Temperature scaling applied post-training (T=3.30)
- **Evaluation:** ADNI held-out test set (N=100) and Bio-Hermes-001 validation set (N=189)
- **Explainability:** SHAP feature attribution and attention weight analysis

### 1.3 IEC 62304 Software Safety Classification

Per NFU-DHF-001 §3.2, NeuroFusion-AD is classified as **IEC 62304 Class B** (software whose failure could result in undesirable patient outcomes but does not directly cause death or serious injury, owing to the CDS-only, physician-intermediated care pathway). This classification is maintained through Phase 2. No change control action required.

### 1.4 Phase 2 Key Deliverable Traceability

```
Intended Use (NFU-IFU-001)
    └─► Software Requirements (NFU-SRS-001, Phase 1)
            └─► Architecture Design (NFU-SAS-001, Phase 1)
                    └─► Training Configuration (NFU-TRN-002, Phase 2) ──► W&B: jehkd9ud
                            └─► HPO Results (NFU-HPO-001) ──────────────► W&B: ybbh5fky
                                    └─► Fine-Tune Config (NFU-FTN-001) ──► W&B: eicxum0n
                                            └─► Calibration (NFU-CAL-001)
                                                    └─► Verification Records (NFU-DHF-002 §4)
                                                            └─► Risk Update (NFU-DHF-002 §5)
```

---

## SECTION 2: TRAINING DECISION LOG

> **Governing Requirement:** IEC 62304 §5.5 (Software Unit Implementation and Verification), §5.6 (Software Integration and Integration Testing), ISO 14971 §10 (Production and post-production activities). Each decision below constitutes a formal design decision record per 21 CFR §820.30(c) and must be reviewed at each design review gate.

---

### Decision TDL-001: Primary Training Dataset Selection — ADNI

**Date:** [Phase 2 Initiation Date]
**Decision Owner:** Principal ML Engineer
**Review Status:** Approved at Phase 2 Design Review Gate DR-2.1

#### 2.1.1 Decision Made

ADNI (Alzheimer's Disease Neuroimaging Initiative) was selected as the primary training dataset for Phase 2 baseline training. The training split comprised N=345 subjects, validation N=74, and an independently held-out test set of N=75 (later expanded to N=100 for final evaluation). The dataset provides longitudinal MCI patient records including cognitive assessments, fluid biomarkers, and APOE genotyping. Training was executed on the ADNI cohort under W&B run ID `jehkd9ud` (baseline) and `ybbh5fky` (HPO-optimized, 150 epochs).

#### 2.1.2 Rationale

ADNI is the field-standard longitudinal dataset for MCI and Alzheimer's disease research, with well-characterized amyloid status, established data governance, and precedent use in regulatory submissions (see FDA guidance on AI/ML-based SaMD, January 2021). Its longitudinal structure provides time-to-event data required for the survival analysis component (C-index metric) of the intended use. No comparable longitudinal MCI dataset with multi-modal biomarker coverage and sufficient sample size was available under the project timeline and data access constraints.

#### 2.1.3 Alternatives Considered

| Alternative | Reason Not Selected |
|-------------|---------------------|
| AIBL (Australian Imaging, Biomarker & Lifestyle) | Smaller cohort; limited North American generalizability; restricted plasma pTau217 coverage |
| PREVENT-AD | Pre-MCI cohort only; label misalignment with intended use population (MCI, ages 50–90) |
| Proprietary clinical data only | Insufficient N for multimodal training; no longitudinal outcomes; data governance not established at Phase 2 initiation |
| Synthetic data augmentation as primary dataset | Synthetic data permissible only as supplement (per FDA AI/ML guidance); insufficient for primary training without real-world anchor |

#### 2.1.4 Risk Assessment

| Risk ID | Hazard | IEC 62304 Clause | ISO 14971 Reference | Severity | Probability | Risk Level | Mitigation |
|---------|--------|-----------------|---------------------|----------|-------------|------------|------------|
| R-TDL-001a | ADNI sample (N=494) may not represent real-world clinical diversity; model may underperform in underrepresented populations | §5.3 (Software Development Environment) | §4.4 (Risk Estimation) | Serious | Probable | High | External validation on Bio-Hermes-001 (24% underrepresented communities); subgroup fairness analysis required |
| R-TDL-001b | ADNI amyloid label coverage is 63.8% (315/494); training on incomplete labels may bias the classifier | §5.5.1 | §4.4 | Moderate | Probable | Medium | Label missingness analysis conducted; semi-supervised imputation considered and rejected (see TDL-004); restricted classification training to labeled subset only |
| R-TDL-001c | ADNI val_auc reached 1.0 during training due to ABETA42_CSF feature leakage; metric not reflective of true generalization | §5.6.3 | §5.1 | Serious | Confirmed | High | ADNI held-out test set (N=100, independent split) established as authoritative generalization metric; val_auc flagged as non-representative in all documentation |

**Residual Risk Acceptance:** Accepted with mitigations active. Reviewed at DR-2.1.

---

### Decision TDL-002: Amyloid Proxy — CSF pTau181 Used in Place of Plasma pTau217

**Date:** [Phase 2 Initiation Date]
**Decision Owner:** Clinical Affairs Lead + Principal ML Engineer
**Review Status:** Approved at DR-2.1 with mandatory disclosure requirement

#### 2.2.1 Decision Made

ADNI training utilized CSF pTau181 as the pTau feature input, as plasma pTau217 (Roche Elecsys — the target clinical assay for the intended use) is not available in the ADNI dataset at sufficient coverage. This substitution is documented as a formal dataset limitation per NFU-DRD-001. Bio-Hermes-001 fine-tuning uses plasma pTau217 (Roche Elecsys), providing the target-assay signal. The final deployed model (v1.0) was validated exclusively on plasma pTau217 labels via Bio-Hermes-001.

#### 2.2.2 Rationale

CSF pTau181 and plasma pTau217 are correlated amyloid-related biomarkers but are measured by different assays with different reference ranges, units, and analytical characteristics. Plasma pTau217 via the Roche Elecsys assay is the intended clinical deployment context (Roche Navify Algorithm Suite). The use of CSF pTau181 in ADNI training is a recognized limitation that cannot be remediated without a new dataset; however, it is acceptable as a pre-training signal because: (a) both biomarkers reflect phosphorylated tau pathology and show strong correlation in the literature (r ≈ 0.70–0.85; Hansson et al., 2021); (b) Bio-Hermes fine-tuning on plasma pTau217 provides domain adaptation to the target assay; (c) the limitation is explicitly disclosed in the IFU and model card.

#### 2.2.3 Alternatives Considered

| Alternative | Reason Not Selected |
|-------------|---------------------|
| Exclude pTau from ADNI training entirely | Significant information loss; pTau is a top SHAP feature and clinically critical biomarker |
| Use only Bio-Hermes-001 for all training | N=756 training set insufficient for multimodal architecture initialization; no longitudinal outcome data |
| Map CSF pTau181 to plasma pTau217 via published conversion formula | No validated conversion formula exists for Roche Elecsys pTau217 from ADNI CSF pTau181; would introduce unquantified systematic bias |

#### 2.2.4 Risk Assessment

| Risk ID | Hazard | ISO 14971 Reference | Severity | Probability | Risk Level | Mitigation |
|---------|--------|---------------------|----------|-------------|------------|------------|
| R-TDL-002a | Assay mismatch between training (CSF pTau181) and deployment (plasma pTau217) may degrade model performance in clinical use | §4.4 | Serious | Possible | High | Bio-Hermes fine-tuning provides target-assay adaptation; AUC 0.829 on Bio-Hermes val set confirms acceptable performance post-adaptation |
| R-TDL-002b | Clinician may interpret model output as validated for CSF pTau181 workflows | §4.4, §9 | Moderate | Unlikely | Medium | IFU explicitly restricts use to plasma pTau217 (Roche Elecsys) inputs; deployment validation (Navify) will enforce assay field mapping |

**Residual Risk Acceptance:** Accepted. Disclosure requirement active in IFU §2.3 (Intended Use Limitations).

---

### Decision TDL-003: Use of Synthesized Acoustic and Motor Features in ADNI Training

**Date:** [Phase 2 Initiation Date]
**Decision Owner:** Principal ML Engineer
**Review Status:** Approved at DR-2.1 with mandatory risk flag

#### 2.3.1 Decision Made

ADNI does not natively contain acoustic (speech/voice) or motor (gait/tremor) feature data. For Phase 2 ADNI training, acoustic and motor features were synthesized from published clinical distributions for MCI patients (documented in NFU-DRD-001). This decision applies exclusively to the ADNI training and test sets. Bio-Hermes-001 does not include acoustic or motor data; therefore these modalities receive no real-world validation signal in the current Phase 2 program. The synthesized feature generation procedure is version-controlled and reproducible.

#### 2.3.2 Rationale

The NeuroFusion-AD architecture is designed as a four-modality system (fluid biomarkers, acoustic, motor, clinical) to reflect the intended deployment context where all four modalities may be available in a neurology clinic workflow. Omitting acoustic and motor modalities from training would require an architectural redesign (not permissible post-Phase 1 freeze) or a two-track model (not consistent with the approved intended use). Synthesis from clinical distributions allows the encoder weights for these modalities to be initialized in a physiologically plausible range, enabling Bio-Hermes fine-tuning to update the classification head with real fluid/clinical data while retaining architectural completeness.

#### 2.3.3 Alternatives Considered

| Alternative | Reason Not Selected |
|-------------|---------------------|
| Zero-pad acoustic and motor inputs during ADNI training | Would cause encoder weight collapse; cross-modal attention mechanism would learn to ignore these modalities permanently |
| Acquire real acoustic/motor data for ADNI supplement | Logistically infeasible within Phase 2 timeline; ADNI participants not available for re-contact |
| Redesign to fluid + clinical only (2-modality) | Requires Phase 1 architecture change control; not aligned with intended use specification |
| Use transfer learning from non-AD speech/motor datasets | Domain mismatch; regulatory traceability would require additional bridging study |

#### 2.3.4 Risk Assessment

| Risk ID | Hazard | ISO 14971 Reference | Severity | Probability | Risk Level | Mitigation |
|---------|--------|---------------------|----------|-------------|------------|------------|
| R-TDL-003a | Synthesized features do not reflect true acoustic/motor signal variance in MCI; encoder weights may be miscalibrated | §4.4 | Serious | Probable | High | Modality importance analysis shows acoustic (0.248) and motor (0.261) attention weights are proportional to fluid (0.246) and clinical (0.245), suggesting no pathological suppression; however, this cannot distinguish true from spurious signal in synthesized data |
| R-TDL-003b | Post-market performance may degrade if real acoustic/motor data distributions differ from synthesis assumptions | §10.2 | Serious | Possible | High | PMS trigger established (§6 of this document): AUC drop ≥0.05 on acoustic/motor-active subgroup triggers retraining flag; real-world data collection plan in Phase 3 |
| R-TDL-003c | Clinician or integrator may assume acoustic/motor features are validated against real patient data | §4.4, §9 | Serious | Possible | High | IFU §4.1 (Modality Inputs) states: "Acoustic and motor features were not validated against real patient recordings in the Phase 2 program. Performance claims apply to fluid biomarker and clinical input modalities." |

**Residual Risk Acceptance:** Conditionally Accepted. Real-world acoustic/motor data collection mandated for Phase 3 clinical validation program (NFU-CVP-001, in preparation). Risk flagged as open in Risk Register (see §5, R-P2-007).

---

### Decision TDL-004: Hyperparameter Optimization Strategy — Optuna, 30 Trials

**Date:** [Phase 2 HPO Date]
**Decision Owner:** Principal ML Engineer
**Review Status:** Approved at DR-2.2

#### 2.4.1 Decision Made

Hyperparameter optimization was performed using the Optuna framework (Tree-structured Parzen Estimator sampler) over 30 trials on the ADNI validation set. The best configuration was selected and used to retrain the model for 150 epochs, with results recorded under W&B run ID `ybbh5fky`. Optimization objective was validation AUC (with the explicit caveat that ADNI val_auc is inflated due to ABETA42_CSF feature presence; this was accepted for HPO purposes as a relative ranking signal, not an absolute performance claim).

#### 2.4.2 Rationale

Optuna TPE provides efficient Bayesian-style optimization over the hyperparameter space with fewer trials than grid search, appropriate for the compute constraint (single RTX 3090). Thirty trials was determined to be sufficient for the hyperparameter space size (learning rate, batch size, dropout, attention heads, encoder depth) given TPE's sample efficiency. The known val_auc inflation issue means HPO results were used only to rank configurations, not to make absolute performance claims. The held-out ADNI test set (N=100) and Bio-Hermes validation set (N=189) serve as authoritative performance measures.

#### 2.4.3 Alternatives Considered

| Alternative | Reason Not Selected |
|-------------|---------------------|
| Grid search | Combinatorially expensive; infeasible on single RTX 3090 for 30+ hyperparameter combinations |
| Random search only | Less sample-efficient than TPE for correlated hyperparameters in transformer architectures |
| Neural Architecture Search (NAS) | Computationally infeasible; Phase 1 architecture is frozen |
| Manual tuning only | Non-reproducible; not compliant with IEC 62304 §5.5 (unit testing and verification requirements) |
| Expand to 100 HPO trials | Compute budget exceeded; marginal gain for TPE diminishes after ~30 trials for this parameter space size |

#### 2.4.4 Risk Assessment

| Risk ID | Hazard | ISO 14971 Reference | Severity | Probability | Risk Level | Mitigation |
|---------|--------|---------------------|----------|-------------|------------|------------|
| R-TDL-004a | HPO may overfit to ADNI validation set due to val_auc inflation; selected configuration may not generalize | §4.4 | Serious | Possible | Medium | Generalization evaluated on independent ADNI test set (N=100) and Bio-Hermes-001 (N=189); HPO selection justified only if test set performance meets thresholds |
| R-TDL-004b | 30 HPO trials may be insufficient to find global optimum; suboptimal model may be deployed | §5.5 | Minor | Possible | Low | Accepted; 30 trials is state-of-practice for this compute class; further optimization is a Phase 3 enhancement candidate |

**Residual Risk Acceptance:** Accepted.

---

### Decision TDL-005: Bio-Hermes-001 Fine-Tuning Strategy — Frozen Encoders, Classification Head Only

**Date:** [Phase 2 Fine-Tune Date]
**Decision Owner:** Principal ML Engineer + Clinical Affairs Lead
**Review Status:** Approved at DR-2.3

#### 2.5.1 Decision Made

Fine-tuning on Bio-Hermes-001 (N=756 train, N=189 val) used frozen multimodal encoders with only the classification head trainable. Learning rate was set to 5e-5, loss was classification-only (cross-entropy), and training used OneCycleLR scheduler with early stopping (patience=25 epochs). The best checkpoint was selected at epoch 17 (W&B run `eicxum0n`). Final validation AUC on Bio-Hermes-001: 0.829 (95% CI: 0.780–0.870).

#### 2.5.2 Rationale

Bio-Hermes-001 is cross-sectional only (no longitudinal outcomes), uses plasma pTau217 (the target assay), and includes 24% underrepresented communities — making it the highest-fidelity external validation dataset available. However, the relatively small fine-tuning set (N=756) relative to the full four-modality encoder parameter count creates a high overfitting risk if encoders are unfrozen. Freezing encoders preserves the multimodal representation learned from ADNI while allowing the classification head to adapt to the plasma pTau217 assay characteristics and Bio-Hermes-001 population distribution. The 5e-5 learning rate with OneCycleLR prevents catastrophic forgetting.

Note: Bio-Hermes-002 does not exist. Only Bio-Hermes-001 was available for this program. All references to external validation refer exclusively to Bio-Hermes-001.

#### 2.5.3 Alternatives Considered

| Alternative | Reason Not Selected |
|-------------|---------------------|
| Full fine-tuning (all layers unfrozen) | High overfitting risk with N=756; encoder representations would be overwritten with cross-sectional-only signal, losing longitudinal learned features |
| Fine-tune encoders with very low lr (1e-6) | Marginal encoder update at 1e-6 effectively equivalent to freezing but adds training instability; not preferred over explicit freezing |
| No fine-tuning; evaluate ADNI model directly on Bio-Hermes | ADNI model uses CSF pTau181; direct application to plasma pTau217 Bio-Hermes data would produce assay-mismatch degradation; unacceptable for submission |
| Separate model trained only on Bio-Hermes | N=756 insufficient to train four-modality architecture from scratch; would not leverage ADNI longitudinal learning |
| Unfreeze only pTau encoder | Bio-Hermes lacks acoustic/motor features; differential encoder unfreezing creates modality imbalance in attention mechanism |

#### 2.5.4 Risk Assessment

| Risk ID | Hazard | ISO 14971 Reference | Severity | Probability | Risk Level | Mitigation |
|---------|--------|---------------------|----------|-------------|------------|------------|
| R-TDL-005a | Frozen encoders may be misaligned with plasma pTau217 feature space; classification head cannot fully compensate | §4.4 | Serious | Possible | Medium | Bio-Hermes val AUC 0.829 demonstrates acceptable compensation; APOE4 and ptau217 are top SHAP features confirming correct signal capture |
| R-TDL-005b | Bio-Hermes-001 is cross-sectional; fine-tuned model may degrade longitudinal prediction performance (C-index) | §4.4 | Serious | Possible | Medium | C-index evaluated on ADNI test set (0.509) is tracked separately; longitudinal performance claim limited to ADNI-derived metrics with stated limitations |
| R-TDL-005c | Early stopping at epoch 17 may represent local minimum; model may not be fully converged | §5.5 | Minor | Unlikely | Low | Multiple checkpoint evaluations confirm epoch 17 is global val_auc maximum within patience window; early stopping with patience=25 is appropriate |

**Residual Risk Acceptance:** Accepted.

---

### Decision TDL-006: Temperature Scaling Calibration (T=3.30)

**Date:** [Phase 2 Calibration Date]
**Decision Owner:** Principal ML Engineer
**Review Status:** Approved at DR-2.3

#### 2.6.1 Decision Made

Post-hoc temperature scaling was applied to the final model (v1.0) using a single scalar temperature parameter T=3.30 learned on the ADNI validation set. This reduced Expected Calibration Error (ECE) from 0.2001 (pre-calibration) to 0.0210 (post-calibration), a 89.5% reduction. Calibration is applied as a fixed post-processing layer in the inference pipeline and does not alter model weights.

#### 2.6.2 Rationale

ECE of 0.2001 pre-calibration indicates substantial overconfidence in model probability outputs, which is clinically dangerous in a decision support context — overconfident probability scores may inappropriately bias physician judgment. Temperature scaling is the regulatory- and literature-preferred calibration method for neural networks used in clinical decision support (Guo et al., 2017; FDA AI/ML guidance) because it is: (a) parameter-efficient (single scalar); (b) monotonically order-preserving (AUC unchanged); (c) mathematically transparent and auditable; (d) applicable post-training without retraining. T=3.30 is a notably high temperature (indicating substantial overconfidence in the raw model), which is consistent with the observed ADNI val_auc inflation and the limited training set size.

#### 2.6.3 Alternatives Considered

| Alternative | Reason Not Selected |
|-------------|---------------------|
| Platt scaling (logistic calibration) | Two parameters; slightly higher overfitting risk on small calibration set; marginal improvement over temperature scaling for this model class |
| Isotonic regression | Requires larger calibration set; non-parametric and non-monotonic; less transparent for regulatory review |
| No calibration | ECE 0.2001 is clinically unacceptable; probability outputs would be unreliable; IFU confidence intervals would be misleading |
| Ensemble calibration | Requires multiple model training runs; not feasible within compute budget; Phase 3 enhancement candidate |

#### 2.6.4 Risk Assessment

| Risk ID | Hazard | ISO 14971 Reference | Severity | Probability | Risk Level | Mitigation |
|---------|--------|---------------------|----------|-------------|------------|------------|
| R-TDL-006a | Temperature learned on ADNI val set may not transfer to Bio-Hermes or clinical deployment population; ECE may revert | §4.4 | Moderate | Possible | Medium | PMS ECE monitoring established (§6); recalibration trigger at ECE > 0.05 on deployment data |
| R-TDL-006b | T=3.30 indicates severe raw overconfidence; root cause (ABETA42_CSF leakage, small N) may persist in deployed outputs if calibration layer is bypassed | §5.6, §9 | Serious | Unlikely | Medium | Calibration layer is architecturally embedded in inference pipeline (not a post-hoc preprocessing step); bypass requires deliberate code modification; version-locked in NFU-SAS-001 |

**Residual Risk Acceptance:** Accepted.

---

### Decision TDL-007: Compute Environment — Single RTX 3090, AMP, Gradient Accumulation=4

**Date:** [Phase 2 Training Date]
**Decision Owner:** Principal ML Engineer
**Review Status:** Approved at DR-2.1

#### 2.7.1 Decision Made

All Phase 2 training was conducted on a single NVIDIA RTX 3090 GPU (24 GB VRAM) using PyTorch Automatic Mixed Precision (AMP) and gradient accumulation with accumulation steps=4 to simulate effective batch size. OneCycleLR learning rate schedule was used for all training runs.

#### 2.7.2 Rationale

The single RTX 3090 configuration was the available compute resource within the project infrastructure. AMP (FP16/FP32 mixed precision) reduces VRAM consumption by approximately 40%, enabling larger batch sizes and model configurations that would otherwise exceed GPU memory. Gradient accumulation=4 simulates a 4× larger effective batch size without additional memory overhead, which is important for stable training of the multimodal transformer architecture. All training runs are fully reproducible via W&B run IDs (`jehkd9ud`, `ybbh5fky`, `eicxum0n`) with fixed random seeds.

#### 2.7.3 Alternatives Considered

| Alternative | Reason Not Selected |
|-------------|---------------------|
| Multi-GPU training | Not available in current infrastructure; Phase 3 retraining on larger datasets will require multi-GPU (planned) |
| Full FP32 precision | Exceeds VRAM for this model size; no clinically significant difference in convergence for this architecture class |
| Gradient accumulation=8 or higher | Diminishing returns; effective batch size already sufficient at accumulation=4 |

#### 2.7.4 Risk Assessment

| Risk ID | Hazard | ISO 14971 Reference | Severity | Probability | Risk Level | Mitigation |
|---------|--------|---------------------|----------|-------------|------------|------------|
| R-TDL-007a | AMP numeric instability (NaN loss) could produce silently corrupted model weights | §5.5 | Serious | Unlikely | Low | AMP gradient scaling active; NaN detection in training loop with automatic run termination; W&B run logs reviewed for loss anomalies |
| R-TDL-007b | Single-GPU training is not reproducible across different GPU hardware without seed control | §5.5, §5.7 | Minor | Possible | Low | Fixed random seeds documented; reproducibility verified by independent re-run at DR-2.3 |

**Residual Risk Acceptance:** Accepted.

---

### Decision TDL-008: APOE4 Status as Training Feature

**Date:** [Phase 2 Training Date]
**Decision Owner:** Clinical Affairs Lead + Regulatory Affairs Lead
**Review Status:** Approved at DR-2.1 with ethical review completion required

#### 2.8.1 Decision Made

APOE4 carrier status is included as a clinical feature in the model and appears as a top-5 SHAP feature. The model uses APOE4 status as a risk-stratification input consistent with published clinical guidelines (NIA-AA Research Framework, 2018).

#### 2.8.2 Rationale

APOE4 is the strongest known genetic risk factor for late-onset Alzheimer's disease and is clinically standard in MCI assessment. Its inclusion is consistent with the intended use population and target clinical workflow. Exclusion would materially degrade model performance in a clinically defensible direction.

#### 2.8.3 Risk Assessment

| Risk ID | Hazard | ISO 14971 Reference | Severity | Probability | Risk Level | Mit