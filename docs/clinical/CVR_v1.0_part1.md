---
document_id: cvr-sections-1-5
generated: 2026-03-11
batch_id: msgbatch_01HZUXhy6DzGoszEMVS44MBf
status: DRAFT — requires human review before submission
---

# CLINICAL VALIDATION REPORT

---

**Document ID:** CVR-001
**Version:** 1.0
**Date:** 2025-07-14
**Status:** DRAFT
**Authors:** TBD
**Product:** NeuroFusion-AD — Clinical Decision Support SaMD
**Regulatory Pathway:** FDA De Novo / EU MDR Class IIa
**Standards:** IEC 62304 Class B | ISO 14971 | IEC 62366-1
**Target Integration Platform:** Roche Information Solutions — Navify Algorithm Suite

---

> **CONFIDENTIAL — DRAFT FOR INTERNAL REVIEW ONLY**
> This document contains preliminary validation findings and has not been approved for submission or external distribution. All results are subject to change pending final quality review.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Intended Use Statement](#2-intended-use-statement)
3. [Study Design](#3-study-design)
   - 3.1 [Training Cohort (ADNI)](#31-training-cohort-adni)
   - 3.2 [External Validation Cohort (Bio-Hermes-001)](#32-external-validation-cohort-bio-hermes-001)
4. [Methods](#4-methods)
   - 4.1 [Model Architecture](#41-model-architecture)
   - 4.2 [Training Methodology](#42-training-methodology)
   - 4.3 [Statistical Analysis Plan](#43-statistical-analysis-plan)
5. [Primary Validation Results](#5-primary-validation-results)

---

## 1. Executive Summary

### 1.1 Purpose

This Clinical Validation Report (CVR-001 v1.0) documents the design, execution, and results of the primary clinical validation for **NeuroFusion-AD**, a Software as a Medical Device (SaMD) that provides clinical decision support for the assessment of amyloid progression risk in patients with Mild Cognitive Impairment (MCI). This report is prepared in accordance with FDA Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device guidance, IEC 62304 Class B software lifecycle requirements, and ISO 14971 risk management principles.

### 1.2 Product Overview

NeuroFusion-AD is a multimodal graph neural network (GNN)-based clinical decision support tool. It integrates fluid biomarker data, acoustic speech features, motor function features, and structured clinical variables to generate an amyloid progression risk score. The system is intended to **aid**, not replace, clinical judgment in the evaluation of MCI patients aged 50–90 years. The device is intended for deployment within the Roche Information Solutions Navify Algorithm Suite.

### 1.3 Validation Cohorts

Validation was conducted across two independent cohorts:

| Cohort | Role | N (Total) | N (Labeled) | Key Characteristic |
|---|---|---|---|---|
| ADNI | Internal training/validation/test | 494 | 315 (63.8%) | CSF Aβ42 amyloid labels; synthesized acoustic/motor features |
| Bio-Hermes-001 | External fine-tuning validation | 945 | Not separately reported | Plasma pTau217 (Roche Elecsys); 24% underrepresented communities |

### 1.4 Key Performance Findings

**ADNI Held-Out Test Set (N = 100):**

| Metric | Value | 95% CI |
|---|---|---|
| Classification AUC | 0.579 | 0.452–0.671 |
| Sensitivity | 0.750 | — |
| Specificity | 0.467 | — |
| AUPRC | 0.479 | — |
| Survival C-index | 0.509 | 0.444–0.591 |
| MMSE RMSE | 2.23 pts/year | — |
| MMSE R² | −0.842 | — |
| ECE (pre-calibration) | 0.2001 | — |
| ECE (post-calibration, T=3.30) | 0.0210 | — |

**Bio-Hermes-001 Validation Set (N = 189):**

| Metric | Value | 95% CI |
|---|---|---|
| Classification AUC | 0.829 | 0.780–0.870 |

### 1.5 Summary of Critical Findings

The internal ADNI test set results reveal **materially limited discriminative performance**, with a classification AUC of 0.579 (95% CI: 0.452–0.671) and a Survival C-index of 0.509 (95% CI: 0.444–0.591), both approaching chance-level discrimination. The MMSE regression R² of −0.842 indicates that the regression head performs worse than a mean-prediction baseline on this cohort. Calibration is substantially improved by temperature scaling (ECE reduced from 0.2001 to 0.0210 at T=3.30); however, the underlying discriminative performance on internal data remains a primary concern.

In contrast, the Bio-Hermes-001 external validation set demonstrates substantially higher discriminative performance (AUC 0.829, 95% CI: 0.780–0.870), suggesting meaningful domain-specific generalization on the target deployment assay (Roche Elecsys plasma pTau217). However, this improvement must be interpreted cautiously given key methodological differences between cohorts.

**Subgroup fairness analysis on the ADNI test set did not meet the pre-specified fairness threshold** (fairness_pass: false), with a maximum AUC gap of 0.189 between APOE4 non-carriers (AUC 0.690) and APOE4 carriers (AUC 0.501). This finding carries direct clinical risk implications and requires resolution prior to regulatory submission.

### 1.6 Critical Limitations Summary

The following limitations are formally identified and require consideration by reviewers:

1. **Synthesized features (DRD-001):** Acoustic and motor features in the ADNI cohort are synthesized from clinical distributions rather than collected from real patients. This fundamentally limits the interpretability of modality importance weights from the ADNI test set.
2. **Assay proxy mismatch:** ADNI uses CSF pTau181 as a proxy for plasma pTau217 (the Roche Elecsys target assay). These are biologically and analytically distinct measurements.
3. **ADNI amyloid label incompleteness:** Only 63.8% (315/494) of ADNI participants carry valid CSF Aβ42 amyloid labels, introducing potential selection bias.
4. **Cross-sectional external validation:** Bio-Hermes-001 is a cross-sectional study; no longitudinal outcome data are available to validate progression claims.
5. **Training data leakage risk (ADNI val_auc = 1.0):** The ADNI validation AUC of 1.0 observed during training is attributed to the presence of the ABETA42_CSF feature during training. This does not reflect generalized performance and confirms the necessity of reporting held-out test set metrics only.
6. **APOE4 carrier subgroup failure:** Near-chance discrimination in APOE4 carriers (AUC 0.501) represents a clinically significant equity and safety concern.

### 1.7 Regulatory Conclusions

Based on current validation evidence, NeuroFusion-AD **does not meet the performance thresholds required for regulatory submission** in its current form. The internal validation results, synthesized modality data, and fairness failures constitute material deficiencies. The Bio-Hermes-001 results are encouraging but insufficient as a standalone basis for clearance given cross-sectional limitations and the absence of full sensitivity/specificity/PPV/NPV reporting for that cohort. Remediation activities are required prior to De Novo or MDR submission. Specific recommendations are documented in the Risk Management File and the post-market surveillance plan per ISO 14971.

---

## 2. Intended Use Statement

### 2.1 Device Intended Use

NeuroFusion-AD is intended to **aid clinicians in the assessment of amyloid progression risk** in adult patients with Mild Cognitive Impairment (MCI). The device generates a risk score and supporting evidence summary, presented as a clinical decision support output within the Roche Navify Algorithm Suite. The output is intended to supplement, not supplant, clinical evaluation, neuroimaging, and laboratory assessment performed by qualified healthcare professionals.

### 2.2 Target Patient Population

| Characteristic | Specification |
|---|---|
| Diagnosis | Mild Cognitive Impairment (MCI) — confirmed by clinical evaluation |
| Age range | 50–90 years |
| Setting | Outpatient memory clinic or specialist neurology practice |
| Input requirements | Minimum required: fluid biomarker panel (CSF or plasma pTau), structured clinical variables (MMSE, age, APOE4 status); optional: acoustic and motor assessment data |

**Contraindications:**

- Patients with confirmed dementia diagnosis (beyond MCI stage)
- Patients younger than 50 years or older than 90 years (outside validated range)
- Patients without available fluid biomarker data (minimum required input not met)
- Patients for whom the output would be used as the **sole basis** for treatment decisions, without clinician review

**Populations for which use is not validated:**

- Patients with non-Alzheimer's neurodegenerative diagnoses (e.g., Parkinson's disease, frontotemporal dementia) as the primary presentation
- Pediatric populations
- Populations outside the age range 50–90 years

### 2.3 Intended User Type

NeuroFusion-AD is intended for use by **qualified clinical professionals** including:

- Neurologists
- Geriatric psychiatrists
- Specialist dementia clinicians
- Clinical staff operating within a supervised specialist memory assessment service

The device is not intended for use by patients directly, nor for use by primary care clinicians operating without specialist supervision. Users are expected to have sufficient clinical background to interpret probabilistic risk outputs in the context of full patient history and clinical examination findings.

### 2.4 Clinical Context of Use

The device is intended to be used as an **adjunct** at the point of specialist clinical assessment, where a clinician is evaluating whether to initiate further amyloid-related diagnostic workup (e.g., amyloid PET, lumbar puncture for CSF biomarkers) or to consider eligibility for amyloid-targeting therapeutic interventions. The device output is one input among multiple clinical data sources and does not constitute a diagnosis.

### 2.5 Contraindications and Precautions Summary

| Condition | Classification |
|---|---|
| Use as sole diagnostic decision basis | Contraindicated |
| Patient age < 50 or > 90 years | Contraindicated |
| Absence of minimum required input features | Contraindicated |
| APOE4 carrier status (current evidence) | **Use with particular caution** — subgroup AUC 0.501; discriminative performance not established |
| Synthesized acoustic/motor feature inputs | Not applicable in deployment; see Section 3.1 limitation note |

---

## 3. Study Design

### 3.1 Training Cohort (ADNI)

#### 3.1.1 Cohort Overview

The Alzheimer's Disease Neuroimaging Initiative (ADNI) cohort served as the primary training, internal validation, and held-out test dataset for NeuroFusion-AD. ADNI is a longitudinal, multi-site observational study of MCI and Alzheimer's disease progression. All ADNI data use was conducted under applicable data use agreements.

#### 3.1.2 Patient Characteristics and Split Summary

| Parameter | Total | Train | Validation | Test |
|---|---|---|---|---|
| N (total) | 494 | 345 | 74 | 75† |
| N with valid amyloid label (CSF Aβ42) | 315 (63.8%) | — | — | 100‡ |
| Age range | 50–90 yrs | — | — | — |
| Amyloid label source | CSF Aβ42 | — | — | — |
| pTau assay | CSF pTau181 (proxy) | — | — | — |
| Acoustic features | Synthesized (DRD-001) | — | — | — |
| Motor features | Synthesized (DRD-001) | — | — | — |

> † The held-out test set used for all reported test metrics contains N=100 labeled participants per the evaluation run metadata; the 75/100 discrepancy between the original split specification (test=75) and the evaluation label count (N_labeled=100) is noted as a documentation item requiring reconciliation in the final report version.
>
> ‡ All 100 test-set participants used in the evaluation run carry valid amyloid labels.

#### 3.1.3 Amyloid Label Coverage

Amyloid classification labels based on CSF Aβ42 were available for 315 of 494 (63.8%) participants. The remaining 36.2% of participants lack a valid amyloid classification label. This incomplete label coverage introduces the risk of non-random missingness (e.g., if lumbar puncture was differentially performed in higher-risk or more symptomatic patients), which may bias the estimated performance metrics.

#### 3.1.4 Known Limitations of the ADNI Cohort

The following limitations are formally documented:

| Limitation ID | Description | Impact |
|---|---|---|
| LIM-ADNI-001 | **Synthesized acoustic and motor features (DRD-001):** Acoustic and motor modality features were not collected from real ADNI participants but were synthesized from clinical distributions. | Modality importance weights for acoustic (0.248) and motor (0.261) features derived from ADNI cannot be considered to reflect true clinical signal for these modalities. Performance attributable to these modalities on ADNI data is not clinically interpretable. |
| LIM-ADNI-002 | **Assay proxy mismatch:** ADNI uses CSF pTau181 as a proxy for the intended deployment assay, plasma pTau217 (Roche Elecsys). These assays differ in biological substrate (CSF vs. plasma), epitope (pTau181 vs. pTau217), and analytical platform. | Performance estimates derived from ADNI may not translate directly to deployment performance with Roche Elecsys plasma pTau217. |
| LIM-ADNI-003 | **Training data leakage (ABETA42_CSF):** ADNI training and validation AUC reached 1.0 during training, attributed to the presence of ABETA42_CSF as a direct proxy for the classification label. | Internal validation AUC of 1.0 does not reflect generalized model performance. Held-out test set metrics are the only valid ADNI performance estimate. |
| LIM-ADNI-004 | **Incomplete amyloid label coverage:** Valid CSF Aβ42 labels available for only 63.8% of participants. | Risk of selection bias; labeled subgroup may not be representative of the full MCI population. |
| LIM-ADNI-005 | **Single geographic/demographic origin:** ADNI is a predominantly North American, predominantly non-Hispanic White cohort, limiting representativeness for global deployment. | Potential performance degradation in underrepresented demographic groups not well-captured in ADNI. |

---

### 3.2 External Validation Cohort (Bio-Hermes-001)

#### 3.2.1 Cohort Overview

The Bio-Hermes-001 study served as the external validation and fine-tuning cohort. It is an independent study cohort with 945 total participants, split into a fine-tuning training set (N=756) and a validation set (N=189).

**Note:** Bio-Hermes-002 does not exist. References to any second Bio-Hermes cohort in any preceding documentation should be considered erroneous.

#### 3.2.2 Cohort Characteristics

| Parameter | Value |
|---|---|
| Total N | 945 |
| Fine-tuning train set (N) | 756 |
| Validation set (N) | 189 |
| Underrepresented community representation | 24% |
| pTau assay | Plasma pTau217 (Roche Elecsys) — target deployment assay |
| Study design | Cross-sectional |
| Longitudinal outcome data | Not available |

#### 3.2.3 Strengths of the Bio-Hermes-001 Cohort

| Strength | Description |
|---|---|
| **Target assay alignment** | Uses Roche Elecsys plasma pTau217 — the exact assay targeted in the clinical deployment context, providing direct analytical validity for the intended use setting. |
| **Demographic diversity** | 24% representation from underrepresented communities, providing a more representative validation of performance across a broader patient population than ADNI alone. |
| **Independent cohort** | Fully independent of the ADNI training data, supporting external validity assessment. |
| **Sample size** | Validation set of N=189 provides a reasonable basis for AUC estimation, though subgroup-level analysis is limited. |

#### 3.2.4 Limitations of the Bio-Hermes-001 Cohort

| Limitation ID | Description | Impact |
|---|---|---|
| LIM-BH-001 | **Cross-sectional design:** Bio-Hermes-001 is a cross-sectional study with no longitudinal follow-up data. | Amyloid *progression* claims — central to the device's intended use — cannot be directly validated from Bio-Hermes-001 data. The AUC reflects cross-sectional risk discrimination only. |
| LIM-BH-002 | **Incomplete performance reporting:** Sensitivity, specificity, PPV, NPV, and F1 are not reported for the Bio-Hermes-001 validation set. | Clinical performance characterization (e.g., false negative rate, false positive rate) cannot be assessed for deployment planning from available Bio-Hermes-001 data. |
| LIM-BH-003 | **Fine-tuning data contamination risk:** The Bio-Hermes fine-tuning used N=756 of 945 participants. The validation set (N=189) was held out from fine-tuning but not from cohort-level exploratory analysis. | Potential for optimistic bias in validation AUC if any cohort-level characteristics influenced fine-tuning design decisions. |
| LIM-BH-004 | **Single checkpoint reporting:** The reported AUC (0.829) reflects the best checkpoint at epoch 17 with early stopping. | Checkpoint selection based on validation AUC introduces some optimistic bias; an independent held-out set was not used for Bio-Hermes final reporting. |
| LIM-BH-005 | **No acoustic/motor real data confirmed:** It is not confirmed whether Bio-Hermes-001 acoustic and motor features are real or synthesized. If synthesized, limitations analogous to LIM-ADNI-001 apply. | Subject to clarification in final report version. |

---

## 4. Methods

### 4.1 Model Architecture

#### 4.1.1 Overview

NeuroFusion-AD is a multimodal graph neural network (GNN) architecture designed to integrate heterogeneous clinical data modalities into a unified amyloid progression risk representation. The architecture processes four distinct input modalities: (1) fluid biomarkers, (2) acoustic speech features, (3) motor function features, and (4) structured clinical variables.

#### 4.1.2 Multimodal Input Encoders

Each modality is processed by a dedicated encoder branch:

| Modality | Encoder Type | Key Features |
|---|---|---|
| Fluid biomarkers | Feature encoder (MLP-based) | CSF/plasma Aβ42, pTau181/217 |
| Acoustic features | Temporal/spectral encoder | Speech-derived biomarker proxies |
| Motor features | Feature encoder | Motor function clinical distributions |
| Clinical variables | Structured encoder | MMSE, age, APOE4 status, demographics |

Encoded modality representations are passed to a **cross-modal attention fusion layer**, which produces learned attention weights per modality per sample. These weights are used for modality importance attribution and are reported in the validation results.

#### 4.1.3 Graph Neural Network Integration

Patient-level representations from each modality encoder are integrated within a graph-structured architecture. Nodes represent individual patients or patient-feature clusters; edges encode relationships based on clinical similarity. Graph message-passing operations enable the model to leverage inter-patient relational structure during training.

#### 4.1.4 Output Heads

NeuroFusion-AD produces three simultaneous outputs:

| Output | Task | Metric Reported |
|---|---|---|
| Amyloid progression risk score | Binary classification | AUC, Sensitivity, Specificity, AUPRC |
| MMSE trajectory estimate | Regression | RMSE, R² |
| Time-to-progression estimate | Survival analysis | C-index |

#### 4.1.5 Post-hoc Calibration

Temperature scaling is applied post-hoc to the classification output. The optimal temperature parameter T=3.30 was determined on the ADNI validation set, reducing ECE from 0.2001 to 0.0210.

#### 4.1.6 Explainability

SHAP (SHapley Additive exPlanations) values are computed for feature-level attribution. Top SHAP features identified on the ADNI test set are: `abeta42_csf`, `ptau217`, `mmse_baseline`, `age`, `apoe4`.

Cross-modal attention weights from the ADNI test set are:

| Modality | Mean Attention Weight |
|---|---|
| Fluid | 0.246 |
| Acoustic | 0.248 |
| Motor | 0.261 |
| Clinical | 0.245 |

> **Interpretive caveat:** Given that acoustic and motor features in the ADNI cohort are synthesized (LIM-ADNI-001), the approximate uniformity of attention weights across modalities may reflect noise fitting to synthesized distributions rather than genuine clinical signal. These weights should not be used to draw conclusions about the clinical utility of acoustic or motor modalities until validated with real patient data.

---

### 4.2 Training Methodology

#### 4.2.1 Training Pipeline Overview

| Stage | Dataset | W&B Run ID | Key Configuration |
|---|---|---|---|
| Baseline training + HPO | ADNI | `jehkd9ud` | Optuna, 30 trials |
| Full retraining (best config) | ADNI | `ybbh5fky` | 150 epochs |
| External fine-tuning | Bio-Hermes-001 | `eicxum0n` | Frozen encoders, 17 epochs (early stopping) |

#### 4.2.2 Hyperparameter Optimization

Hyperparameter search was performed using **Optuna** with 30 trials on the ADNI training/validation split (train=345, val=74). The best configuration identified by Optuna was used to retrain the model for 150 epochs on ADNI data (W&B run `ybbh5fky`).

#### 4.2.3 Compute and Training Infrastructure

| Parameter | Value |
|---|---|
| Hardware | Single NVIDIA RTX 3090 GPU |
| Mixed precision training | Enabled (Automatic Mixed Precision, AMP) |
| Gradient accumulation steps | 4 |
| Learning rate schedule | OneCycleLR |
| Early stopping patience | 25 epochs |

#### 4.2.4 Bio-Hermes-001 Fine-Tuning

Fine-tuning on Bio-Hermes-001 was performed with the following constraints:

| Parameter | Value |
|---|---|
| Frozen components | All encoder modules (frozen weights) |
| Trainable components | Classification head only |
| Loss function | Classification-only loss |
| Learning rate | 5 × 10⁻⁵ |
| Best checkpoint epoch | 17 (early stopping triggered) |
| W&B Run ID | `eicxum0n` |

The decision to freeze encoders during Bio-Hermes fine-tuning limits the risk of catastrophic forgetting but also constrains the model's ability to adapt encoder-level representations to the plasma pTau217 assay characteristics.

#### 4.2.5 Training Data Leakage Note

During ADNI training, the validation AUC reached 1.0. This is attributed to the inclusion of ABETA42_CSF as a direct proxy for the classification label in the ADNI dataset. This feature provides near-perfect discriminative signal, rendering the ADNI validation AUC uninformative as a generalization estimate. The held-out test set results (AUC 0.579) are the only valid ADNI performance benchmark. This finding has been logged and all ADNI training AUC values from runs `jehkd9ud` and `ybbh5fky` should be disregarded for performance reporting purposes.

---

### 4.3 Statistical Analysis Plan

#### 4.3.1 Primary Performance Metrics

The primary performance metric for the binary classification task is the **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)**. Secondary metrics include sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), F1 score, and Area Under the Precision-Recall Curve (AUPRC).

For the regression task (MMSE trajectory), primary metrics are Root Mean Squared Error (RMSE) and R².

For the survival analysis task, the primary metric is the concordance index (C-index).

Calibration is assessed using Expected Calibration Error (ECE) before and after temperature scaling.

#### 4.3.2 Confidence Intervals

All AUC values are reported with **95% confidence intervals** computed using bootstrap resampling (stratified by outcome class). C-index confidence intervals are reported as derived from the evaluation run.

#### 4.3.3 Subgroup Analysis Plan

Pre-specified subgroup analyses were conducted on the ADNI test set across the following stratifications:

| Subgroup | Stratification Variable |
|---|---|
| Age < 65 years | Age at enrollment |
| Age 65–75 years | Age at enrollment |
| Age > 75 years | Age at enrollment |
| Male | Biological sex |
| Female | Biological sex |
| APOE4 non-carrier | APOE4 genotype |
| APOE4 carrier | APOE4 genotype |

**Fairness pass criterion:** A pre-specified maximum AUC gap threshold was applied. If the maximum AUC difference between any two subgroups within a stratification variable exceeded the threshold, the fairness check was flagged as failed (fairness_pass: false).

#### 4.3.4 SHAP Feature Attribution

SHAP values were computed on the ADNI test set to provide feature-level attribution for the classification output. The top 5 SHAP features are reported by mean absolute SHAP value.

#### 4.3.5 Calibration Assessment

ECE was computed using equal-width probability bins before and after temperature scaling. The optimal temperature parameter T was determined by minimizing ECE on the ADNI validation set and applied as a fixed post-processing transformation.

---

## 5. Primary Validation Results

### 5.1 Full Metrics Table — ADNI Held-Out Test Set (N=100)

| Metric | Value | 95% CI | Notes |
|---|---|---|---|
| Classification AUC | **0.579** | 0.452–0.671 | Primary metric; held-out test set |
| Sensitivity | 0.750 | — | At optimal threshold (not separately reported) |
| Specificity | 0.467 | — | At optimal threshold (not separately reported) |
| PPV | N/A | — | Not reported in evaluation run |
| NPV | N/A | — | Not reported in evaluation run |
| F1 Score | N/A | — | Not reported in evaluation run |
| AUPRC | 0.479 | — | Area under precision-recall curve |
| Survival C-index | 0.509 | 0.444–0.591 | Near-chance; progression discrimination not established |
| MMSE RMSE | 2.23 pts/year | — | Regression head performance |
| MMSE R² | −0.842 | — | Negative R²: worse than mean-prediction baseline |
| ECE (pre-calibration) | 0.2001 | — | Before temperature scaling |
| ECE (post-calibration) | **0.0210** | — | After temperature scaling, T=3.30 |

### 5.2 Full Metrics Table — Bio-Hermes-001 Validation Set (N=189)

| Metric | Value | 95% CI | Notes |
|---|---|---|---|
| Classification AUC | **0.829** | 0.780–0.870 | Best checkpoint, epoch 17 |
| Sensitivity | N/A | — | Not reported |
| Specificity | N/A | — | Not reported |
| PPV | N/A | — | Not reported |
| NPV | N/A | — | Not reported |
| F1 Score | N/A | — | Not reported |
| AUPRC | N/A | — | Not reported |
| Survival C-index | N/A | — | Cross-sectional cohort; not applicable |
| MMSE R² | N/A | — | Not reported |
| ECE | N/A | — | Not reported |

### 5.3 Subgroup Analysis Results — ADNI Test Set

| Subgroup | N | AUC | 95% CI | Fairness Status |
|---|---|---|---|---|
| Age < 65 years | 33 | 0.516 | 0.321–0.723 | — |
| Age 65–75 years | 34 | 0.618 | 0.414–0.762 | — |
| Age > 75 years | 33 | 0.573 | 0.370–0.783 | — |
| Male | 44 | 0.543 | 0.340–0.705 | — |
| Female | 56 | 0.592 | 0.414–0.731 | — |
| APOE4 non-carrier | 42 | 0.690 | 0.530–0.833 | — |
| APOE4 carrier | 58 | 0.501 | 0.369–0.650 | ⚠️ Near-chance |
| **Maximum AUC gap** | — | **0.189** | — | ❌ **Fairness FAIL** |

### 5.4 Modality Attention Weights — ADNI Test Set

| Modality | Mean Attention Weight |
|---|---|
| Fluid biomarkers | 0.246 |
| Acoustic | 0.248 |
| Motor | 0.261 |
| Clinical | 0.245 |

> **Caveat:** See LIM-ADNI-001. Acoustic and motor weights are not clinically interpretable due to synthesized feature inputs.

### 5.5 Top SHAP Features — ADNI Test Set

Ranked by mean absolute SHAP value:

1. `abeta42_csf`
2. `ptau217`
3. `mmse_baseline`
4. `age`
5. `apoe4`

> The dominance of `abeta42_csf` in SHAP rankings is consistent with the known training leakage issue (Section 4.2.5 and LIM-ADNI-003). This feature is a near-direct proxy for the ADNI classification label and its SHAP prominence does not reflect novel model learning.

---

### 5.6 Discussion of Domain Generalization

#### 5.6.1 Internal vs. External Performance Disparity

The most notable finding across the validation program is the **large performance gap between the ADNI internal test set (AUC 0.579) and the Bio-Hermes-001 external validation set (AUC 0.829)**. This 0.250-point AUC difference warrants careful interpretation.

Several factors plausibly contribute to the improved Bio-Hermes-001 performance:

1. **Assay alignment:** Bio-Hermes-001 uses Roche Elecsys plasma pTau217 — the target deployment assay — while ADNI uses CSF pTau181 as a proxy. The fine-tuned model has direct access to the target assay signal during Bio-Hermes training. This is the most likely primary driver of performance improvement.

2. **Fine-