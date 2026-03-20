# Clinical Validation Report

---

## DOCUMENT HEADER

| Field | Value |
|---|---|
| **Document ID** | CVR-001 |
| **Version** | 1.0 |
| **Date** | 2025-07-14 |
| **Status** | Superseded — See CVR v2.0 (NF-AD-CVR-002) |
| **Product** | NeuroFusion-AD |
| **Product Type** | Software as a Medical Device (SaMD) — Clinical Decision Support |
| **Regulatory Pathway** | FDA De Novo Authorization; EU MDR Class IIa |
| **Software Safety Classification** | IEC 62304 Class B |
| **Risk Management Standard** | ISO 14971 |
| **Intended Platform** | Roche Information Solutions — Navify Algorithm Suite |
| **Authors** | NeuroFusion-AD Development Team |
| **Reviewers** | Clinical Evaluation Team |
| **Approvers** | Chief Medical Officer |
| **Document Owner** | Clinical Documentation Specialist, NeuroFusion-AD Program |

---

> **SUPERSESSION NOTICE:** CVR v1.0 reported Phase 2 results that contained a data leakage artifact. All performance figures herein are invalid. See CVR v2.0 (NF-AD-CVR-002) for authoritative Phase 2B corrected results.

---

## TABLE OF CONTENTS

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

### 1.1 Purpose and Scope

This Clinical Validation Report (CVR-001 v1.0) documents the design, execution, and outcomes of the Phase 2B clinical validation of NeuroFusion-AD, a multimodal Software as a Medical Device (SaMD) intended to aid clinicians in assessing amyloid progression risk in adults with Mild Cognitive Impairment (MCI). This report covers the remediated Phase 2B model following the identification and correction of a critical data leakage issue present in prior development phases. The report is prepared in accordance with FDA AI/ML-Based Software as a Medical Device guidance, IEC 62304 software lifecycle requirements (Class B), and ISO 14971 risk management principles.

### 1.2 Background: Phase 2B Remediation

During internal quality review preceding Phase 2B, a critical data leakage defect (Defect Record DRD-001) was identified. The feature `ABETA42_CSF` (cerebrospinal fluid amyloid beta-42), exhibiting a Pearson correlation of r = −0.864 with the amyloid classification label, had been included in the fluid biomarker feature set. Because this feature constitutes or is a direct surrogate for the ground-truth outcome label, its inclusion in model training produced artificially inflated performance estimates in prior phases. This feature was removed prior to all Phase 2B training.

As a consequence of remediation, the fluid feature set was reduced from six features to two features: **plasma pTau217** (`PTAU217`) and **plasma neurofilament light chain** (`NFL_PLASMA`). Model architecture was correspondingly simplified (embedding dimension reduced from 768 to 256; parameter count reduced from approximately 60 million to approximately 2.24 million; dropout increased to 0.4). All Phase 2B training and evaluation were conducted on data that excluded `ABETA42_CSF` from input features and from training labels where applicable.

### 1.3 Validation Cohorts

Two independent cohorts were used for validation:

- **ADNI (Internal Validation):** 494 MCI patients drawn from the Alzheimer's Disease Neuroimaging Initiative. The dataset was partitioned into training (N=345), validation (N=74), and held-out test (N=75) subsets. Amyloid label coverage was 63.8% (315/494 patients with valid CSF Aβ42 labels); the test set included 44 labeled patients out of 75. This cohort served as the primary internal validation dataset and supported hyperparameter optimization.

- **Bio-Hermes-001 (External Validation):** 945 participants with stratified 70/15/15 splitting yielding a held-out test set of N=142. This cohort includes 24% participants from underrepresented communities, providing a more diverse validation population. The Bio-Hermes-001 cohort uses plasma pTau217 measured by the Roche Elecsys assay, which is the target deployment assay.

### 1.4 Key Performance Results

**Table 1.4-A: Summary of Primary Performance Metrics**

| Metric | ADNI Test (N=44 labeled) | Bio-Hermes-001 Test (N=142) |
|---|---|---|
| AUC (95% CI) | 0.890 (0.782–0.964) | 0.907 (0.855–0.959) |
| Sensitivity | 0.793 | 0.902 |
| Specificity | 0.933 | 0.879 |
| PPV | 0.958 | 0.807 |
| NPV | 0.700 | 0.941 |
| F1 Score | 0.868 | 0.852 |
| Optimal Threshold | 0.644 | — |
| ECE (post-calibration) | 0.083 (T=0.76) | — |
| MMSE RMSE | 1.80 pts/year | — |
| Survival C-index (95% CI) | 0.651 (0.525–0.788) | — |

Performance was consistent or improved in the external cohort relative to internal validation across primary classification metrics, with Bio-Hermes-001 achieving AUC 0.907 versus 0.890 in ADNI, indicating acceptable domain generalization. Sensitivity was substantially higher on Bio-Hermes-001 (0.902 vs. 0.793), which has favorable implications for a screening-adjacent clinical use case where missed amyloid-positive cases carry higher clinical consequence.

### 1.5 Key Limitations

The following limitations are documented and must be considered in the context of any regulatory or clinical interpretation of this report:

1. **Synthetic Acoustic and Motor Data (ADNI):** Acoustic speech features and motor/gait features used in ADNI training and validation are synthesized from published clinical distributions (per DRD-001). This introduces unknown bias into modality-specific performance contributions. No real-world speech or gait data were available in the ADNI cohort.

2. **pTau Assay Discordance (ADNI):** The ADNI cohort uses CSF pTau181 as a proxy for plasma pTau217 (Roche Elecsys). These represent different matrices (CSF vs. plasma) and different assay technologies. This substitution may introduce systematic error into ADNI-based estimates of fluid biomarker contribution and performance.

3. **Incomplete Amyloid Label Coverage (ADNI):** Only 63.8% of the ADNI cohort (315/494 patients) have valid amyloid labels from CSF Aβ42. The ADNI test set evaluation for classification metrics is based on N=44 of 75 total test patients. This substantially limits the statistical precision of ADNI-based performance estimates, as reflected in wide confidence intervals (e.g., AUC CI: 0.782–0.964).

4. **Cross-Sectional External Validation:** Bio-Hermes-001 is a cross-sectional cohort. No longitudinal outcome data are available, precluding external validation of progression trajectory predictions, MMSE slope estimation, or survival/time-to-progression endpoints on this cohort.

5. **Fairness Criterion Not Met (ADNI Subgroup Analysis):** The maximum AUC gap across demographic subgroups on the ADNI test set is 0.225, which exceeds the predefined fairness passing threshold. The fairness evaluation on ADNI is flagged as **FAIL** (`fairness_pass: false`). The primary source of this gap appears to be the APOE4 carrier subgroup (AUC=0.775 vs. non-carriers AUC=0.906) and the `age < 65` subgroup (AUC=1.0, N=11, which is insufficiently powered for reliable estimation). Dedicated fairness analysis on Bio-Hermes-001, which includes 24% participants from underrepresented communities, is deferred to a supplementary analysis section.

6. **Limited HPO Budget:** Phase 2B hyperparameter optimization was constrained to 15 Optuna trials due to computational budget limitations. The best trial achieved a validation AUC of 0.9081. It cannot be excluded that additional HPO trials would yield a superior configuration.

7. **Calibration:** Pre-calibration Expected Calibration Error (ECE) on ADNI was 0.1120, reduced to 0.0831 after temperature scaling (T=0.76). While improved, residual calibration error remains. Calibration on Bio-Hermes-001 has not been independently assessed.

### 1.6 Conclusions

NeuroFusion-AD Phase 2B demonstrates clinically meaningful discriminative performance for amyloid progression risk classification in MCI patients following remediation of the critical data leakage defect. The device achieves AUC ≥ 0.890 on internal validation and AUC ≥ 0.907 on external validation, with acceptable sensitivity and specificity profiles for a clinical decision support context. However, the findings are qualified by the limitations enumerated above, including synthetic modality data in ADNI, pTau assay discordance, incomplete label coverage, absence of longitudinal external validation, a failed fairness criterion on ADNI, and limited HPO budget. These limitations are assessed within the ISO 14971 risk management framework and must be addressed in the Post-Market Clinical Follow-Up (PMCF) plan and in the next development phase prior to regulatory submission.

---

## 2. Intended Use Statement

### 2.1 Device Name and Type

**Device Name:** NeuroFusion-AD
**Device Type:** Software as a Medical Device (SaMD) — Clinical Decision Support (CDS)
**Regulatory Classification:** FDA De Novo Authorization; EU MDR Class IIa

### 2.2 Intended Use

NeuroFusion-AD is intended to aid clinicians in the assessment of amyloid progression risk in adult patients with Mild Cognitive Impairment (MCI). The device integrates multimodal patient data — including plasma biomarkers, clinical assessments, acoustic speech features, and motor/gait features — to generate a probabilistic risk score and classification output indicating the likelihood of amyloid pathology progression. The device output is intended to inform, but not replace, clinical judgment and is intended for use as a supplementary decision support tool within a broader diagnostic and care planning workflow.

The device is **not** intended to provide a definitive diagnosis of Alzheimer's disease or amyloid pathology. It does not replace confirmatory diagnostic procedures including CSF analysis, amyloid PET imaging, or other biomarker-based diagnostic evaluations.

### 2.3 Intended Population

- **Condition:** Mild Cognitive Impairment (MCI), as established by standard clinical criteria (e.g., NIA-AA MCI criteria)
- **Age Range:** 50 to 90 years of age
- **Clinical Setting:** Memory clinics, neurology outpatient services, and academic medical centers with access to plasma biomarker testing (specifically plasma pTau217 via Roche Elecsys assay and plasma neurofilament light chain)
- **Deployment Platform:** Roche Information Solutions — Navify Algorithm Suite

### 2.4 Intended User

The device is intended for use by qualified healthcare professionals, including:

- Neurologists
- Geriatric psychiatrists
- Geriatricians
- Specialist memory clinic practitioners

Users are expected to have clinical training in the assessment and management of cognitive disorders. The device is not intended for self-use by patients or for use by non-specialist primary care providers without appropriate specialist oversight.

### 2.5 Intended Use Environment

The device operates as an integrated component within the Roche Navify Algorithm Suite in hospital and clinic-based healthcare information technology environments. It processes structured data inputs derived from clinical records, laboratory systems, and device-generated assessment outputs. The deployment environment is expected to conform to applicable healthcare IT security and data governance standards.

### 2.6 Contraindications

The device is **contraindicated** in the following circumstances:

1. **Age outside validated range:** Patients younger than 50 years or older than 90 years. The model has not been validated in these populations and performance characteristics are unknown.

2. **Absence of required biomarker inputs:** The device requires plasma pTau217 (Roche Elecsys assay) and plasma neurofilament light chain (NFL) as mandatory fluid biomarker inputs. The device must not be used as a primary risk assessment tool in the absence of these measurements.

3. **Confirmed Alzheimer's disease diagnosis:** The device is intended for use in MCI populations. Use in patients with confirmed dementia diagnoses is outside the intended use and the device has not been validated in this population.

4. **Non-MCI cognitive status:** The device has not been validated in cognitively normal individuals or in patients with other primary dementia diagnoses (e.g., frontotemporal dementia, Lewy body dementia, vascular dementia as the primary etiology).

5. **Pediatric populations:** The device is not intended for use in individuals under 18 years of age.

### 2.7 Warnings and Precautions

- The device output must be interpreted by a qualified clinician in the context of the complete clinical picture and must not be used as the sole basis for treatment decisions.
- The device has not been validated for use with pTau217 assays other than the Roche Elecsys platform. Use with alternative assays may produce unreliable outputs.
- Performance in populations substantially different from the validation cohorts (ADNI, Bio-Hermes-001) may differ from reported metrics. Particular caution is warranted in populations with low APOE4 carrier status representation or extreme age ranges at the boundary of the validated population.
- The probabilistic output of the device has been calibrated using temperature scaling on the ADNI cohort (T=0.76; ECE post-calibration=0.083). Clinicians should be aware that residual calibration error is present and that the output probability is an estimate subject to uncertainty.
- The device does not provide longitudinal trajectory predictions suitable for treatment initiation or monitoring decisions without supplementary clinical assessment.

---

## 3. Study Design

### 3.1 Training Cohort (ADNI)

#### 3.1.1 Cohort Overview

The Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset was used as the primary internal training and validation cohort for NeuroFusion-AD Phase 2B. ADNI is a longitudinal, multi-site observational study of aging, MCI, and Alzheimer's disease, providing standardized clinical, neuroimaging, and biomarker data.

#### 3.1.2 Dataset Characteristics

**Table 3.1-A: ADNI Cohort Characteristics**

| Characteristic | Value |
|---|---|
| Total Patients | 494 MCI patients |
| Age Range (Intended Use) | 50–90 years |
| Amyloid Label Coverage | 63.8% (315/494 patients with valid CSF Aβ42) |
| Amyloid Label Source | CSF Amyloid Beta-42 (Aβ42) |
| pTau Input Feature | CSF pTau181 (proxy for plasma pTau217) |
| Plasma NFL Input Feature | Available (assay details per ADNI protocol) |
| Acoustic Features | **Synthesized** from published clinical distributions |
| Motor/Gait Features | **Synthesized** from published clinical distributions |

#### 3.1.3 Dataset Partitioning

**Table 3.1-B: ADNI Dataset Split**

| Split | N (Total) | N (Amyloid Labeled) |
|---|---|---|
| Training | 345 | ~220 (estimated proportional) |
| Validation | 74 | ~30 (estimated proportional) |
| Test (Held-Out) | 75 | 44 |
| **Total** | **494** | **315 (63.8%)** |

> **Note:** Training and validation label counts are estimated from the reported 63.8% overall label coverage. The test set labeled count of 44/75 is reported directly from the evaluation run. Classification metrics on ADNI are computed over the 44 labeled test patients only.

#### 3.1.4 ADNI Cohort — Known Limitations

The following limitations of the ADNI cohort are formally documented for inclusion in the risk management file:

**Limitation ADNI-L-001 — Synthetic Acoustic and Motor Features:**
Acoustic speech features (used as one of four modality inputs in NeuroFusion-AD) and motor/gait features (used as a second modality) were not collected as part of the ADNI study protocol. For Phase 2B development, these features were synthesized by sampling from published clinical distributions characterizing MCI populations. This approach does not capture individual patient variability, inter-site collection variability, or real-world signal artifacts present in actual acoustic and motor recordings. The effect of this synthetic data on model training and on the ecological validity of ADNI-derived performance metrics is unknown. This limitation is cross-referenced with Defect Record DRD-001.

**Limitation ADNI-L-002 — pTau Assay Discordance:**
The intended deployment configuration of NeuroFusion-AD requires plasma pTau217 measured by the Roche Elecsys immunoassay platform. The ADNI cohort does not include plasma pTau217 measurements by this assay. CSF pTau181 was used as a proxy. CSF pTau181 and plasma pTau217 differ in biological matrix (cerebrospinal fluid versus whole blood plasma), analytical methodology, and the specific phosphorylation epitope measured. While both biomarkers track tau pathology and show correlation with amyloid burden, the quantitative relationship and dynamic range differ. Performance estimates derived from ADNI using CSF pTau181 as input may not accurately reflect performance in deployment with plasma pTau217.

**Limitation ADNI-L-003 — Incomplete Amyloid Label Coverage:**
Only 63.8% of ADNI MCI patients in the dataset have valid amyloid labels derived from CSF Aβ42. The remaining 36.2% of patients are unlabeled for the classification outcome. The test set evaluation is therefore limited to 44 of 75 total test patients. This incomplete coverage reduces the statistical power of test set performance estimates and introduces the possibility that labeled and unlabeled patients differ systematically in ways that could bias reported metrics.

**Limitation ADNI-L-004 — Cross-Site and Longitudinal Variability:**
ADNI is a multi-site study. While standardization protocols are applied, inter-site variability in clinical assessment, biomarker collection, and data quality may introduce noise not reflected in single-site deployment scenarios or in the homogeneous evaluation metrics reported here.

---

### 3.2 External Validation Cohort (Bio-Hermes-001)

#### 3.2.1 Cohort Overview

Bio-Hermes-001 is an independently collected cohort of 945 participants used for external validation and fine-tuning of NeuroFusion-AD Phase 2B. This cohort was not used during primary model architecture development or hyperparameter optimization, and the test partition was held out through all phases of training and fine-tuning.

#### 3.2.2 Dataset Characteristics

**Table 3.2-A: Bio-Hermes-001 Cohort Characteristics**

| Characteristic | Value |
|---|---|
| Total Participants | 945 |
| Underrepresented Community Representation | 24% |
| Study Design | Cross-sectional |
| Longitudinal Follow-up | Not available |
| pTau217 Assay | Plasma pTau217 — Roche Elecsys (target deployment assay) |
| Dataset Split | Stratified 70/15/15 |

#### 3.2.3 Dataset Partitioning

**Table 3.2-B: Bio-Hermes-001 Dataset Split**

| Split | N |
|---|---|
| Training (fine-tuning) | 661 |
| Validation | 142 |
| Test (Held-Out) | 142 |
| **Total** | **945** |

> The stratified 70/15/15 split was applied prior to any model contact with Bio-Hermes-001 data. The test partition (N=142) was held out through all fine-tuning and validation phases and was evaluated once for final performance reporting.

#### 3.2.4 Bio-Hermes-001 Cohort — Strengths

- **Target Assay Alignment:** Bio-Hermes-001 uses plasma pTau217 measured by the Roche Elecsys platform, which is the target assay for NeuroFusion-AD deployment. Performance metrics on this cohort therefore reflect expected deployment conditions more accurately than ADNI-based metrics.
- **Demographic Diversity:** The inclusion of 24% participants from underrepresented communities represents an improvement over many historical Alzheimer's biomarker research cohorts and supports an initial assessment of performance equity across demographic groups.
- **Cohort Size:** With 142 held-out test patients, Bio-Hermes-001 provides a more statistically powered test set than the ADNI test set (44 labeled patients), yielding narrower confidence intervals on primary metrics.

#### 3.2.5 Bio-Hermes-001 Cohort — Limitations

**Limitation BH-L-001 — Cross-Sectional Design:**
Bio-Hermes-001 is a cross-sectional study. No longitudinal follow-up data are available. This precludes external validation of NeuroFusion-AD's prognostic outputs, including MMSE slope estimation (MMSE RMSE), time-to-progression survival analysis (C-index), and amyloid progression trajectory predictions. Longitudinal validation on an independent cohort with follow-up data remains an outstanding requirement for full clinical validation.

**Limitation BH-L-002 — Fine-Tuning Data Overlap:**
The training partition of Bio-Hermes-001 (N=661) was used for fine-tuning NeuroFusion-AD (with frozen encoders). While the held-out test set was maintained independently, the use of any Bio-Hermes-001 data in fine-tuning means that Bio-Hermes-001 is not a fully independent external validation in the strictest sense. A truly independent external validation dataset — one with no role in any phase of model development — would be required for confirmatory regulatory validation.

**Limitation BH-L-003 — Acoustic and Motor Data Provenance:**
The provenance and collection protocol of acoustic and motor features in Bio-Hermes-001, including whether these represent real-world recordings or simulated/standardized assessments, should be confirmed. If these features in Bio-Hermes-001 are also derived from standardized or synthetic sources, the generalizability of performance metrics across real-world heterogeneous recording conditions may be limited.

> **Note:** Bio-Hermes-002 does not exist at time of writing. References to a "Bio-Hermes-002" dataset in any prior documentation should be treated as erroneous.

---

## 4. Methods

### 4.1 Model Architecture

#### 4.1.1 Overview

NeuroFusion-AD is a multimodal deep learning model implementing a graph neural network (GNN) — based fusion architecture. The model integrates four distinct input modalities: (1) **fluid biomarkers**, (2) **acoustic speech features**, (3) **motor/gait features**, and (4) **clinical assessments**. Each modality is processed by a dedicated encoder, and the resulting modality-specific representations are integrated via an attention-based fusion mechanism.

#### 4.1.2 Phase 2B Architecture Parameters

**Table 4.1-A: NeuroFusion-AD Phase 2B Architecture Summary**

| Parameter | Value |
|---|---|
| Architecture Type | Multimodal GNN with attention-based fusion |
| Embedding Dimension | 256 |
| Dropout Rate | 0.40 |
| Approximate Parameter Count | ~2.24 million |
| Fluid Feature Count | 2 (PTAU217, NFL_PLASMA) |
| Modalities | Fluid, Acoustic, Motor, Clinical |
| Output | Probabilistic amyloid risk score (binary classification head) + auxiliary regression heads |

#### 4.1.3 Input Features

**Fluid Biomarkers (2 features):**
- `PTAU217`: Plasma phospho-tau 217 (Roche Elecsys in deployment; CSF pTau181 proxy in ADNI)
- `NFL_PLASMA`: Plasma neurofilament light chain

> **Critical Note:** `ABETA42_CSF` was removed from the fluid feature set prior to all Phase 2B training following identification of critical data leakage (Pearson r = −0.864 with amyloid label). This feature is excluded from all Phase 2B model inputs and must not be reintroduced in any future version without full revalidation.

**Additional Modalities:**
- Acoustic speech features: Processed by dedicated acoustic encoder
- Motor/gait features: Processed by dedicated motor encoder
- Clinical features: Including MMSE baseline score, age, APOE4 status, and related variables

#### 4.1.4 Output Heads

The model implements multiple output heads trained jointly:

1. **Classification Head:** Binary amyloid risk classification (amyloid positive / negative) with probabilistic output
2. **Regression Head (MMSE):** MMSE trajectory prediction (MMSE RMSE, R²)
3. **Survival Head:** Time-to-progression estimation (C-index)

#### 4.1.5 Modality Attention Weights

Attention-based fusion produces interpretable modality importance weights. Mean attention weights on the ADNI test set are reported in Table 4.1-B.

**Table 4.1-B: Mean Modality Attention Weights (ADNI Test Set)**

| Modality | Mean Attention Weight |
|---|---|
| Clinical | 0.2855 |
| Acoustic | 0.2618 |
| Motor | 0.2396 |
| Fluid | 0.2130 |
| **Total** | **1.000** |

All four modalities contribute meaningfully to the model output, with clinical features receiving the highest mean attention weight (0.2855) and fluid biomarkers the lowest (0.2130). The relatively balanced distribution of attention weights suggests that no single modality dominates model predictions, which is consistent with the multimodal design intent. However, given that acoustic and motor features in ADNI are synthesized, the attention weights assigned to those modalities in ADNI may not accurately reflect their contribution in real-world deployment.

#### 4.1.6 SHAP Feature Importance

Top-ranked features by mean absolute SHAP value on the ADNI test set are:

1. `ptau217`
2. `nfl_plasma`
3. `mmse_baseline`
4. `age`
5. `apoe4`

These features align with established clinical understanding of Alzheimer's disease risk biomarkers, supporting face validity of the model's learned representations.

---

### 4.2 Training Methodology

#### 4.2.1 Phase 2B Training Pipeline Overview

Phase 2B training followed a two-stage pipeline:

**Stage 1 — ADNI Baseline Training with HPO:**
The model was trained on the ADNI training partition (N=345) using the Optuna hyperparameter optimization framework. Due to computational budget constraints, HPO was limited to 15 trials. The best configuration identified by Optuna (W&B run ID: `k58caevv` for baseline; `t9s3ngbx` for best remediated model) achieved a validation AUC of 0.9081 on the ADNI validation set (N=74).

**Stage 2 — Bio-Hermes-001 Fine-Tuning:**
The model trained in Stage 1 was subsequently fine-tuned on the Bio-Hermes-001 training partition (N=661) using a transfer learning approach. During fine-tuning, all modality encoders were **frozen**; only the classification head parameters were updated. This approach preserves the learned multimodal representations while adapting the classification boundary to the Bio-Hermes-001 data distribution (W&B run ID: `o4pcjy3r`).

#### 4.2.2 Hardware and Training Configuration

**Table 4.2-A: Training Infrastructure and Configuration**

| Parameter | Value |
|---|---|
| Hardware | Single NVIDIA RTX 3090 GPU |
| Mixed Precision Training | Automatic Mixed Precision (AMP) enabled |
| Gradient Accumulation Steps | 4 |
| Learning Rate Schedule | OneCycleLR |
| Early Stopping Patience | 25 epochs |
| HPO Framework | Optuna |
| HPO Trials | 15 (budget constrained) |
| Experiment Tracking | Weights & Biases (W&B) |

#### 4.2.3 Hyperparameter Optimization

HPO was conducted over the ADNI training/validation split using the Optuna framework with a budget of 15 trials. The optimization objective was validation set AUC. The best trial configuration yielded validation AUC = 0.9081. The constrained HPO budget (15 trials versus a recommended minimum of 50–100 trials for thorough search in comparable architectures) represents a known limitation. It is acknowledged that additional HPO exploration may identify superior configurations.

#### 4.2.4 Loss Function and Training Objective

The model was trained using a joint loss combining:
- Classification loss (binary cross-entropy on amyloid risk classification)
- Auxiliary regression loss (MMSE trajectory)
- Auxiliary survival loss (time-to-progression)

Fine-tuning on Bio-Hermes-001 used classification-only loss, consistent with the frozen encoder / classification head update strategy.

#### 4.2.5 Calibration

Post-training probability calibration was performed on the ADNI dataset using temperature scaling. The calibration procedure identified an optimal temperature parameter T = 0.76, reducing Expected Calibration Error from ECE = 0.1120 (pre-calibration) to ECE = 0.0831 (post-calibration). Temperature scaling was applied to model outputs prior to threshold-based classification.

---

### 4.3 Statistical Analysis Plan

#### 4.3.1 Primary Endpoints

The primary statistical endpoints for NeuroFusion-AD validation are:

1. **Classification AUC** (Area Under the Receiver Operating Characteristic Curve): Primary discrimination metric for binary amyloid risk classification.
2. **Sensitivity and Specificity** at the optimal operating threshold.
3. **Positive Predictive Value (PPV) and Negative Predictive Value (NPV)** at the optimal operating threshold.

#### 4.3.2 Secondary Endpoints

1. **F1 Score:** Harmonic mean of precision and recall at optimal threshold.
2. **MMSE RMSE:** Root Mean Squared Error of MMSE trajectory regression in pts/year (ADNI only, longitudinal data available).
3. **MMSE R²:** Coefficient of determination for MMSE trajectory regression (ADNI only).
4. **Survival C-index:** Harrell's concordance index for time-to-progression prediction (ADNI only).
5. **Expected Calibration Error (ECE):** Pre- and post-calibration (ADNI only).

#### 4.3.3 Confidence Intervals

All AUC estimates are reported with 95% confidence intervals (CI). Confidence intervals for AUC were computed using bootstrap resampling. Confidence intervals for survival C-index were computed by bootstrap resampling of test set predictions.

#### 4.3.4 Optimal Threshold Selection

The optimal classification threshold was selected by maximizing the Youden Index (sensitivity + specificity − 1) on the validation set. For ADNI, the optimal threshold was identified as 0.6443. The same threshold was applied at test time for threshold-dependent metrics (sensitivity, specificity, PPV, NPV, F1).

#### 4.3.5 Subgroup Analysis