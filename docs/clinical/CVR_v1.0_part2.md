---
document_id: cvr-sections-6-11
generated: 2026-03-11
batch_id: msgbatch_01HZUXhy6DzGoszEMVS44MBf
status: DRAFT — requires human review before submission
---

# NeuroFusion-AD Clinical Validation Report
## Sections 6–11 and Appendices

**Document ID:** NFD-CVR-2024-001-B
**Version:** 1.0
**Status:** DRAFT — For Regulatory Review
**Continuation of:** NFD-CVR-2024-001-A (Sections 1–5)
**Classification:** Confidential — Restricted Distribution

---

# SECTION 6: EXPLAINABILITY ANALYSIS

## 6.1 Overview and Rationale

Explainability is a regulatory and clinical necessity for AI-based Clinical Decision Support tools operating in high-stakes diagnostic contexts. FDA guidance on AI/ML-based SaMD and EU MDR Article 61 both emphasize that clinicians using AI outputs must be able to understand and interrogate the basis for those outputs. For NeuroFusion-AD, explainability is implemented at two complementary levels: (1) **multimodal attention weighting**, which describes the relative contribution of each input modality to a given prediction, and (2) **SHAP (SHapley Additive exPlanations) feature importance**, which identifies the specific clinical variables driving individual risk estimates.

This section presents both analyses as derived from the ADNI held-out test set (N=100) and discusses clinical interpretation of representative cases. All explainability outputs are surfaced to end users through the Navify Algorithm Suite interface alongside the numerical risk score.

> ⚠️ **Explainability Caveat:** Attention weights and SHAP values reflect statistical associations learned from training data. They do not imply causality. Two of the four modalities (acoustic and motor) are derived from **synthesized feature distributions** in the ADNI dataset (Data Restriction Document DRD-001). Modality importance scores for these two channels should be interpreted with caution and may not reflect true real-world signal contribution. This limitation is propagated to all clinical users through the Navify interface warning system.

---

## 6.2 Multimodal Attention Weights

NeuroFusion-AD employs a learnable cross-modal attention mechanism in the fusion layer. At inference time, the model computes normalized attention weights across the four input modalities for each patient. The values reported below represent **mean attention weights** across all N=100 ADNI test-set patients, averaged over the final-layer attention heads.

### Table 6.1: Mean Modality Attention Weights — ADNI Test Set (N=100)

| Modality | Mean Attention Weight | Rank | Data Source in ADNI | Notes |
|---|---|---|---|---|
| **Motor** | 0.2612 | 1 | ⚠️ Synthesized (DRD-001) | Highest mean weight; caution required |
| **Acoustic** | 0.2480 | 2 | ⚠️ Synthesized (DRD-001) | Second highest; synthetic origin limits interpretation |
| **Fluid (Biomarker)** | 0.2460 | 3 | CSF Aβ42 / pTau181 proxy | Real biomarker data; assay mismatch with target |
| **Clinical** | 0.2448 | 4 | MMSE, age, APOE4 status | Lowest weight; reflects structured EHR data |
| **Total** | **1.0000** | — | — | Normalized softmax output |

### 6.2.1 Interpretation

The attention weight distribution is **remarkably uniform across all four modalities** (range: 0.2448–0.2612, spread = 0.0164), suggesting that the fusion mechanism has not learned to preferentially rely on any single input channel. Several interpretations are possible:

**Hypothesis A — Genuine multimodal integration:** The model successfully fuses information from all four modalities, and each provides complementary signal not captured by the others. This would be the ideal operating condition.

**Hypothesis B — Undifferentiated weighting due to synthetic features:** Because acoustic and motor features are synthesized from clinical distributions rather than independently measured, they may not provide genuinely orthogonal information relative to the clinical modality. The attention mechanism may therefore assign weights by default rather than by learned discriminability. This hypothesis is supported by the fact that motor features (entirely synthetic) receive the *highest* attention weight — a result that is clinically implausible given the absence of real motor measurement data.

**Hypothesis C — Insufficient training signal:** With only 345 training patients (ADNI), the attention mechanism may lack sufficient examples to learn meaningful cross-modal weighting, defaulting toward uniform allocation.

**Clinical Guidance:** End users should be informed that modality weights in the current validation represent approximate equal weighting. Clinicians should not interpret high motor or acoustic weights as indicative of genuine motor or speech signal in the current ADNI-trained model. Bio-Hermes-001 fine-tuning used plasma pTau217 (real assay) but is cross-sectional and does not permit longitudinal modality-level assessment. Revision of modality importance analysis is planned following collection of real acoustic and motor data (see Section 9, Limitation 3).

---

## 6.3 SHAP Feature Importance

SHAP values were computed using the TreeSHAP-compatible approximation method applied to the post-fusion classification head of NeuroFusion-AD. Values represent mean absolute SHAP contributions across the ADNI test set (N=100), indicating the average magnitude of each feature's impact on the model's output probability.

### 6.3.1 Top-5 SHAP Features

The five highest-ranking features by mean |SHAP| value are listed below in rank order:

| Rank | Feature | Modality | Clinical Significance | Interpretability Concern |
|---|---|---|---|---|
| 1 | `abeta42_csf` | Fluid | Gold-standard amyloid proxy; direct disease relevance | Only 63.8% label coverage in ADNI (N=315/494) |
| 2 | `ptau217` | Fluid | Tau phosphorylation marker; Roche Elecsys assay in Bio-Hermes | CSF pTau181 used as proxy in ADNI (assay mismatch) |
| 3 | `mmse_baseline` | Clinical | Established cognitive screening tool; reflects current deficit | Cross-sectional; does not capture trajectory |
| 4 | `age` | Clinical | Known risk modifier for amyloid progression | Non-modifiable; raises fairness considerations |
| 5 | `apoe4` | Clinical | Highest known genetic risk factor for late-onset AD | Binary encoding obscures dosage effect (ε4/ε4 vs ε3/ε4) |

### 6.3.2 Interpretation and Clinical Alignment

The top SHAP features are **clinically coherent** — the model is primarily driven by established AD biomarkers (Aβ42, pTau217) and validated risk factors (MMSE, age, APOE4). This alignment with clinical knowledge provides a degree of face validity. However, several concerns merit discussion:

**Dominance of fluid biomarkers:** Features ranked 1 and 2 are both fluid biomarkers. In the ADNI dataset, `abeta42_csf` carries label-proximate information (CSF Aβ42 is used to derive the amyloid progression label in a subset of patients). This creates a **potential target leakage risk**: if the amyloid label was partially derived from or strongly correlated with `abeta42_csf` in the same patients, SHAP dominance of this feature may reflect label correlation rather than genuine prospective predictive signal. This concern is documented in the risk register (ISO 14971 Risk ID: ML-SHAP-001) and is under active investigation.

**pTau217 assay mismatch:** `ptau217` as used in ADNI refers to CSF pTau181, not plasma pTau217 (Roche Elecsys). The SHAP importance of this feature in ADNI-derived explanations may not translate to the same feature importance when using the target assay in deployment. The Bio-Hermes-001 fine-tuning uses the correct Roche Elecsys plasma pTau217 assay, but SHAP analysis was not re-derived from the Bio-Hermes validation set (which is cross-sectional and lacks outcome labels for full SHAP decomposition). This constitutes a gap in the explainability evidence base.

**Age as a high-ranking feature:** Age appearing in the top 5 SHAP features has direct fairness implications. A model that uses age as a primary driver of risk scores may produce systematically higher risk scores for elderly patients independent of actual biomarker status. This intersects with the subgroup findings reported in Section 7.

---

## 6.4 Clinical Case Studies

Three clinical case studies are presented to illustrate how NeuroFusion-AD outputs, explainability information, and uncertainty estimates interact in realistic clinical scenarios. Cases were selected from the ADNI test set according to the following pre-specified criteria:

**Selection Criteria:**
- **Case A (True Positive / High Confidence):** Patient with confirmed amyloid positivity (CSF Aβ42 below threshold), high model risk score (>0.75), and SHAP explanation dominated by fluid biomarkers. Selected to demonstrate the model's best-case operating condition.
- **Case B (False Negative / Low Confidence):** Patient with confirmed amyloid positivity but low model risk score (<0.40), representing a missed case. Selected to demonstrate failure mode transparency and the clinical importance of not over-relying on the score.
- **Case C (Uncertain / Subgroup-Relevant):** Patient from the APOE ε4 carrier subgroup (the subgroup with the lowest observed AUC, 0.501) with an ambiguous risk score (0.45–0.55). Selected to demonstrate the interaction between fairness concerns and individual-level explainability.

All identifying information has been removed. Ages are reported in 5-year bands. Clinical details are paraphrased to prevent re-identification.

---

### Case Study A: High-Confidence True Positive

**Patient Profile:**
- Age band: 70–75 | Sex: Female | APOE: ε3/ε4 (carrier)
- MMSE Baseline: 24/30 | CSF Aβ42: Below positive threshold
- Clinical context: Referred for memory concerns; family history of AD

**Model Output:**
- Risk Score: 0.81 (post-calibration, T=3.30)
- Classification: High Risk (above optimal threshold)
- Survival C-index prediction: Consistent with early progression trajectory

**Attention Weights (this patient):**
- Fluid: 0.31 | Clinical: 0.26 | Motor: 0.22 | Acoustic: 0.21

**Top SHAP Features (this patient):**
1. `abeta42_csf` (+0.18) — strongly positive for amyloid
2. `ptau217` (+0.12) — elevated tau signal
3. `mmse_baseline` (+0.07) — mild deficit at baseline
4. `apoe4` (+0.05) — carrier status adds risk
5. `age` (+0.03) — moderate contribution

**Clinical Interpretation:**
This case demonstrates the model operating as intended. The fluid modality receives elevated attention weight (0.31 vs. population mean of 0.246), consistent with strong biomarker signal. SHAP values are directionally coherent — elevated Aβ42 and tau drive risk upward, consistent with established biomarker models of AD. The MMSE deficit of 24 provides corroborating clinical evidence. The attending clinician, reviewing this output in Navify, would see a high risk score supported by specific, clinically recognizable features. This case passes the explainability face-validity check.

**Caveat:** This patient is an APOE ε4 carrier, the subgroup for which the model shows the weakest aggregate performance (AUC=0.501). Individual-level performance may exceed group-level AUC in cases with strong biomarker signal, underscoring that subgroup AUC reflects average performance, not the impossibility of correct classification within the subgroup.

---

### Case Study B: False Negative — Missed Progression Case

**Patient Profile:**
- Age band: 65–70 | Sex: Male | APOE: ε3/ε4 (carrier)
- MMSE Baseline: 27/30 | CSF Aβ42: Below positive threshold (confirmed amyloid positive)
- Clinical context: Subtle memory complaints; no strong family history

**Model Output:**
- Risk Score: 0.33 (post-calibration)
- Classification: Low Risk (below threshold)
- Ground Truth: Amyloid positive — **FALSE NEGATIVE**

**Attention Weights (this patient):**
- Motor: 0.29 | Acoustic: 0.27 | Clinical: 0.24 | Fluid: 0.20

**Top SHAP Features (this patient):**
1. `mmse_baseline` (-0.09) — near-normal score reduces risk estimate
2. `age` (-0.04) — lower-end age for the cohort
3. `abeta42_csf` (+0.06) — positive signal, but weight insufficient to overcome
4. `apoe4` (+0.04) — carrier, but SHAP impact modest
5. `ptau217` (+0.02) — minimal tau elevation

**Clinical Interpretation:**
This case illustrates a critical failure mode. Despite confirmed amyloid positivity (the ground-truth label), the model assigns a low risk score, driven primarily by a near-normal MMSE (27/30) that suppresses the risk estimate. The motor and acoustic modalities receive elevated attention weight (0.29 and 0.27), but these features are synthesized in ADNI and may carry noise rather than signal. The fluid modality — which contains the genuine positive signal — is paradoxically assigned the *lowest* attention weight (0.20) in this patient.

This false negative has significant clinical consequences: a patient with early amyloid pathology who presents with only subtle cognitive symptoms may be falsely reassured by a low risk score. The NeuroFusion-AD labeling explicitly states the tool is an **aid to assessment, not a definitive diagnostic tool**, and this case exemplifies the necessity of that framing. Clinicians should be instructed that a low NeuroFusion-AD score does not rule out amyloid progression, particularly in patients with biomarker evidence of early pathology but preserved cognition.

This pattern likely contributes to the model's sensitivity of 0.750 paired with specificity of only 0.467 — the model is aggressive in flagging risk based on cognitive scores but misses cases where biomarker positivity precedes cognitive decline.

---

### Case Study C: Uncertain Score in Low-Performance Subgroup

**Patient Profile:**
- Age band: 60–65 | Sex: Female | APOE: ε4/ε4 (homozygous carrier)
- MMSE Baseline: 26/30 | CSF Aβ42: Near threshold (borderline)
- Clinical context: Research participant; proactive cognitive health monitoring

**Model Output:**
- Risk Score: 0.49 (post-calibration) — near-decision boundary
- Classification: Indeterminate (within ±0.05 of optimal threshold)
- Confidence interval on score: Wide (reflects model uncertainty)

**Attention Weights (this patient):**
- Clinical: 0.27 | Fluid: 0.25 | Motor: 0.25 | Acoustic: 0.23

**Top SHAP Features (this patient):**
1. `apoe4` (+0.08) — homozygous carrier; strong genetic risk
2. `age` (+0.05) — 60–65 band
3. `abeta42_csf` (+0.03) — borderline signal
4. `mmse_baseline` (-0.06) — relatively preserved cognition
5. `ptau217` (+0.01) — minimal contribution

**Clinical Interpretation:**
This case is architecturally important for understanding NeuroFusion-AD's uncertainty boundaries. The score of 0.49 places this patient at the decision boundary where classification is most sensitive to small perturbations in input. The model is essentially uncertain. The SHAP decomposition shows competing forces: strong APOE4 genetic risk (ε4/ε4, the highest-risk genotype) pulling the score upward, partially offset by relatively preserved MMSE.

This patient belongs to the APOE ε4 carrier subgroup, for which the aggregate AUC is 0.501 — barely above chance. The near-chance performance in this subgroup means the model provides clinically unreliable guidance specifically for the patient population at highest genetic risk for AD. This is a paradox with serious clinical implications: the patients most likely to benefit from early detection (APOE ε4/ε4 carriers) are precisely those for whom the model performs worst.

The Navify interface should display an **explicit uncertainty flag** for scores within 0.05 of the decision threshold, and the APOE carrier performance limitation must be communicated in the Instructions for Use (IFU). Clinicians encountering such cases should be directed to rely more heavily on longitudinal biomarker trends and specialist consultation rather than the NeuroFusion-AD score alone.

---

# SECTION 7: SUBGROUP FAIRNESS ANALYSIS

## 7.1 Overview and Regulatory Basis

Subgroup fairness analysis is required under FDA guidance on AI/ML-based SaMD (2021), the EU MDR Annex XIV post-market clinical follow-up requirements, and the principles of health equity established by the NIH and NASEM. NeuroFusion-AD is intended for use across a broad population of MCI patients aged 50–90. Differential performance across demographic subgroups could lead to systematic underdiagnosis or overdiagnosis in specific populations, amplifying existing health disparities in Alzheimer's disease care — a field in which underdiagnosis among women, minority populations, and lower-education patients is already well-documented.

The pre-specified fairness threshold for this validation is **maximum AUC gap ≤ 0.07** across any pairwise subgroup comparison within a single demographic axis. This threshold was established during the study protocol development phase (Protocol NFD-PROT-2024-001) based on precedent from FDA-cleared cardiovascular AI tools and the NHS AI fairness framework.

---

## 7.2 Subgroup AUC Results — ADNI Test Set

### Table 7.1: AUC by Age Group — ADNI Test Set (N=100)

| Age Group | N | AUC | 95% CI (lower) | 95% CI (upper) | Interpretation |
|---|---|---|---|---|---|
| Age < 65 | 33 | 0.516 | 0.321 | 0.723 | Near-chance; very wide CI |
| Age 65–75 | 34 | 0.618 | 0.414 | 0.762 | Modest; best age group |
| Age > 75 | 33 | 0.573 | 0.370 | 0.783 | Below acceptable threshold |
| **Max pairwise gap** | — | **0.102** | — | — | ⚠️ Exceeds 0.07 threshold |

> **Finding:** The age group analysis reveals a 10.2-point AUC gap between the best-performing (65–75: AUC=0.618) and worst-performing (age <65: AUC=0.516) groups. This gap exceeds the pre-specified 0.07 threshold. However, the wide confidence intervals (overlapping substantially across all three groups) indicate that sample sizes at n≈33 per cell are insufficient to draw definitive conclusions. The apparent gap may reflect sampling variation rather than systematic differential performance. Nonetheless, the point estimate gap is flagged as a finding requiring longitudinal monitoring.

### Table 7.2: AUC by Sex — ADNI Test Set (N=100)

| Sex | N | AUC | 95% CI (lower) | 95% CI (upper) | Interpretation |
|---|---|---|---|---|---|
| Male | 44 | 0.543 | 0.340 | 0.705 | Below acceptable threshold |
| Female | 56 | 0.592 | 0.414 | 0.731 | Below acceptable threshold |
| **Max pairwise gap** | — | **0.049** | — | — | ✅ Within 0.07 threshold |

> **Finding:** The male-female AUC gap of 0.049 is within the pre-specified 0.07 fairness threshold. However, both groups exhibit AUCs near or below 0.60, indicating broadly poor model performance for both sexes in this cohort. The slightly higher female AUC (0.592 vs. 0.543) is clinically non-significant given overlapping confidence intervals. No sex-based differential harm is detectable at this sample size, but the universal underperformance warrants attention.

### Table 7.3: AUC by APOE Status — ADNI Test Set (N=100)

| APOE Status | N | AUC | 95% CI (lower) | 95% CI (upper) | Interpretation |
|---|---|---|---|---|---|
| Non-carrier (ε2/ε3, ε3/ε3) | 42 | 0.690 | 0.530 | 0.833 | Acceptable range |
| Carrier (ε3/ε4, ε4/ε4) | 58 | 0.501 | 0.369 | 0.650 | Near-chance performance |
| **Max pairwise gap** | — | **0.189** | — | — | ❌ Substantially exceeds threshold |

> **Finding:** The APOE carrier vs. non-carrier AUC gap of **0.189** is the most clinically alarming finding in this validation. APOE ε4 carrier status is the strongest known genetic risk factor for late-onset AD, meaning the subgroup for whom accurate risk stratification is most clinically impactful is precisely the subgroup for which the model performs worst (AUC=0.501, effectively at chance). This is the primary driver of the **fairness_pass: false** determination.

---

## 7.3 Aggregate Fairness Assessment

| Demographic Axis | Max AUC Gap | Threshold | Status |
|---|---|---|---|
| Age (3-group) | 0.102 | ≤ 0.070 | ❌ FAIL |
| Sex (2-group) | 0.049 | ≤ 0.070 | ✅ PASS |
| APOE Status (2-group) | 0.189 | ≤ 0.070 | ❌ FAIL |
| **Overall Fairness Determination** | **0.189 (max)** | **≤ 0.070** | **❌ FAIL** |

**Overall Result: `fairness_pass: false`**

The model fails the pre-specified fairness criteria on two of three demographic axes. The APOE status gap of 0.189 represents a 2.7× exceedance of the 0.07 threshold and constitutes a **Class 1 performance limitation** requiring explicit disclosure in the Instructions for Use and on the Navify interface.

---

## 7.4 Health Equity Implications

### 7.4.1 APOE Carrier Performance

The near-chance performance (AUC=0.501) for APOE ε4 carriers has a plausible mechanistic explanation: APOE ε4 carriers may have a different amyloid progression trajectory than non-carriers, with steeper and more variable pathological onset timescales. If the training data (ADNI N=345) did not provide sufficient longitudinal coverage of APOE carrier progression, the model may have learned biomarker patterns predominantly from non-carriers and generalized poorly to the carrier subgroup. Additionally, the assay proxy issue (CSF pTau181 used instead of plasma pTau217) may differentially affect pTau signal interpretation across APOE genotypes — a phenomenon documented in the pTau assay literature.

### 7.4.2 Younger Patient Performance

The near-chance performance for patients under 65 (AUC=0.516) raises concerns about the model's utility in younger-onset MCI cases. Earlier-onset amyloid pathology may have different biomarker signatures, and the ADNI training cohort's demographic composition (predominantly older patients) may underrepresent this population. The Bio-Hermes-001 dataset's 24% representation from underrepresented communities provides broader demographic coverage for the fine-tuned model, but subgroup-level AUC for the Bio-Hermes validation was not decomposed by age or APOE status (cross-sectional dataset; outcome labels unavailable for all participants).

### 7.4.3 Implications for Deployment

Given the fairness analysis results, the following deployment restrictions are recommended:

1. **APOE ε4 Carriers:** The Navify interface must display a prominent warning when a patient is identified as an APOE ε4 carrier, indicating that model performance is significantly reduced in this subgroup and that the risk score should be considered with substantially reduced confidence.

2. **Patients Under 65:** The risk score output should include a low-confidence flag for patients aged 50–64, given point-estimate AUC of 0.516 and wide confidence intervals.

3. **Post-Market Surveillance:** A targeted fairness re-evaluation must be conducted on the first 500 real-world deployment cases, stratified by APOE status, within 12 months of initial deployment.

4. **Model Development Roadmap:** Version 2.0 of NeuroFusion-AD should prioritize training data enrichment for APOE ε4 carriers and younger-onset MCI patients. Collaboration with APOE-specific cohort studies (e.g., PREVENT Dementia, DIAN) is recommended.

---

# SECTION 8: CALIBRATION ANALYSIS

## 8.1 Overview

Calibration refers to the correspondence between a model's predicted probability and the empirically observed frequency of the outcome. A well-calibrated model that predicts 70% probability for a set of patients should have approximately 70% of those patients testing positive for amyloid progression. Calibration is distinct from discrimination (AUC) — a model can discriminate well (high AUC) but be systematically over- or underconfident in its probability estimates, or vice versa.

For a Clinical Decision Support tool, calibration has direct clinical consequences. Overconfident predictions (predicted probability >> true probability) may cause clinicians to pursue unnecessary invasive follow-up (CSF lumbar puncture, PET imaging) in low-risk patients. Underconfident predictions (predicted probability << true probability) may fail to trigger appropriate clinical action in high-risk patients. Expected Calibration Error (ECE) is the primary calibration metric used in this analysis.

**ECE Definition:** ECE is computed by partitioning predictions into M equal-width bins (M=10 used here), computing the absolute difference between mean predicted probability and observed positive fraction within each bin, and taking the weighted average across bins.

---

## 8.2 Calibration Results

### Table 8.1: Expected Calibration Error — ADNI Test Set (N=100)

| Calibration Stage | ECE | Interpretation |
|---|---|---|
| **Pre-calibration (raw model output)** | **0.2001** | Severe miscalibration |
| **Post-calibration (temperature scaling, T=3.30)** | **0.0210** | Acceptable calibration |
| **Reduction** | 0.1791 (89.5% reduction) | Temperature scaling highly effective |
| **Temperature parameter (T)** | **3.30** | Indicates substantial pre-calibration overconfidence |

### 8.2.1 Pre-Calibration Analysis

The raw model ECE of **0.2001** indicates severe miscalibration. A temperature parameter of T=3.30 required to achieve acceptable ECE confirms that the model was substantially **overconfident** in its pre-calibration state — producing predicted probabilities concentrated near 0 and 1, while the true positive fraction remained closer to the base rate. This pattern is characteristic of models trained with cross-entropy loss on imbalanced datasets, and is further exacerbated by the presence of features with high information content relative to the label (specifically `abeta42_csf`, which carries near-direct label signal).

The pre-calibration ECE of 0.2001 would be clinically unacceptable for deployment. A physician receiving an 85% risk score from an uncalibrated model, where the true positive rate for patients in that predicted range is approximately 50%, would be systematically misled about the certainty of the prediction.

### 8.2.2 Post-Calibration Analysis

Temperature scaling with T=3.30 reduces ECE to **0.0210**, representing an 89.5% improvement. Post-calibration ECE of 0.021 is within the clinically acceptable range (threshold: ECE ≤ 0.05 per internal quality criteria). Temperature scaling is a post-hoc calibration method that applies a single learned scalar to the model's logits before softmax conversion, rescaling the entire output distribution without modifying the model's rank-ordering of patients (discrimination is preserved).

### 8.2.3 Important Caveats on Temperature Scaling

While the post-calibration ECE is acceptable, several limitations of the temperature scaling approach must be acknowledged:

1. **Calibration set dependency:** The temperature parameter T=3.30 was derived from the ADNI validation set (N=74). If the deployment population has different base rates or feature distributions than the ADNI cohort, the calibration may degrade. The Bio-Hermes-001 population (different recruitment, different assays, 24% underrepresented communities) may require a different temperature parameter. A separate calibration evaluation on Bio-Hermes-001 data is recommended but was not performed in this validation cycle due to the cross-sectional nature of that dataset (no outcome labels for all participants).

2. **Subgroup calibration not assessed:** ECE was computed at the aggregate level (N=100). Calibration may be acceptable overall while remaining poor within specific subgroups (e.g., APOE ε4 carriers). Subgroup-level calibration analysis is recommended in the Version 2.0 validation cycle.

3. **Temporal stability:** Temperature scaling is a static parameter derived from a fixed validation set. Concept drift in deployment data (e.g., changing patient population, updated assay protocols) could invalidate the calibration without triggering any automated alert. The post-market surveillance plan must include periodic calibration monitoring using real-world data.

---

## 8.3 Clinical Implications of Miscalibration in CDS

The pre-calibration ECE of 0.2001 provides a useful illustration of the clinical stakes of deploying miscalibrated AI tools. In a population with a 50% base rate of amyloid progression (approximate in the ADNI MCI cohort), an uncalibrated model producing a predicted probability of 0.82 for a patient might correspond to a true positive fraction of only 0.60 in that predicted bin. If a clinician uses the score to justify ordering an [18F]-florbetapir amyloid PET (approximate cost: $3,500–$6,000; radiation exposure: ~7 mSv), and 40% of patients in that predicted bin are actually amyloid negative, the clinical and economic harm is substantial.

Post-calibration, a score of 0.82 would more accurately reflect approximately 80% true positive fraction, enabling more appropriate clinical thresholding. The post-deployment requirement to maintain ECE ≤ 0.05 using rolling calibration monitoring is therefore classified as a **safety-critical requirement** in the NeuroFusion-AD risk register (ISO 14971 Risk ID: CAL-001).

---

# SECTION 9: LIMITATIONS

## 9.1 Overview

This section presents a comprehensive, structured account of the known limitations of NeuroFusion-AD as validated in this report. Transparent limitation disclosure is required by FDA guidance (Predetermined Change Control Plan, AI/ML Action Plan 2021), EU MDR Article 61, and the principles of responsible AI deployment in healthcare. All limitations documented here are propagated to the Instructions for Use, the Navify interface warning system, and the clinician training materials.

Limitations are classified by severity and type. **Mandatory limitations** (ML prefix) are those required to be disclosed per study protocol and regulatory guidance. **Technical limitations** (TL prefix) reflect implementation and data constraints. **Operational limitations** (OL prefix) affect real-world deployment.

---

## 9.2 Mandatory Limitations

### ML-001: Synthesized Acoustic and Motor Features (DRD-001)

**Description:** In the ADNI training and test datasets, acoustic features (voice analysis metrics) and motor features (gait, fine motor coordination metrics) were not directly collected. Instead, these features were synthesized by sampling from clinical distributions derived from published literature and the available ADNI clinical variables (Data Restriction Document DRD-001).

**Impact:** The model's acoustic and motor processing sub-networks were trained on artificial data that may not reflect the statistical properties, variability, or clinical correlations of real-world acoustic and motor measurements. Modality