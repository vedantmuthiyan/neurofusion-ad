---
document_id: fairness-report
generated: 2026-03-11
batch_id: msgbatch_01HZUXhy6DzGoszEMVS44MBf
status: DRAFT — requires human review before submission
---

# FAIRNESS AND BIAS ANALYSIS REPORT
## NeuroFusion-AD Clinical Decision Support System
### Document ID: FAIR-001 v1.0
### Classification: Controlled Document

---

| Field | Value |
|---|---|
| Product | NeuroFusion-AD |
| Version | 1.0 |
| Document ID | FAIR-001 |
| Regulatory Context | FDA De Novo / EU MDR Class IIa |
| Standard | FDA AI/ML Action Plan (2021), FDA Guidance on Predetermined Change Control Plans |
| Author | Clinical Documentation Specialist |
| Status | **DRAFT — Pending Clinical Review** |
| Review Cycle | Annual or upon dataset update |

---

> **⚠ CRITICAL PREFATORY STATEMENT**
>
> The fairness analysis presented in this document reflects results from the ADNI held-out test set (N=100) and the Bio-Hermes-001 external validation cohort (N=189). The overall ADNI test AUC of 0.579 (95% CI: 0.452–0.671) falls below the minimum acceptable performance threshold for a diagnostic aid. All subgroup analyses reported herein are therefore bounded by a low-performing baseline model. Fairness findings must be interpreted alongside the primary performance limitations documented in PERF-001. This system is **not currently cleared for clinical deployment.**

---

## 1. FAIRNESS FRAMEWORK

### 1.1 Regulatory Basis

This framework is constructed in response to:

- FDA's *Action Plan for AI/ML-Based Software as a Medical Device* (January 2021), which identifies bias and equity as explicit post-market concerns
- FDA's *Guidance on Clinical Decision Support Software* (September 2022), Section IV.B, requiring evaluation of intended-use population subgroups
- EU MDR Annex I, §1 (General Safety and Performance Requirements), specifically the obligation to demonstrate absence of unacceptable risk in foreseeable populations
- ISO 14971:2019, Clause 4.4 (hazard identification), applied to algorithmic disparate impact as a foreseeable harm
- ONC's *Health Equity Framework for Health IT*, requiring documentation of performance across race, ethnicity, sex, and age

### 1.2 Fairness Definitions Adopted

NeuroFusion-AD adopts the following operational definitions of fairness for clinical decision support systems:

| Fairness Criterion | Definition | Rationale |
|---|---|---|
| **Equalized Opportunity** | Sensitivity ≤ 15 percentage points absolute difference across subgroups | Missing a true amyloid-positive case (false negative) carries the primary clinical harm |
| **Predictive Parity** | PPV ≤ 15 percentage points absolute difference across subgroups | Avoids differential false alarm burden |
| **AUC Parity** | AUC gap ≤ 0.10 between any two subgroups | Threshold-independent discrimination measure |
| **Calibration Parity** | ECE ≤ 0.05 within each subgroup | Ensures confidence scores are reliable across groups |

These criteria are not mutually exclusive. The system must satisfy **all four** to pass a fairness gate. A failure on any single criterion is documented as a residual risk requiring mitigation prior to deployment.

### 1.3 Threshold Justification

The 0.10 AUC gap threshold is derived from:

- Published thresholds in FDA-submitted AI/ML SaMD submissions reviewed in the public literature (Obermeyer et al., 2019; Vokinger et al., 2021)
- The clinical consequence model: in MCI populations, an AUC difference of >0.10 translates to materially different referral rates for downstream PET or CSF testing, which carries both financial and procedural burden
- The ADNI test set sample sizes (n=33–56 per subgroup) are insufficient to power narrower thresholds with statistical confidence; the 0.10 threshold is therefore a minimum bar, not a sufficient one

### 1.4 Metrics Computed vs. Metrics Available

| Metric | ADNI Test | Bio-Hermes-001 Val | Status |
|---|---|---|---|
| AUC by subgroup | ✅ Available | ❌ Not stratified | Partial |
| Sensitivity by subgroup | ❌ Not reported | ❌ Not reported | **Gap — see Section 6** |
| Specificity by subgroup | ❌ Not reported | ❌ Not reported | **Gap** |
| PPV/NPV by subgroup | ❌ Not reported | ❌ Not reported | **Gap** |
| ECE by subgroup | ❌ Not computed | ❌ Not computed | **Gap** |
| Race/ethnicity stratification | ❌ Not available (ADNI) | ❌ Not stratified | **Critical Gap** |

---

## 2. DATASET DIVERSITY ASSESSMENT

### 2.1 ADNI Internal Validation Cohort

| Attribute | Value | Bias Risk |
|---|---|---|
| Total N | 494 (train=345, val=74, test=75; held-out test=100) | — |
| Amyloid label coverage | 63.8% (315/494 with valid CSF Aβ42) | Missingness may be non-random |
| Age range | 50–90 years (intended use population) | Low |
| Racial composition | Predominantly White (~88–92% based on ADNI public demographics) | **HIGH** |
| Ethnic composition | Predominantly non-Hispanic (~95% based on ADNI public demographics) | **HIGH** |
| Recruitment setting | Academic medical centers only | **HIGH** |
| Biofluid assay | CSF pTau181 used as **proxy** for plasma pTau217 | **HIGH** — assay mismatch |
| Acoustic/motor features | **SYNTHESIZED** from clinical distributions (DRD-001) | **CRITICAL** |
| Longitudinal data | Available (multi-visit) | Low |

**ADNI Limitations — Narrative:**

**2.1.1 Racial Homogeneity.** ADNI was recruited predominantly from academic medical centers in the United States and Canada beginning in 2004. The cohort is approximately 88–92% non-Hispanic White. This is not representative of the U.S. MCI population aged 50–90, in which Black Americans have approximately 1.5–2× higher incidence of dementia and Hispanic Americans have elevated risk profiles due to vascular comorbidity burden. Any model trained on ADNI inherits this demographic skew, and performance estimates derived from ADNI do not generalize to these populations without direct validation evidence.

**2.1.2 Academic Center Recruitment Bias.** ADNI participants are systematically more educated (mean education ~15.5 years), more physically healthy at baseline, and more cognitively intact than community-dwelling MCI populations. This selection effect inflates specificity estimates and may depress sensitivity in real-world use, where floor effects on MMSE are more common and comorbidity patterns differ.

**2.1.3 Assay Mismatch (CSF pTau181 → Plasma pTau217).** The intended use of NeuroFusion-AD incorporates plasma pTau217 (Roche Elecsys), which is the assay used in Bio-Hermes-001. ADNI data uses CSF pTau181 as a proxy. These are not equivalent: plasma pTau217 has a different analytical sensitivity (LOD ~0.10 pg/mL Elecsys), different reference ranges, and different relationships to amyloid PET status. The model trained on ADNI therefore learned fluid-biomarker representations that do not directly correspond to the inference-time assay. This constitutes a systematic measurement bias that is not correctable through fine-tuning alone.

**2.1.4 Synthesized Acoustic and Motor Features (DRD-001).** Per deviation record DRD-001, acoustic features (speech prosody, pause rate, phonation duration) and motor features (gait cadence, tremor frequency) were not collected in ADNI and were instead synthesized from published clinical distributions. This means the model's acoustic encoder (attention weight: 0.248) and motor encoder (attention weight: 0.261) were trained on statistically plausible but non-patient-derived data. No subgroup-level synthesis was documented; it is unknown whether synthesis preserved demographic variation in these features. This is the single largest source of construct validity threat in this dataset.

### 2.2 Bio-Hermes-001 External Validation Cohort

| Attribute | Value | Bias Risk |
|---|---|---|
| Total N | 945 (train=756, val=189) | — |
| Underrepresented communities | 24% (≈227 participants) | Low relative to ADNI |
| Biofluid assay | Plasma pTau217, Roche Elecsys | **Target assay — correct** |
| Longitudinal data | **Cross-sectional only** | **HIGH** for temporal claims |
| Subgroup stratification | Not provided in current evaluation run | **Gap** |
| Race/ethnicity breakdown | Not specified beyond "24% underrepresented" | **Gap** |

**Bio-Hermes-001 Strengths:**

The 24% underrepresented community enrollment represents a material improvement over ADNI. This cohort uses the target inference-time assay (Roche Elecsys plasma pTau217), establishing construct validity for the fluid biomarker pathway. The external validation AUC of 0.829 (95% CI: 0.780–0.870) is substantially higher than the ADNI test AUC of 0.579, which is consistent with the model performing better when the training-to-inference assay mismatch is resolved.

**Bio-Hermes-001 Limitations:**

The dataset is cross-sectional. NeuroFusion-AD outputs include MMSE trajectory prediction (RMSE: 2.23 pts/year) and a survival C-index (0.509), both of which require longitudinal ground truth for meaningful validation. Bio-Hermes-001 cannot validate these outputs. Furthermore, "24% underrepresented communities" is not disaggregated: it is unknown what proportion are Black, Hispanic, Asian, or other groups, making targeted subgroup analysis impossible with current data. The label "Bio-Hermes-002" does not exist; no second cohort is available.

---

## 3. SUBGROUP PERFORMANCE TABLE

All values from ADNI held-out test set (N=100). Bio-Hermes-001 subgroup breakdowns were not available in the current evaluation run.

### 3.1 Primary Subgroup Results

| Subgroup | N | AUC | 95% CI Lower | 95% CI Upper | ΔAUCᵃ | Fairness Pass (≤0.10 gap)? |
|---|---|---|---|---|---|---|
| **Overall** | 100 | 0.579 | 0.452 | 0.671 | — | — |
| Age < 65 | 33 | 0.516 | 0.321 | 0.723 | −0.063 | ✅ (vs. overall) |
| Age 65–75 | 34 | 0.618 | 0.414 | 0.762 | +0.039 | ✅ (vs. overall) |
| Age > 75 | 33 | 0.573 | 0.370 | 0.783 | −0.006 | ✅ (vs. overall) |
| **Max age gap (65–75 vs. <65)** | — | — | — | — | **0.102** | ❌ **FAIL** |
| Sex: Male | 44 | 0.543 | 0.340 | 0.705 | — | — |
| Sex: Female | 56 | 0.592 | 0.414 | 0.731 | — | — |
| **Sex gap (F vs. M)** | — | — | — | — | **0.050** | ✅ |
| APOE4 Non-carrier | 42 | 0.690 | 0.530 | 0.833 | — | — |
| APOE4 Carrier | 58 | 0.501 | 0.369 | 0.650 | — | — |
| **APOE4 gap (Non-carrier vs. Carrier)** | — | — | — | — | **0.189** | ❌ **FAIL** |

ᵃ ΔAUC computed as subgroup AUC minus reference group AUC within each pair. Max gap = 0.189 (APOE4 status), system-reported `max_auc_gap` = 0.1885521885521887.

**System-level fairness gate: `fairness_pass = false`**

### 3.2 Interpretation

**APOE4 Carrier vs. Non-Carrier Disparity (ΔAUC = 0.189).**
This is the most clinically consequential finding. APOE4 carriers constitute 58% of the ADNI test subgroup and represent the highest-risk MCI population for amyloid progression. The model performs at near-chance for APOE4 carriers (AUC = 0.501, CI: 0.369–0.650), while achieving moderate discrimination in non-carriers (AUC = 0.690, CI: 0.530–0.833). The paradox — poorer performance in higher-risk individuals — suggests the model may be relying on APOE4 status as a shortcut feature while failing to integrate other biomarker signals meaningfully for this subgroup. SHAP analysis confirms APOE4 ranks as a top-5 feature, consistent with this interpretation.

**Age Subgroup Gap (ΔAUC = 0.102).**
The gap between the 65–75 age band (AUC = 0.618) and the under-65 group (AUC = 0.516) marginally exceeds the 0.10 threshold. Confidence intervals are wide due to small subgroup sizes (n=33–34), and this finding should be interpreted cautiously. However, the youngest subgroup (50–64 years) is precisely the population where early intervention is most valuable; near-chance performance in this group is a clinical safety concern independent of the statistical threshold analysis.

**Sex Subgroup (ΔAUC = 0.050).**
The sex difference (Female: 0.592, Male: 0.543) falls within the acceptable threshold. However, sensitivity and specificity are not stratified by sex in the current output, and PPV/NPV are not reported. This gap cannot be fully characterized with AUC alone.

**Race/Ethnicity: Not Analyzed.**
No race or ethnicity subgroup AUC is available from either dataset in the current evaluation run. This constitutes a critical fairness documentation gap given the known racial disparities in Alzheimer's disease incidence, biomarker expression, and care access. This gap must be resolved before any regulatory submission.

---

## 4. KNOWN BIAS SOURCES

Each source below is classified by origin type, mechanism of effect, and magnitude assessment.

### 4.1 Bias Source Table

| ID | Source | Type | Mechanism | Estimated Magnitude | Remediable? |
|---|---|---|---|---|---|
| BS-01 | ADNI racial homogeneity | **Training data bias** | Model learns biomarker-outcome relationships calibrated to White participants; may not generalize across populations with different amyloid kinetics or baseline biomarker distributions | HIGH | Partial (Bio-Hermes fine-tuning) |
| BS-02 | Synthesized acoustic/motor features (DRD-001) | **Construct validity bias** | Synthetic data may not preserve demographic or disease-stage variation in speech/gait; model learns phantom correlations with no real patient source | CRITICAL | No — requires real data collection |
| BS-03 | CSF pTau181 → plasma pTau217 assay mismatch | **Measurement bias** | Training fluid features do not correspond to inference-time assay; systematic shift in feature space | HIGH | Partial (Bio-Hermes fine-tuning) |
| BS-04 | APOE4 shortcut learning | **Algorithmic bias** | Model over-relies on APOE4 carrier status, achieving near-chance AUC in carriers despite APOE4 being a known risk factor | HIGH | Requires retraining |
| BS-05 | ABETA42_CSF leakage in ADNI validation | **Evaluation bias** | ADNI val_auc = 1.0 during training, attributed to ABETA42_CSF feature; training signal was contaminated by near-perfect label proxy | HIGH | Mitigated in held-out test; not in training |
| BS-06 | Academic center selection bias | **Sampling bias** | Higher education, lower vascular comorbidity burden, greater cognitive reserve in ADNI vs. real-world MCI population; performance may degrade in community settings | MODERATE | Partial (Bio-Hermes) |
| BS-07 | Amyloid label missingness (36.2% unlabeled) | **Label bias** | If missingness correlates with disease severity, age, or socioeconomic status, the labeled subset is non-representative; model trained on a biased outcome sample | MODERATE | Requires missingness analysis |
| BS-08 | Cross-sectional Bio-Hermes validation | **Evaluation bias** | Longitudinal outputs (MMSE trajectory, survival C-index) cannot be validated on Bio-Hermes-001; performance on these outputs is unverified | MODERATE | Requires longitudinal cohort |
| BS-09 | Small subgroup N (n=33–58) | **Statistical bias** | Wide confidence intervals on all subgroup estimates preclude definitive fairness conclusions | HIGH | Requires larger validation set |
| BS-10 | Fine-tuning on frozen encoders | **Transfer bias** | Acoustic/motor encoders trained on synthetic data are frozen during Bio-Hermes fine-tuning; modality-specific biases cannot be corrected at fine-tuning stage | MODERATE | Requires full retraining |

### 4.2 Detailed Explanation of Critical Sources

**BS-02 — Synthesized Acoustic/Motor Features.**
The acoustic and motor modalities collectively receive 50.9% of total mean attention weight (acoustic: 0.248, motor: 0.261). These are the two highest-weighted modalities in the attention fusion layer. Both were trained entirely on synthetic data. This means the model's two primary input pathways at inference were never calibrated on real patient speech or gait recordings. The clinical correlations these features are intended to capture — phonation changes in MCI, gait variability as a motor-cognitive biomarker — may not be faithfully represented in the synthetic distributions. This is not a minor data quality issue; it is a fundamental validity threat to the model's input representation.

**BS-04 — APOE4 Shortcut Learning.**
APOE4 carrier status is listed in the top-5 SHAP features. APOE4 carriers have a known higher a priori probability of amyloid positivity. A model that learns to use APOE4 as a primary signal will assign high risk scores to most carriers regardless of other biomarker state. The result is a compressed score distribution for carriers — high scores for all carriers — which destroys discrimination within the carrier subgroup. AUC = 0.501 in carriers is consistent with this failure mode. The clinical consequence is that the model provides no useful prognostic discrimination for the highest-risk patient segment.

**BS-05 — ABETA42_CSF Training Leakage.**
The W&B training log (run: jehkd9ud → ybbh5fky) recorded val_auc = 1.0 during ADNI training. This is attributable to ABETA42_CSF being included in the training feature set when it is also the basis for the amyloid progression label. This constitutes target leakage: the model was optimizing a signal that was functionally identical to the label during training. The held-out test AUC of 0.579 represents the true generalization performance after this feature's contribution is assessed on an independent split. However, the training run itself may have learned feature interaction weights that are distorted by this leakage, meaning the learned representations in other modalities may be suboptimal.

---

## 5. MITIGATIONS IMPLEMENTED

| Mitigation | Implementation Detail | Addresses Bias Source(s) | Evidence of Effectiveness |
|---|---|---|---|
| **Bio-Hermes-001 external validation** | 945-participant cohort, 24% underrepresented communities, target assay (Roche Elecsys pTau217) | BS-01, BS-03, BS-06 | AUC improved from 0.579 → 0.829; assay mismatch largely resolved |
| **Bio-Hermes fine-tuning** | Frozen encoders, classification loss only, lr=5e-5, early stopping at epoch 17 | BS-01, BS-03 | Validated by AUC 0.829 on Bio-Hermes val set |
| **ADNI held-out test set (N=100, independent split)** | Separated from training before HPO; prevents evaluation bias from hyperparameter selection | BS-05 | True AUC = 0.579, exposing leakage not visible in val_auc |
| **Temperature scaling calibration** | T=3.30; ECE reduced from 0.2001 to 0.0210 overall | Calibration parity (prerequisite for PPV/NPV parity) | ECE 0.0210 meets overall threshold; subgroup ECE not yet computed |
| **Optuna HPO (30 trials)** | Systematic hyperparameter search to reduce optimization variance | BS-09 (reduces variance but not sample size) | Best config selected at epoch 150 |
| **Fairness gate in evaluation pipeline** | `fairness_pass` flag computed from max_auc_gap; blocks deployment if >0.10 | BS-04, BS-09 | Gate correctly returned `false` in current evaluation |
| **Modality attention weighting** | Learned attention fusion allows model to down-weight unreliable modalities at inference | BS-02, BS-10 (partial) | Acoustic/motor weights (0.248, 0.261) are not suppressed — mitigation is incomplete |
| **Amyloid label coverage documentation** | Missingness rate (36.2%) documented in PERF-001 and here | BS-07 | Documentation only; missingness mechanism not yet characterized |

**Mitigation Effectiveness Summary:**

The most effective mitigation was Bio-Hermes-001 external validation and fine-tuning, which resolved the assay mismatch and improved AUC by 0.250 absolute points. The least effective mitigation was the modality attention weighting, which did not suppress the synthetic feature encoders — the motor modality (synthetic) retains the highest attention weight in the model (0.261), indicating the model is actively using the least trustworthy input pathway.

---

## 6. RESIDUAL BIAS RISKS

The following risks remain unmitigated after all implemented controls. Each is assigned a risk level per ISO 14971 using a severity × probability framework.

### 6.1 Residual Risk Register

| Risk ID | Description | Severity | Probability | Risk Level | Acceptability |
|---|---|---|---|---|---|
| RR-01 | Model provides near-chance discrimination for APOE4 carriers (AUC=0.501), the highest-risk MCI subpopulation | Critical | High | **UNACCEPTABLE** | ❌ Not acceptable |
| RR-02 | No race/ethnicity subgroup validation available; unknown performance in Black and Hispanic MCI patients | Critical | High | **UNACCEPTABLE** | ❌ Not acceptable |
| RR-03 | Acoustic and motor encoders trained on synthetic data; input validity unverified for any real patient population | Serious | High | **UNACCEPTABLE** | ❌ Not acceptable |
| RR-04 | MMSE trajectory (RMSE=2.23) and survival C-index (0.509) validated on ADNI only; no longitudinal validation in Bio-Hermes | Serious | High | **Unacceptable** | ❌ Not acceptable |
| RR-05 | Subgroup ECE not computed; calibration parity unknown across age, sex, APOE4 | Moderate | High | **High** | ❌ Not acceptable for deployment |
| RR-06 | Young-onset MCI subgroup (age <65, AUC=0.516) performs at near-chance; intervention window most valuable in this group | Serious | Moderate | **High** | ❌ Not acceptable |
| RR-07 | Bio-Hermes-001 "24% underrepresented" not disaggregated; cannot confirm performance for specific racial/ethnic groups | Serious | Moderate | **High** | ❌ Not acceptable |
| RR-08 | APOE4 shortcut pattern may persist in Bio-Hermes fine-tuned model; not tested due to lack of APOE4-stratified Bio-Hermes results | Moderate | Moderate | **Moderate** | Requires monitoring |
| RR-09 | Amyloid label missingness (36.2%) mechanism unknown; non-random missingness could bias all performance estimates | Moderate | Moderate | **Moderate** | Requires analysis |

**Overall Residual Risk Assessment: UNACCEPTABLE for clinical deployment.** Three residual risks are classified as unacceptable under ISO 14971 criteria. No benefit-risk justification can be constructed that would support deployment given the absence of race/ethnicity validation data and APOE4 carrier performance at chance level.

---

## 7. POST-MARKET MONITORING PLAN FOR BIAS DETECTION

This section is written prospectively. It describes the monitoring program that would be activated upon a future clearance event. It does not constitute authorization for current deployment.

### 7.1 Monitoring Governance

| Role | Responsibility |
|---|---|
| Medical Director (Neurology) | Clinical review of performance alerts; final decision on action triggers |
| ML Engineering Lead | Dashboard maintenance, drift detection implementation |
| Clinical Documentation Specialist | Regulatory reporting, FAIR-001 update authorship |
| Biostatistician | Subgroup analysis, confidence interval computation |
| Patient Advocacy Representative | Review of equity reports; community stakeholder communication |

Review cadence: **Quarterly operational review**, **Annual comprehensive fairness reassessment**, **Immediate review upon any triggered alert**.

### 7.2 Data Collection Requirements

At each site of deployment, the following data must be collected prospectively for monitoring:

- Patient demographics: age, self-reported race, self-reported ethnicity, sex, APOE4 status (if available)
- Model outputs: risk score (continuous), binary threshold classification, modality confidence weights
- Clinical outcome (when available, 12–24 month follow-up): amyloid PET result, CSF Aβ42, clinical diagnosis
- Clinician action: whether clinician overrode model output, referral decision
- Site metadata: academic vs. community setting, geographic region

Minimum N for quarterly subgroup analysis: 50 per demographic subgroup. Sites falling below this threshold will have data pooled quarterly for regional analysis.

### 7.3 Performance Drift Triggers

The following thresholds will trigger an immediate clinical safety review:

| Metric | Monitoring Frequency | Alert Threshold | Action |
|---|---|---|---|
| Overall AUC (rolling 90-day) | Monthly | Drop >0.05 from deployment baseline | Immediate safety review; consider temporary suspension |
| APOE4 carrier AUC | Quarterly | AUC <0.55 sustained over two quarters | Mandatory model update cycle |
| Max subgroup AUC gap | Quarterly | Gap >0.15 across any two subgroups | Bias investigation report within 30 days |
| Race/ethnicity AUC parity | Quarterly | AUC gap >0.10 between any two racial groups | Escalation to Medical Director and Regulatory |
| Calibration ECE by subgroup | Quarterly | ECE >0.05 in any subgroup | Recalibration review |
| Clinician override rate by subgroup | Monthly | Override rate >30% in any demographic group | Clinical usability review |
| Model score distribution shift | Monthly | KL divergence >0.15 vs. deployment baseline | Covariate drift investigation |

### 7.4 Race/Ethnicity Monitoring Protocol

Given the absence of validated race/ethnicity performance data at the time of initial filing, a prospective race/ethnicity monitoring commitment is required as a condition of any De Novo clearance:

1. **Month 0–6 post-deployment:** Collect demographic data from ≥500 patients across ≥3 sites, targeting ≥100 Black patients and ≥100 Hispanic patients
2. **Month 6:** Submit interim race/ethnicity stratified AUC report to FDA
3. **Month 12:** Submit full subgroup validation report; if any racial/ethnic AUC gap >0.10, submit corrective action plan within 60 days
4. **Ongoing:** Include race/ethnicity AUC in all annual performance reports

### 7.5 Feedback Loop and Model Update Protocol

- **Labeling pipeline:** Sites with amyloid PET confirmation will contribute outcome labels to a continuously growing labeled dataset
- **Retraining trigger:** If ≥500 new labeled outcomes are available AND a performance drift alert has been sustained for >90 days, a model retraining cycle is initiated per the Predetermined Change Control Plan (PCCP-001)
- **Fairness gate on all retrains:** Every retrained model must pass the four fairness criteria in Section 1.2 before deployment; `fairness_pass = false` blocks release
- **Synthetic feature replacement:** Priority action is collection of real acoustic and motor data from ≥200 MCI patients to replace DRD-001 synthetic features; target completion 18 months post-deployment

### 7.6 Reporting Obligations

| Report | Recipient | Frequency | Trigger |
|---|---|---|---|
| Quarterly Equity Report | Internal Safety Committee | Quarterly | Scheduled |
| Annual Bias Assessment (FAIR-001 update) | FDA, Notified Body | Annual | Scheduled |
| Expedited Safety Report | FDA (MedWatch / 510k post-market) | As needed | Any unacceptable residual risk identified |
| Site Performance Report | Deploying institution | Quarterly | Scheduled |
| Patient Community Brief | Advocacy partners | Semi-annual | Scheduled |

---

## 8. SUMMARY AND DISPOSITION

### 8.1 Fairness Gate Status

| Gate | Result |
|---|---|
| AUC parity (≤0.10 gap) | ❌ **FAIL** — Max gap = 0.189 (APOE4 status) |
| Age subgroup parity | ❌ **FAIL** — Gap = 0.102 (65–75 vs. <65) |
| Sex subgroup parity | ✅ PASS — Gap = 0.050 |
| Race/ethnicity parity | ❌ **NOT EVALUABLE** — No data |
| Calibration parity | ❌ **NOT EVALUABLE** — Subgroup ECE not computed |
| Overall fairness disposition | ❌ **FAIL** |

### 8.2 Required Actions Before Next Regulatory Submission

| Priority | Action | Owner | Target Date |
|---|---|---|---|
| P0 | Collect real acoustic and motor data; retire DRD-001 synthetic features | ML Engineering | Prior to next training cycle |
| P0 | Obtain race/ethnicity-stratified performance data from ≥100 Black and ≥100 Hispanic patients | Clinical Operations | Prior to De Novo submission |
| P0 | Investigate and remediate APOE4 carrier near-chance performance (AUC=0.501) | ML Engineering + Clinical | Prior to De Novo submission |
| P1 | Compute subgroup ECE for all demographic groups | Biostatistics | 60 days |
| P1 | Characterize amyloid label missingness mechanism | Biostatistics | 60 days |
| P1 | Obtain longitudinal validation cohort for MMSE trajectory and survival outputs | Clinical Partnerships | 6 months |
| P2 | Disaggregate Bio-Hermes-001 "24% underrepresented" by specific race/ethnicity | Data Governance | 30 days |
| P2 | Compute sensitivity/specificity/PPV/NPV by all subgroups | Biostatistics | 60 days |

---

*Document prepared by: Clinical Documentation Specialist, NeuroFusion-AD Program*
*Next scheduled review: Upon completion of P0 action items or 12 months from issue date, whichever is sooner