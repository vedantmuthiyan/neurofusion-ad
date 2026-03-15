---
document_id: fairness-report
generated: 2026-03-15
batch_id: msgbatch_01G4xrs23ARV9Qg7oCHV4nen
status: DRAFT — requires human review before submission
---

# FAIR-001 v1.0
## Fairness and Bias Analysis Report
### NeuroFusion-AD — Clinical Decision Support (SaMD)

---

**Document ID:** FAIR-001
**Version:** 1.0
**Classification:** Regulatory Submission Document
**Applicable Standards:** FDA AI/ML-Based SaMD Action Plan (2021), FDA Guidance on Predetermined Change Control Plans (2023), ISO 14971:2019 Risk Management, EU MDR 2017/745 Annex I (GSPR)
**Regulatory Pathway:** FDA De Novo + EU MDR Class IIa
**Prepared by:** Clinical Documentation Specialist, NeuroFusion-AD Program
**Review Status:** Final — For Regulatory Submission

---

## 1. Fairness Framework

### 1.1 Governing Rationale

NeuroFusion-AD generates amyloid progression risk assessments that directly inform clinical decisions in MCI patients aged 50–90. Biased performance across demographic subgroups could cause systematic harm: false negatives delay access to intervention; false positives expose patients to unnecessary diagnostic workup, anxiety, and cost. Because the device operates in a population where age, sex, and APOE4 carrier status are biological moderators of amyloid pathology — not merely confounders — the fairness framework must distinguish between clinically meaningful biological variation and analytically induced performance disparities.

The framework adopts three principles:

1. **Equal opportunity**: Sensitivity (true positive rate) must not differ materially across demographic subgroups, because missing amyloid progression in any group constitutes direct patient harm.
2. **Predictive parity**: PPV and NPV must remain stable across subgroups so that clinician interpretation of a positive or negative output is consistent regardless of the patient's demographic context.
3. **Performance stability**: AUC differences across subgroups must remain within a pre-specified tolerance; subgroups showing degraded discrimination trigger mandatory mitigation.

### 1.2 Primary Fairness Metrics

| Metric | Definition | Clinical Rationale |
|---|---|---|
| AUC per subgroup | Area under ROC curve | Overall discrimination capacity; primary regulatory endpoint |
| Maximum AUC gap | Max(AUC_subgroup) − Min(AUC_subgroup) across all defined groups | Device-level fairness signal |
| Sensitivity per subgroup | TP / (TP + FN) | Missing amyloid risk = direct harm |
| Specificity per subgroup | TN / (TN + FP) | Unnecessary workup = indirect harm |
| Expected Calibration Error (ECE) per subgroup | Calibration deviation | Probability outputs must be interpretable uniformly |

### 1.3 Fairness Thresholds and Pass/Fail Criteria

The following thresholds are pre-specified in the Risk Management Plan (RMP-001) and are not adjusted post-hoc:

| Metric | Threshold | Basis |
|---|---|---|
| Maximum AUC gap across demographic subgroups | ≤ 0.10 | Adapted from FDA Table of Contents for Algorithmic Bias; threshold consistent with FDA AI/ML Action Plan |
| Minimum subgroup AUC | ≥ 0.80 | Below this, discriminative utility is insufficient for clinical use |
| Sensitivity in any subgroup | ≥ 0.75 | Minimum acceptable to ensure no subgroup faces systematically higher miss rate |
| ECE after calibration in any subgroup | ≤ 0.10 | Probability outputs must support shared decision-making |
| Minimum subgroup N for reportable analysis | N ≥ 10 | Below this, estimates are statistically unstable; reported with explicit caveat |

### 1.4 Protected Attributes Evaluated

- **Age bands**: < 65, 65–75, > 75 (pre-specified; age-related amyloid prevalence increases, so differential performance must be assessed)
- **Sex**: Male, Female
- **APOE4 status**: Carrier (≥1 ε4 allele), Non-carrier
- **Race/Ethnicity** (Bio-Hermes-001 only; ADNI lacks adequate diversity for this analysis — see Section 2)

---

## 2. Dataset Diversity Assessment

### 2.1 ADNI (Internal Validation Cohort)

**Cohort summary:** 494 MCI patients; train N=345, val N=74, test N=75. Amyloid label coverage: 63.8% (315/494); test set: 44/75 labeled.

**Diversity limitations — rated HIGH SEVERITY for regulatory purposes:**

| Limitation | Detail | Impact on Fairness Analysis |
|---|---|---|
| **Racial homogeneity** | ADNI is predominantly Non-Hispanic White (~85–90% historically); specific racial/ethnic breakdown not available at subgroup level in current dataset | Race/ethnicity fairness analysis on ADNI is statistically invalid and is not reported |
| **Socioeconomic homogeneity** | ADNI recruitment through academic medical centers creates selection bias toward higher education, higher health literacy, and geographic proximity to academic centers | Underrepresents patients from rural, lower-income, or limited-English-proficiency populations |
| **Assay mismatch — pTau** | ADNI uses CSF pTau181 as a proxy for plasma pTau217 (Roche Elecsys) — different analyte, different matrix, different assay characteristics | PTAU217 is the top SHAP feature; assay mismatch introduces systematic measurement error that may affect subgroups differentially if CSF access correlates with demographics |
| **Synthesized modalities** | Acoustic and motor features are synthesized from clinical distributions (DRD-001); these are NOT real patient recordings | Synthetic generation preserves aggregate distributions but cannot capture true demographic variation in speech or motor patterns |
| **Amyloid label incompleteness** | 63.8% label coverage overall; 44/75 (58.7%) in test set | Unlabeled patients could differ systematically in age, sex, or APOE4 status; subgroup AUC estimates in ADNI are computed on partially labeled subsets |

**Overall ADNI fairness analysis reliability: LIMITED.** ADNI subgroup results inform hypothesis generation and safety monitoring but cannot constitute primary fairness evidence.

### 2.2 Bio-Hermes-001 (External Validation Cohort)

**Cohort summary:** 945 participants; train N=661, val N=142, test N=142. Uses plasma pTau217 (Roche Elecsys) — the intended clinical assay.

**Diversity strengths:**

| Attribute | Detail |
|---|---|
| **Underrepresented community representation** | 24% of participants from underrepresented communities — materially exceeds ADNI representation |
| **Assay fidelity** | Plasma pTau217 via Roche Elecsys matches the intended deployment assay; no proxy substitution |
| **Geographic distribution** | Multi-site recruitment in Bio-Hermes design; broader socioeconomic range than single-center academic cohort |

**Bio-Hermes-001 limitations:**

| Limitation | Detail | Impact |
|---|---|---|
| **Cross-sectional design** | No longitudinal outcomes available | Cannot assess whether fairness holds across time; progression prediction accuracy by subgroup is unvalidatable in this cohort |
| **Subgroup-level demographic breakdown** | Granular subgroup AUC by race/ethnicity within Bio-Hermes-001 test set (N=142) is underpowered for all subgroups at N ≥ 30 threshold | Specific subgroup AUC CIs are wide; results directionally informative only |
| **"Underrepresented communities" definition** | Composite category; specific racial/ethnic breakdown within the 24% not fully disaggregated in available data | Limits ability to detect differential performance within subgroups (e.g., Black vs. Hispanic vs. Asian) |

**Critical note:** Bio-Hermes-002 does not exist. All external validation references in this document pertain exclusively to Bio-Hermes-001.

---

## 3. Subgroup Performance Analysis

### 3.1 ADNI Test Set Subgroup Results (N=75 total; N_labeled=44)

Results derived from validated evaluation run (W&B run ID: t9s3ngbx). Fairness pass/fail applied against thresholds in Section 1.3.

**Table 3.1 — ADNI Test Set Subgroup AUC**

| Subgroup | N (total) | AUC | 95% CI Lower | 95% CI Upper | Meets AUC ≥ 0.80 Threshold | Notes |
|---|---|---|---|---|---|---|
| Age < 65 | 11 | 1.000 | 1.000 | 1.000 | ✅ PASS | N=11; below N=30 reliability threshold; treat as exploratory |
| Age 65–75 | 40 | 0.865 | 0.699 | 1.000 | ✅ PASS | Largest age subgroup; most reliable estimate |
| Age > 75 | 24 | 0.939 | 0.814 | 1.000 | ✅ PASS | Adequate performance in highest-prevalence age group |
| Sex: Male | 49 | 0.900 | 0.762 | 0.986 | ✅ PASS | |
| Sex: Female | 26 | 0.875 | 0.593 | 1.000 | ✅ PASS | Wide CI due to small N; lower bound approaches threshold |
| APOE4 Carrier | 36 | 0.775 | 0.416 | 1.000 | ❌ **FAIL** | Below 0.80 threshold; extremely wide CI; primary safety concern |
| APOE4 Non-carrier | 39 | 0.906 | 0.726 | 1.000 | ✅ PASS | |

**Table 3.2 — ADNI Fairness Summary**

| Fairness Metric | Observed Value | Threshold | Status |
|---|---|---|---|
| Maximum AUC gap (overall) | 0.225 (Age<65 vs. APOE4 carrier) | ≤ 0.10 | ❌ **FAIL** |
| Maximum AUC gap (excluding N<30 subgroups) | 0.131 (APOE4 carrier 0.775 vs. Non-carrier 0.906) | ≤ 0.10 | ❌ **FAIL** |
| Minimum subgroup AUC | 0.775 (APOE4 carrier) | ≥ 0.80 | ❌ **FAIL** |
| Overall fairness assessment | — | — | ❌ **FAIL** |

**Programmatic flag:** `fairness_pass: false` (confirmed from evaluation output).

**Note on Age < 65 AUC = 1.000:** A perfect AUC in N=11 patients is statistically uninterpretable. This result reflects insufficient sample size rather than genuine perfect discrimination. It is excluded from the max AUC gap calculation for primary fairness assessment and flagged for monitoring.

### 3.2 Bio-Hermes-001 Test Set Performance (N=142)

Granular demographic subgroup AUC within Bio-Hermes-001 test set is not reported at this time due to insufficient N for reliable subgroup estimates at the racial/ethnic level. The aggregate test set performance is:

| Metric | Value | 95% CI |
|---|---|---|
| AUC | 0.907 | 0.855–0.959 |
| Sensitivity | 0.902 | — |
| Specificity | 0.879 | — |
| PPV | 0.807 | — |
| NPV | 0.941 | — |
| F1 | 0.852 | — |

Bio-Hermes subgroup analysis is a **designated post-market priority** (see Section 7) pending dataset expansion to N ≥ 200 per racial/ethnic subgroup.

---

## 4. Known Bias Sources

Each bias source is categorized by type, mechanism, and estimated impact severity.

### 4.1 APOE4 Carrier Performance Degradation

**Type:** Algorithmic bias (feature-label interaction)
**Mechanism:** APOE4 is the fifth-ranked SHAP feature. APOE4 ε4 carriers have higher baseline amyloid burden and steeper progression trajectories, but also more variable biomarker profiles. The Phase 2B HPO used only 15 trials due to budget constraints, which may have underfit the hyperparameter space for subpopulations with complex feature interactions. The APOE4 carrier subgroup AUC of 0.775 falls below the minimum threshold of 0.80. This is the primary identified bias risk.
**Severity:** HIGH
**Affected population size:** ~35–40% of MCI patients carry at least one APOE4 allele — a clinically significant proportion.

### 4.2 CSF pTau181 / Plasma pTau217 Assay Proxy Mismatch (ADNI)

**Type:** Measurement bias
**Mechanism:** PTAU217 is the top-ranked SHAP feature by attention weight and SHAP analysis. In the ADNI training and validation cohort, CSF pTau181 was substituted as a proxy. These are different analytes (different phosphorylation site), different biological matrices (CSF vs. plasma), and different assay platforms. The correlation between CSF pTau181 and plasma pTau217 is moderate but imperfect. Any systematic demographic differences in how this proxy substitution performs (e.g., if CSF availability correlates with sex, age, or race) will introduce correlated errors into training.
**Severity:** MODERATE-HIGH
**Mitigation status:** Partially mitigated — Bio-Hermes-001 fine-tuning uses correct Roche Elecsys plasma pTau217 assay; however, base model weights were initialized from ADNI training.

### 4.3 Synthesized Acoustic and Motor Features (ADNI)

**Type:** Data generation bias
**Mechanism:** Acoustic and motor features in ADNI are synthesized from clinical distributions (per DRD-001). Synthesis preserves aggregate statistical properties but cannot capture true demographic variation in speech patterns, motor signatures, or technology-related performance differences (e.g., older patients may perform differently on digital motor tasks; non-native English speakers show different acoustic feature distributions). Acoustic features carry the second-highest mean attention weight (0.2618); motor features carry the third-highest (0.2396). Biases in these modalities directly affect model output.
**Severity:** MODERATE
**Mitigation status:** NOT MITIGATED in current version. Requires real data collection.

### 4.4 ADNI Racial/Ethnic Homogeneity

**Type:** Training data representation bias
**Mechanism:** ADNI is predominantly Non-Hispanic White. The model's fluid, acoustic, motor, and clinical feature distributions are learned from a racially non-representative cohort. Race-correlated differences in amyloid pathophysiology, biomarker distributions, and clinical presentation (e.g., NFL plasma levels vary with comorbidity burden which correlates with race in the US healthcare context) are not adequately represented in training.
**Severity:** HIGH (cannot be quantified from available data — absence of evidence is not evidence of absence)
**Mitigation status:** PARTIALLY MITIGATED via Bio-Hermes-001 fine-tuning (24% underrepresented communities), but fine-tuning used frozen encoders — base feature representations remain ADNI-derived.

### 4.5 Academic Medical Center Selection Bias

**Type:** Recruitment/sampling bias
**Mechanism:** Both ADNI and Bio-Hermes-001 recruit through academic medical centers and research networks. Patients at these centers are more likely to be highly educated, English-speaking, have reliable transportation, and have healthcare engagement patterns that differ from community-based populations. The Navify Algorithm Suite deployment environment may reach a broader clinical context than the training population.
**Severity:** MODERATE
**Mitigation status:** NOT MITIGATED in current version.

### 4.6 Incomplete Amyloid Label Coverage (ADNI)

**Type:** Label bias
**Mechanism:** Only 63.8% of ADNI patients (315/494) have valid amyloid labels; 44/75 (58.7%) of the test set are labeled. If label missingness is non-random — e.g., if younger patients, patients from lower socioeconomic strata, or patients with milder symptoms are less likely to have undergone CSF collection or amyloid PET — then the labeled training and evaluation population is systematically different from the intended use population.
**Severity:** MODERATE
**Mitigation status:** NOT MITIGATED. Missing label mechanism is unknown and cannot be assumed to be missing-at-random.

### 4.7 Limited HPO Budget (15 Trials)

**Type:** Algorithmic optimization bias
**Mechanism:** The Phase 2B Optuna HPO was constrained to 15 trials due to budget limitations. Best trial validation AUC was 0.9081. Fifteen trials is insufficient to thoroughly explore the hyperparameter space (embed_dim, dropout, learning rate, weight decay, etc.), increasing the risk that the selected configuration is locally optimal for the dominant demographic in the validation set (predominantly White, ADNI) rather than globally optimal across subgroups.
**Severity:** LOW-MODERATE
**Mitigation status:** NOT MITIGATED. Expanded HPO is designated a Phase 3 priority.

---

## 5. Mitigations Implemented

### 5.1 Critical Data Leakage Remediation (Phase 2B)

**Action:** Removal of ABETA42_CSF from fluid features. This analyte had Pearson r = −0.864 with the amyloid label — a near-perfect linear predictor constituting target leakage. Its presence in Phase 2A would have masked true subgroup performance disparities, particularly in subgroups with different CSF collection patterns.
**Fairness impact:** Removal forces the model to learn from genuinely predictive features (PTAU217, NFL_PLASMA, acoustic, motor, clinical), producing fairness estimates that reflect real-world performance rather than label leakage artifacts.
**Status:** COMPLETE. Fluid features = [PTAU217, NFL_PLASMA] only.

### 5.2 Bio-Hermes-001 Fine-Tuning on Diverse Cohort

**Action:** After ADNI-based training, the model was fine-tuned on Bio-Hermes-001 (N=661 train, N=142 val) using a frozen-encoder, classification-only training protocol (W&B run ID: o4pcjy3r). Bio-Hermes-001 includes 24% underrepresented communities and uses the correct pTau217 assay.
**Fairness impact:** Shifts the classification head's decision boundary toward a more diverse population; corrects for some assay proxy mismatch from ADNI pTau181.
**Limitation:** Frozen encoders mean base feature representations remain ADNI-derived. Full fine-tuning was not performed due to overfitting risk on the classification head with reduced model size (~2.24M parameters).
**Status:** COMPLETE.

### 5.3 Calibration via Temperature Scaling

**Action:** Post-hoc probability calibration applied; optimal temperature T = 0.76; ECE reduced from 0.1120 to 0.0831 on ADNI test set.
**Fairness impact:** Well-calibrated probabilities are essential for equitable clinical use — a probability output of 0.80 must mean the same thing regardless of patient demographics. Post-calibration ECE of 0.0831 approaches but does not yet meet the ≤ 0.10 subgroup-level threshold (ECE is reported at aggregate level; subgroup-level calibration is not yet assessed — see Section 6).
**Status:** COMPLETE at aggregate level. Subgroup calibration assessment is PENDING.

### 5.4 Model Size Reduction (Regularization for Generalization)

**Action:** Architecture reduced from embed_dim=768 / ~60M parameters to embed_dim=256 / ~2.24M parameters; dropout increased to 0.4.
**Fairness impact:** Smaller model with higher dropout is less likely to memorize ADNI-specific demographic patterns. Improved generalization across the distribution shift between ADNI and Bio-Hermes-001 (AUC 0.907 on Bio-Hermes vs. 0.890 on ADNI) supports this interpretation.
**Status:** COMPLETE.

### 5.5 Stratified Data Splitting

**Action:** Bio-Hermes-001 used stratified 70/15/15 split; test set held out through all training and hyperparameter selection.
**Fairness impact:** Prevents demographic imbalance in evaluation splits from inflating fairness estimates; test set performance reflects unbiased model behavior.
**Status:** COMPLETE.

### 5.6 SHAP-Based Feature Attribution Transparency

**Action:** SHAP analysis identifies top features as: ptau217, nfl_plasma, mmse_baseline, age, apoe4. This attribution is disclosed in labeling and instructions for use.
**Fairness impact:** Clinical users can contextualize outputs relative to known biological drivers; APOE4 being a top-5 feature informs the APOE4 carrier performance concern in Section 4.1.
**Status:** COMPLETE. Feature attribution included in IFU.

---

## 6. Residual Bias Risks

The following risks remain unresolved after Phase 2B remediation and constitute open items for Phase 3 and post-market surveillance.

### 6.1 APOE4 Carrier AUC Below Threshold [SEVERITY: HIGH — UNRESOLVED]

**Residual risk:** AUC = 0.775 in APOE4 carriers on ADNI test set (N=36). This falls below the pre-specified minimum of 0.80. While the confidence interval is wide (0.416–1.000) due to small N, the point estimate constitutes a threshold failure.
**Current control:** Mandatory disclosure in IFU: "Performance in APOE4 ε4 carriers has not been validated to the same standard as non-carriers. Clinical judgment is particularly important in this subgroup."
**Residual harm potential:** APOE4 carriers have the highest a priori risk of amyloid progression — exactly the subgroup where accurate risk stratification is most clinically consequential. Degraded performance in this group is not a low-stakes limitation.
**Path to resolution:** Phase 3 expanded dataset with enriched APOE4 carrier representation; full (unfrozen) fine-tuning on diverse cohort; expanded HPO budget.

### 6.2 Race/Ethnicity Subgroup Performance Unquantified [SEVERITY: HIGH — UNRESOLVED]

**Residual risk:** No validated subgroup AUC estimates by race/ethnicity exist for either ADNI (inadequate diversity) or Bio-Hermes-001 (insufficient N per subgroup). The model may perform materially differently across racial/ethnic groups and this cannot currently be detected.
**Current control:** Deployment restricted to "aid assessment" role — CDS output is non-binding; clinician retains final decision authority. Explicit IFU disclosure of unvalidated race/ethnicity performance.
**Residual harm potential:** If Black or Hispanic MCI patients systematically receive lower risk scores due to training data underrepresentation, downstream amyloid workup rates in these groups will be lower, replicating existing healthcare disparities in AD diagnosis.
**Path to resolution:** Prospective diverse data collection; minimum N=200 per racial/ethnic subgroup prior to Phase 3 primary fairness analysis.

### 6.3 Synthesized Acoustic and Motor Feature Bias [SEVERITY: MODERATE — UNRESOLVED]

**Residual risk:** Acoustic (attention weight 0.2618) and motor (0.2396) features account for ~50% of model attention yet are entirely synthesized from clinical distributions in ADNI. Real patient speech and motor data from diverse populations may produce feature distributions that the model has never encountered.
**Current control:** DRD-001 data risk disclosure; IFU advisory that acoustic and motor inputs require validated collection hardware and clinical protocols.
**Residual harm potential:** Systematic misclassification for patients with accents, non-English primary language, motor comorbidities (e.g., essential tremor), or technology-naive populations.
**Path to resolution:** Real acoustic and motor data collection across diverse populations; adversarial testing with non-standard acoustic inputs.

### 6.4 Subgroup-Level Calibration Not Assessed [SEVERITY: MODERATE — UNRESOLVED]

**Residual risk:** ECE of 0.0831 is reported at aggregate level only. Calibration may be systematically miscalibrated for specific subgroups (e.g., older patients, APOE4 carriers) even if aggregate ECE appears acceptable. A model with ECE=0.08 overall but ECE=0.25 in APOE4 carriers would have clinically meaningful probability interpretation failures.
**Current control:** None specific to subgroup calibration.
**Path to resolution:** Subgroup-level ECE assessment in Phase 3 validation; subgroup-specific temperature scaling if warranted.

### 6.5 Temporal Fairness Unassessed [SEVERITY: MODERATE — UNRESOLVED]

**Residual risk:** Bio-Hermes-001 is cross-sectional only. Whether performance differences across demographic subgroups emerge or widen over longitudinal follow-up is unknown. Survival C-index of 0.651 (95% CI: 0.525–0.788) on ADNI is modest; subgroup-level C-index is not available.
**Path to resolution:** Longitudinal outcome data collection; minimum 18-month follow-up for progression endpoint validation.

### 6.6 Academic Center Deployment Gap [SEVERITY: LOW-MODERATE — UNRESOLVED]

**Residual risk:** Both training datasets are academic medical center-derived. Community clinic, rural hospital, or federally qualified health center deployment contexts may introduce demographic distributions outside the training population.
**Path to resolution:** Real-world evidence study in community deployment settings; post-market registry.

---

## 7. Post-Market Monitoring Plan for Bias Detection

### 7.1 Monitoring Architecture

Post-market bias monitoring is a regulatory commitment under FDA AI/ML SaMD guidance and EU MDR Article 83 (Post-Market Surveillance). Monitoring is operationalized through three mechanisms:

1. **Prospective outcome registry** — linked clinical outcomes with device outputs
2. **Statistical process control on subgroup performance metrics** — automated drift detection
3. **Periodic mandatory review** — scheduled independent analysis

### 7.2 Key Performance Indicators (KPIs) for Bias Monitoring

| KPI | Metric | Alert Threshold | Action Threshold |
|---|---|---|---|
| APOE4 carrier AUC | Rolling 90-day AUC estimate (N≥50) | < 0.80 | < 0.75 |
| Race/ethnicity AUC gap | Max AUC gap across reported race/ethnicity groups | > 0.10 | > 0.15 |
| Sex AUC gap | AUC(male) − AUC(female) | > 0.10 | > 0.15 |
| Age subgroup AUC | Minimum AUC across age bands | < 0.82 | < 0.78 |
| Subgroup ECE | ECE per subgroup post-calibration | > 0.10 | > 0.15 |
| False negative rate by race | Subgroup-specific 1 − Sensitivity | > 0.30 | > 0.40 |
| Score distribution shift | KL divergence of output scores by demographic group vs. baseline | > 0.05 | > 0.10 |

**Alert threshold:** Triggers mandatory review within 30 days and disclosure to Roche Information Solutions deployment lead.
**Action threshold:** Triggers mandatory model quarantine review and FDA/notified body notification within 15 days.

### 7.3 Monitoring Schedule

| Activity | Frequency | Responsible Party | Output |
|---|---|---|---|
| Automated KPI dashboard update | Continuous (real-time) | Data Science Lead | Live dashboard |
| Subgroup performance statistical review | Quarterly | Clinical Documentation Specialist + Biostatistician | Quarterly Bias Report |
| Full fairness re-assessment (FAIR-001 update) | Annually | Full clinical team | FAIR-001 vX.Y |
| Independent external audit of fairness | Every 24 months | Third-party clinical AI auditor | External Audit Report |
| Regulatory submission update | Per PMA supplement / PCCP update cycle | Regulatory Affairs | FDA/MDR filing |

### 7.4 Minimum N Requirements for Triggered Analyses

Bias monitoring analyses are only reported when the following minimum N thresholds are met to prevent false alert signals from noise:

| Analysis Type | Minimum N | If Below Threshold |
|---|---|---|
| Subgroup AUC | 50 per subgroup | Flag as "insufficient data"; do not report point estimate |
| Race/ethnicity AUC | 30 per racial/ethnic group | Aggregate into "underrepresented communities" category |
| Calibration (ECE) per subgroup | 40 per subgroup | Pool with adjacent subgroup or report as underpowered |

### 7.5 Data Collection Requirements for Bias Monitoring

Post-market bias monitoring requires structured demographic data capture at point of care. The following fields are required in the Navify Algorithm Suite integration:

- Patient age (exact, not banded — banding applied analytically)
- Sex at birth (Male / Female / Not reported)
- Self-reported race and ethnicity (per US OMB categories or EU equivalent)
- APOE4 genotype status (Carrier / Non-carrier / Unknown)
- Site type (Academic medical center / Community clinic / Other)

**Data governance:** All demographic data collected for bias monitoring is de-identified prior to transmission to the monitoring registry. Collection is governed by a dedicated Data Use Agreement between the deploying institution and NeuroFusion-AD.

### 7.6 Feedback Loop and Model Update Triggers

If any Action Threshold (Section 7.2) is met, the following predetermined change control process (per FDA PCCP guidance) is initiated:

1. **Immediate:** Model output flagged in Navify interface with advisory to clinicians
2. **Within 15 days:** Root cause analysis submitted to FDA/notified body
3. **Within 90 days:** Remediated model version with updated validation package submitted for review
4. **Prospective commitment:** APOE4 carrier subgroup validation to N ≥ 100 prior to removal of IFU advisory (Section 6.1)

---

## 8. Summary and Regulatory Status

| Domain | Status | Primary Finding |
|---|---|---|
| Fairness framework | ✅ Defined | Thresholds pre-specified; metrics registered |
| ADNI diversity | ❌ Inadequate | Predominantly White, academic center, synthesized features |
| Bio-Hermes diversity | ⚠️ Partial | 24% underrepresented communities; cross-sectional only |
| APOE4 subgroup AUC | ❌ **FAILS threshold** | AUC=0.775 vs. threshold 0.80; primary unresolved bias risk |
| Overall AUC gap | ❌ **FAILS threshold** | Gap=0.225 (including small N) / 0.131 (N≥10); both exceed 0.10 |
| Age subgroup performance | ✅ Pass (N≥10 groups) | AUC 0.865–0.939 for age 65–75 and >75 |
| Sex subgroup performance | ✅ Pass | AUC 0.875–0.900; female CI wide |
| Race/ethnicity validation | ❌ Not achievable in current data | Insufficient diversity in ADNI; insufficient N per group in Bio-Hermes |
| Calibration (aggregate) | ✅ Pass | ECE=0.0831 after temperature scaling |
| Calibration (subgroup) | ⚠️ Not assessed | Required before Phase 3 claims |
| Post-market monitoring | ✅ Defined | Automated KPI + quarterly review + annual FAIR update |

**Overall Fairness Determination:** **CONDITIONAL APPROVAL FOR LIMITED DEPLOYMENT** — NeuroFusion-AD demonstrates adequate overall discrimination (AUC 0.890–0.907) but fails pre-specified fairness thresholds on APOE4 carrier subgroup performance and maximum AUC gap. Deployment is permissible under the following conditions:

1. IFU must include explicit advisory regarding APOE4 carrier performance limitations