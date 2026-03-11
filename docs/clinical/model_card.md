---
document_id: model-card
generated: 2026-03-11
batch_id: msgbatch_01HZUXhy6DzGoszEMVS44MBf
status: DRAFT — requires human review before submission
---

# NeuroFusion-AD Model Card

**Version:** 1.0.0
**Date:** 2025-07-14
**Format:** Mitchell et al. (2019), adapted for FDA AI/ML Action Plan compliance
**Regulatory Status:** FDA De Novo (pending) | EU MDR Class IIa | IEC 62304 Class B | ISO 14971
**Document ID:** NF-AD-MC-001

---

## Table of Contents

1. [Model Details](#1-model-details)
2. [Intended Use](#2-intended-use)
3. [Out-of-Scope Uses](#3-out-of-scope-uses)
4. [Factors](#4-factors)
5. [Metrics](#5-metrics)
6. [Evaluation Data](#6-evaluation-data)
7. [Training Data](#7-training-data)
8. [Ethical Considerations](#8-ethical-considerations)
9. [Caveats and Recommendations](#9-caveats-and-recommendations)

---

## 1. Model Details

### 1.1 Model Description

NeuroFusion-AD is a multimodal clinical decision support (CDS) software as a medical device (SaMD) designed to aid clinicians in assessing the risk of amyloid progression in patients diagnosed with mild cognitive impairment (MCI). The system fuses four input modalities — fluid biomarkers, acoustic speech features, fine motor measurements, and structured clinical variables — through an attention-based deep neural network. The model produces a continuous amyloid progression risk score and an estimated cognitive trajectory (MMSE), intended to supplement, not replace, clinician judgment.

### 1.2 Model Architecture and Training

| Component | Specification |
|-----------|--------------|
| Architecture | Multimodal attention fusion network with modality-specific encoders |
| Input modalities | Fluid biomarkers, acoustic features, motor features, clinical variables |
| Output | Amyloid progression risk score (0–1), MMSE trajectory estimate, survival risk index |
| Hyperparameter optimization | Optuna, 30 trials, ADNI validation split |
| Baseline training | 150 epochs, ADNI (N=345 train), W&B run `ybbh5fky` |
| Fine-tuning | Bio-Hermes-001 (N=756 train, frozen encoders, classification head only), lr=5e-5, W&B run `eicxum0n` |
| Hardware | Single NVIDIA RTX 3090, AMP mixed precision, gradient accumulation = 4 |
| Scheduler | OneCycleLR; early stopping patience = 25 epochs |

### 1.3 Deployment Target

Roche Information Solutions — Navify Algorithm Suite

### 1.4 Model Developers and Contact

**Organization:** NeuroFusion-AD Development Team
**Clinical Documentation Specialist:** [Name on file]
**Regulatory Contact:** [Contact on file]
**For questions regarding this model card:** [Contact information on file]

### 1.5 License and Access

Proprietary. Access restricted to authorized clinical deployment partners under executed data use and software license agreements. Not available for open redistribution.

---

## 2. Intended Use

### 2.1 Primary Intended Use

NeuroFusion-AD is intended to **aid** the clinical assessment of amyloid progression risk in patients aged **50–90 years** with an established diagnosis of **Mild Cognitive Impairment (MCI)**. The system is designed to be used as one input among multiple information sources reviewed by a qualified physician or licensed clinician during a patient evaluation. It is not intended to be the sole basis for any clinical decision.

### 2.2 Intended Users

- Neurologists and geriatricians with training in MCI and dementia assessment
- Memory clinic clinicians operating within a supervised diagnostic workflow
- Clinical teams using the Navify Algorithm Suite within a Roche Information Solutions deployment environment

### 2.3 Clinical Context

The model is intended for use within a structured clinical encounter where a physician has already established an MCI diagnosis and seeks additional decision support regarding the likelihood of amyloid pathology progression. The risk score is intended to inform — but not determine — decisions about further diagnostic workup (e.g., amyloid PET, lumbar puncture), monitoring frequency, or care planning discussions.

### 2.4 Regulatory Classification

| Jurisdiction | Classification |
|---|---|
| United States (FDA) | De Novo, SaMD Clinical Decision Support |
| European Union | EU MDR Class IIa |
| Software Safety Class | IEC 62304 Class B |
| Risk Management | ISO 14971 compliant |

---

## 3. Out-of-Scope Uses

The following uses are **explicitly outside the intended use** of NeuroFusion-AD and are **contraindicated**. Deployment in any of the following contexts is not supported and may introduce patient safety risks.

### 3.1 Prohibited Patient Populations

| # | Out-of-Scope Use | Rationale |
|---|---|---|
| 1 | **Patients under 50 years of age** | No training or validation data exists for this age group. Model behavior is undefined and potentially harmful in younger-onset presentations. |
| 2 | **Patients without an established MCI diagnosis** (e.g., cognitively normal individuals, subjective cognitive decline only, dementia already diagnosed) | The model was trained and validated exclusively in MCI cohorts. Application to pre-MCI or post-MCI populations is clinically inappropriate and statistically unvalidated. |
| 3 | **Patients with dementia (any stage) or other primary neurodegenerative diagnoses** (e.g., Parkinson's disease dementia, Lewy body disease, frontotemporal dementia) | These populations were excluded from all training and validation data. Risk scores in these populations have no validated clinical meaning. |

### 3.2 Prohibited Use Patterns

| # | Out-of-Scope Use | Rationale |
|---|---|---|
| 4 | **Standalone diagnostic use** — using the model output as a definitive diagnosis of amyloid pathology or Alzheimer's disease without confirmatory clinical or biomarker evaluation | NeuroFusion-AD is a risk stratification aid, not a diagnostic test. It does not replace amyloid PET, CSF analysis, or equivalent confirmatory workup. |
| 5 | **Use without physician review** — automated clinical action (e.g., treatment initiation, referral, care-level changes) triggered solely by model output without qualified clinician oversight | All outputs require interpretation by a licensed physician. No clinical pathway should be automated solely on the basis of model scores. |
| 6 | **Population-level screening** in unselected or cognitively normal community cohorts | The model is not validated for screening utility in populations without established MCI. Positive predictive value in low-prevalence settings is unknown and likely poor. |
| 7 | **Prognostic use in patients over 90 years of age** | The validation cohorts enrolled patients up to age 90. Extrapolation beyond this age range is not supported. |
| 8 | **Use as a sole determinant for clinical trial enrollment or exclusion** | Risk score performance is insufficient (see Section 5) to serve as a standalone enrollment criterion without additional validated biomarker confirmation. |
| 9 | **Use in paediatric patients** | No data, no validation, no regulatory authorization. |
| 10 | **Deployment outside the Navify Algorithm Suite** or in any infrastructure not validated by Roche Information Solutions and NeuroFusion-AD development team | Software performance is validated only within the specified deployment environment. |

---

## 4. Factors

This section documents the factors — demographic, clinical, and technical — anticipated to influence model behavior, as required under Mitchell et al. (2019) and aligned with FDA guidance on algorithmic bias evaluation.

### 4.1 Demographic Factors

| Factor | Notes |
|---|---|
| **Age** | Three subgroups evaluated: <65, 65–75, >75 (see Section 5.3). Performance differences observed across strata, though confidence intervals overlap substantially. |
| **Sex** | Male and female subgroups evaluated. Female subgroup shows marginally higher AUC (0.592 vs. 0.543), but confidence intervals overlap; difference is not statistically significant at N=44–56 per group. |
| **APOE4 carrier status** | Meaningful performance divergence observed. Non-carriers: AUC 0.690 (95% CI: 0.530–0.833); Carriers: AUC 0.501 (95% CI: 0.369–0.650). The model shows near-chance performance in APOE4 carriers on the ADNI test set. |
| **Race and ethnicity** | ADNI cohort is predominantly non-Hispanic White. Bio-Hermes-001 includes 24% participants from underrepresented communities. Subgroup-level performance by race/ethnicity is not currently reportable due to sample size constraints in the test set. |

### 4.2 Clinical Factors

| Factor | Notes |
|---|---|
| **MMSE baseline score** | Top-3 SHAP feature. Performance may vary across MCI severity spectrum (mild vs. borderline). |
| **Amyloid biomarker availability** | Only 63.8% of ADNI participants had valid CSF Aβ42 labels. Label missingness may introduce selection bias in the training labels. |
| **APOE4 genotype** | Top-5 SHAP feature; also a key stratification variable for subgroup performance differences. |
| **Comorbidities** | Patients with psychiatric comorbidities, cerebrovascular disease, or other confounding neurological conditions were not systematically excluded from ADNI but may affect generalizability. |

### 4.3 Technical and Instrumentation Factors

| Factor | Notes |
|---|---|
| **Biomarker assay platform** | ADNI uses CSF pTau181 as a proxy for plasma pTau217. Bio-Hermes-001 uses the Roche Elecsys plasma pTau217 assay — the target deployment assay. This assay mismatch in training data is a material limitation. |
| **Acoustic and motor feature synthesis** | ADNI acoustic and motor features were synthesized from clinical distributions (data record DRD-001) rather than collected from real patients. Real-world acoustic and motor inputs may differ systematically. |
| **Cross-site variability** | ADNI data is multi-site; Bio-Hermes-001 is a separate external cohort. Site-level harmonization may not fully eliminate acquisition differences. |
| **Modality completeness** | The model assumes all four modality inputs are available. Behavior under partial missingness in production has not been fully characterized. |

---

## 5. Metrics

Evaluation metrics follow recommendations from FDA's AI/ML-Based SaMD Action Plan and are reported separately for internal validation (ADNI) and external validation (Bio-Hermes-001) cohorts. All classificationAUC confidence intervals are bootstrapped (N=1000, BCa method) unless otherwise noted.

> ⚠️ **Critical Interpretive Note:** ADNI validation-set AUC during training reached 1.0, attributable to the strong influence of the `ABETA42_CSF` feature, which is the model's training label proxy. This inflated metric is **not** a valid performance estimate. All performance claims in this model card are drawn from the held-out test set (ADNI, N=100) and the external fine-tuning validation set (Bio-Hermes-001, N=189).

### 5.1 Classification Performance

| Metric | ADNI Test Set (Internal) | Bio-Hermes-001 Val Set (External) |
|---|---|---|
| **N evaluated** | 100 | 189 |
| **AUC (95% CI)** | **0.579 (0.452–0.671)** | **0.829 (0.780–0.870)** |
| Sensitivity | 0.750 | Not available |
| Specificity | 0.467 | Not available |
| PPV | Not calculable | Not available |
| NPV | Not calculable | Not available |
| F1 Score | Not calculable | Not available |
| AUPRC | 0.479 | Not available |
| Optimal threshold | Not established | Not established |

**Notes on ADNI classification metrics:**
PPV, NPV, F1, and optimal threshold are reported as not calculable for the ADNI test set due to insufficient labeled-positive cases at the evaluated threshold and the provisional nature of the amyloid label derived from CSF Aβ42 proxy. These should be established with the target assay on a prospectively collected cohort before clinical deployment.

**Notes on Bio-Hermes-001 metrics:**
AUC is taken from the best validation checkpoint at epoch 17 (early stopping). Sensitivity, specificity, PPV, and NPV are not reported because Bio-Hermes-001 is cross-sectional and does not include longitudinal outcome labels sufficient for threshold-dependent metric calculation in this fine-tuning run.

### 5.2 Regression and Survival Performance (ADNI Test Set)

| Metric | Value | Interpretation |
|---|---|---|
| MMSE RMSE | 2.23 points/year | Moderate absolute error in cognitive trajectory estimation |
| MMSE R² | **−0.842** | ⚠️ Negative R² indicates the MMSE regression component performs **worse than a mean predictor**. This output should not be used clinically until retrained and re-validated. |
| Survival C-index | 0.509 (95% CI: 0.444–0.591) | Equivalent to random chance (0.5). Survival risk index is **not clinically informative** in its current form. |

### 5.3 Calibration

| Stage | ECE | Notes |
|---|---|---|
| Pre-calibration | 0.200 | Substantially miscalibrated; risk scores systematically biased |
| Post-temperature scaling | **0.021** (T = 3.30) | Acceptable calibration after scaling. Temperature of 3.30 indicates the model was overconfident prior to calibration. |

Temperature scaling was applied as a post-hoc calibration step. Calibration should be re-evaluated if the model is retrained, fine-tuned, or deployed on a new patient population.

### 5.4 Subgroup Performance (ADNI Test Set)

| Subgroup | N | AUC | 95% CI |
|---|---|---|---|
| Age < 65 | 33 | 0.516 | 0.321–0.723 |
| Age 65–75 | 34 | 0.618 | 0.414–0.762 |
| Age > 75 | 33 | 0.573 | 0.370–0.783 |
| Sex: Male | 44 | 0.543 | 0.340–0.705 |
| Sex: Female | 56 | 0.592 | 0.414–0.731 |
| APOE4 Non-carrier | 42 | 0.690 | 0.530–0.833 |
| APOE4 Carrier | 58 | 0.501 | 0.369–0.650 |

**Fairness assessment:**
- Maximum AUC gap across subgroups: **0.189** (APOE4 non-carrier vs. carrier)
- Fairness threshold: **FAIL**
- The APOE4 carrier subgroup (N=58) shows near-chance discrimination. Given that APOE4 carriers represent a clinically high-risk population for amyloid pathology, this failure is of direct clinical relevance and must be addressed before deployment.

### 5.5 Modality Attention Weights (ADNI Test Set)

Mean attention weights across modalities are approximately uniform, suggesting no single modality dominates the fusion:

| Modality | Mean Attention Weight |
|---|---|
| Motor features | 0.261 |
| Acoustic features | 0.248 |
| Fluid biomarkers | 0.246 |
| Clinical variables | 0.245 |

> **Caution:** Despite the approximately uniform attention weights, SHAP feature importance identifies `abeta42_csf` and `ptau217` as the top predictive features. Attention weights reflect architectural routing, not causal feature importance. These two representations are not equivalent and should not be used interchangeably for interpretability.

### 5.6 Top SHAP Features

In ranked order of mean absolute SHAP value on the ADNI test set:

1. `abeta42_csf`
2. `ptau217`
3. `mmse_baseline`
4. `age`
5. `apoe4`

The dominance of `abeta42_csf` in SHAP rankings is consistent with this feature being the label proxy during ADNI training, and is a significant confound in interpreting ADNI test-set performance as true generalization.

---

## 6. Evaluation Data

### 6.1 ADNI — Internal Validation and Test Set

| Attribute | Value |
|---|---|
| Dataset | Alzheimer's Disease Neuroimaging Initiative (ADNI) |
| Total cohort used | 494 MCI patients |
| Split | Train: 345 / Val: 74 / Test: 75 (held-out test set expanded to N=100 for evaluation) |
| Age range | 50–90 years |
| Amyloid label coverage | 63.8% (315/494 with valid CSF Aβ42) |
| Biomarker assay | CSF pTau181 (proxy for plasma pTau217 — **assay mismatch with deployment target**) |
| Acoustic/motor features | **Synthesized** from clinical distributions (DRD-001) — not collected from real patients |
| Longitudinal data | Available |
| Cohort diversity | Predominantly non-Hispanic White; limited demographic diversity |
| W&B training runs | Baseline HPO: `jehkd9ud`; Best model: `ybbh5fky` |

**Known limitations of ADNI evaluation:**
- CSF pTau181 ≠ plasma pTau217. The assay used for training labels is not the assay used in the deployment target, introducing systematic measurement discordance.
- Synthesized acoustic and motor features may not reflect real-world signal distributions. Acoustic/motor model components are not validated against real patient data in this cohort.
- The 1.0 validation AUC observed during training (`ybbh5fky`) reflects `ABETA42_CSF` data leakage through the label proxy and is explicitly excluded from all performance claims.

### 6.2 Bio-Hermes-001 — External Validation Set

| Attribute | Value |
|---|---|
| Dataset | Bio-Hermes-001 |
| Total cohort size | 945 participants |
| Split used | Train: 756 / Val: 189 |
| Age range | Not specified (enrolled adults) |
| Underrepresented communities | 24% of cohort |
| Biomarker assay | Plasma pTau217 — Roche Elecsys (**matches deployment target assay**) |
| Longitudinal data | **Not available** — cross-sectional only |
| Fine-tuning W&B run | `eicxum0n` |
| Best checkpoint | Epoch 17 (early stopping) |

**Known limitations of Bio-Hermes-001 evaluation:**
- Cross-sectional design means no longitudinal outcome labels are available. The AUC of 0.829 reflects prediction of biomarker status at a single time point, not prediction of future amyloid progression. This limits direct clinical interpretation of the external AUC.
- The validation set (N=189) is the same split used for early stopping during fine-tuning. An independent held-out test set from Bio-Hermes-001 does not exist. The external AUC may therefore be optimistic.
- **Bio-Hermes-002 does not exist.** Any reference to a second Bio-Hermes dataset in related documents should be treated as erroneous.

---

## 7. Training Data

### 7.1 Training Summary

| Phase | Dataset | N (train) | Key Configuration |
|---|---|---|---|
| Phase 1: Baseline | ADNI | 345 | Optuna HPO (30 trials) → retrain best config, 150 epochs |
| Phase 2: Fine-tuning | Bio-Hermes-001 | 756 | Frozen encoders, classification head only, lr=5e-5, early stopping patience=25 |

### 7.2 Label Construction

Amyloid progression labels in ADNI were derived from **CSF Aβ42 values** (threshold-based binary classification). This is a proxy label, not a direct measure of amyloid progression, and introduces the following consequences:

- The model may have learned to detect CSF Aβ42 status rather than clinical amyloid progression trajectory
- Feature `abeta42_csf` is the dominant SHAP predictor, consistent with partial label leakage
- The ADNI validation AUC of 1.0 during training is entirely explained by this phenomenon and must not be cited as model performance

Bio-Hermes-001 fine-tuning used plasma pTau217 (Roche Elecsys) labels. This represents the operationally correct target assay for deployment and provides a more clinically relevant training signal for the classification head.

### 7.3 Data Limitations Affecting Training

1. **Synthesized modalities:** ADNI acoustic and motor features are computationally generated, not recorded from patients. The encoders for these modalities are trained on synthetic distributions and their real-world transfer performance is uncharacterized.
2. **Label missingness:** 36.2% of ADNI participants lacked valid amyloid labels and were excluded from supervised training, introducing potential selection bias.
3. **Assay heterogeneity across training phases:** Phase 1 (ADNI) used CSF pTau181; Phase 2 (Bio-Hermes) used plasma pTau217. The model was exposed to discordant biomarker representations of the same biological signal during different training phases.
4. **Single fine-tuning site/cohort:** Generalizability of the fine-tuned classification head beyond Bio-Hermes-001 recruitment sites has not been established.

---

## 8. Ethical Considerations

### 8.1 Intended Benefit and Harm Potential

NeuroFusion-AD is intended to support earlier and more accurate risk stratification of amyloid pathology in MCI patients, potentially enabling earlier intervention and care planning. However, the following harms are possible if the system is misused or if its limitations are not communicated and respected:

- **False reassurance:** A low risk score in a patient with true amyloid progression may delay workup, treatment, or supportive care.
- **False alarm and unnecessary intervention:** A high risk score in an amyloid-negative patient may lead to unnecessary invasive procedures (e.g., lumbar puncture), anxiety, or inappropriate treatment.
- **Disparate impact:** The model shows near-chance performance in APOE4 carriers (AUC 0.501), a clinically important high-risk group. Deployment without disclosure of this limitation could result in systematically worse care guidance for this population.
- **Automation bias:** Clinicians may over-rely on model output, particularly if the risk score is displayed prominently in the clinical interface without adequate uncertainty communication.

### 8.2 Equity and Fairness

- The ADNI training cohort is predominantly non-Hispanic White and does not adequately represent the demographic diversity of the MCI population. This limits confidence in equitable performance across racial and ethnic groups.
- Bio-Hermes-001 includes 24% participants from underrepresented communities, improving but not resolving the diversity gap. Subgroup performance by race and ethnicity cannot currently be reported with statistical reliability due to test-set sample size constraints.
- The fairness evaluation **fails** the pre-specified AUC gap threshold (observed gap = 0.189), primarily driven by APOE4 stratification. This must be disclosed to clinical deployers and end users.
- A prospective equity audit with adequate sample sizes across demographic subgroups is required before broad deployment.

### 8.3 Privacy and Data Governance

- ADNI data was accessed under an approved data use agreement with the ADNI Data Sharing and Publications Committee.
- Bio-Hermes-001 data was accessed under applicable IRB approval and data use agreement.
- No patient-identifiable data is included in this model card or in model weights as distributed.
- All training data handling complied with applicable HIPAA, GDPR, and institutional data governance requirements.

### 8.4 Informed Consent and Transparency

- Patients whose data were used in training and validation provided informed consent for research use under the respective study protocols.
- It is recommended that healthcare institutions deploying NeuroFusion-AD disclose to patients that AI-assisted clinical decision support tools are used in their care pathway, consistent with emerging best practices and applicable local regulations.

### 8.5 Human Oversight Requirement

The model is explicitly designed for **human-in-the-loop** use. Any deployment configuration that permits automated clinical action without qualified physician review is a misuse of the system (see Section 3, Out-of-Scope Uses). Integration partners are contractually required to implement a physician review step before any clinical output is acted upon.

---

## 9. Caveats and Recommendations

### 9.1 Mandatory Limitations

The following five limitations are **mandatory disclosures** that must accompany any clinical deployment, integration documentation, or publication of NeuroFusion-AD performance results.

---

**MANDATORY LIMITATION 1: Assay Mismatch Between Training and Target Deployment**

The ADNI training data uses **CSF pTau181** as a proxy for plasma pTau217. The intended deployment assay is **plasma pTau217 (Roche Elecsys)**. These are different analytes measured by different platforms with different reference ranges and clinical correlates. The model's fluid biomarker encoder was trained on a non-target assay. The Bio-Hermes-001 fine-tuning partially addresses this through classification head adaptation, but the underlying encoder representations were not retrained on the target assay. Performance claims derived from ADNI data must not be interpreted as reflective of performance on plasma pTau217 inputs. Prospective validation using exclusively plasma pTau217 inputs is required.

---

**MANDATORY LIMITATION 2: Synthesized Acoustic and Motor Features in Training Data**

ADNI acoustic and motor input features were **computationally synthesized** from clinical distributions (DRD-001) and do not represent real patient recordings or measurements. The modality encoders for acoustic and motor inputs were trained on synthetic data. Despite attention weights suggesting these modalities contribute approximately equally to model decisions (~25% each), their contribution in real-world deployment — where acoustic and motor inputs are real — is **unvalidated**. Real-world acoustic and motor data may exhibit distributional properties not present in the synthetic training data, potentially degrading or unpredictably altering model behavior. Independent validation on real acoustic and motor data is required before clinical deployment.

---

**MANDATORY LIMITATION 3: ADNI Test-Set AUC Is Near Chance; MMSE Regression and Survival Index Are Not Clinically Informative**

On the ADNI held-out test set (N=100), the classification AUC is **0.579 (95% CI: 0.452–0.671)**. The lower bound of the confidence interval includes values below 0.5, indicating that discriminative performance in this cohort cannot be statistically distinguished from chance at conventional significance levels. Furthermore:
- The MMSE trajectory regression yields **R² = −0.842**, meaning it performs worse than predicting the mean MMSE for all patients. This output must not be used clinically.
- The survival C-index is **0.509 (95% CI: 0.444–0.591)**, equivalent to random guessing. The survival risk index must not be used clinically.

The Bio-Hermes-001 external AUC of 0.829 is more encouraging, but it is derived from a cross-sectional cohort using a validation split that was also used for early stopping, and thus may be optimistic. Neither result is sufficient to support standalone diagnostic use.

---

**MANDATORY LIMITATION 4: Fairness Evaluation Fails; Near-Chance Performance in APOE4 Carriers**

The pre-specified fairness evaluation **fails**. The AUC gap between APOE4 non-carriers (AUC = 0.690) and APOE4 carriers (AUC = 0.501) is **0.189**, exceeding the acceptable threshold. APOE4 carrier status is the strongest known genetic risk factor for late-onset Alzheimer's disease. The model's inability to discriminate amyloid progression risk in this clinically critical subgroup represents a material patient safety concern. Deploying NeuroFusion-AD without communicating this limitation to clinicians risks systematic underestimation of risk in a high-risk population. Mitigation — including targeted data collection, APOE4-stratified retraining, or explicit suppression of the risk score output for APOE4 carriers pending revalidation — must be evaluated before deployment.

---

**MANDATORY LIMITATION 5: Absence of Longitudinal Outcome Validation**

Bio-Hermes-001, the external validation cohort providing the highest AUC estimate, is **cross-sectional**. It contains no longitudinal follow-up data linking baseline risk scores to future amyloid progression events. The AUC of 0.829 reflects the model's ability to predict biomarker status at a single time point, not its ability to predict who will progress to Alzheimer's dementia over time — which is the clinically meaningful use case. The ADNI longitudinal data provides partial evidence for temporal prediction, but performance in that cohort is poor (C-index ≈ 0.5). No prospective validation study with longitudinal outcome follow-up has been completed. Such a study is required to substantiate any claim that NeuroFusion-AD predicts amyloid progression in a clinically meaningful sense.

---

### 9.2 Additional Recommendations

| # | Recommendation | Priority |
|---|---|---|
| R1 | Commission a prospective validation study using plasma pTau217 (Roche Elecsys) inputs with real acoustic and motor measurements and longitudinal outcomes (minimum 12-month follow-up). | **Critical** |
| R2 | Address APOE4 carrier fairness failure before broad deployment. Consider APOE4-stratified model variants, additional training data for carriers, or output suppression with clinical advisory for this subgroup. | **Critical** |
| R3 | Retire or clearly gate the MMSE trajectory regression and survival risk index outputs until these components can be retrained and validated. R² < 0 and C-index ≈ 0.5 are unacceptable for clinical use. | **Critical** |
| R4 | Collect and validate real acoustic and motor feature data to replace synthesized ADNI inputs in retraining. The current acoustic/motor encoders cannot be assumed to generalize to real data. | **High** |
| R5 | Obtain an independent held-out test set from Bio-Hermes-001 (or an equivalent external cohort) that was not used at any point in training or early stopping decisions, to provide an unbiased external performance estimate. | **High** |
| R6 | Expand demographic diversity in future training and validation data, particularly across racial and ethnic groups, to enable statistically reliable subgroup performance reporting. | **High** |
| R7 | Re-evaluate model calibration on the deployment target population (plasma pTau217 inputs). Temperature scaling parameters (T=3.30) derived from ADNI may not transfer directly to Bio-Hermes or real-world distributions. | **Medium** |
| R8 | Implement a post-market surveillance plan capturing real-world model performance, adverse events, and user feedback, consistent with FDA Total Product Lifecycle (TPLC) guidance for AI/ML SaMD. | **Medium** |
| R9 | Ensure all clinical interface implementations display confidence intervals alongside point estimates and include explicit warnings for subgroups where model performance is degraded (particularly APOE4 carriers). | **Medium** |
| R10 | Document and disclose to deployers that Bio-Hermes-002 does not exist. Any reference to Bio-Hermes-002 in associated materials is erroneous and must be corrected. | **Low** |

---

## Document Control

| Field | Value |
|---|---|
| Document ID | NF-AD-MC-001 |
| Version | 1.0.0 |
| Status | Draft — Pending Regulatory Review |
| Prepared by | Clinical Documentation Specialist, NeuroFusion-AD |
| Review required | Regulatory Affairs, Clinical Affairs, Biostatistics, Ethics Review Board |
| Next review date | [Per post-market surveillance schedule] |
| Format reference | Mitchell et al. (2019). "Model Cards for Model Reporting." *FAccT 2019*. |
| Regulatory alignment | FDA AI/ML-Based SaMD Action