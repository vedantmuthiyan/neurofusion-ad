---
document_id: model-card
generated: 2026-03-15
batch_id: msgbatch_01G4xrs23ARV9Qg7oCHV4nen
status: DRAFT — requires human review before submission
---

# NeuroFusion-AD Model Card

**Version:** 2.0 (Phase 2B Remediated)
**Date:** 2025
**Format:** Mitchell et al. (2019), adapted for FDA AI/ML SaMD guidance
**Document Control:** CDS-MC-001-v2.0
**Regulatory Classification:** FDA De Novo + EU MDR Class IIa | IEC 62304 Class B | ISO 14971

---

## 1. Model Details

| Field | Value |
|---|---|
| **Full Name** | NeuroFusion-AD Clinical Decision Support System |
| **Version** | 2.0 — Phase 2B Remediated |
| **Model Type** | Multimodal transformer-based fusion; SaMD — Clinical Decision Support |
| **Architecture** | Embed dim = 256; dropout = 0.4; ~2.24M parameters |
| **Input Modalities** | Fluid biomarkers, acoustic features, motor features, clinical/cognitive measures |
| **Output** | Amyloid progression risk score (0–1) + MMSE trajectory estimate |
| **Intended Platform** | Roche Information Solutions — Navify Algorithm Suite |
| **Target Population** | Adults aged 50–90 with diagnosed Mild Cognitive Impairment (MCI) |
| **Training Framework** | PyTorch (AMP); single NVIDIA RTX 3090 |
| **HPO** | Optuna; 15 trials (Phase 2B budget constraint); best trial val AUC = 0.9081 |
| **W&B Run IDs** | Baseline (ADNI): `k58caevv` · Best model: `t9s3ngbx` · Bio-Hermes fine-tune: `o4pcjy3r` |
| **Key Remediation (Phase 2B)** | ABETA42_CSF removed (Pearson r = −0.864 with label; critical data leakage); fluid features reduced from 6 → 2: [PTAU217, NFL_PLASMA] |
| **Developed By** | NeuroFusion-AD Clinical Documentation Team |
| **License / Distribution** | Proprietary — Roche Navify deployment only |
| **Contact** | clinical-documentation@neurofusion-ad.example.com |

### 1.1 Phase 2B Remediation Summary

The transition from Phase 2A to Phase 2B addressed a **critical data leakage event**. CSF Aβ42 (ABETA42_CSF) exhibited a Pearson correlation of r = −0.864 with the amyloid progression label, indicating it was effectively a direct proxy for the prediction target rather than an independent predictor. Its inclusion in Phase 2A inflated performance metrics and rendered those results scientifically invalid for regulatory submission.

**Changes enacted in Phase 2B:**
- ABETA42_CSF removed from all fluid feature inputs
- Fluid feature vector reduced from 6 features → 2 features: [PTAU217, NFL_PLASMA]
- Model capacity reduced accordingly: embed_dim 768 → 256; parameter count ~60M → ~2.24M
- All training, validation, and evaluation runs restarted from clean data pipeline
- HPO re-executed (15 trials under budget constraint) before final retraining

---

## 2. Intended Use

### 2.1 Primary Intended Use

NeuroFusion-AD is intended to **aid** the clinical assessment of amyloid progression risk in adults aged 50–90 who have received a clinical diagnosis of Mild Cognitive Impairment (MCI). The system generates a continuous risk score (0–1) and an estimated MMSE trajectory to support — not replace — clinician judgment in deciding whether further diagnostic workup (e.g., PET imaging, CSF analysis) may be warranted.

### 2.2 Clinical Context

The device is designed for use within the Roche Navify Algorithm Suite by qualified healthcare professionals with appropriate training in neurodegenerative disease assessment. All outputs must be reviewed by a licensed physician before any clinical action is taken. The tool is positioned as one input among multiple sources of clinical evidence and should be interpreted within the full clinical picture of each patient.

### 2.3 Regulatory Designation

- **US:** FDA De Novo authorization (SaMD — Clinical Decision Support)
- **EU:** MDR Class IIa
- **Software safety class:** IEC 62304 Class B
- **Risk management:** ISO 14971

---

## 3. Out-of-Scope Uses

The following uses are **explicitly prohibited** and fall outside the validated intended use of NeuroFusion-AD. Deployment in any of these contexts has not been validated and may cause patient harm.

| # | Prohibited Use | Rationale |
|---|---|---|
| **3.1** | **Patients under age 50** | Training and validation cohorts were restricted to ages 50–90. Performance in younger adults is entirely unvalidated; amyloid pathophysiology and biomarker distributions differ substantially below age 50. |
| **3.2** | **Non-MCI patient populations** | The model was trained exclusively on MCI-diagnosed patients. Application to cognitively normal individuals, patients with Alzheimer's disease dementia, other dementia subtypes (e.g., LBD, FTD, VaD), or other neurological conditions is not validated and is prohibited. |
| **3.3** | **Standalone diagnostic use** | NeuroFusion-AD does not provide a diagnosis of Alzheimer's disease, amyloid positivity, or any other clinical condition. The risk score is a decision support output and must not be used as the sole basis for diagnosis, treatment initiation, or exclusion from treatment. |
| **3.4** | **Use without physician review** | No clinical action may be taken on the basis of a NeuroFusion-AD output without review and clinical judgment from a licensed, qualified physician. Automated downstream clinical decisions without human review are prohibited. |
| **3.5** | **Screening of asymptomatic individuals** | The model is validated only for patients with an existing MCI diagnosis, not for population-level screening or risk stratification in asymptomatic adults. |
| **3.6** | **Prognosis beyond model output scope** | NeuroFusion-AD does not predict conversion to dementia with validated accuracy; survival C-index is exploratory only (see Section 5). |
| **3.7** | **Replacement of PET or CSF confirmatory testing** | The model output does not replace amyloid PET or lumbar puncture/CSF assay results for confirmatory diagnosis. |
| **3.8** | **Deployment outside Navify Algorithm Suite** | The model has been validated for the Roche Navify integration context. Performance in other electronic health record or PACS environments is unvalidated. |

---

## 4. Factors

### 4.1 Relevant Factors Evaluated

The following factors were evaluated in subgroup analyses to assess performance variation. Factor selection was guided by clinical relevance, known sources of bias in Alzheimer's research, and FDA AI/ML guidance on equity.

| Factor | Categories Evaluated | Source Dataset |
|---|---|---|
| Age | < 65 / 65–75 / > 75 | ADNI test set |
| Sex | Male / Female | ADNI test set |
| APOE4 status | Carrier / Non-carrier | ADNI test set |
| Community representation | Underrepresented communities (24%) | Bio-Hermes-001 |

### 4.2 Factors Not Yet Evaluated

The following clinically relevant factors could not be fully evaluated due to dataset limitations and represent gaps requiring post-market surveillance:

- **Race/ethnicity subgroup performance** — ADNI lacks sufficient granularity; Bio-Hermes-001 reports 24% underrepresented communities as a single category without further stratification available in current analyses
- **Education level** — Known confounder for MMSE scores; not systematically captured across both datasets
- **Language background** — Acoustic features may be influenced by native language; unvalidated in non-English speakers
- **Comorbidity burden** — Interaction with cardiovascular disease, diabetes, and other conditions known to affect biomarkers has not been systematically analyzed
- **Assay platform variability** — pTau217 assay differences across clinical laboratories beyond the Roche Elecsys platform

---

## 5. Metrics

### 5.1 Metric Definitions and Clinical Rationale

| Metric | Definition | Clinical Relevance |
|---|---|---|
| AUC (AUROC) | Area under ROC curve | Primary performance metric; threshold-independent discrimination |
| Sensitivity | True positive rate at optimal threshold | Ability to identify true amyloid-risk cases; high sensitivity reduces missed cases |
| Specificity | True negative rate at optimal threshold | Ability to correctly exclude low-risk patients; reduces unnecessary workup |
| PPV | Positive predictive value | Probability that a positive output reflects true amyloid risk |
| NPV | Negative predictive value | Probability that a negative output correctly excludes amyloid risk |
| F1 Score | Harmonic mean of precision and recall | Balance of precision and sensitivity |
| MMSE RMSE | Root mean squared error of MMSE trajectory (pts/year) | Accuracy of cognitive decline trajectory estimation |
| MMSE R² | Variance explained in MMSE trajectory | Predictive utility of trajectory estimation |
| Survival C-index | Concordance index for time-to-event | Exploratory; ordering of event timing predictions |
| ECE | Expected Calibration Error | Calibration quality; confidence score reliability for clinical use |
| Optimal Threshold | Youden-optimal classification threshold | Operating point for sensitivity/specificity reporting |

### 5.2 Internal Validation — ADNI Test Set

> **N = 75 total; N_labeled = 44** (amyloid labels available for 44/75; see Section 7 for limitation)
> Classification metrics computed on labeled subset (N = 44).

| Metric | Value | 95% CI |
|---|---|---|
| **AUC (AUROC)** | **0.890** | 0.782 – 0.964 |
| Sensitivity | 0.793 | — |
| Specificity | 0.933 | — |
| PPV | 0.958 | — |
| NPV | 0.700 | — |
| F1 Score | 0.868 | — |
| Optimal Threshold | 0.6443 | — |
| MMSE RMSE | 1.80 pts/year | — |
| MMSE R² | 0.047 | — |
| Survival C-index | 0.651 | 0.525 – 0.788 |
| ECE (pre-calibration) | 0.1120 | — |
| **ECE (post-calibration)** | **0.0831** | — (T = 0.76) |

> ⚠️ **ADNI-specific caveats:** (1) Amyloid label coverage is 63.8% overall (315/494); test set coverage is 58.7% (44/75). Classification metrics reflect only labeled cases and may not generalize to the full population. (2) ADNI does not contain plasma pTau217 (Roche Elecsys); CSF pTau181 was used as an assay proxy. This introduces an unknown systematic bias. (3) Acoustic and motor features in ADNI are **synthesized** from clinical distributions (DRD-001) and do not reflect real recorded data.

### 5.3 External Validation — Bio-Hermes-001 Test Set

> **N = 142** | Stratified 70/15/15 split; test set held out through all training and HPO
> Uses plasma pTau217 (Roche Elecsys) — the target clinical assay
> 24% underrepresented communities included

| Metric | Value | 95% CI |
|---|---|---|
| **AUC (AUROC)** | **0.907** | 0.855 – 0.959 |
| Sensitivity | 0.902 | — |
| Specificity | 0.879 | — |
| PPV | 0.807 | — |
| NPV | 0.941 | — |
| F1 Score | 0.852 | — |

> ✅ Bio-Hermes-001 is considered the stronger external validation dataset for the target deployment context due to its use of the Roche Elecsys pTau217 assay and diverse participant inclusion. However, it is **cross-sectional only** — no longitudinal outcome data are available (see Section 9).

### 5.4 Subgroup Performance — ADNI Test Set

> ⚠️ Small subgroup sizes limit interpretability. Wide confidence intervals should be noted.

| Subgroup | N | AUC | 95% CI | Clinical Note |
|---|---|---|---|---|
| Age < 65 | 11 | **1.000** | 1.000 – 1.000 | n = 11; result unreliable due to small N |
| Age 65–75 | 40 | 0.865 | 0.699 – 1.000 | Primary age stratum |
| Age > 75 | 24 | 0.939 | 0.814 – 1.000 | Strong performance in older subgroup |
| Sex: Male | 49 | 0.900 | 0.762 – 0.986 | |
| Sex: Female | 26 | 0.875 | 0.593 – 1.000 | Wider CI; fewer labeled females |
| APOE4 Non-carrier | 39 | 0.906 | 0.726 – 1.000 | |
| APOE4 Carrier | 36 | **0.775** | 0.416 – 1.000 | ⚠️ Lower performance; very wide CI |
| **Max AUC gap** | — | **0.225** | — | ⚠️ Fairness threshold not met |

> 🚨 **Fairness flag: FAIL.** The maximum AUC gap across subgroups is 0.225 (APOE4 carrier vs. non-carrier, and age < 65 vs. age 65–75). The pre-specified fairness threshold was not met. The APOE4 carrier subgroup (AUC = 0.775, CI: 0.416–1.000) warrants particular attention, as APOE4 carriers are a high-clinical-priority population. This finding must be disclosed to end users and addressed in post-market surveillance.

### 5.5 Modality Importance (ADNI Test Set — Mean Attention Weights)

| Modality | Mean Attention Weight | Notes |
|---|---|---|
| Clinical/Cognitive | 0.2855 | Highest weight; MMSE, age, APOE4 dominant |
| Acoustic | 0.2618 | ⚠️ Synthesized features in ADNI (DRD-001) |
| Motor | 0.2396 | ⚠️ Synthesized features in ADNI (DRD-001) |
| Fluid Biomarkers | 0.2130 | pTau217, NFL_PLASMA only (post-remediation) |

### 5.6 Top SHAP Features (Global Importance)

Ranked by mean absolute SHAP value on ADNI test set:

1. `ptau217` — Plasma phosphorylated tau 217
2. `nfl_plasma` — Plasma neurofilament light chain
3. `mmse_baseline` — Baseline MMSE score
4. `age` — Patient age
5. `apoe4` — APOE4 carrier status

---

## 6. Evaluation Data

### 6.1 ADNI (Internal Validation)

| Attribute | Detail |
|---|---|
| **Full name** | Alzheimer's Disease Neuroimaging Initiative |
| **Role** | Internal validation and baseline training |
| **N total** | 494 MCI patients |
| **Split** | Train = 345 / Val = 74 / Test = 75 |
| **Amyloid label coverage** | 63.8% overall (315/494); 58.7% test set (44/75) |
| **pTau assay** | CSF pTau181 (proxy for plasma pTau217 — different assay) |
| **Acoustic/Motor features** | Synthesized from clinical distributions (deviation report DRD-001) |
| **Age range** | 50–90 (MCI-diagnosed) |
| **Known limitations** | Predominantly non-Hispanic White; academic medical center recruitment; proxy assay for fluid biomarker; synthesized non-fluid modalities; incomplete amyloid labeling |

### 6.2 Bio-Hermes-001 (External Validation)

| Attribute | Detail |
|---|---|
| **Full name** | Bio-Hermes-001 |
| **Role** | External validation and fine-tuning (frozen encoders, classification head only) |
| **N total** | 945 participants |
| **Split** | Train = 661 / Val = 142 / Test = 142 |
| **pTau assay** | Plasma pTau217 — Roche Elecsys (target clinical assay) |
| **Diverse representation** | 24% underrepresented communities |
| **Study design** | Cross-sectional — no longitudinal follow-up data available |
| **Key strength** | Real plasma pTau217 data; community diversity; held-out test integrity confirmed |
| **Key limitation** | No longitudinal outcome data; single time-point only |

> **Note:** Bio-Hermes-002 does not exist. All references to an external replication dataset refer exclusively to Bio-Hermes-001.

---

## 7. Training Data

### 7.1 Training Pipeline Summary

```
ADNI (N=345 train) 
    → Baseline training (W&B: k58caevv)
    → Optuna HPO: 15 trials (budget constraint)
        Best trial val AUC = 0.9081
    → Retrain best config → Remediated model (W&B: t9s3ngbx)

Bio-Hermes-001 (N=661 train)
    → Fine-tuning: frozen encoders
    → Classification head only (W&B: o4pcjy3r)
```

### 7.2 Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Gradient accumulation | 4 steps |
| Precision | AMP (automatic mixed precision) |
| Early stopping patience | 25 epochs |
| Hardware | Single NVIDIA RTX 3090 |
| HPO framework | Optuna (15 trials) |

### 7.3 Critical Training Data Limitations

1. **ABETA42_CSF removed (Phase 2B remediation):** This feature had Pearson r = −0.864 with the amyloid label and constituted data leakage. All Phase 2A results are invalid and must not be cited.
2. **Synthesized acoustic and motor features (ADNI):** These modalities were not recorded in ADNI participants; values were synthesized from published clinical distributions (DRD-001). The model's acoustic and motor encoders were trained on non-real data in the ADNI phase.
3. **pTau assay mismatch (ADNI):** Training used CSF pTau181 as a proxy for plasma pTau217. Systematic differences between these assays introduce unknown bias into the fluid encoder weights from the ADNI training phase.
4. **HPO budget constraint:** Only 15 Optuna trials were conducted due to compute constraints. The hyperparameter search space is unlikely to have been exhaustively explored; the selected configuration may not represent a global optimum.
5. **Bio-Hermes fine-tuning scope:** Encoder weights were frozen during Bio-Hermes fine-tuning; only the classification head was updated. This limits the degree to which Bio-Hermes-001 data corrects assay-mismatch biases from ADNI training.

---

## 8. Ethical Considerations

### 8.1 Patient Safety

NeuroFusion-AD is a Clinical Decision Support tool. A risk score generated by this system must never be used as a sole or primary basis for clinical action. Misuse — including use without physician review, use in prohibited populations, or use as a standalone diagnostic — poses direct risk of patient harm through missed diagnosis, unnecessary invasive testing, inappropriate treatment initiation, or inappropriate withholding of further evaluation.

### 8.2 Equity and Fairness

- The **ADNI dataset is predominantly non-Hispanic White** with academic medical center recruitment bias. This is a well-documented limitation of ADNI and introduces risk of underperformance in underrepresented racial and ethnic groups.
- Bio-Hermes-001 includes 24% participants from underrepresented communities, but granular subgroup performance by race or ethnicity was not available for the current evaluation.
- The **APOE4 carrier subgroup shows lower AUC (0.775)** compared to non-carriers (0.906). Given that APOE4 carriers represent a clinically high-priority population for amyloid risk assessment, this performance gap raises equity concerns and must be disclosed to deploying clinicians.
- Fairness evaluation has formally **failed the pre-specified threshold** (max AUC gap = 0.225). Deployment must be accompanied by clinician awareness of differential performance.

### 8.3 Data Privacy and Consent

- ADNI data were used under the ADNI Data Use Agreement. All ADNI participants provided informed consent under IRB-approved protocols.
- Bio-Hermes-001 data were used under applicable data use agreements. All participants provided informed consent.
- No patient-identifiable information is retained in model weights or outputs.

### 8.4 Transparency

- The removal of ABETA42_CSF and the circumstances of the Phase 2A data leakage are disclosed fully in this document and in regulatory submissions. Phase 2A performance results must not be cited in any external communication.
- Synthesized feature usage in ADNI training is disclosed and documented (DRD-001).
- This Model Card is intended for regulatory reviewers, integration partners (Roche), and qualified clinical users.

### 8.5 Misuse Risk

The continuous nature of the risk score (0–1) may invite threshold-based decision-making by non-specialist users without appropriate clinical context. Deployment workflows within Navify must include interface-level safeguards communicating that (a) the score is a support tool only, (b) physician review is mandatory, and (c) the score does not constitute a diagnosis.

---

## 9. Caveats and Recommendations

### ⚠️ MANDATORY LIMITATIONS (5 Required)

The following five limitations are mandatory disclosures for regulatory submission, clinician-facing documentation, and deployment partner communications.

---

**LIMITATION 1 — Synthesized Acoustic and Motor Features (ADNI Dataset)**

Acoustic and motor features in the ADNI training and validation cohort are **computationally synthesized** from published clinical distributions rather than derived from real patient recordings (deviation report: DRD-001). Consequently, the acoustic and motor encoders were not trained on authentic signal data in the ADNI phase. Although Bio-Hermes-001 fine-tuning used real data for the classification head, encoder weights for these modalities retain ADNI-trained parameters. The model's modality attention weights assign 26.2% (acoustic) and 24.0% (motor) weight to these modalities, representing a substantial fraction of model decisions based on encoders trained on synthetic data.

*Recommendation:* Treat acoustic and motor feature contributions with caution in clinical interpretation. Future work must include a validation study using real acoustic and motor recordings across the intended population before these modalities are treated as fully validated inputs.

---

**LIMITATION 2 — pTau Assay Mismatch: CSF pTau181 ≠ Plasma pTau217**

ADNI does not contain plasma pTau217 data as measured by the Roche Elecsys assay. CSF pTau181 was used as a proxy during training. These are **different analytes measured in different biological compartments using different assay platforms**. The correlation between CSF pTau181 and plasma pTau217 is imperfect; systematic differences in dynamic range, reference intervals, and clinical sensitivity exist. This mismatch introduces unknown directional bias into the ADNI-phase fluid encoder. The impact is partially mitigated by Bio-Hermes-001 fine-tuning (which used Roche Elecsys plasma pTau217), but encoder weights were frozen during this phase and the underlying ADNI-derived representations persist.

*Recommendation:* The ADNI internal validation results (AUC = 0.890) should be interpreted as a lower-confidence internal benchmark. The Bio-Hermes-001 external results (AUC = 0.907) using the target assay are the more reliable performance estimate for the intended deployment context. Future training cycles should seek ADNI-equivalent datasets with native plasma pTau217 measurements.

---

**LIMITATION 3 — Incomplete Amyloid Label Coverage in ADNI**

Amyloid progression labels are available for only **63.8% of the full ADNI cohort** (315/494 patients) and **58.7% of the ADNI test set** (44/75 patients). Classification metrics for internal validation are computed exclusively on the labeled subset (N = 44). The 31 unlabeled test patients were excluded from classification evaluation. If labeling is systematically non-random — for example, if amyloid testing was more frequently performed in patients with more advanced or atypical presentations — the labeled subset may not be representative of the full MCI population, and ADNI classification metrics may be biased in an unknown direction.

*Recommendation:* Reported ADNI classification metrics (AUC = 0.890, sensitivity = 0.793, specificity = 0.933) should be interpreted in the context of this labeling gap. The Bio-Hermes-001 test set (N = 142; full label coverage) provides a more complete performance estimate. Post-market surveillance should track model performance on patients with complete amyloid confirmation.

---

**LIMITATION 4 — Subgroup Fairness Failure: APOE4 Carriers and Small Subgroup Reliability**

The pre-specified fairness evaluation has **formally failed** (max AUC gap = 0.225). Notable subgroup findings:
- APOE4 **carriers** show AUC = 0.775 (95% CI: 0.416–1.000) vs. non-carriers AUC = 0.906 (95% CI: 0.726–1.000)
- The **age < 65 subgroup** shows AUC = 1.000 based on N = 11 only — this result is not statistically meaningful and must not be cited as evidence of strong performance
- Female subgroup CI (0.593–1.000) is very wide, reflecting limited labeled female representation

APOE4 carriers represent a clinically critical population for amyloid risk assessment; lower model performance in this group could lead to disproportionate rates of missed risk identification precisely among high-risk patients.

*Recommendation:* Deploying clinicians must be informed of differential performance across APOE4 status. Clinical review should apply heightened scrutiny to model outputs for known APOE4 carriers. A dedicated fairness remediation study with larger, balanced subgroup representation is required before claims of equitable performance can be made.

---

**LIMITATION 5 — No Longitudinal Outcome Validation; Survival C-index is Exploratory**

Bio-Hermes-001, the primary external validation dataset, is **cross-sectional** — it contains no longitudinal follow-up data. NeuroFusion-AD's core clinical claim involves amyloid *progression* risk; however, the model's ability to predict actual longitudinal progression to Alzheimer's dementia has not been validated in prospective outcome data.

The survival C-index reported on the ADNI test set is **0.651 (95% CI: 0.525–0.788)** — modestly above the null value of 0.5 but with a wide confidence interval spanning weak to moderate concordance. The MMSE trajectory R² is 0.047, indicating that the model explains only ~5% of variance in MMSE change over time.

These exploratory results do not constitute validated longitudinal prognostic performance. Clinicians must not interpret the risk score as a validated predictor of time-to-dementia conversion.

*Recommendation:* A prospective longitudinal validation study with minimum 18-month follow-up and confirmed amyloid outcomes (PET or CSF) is required to substantiate longitudinal prognostic claims. Until such data are available, the risk score should be communicated to clinicians as a cross-sectional amyloid burden risk estimate, not a progression trajectory prediction.

---

### 9.1 Additional Recommendations

| Recommendation | Priority |
|---|---|
| Obtain real acoustic and motor recordings for ADNI-comparable cohort | High |
| Conduct longitudinal prospective validation study (≥18 months) | High |
| Expand subgroup analysis with granular race/ethnicity data | High |
| Re-run HPO with expanded trial budget (≥100 trials) before next version | Medium |
| Conduct dedicated APOE4-carrier fairness remediation analysis | High |
| Validate on non-English-speaking populations before international expansion | Medium |
| Implement post-market surveillance registry for real-world performance tracking | High |
| Develop assay-harmonization module for non-Elecsys pTau217 platforms | Medium |

---

## Document Sign-off

| Role | Name | Date |
|---|---|---|
| Clinical Documentation Specialist | *[Signature required]* | 2025 |
| Medical Director | *[Signature required]* | 2025 |
| Regulatory Affairs Lead | *[Signature required]* | 2025 |
| Data Science Lead | *[Signature required]* | 2025 |

---

*This Model Card follows Mitchell et al. (2019) "Model Cards for Model Reporting," adapted for FDA AI/ML SaMD guidance and EU MDR requirements. All performance metrics are derived from validated evaluation runs on held-out test sets. Phase 2A results are invalidated due to data leakage and must not be cited in any form.*