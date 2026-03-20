# RISK MANAGEMENT FILE

## NeuroFusion-AD: Multimodal GNN for Alzheimer's Disease Progression Prediction

---

```
Document ID:        RMF-001
Version:            1.0
Classification:     CONFIDENTIAL – REGULATORY AFFAIRS
Status:             APPROVED FOR SUBMISSION
Date Created:       2025-01-01
Last Revised:       2025-01-01
Device Name:        NeuroFusion-AD
Device Version:     1.0.0
Intended Use:       Clinical Decision Support for AD Progression Risk Assessment
                    in MCI Patients (Age 50–90)
Regulatory Class:   SaMD – IEC 62304 Class B
Standards Applied:  ISO 14971:2019, IEC 62304:2015+AMD1:2015,
                    IEC 62366-1:2015, ISO 13485:2016, FDA 21 CFR Part 11
```

---

## DOCUMENT CONTROL

| Version | Date | Author | Reviewer | Approver | Changes |
|---------|------|--------|----------|----------|---------|
| 0.1 | 2024-10-01 | J. Chen, RAO | M. Patel, QA | — | Initial draft |
| 0.2 | 2024-11-15 | J. Chen, RAO | M. Patel, QA | — | Hazard analysis expanded |
| 0.3 | 2024-12-01 | J. Chen, RAO | Dr. A. Ross, Clinical | — | Residual risk review |
| 1.0 | 2025-01-01 | J. Chen, RAO | M. Patel, QA | Dr. L. Huang, CMO | Approved for submission |

---

## TABLE OF CONTENTS

1. [Scope and Purpose](#1-scope-and-purpose)
2. [Risk Management Plan](#2-risk-management-plan)
   - 2.1 Intended Use and Intended Users
   - 2.2 Scope of Risk Management Activities
   - 2.3 Responsibilities and Authorities
   - 2.4 Risk Management Process Activities
   - 2.5 Criteria for Risk Acceptability
   - 2.6 Verification of Risk Control Measures
   - 2.7 Residual Risk Evaluation Policy
3. [Risk Analysis](#3-risk-analysis)
   - 3.1 Intended Use and Foreseeable Misuse Analysis
   - 3.2 Hazard Identification Methodology
   - 3.3 Risk Estimation Scales
4. [Hazard Register](#4-hazard-register)
5. [Risk Evaluation and Control](#5-risk-evaluation-and-control)
6. [Residual Risk Summary](#6-residual-risk-summary)
7. [Overall Residual Risk Evaluation](#7-overall-residual-risk-evaluation)
8. [Risk Management Review](#8-risk-management-review)
9. [Post-Market Risk Activities](#9-post-market-risk-activities)
10. [Traceability Matrix](#10-traceability-matrix)
11. [References and Standards](#11-references-and-standards)

---

## 1. SCOPE AND PURPOSE

### 1.1 Purpose

This Risk Management File (RMF) documents all risk management activities performed for NeuroFusion-AD in accordance with **ISO 14971:2019** — *Medical devices – Application of risk management to medical devices*. This document serves as the master record for the identification, analysis, evaluation, control, and monitoring of risks associated with the NeuroFusion-AD software as a medical device (SaMD).

This RMF is intended to support:
- **FDA De Novo clearance** (21 CFR Part 515B), referencing predicate Prenosis Sepsis ImmunoScore
- **EU MDR Class IIa certification** (EU 2017/745), Notified Body review per Annex IX/XI
- Ongoing post-market surveillance and risk monitoring obligations

### 1.2 Device Description

**NeuroFusion-AD** is a multimodal Graph Neural Network (GNN)-based Clinical Decision Support (CDS) system that aids clinicians in assessing Alzheimer's Disease (AD) progression risk in patients with Mild Cognitive Impairment (MCI). The device is not intended to replace clinical judgment; it provides quantitative risk stratification as a supplementary tool.

**Technical Architecture:**
- Four modality encoders: fluid biomarkers, acoustic, motor, clinical/demographic
- Cross-Modal Attention mechanism (768-dimensional, 8 attention heads)
- GraphSAGE GNN (3 layers) for multi-patient relational modeling
- Multi-task outputs: risk classification (target AUC ≥ 0.85), CDRSOB regression (RMSE ≤ 3.0), survival analysis (C-index ≥ 0.75)
- Deployment: FastAPI microservice, PostgreSQL 14, Docker/Kubernetes

**Intended Patient Population:** MCI patients aged 50–90 years

**Intended Users:** Licensed neurologists, geriatricians, and trained clinical staff in hospital and specialty clinic settings

**Operating Environment:** Clinical information systems (EHR-integrated or standalone), requiring connectivity and validated data input pipelines

---

## 2. RISK MANAGEMENT PLAN

> **Regulatory Basis:** ISO 14971:2019, §4 — Risk Management Plan

### 2.1 Intended Use and Intended Users

#### 2.1.1 Intended Use Statement

NeuroFusion-AD is intended as a prescription-use clinical decision support tool to assist qualified clinicians in quantifying Alzheimer's Disease progression risk in patients diagnosed with Mild Cognitive Impairment (MCI), aged 50–90 years. The system integrates fluid biomarker data, acoustic speech analysis, motor assessment data, and clinical/demographic information to generate a probabilistic risk stratification score, a projected cognitive decline trajectory, and a survival analysis estimate.

> ⚠️ **The device output is advisory only. Clinical decisions remain the sole responsibility of the treating clinician. The device is not intended for use as the sole basis for diagnosis, treatment initiation, or prognosis.**

#### 2.1.2 Intended Users

| User Class | Qualification Requirements | Training Required |
|------------|---------------------------|-------------------|
| Primary User | Licensed neurologist or geriatrician | NeuroFusion-AD operator training (≥4h) |
| Secondary User | Clinical nurse specialist, PA (under supervision) | NeuroFusion-AD operator training (≥4h) |
| IT/Admin | Healthcare IT administrator | System administration training |
| Excluded | General practitioners (without specialist oversight) | N/A |

#### 2.1.3 Foreseeable Misuse

| Misuse Scenario | Likelihood | Addressed In |
|-----------------|------------|--------------|
| Use in patients outside age range (50–90) | Medium | H-004, H-008; input validation |
| Use as sole diagnostic basis without clinical review | High | H-006, labeling, UI design |
| Operation with corrupted or missing input data | Medium | H-004; input validation module |
| Use on non-MCI patients (e.g., normal cognition, advanced dementia) | Medium | H-008, labeling, indication constraint |
| Unauthorized data access or system intrusion | Low | H-005; cybersecurity controls |

### 2.2 Scope of Risk Management Activities

Risk management activities apply to the **entire lifecycle** of NeuroFusion-AD, encompassing:

- **Design and development phase:** Architecture decisions, training data curation, model validation
- **Verification and validation phase:** Clinical study design, performance benchmarking, subgroup analysis
- **Manufacturing/deployment phase:** Containerization, CI/CD pipeline security, infrastructure hardening
- **Post-market phase:** Performance monitoring, adverse event surveillance, model retraining protocol

**Exclusions:** Risk management for downstream clinical decision execution (e.g., medication dosing, hospitalization decisions) is outside device scope. Risks arising from clinician judgment following review of device output are addressed through use-error mitigation and labeling only.

### 2.3 Responsibilities and Authorities

| Role | Name/Title | Responsibilities |
|------|-----------|-----------------|
| Risk Management Owner | Chief Medical Officer (CMO) | Ultimate accountability for RMF approval and residual risk acceptance |
| Regulatory Affairs Officer (RAO) | Regulatory Affairs Lead | RMF authorship, submission coordination, ISO 14971 compliance |
| Quality Assurance Manager | QA Director | Process compliance, document control, audit readiness |
| Clinical Affairs Lead | Senior Clinical Scientist | Clinical hazard identification, harm severity classification |
| ML/Software Engineering Lead | Principal ML Engineer | Technical risk identification, mitigation implementation |
| Cybersecurity Officer | Information Security Lead | Security threat modeling, cybersecurity controls |
| Usability Engineering Lead | Human Factors Specialist | Use-error analysis, IEC 62366-1 alignment |
| Risk Management Review Board | Cross-functional (above + Legal) | Periodic risk review, residual risk acceptance decisions |

**Risk Management Review Cadence:**
- Quarterly during development and clinical trial phases
- Semi-annually post-market (minimum)
- Ad hoc following any serious adverse event, significant software update, or field safety signal

### 2.4 Risk Management Process Activities

Per ISO 14971:2019, §4.1, the following activities constitute the complete risk management process:

```
┌─────────────────────────────────────────────────────────────┐
│                    RISK MANAGEMENT PROCESS                  │
│                                                             │
│  1. RISK ANALYSIS                                           │
│     ├── Intended use & foreseeable misuse identification    │
│     ├── Hazard identification (FMEA, FTA, HAZOP, HTA)      │
│     └── Risk estimation (severity × probability)           │
│                                                             │
│  2. RISK EVALUATION                                         │
│     ├── Compare against acceptability criteria             │
│     └── Prioritize for risk control                        │
│                                                             │
│  3. RISK CONTROL                                           │
│     ├── Option analysis (inherent safety > protection >    │
│     │   information for safety)                            │
│     ├── Implementation verification                        │
│     └── Residual risk and new risk assessment              │
│                                                             │
│  4. OVERALL RESIDUAL RISK EVALUATION                       │
│     └── Benefit-risk analysis                              │
│                                                             │
│  5. RISK MANAGEMENT REVIEW                                 │
│     └── Completeness, consistency, approval                │
│                                                             │
│  6. PRODUCTION & POST-MARKET INFORMATION                   │
│     └── Surveillance, feedback loops, RMF updates          │
└─────────────────────────────────────────────────────────────┘
```

### 2.5 Criteria for Risk Acceptability

> **Regulatory Basis:** ISO 14971:2019, §4.2(c)

#### 2.5.1 Severity Scale

| Level | Code | Definition | Clinical Example |
|-------|------|-----------|-----------------|
| Catastrophic | S5 | Patient death or permanent severe disability | Fatal medication error from misdiagnosis |
| Critical | S4 | Serious, irreversible health consequences | Inappropriate cessation of neuroprotective therapy; institutionalization |
| Serious | S3 | Reversible or moderate long-term health consequences | Delayed treatment initiation; unnecessary lumbar puncture |
| Moderate | S2 | Minor health consequence, manageable clinical impact | Unnecessary follow-up testing; patient anxiety |
| Negligible | S1 | No clinically meaningful health consequence | Cosmetic UI error; non-impacting data display issue |

#### 2.5.2 Probability Scale

| Level | Code | Qualitative Definition | Quantitative Estimate |
|-------|------|----------------------|----------------------|
| High | P4 | Likely to occur frequently in device lifetime | > 1 in 100 uses |
| Medium | P3 | Likely to occur occasionally | 1 in 100 to 1 in 1,000 uses |
| Low | P2 | Unlikely but possible | 1 in 1,000 to 1 in 10,000 uses |
| Very Low | P1 | Highly unlikely, only in exceptional circumstances | < 1 in 10,000 uses |

#### 2.5.3 Risk Acceptability Matrix

```
                        SEVERITY
                ┌──────────────────────────────────────────────────┐
                │  Negligible  Moderate   Serious   Critical  Catas.│
                │    (S1)       (S2)       (S3)       (S4)     (S5) │
  P ┌───────────┼──────────────────────────────────────────────────┤
  R │ High (P4) │  ALARP     ALARP    UNACCEPTABLE UNACCEPT. UNACCEPT│
  O ├───────────┼──────────────────────────────────────────────────┤
  B │ Med. (P3) │  ACCEPT    ALARP      ALARP     UNACCEPT. UNACCEPT│
  A ├───────────┼──────────────────────────────────────────────────┤
  B │ Low (P2)  │  ACCEPT    ACCEPT     ALARP       ALARP    UNACCEPT│
  I ├───────────┼──────────────────────────────────────────────────┤
  L │VLow (P1)  │  ACCEPT    ACCEPT     ACCEPT      ALARP    ALARP  │
  Y └───────────┴──────────────────────────────────────────────────┘

  LEGEND:
  ■ UNACCEPTABLE — Risk must be reduced; device cannot be released
  ▣ ALARP        — Risk must be As Low As Reasonably Practicable
  □ ACCEPTABLE   — Risk acceptable with standard controls; document rationale
```

#### 2.5.4 Risk Acceptability Decision Rules

| Decision | Criteria | Required Action |
|----------|----------|----------------|
| **UNACCEPTABLE** | Catastrophic at any probability; Critical at Medium or above; Serious at High | Mandatory risk reduction before proceeding; escalate to CMO |
| **ALARP** | All medium/high residual risks not meeting unacceptable threshold | Must demonstrate ALARP through documented analysis; benefit-risk justification required |
| **ACCEPTABLE** | Low or Very Low risk after mitigation; residual risks in acceptable zone | Document rationale; no further action required unless post-market signals emerge |

### 2.6 Verification of Risk Control Measures

All risk control measures shall be:

1. **Implemented** – Confirmed by engineering change notice or configuration record
2. **Verified** – Confirmed as effective through documented testing (unit test, integration test, clinical validation, penetration test, etc.)
3. **Validated** – Where applicable, confirmed through usability testing per IEC 62366-1
4. **Assessed for new hazards** – Mitigation measures themselves must not introduce new risks above the acceptable threshold
5. **Traced** – Each control measure linked to a specific hazard ID in the Traceability Matrix (§10)

### 2.7 Residual Risk Evaluation Policy

Following implementation of all risk controls:
- All residual risks classified as **ACCEPTABLE** or **ALARP with documented justification** shall be compiled in §6
- The **overall residual risk** shall be evaluated per ISO 14971:2019, §8, using a benefit-risk analysis
- If any residual risk remains UNACCEPTABLE, the device shall **not** be released until further risk reduction is achieved
- The CMO shall provide written approval of the overall residual risk evaluation prior to regulatory submission

---

## 3. RISK ANALYSIS

> **Regulatory Basis:** ISO 14971:2019, §5

### 3.1 Hazard Identification Methodology

The following structured analysis methods were employed to comprehensively identify hazards:

| Method | Application Area | Lead |
|--------|-----------------|------|
| **FMEA** (Failure Mode & Effects Analysis) | Software components, ML pipeline, data flows | ML Engineering |
| **FTA** (Fault Tree Analysis) | System-level failure scenarios, integration failures | Software Engineering |
| **HAZOP** (Hazard and Operability Study) | Operational workflows, user interface interactions | Usability Engineering |
| **STRIDE Threat Modeling** | Cybersecurity and PHI protection | Cybersecurity Officer |
| **Clinical Expert Review** | Harm classification, clinical scenario analysis | Clinical Affairs |
| **Human Factors Analysis (HTA)** | Use-error scenarios, automation bias assessment | Human Factors |
| **Literature Review** | Known failure modes for AI/ML CDS in neurology | Clinical Affairs + RAO |

### 3.2 Hazard Categories

Eight primary hazard categories were identified:

| # | Category | Hazard ID Range |
|---|----------|----------------|
| 1 | Diagnostic Output Error – False Negative | H-001 |
| 2 | Diagnostic Output Error – False Positive | H-002 |
| 3 | Algorithmic Bias – Demographic Subgroups | H-003 |
| 4 | Data Quality Failure – Corrupted/Invalid Inputs | H-004 |
| 5 | Cybersecurity Breach – PHI Exposure | H-005 |
| 6 | Use Error – Automation Bias / Over-Reliance | H-006 |
| 7 | System Availability Failure – Downtime | H-007 |
| 8 | Software Error – Incorrect Calculation | H-008 |

### 3.3 Risk Estimation Scales Reference

See §2.5.1 (Severity), §2.5.2 (Probability), and §2.5.3 (Acceptability Matrix).

---

## 4. HAZARD REGISTER

> **Regulatory Basis:** ISO 14971:2019, §§5, 6, 7

---

### H-001: False Negative — Missed High-Risk AD Progression

| Field | Detail |
|-------|--------|
| **Hazard ID** | H-001 |
| **Hazard Description** | The model outputs a low-risk classification or understates cognitive decline trajectory for a patient who is at high risk of or actively experiencing rapid AD progression |
| **Hazardous Situation** | A clinician reviews the NeuroFusion-AD output showing low/moderate risk and reduces monitoring frequency, delays referral for specialist evaluation, or withholds initiation of disease-modifying therapy (e.g., lecanemab) based on the erroneous low-risk output |
| **Harm** | Delayed or missed opportunity for treatment of AD progression; irreversible cognitive decline that could have been slowed with timely intervention; psychological harm to patient and family due to unanticipated disease acceleration |
| **Severity (Pre-Mitigation)** | **Critical (S4)** — Missed treatment window for disease-modifying therapy represents serious, potentially irreversible clinical harm |
| **Probability (Pre-Mitigation)** | **Medium (P3)** — Model AUC target ≥0.85 implies residual false-negative rate; complex biomarker profiles in atypical AD phenotypes increase risk; performance varies across subpopulations |
| **Risk Level (Pre-Mitigation)** | 🔴 **UNACCEPTABLE** (S4 × P3) |
| **Causal Factors** | Model underperformance in rare biomarker profiles; training data underrepresentation; input data noise or out-of-range values handled silently; over-smoothing in GNN layers; distribution shift in deployment data |

**Risk Control Measures:**

| Control ID | Type | Description | Verification Method |
|-----------|------|-------------|-------------------|
| RC-001-A | Inherent Safety | Minimum performance gate: AUC ≥ 0.85 on held-out clinical validation set before release; continuous monitoring with automated alert if AUC drops below 0.82 in production | Clinical validation report; performance dashboard |
| RC-001-B | Inherent Safety | Uncertainty quantification: Monte Carlo Dropout ensemble (N=50 forward passes) provides confidence intervals; high-uncertainty outputs flagged with explicit warning overlay | Technical validation; unit testing |
| RC-001-C | Protective Measure | Clinician override protocol: All low-risk outputs accompanied by mandatory acknowledgment screen listing key clinical factors requiring independent verification | Usability test; UI specification |
| RC-001-D | Information for Safety | Labeling and IFU: Explicit statement that negative/low-risk output does not exclude AD progression; device is supplementary tool only; sensitivity/specificity statistics provided per subgroup | Label review; submission package |
| RC-001-E | Inherent Safety | Subgroup performance monitoring: Automated disaggregated metrics tracking by age decile, sex, APOE ε4 status, and race/ethnicity; alert threshold if any subgroup AUC < 0.80 | Post-market surveillance plan |
| RC-001-F | Protective Measure | Mandatory clinical correlation requirement embedded in UI: Structured prompt for clinician to document independent assessment of MMSE trajectory before acting on low-risk output | UI specification; usability validation |

| Field | Detail |
|-------|--------|
| **Severity (Post-Mitigation)** | **Critical (S4)** — Severity of harm unchanged by controls |
| **Probability (Post-Mitigation)** | **Very Low (P1)** — Uncertainty flags, mandatory acknowledgment, and performance monitoring substantially reduce probability of undetected false negatives reaching harmful decisions |
| **Residual Risk Level** | 🟡 **ALARP** (S4 × P1) — Benefit-risk justification provided in §7 |
| **Residual Risk Acceptability** | Accepted with documented ALARP justification; no further technically feasible risk reduction identified without compromising device utility |

---

### H-002: False Positive — Unnecessary Diagnostic Workup

| Field | Detail |
|-------|--------|
| **Hazard ID** | H-002 |
| **Hazard Description** | The model outputs a high-risk classification for a patient who does not have significant AD progression risk |
| **Hazardous Situation** | A clinician acting on the high-risk output initiates an unnecessary invasive diagnostic workup (e.g., lumbar puncture for CSF biomarkers, PET amyloid imaging, or referral for clinical trial enrollment) or initiates pharmacological treatment with significant side-effect profiles (e.g., ARIA risk with anti-amyloid antibody therapy) |
| **Harm** | Physical harm from unnecessary invasive procedures (LP complications: post-dural puncture headache, nerve injury); significant psychological distress, anxiety, and depression from false AD prognosis; financial harm; unnecessary exposure to drug adverse effects; stigma and social consequences |
| **Severity (Pre-Mitigation)** | **Serious (S3)** — Reversible physical harms from procedures; significant psychological harm; no permanent neurological sequelae expected from false positive alone |
| **Probability (Pre-Mitigation)** | **Medium (P3)** — Model specificity at AUC 0.85 operating threshold implies non-trivial false-positive rate; conservative clinician behavior may increase unnecessary testing |
| **Risk Level (Pre-Mitigation)** | 🟠 **ALARP** (S3 × P3) |
| **Causal Factors** | Model overfitting to high-risk patterns; biomarker values near decision boundaries triggering high-risk output; demographic features (older age) driving false elevation; lack of clinician familiarity with model operating characteristics |

**Risk Control Measures:**

| Control ID | Type | Description | Verification Method |
|-----------|------|-------------|-------------------|
| RC-002-A | Inherent Safety | Optimized operating threshold: Decision threshold selected to maximize balanced accuracy; F1-score and NPV/PPV tradeoffs documented; multiple threshold options provided to institution (e.g., high-sensitivity vs. balanced mode) | Validation report; ROC analysis |
| RC-002-B | Inherent Safety | Confidence score display: All outputs show posterior probability with 95% credible interval; outputs near decision boundary (e.g., 0.45–0.55 risk score) flagged as "Indeterminate – Clinical Judgment Required" | Technical specification; UI validation |
| RC-002-C | Information for Safety | PPV/NPV statistics by prevalence and subgroup provided in IFU; expected false-positive rates per 1,000 patients communicated clearly in labeling | Label review; clinical affairs review |
| RC-002-D | Protective Measure | Clinical workflow safeguard: UI requires structured documentation of corroborating clinical evidence before invasive follow-up action is recorded; soft interlock warning when risk score is moderate-high without corroborating MMSE decline | UI specification; usability validation |
| RC-002-E | Information for Safety | Mandatory operator training module covering model limitations, expected false-positive rates, and clinical corroboration requirements; competency assessment required before clinical use | Training records; competency assessment |

| Field | Detail |
|-------|--------|
| **Severity (Post-Mitigation)** | **Serious (S3)** |
| **Probability (Post-Mitigation)** | **Low (P2)** |
| **Residual Risk Level** | 🟡 **ALARP** (S3 × P2) |
| **Residual Risk Acceptability** | Accepted; residual probability reflects irreducible model uncertainty; clinical corroboration requirement and IFU communication are the primary final barriers |

---

### H-003: Algorithmic Bias — Disparate Performance Across Demographic Subgroups

| Field | Detail |
|-------|--------|
| **Hazard ID** | H-003 |
| **Hazard Description** | The GNN model exhibits significantly different predictive performance (AUC, sensitivity, specificity) across demographic subgroups defined by race/ethnicity, sex, age decile, socioeconomic status, language background, or APOE ε4 carrier status |
| **Hazardous Situation** | Subgroup-specific model underperformance results in systematically higher false-negative rates for underrepresented populations (e.g., Black/African American patients, Hispanic patients, or patients with non-English language backgrounds), leading to unequal access to timely AD diagnosis and treatment for those populations |
| **Harm** | Systematic health equity harm: entire demographic groups receive disproportionately delayed or missed diagnosis; amplification of existing healthcare disparities in AD care; disproportionate exposure to false positives in over-represented groups |
| **Severity (Pre-Mitigation)** | **Critical (S4)** — Systematic, population-level harm affecting a definable patient group; irreversible consequences of delayed AD treatment |
| **Probability (Pre-Mitigation)** | **High (P4)** — Well-documented phenomenon in AI/ML medical devices; training datasets for AD (e.g., ADNI) have historically been predominantly white, non-Hispanic; acoustic and motor encoders may show cultural/linguistic variation |
| **Risk Level (Pre-Mitigation)** | 🔴 **UNACCEPTABLE** (S4 × P4) |
| **Causal Factors** | Non-representative training data; proxy variable encoding (speech, motor patterns vary with education, language, culture); insufficient sample sizes for underrepresented subgroups in clinical validation; GNN neighborhood aggregation may amplify majority-class patterns |

**Risk Control Measures:**

| Control ID | Type | Description | Verification Method |
|-----------|------|-------------|-------------------|
| RC-003-A | Inherent Safety | Training data diversity requirement: Prospective data collection protocol mandates minimum subgroup representation (≥15% Black/AA, ≥15% Hispanic/Latino, ≥45% female, ≥20% APOE ε4 carriers); supplemented with NACC, MAACS, WHICAP cohort data | Data audit report; data card documentation |
| RC-003-B | Inherent Safety | Fairness-aware training: Adversarial debiasing layer applied during GNN training to penalize demographic performance disparities; equalized odds constraints incorporated in multi-task loss function | Model training specification; ablation study |
| RC-003-C | Inherent Safety | Mandatory disaggregated validation: Pre-market clinical validation report must include AUC, sensitivity, and specificity broken out by sex, age decile, race/ethnicity, education level, and APOE ε4 status; no subgroup may have AUC < 0.80 | Clinical validation protocol; statistical analysis plan |
| RC-003-D | Protective Measure | Deployment-time subgroup monitoring: Automated drift detection (PSI, KS test) per subgroup in production; performance dashboard alerts if any subgroup AUC drops below 0.80 or diverges > 0.05 from overall AUC | Software specification; monitoring runbook |
| RC-003-E | Information for Safety | Labeling disclosure: Subgroup performance statistics published in IFU; explicit statement of known limitations for subgroups with <500 validation patients; intended use restriction to populations with ≥100 patients in training data | Label review |
| RC-003-F | Protective Measure | Model card and algorithmic impact assessment: Published as part of regulatory submission and public documentation; reviewed annually for equity | Model card document; annual review record |

| Field | Detail |
|-------|--------|
| **Severity (Post-Mitigation)** | **Critical (S4)** — Severity remains if controls fail |
| **Probability (Post-Mitigation)** | **Low (P2)** — Active debiasing, diverse data, and monitoring substantially reduce systematic disparity |
| **Residual Risk Level** | 🟡 **ALARP** (S4 × P2) |
| **Residual Risk Acceptability** | ALARP accepted with conditions: Subgroup AUC requirement (≥0.80) is a release gate; post-market equity monitoring is mandatory |

---

### H-004: Data Quality Failure — Corrupted, Missing, or Out-of-Range Input Data

| Field | Detail |
|-------|--------|
| **Hazard ID** | H-004 |
| **Hazard Description** | Input biomarker values, acoustic features, motor data, or clinical/demographic variables are corrupted, missing, outside validated ranges, or from an incorrect patient, and the system processes them without adequate detection or rejection |
| **Hazardous Situation** | The model silently ingests corrupt or invalid data (e.g., pTau-217 value of 0.05 pg/mL — below minimum validated range of 0.1 pg/mL; MMSE score of 35 — above maximum of 30; wrong-patient data substitution due to EHR mapping error) and generates an erroneous risk prediction that is presented to the clinician without any indication of data quality issues |
| **Harm** | Patient receives clinical decision based on erroneous risk score; potential false negative or false positive harm cascades (see H-001, H-002); in worst case, wrong-patient data results in treatment decision for the wrong individual |
| **Severity (Pre-Mitigation)** | **Critical (S4)** — Wrong-patient scenario could result in harmful treatment for incorrect patient; even correct patient with corrupt data may receive seriously misleading risk estimate |
| **Probability (Pre-Mitigation)** | **Medium (P3)** — EHR data quality issues are well-documented; lab transcription errors, sensor calibration drift, network transmission corruption, and mapping errors are realistic failure modes |
| **Risk Level (Pre-Mitigation)** | 🔴 **UNACCEPTABLE** (S4 × P3) |
| **Causal Factors** | EHR interface mapping errors; lab instrument calibration drift; acoustic/motor sensor hardware failure; network packet corruption during data transmission; API input not validated before model inference; missing modality not handled gracefully |

**Hard Constraint Input Validation Ranges:**

| Biomarker | Valid Range | Action if Out-of-Range |
|-----------|------------|----------------------|
| pTau-217 | 0.1 – 100 pg/mL | REJECT inference; return error code E-001 |
| Aβ42/40 ratio | 0.01 – 0.30 | REJECT inference; return error code E-002 |
| NfL | 5 – 200 pg/mL | REJECT inference; return error code E-003 |
| MMSE | 0 – 30 | REJECT inference; return error code E-004 |
| Age | 50 – 90 years | WARNING; proceed