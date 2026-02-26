---
document_id: regulatory-strategy
generated: 2026-02-26T22:55:20.924914
batch_id: msgbatch_01DTMbBbcyvTviGxwBhePxKr
status: DRAFT — requires human review before approval
---

# REGULATORY STRATEGY DOCUMENT

---

**Document ID:** REG-001 v1.0
**Product:** NeuroFusion-AD — Multimodal Graph Neural Network for Alzheimer's Disease Progression Prediction
**Document Type:** Regulatory Strategy Document
**Version:** 1.0
**Prepared By:** Regulatory Affairs Office
**Review Status:** DRAFT — For Internal Review
**Date:** 2025-07-14
**Classification:** CONFIDENTIAL — Regulatory Sensitive

---

## DOCUMENT CONTROL

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2025-06-01 | RA Office | Initial draft |
| 0.9 | 2025-07-01 | RA Office | Internal review incorporation |
| 1.0 | 2025-07-14 | RA Office | Released for stakeholder review |

**Approved By:**
- [ ] Chief Medical Officer
- [ ] VP Engineering
- [ ] Legal Counsel
- [ ] Quality Assurance Director

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [FDA De Novo Pathway Analysis](#2-fda-de-novo-pathway-analysis)
3. [EU MDR Class IIa Analysis](#3-eu-mdr-class-iia-analysis)
4. [IEC 62304 Software Safety Classification and Compliance Plan](#4-iec-62304-software-safety-classification-and-compliance-plan)
5. [ISO 14971 Risk Management Plan Summary](#5-iso-14971-risk-management-plan-summary)
6. [Key Regulatory Milestones and Timeline](#6-key-regulatory-milestones-and-timeline)
7. [Regulatory Risks and Mitigations](#7-regulatory-risks-and-mitigations)
8. [References and Applicable Standards](#8-references-and-applicable-standards)

---

## 1. EXECUTIVE SUMMARY

### 1.1 Product Overview

NeuroFusion-AD is a Software as a Medical Device (SaMD) that employs a multimodal Graph Neural Network (GNN) architecture to aid clinicians in assessing Alzheimer's Disease (AD) progression risk in patients diagnosed with Mild Cognitive Impairment (MCI) between the ages of 50 and 90. The system integrates four biomarker modalities — fluid biomarkers (pTau-217, Abeta42/40, NfL), acoustic features, motor performance metrics, and clinical/demographic data — through a cross-modal attention mechanism (768-dimensional, 8-head) and a GraphSAGE GNN (3 layers). NeuroFusion-AD delivers three complementary outputs: binary risk classification (target AUC ≥ 0.85), continuous cognitive trajectory regression (target RMSE ≤ 3.0 MMSE points), and time-to-progression survival analysis (target C-index ≥ 0.75).

The device is intended solely as a **Clinical Decision Support (CDS)** tool. It does not autonomously initiate, modify, or terminate therapeutic interventions. All clinical decisions remain the responsibility of the treating physician.

### 1.2 Regulatory Approach Summary

NeuroFusion-AD will pursue **dual regulatory clearance** in the United States and the European Union through parallel, coordinated pathways:

| Jurisdiction | Pathway | Classification | Target Milestone |
|---|---|---|---|
| United States | FDA De Novo | SaMD, non-exempt | Month 16 (submission) |
| European Union | EU MDR | Class IIa, Rule 11 | Month 16 (Technical File completion) |
| International | IEC 62304 | Software Safety Class B | Continuous (development lifecycle) |
| Quality/Risk | ISO 14971 | Full risk management process | Continuous (parallel to development) |

The De Novo pathway is selected over 510(k) because no substantially equivalent predicate with identical technological characteristics exists. The Prenosis Sepsis ImmunoScore (DEN200057), a multimodal AI-based risk stratification tool granted De Novo clearance, represents the most functionally analogous predicate device and is used to anchor our equivalence argument at the intended use and technological level.

In the EU, NeuroFusion-AD is classified as a **Class IIa medical device** under MDR (EU) 2017/745, Rule 11 (software intended to provide information used to make decisions with diagnosis or therapeutic purposes), requiring Notified Body involvement for conformity assessment.

### 1.3 Strategic Priorities

The following strategic priorities govern this regulatory program:

> **Priority 1 — Pre-submission Engagement:** Secure FDA Pre-Sub feedback on predicate justification, performance thresholds, and clinical study design before investing in full study execution. This is the single highest-leverage regulatory action in Phase 1.

> **Priority 2 — Clinical Evidence Quality:** The strength of clinical evidence — particularly subgroup performance data across age, sex, race/ethnicity, and APOE-ε4 carrier status — will be the primary determinant of submission success in both jurisdictions.

> **Priority 3 — Algorithm Transparency:** Explainability features (attention weights, SHAP values, confidence intervals) must be validated as clinically meaningful, not merely technically present, to satisfy FDA's PCCP expectations and MDR General Safety and Performance Requirements (GSPR) Annex I.

> **Priority 4 — Harmonization:** Where possible, technical documentation, risk management artifacts, and clinical data packages should be architected to serve both FDA and EU submissions simultaneously, reducing duplication and timeline risk.

---

## 2. FDA DE NOVO PATHWAY ANALYSIS

### 2.1 Pathway Selection Rationale

#### 2.1.1 De Novo vs. 510(k): Decision Framework

The FDA 510(k) pathway requires demonstration of **substantial equivalence** to a legally marketed predicate device, meaning the new device must have the same intended use and the same or different technological characteristics, with the latter not raising new questions of safety/effectiveness. The De Novo pathway, by contrast, is appropriate for novel, low-to-moderate risk devices for which no valid predicate exists.

The decision to pursue De Novo rather than 510(k) is based on the following analysis:

| Evaluation Criterion | 510(k) Assessment | De Novo Assessment |
|---|---|---|
| Legally marketed predicate available? | No direct predicate; closest analogy is DEN200057 | De Novo does not require exact predicate |
| Same intended use as predicate? | Partially (AI-based risk stratification) | Yes, at a functional level |
| Same technological characteristics? | No — GNN architecture is novel in this context | Novel technology → De Novo appropriate |
| New safety/effectiveness questions raised? | Yes — multimodal fusion, GNN explainability, longitudinal outputs | Addressed through special controls |
| Risk classification | Class II | Class II (target) |
| Regulatory flexibility needed? | Limited under 510(k) | Yes — special controls tailored to device |

**Conclusion:** No legally marketed device exists with substantially equivalent intended use *and* technological characteristics to NeuroFusion-AD. The 510(k) pathway is therefore unavailable. De Novo under 21 CFR Part 515A is the appropriate pathway.

#### 2.1.2 Statutory Basis

- 21 U.S.C. § 513(f)(2) — De Novo classification request
- 21 CFR Part 515A — Procedures for a De Novo classification request
- FDA Guidance: "De Novo Classification Process (Evaluation of Automatic Class III Designation)," October 2021
- FDA Guidance: "Clinical Decision Support Software," September 2022
- FDA Guidance: "Artificial Intelligence/Machine Learning (AI/ML)-Based Software as a Medical Device Action Plan," January 2021

### 2.2 Predicate Device Analysis: Prenosis Sepsis ImmunoScore (DEN200057)

#### 2.2.1 Predicate Device Description

The Prenosis Sepsis ImmunoScore received De Novo clearance (DEN200057) as a Class II device with special controls. It is a machine learning-based clinical decision support tool that integrates multiple biomarker inputs to generate a sepsis risk stratification score for Emergency Department patients. The FDA's special controls established for DEN200057 include requirements for algorithm transparency, performance validation, labeling, and post-market surveillance.

#### 2.2.2 Substantial Equivalence Argument

While NeuroFusion-AD targets a different disease state, the **functional regulatory analogy** to DEN200057 is strong at the level of intended use, device category, and risk profile. The following table presents the equivalence argument:

| Equivalence Dimension | Prenosis Sepsis ImmunoScore (DEN200057) | NeuroFusion-AD | Assessment |
|---|---|---|---|
| **Device Category** | SaMD — AI/ML-based risk stratification | SaMD — AI/ML-based risk stratification | ✅ Equivalent |
| **Intended Use** | Aids clinical assessment of sepsis risk using multi-biomarker input | Aids clinical assessment of AD progression risk using multi-modal input | ✅ Equivalent at functional level |
| **User** | Emergency physician | Neurologist / primary care physician | ✅ Comparable HCP user |
| **Output type** | Risk score (not autonomous treatment) | Risk classification + trajectory + survival estimate (not autonomous treatment) | ✅ Equivalent: decision support only |
| **Technology** | ML-based multi-biomarker integration | GNN-based multimodal fusion | ⚠️ Different technology (justification below) |
| **Risk class** | Class II with special controls | Class II with special controls (target) | ✅ Equivalent risk tier |
| **Patient population** | Adult ED patients | Adults age 50–90 with MCI | ✅ Comparable adult population |
| **Clinical consequences of error** | Missed sepsis (life-threatening) | Missed progression (serious, not immediately life-threatening) | ✅ NeuroFusion-AD carries comparable or lower risk |

**Technological Differences and Safety/Effectiveness Assessment:**

NeuroFusion-AD employs a Graph Neural Network (GraphSAGE, 3 layers) with cross-modal attention, which differs from the ML architecture used in DEN200057. We assert that these technological differences **do not raise new questions of safety or effectiveness** for the following reasons:

1. **Explainability:** GNN attention weights provide interpretable, modality-level explanations. This is more, not less, transparent than opaque ensemble methods.
2. **Multimodal fusion:** Integration of heterogeneous inputs via attention mechanisms is an established technique in AI/ML SaMD (see FDA's PCCP framework).
3. **Output interpretability:** Multi-task outputs (classification, regression, survival) are accompanied by uncertainty quantification and confidence intervals, supporting informed clinical interpretation.
4. **Validation rigor:** Performance thresholds (AUC ≥ 0.85, RMSE ≤ 3.0, C-index ≥ 0.75) will be validated on independent, demographically representative datasets.

### 2.3 Special Controls Framework

Based on the DEN200057 order and FDA's AI/ML SaMD guidance, the following special controls are anticipated and have been proactively designed into NeuroFusion-AD's development program:

#### 2.3.1 Performance Validation Requirements

```
Special Control: Performance Testing — Statistical Thresholds
├── Classification: AUC ≥ 0.85 (95% CI lower bound ≥ 0.82)
├── Regression: RMSE ≤ 3.0 MMSE points on held-out test set
├── Survival: C-index ≥ 0.75 (bootstrapped 95% CI)
├── Subgroup analysis: Age (50-65, 66-75, 76-90), Sex, Race/Ethnicity (≥3 groups), APOE-ε4 status
└── Demographic parity: No subgroup AUC < 0.80 without labeled justification
```

#### 2.3.2 Algorithm Transparency and Explainability

```
Special Control: Transparency
├── Modality contribution: Attention weight visualization per inference
├── Feature attribution: SHAP values for top-5 contributing features
├── Confidence output: Prediction intervals with calibration evidence
├── Model card: Published performance by subgroup and input modality
└── PCCP: Pre-Specified Change Control Plan per FDA ML action plan
```

#### 2.3.3 Labeling Requirements

The intended use statement, indications for use, contraindications, and performance characteristics must be clearly specified in device labeling per 21 CFR 801 and FDA's CDS guidance. Critical labeling elements include:

- **Intended use:** "NeuroFusion-AD is intended as a clinical decision support tool to aid clinicians in assessing the risk of Alzheimer's Disease progression in patients aged 50–90 diagnosed with Mild Cognitive Impairment (MCI). The device provides a risk classification score, a projected cognitive trajectory, and a time-to-progression estimate. Clinical decisions must be made by a qualified healthcare professional and must not be based solely on NeuroFusion-AD outputs."
- **Intended user:** Neurologists, geriatricians, and primary care physicians with neurology training
- **Contraindications:** Patients outside the 50–90 age range; input values outside validated ranges (see §2.3.5); patients with dementia diagnoses prior to assessment
- **Limitations of use:** Not a diagnostic device; not validated for non-MCI populations; performance may vary outside validated demographic subgroups

#### 2.3.4 Post-Market Performance Monitoring

```
Special Control: Post-Market Surveillance
├── Real-world performance tracking: AUC, RMSE monitored quarterly
├── Input drift detection: Distribution shift alerts for all 4 modalities
├── Adverse event reporting: 21 CFR Part 803 MDR compliance
├── Annual performance reports submitted to FDA
└── Model update protocol: PCCP governs all algorithm updates
```

#### 2.3.5 Input Validation and Out-of-Range Handling

Hard constraints on input biomarker ranges are treated as a special control for safe operation:

| Biomarker | Valid Range | Out-of-Range Response |
|---|---|---|
| pTau-217 | 0.1 – 100 pg/mL | Inference blocked; clinician alert generated |
| Abeta42/40 | 0.01 – 0.30 | Inference blocked; clinician alert generated |
| NfL | 5 – 200 pg/mL | Inference blocked; clinician alert generated |
| MMSE | 0 – 30 | Inference blocked; clinician alert generated |

All out-of-range events are logged to the audit trail (ISO 27001-aligned) for post-market surveillance review.

### 2.4 Submission Timeline Estimate (Phase 3: Months 15–16)

The De Novo submission is targeted for **Month 16** of the program, consistent with Phase 3 execution. The following outlines the submission package composition:

#### 2.4.1 Submission Package Structure

```
De Novo Submission Package (21 CFR 515A)
│
├── Cover Letter and Administrative Section
│   ├── Device identification and intended use
│   ├── Proposed classification (Class II, with special controls)
│   └── Proposed product code and panel designation
│
├── Section A: Device Description
│   ├── Architecture description (GNN, attention mechanism, multi-task heads)
│   ├── Input/output specification
│   ├── Software description (IEC 62304 compliance summary)
│   └── System architecture diagram
│
├── Section B: Proposed Intended Use and Indications for Use
│   ├── Intended use statement
│   ├── Indications for use (21 CFR 801.109)
│   └── Contraindications and limitations
│
├── Section C: Substantial Equivalence Comparison to DEN200057
│   ├── Predicate device description
│   ├── Equivalence matrix (Table 2.2.2 of this document)
│   └── Technological difference justification
│
├── Section D: Risk-Benefit Analysis
│   ├── ISO 14971 risk management summary
│   ├── Benefit characterization (clinical utility data)
│   └── Residual risk acceptability statement
│
├── Section E: Performance Data
│   ├── Training/validation/test split methodology
│   ├── Primary performance metrics (AUC, RMSE, C-index)
│   ├── Subgroup performance analysis
│   ├── Calibration curves and reliability diagrams
│   ├── Failure mode analysis
│   └── Comparison to clinical standard of care benchmark
│
├── Section F: Labeling
│   ├── Instructions for use (IFU)
│   ├── Quick reference guide
│   └── Sample user interface screenshots
│
├── Section G: Proposed Special Controls
│   └── Draft special controls document
│
└── Section H: Post-Market Surveillance Plan
    ├── Real-world performance monitoring protocol
    ├── PCCP (Pre-Specified Change Control Plan)
    └── Adverse event reporting procedures
```

#### 2.4.2 FDA Review Timeline Estimate

| Phase | Activity | Estimated Duration | Notes |
|---|---|---|---|
| Administrative | Filing review | 15 business days | FDA confirms acceptance |
| Substantive review | Technical, clinical, software review | 90–150 calendar days | Novel device; full AI/ML review expected |
| Interactive review | Deficiency response, Q&A | 30–60 calendar days | Pre-Sub engagement reduces deficiencies |
| Decision | Clearance or denial | — | Target: De Novo clearance |
| **Total post-submission** | | **~6–9 months** | Based on comparable AI/ML De Novo precedents |

> **Note:** FDA's average De Novo review time for AI/ML SaMD has ranged from 6 to 14 months. Early Pre-Sub engagement (Phase 2) is the primary lever for compressing this timeline.

---

## 3. EU MDR CLASS IIA ANALYSIS

### 3.1 Classification Analysis

#### 3.1.1 MDR Rule 11 Application

Under MDR (EU) 2017/745, Annex VIII, Classification Rules, **Rule 11** applies to software:

> *"Software intended to provide information which is used to take decisions with diagnosis or therapeutic purposes is classified as Class IIa, except if such decisions have an impact that may cause: death or an irreversible deterioration of a person's state of health, in which case it is in Class III; or a serious deterioration in a person's state of health or a surgical intervention, in which case it is in Class IIb."*

**Application to NeuroFusion-AD:**

| Rule 11 Criterion | NeuroFusion-AD Assessment | Classification Implication |
|---|---|---|
| Is it software? | Yes — SaMD running on cloud/on-premise infrastructure | Rule 11 applies |
| Does it provide information for clinical decisions? | Yes — risk score, trajectory, survival estimate used by clinician | Rule 11 triggering condition met |
| Can decisions cause death or irreversible deterioration? | No — CDS only; physician retains decision authority; AD progression is not immediately life-threatening | Class III does NOT apply |
| Can decisions cause serious deterioration or surgical intervention? | No — no surgical pathway; misclassification may delay treatment adjustment but is not immediately serious | Class IIb does NOT apply |
| **Conclusion** | Moderate risk, diagnostic information support | **Class IIa** ✅ |

#### 3.1.2 Regulatory Classification Confirmation

- **MDR Classification:** Class IIa
- **Rule Applied:** Rule 11 (Annex VIII, MDR 2017/745)
- **MDCG Guidance Applied:** MDCG 2019-11 (Guidance on Qualification and Classification of Software in Regulation (EU) 2017/745), MDCG 2020-1 (Guidance on Clinical Evaluation for MDR)
- **NB Involvement Required:** Yes — Conformity assessment by Notified Body under Annex IX (QMS audit + technical documentation review) or Annex XI (product verification)

### 3.2 IVDR vs. MDR Determination

#### 3.2.1 Analysis Framework

The distinction between MDR (EU 2017/745) and IVDR (EU 2017/746) hinges on whether the device's primary intended purpose is to analyze specimens (in vitro) or to process clinical/physiological data more broadly.

| Criterion | MDR Applicability | IVDR Applicability | NeuroFusion-AD |
|---|---|---|---|
| Primary purpose | Medical device for diagnosis, monitoring, treatment | In vitro diagnostic — examination of specimens derived from the human body | Processes multi-modal data including fluid biomarkers *and* acoustic, motor, clinical data |
| Specimen analysis? | Not primary | Yes — primary function | Biomarkers are ONE of FOUR input modalities |
| Output target | Clinical assessment / patient management | Diagnostic information from specimen examination | Risk stratification (clinical assessment) |
| Predominant function | Integrative clinical decision support | Laboratory/diagnostic analysis | CDS across modalities |

**Conclusion: MDR Applies**

While NeuroFusion-AD ingests fluid biomarker values (pTau-217, Abeta42/40, NfL), these values are **not generated by the device** — they are input parameters originating from separately certified laboratory instruments. NeuroFusion-AD does not perform in vitro analysis; it performs **multimodal data integration and risk stratification**. The predominant intended purpose is clinical decision support for patient management, not in vitro specimen examination.

This determination is consistent with MDCG 2019-11 guidance, which directs classification based on the **primary intended purpose** and the **predominant mechanism of action** of the device.

> **Regulatory Recommendation:** Document this IVDR/MDR determination in the Technical File with explicit reference to MDCG 2019-11, §3.2. Obtain written Notified Body concurrence during Phase 1 engagement to preempt classification challenges during conformity assessment.

### 3.3 Notified Body Requirements

#### 3.3.1 Conformity Assessment Route

For Class IIa devices, MDR Article 52 requires conformity assessment. The applicable conformity assessment procedures are:

**Option A (Selected Pathway): Annex IX (QMS + Technical Documentation)**

```
Annex IX Conformity Assessment
├── Chapter I: QMS Audit
│   ├── ISO 13485:2016 certified Quality Management System
│   ├── NB audits design/development, production, post-market surveillance
│   └── Annual surveillance audits post-certification
│
└── Chapter II: Technical Documentation Assessment
    ├── NB reviews Technical Documentation (representative samples for Class IIa)
    ├── Clinical evaluation report reviewed
    └── Post-Market Clinical Follow-up (PMCF) plan assessed
```

**Rationale for Annex IX selection:** Annex IX provides a more comprehensive ongoing relationship with the Notified Body, which is strategically advantageous for an AI/ML device that will undergo iterative updates governed by our PCCP equivalent (Post-Market Performance Monitoring Plan). Annex XI (product verification) is more suitable for stable, non-adaptive devices.

#### 3.3.2 Notified Body Selection Criteria

The following criteria govern Notified Body selection:

| Criterion | Requirement |
|---|---|
| NANDO designation | Designated under MDR 2017/745 for Class IIa software/AI |
| Technical expertise | Demonstrated competence in AI/ML SaMD and neurology applications |
| Capacity | Availability within Phase 1 timeline; acceptable queue times |
| Geographic presence | EU-based with English-language review capability |
| Precedent | Prior Class IIa SaMD certifications under MDR |

**Candidate Notified Bodies (to be finalized in Phase 1):**
- TÜV SÜD Product Service GmbH (NB 0123)
- BSI Group The Netherlands B.V. (NB 2797)
- SGS Belgium NV (NB 1639)

> **Action Item [Phase 1, Month 2]:** Issue NB solicitation package including device description, intended use, preliminary classification rationale, and project timeline. Evaluate responses against criteria above. Execute NB engagement agreement by Month 4.

#### 3.3.3 Technical File Structure (MDR Annex II + III)

The Technical File for EU MDR submission will include the following mandatory elements:

```
Technical File — NeuroFusion-AD (MDR Annex II + III)
│
├── 1. Device Description and Specification (Annex II §1)
│   ├── Intended purpose
│   ├── UMDNS/GMDN code
│   ├── Risk class and applicable rule
│   └── New features vs. predicate (if applicable)
│
├── 2. Information Supplied by Manufacturer (Annex II §2)
│   ├── Label and IFU (MDR Annex I §23)
│   └── UDI (EUDAMED registration)
│
├── 3. Design and Manufacturing Information (Annex II §3)
│   ├── Software development lifecycle (IEC 62304 documentation)
│   ├── SOUP (Software of Unknown Provenance) list
│   └── Cybersecurity documentation
│
├── 4. GSPR Compliance (Annex II §4 → Annex I)
│   ├── General safety and performance requirements checklist
│   ├── Harmonized standard compliance matrix
│   └── CS (Common Specifications) compliance where applicable
│
├── 5. Benefit-Risk Analysis and Risk Management (Annex II §5)
│   ├── ISO 14971:2019 Risk Management File
│   └── Summary of residual risks and benefit-risk determination
│
├── 6. Product Verification and Validation (Annex II §6)
│   ├── Usability validation (IEC 62366-1)
│   ├── Performance validation data
│   ├── Cybersecurity testing results
│   └── Input validation test results
│
├── 7. Post-Market Surveillance (Annex II §7 → Annex III)
│   ├── PMS Plan
│   ├── PMCF Plan (Annex XIV, Part B)
│   └── Periodic Safety Update Report (PSUR) schedule
│
└── 8. Clinical Evaluation Report (MDR Annex XIV, Part A)
    ├── State of the art review
    ├── Clinical data from development studies
    ├── Equivalence analysis (if applicable)
    └── Clinical evaluation conclusion
```

### 3.4 Clinical Evaluation Pathway

#### 3.4.1 Clinical Evaluation Requirements (MDR Article 61, Annex XIV)

MDR requires a **Clinical Evaluation Report (CER)** demonstrating conformity with General Safety and Performance Requirements (GSPR) based on clinical data. For NeuroFusion-AD, the clinical evaluation will follow the **Performance Data from Clinical Investigations** pathway (Annex XIV, Part A, §6.1(c)) supplemented by literature review.

#### 3.4.2 Clinical Data Strategy

```
Clinical Evaluation Data Package
│
├── Arm 1: Internal Validation Study
│   ├── Dataset: Independent test cohort (n ≥ 500, demographically stratified)
│   ├── Outcomes: AUC, RMSE, C-index, subgroup analysis
│   ├── Reference standard: Expert clinical consensus + longitudinal follow-up
│   └── Study protocol: IEC 62366-1 compliant usability protocol
│
├── Arm 2: Prospective Real-World Validation
│   ├── Sites: ≥3 EU clinical sites (target: DE, NL, SE)
│   ├── Design: Observational, prospective, 12-month follow-up
│   ├── Primary endpoint: Concordance between NeuroFusion-AD prediction and clinical outcome
│   └── Ethics: EU CTR / applicable national IRB approvals
│
└── Arm 3: Published Literature Review
    ├── PICO framework: MCI patients, AI-based risk stratification, AD progression
    ├── Databases: PubMed, Embase, Cochrane
    ├── Evidence synthesis: Systematic review methodology
    └── Contribution: State-of-the-art and clinical gap analysis
```

#### 3.4.3 Post-Market Clinical Follow-Up (PMCF)

PMCF is mandatory for Class IIa devices under MDR Annex XIV, Part B. The PMCF plan will include:

- **Systematic literature review:** Annual updates to CER
- **Registry participation:** Target enrollment in European Dementia Registry or equivalent
- **User surveys:** Annual structured feedback from clinical users at certified sites
- **Real-world performance data collection:** Continuous performance monitoring feeding PSUR
- **PMCF Report frequency:** Annual, feeding into PSUR (per MDR Article 86)

---

## 4. IEC 62304 SOFTWARE SAFETY CLASSIFICATION AND COMPLIANCE PLAN

### 4.1 Software Safety Class Determination

#### 4.1.1 Classification Rationale: Class B

IEC 62304:2006+AMD1:2015 defines three software safety classes based on the potential severity of hazard arising from software failure:

| Class | Severity of Injury | Definition |
|---|---|---|
| A | No injury | Software failure cannot contribute to a hazardous situation |
| **B** | **Non-serious injury** | **Software failure could contribute to injury that is not life-threatening or result in permanent impairment** |
| C | Serious injury or death | Software failure could contribute to life-threatening injury or death |

**Class B Rationale for NeuroFusion-AD:**

The software's function is to provide **clinical decision support** — it generates a risk score, cognitive trajectory, and survival estimate, but does not autonomously execute any therapeutic action. The pathway from software failure to patient harm requires an intervening human decision by a qualified clinician.

Analysis of the worst-case failure modes:

```
Failure Mode Analysis (IEC 62304 Class Determination)

Scenario 1: False Negative (low risk predicted, high risk actual)
├── Software output: Low-risk classification
├── Clinician action: Reduced monitoring frequency
├── Patient consequence: Delayed treatment initiation
├── Severity: Moderate — NOT immediately life-threatening
└── Class B appropriate ✅

Scenario 2: False Positive (high risk predicted, low risk actual)
├── Software output: High-risk classification
├── Clinician action: Unnecessary workup or treatment
├── Patient consequence: Over-treatment burden, anxiety
├── Severity: Minor to moderate — NOT life-threatening
└── Class B appropriate ✅

Scenario 3: System Crash / Unavailability
├── Software output: None (system unavailable)
├── Clinician action: Reverts to standard clinical assessment
├── Patient consequence: No NeuroFusion-AD-assisted assessment
├── Severity: No direct harm; standard of care maintained
└── Class A/B — Class B conservative assignment ✅
```

**Conclusion:** NeuroFusion-AD software is assigned **IEC 62304 Safety Class B**. The device provides decision support; all therapeutic decisions are made by a licensed clinician. Software failure cannot directly cause death or serious irreversible injury without an independent clinical error of judgment. Class B imposes appropriate rigor without the disproportionate overhead of Class C.

> **Important Caveat:** This classification must be reviewed if device indications expand to include autonomous alert generation, medication dosing recommendations, or any output that can directly trigger a care pathway without clinician confirmation. Any such expansion would trigger re-evaluation to Class C.

### 4.2 IEC 62304 Class B Required Activities

#### 4.2.1 Activity Matrix

| IEC 62304 Clause | Activity | Class B Requirement | NeuroFusion-AD Implementation |
|---|---|---|---|
| §5.1 | Software development planning | Required | SDL plan with milestones, roles, tools |
| §5.2 | Software requirements analysis | Required | SRS document (functional + safety requirements) |
| §5.3 | Software architectural design | Required | Architecture document (see §MODEL ARCHITECTURE) |
| §5.4 | Software detailed design | Required | Detailed design specs per module |
| §5.5 | Software unit implementation | Required (no formal unit verification required for Class B) | Code review + static analysis |
| §5.6 | Software integration and testing | Required | Integration test plan and results |
| §5.7 | Software system testing | Required |