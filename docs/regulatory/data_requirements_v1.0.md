---
document_id: data-requirements
generated: 2026-02-26T22:55:20.935915
batch_id: msgbatch_01DTMbBbcyvTviGxwBhePxKr
status: DRAFT — requires human review before approval
---

# Data Requirements Document

## NeuroFusion-AD: Multimodal Graph Neural Network for Alzheimer's Disease Progression Prediction

---

| **Document ID** | DRD-001 |
|---|---|
| **Version** | 1.0 |
| **Status** | DRAFT — FOR INTERNAL REVIEW |
| **Date** | 2025-01-31 |
| **Author** | Regulatory Affairs Office |
| **Classification** | Confidential — Restricted Distribution |
| **Applicable Standards** | 21 CFR Part 11, HIPAA 45 CFR Parts 160/164, EU GDPR, ISO 13485, IEC 62304, FDA AI/ML-Based SaMD Action Plan |

---

> ⚠️ **SYNTHETIC DATA DISCLOSURE — PHASE 1**
> All development activities prior to ADNI/Bio-Hermes-001/DementiaBank data approval are conducted exclusively on **synthetically generated data**. Synthetic data does not contain real patient information and must not be used for final regulatory submission, clinical validation, or performance claims. This disclosure must be reproduced in all derived documents, model cards, and training reports until real-world data access is confirmed in writing.

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Referenced Documents](#2-referenced-documents)
3. [Data Sources](#3-data-sources)
4. [Data Schema Specification](#4-data-schema-specification)
5. [Data Quality Requirements](#5-data-quality-requirements)
6. [Data Preprocessing Specification](#6-data-preprocessing-specification)
7. [Phase 1 Synthetic Data Plan](#7-phase-1-synthetic-data-plan)
8. [Data Privacy and Security](#8-data-privacy-and-security)
9. [Traceability and Governance](#9-traceability-and-governance)
10. [Document History and Approval](#10-document-history-and-approval)

---

## 1. Purpose and Scope

### 1.1 Purpose

This Data Requirements Document (DRD) defines the authoritative specifications governing data acquisition, structure, quality, preprocessing, and privacy for all datasets used in the development, validation, and post-market monitoring of **NeuroFusion-AD** — a multimodal Graph Neural Network–based Clinical Decision Support (CDS) tool for Alzheimer's Disease (AD) progression prediction in patients with Mild Cognitive Impairment (MCI).

This document is a primary input artifact for the Technical File (EU MDR Article 10) and the Design History File (FDA 21 CFR Part 820.30). It governs all data lifecycle activities from initial acquisition through model training, validation, and deployment audit.

### 1.2 Scope

This document applies to:

- All training, validation, and test datasets used across all NeuroFusion-AD model versions
- All data modalities: fluid biomarkers, acoustic features, motor features, and clinical/demographic variables
- All personnel and systems handling NeuroFusion-AD data, including contracted data science vendors
- Phase 1 synthetic data and all subsequent real-world data cohorts

### 1.3 Out of Scope

- Neuroimaging data (MRI/PET) — reserved for future model versions; addressed in DRD-002
- Post-market surveillance data collection protocols — addressed in PMS-001
- Electronic Health Record (EHR) integration data contracts — addressed in IFU-003

### 1.4 Intended Use Alignment

Per the Intended Use Statement approved in RD-001:

> *NeuroFusion-AD is a Software as a Medical Device (SaMD) intended to aid clinicians in assessing Alzheimer's Disease progression risk in patients aged 50–90 diagnosed with Mild Cognitive Impairment (MCI). The device provides risk stratification scores, estimated cognitive decline trajectories, and time-to-progression estimates to support — not replace — clinical judgment.*

All data requirements in this document are derived from and traceable to this intended use statement, the target patient population (MCI, age 50–90), and the four active model input modalities.

---

## 2. Referenced Documents

| Document ID | Title | Version | Relationship |
|---|---|---|---|
| RD-001 | Intended Use and Indications for Use Statement | 1.0 | Parent requirement |
| SRS-001 | Software Requirements Specification | 1.0 | Derives data interface requirements |
| RISK-001 | Risk Management File (ISO 14971) | 1.0 | Informs data hazard analysis |
| VAL-001 | Clinical Validation Protocol | 0.9 DRAFT | Consumes datasets defined here |
| SEC-001 | Information Security Management Plan | 1.0 | Governs data security controls |
| IEC 62304 | Medical Device Software Lifecycle | 2006+AMD1:2015 | Normative reference |
| ISO 14971 | Application of Risk Management to MD | 2019 | Normative reference |
| FDA Guidance | Predetermined Change Control Plan for AI/ML | 2023 | Regulatory reference |
| EU MDR | Regulation (EU) 2017/745 | Annex I, II, XIV | Regulatory reference |
| ADNI Protocol | ADNI3 Study Protocol | Current | Data source reference |

---

## 3. Data Sources

### 3.1 Overview

NeuroFusion-AD training and validation relies on three primary external data sources, selected for their complementary coverage of the four input modalities, established de-identification practices, and suitability for MCI progression research.

| Source | Modalities Covered | Expected N (MCI) | Access Mechanism | Timeline |
|---|---|---|---|---|
| ADNI | Fluid biomarkers, Clinical/Demographic | ~1,200 | DUA via LONI Portal | 1–2 weeks |
| Bio-Hermes-001 | Plasma biomarkers (extended panel) | ~500 | DUA via study PI | 2–4 weeks |
| DementiaBank (Pitt Corpus) | Acoustic (speech) features | ~300 recordings | DUA via TalkBank | 1–2 weeks |

---

### 3.2 ADNI — Alzheimer's Disease Neuroimaging Initiative

#### 3.2.1 Dataset Description

The Alzheimer's Disease Neuroimaging Initiative (ADNI) is a longitudinal multi-site observational study initiated in 2004. The ADNI3 phase (2016–present) provides the most clinically relevant cohort for NeuroFusion-AD, including plasma and CSF biomarkers, neuropsychological assessments, and longitudinal follow-up. ADNI data is managed by the Laboratory of Neuro Imaging (LONI) at the University of Southern California.

**Anticipated cohort selection criteria applied post-access:**

- Diagnosis: MCI (early MCI or late MCI) at baseline
- Age: 50–90 years
- Minimum follow-up: ≥2 time points separated by ≥6 months
- Availability of: MMSE, pTau-217 or surrogate (pTau-181), Aβ42/40 ratio, NfL, APOE genotype

**Expected sample:** ~1,200 MCI patients meeting inclusion criteria (based on published ADNI3 cohort demographics).

#### 3.2.2 Access Process

```
Step 1: Registration
  └─ URL: https://ida.loni.usc.edu
  └─ Action: Create institutional account with valid .edu or institutional email
  └─ Timeline: Immediate

Step 2: Data Use Agreement (DUA)
  └─ Action: Submit signed DUA on institutional letterhead
  └─ Signatory: Institutional official (IRB office or department head)
  └─ Required fields: Institution name, PI name, study purpose, data security attestation
  └─ Timeline: 1–5 business days for ADNI review

Step 3: Data Download
  └─ Action: Select ADNI3 cohort; apply inclusion filters via LONI Image and Data Archive
  └─ File format: CSV (tabular biomarkers), XML (metadata)
  └─ Download method: LONI DTA secure download client (authenticated HTTPS)

Step 4: Internal Registration
  └─ Action: Log dataset receipt in Data Asset Register (DAR-001)
  └─ SHA-256 hash of downloaded files to be recorded for data integrity verification
```

#### 3.2.3 Permitted Uses and Restrictions

Per the ADNI DUA (current version):

- ✅ Algorithm development and training
- ✅ Academic publication of aggregate results
- ❌ Commercial redistribution of raw data
- ❌ Re-identification attempts
- ❌ Linkage with external datasets without separate approval

**Regulatory Note:** ADNI data use must be disclosed in FDA Pre-Submission and EU MDR Technical File. The DUA and executed institutional agreement shall be retained in the Design History File.

---

### 3.3 Bio-Hermes-001

#### 3.3.1 Dataset Description

Bio-Hermes-001 is a prospective observational study (NCT04937972) specifically designed to evaluate plasma-based biomarkers for AD detection, including pTau-217 (Lilly Quanterix Simoa assay), Aβ42/40, NfL, and GFAP. It provides a community-based (non-academic medical center) cohort that complements ADNI's predominantly academic recruitment, improving demographic generalizability.

**Key distinguishing features:**

- pTau-217 measured on the same Quanterix Simoa platform targeted by NeuroFusion-AD
- Community-dwelling participants with broader socioeconomic and ethnic diversity than ADNI
- Includes plasma biomarkers at ≥2 time points for a subset of participants

**Expected sample:** ~500 patients with plasma biomarker panels meeting inclusion criteria.

#### 3.3.2 Access Process

```
Step 1: Investigator Contact
  └─ Action: Contact study PI (published in NCT04937972 record)
  └─ Purpose: Express data sharing interest; request Data Sharing Agreement template
  └─ Timeline: 1–2 weeks for initial response

Step 2: Data Sharing Agreement (DSA)
  └─ Parties: NeuroFusion-AD institution + Bio-Hermes-001 study site(s)
  └─ Legal review required: Yes — institutional legal counsel review mandated
  └─ Key provisions to negotiate:
       - Permitted analysis scope
       - Publication rights and embargo periods
       - IP ownership of derived models
       - Data retention and destruction schedule
  └─ Timeline: 2–4 weeks

Step 3: IRB/Ethics Coordination
  └─ Action: Confirm NeuroFusion-AD institution's IRB covers secondary analysis
  └─ If not: Submit IRB amendment or reliance agreement
  └─ Timeline: Variable (2–8 weeks if amendment required)

Step 4: Secure Transfer
  └─ Method: SFTP with PGP encryption OR institution-provided secure data enclave
  └─ Action: Record receipt in DAR-001; verify SHA-256 integrity hash
```

#### 3.3.3 Data Unique Contribution

| Feature | ADNI | Bio-Hermes-001 |
|---|---|---|
| pTau-217 (Simoa) | Surrogate (pTau-181) | ✅ Direct measurement |
| Community-based recruitment | Limited | ✅ Primary design feature |
| Ethnic diversity | Predominantly White | Broader representation |
| GFAP plasma | Limited | ✅ Available |

---

### 3.4 DementiaBank — Pitt Corpus

#### 3.4.1 Dataset Description

DementiaBank, hosted by TalkBank (Carnegie Mellon University), provides the Pitt Corpus — a collection of speech samples from participants performing the Boston Diagnostic Aphasia Examination Cookie Theft picture description task. Audio recordings include participants diagnosed with AD, MCI, and cognitively normal controls, with longitudinal recordings for a subset.

NeuroFusion-AD uses this corpus to train the acoustic modality encoder, targeting 15 acoustic feature dimensions extracted via openSMILE (speech rate, pause frequency, pitch variance, lexical diversity, disfluency markers, and related features).

**Expected sample:** ~300 audio recordings from MCI participants (at least 1 recording per participant; longitudinal pairs available for ~40% of cohort).

#### 3.4.2 Access Process

```
Step 1: TalkBank Registration
  └─ URL: https://dementia.talkbank.org
  └─ Action: Register as researcher; agree to TalkBank terms of use
  └─ Timeline: 1–3 business days

Step 2: Data Use Agreement
  └─ Action: Submit signed DUA via TalkBank researcher portal
  └─ Required: Institutional affiliation, research purpose statement, IRB documentation
  └─ Timeline: 1–5 business days

Step 3: Ethics Confirmation
  └─ Confirm: Secondary analysis of de-identified audio is covered under existing IRB
  └─ Note: Audio recordings require heightened privacy consideration
         (voice is a biometric identifier under HIPAA/GDPR)
  └─ Action: Confirm voice data handling procedures with privacy officer

Step 4: Feature Extraction (On-Access)
  └─ Action: Extract acoustic features using openSMILE 3.0 IMMEDIATELY upon access
  └─ Store: Feature vectors only (float32 arrays); delete raw audio from active systems
  └─ Rationale: Minimizes biometric data exposure; audio reconstruction impossible
         from feature vectors
  └─ Timeline: Complete within 5 business days of access
```

#### 3.4.3 Acoustic Data Special Handling Requirements

> ⚠️ **BIOMETRIC DATA WARNING:** Voice recordings constitute biometric data under HIPAA, GDPR Article 9, and several U.S. state privacy laws (CCPA, BIPA). Raw audio must never be stored in the NeuroFusion-AD production database. Only extracted numerical feature vectors are permitted in the training pipeline and inference system.

---

## 4. Data Schema Specification

### 4.1 Schema Design Principles

All fields are mapped to **HL7 FHIR R4** resources to ensure interoperability with EHR systems and to support future integration with clinical workflows. FHIR mappings are provided at the element level. Where no direct FHIR element exists, FHIR extensions (indicated with `ext:`) are specified.

**Null representation:** `NULL` in PostgreSQL; `NaN` in PyTorch tensors. Empty strings are not permitted.
**Timestamp format:** ISO 8601 (UTC): `YYYY-MM-DDTHH:MM:SSZ`
**Version tracking:** All schema changes require DRD version increment and SRS-001 update.

---

### 4.2 Fluid Biomarker Schema

**Source tables:** ADNI (CSF/plasma panels), Bio-Hermes-001 (plasma panels)
**FHIR Resource Base:** `Observation` (profile: `us-core-observation-lab`)

| Field Name | Type | Valid Range | Units | Required / Optional | FHIR Mapping |
|---|---|---|---|---|---|
| `patient_id` | VARCHAR(36) | UUID v4 format | — | **Required** | `Observation.subject.reference` |
| `visit_id` | VARCHAR(36) | UUID v4 format | — | **Required** | `Observation.encounter.reference` |
| `visit_date` | DATE | 2000-01-01 to present | ISO 8601 | **Required** | `Observation.effectiveDateTime` |
| `visit_number` | INTEGER | 1–20 | — | **Required** | `Observation.component[visit_number]` |
| `ptau_217` | FLOAT | 0.1–100.0 | pg/mL | **Required** | `Observation.valueQuantity` (LOINC: 98979-8) |
| `ptau_217_assay` | VARCHAR(50) | Enum: {Simoa_Quanterix, MSD, Elecsys} | — | **Required** | `Observation.method.coding` |
| `ptau_217_cv` | FLOAT | 0.0–25.0 | % CV | Optional | `Observation.component[cv]` |
| `abeta42` | FLOAT | 100–3000 | pg/mL | Optional† | `Observation.valueQuantity` (LOINC: 42911-7) |
| `abeta40` | FLOAT | 1000–30000 | pg/mL | Optional† | `Observation.valueQuantity` (LOINC: ext:abeta40) |
| `abeta42_40_ratio` | FLOAT | 0.01–0.30 | ratio | **Required** | `Observation.valueQuantity` (LOINC: 98978-0) |
| `nfl` | FLOAT | 5.0–200.0 | pg/mL | **Required** | `Observation.valueQuantity` (LOINC: 99750-2) |
| `nfl_assay` | VARCHAR(50) | Enum: {Simoa_Quanterix, Lumipulse} | — | **Required** | `Observation.method.coding` |
| `gfap` | FLOAT | 20.0–2000.0 | pg/mL | Optional | `Observation.valueQuantity` (LOINC: ext:gfap-plasma) |
| `apoe_genotype` | VARCHAR(5) | Enum: {e2e2, e2e3, e2e4, e3e3, e3e4, e4e4} | — | **Required** | `Observation.valueCodeableConcept` (LOINC: 69551-0) |
| `apoe_e4_count` | INTEGER | 0, 1, 2 | allele count | **Required** | `Observation.component[e4_count]` |
| `specimen_type` | VARCHAR(20) | Enum: {plasma, CSF} | — | **Required** | `Observation.specimen.type` |
| `lab_site_id` | VARCHAR(36) | UUID v4 format | — | **Required** | `Observation.performer.reference` |
| `fasting_status` | BOOLEAN | TRUE/FALSE | — | Optional | `Observation.component[fasting]` |

† `abeta42` and `abeta40` individually optional; `abeta42_40_ratio` is required. Ratio may be provided directly or computed from individual values. If computed, `abeta42` and `abeta40` must both be present.

**Hard Validation Constraints (enforced at ingestion):**

```python
BIOMARKER_HARD_LIMITS = {
    "ptau_217":        {"min": 0.1,  "max": 100.0,  "unit": "pg/mL"},
    "abeta42_40_ratio":{"min": 0.01, "max": 0.30,   "unit": "ratio"},
    "nfl":             {"min": 5.0,  "max": 200.0,  "unit": "pg/mL"},
    "gfap":            {"min": 20.0, "max": 2000.0, "unit": "pg/mL"},
}
# Values outside these ranges trigger HARD REJECT — record excluded from dataset
# Values flagged as outliers (|z-score| > 4.0) trigger SOFT FLAG — manual review required
```

---

### 4.3 Clinical and Demographic Schema

**Source tables:** ADNI (PTDEMOG, MMSE tables), Bio-Hermes-001 (clinical assessments)
**FHIR Resource Base:** `Patient`, `Observation`, `Condition`

| Field Name | Type | Valid Range | Units | Required / Optional | FHIR Mapping |
|---|---|---|---|---|---|
| `patient_id` | VARCHAR(36) | UUID v4 format | — | **Required** | `Patient.id` |
| `age_at_visit` | FLOAT | 50.0–90.0 | years | **Required** | `Patient.birthDate` (derived) |
| `sex` | VARCHAR(10) | Enum: {Male, Female, Intersex} | — | **Required** | `Patient.gender` |
| `race` | VARCHAR(50) | OMOP CDM race vocabulary | — | Optional | `Patient.extension[us-core-race]` |
| `ethnicity` | VARCHAR(50) | OMOP CDM ethnicity vocabulary | — | Optional | `Patient.extension[us-core-ethnicity]` |
| `education_years` | INTEGER | 0–30 | years | **Required** | `Observation.valueQuantity` (ext:education-years) |
| `diagnosis_at_visit` | VARCHAR(20) | Enum: {CN, SMC, EMCI, LMCI, AD} | — | **Required** | `Condition.code` (SNOMED: 386807006) |
| `mmse_total` | INTEGER | 0–30 | score | **Required** | `Observation.valueInteger` (LOINC: 72107-6) |
| `mmse_date` | DATE | 2000-01-01 to present | ISO 8601 | **Required** | `Observation.effectiveDateTime` |
| `cdr_global` | FLOAT | Enum: {0, 0.5, 1, 2, 3} | score | **Required** | `Observation.valueQuantity` (LOINC: 52491-8) |
| `cdr_sum_boxes` | FLOAT | 0.0–18.0 | score | Optional | `Observation.valueQuantity` (LOINC: ext:cdr-sob) |
| `moca_total` | INTEGER | 0–30 | score | Optional | `Observation.valueInteger` (LOINC: 72172-0) |
| `adas_cog_13` | FLOAT | 0.0–85.0 | score | Optional | `Observation.valueQuantity` (LOINC: ext:adas-cog13) |
| `faq_total` | INTEGER | 0–30 | score | Optional | `Observation.valueInteger` (LOINC: ext:faq) |
| `depression_gds` | INTEGER | 0–15 | score | Optional | `Observation.valueInteger` (LOINC: 48543-6) |
| `medication_cholinesterase` | BOOLEAN | TRUE/FALSE | — | **Required** | `MedicationStatement.status` (ATC: N06DA) |
| `medication_memantine` | BOOLEAN | TRUE/FALSE | — | **Required** | `MedicationStatement.status` (ATC: N06DX01) |
| `years_since_mci_diagnosis` | FLOAT | 0.0–30.0 | years | **Required** | `Condition.onsetDateTime` (derived) |
| `family_history_ad` | BOOLEAN | TRUE/FALSE | — | Optional | `FamilyMemberHistory.condition` |
| `progression_label` | VARCHAR(20) | Enum: {Stable_MCI, Progressive_MCI, Converted_AD} | — | **Required** | `Observation.valueCodeableConcept` (ext:ad-progression-label) |
| `progression_confirmed_date` | DATE | 2000-01-01 to present | ISO 8601 | **Required** | `Observation.effectiveDateTime` |
| `time_to_conversion_months` | FLOAT | 0.0–120.0 | months | Optional‡ | `Observation.valueQuantity` (ext:time-to-conversion) |
| `censored` | BOOLEAN | TRUE/FALSE | — | **Required** | `Observation.component[censored]` (survival analysis) |

‡ Required for survival analysis output; if not available, patient is excluded from survival model training set (included in classification/regression subsets).

---

### 4.4 Acoustic Feature Schema

**Source:** DementiaBank Pitt Corpus (features extracted via openSMILE 3.0)
**FHIR Resource Base:** `Observation` (ext: `speech-analysis-panel`)
**Note:** All 15 features are FLOAT32; extracted from standardized Cookie Theft picture description task recordings (90-second administration window).

| Field Name | Type | Valid Range | Units | Required / Optional | FHIR Mapping |
|---|---|---|---|---|---|
| `patient_id` | VARCHAR(36) | UUID v4 format | — | **Required** | `Observation.subject.reference` |
| `recording_id` | VARCHAR(36) | UUID v4 format | — | **Required** | `Observation.identifier` |
| `recording_date` | DATE | 2000-01-01 to present | ISO 8601 | **Required** | `Observation.effectiveDateTime` |
| `task_type` | VARCHAR(50) | Enum: {Cookie_Theft, Free_Recall, Fluency_Animals} | — | **Required** | `Observation.component[task]` |
| `recording_duration_sec` | FLOAT | 10.0–300.0 | seconds | **Required** | `Observation.component[duration]` |
| `acou_f0_mean` | FLOAT32 | 50.0–400.0 | Hz | **Required** | `Observation.component[f0-mean]` |
| `acou_f0_std` | FLOAT32 | 0.0–100.0 | Hz | **Required** | `Observation.component[f0-std]` |
| `acou_speech_rate` | FLOAT32 | 0.0–8.0 | syllables/sec | **Required** | `Observation.component[speech-rate]` |
| `acou_pause_rate` | FLOAT32 | 0.0–5.0 | pauses/min | **Required** | `Observation.component[pause-rate]` |
| `acou_pause_duration_mean` | FLOAT32 | 0.0–10.0 | seconds | **Required** | `Observation.component[pause-dur-mean]` |
| `acou_pause_duration_std` | FLOAT32 | 0.0–5.0 | seconds | **Required** | `Observation.component[pause-dur-std]` |
| `acou_jitter` | FLOAT32 | 0.0–5.0 | % | **Required** | `Observation.component[jitter]` |
| `acou_shimmer` | FLOAT32 | 0.0–15.0 | % | **Required** | `Observation.component[shimmer]` |
| `acou_hnr` | FLOAT32 | -10.0–40.0 | dB | **Required** | `Observation.component[hnr]` |
| `acou_type_token_ratio` | FLOAT32 | 0.0–1.0 | ratio | **Required** | `Observation.component[ttr]` |
| `acou_mlu` | FLOAT32 | 0.0–30.0 | words/utterance | **Required** | `Observation.component[mlu]` |
| `acou_disfluency_rate` | FLOAT32 | 0.0–10.0 | events/min | **Required** | `Observation.component[disfluency]` |
| `acou_content_unit_count` | FLOAT32 | 0.0–50.0 | count | **Required** | `Observation.component[content-units]` |
| `acou_information_units` | FLOAT32 | 0.0–1.0 | ratio | **Required** | `Observation.component[info-units]` |
| `acou_silence_fraction` | FLOAT32 | 0.0–1.0 | proportion | **Required** | `Observation.component[silence-fraction]` |
| `opensmile_version` | VARCHAR(10) | Enum: {3.0.0, 3.0.1, 3.0.2} | — | **Required** | `Observation.device.reference` |
| `feature_extraction_date` | DATE | 2000-01-01 to present | ISO 8601 | **Required** | `Observation.issued` |

---

### 4.5 Motor Feature Schema

**Source:** Future wearable device integration (Phase 2); stub schema included for IEC 62304 completeness.
**Status:** ⚠️ PLACEHOLDER — Motor data collection protocol under development. See MOT-PROTO-001.
**FHIR Resource Base:** `Observation` (ext: `motor-assessment-panel`)

| Field Name | Type | Valid Range | Units | Required / Optional | FHIR Mapping |
|---|---|---|---|---|---|
| `patient_id` | VARCHAR(36) | UUID v4 format | — | **Required** | `Observation.subject.reference` |
| `assessment_id` | VARCHAR(36) | UUID v4 format | — | **Required** | `Observation.identifier` |
| `assessment_date` | DATE | 2000-01-01 to present | ISO 8601 | **Required** | `Observation.effectiveDateTime` |
| `device_type` | VARCHAR(50) | Enum: {Actigraph_GT9X, GENEActiv, AppleWatch_S9} | — | **Required** | `Device.type` |
| `motor_gait_speed` | FLOAT32 | 0.0–3.0 | m/s | **Required** | `Observation.component[gait-speed]` |
| `motor_stride_length` | FLOAT32 | 0.0–2.0 | m | **Required** | `Observation.component[stride-length]` |
| `motor_cadence` | FLOAT32 | 0.0–200.0 | steps/min | **Required** | `Observation.component[cadence]` |
| `motor_step_asymmetry` | FLOAT32 | 0.0–100.0 | % | **Required** | `Observation.component[step-asymmetry]` |
| `motor_double_support_time` | FLOAT32 | 0.0–60.0 | % gait cycle | **Required** | `Observation.component[double-support]` |
| `motor_gait_variability` | FLOAT32 | 0.0–50.0 | CV % | **Required** | `Observation.component[gait-variability]` |
| `motor_turn_duration` | FLOAT32 | 0.0–5.0 | seconds | **Required** | `Observation.component[turn-duration]` |
| `motor_turn_angle` | FLOAT32 | 0.0–360.0 | degrees | **Required** | `Observation.component[turn-angle]` |
| `motor_grip_strength_dom` | FLOAT32 | 0.0–100.0 | kg | Optional | `Observation.component[grip-dominant]` |
| `motor_grip_strength_nondom` | FLOAT32 | 0.0–100.0 | kg | Optional | `Observation.component[grip-nondominant]` |
| `motor_tremor_amplitude` | FLOAT32 | 0.0–10.0 | mm | Optional | `Observation.component[tremor-amplitude]` |
| `motor_tremor_frequency` | FLOAT32 | 0.0–20.0 | Hz | Optional | `Observation.component[tremor-frequency]` |
| `motor_tug_time` | FLOAT32 | 0.0–60.0 | seconds | **Required** | `Observation.component[tug-time]` |
| `motor_balance_sway` | FLOAT32 | 0.0–100.0 | mm | Optional | `Observation.component[balance-sway]` |
| `motor_reaction_time` | FLOAT32 | 100.0–2000.0 | ms | Optional | `Observation.component[reaction-time]` |
| `motor_tapping_frequency` | FLOAT32