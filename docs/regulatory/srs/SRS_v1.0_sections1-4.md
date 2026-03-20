# SOFTWARE REQUIREMENTS SPECIFICATION
## NeuroFusion-AD: Multimodal Graph Neural Network for Alzheimer's Disease Progression Prediction

---

| **Document ID** | SRS-001 |
|---|---|
| **Version** | 1.0 |
| **Status** | Final |
| **Date** | 2025-01-01 |
| **Classification** | Confidential — Regulatory Submission Document |
| **Prepared By** | Regulatory Affairs Office |
| **Reviewed By** | *[Software Quality Assurance — Pending]* |
| **Approved By** | *[Chief Medical Officer — Pending]* |

---

### Document Control

| Version | Date | Author | Description of Changes |
|---|---|---|---|
| 0.1 | 2024-10-15 | Regulatory Affairs | Initial draft — Sections 1–2 |
| 0.2 | 2024-11-20 | Regulatory Affairs | Added Sections 3–4 skeleton |
| 1.0 | 2025-01-01 | Regulatory Affairs | First complete release for QA review |

---

### Compliance Notices

This document is prepared in accordance with:

- **IEC 62304:2006+AMD1:2015** — Software life cycle processes (Class B designation)
- **IEC 62304 Section 5.2** — Software requirements analysis
- **ISO 14971:2019** — Application of risk management to medical devices
- **FDA Guidance: Software as a Medical Device (SaMD)** — De Novo Request per 21 CFR 513(f)(2)
- **EU MDR 2017/745 Annex I** — General Safety and Performance Requirements (GSPR)
- **FDA 21 CFR Part 11** — Electronic Records and Electronic Signatures
- **IEC 82304-1:2016** — Health software — General requirements for product safety
- **ISO/IEC 25010:2011** — Software product quality characteristics

---

## TABLE OF CONTENTS

1. [Introduction](#1-introduction)
   - 1.1 Purpose
   - 1.2 Scope
   - 1.3 Definitions
   - 1.4 Abbreviations and Acronyms
   - 1.5 Document Overview
   - 1.6 Relationship to Other Documents

2. [Overall Description](#2-overall-description)
   - 2.1 Product Perspective
   - 2.2 Product Functions
   - 2.3 User Classes and Characteristics
   - 2.4 Operating Environment and Constraints
   - 2.5 Design and Implementation Constraints
   - 2.6 Assumptions and Dependencies

3. [Functional Requirements — Data Ingestion (FRI-001–FRI-020)](#3-functional-requirements--data-ingestion)

4. [Functional Requirements — Preprocessing (FRP-001–FRP-015)](#4-functional-requirements--preprocessing)

---

## 1. Introduction

### 1.1 Purpose

This Software Requirements Specification (SRS) defines the functional, performance, safety, and regulatory requirements for **NeuroFusion-AD**, a multimodal Graph Neural Network (GNN)-based Software as a Medical Device (SaMD) intended to aid clinicians in the assessment of Alzheimer's Disease (AD) progression risk in patients diagnosed with Mild Cognitive Impairment (MCI).

This document serves as the authoritative requirements baseline for:

1. **FDA De Novo clearance** under 21 CFR 513(f)(2), with Prenosis Sepsis ImmunoScore (DEN200080) as the identified predicate device for the multimodal AI/ML-based clinical decision support framework;
2. **EU MDR 2017/745 Class IIa certification** under Rule 11 (software intended to provide information used to make clinical decisions) and Rule 10 (active devices for diagnosis), subject to Notified Body audit;
3. **IEC 62304 Class B software development lifecycle** compliance documentation, establishing the traceability link between requirements, architecture, implementation, verification, and validation activities;
4. **ISO 14971:2019 risk management** integration, providing input hazard identification data for the Risk Management File (RMF-001).

This SRS governs **Sections 1 through 4** of the complete specification. Subsequent sections covering non-functional requirements, security requirements, interface requirements, and verification/validation requirements are documented in SRS-002 through SRS-005 respectively.

All requirements stated herein are mandatory unless explicitly designated as *[OPTIONAL]* or *[FUTURE RELEASE]*. Each requirement is uniquely identified, testable by a defined verification method, and traceable to upstream design inputs and downstream test cases in the Master Verification and Validation Plan (MVVP-001).

### 1.2 Scope

#### 1.2.1 System Identification

**System Name:** NeuroFusion-AD
**System Identifier:** NFA-SYS-001
**Software Safety Classification:** IEC 62304 Class B (software failure can result in unacceptable risk of injury; no serious injury pathway identified that is not mitigated by independent clinical judgment)

#### 1.2.2 Intended Use Statement

NeuroFusion-AD is a prescription-use clinical decision support (CDS) software intended to aid qualified healthcare professionals in assessing the risk of Alzheimer's Disease progression in adult patients (ages 50–90) with a confirmed diagnosis of Mild Cognitive Impairment (MCI). The device integrates and analyzes four input modality streams — fluid biomarkers, acoustic speech features, motor function metrics, and clinical/demographic data — to generate a multimodal risk score, a projected MMSE trajectory, and an event-time survival estimate for conversion from MCI to AD dementia.

**NeuroFusion-AD is intended to augment, not replace, clinical judgment.** Output scores must be interpreted by a licensed clinician in the context of complete patient history, physical examination, and applicable clinical guidelines. The device does not autonomously diagnose Alzheimer's Disease.

#### 1.2.3 Indications for Use

NeuroFusion-AD is indicated for use in outpatient neurology, geriatric psychiatry, and memory disorder clinic settings where:

- The patient has received a formal MCI diagnosis per NIA-AA 2011 or equivalent diagnostic criteria;
- The patient is between 50 and 90 years of age;
- At least one fluid biomarker (pTau-217, Aβ42/40, or NfL) measurement is available from a CLIA-certified laboratory;
- Acoustic and/or motor assessments have been acquired using validated, integrated data collection protocols compatible with NeuroFusion-AD's input specifications.

#### 1.2.4 Contraindications and Out-of-Scope Use

The following use cases are explicitly **out of scope** and unsupported:

- Diagnosis of Alzheimer's Disease or other dementias as a standalone assessment;
- Use in patients under age 50 or over age 90;
- Screening of cognitively normal individuals;
- Use in emergency, intensive care, or inpatient acute care settings without specialist oversight;
- Genetic counseling or hereditary risk determination;
- Pediatric applications;
- Automated treatment selection or medication management.

#### 1.2.5 Software Items in Scope

This SRS governs all software items within the NeuroFusion-AD system boundary, including:

| Software Item | Identifier | IEC 62304 Class |
|---|---|---|
| Data Ingestion Service | NFA-SW-001 | B |
| Preprocessing Pipeline | NFA-SW-002 | B |
| Fluid Biomarker Encoder | NFA-SW-003 | B |
| Acoustic Feature Encoder | NFA-SW-004 | B |
| Motor Feature Encoder | NFA-SW-005 | B |
| Clinical/Demographic Encoder | NFA-SW-006 | B |
| Cross-Modal Attention Module | NFA-SW-007 | B |
| GraphSAGE GNN Module | NFA-SW-008 | B |
| Multi-Task Output Head | NFA-SW-009 | B |
| FastAPI REST Interface | NFA-SW-010 | B |
| Result Persistence Service | NFA-SW-011 | B |
| Audit Trail Manager | NFA-SW-012 | B |

### 1.3 Definitions

The following definitions apply throughout this document and all subordinate technical documents in the NeuroFusion-AD Design History File (DHF):

| Term | Definition |
|---|---|
| **Alzheimer's Disease (AD)** | A progressive neurodegenerative disorder characterized by amyloid plaques, neurofibrillary tangles, synaptic loss, and cognitive decline, culminating in dementia. Diagnosis per NIA-AA 2018 research framework. |
| **Mild Cognitive Impairment (MCI)** | A clinical syndrome defined by subjective and objective cognitive decline beyond expected aging without functional impairment in daily activities, representing an at-risk state for dementia conversion. |
| **Software as a Medical Device (SaMD)** | Software intended to be used for one or more medical purposes that perform these purposes without being part of a hardware medical device, per IMDRF/SaMD N10:2013. |
| **Clinical Decision Support (CDS)** | Software that provides clinician-facing analytical information and patient-specific recommendations to enhance clinical decision-making, without autonomous clinical action. |
| **Multimodal Input** | Patient data derived from two or more distinct physiological or clinical measurement domains (e.g., blood biomarkers, speech acoustics, movement kinematics, demographics). |
| **Graph Neural Network (GNN)** | A class of deep learning architectures that operates on graph-structured data, capturing relational dependencies between nodes (patient features or clinical entities) via message-passing algorithms. |
| **GraphSAGE** | Graph Sample and Aggregation; a GNN variant that generates node embeddings by sampling and aggregating features from local neighborhoods, enabling inductive generalization to unseen nodes. |
| **Cross-Modal Attention** | A transformer-derived attention mechanism that computes inter-modal relevance weights between latent representations of distinct input modalities, producing a fused joint embedding. |
| **FHIR** | Fast Healthcare Interoperability Resources (HL7 FHIR R4); a standard for healthcare data exchange, used herein for structured biomarker and clinical observation ingestion. |
| **Observation Resource** | An HL7 FHIR R4 resource type representing measurements, simple assertions, and calculated values in clinical contexts. |
| **pTau-217** | Phosphorylated tau protein at threonine-217; a plasma or CSF biomarker with high specificity for Alzheimer's pathology, quantified in picograms per milliliter (pg/mL). |
| **Aβ42/40** | Amyloid-beta 42 to 40 peptide ratio; a plasma or CSF biomarker reflecting amyloid plaque burden; lower ratios indicate greater amyloid pathology. |
| **NfL** | Neurofilament light chain; a plasma or CSF biomarker of neuroaxonal injury and neurodegeneration, quantified in pg/mL. |
| **MMSE** | Mini-Mental State Examination; a 30-point standardized neuropsychological screening instrument assessing orientation, memory, attention, language, and visuospatial function. |
| **APOE ε4** | The ε4 allele of Apolipoprotein E; the strongest known genetic risk factor for late-onset AD, used as a binary or additive encoded covariate in this system. |
| **Z-score Normalization** | A statistical transformation mapping a raw value *x* to *(x − μ) / σ*, where *μ* and *σ* are the population mean and standard deviation derived from the training dataset. |
| **Median Imputation** | A missing data handling strategy replacing absent values with the feature-wise median computed from the training dataset, stratified by applicable cohort strata. |
| **C-index** | Concordance index (Harrell's C-statistic); a measure of discriminative ability for survival models ranging from 0.5 (random) to 1.0 (perfect discrimination). |
| **AUC** | Area Under the Receiver Operating Characteristic Curve; a threshold-independent performance metric for binary classification ranging from 0.5 to 1.0. |
| **RMSE** | Root Mean Square Error; a measure of regression accuracy representing the square root of the mean of squared prediction errors. |
| **Input Validation** | The process of verifying that received data values are present, correctly formatted, and within clinically and technically acceptable ranges prior to model inference. |
| **Hardcoded Constraint** | A validation boundary defined in software that cannot be overridden at runtime without a change-controlled software update. |
| **Inference Latency** | The elapsed time from receipt of a complete, validated inference request to delivery of the model output response. |
| **Audit Trail** | An immutable, timestamped chronological record of system events, user actions, data access, and configuration changes, maintained per 21 CFR Part 11 requirements. |
| **Design History File (DHF)** | The compilation of records that describes the design history of a finished device, per 21 CFR 820.30(j). |
| **Risk Management File (RMF)** | The set of records and documents produced by the risk management process, per ISO 14971:2019 Section 4.5. |
| **SOUP** | Software of Unknown Provenance; pre-existing software components not developed under the project's quality management system, requiring documented evaluation per IEC 62304 Section 8. |
| **Notified Body** | A conformity assessment organization designated by an EU Member State under MDR 2017/745 to perform third-party review for Class IIa+ devices. |

### 1.4 Abbreviations and Acronyms

| Abbreviation | Expansion |
|---|---|
| AD | Alzheimer's Disease |
| API | Application Programming Interface |
| APOE | Apolipoprotein E |
| AUC | Area Under the ROC Curve |
| Aβ42/40 | Amyloid-beta 42/40 Ratio |
| CDS | Clinical Decision Support |
| CFR | Code of Federal Regulations |
| CI | Confidence Interval |
| CLIA | Clinical Laboratory Improvement Amendments |
| CSF | Cerebrospinal Fluid |
| DHF | Design History File |
| DICOM | Digital Imaging and Communications in Medicine |
| EHR | Electronic Health Record |
| EU | European Union |
| FDA | U.S. Food and Drug Administration |
| FHIR | Fast Healthcare Interoperability Resources |
| GNN | Graph Neural Network |
| GSPR | General Safety and Performance Requirements |
| GUI | Graphical User Interface |
| HL7 | Health Level Seven International |
| IEC | International Electrotechnical Commission |
| ISO | International Organization for Standardization |
| JSON | JavaScript Object Notation |
| JWT | JSON Web Token |
| LOINC | Logical Observation Identifiers Names and Codes |
| MCI | Mild Cognitive Impairment |
| MDR | Medical Device Regulation (EU 2017/745) |
| MMSE | Mini-Mental State Examination |
| MRI | Magnetic Resonance Imaging |
| MVVP | Master Verification and Validation Plan |
| NfL | Neurofilament Light Chain |
| NIA-AA | National Institute on Aging — Alzheimer's Association |
| PII | Personally Identifiable Information |
| PHI | Protected Health Information |
| pTau-217 | Phosphorylated Tau at Threonine-217 |
| QA | Quality Assurance |
| QMS | Quality Management System |
| REST | Representational State Transfer |
| RMF | Risk Management File |
| RMSE | Root Mean Square Error |
| ROC | Receiver Operating Characteristic |
| SaMD | Software as a Medical Device |
| SNOMED CT | Systematized Nomenclature of Medicine — Clinical Terms |
| SOUP | Software of Unknown Provenance |
| SRS | Software Requirements Specification |
| TLS | Transport Layer Security |
| UUID | Universally Unique Identifier |

### 1.5 Document Overview

This SRS is organized into the following major sections:

| Section | Title | Content Summary |
|---|---|---|
| 1 | Introduction | Purpose, scope, definitions, and abbreviations establishing the regulatory and technical context |
| 2 | Overall Description | System architecture context, core functional summary, user classes, operating environment, regulatory constraints, and key assumptions |
| 3 | Functional Requirements — Data Ingestion (FRI-001–FRI-020) | Detailed, testable requirements for FHIR-based data ingestion, biomarker parsing, input validation bounds enforcement, and error handling |
| 4 | Functional Requirements — Preprocessing (FRP-001–FRP-015) | Detailed, testable requirements for normalization, missing data imputation, categorical encoding, and preprocessing audit logging |
| 5 *(future)* | Functional Requirements — Model Inference | Encoder, attention, GNN, and multi-task output head requirements |
| 6 *(future)* | Functional Requirements — Output and Reporting | Score presentation, confidence intervals, explainability outputs |
| 7 *(future)* | Non-Functional Requirements | Performance, availability, security, scalability |
| 8 *(future)* | Interface Requirements | API contracts, EHR integration, external services |
| 9 *(future)* | Verification and Validation Requirements | Test strategy traceability matrix |

### 1.6 Relationship to Other Documents

This document is part of the NeuroFusion-AD Design History File and shall be read in conjunction with:

| Document ID | Title | Relationship |
|---|---|---|
| SDD-001 | Software Design Description | Downstream — architecture constrained by this SRS |
| RMF-001 | Risk Management File | Bidirectional — hazards traced to requirements |
| MVVP-001 | Master Verification and Validation Plan | Downstream — test cases trace to requirements herein |
| IFU-001 | Instructions for Use | Downstream — constrained by intended use in §1.2 |
| TRAIN-001 | Model Training and Validation Report | Upstream — performance benchmarks inform §3–4 constraints |
| SOUP-001 | SOUP Evaluation Register | Lateral — SOUP items subject to requirements herein |
| CYB-001 | Cybersecurity Architecture Document | Lateral — security requirements supplement §7 |

---

## 2. Overall Description

### 2.1 Product Perspective

#### 2.1.1 System Context

NeuroFusion-AD is a standalone SaMD that integrates into the existing healthcare information ecosystem as a host-independent, cloud-deployable clinical decision support service. It is not a component of a larger medical device hardware system. The system operates as a **prescription-only CDS tool** accessed by authorized clinicians through a RESTful API interface or an integrated EHR plugin.

The following context diagram describes the system boundary and primary data flows:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      HEALTHCARE ECOSYSTEM                           │
│                                                                     │
│  ┌─────────────┐    FHIR R4      ┌──────────────────────────────┐  │
│  │  EHR System │ ─────────────▶  │                              │  │
│  │ (Epic/Cerner│                 │       NeuroFusion-AD          │  │
│  │  /Meditech) │                 │        SaMD Platform          │  │
│  └─────────────┘                 │                              │  │
│                                  │  ┌──────────────────────┐   │  │
│  ┌─────────────┐    REST/JSON    │  │  Data Ingestion &    │   │  │
│  │  Lab System │ ─────────────▶  │  │  Validation Layer    │   │  │
│  │  (LIMS/LIS) │                 │  └──────────┬───────────┘   │  │
│  └─────────────┘                 │             │               │  │
│                                  │  ┌──────────▼───────────┐   │  │
│  ┌─────────────┐    Structured   │  │  Preprocessing       │   │  │
│  │  Acoustic & │    Feature JSON │  │  Pipeline            │   │  │
│  │  Motor Data │ ─────────────▶  │  └──────────┬───────────┘   │  │
│  │  Capture    │                 │             │               │  │
│  └─────────────┘                 │  ┌──────────▼───────────┐   │  │
│                                  │  │  Multimodal GNN      │   │  │
│  ┌─────────────┐                 │  │  Inference Engine    │   │  │
│  │  Clinician  │◀──── Report ──  │  └──────────┬───────────┘   │  │
│  │  Workstation│                 │             │               │  │
│  └─────────────┘                 │  ┌──────────▼───────────┐   │  │
│                                  │  │  Output & Audit      │   │  │
│  ┌─────────────┐                 │  │  Service (PostgreSQL) │   │  │
│  │  Audit/     │◀─── Audit ───   │  └──────────────────────┘   │  │
│  │  Compliance │    Records      │                              │  │
│  └─────────────┘                 └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2.1.2 Technology Infrastructure

NeuroFusion-AD is deployed on the following technology stack, all components of which are governed by the SOUP Evaluation Register (SOUP-001):

| Component | Technology | Version | Role |
|---|---|---|---|
| ML Framework | PyTorch | 2.1.2 | Model training and inference |
| GNN Library | PyTorch Geometric | 2.5.0 | GraphSAGE implementation |
| API Framework | FastAPI | ≥0.104.0 | REST endpoint management |
| Database | PostgreSQL | 14.x | Result persistence and audit trail |
| Containerization | Docker | ≥24.0 | Deployment packaging |
| Orchestration | Kubernetes | ≥1.28 | Scalability and availability |
| Encryption at Rest | AES-256 | N/A | PHI data protection |
| Transport Security | TLS | 1.3 | Data-in-transit protection |

### 2.2 Product Functions

NeuroFusion-AD performs the following principal functions, elaborated in detail in Sections 3–8:

| Function ID | Function Name | Summary |
|---|---|---|
| F-001 | FHIR Data Ingestion | Parse and validate HL7 FHIR R4 Observation resources for all four input modalities |
| F-002 | Input Validation | Enforce hardcoded clinical and technical range constraints on all biomarker and feature inputs |
| F-003 | Preprocessing | Apply z-score normalization, median imputation, and categorical encoding to all input features |
| F-004 | Fluid Biomarker Encoding | Generate a 768-dimensional latent embedding from pTau-217, Aβ42/40, and NfL inputs |
| F-005 | Acoustic Feature Encoding | Generate a modality-specific embedding from speech-derived acoustic features |
| F-006 | Motor Feature Encoding | Generate a modality-specific embedding from kinematic and motor assessment features |
| F-007 | Clinical/Demographic Encoding | Generate an embedding from MMSE, age, sex, APOE status, and comorbidity features |
| F-008 | Cross-Modal Attention Fusion | Fuse four modality embeddings via 8-head, 768-dimensional cross-modal attention |
| F-009 | GNN Inference | Apply 3-layer GraphSAGE to produce patient-level graph-contextualized representations |
| F-010 | Multi-Task Output Generation | Output AD progression risk classification, MMSE trajectory regression, and MCI-to-AD survival prediction |
| F-011 | Uncertainty Quantification | Report calibrated confidence intervals for all three output tasks |
| F-012 | Explainability Output | Generate SHAP-based feature attribution scores for each modality contribution |
| F-013 | Result Persistence | Store all inference inputs, outputs, confidence scores, and metadata to PostgreSQL with full audit logging |
| F-014 | Audit Trail Management | Maintain tamper-evident, timestamped event log per 21 CFR Part 11 |
| F-015 | Access Control | Enforce role-based access control (RBAC) and JWT authentication for all API endpoints |

### 2.3 User Classes and Characteristics

#### 2.3.1 Primary Users

| User Class | Description | Technical Proficiency | Interaction Mode |
|---|---|---|---|
| **Neurologist** | Licensed physician specializing in neurological disorders; primary clinical interpreter of NeuroFusion-AD outputs | Moderate to high clinical; moderate technical | EHR-integrated CDS panel or web interface |
| **Geriatric Psychiatrist** | Licensed physician managing cognitive disorders in older adults; secondary clinical user | Moderate clinical; moderate technical | EHR-integrated or direct API |
| **Memory Disorder Nurse Practitioner** | Advanced practice clinician in memory disorder clinics; may initiate requests and review outputs | Moderate clinical | Web interface |
| **Clinical Researcher** | Academic or industry researcher conducting approved retrospective or prospective studies with NeuroFusion-AD data | High technical; moderate clinical | Batch API with IRB approval |

#### 2.3.2 Administrative and Technical Users

| User Class | Description | Technical Proficiency | Interaction Mode |
|---|---|---|---|
| **System Administrator** | Responsible for deployment, configuration, monitoring, and access provisioning | High technical | Kubernetes/Docker management console, admin API |
| **Biomedical Informatics Engineer** | Responsible for EHR/FHIR integration and data pipeline configuration | High technical | API integration layer |
| **Quality Assurance Auditor** | Reviews audit trails, validates software operation against requirements | Moderate technical, high QA | Audit trail viewer, compliance dashboard |

#### 2.3.3 User Exclusions

The following individuals are **not** intended users and must be prevented from accessing NeuroFusion-AD outputs without appropriate clinical supervision:

- Patients and their family members;
- Non-licensed clinical staff acting without supervising clinician oversight;
- Unauthorized third parties.

### 2.4 Operating Environment and Constraints

#### 2.4.1 Deployment Environment

NeuroFusion-AD shall operate in the following deployment configurations:

| Configuration | Description |
|---|---|
| **Cloud-Hosted SaaS** | Deployed on HIPAA-compliant cloud infrastructure (e.g., AWS GovCloud, Microsoft Azure Healthcare APIs) managed by the manufacturer |
| **On-Premise Enterprise** | Deployed within a healthcare organization's private Kubernetes cluster, managed by the organization's IT department under manufacturer deployment specifications |
| **Hybrid** | Compute on-premise with manufacturer-managed model versioning and update services |

#### 2.4.2 Regulatory and Standards Constraints

The following regulatory constraints are non-negotiable design inputs:

| Constraint ID | Constraint | Source |
|---|---|---|
| OC-001 | All PHI at rest shall be encrypted using AES-256 | HIPAA 45 CFR §164.312(a)(2)(iv); FDA Cybersecurity Guidance 2023 |
| OC-002 | All data in transit shall use TLS 1.3 or higher; TLS 1.2 shall not be used after 2025-12-31 | NIST SP 800-52 Rev.2; FDA Cybersecurity Guidance 2023 |
| OC-003 | Electronic records shall comply with 21 CFR Part 11 | FDA 21 CFR Part 11 |
| OC-004 | Software shall maintain an immutable audit trail of all inference events and user actions | IEC 62304 §9.8; 21 CFR Part 11 §11.10(e) |
| OC-005 | All changes to validated software shall be subject to change control per IEC 62304 §6.3 | IEC 62304 §6.3 |
| OC-006 | Inference outputs shall be labeled with version identifier of the deployed model | FDA AI/ML-Based SaMD Action Plan 2021 |
| OC-007 | System shall not provide autonomous clinical decisions; all outputs require clinician review | EU MDR Annex I §23.4; FDA CDS Guidance 2022 |
| OC-008 | Input validation ranges shall be hardcoded and version-controlled; runtime modification requires change control | ISO 14971 §3.12; IEC 62304 §5.2.5 |

#### 2.4.3 Performance Constraints

| Constraint ID | Constraint | Target | Source |
|---|---|---|---|
| PC-001 | Inference latency (p95) | < 2.0 seconds from validated request receipt to response delivery | System Design Specification SDD-001 |
| PC-002 | System availability | ≥ 99.5% uptime measured monthly, excluding scheduled maintenance | Service Level Agreement SLA-001 |
| PC-003 | Concurrent inference sessions | ≥ 50 simultaneous requests without degradation beyond p95 latency target | Capacity planning estimate |
| PC-004 | Database response time (p95) | < 500 ms for audit write operations | SDD-001 |

### 2.5 Design and Implementation Constraints

| Constraint ID | Constraint Description |
|---|---|
| DIC-001 | The GNN architecture shall implement GraphSAGE with exactly 3 message-passing layers, as validated in TRAIN-001 |
| DIC-002 | The cross-modal attention mechanism shall use 8 attention heads with a 768-dimensional embedding space, matching the validated model architecture |
| DIC-003 | All model weights shall be loaded from cryptographically signed checkpoint files; unsigned checkpoints shall be rejected at initialization |
| DIC-004 | The preprocessing normalization parameters (means and standard deviations) shall be fixed to training dataset statistics documented in TRAIN-001 and shall not adapt at inference time |
| DIC-005 | The system shall not perform any form of online learning, model weight updates, or active learning from inference-time patient data without a complete revalidation and regulatory update cycle |
| DIC-006 | All software components shall be containerized using Docker images with reproducible, pinned base image digests |
| DIC-007 | Python runtime version shall be 3.11.x; PyTorch version shall be 2.1.2; PyTorch Geometric version shall be 2.5.0 |
| DIC-008 | The REST API shall conform to OpenAPI 3.1 specification; the schema shall be version-controlled |

### 2.6 Assumptions and Dependencies

#### 2.6.1 Assumptions

| ID | Assumption |
|---|---|
| A-001 | Fluid biomarker measurements (pTau-217, Aβ42/40, NfL) are obtained from CLIA-certified laboratories using validated analytical methods; the accuracy of laboratory measurements is outside the scope of NeuroFusion-AD's validation |
| A-002 | Clinicians accessing the system have received manufacturer-provided training on the interpretation of NeuroFusion-AD outputs as described in IFU-001 |
| A-003 | The deploying healthcare organization maintains HIPAA-compliant network infrastructure and physical security controls for on-premise deployments |
| A-004 | Acoustic feature extraction from patient speech samples has been performed by a validated pre-processing tool conforming to the feature specification in Appendix A of this document |
| A-005 | Motor feature data has been captured using validated digital assessment protocols; the clinical validity of the assessment instruments is assumed to be established prior to integration |
| A-006 | FHIR R4 Observation resources submitted to the ingestion API accurately represent the underlying laboratory or clinical measurements |