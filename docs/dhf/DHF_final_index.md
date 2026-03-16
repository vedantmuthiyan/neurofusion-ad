---
document: dhf-final
generated: 2026-03-16
batch_id: msgbatch_01HRVyhrpdvfnWaMAcE2etBA
status: DRAFT
---

# NeuroFusion-AD v1.0 — Design History File (DHF) Final Index

**Document ID:** NFU-AD-DHF-IDX-001
**Version:** 2.1
**Effective Date:** 2026-03-31
**Classification:** Controlled Document — Confidential
**Prepared By:** [CMO/CTO Co-Authors], NeuroFusion-AD Development Team
**Reviewed By:** Quality Assurance Lead, Regulatory Affairs Lead
**Approved By:** Chief Executive Officer
**Next Review Date:** 2026-09-30

**Regulatory Basis:**
- IEC 62304:2006/AMD1:2015 §5.8 (Software Release)
- 21 CFR Part 820 Subpart J (Device Master Record / DHF)
- 21 CFR Part 11 (Electronic Records)
- ISO 14971:2019 (Risk Management)
- EU MDR 2017/745 Annex IX (QMS)

---

> **DHF Scope Statement:** This Design History File documents the complete design and development lifecycle of NeuroFusion-AD v1.0, a Software as a Medical Device (SaMD) clinical decision support system for amyloid progression risk stratification in MCI patients aged 50–90. The DHF demonstrates that the device was designed and developed in accordance with the approved design plan and applicable regulatory requirements. This index serves as the master table of contents and status tracker for all constituent DHF documents. All documents referenced herein are stored in the validated document management system (DMS) under project code NFU-AD-001.

---

## DHF Master Status Dashboard

| Section | # Documents | # Approved | # In Review | # Open Items | Regulatory Risk |
|---|---|---|---|---|---|
| 00 Project Management | 8 | 7 | 1 | 2 | Low |
| 01 Requirements | 7 | 6 | 1 | 3 | Medium |
| 02 Architecture | 6 | 6 | 0 | 1 | Low |
| 03 Implementation | 9 | 8 | 1 | 2 | Medium |
| 04 Verification & Validation | 11 | 9 | 2 | 4 | **High** |
| 05 Risk Management | 8 | 7 | 1 | 3 | **High** |
| 06 Configuration Management | 5 | 5 | 0 | 0 | Low |
| 07 Release | 6 | 5 | 1 | 2 | Medium |
| **Design Change Notices** | 3 | 2 | 1 | 1 | **High** |
| **TOTAL** | **63** | **55** | **7** | **18** | — |

**Overall DHF Completeness:** 87.3% (55/63 documents in Approved status)
**DHF Freeze Target:** 2026-04-30 (Pre-De Novo Submission)

---

## SECTION 00 — Project Management

**Folder Path:** `/DHF/00_Project_Management/`
**IEC 62304 Reference:** §4.4 (Development Planning), §5.1 (Software Development Planning)
**21 CFR Reference:** 820.30(b) (Design and Development Planning)

---

### 00-001 — Project Charter

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-PM-CHARTER-001 |
| **Version** | 1.2 |
| **Date** | 2024-01-15 |
| **Author** | CEO / CMO / CTO |
| **Reviewer** | Board of Directors |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Defines the strategic mandate, scope boundaries, and executive sponsorship for NeuroFusion-AD v1.0 development. Establishes the intended use as a SaMD clinical decision support tool (not a standalone diagnostic) targeting MCI patients aged 50–90 for amyloid progression risk stratification and Elecsys pTau-217 triage. Formally authorizes resource allocation of $2.3M development budget across 24-month development timeline (Q1 2024 – Q4 2025) with Phase 2B clinical validation extending to March 2026.

**Open Items:** None.

---

### 00-002 — Software Development Plan (SDP)

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-PM-SDP-001 |
| **Version** | 2.0 |
| **Date** | 2024-02-01 (v1.0); 2025-06-15 (v2.0 major revision) |
| **Author** | CTO, Software Engineering Lead |
| **Reviewer** | QA Lead, Regulatory Affairs Lead |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Establishes the full software development lifecycle methodology (hybrid Agile/V-model with 6-week sprint cycles and formal phase gate reviews at design freeze, verification freeze, and validation freeze). Defines IEC 62304 software safety class as **Class B** (non-serious injury possible; device is decision support, not autonomous treatment), with rationale documented in the Risk Management File (§05-001). Specifies development toolchain, version control system (Git/GitLab, branch protection rules), and required documentation artifacts for each phase gate.

**Open Items:**
- **OI-SDP-001** [LOW]: v2.0 appendix referencing SOUP (Software of Unknown Provenance) list requires cross-reference update to match 03-006 (SOUP Registry v1.1). Target resolution: 2026-04-15. Owner: Software Engineering Lead.

---

### 00-003 — RACI Matrix

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-PM-RACI-001 |
| **Version** | 1.3 |
| **Date** | 2025-09-01 |
| **Author** | Project Manager |
| **Reviewer** | All Team Leads |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Defines Responsible, Accountable, Consulted, and Informed assignments for all 63 DHF document types and 24 major project activities across 11 organizational roles (CEO, CMO, CTO, Clinical Affairs, Regulatory Affairs, Software Engineering, Data Science, QA, Biostatistics, Legal/IP, Clinical Site Coordinators). Captures dual-hat assignments for CMO/CTO co-authorship on regulatory submission documents with appropriate independence controls for review and approval. Establishes conflict-of-interest declaration requirements for all document approvers.

**Open Items:** None.

---

### 00-004 — Phase Gate Review Records

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-PM-PGR-001 through PGR-005 |
| **Version** | 1.0 each |
| **Dates** | PGR-001: 2024-03-15 \| PGR-002: 2024-09-30 \| PGR-003: 2025-03-15 \| PGR-004: 2025-09-30 \| PGR-005: 2026-03-28 |
| **Author** | Project Manager, Phase Gate Review Board |
| **Status** | ✅ ALL APPROVED (5/5) |

**Key Content Summary:**
Five formal phase gate records spanning Design Planning (PGR-001), Requirements Freeze (PGR-002), Architecture Freeze (PGR-003), Verification Freeze (PGR-004), and Validation Freeze / DHF Completion (PGR-005). Each record documents entry criteria checklist, exit criteria verification, attendee signatures, action item log, and formal go/no-go decision with rationale. PGR-005 (2026-03-28) confirmed Phase 2B validation results and authorized DHF freeze subject to resolution of 18 documented open items, of which 12 are LOW risk and 6 are MEDIUM/HIGH risk requiring resolution before De Novo submission.

**Open Items:**
- **OI-PGR-001** [MEDIUM]: PGR-005 action item AI-007 (statistical power re-analysis for ADNI subgroup N=44 labeled subset) is in progress. Target: 2026-04-20. Owner: Biostatistics Lead. *See also §04-003.*

---

### 00-005 — Quality Management Plan (QMP)

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-PM-QMP-001 |
| **Version** | 1.1 |
| **Date** | 2024-02-15 |
| **Author** | QA Lead |
| **Reviewer** | CEO, Regulatory Affairs Lead |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Establishes the quality management framework governing NeuroFusion-AD v1.0 development, aligned with ISO 13485:2016 and 21 CFR Part 820 requirements. Defines document control procedures, non-conformance management process (with mandatory NCR issuance for any test failure or protocol deviation), and the audit schedule (internal QMS audit Q2 2025 completed; notified body audit scheduled Q3 2026). Specifies that all Phase 2B clinical data and statistical analysis products are subject to independent biostatistics review before DHF incorporation.

**Open Items:** None.

---

### 00-006 — Regulatory Strategy Document

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-PM-REG-001 |
| **Version** | 1.4 |
| **Date** | 2026-01-10 |
| **Author** | Regulatory Affairs Lead, CMO |
| **Reviewer** | External Regulatory Counsel |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Documents the dual-pathway regulatory strategy: (1) FDA De Novo classification request under 21 CFR 860.257 targeting product code QMF (clinical decision support software), with SaMD non-device determination analysis per FDA 2022 CDS Guidance documenting that NeuroFusion-AD does *not* qualify for non-device status given its intended use in serious/critical disease context requiring clinical review; (2) EU MDR Class IIa conformity assessment per Annex IX via BSI notified body, with MDR Rule 11 classification rationale. Includes Pre-Sub meeting outcomes (FDA meeting Q4 2025, minutes filed as 00-008) confirming De Novo pathway acceptability and identifying key performance thresholds FDA expects in the submission.

**Open Items:** None.

---

### 00-007 — Pre-Submission Meeting Minutes (FDA Q-Sub)

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-PM-QSUB-001 |
| **Version** | 1.0 |
| **Date** | 2025-11-14 |
| **Author** | Regulatory Affairs Lead |
| **Reviewer** | CMO, CTO |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Official record of FDA Q-Sub meeting confirming acceptability of the De Novo pathway for NeuroFusion-AD v1.0 as clinical decision support SaMD. FDA confirmed acceptance of the two-cohort validation design (ADNI internal + Bio-Hermes-001 external) subject to transparency requirements for the synthesized acoustic/motor feature limitation in ADNI, and specified that the APOE4 subgroup performance gap must be addressed in the device labeling with explicit contraindication or performance limitation statement. FDA noted that CSF pTau181 / plasma pTau217 assay bridging must be formally addressed in the Statistical Analysis Plan.

**Open Items:**
- **OI-QSUB-001** [MEDIUM]: FDA requested clarification letter response regarding the N=44 labeled sample size in ADNI internal test; response drafted and under legal review. Target submission: 2026-04-25. Owner: Regulatory Affairs Lead.

---

### 00-008 — Project Meeting Minutes Archive

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-PM-MTG-001 (Archive) |
| **Version** | Ongoing (v1.0–v24.0 representing 24 months of bi-weekly records) |
| **Date** | 2024-01-15 through 2026-03-28 |
| **Author** | Project Manager (rotating scribe) |
| **Status** | 🔄 IN REVIEW (Final archive compilation) |

**Key Content Summary:**
Consolidated archive of 48 bi-weekly project meeting minutes covering the full 24-month development period, including sprint review records, clinical advisory board meeting minutes (3 meetings), biostatistics review meetings, and regulatory preparation sessions. Minutes document key design decisions with rationale as required by 21 CFR 820.30(j), including the GNN architecture selection decision (2024-07-08), the decision to use CSF pTau181 as ADNI proxy with Bio-Hermes bridging validation (2024-11-22), and the Phase 2B data leakage discovery and remediation decision (2025-08-15, see DCN-001). Final archive compilation in progress pending inclusion of PGR-005 follow-up meeting notes.

**Open Items:**
- **OI-MTG-001** [LOW]: PGR-005 follow-up meeting minutes (2026-03-30) pending transcription and approval. Target: 2026-04-10. Owner: Project Manager.

---

## SECTION 01 — Requirements

**Folder Path:** `/DHF/01_Requirements/`
**IEC 62304 Reference:** §5.2 (Software Requirements Analysis)
**21 CFR Reference:** 820.30(c) (Design Input)

---

### 01-001 — Software Requirements Specification (SRS) v1.0

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-REQ-SRS-001 |
| **Version** | 1.0 (Design Freeze) |
| **Date** | 2024-09-30 (Frozen at PGR-002) |
| **Author** | CTO, Software Engineering Lead, CMO (clinical requirements) |
| **Reviewer** | QA Lead, Clinical Advisory Board, Regulatory Affairs Lead |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Defines 187 functional and non-functional software requirements across 12 requirement categories: input data handling (REQ-INP-001 to REQ-INP-022), GNN model inference (REQ-MOD-001 to REQ-MOD-031), multi-task output generation (REQ-OUT-001 to REQ-OUT-018), explainability and SHAP output (REQ-EXP-001 to REQ-EXP-012), FHIR R4 interoperability (REQ-INT-001 to REQ-INT-024), performance thresholds (REQ-PER-001 to REQ-PER-015), security and access control (REQ-SEC-001 to REQ-SEC-019), audit logging (REQ-AUD-001 to REQ-AUD-011), error handling (REQ-ERR-001 to REQ-ERR-014), deployment environment (REQ-ENV-001 to REQ-ENV-008), data privacy/HIPAA (REQ-PRV-001 to REQ-PRV-009), and labeling/output display (REQ-LBL-001 to REQ-LBL-004). Performance requirements include minimum AUC ≥ 0.85 on external validation cohort, maximum MMSE RMSE ≤ 2.5 pts/year, and maximum inference latency ≤ 3 seconds at 95th percentile.

**Open Items:**
- **OI-SRS-001** [MEDIUM]: REQ-LBL-003 (APOE4 performance limitation display in clinician UI) requires update to align with FDA Q-Sub guidance (00-007). Formal change via DCN-003 (pending). Owner: Regulatory Affairs Lead. Target: 2026-04-18.
- **OI-SRS-002** [LOW]: REQ-ENV-005 (GPU/CPU compute requirements) requires update following cloud deployment architecture decision. Target: 2026-04-22. Owner: CTO.

---

### 01-002 — User Needs Document (UND)

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-REQ-UND-001 |
| **Version** | 1.1 |
| **Date** | 2024-07-15 |
| **Author** | CMO, Clinical Affairs Lead |
| **Reviewer** | Clinical Advisory Board (3 neurologists, 1 primary care physician, 1 geriatrician) |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Documents 34 user needs derived from structured interviews with 12 clinicians (8 neurologists, 3 primary care physicians, 1 geriatrician) and review of clinical workflow literature. User needs span three primary user groups: specialist neurologists in memory clinics (UN-SPEC-001 to UN-SPEC-012), primary care physicians performing initial MCI triage (UN-PCP-001 to UN-PCP-011), and clinical administrators/IT staff responsible for system integration (UN-IT-001 to UN-IT-011). Key user needs include: integration with existing EHR systems without workflow disruption (UN-IT-003), output interpretability sufficient for non-specialist users (UN-PCP-007), clear identification of patients requiring confirmatory pTau-217 testing (UN-SPEC-005), and explicit uncertainty communication for predictions (UN-SPEC-009).

**Open Items:** None.

---

### 01-003 — User Stories Backlog

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-REQ-UST-001 |
| **Version** | 2.1 |
| **Date** | 2025-02-28 |
| **Author** | Product Manager, Software Engineering Lead |
| **Reviewer** | CMO, CTO |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Contains 67 user stories in standard Agile format (As a [role], I want [feature], so that [benefit]) with acceptance criteria, story point estimates, and sprint assignment records. Stories are organized by epic: Data Ingestion (6 stories), Model Inference Engine (14 stories), Clinical Output Dashboard (11 stories), FHIR Integration (9 stories), Explainability Layer (8 stories), Security & Access (7 stories), System Administration (7 stories), and Performance Monitoring (5 stories). All 67 stories achieved "Definition of Done" criteria and are linked to corresponding SRS requirements in the Requirements Traceability Matrix (01-005).

**Open Items:** None.

---

### 01-004 — Intended Use Statement and Indications for Use

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-REQ-IUS-001 |
| **Version** | 1.2 |
| **Date** | 2026-01-20 |
| **Author** | CMO, Regulatory Affairs Lead |
| **Reviewer** | External Regulatory Counsel, QA Lead |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Formally defines the Intended Use, Indications for Use, Contraindications, and Intended User for FDA De Novo submission and EU MDR technical documentation. **Intended Use:** NeuroFusion-AD v1.0 is a software-only clinical decision support tool intended to aid qualified healthcare professionals in assessing the risk of amyloid progression in adult patients (aged 50–90) with a diagnosis of Mild Cognitive Impairment (MCI), by integrating plasma biomarker data, cognitive assessment scores, demographic features, and digital behavioral biomarkers to generate a risk stratification score and triage recommendation for confirmatory Elecsys pTau-217 testing. **Contraindications:** Not validated for patients outside the 50–90 age range; not validated for patients with non-AD dementia subtypes (LBD, FTD, VaD); reduced performance in APOE4 homozygous carriers must be disclosed to the ordering clinician via system output (see REQ-LBL-003). **Not Intended For:** Standalone diagnosis, treatment selection, or any autonomous clinical decision-making without clinician review and override capability.

**Open Items:**
- **OI-IUS-001** [MEDIUM]: Contraindication language for APOE4 carriers under final regulatory affairs review for consistency with DCN-003 labeling update. Target: 2026-04-18. Owner: Regulatory Affairs Lead.

---

### 01-005 — Requirements Traceability Matrix (RTM)

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-REQ-RTM-001 |
| **Version** | 2.0 |
| **Date** | 2026-03-15 |
| **Author** | QA Lead, Software Engineering Lead |
| **Reviewer** | CTO, Regulatory Affairs Lead |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Master traceability matrix linking all 187 SRS requirements bidirectionally to: upstream User Needs (01-002), downstream Architecture components (02-001/02-002), Implementation modules (03-001), Verification test cases (04-002), and Validation test cases (04-004). Matrix confirms 100% forward traceability (all 187 requirements have at least one implementing component) and 98.4% backward traceability (184/187 requirements have at least one verification test case; 3 REQ-ENV requirements verified through deployment qualification protocol rather than unit test). Traceability gaps for the 3 REQ-ENV requirements are formally accepted and documented with alternative verification evidence references.

**Open Items:**
- **OI-RTM-001** [LOW]: RTM v2.0 does not yet reflect DCN-001 (data leakage correction) impact on REQ-PER requirements; update pending DCN-001 closure. Target: 2026-04-25. Owner: QA Lead.

---

### 01-006 — Clinical Input Specifications

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-REQ-CIS-001 |
| **Version** | 1.1 |
| **Date** | 2025-01-10 |
| **Author** | CMO, Data Science Lead |
| **Reviewer** | Clinical Advisory Board, Biostatistics Lead |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Specifies the complete input feature set for NeuroFusion-AD v1.0 inference, including data types, units, valid ranges, missingness handling rules, and clinical measurement standards. Input modalities include: plasma biomarkers (pTau-217 [pg/mL], Aβ42/Aβ40 ratio, NfL [pg/mL], GFAP [pg/mL]); cognitive assessments (MMSE, MoCA, CDR-SB — score ranges, administration requirements, inter-rater reliability thresholds); demographic features (age, sex, education years, APOE4 carrier status — binary); and digital behavioral biomarkers (gait speed [m/s], stride variability [CV%], speech fluency metrics [words per minute, pause frequency]). Documents the handling protocol for synthesized acoustic/motor features in ADNI training data and specifies that production inference requires real digital biomarker data per the Bio-Hermes-001 protocol.

**Open Items:** None.

---

### 01-007 — Use Error Analysis (UEA)

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-REQ-UEA-001 |
| **Version** | 1.0 |
| **Date** | 2025-08-30 |
| **Author** | CMO, Human Factors Lead |
| **Reviewer** | Clinical Advisory Board, QA Lead |
| **Status** | 🔄 IN REVIEW |

**Key Content Summary:**
Systematic analysis of potential use errors by intended users (neurologists, primary care physicians, IT administrators) using task analysis, FMEA-style hazard identification, and structured formative usability evaluation findings from 8 simulated clinical sessions. Identifies 23 potential use errors categorized by severity (Critical: 4, Serious: 9, Minor: 10) with mitigations mapped to UI/UX design controls and labeling requirements. Critical use errors identified include: clinician misinterpretation of risk score as diagnostic confirmation (mitigated by mandatory disclaimer display and required acknowledgment), and failure to communicate APOE4 performance limitation to patient (mitigated by automated flag in output report).

**Open Items:**
- **OI-UEA-001** [MEDIUM]: Summative usability evaluation (formative is complete; summative requires 15-participant study with target users) not yet conducted. Protocol approved; study scheduled Q2 2026. FDA has accepted formative results for De Novo submission with summative results as post-market commitment per Q-Sub agreement. Target summative completion: 2026-06-30. Owner: Human Factors Lead.

---

## SECTION 02 — Architecture

**Folder Path:** `/DHF/02_Architecture/`
**IEC 62304 Reference:** §5.3 (Software Architectural Design)
**21 CFR Reference:** 820.30(d) (Design Output)

---

### 02-001 — Software Architecture Document (SAD) v1.0

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-ARCH-SAD-001 |
| **Version** | 1.0 (Architecture Freeze) |
| **Date** | 2025-03-15 (Frozen at PGR-003) |
| **Author** | CTO, Software Engineering Lead |
| **Reviewer** | QA Lead, Security Lead, Regulatory Affairs Lead |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Defines the complete software architecture for NeuroFusion-AD v1.0 using arc42 architecture documentation framework, covering system context, solution strategy, building blocks (levels 1–3), runtime views, deployment views, and architectural decisions with rationale. The architecture implements a six-layer design: (1) Data Ingestion Layer (FHIR R4 REST API, HL7 v2 adapter, CSV/JSON import); (2) Feature Engineering Pipeline (preprocessing, normalization, missingness imputation using MICE algorithm); (3) Graph Construction Engine (patient similarity graph builder using k-NN with k=5 based on biomarker and demographic feature cosine similarity); (4) GNN Inference Engine (12-layer heterogeneous Graph Attention Network, 12M parameters, PyTorch Geometric); (5) Multi-Task Output Layer (amyloid classification head, MMSE regression head, survival analysis head using DeepSurv); (6) Explainability and Presentation Layer (SHAP-based feature attribution, integrated gradients for graph attention, clinician dashboard API). Architecture Decision Records (ADRs) 001–017 document all major architectural choices with rationale, considered alternatives, and consequences.

**Open Items:**
- **OI-SAD-001** [LOW]: ADR-015 (cloud deployment architecture — AWS vs. Azure) finalization deferred to post-acquisition infrastructure decision. Pre-acquisition deployments use on-premises Docker/Kubernetes. Roche acquisition integration architecture to be addressed in v1.1 DHF. Owner: CTO.

---

### 02-002 — Component Diagrams Package

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-ARCH-CDP-001 |
| **Version** | 1.0 |
| **Date** | 2025-03-15 |
| **Author** | Software Engineering Lead |
| **Reviewer** | CTO, QA Lead |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Package of 14 UML component diagrams (C4 model: System Context, Container, Component levels) covering all major subsystems. Includes: system context diagram showing integration with EHR systems, Roche Elecsys instrument interface, and clinician dashboard; container diagram showing Docker container boundaries, inter-service communication protocols (REST/gRPC), and data persistence layers; component-level diagrams for the GNN inference engine (showing graph attention mechanism, message passing layers, readout functions), the FHIR adapter (HL7 FHIR R4 resource mappings, SMART on FHIR OAuth2 flow), and the explainability module (SHAP value computation pipeline, attention weight extraction). All diagrams version-controlled in GitLab and regenerated from architecture-as-code (Structurizr DSL) to ensure synchronization with implementation.

**Open Items:** None.

---

### 02-003 — Data Architecture and Flow Specification

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-ARCH-DAF-001 |
| **Version** | 1.1 |
| **Date** | 2025-04-10 |
| **Author** | Data Science Lead, Software Engineering Lead |
| **Reviewer** | CMO, Privacy Officer, QA Lead |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Specifies all data flows through the NeuroFusion-AD system, including data at rest and in transit encryption requirements (AES-256 at rest, TLS 1.3 in transit), PHI handling and de-identification procedures (per HIPAA Safe Harbor and Expert Determination methods), data retention policies, and the model inference data pipeline from raw clinical inputs to final risk score output. Documents the patient similarity graph construction algorithm in detail, including the k-NN graph builder parameters (k=5, cosine similarity metric, minimum similarity threshold θ=0.15), edge weight computation, and dynamic graph update protocol for new patients. Includes formal data lineage diagram tracing each input feature from source instrument/EHR field through preprocessing, normalization, and feature vector construction to GNN input tensor.

**Open Items:** None.

---

### 02-004 — API Specification (FHIR R4 / REST)

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-ARCH-API-001 |
| **Version** | 1.2 |
| **Date** | 2025-07-20 |
| **Author** | Software Engineering Lead |
| **Reviewer** | CTO, QA Lead |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Complete OpenAPI 3.0 specification for all NeuroFusion-AD external interfaces, including the FHIR R4 REST endpoint definitions (SMART on FHIR OAuth2 authentication, Patient/$risk-stratify operation, DiagnosticReport resource output format), the Elecsys pTau-217 triage recommendation webhook (outbound notification to ordering system), and the clinical dashboard API (React frontend data contract). Specifies FHIR resource profiles used (US Core Patient, Observation, DiagnosticReport with NeuroFusion-AD extension profile for risk score and triage recommendation), ICD-10 coding requirements (G31.84 for MCI), SNOMED-CT concept mappings for biomarker observations, and LOINC codes for all laboratory inputs. API versioning strategy and backward compatibility commitments documented.

**Open Items:** None.

---

### 02-005 — Model Architecture Specification (GNN)

| Field | Value |
|---|---|
| **Document ID** | NFU-AD-ARCH-MAS-001 |
| **Version** | 1.0 |
| **Date** | 2025-03-15 |
| **Author** | Data Science Lead, CTO |
| **Reviewer** | External ML Advisor, QA Lead |
| **Status** | ✅ APPROVED |

**Key Content Summary:**
Formal mathematical and implementation specification of the NeuroFusion-AD GNN model, constituting a regulatory-grade model card and architecture description. Specifies the heterogeneous Graph Attention Network (GAT) with 12 message-passing layers, multi-head attention (8 heads per layer), node feature dimensionality (128-dimensional patient embeddings), edge feature encoding (biomarker similarity + temporal proximity), readout function (hierarchical graph pooling with DiffPool), and the three task-specific output heads: (1) Binary amyloid classification head (sigmoid output, calibrated with temperature scaling, EC