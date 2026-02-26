# Phase 1: Requirements Analysis & Architecture Design

**Duration**: Weeks 1-8 (2 months)  
**Team Size**: 4 FTE (Full-Time Equivalent)  
**Budget Allocation**: 15% of total project  
**Critical Path**: Yes - all subsequent phases depend on this

---

## Phase Overview

Phase 1 establishes the foundational specifications, architecture, and compliance framework for NeuroFusion-AD. This phase produces all regulatory-required documentation and technical blueprints that will govern development through to deployment.

**Key Principle**: "Design twice, build once." Every decision made here must be documented, reviewed, and approved before coding begins.

---

## Team Composition & Roles

### Core Team

**1. ML Architect (Lead - 100% FTE)**
- **Responsibilities**: 
  - Design GNN architecture
  - Define model input/output specifications
  - Select framework and libraries
  - Establish performance benchmarks
  - Review all technical decisions
- **Qualifications**: 
  - PhD in ML/AI or equivalent
  - 5+ years experience with graph neural networks
  - Healthcare AI experience required
  - Publication record in medical ML preferred

**2. Clinical Domain Specialist (50% FTE)**
- **Responsibilities**:
  - Define clinical requirements and use cases
  - Validate feature selection
  - Establish clinical evaluation criteria
  - Review risk analysis from clinical perspective
  - Liaise with neurologist advisors
- **Qualifications**:
  - MD or PhD in Neuroscience/Neurology
  - Clinical trials experience
  - Understanding of Alzheimer's biomarkers
  - Familiarity with FDA/MDR requirements

**3. Regulatory Consultant (30% FTE)**
- **Responsibilities**:
  - Draft regulatory strategy document
  - Ensure IEC 62304 compliance
  - Develop risk management framework (ISO 14971)
  - Prepare FDA pre-submission materials
  - Review all documentation for regulatory adherence
- **Qualifications**:
  - RAC (Regulatory Affairs Certification) preferred
  - 5+ years medical device regulatory experience
  - FDA 510(k) De Novo submission experience
  - MDR Class IIa experience

**4. Data Engineer (70% FTE)**
- **Responsibilities**:
  - Design data pipeline architecture
  - Define data schemas and formats
  - Plan data acquisition strategy
  - Establish data quality framework
  - Create data flow diagrams
- **Qualifications**:
  - BS/MS in Computer Science or related field
  - 3+ years medical data engineering
  - Experience with FHIR and HL7
  - ADNI dataset experience preferred

### External Advisors (Consultation Basis)

**Neurologist Advisor Panel (3-5 clinicians)**
- Monthly review meetings
- User story validation
- Clinical workflow feedback
- Use case refinement

**Roche Technical Liaison**
- Navify integration specifications
- Elecsys assay technical details
- FHIR resource mapping guidance
- Security and compliance requirements

---

## Week-by-Week Breakdown

### Week 1-2: Project Initiation & Stakeholder Alignment

#### Deliverables
1. **Project Charter** (Complete by Day 3)
2. **Stakeholder Register** (Complete by Day 5)
3. **Communication Plan** (Complete by Day 7)
4. **Initial Risk Register** (Complete by Day 10)

#### Activities

**Day 1: Kickoff Meeting**
- **Attendees**: Full team + advisors
- **Agenda**:
  - Project vision and business case review
  - Roles and responsibilities clarification
  - Communication protocols establishment
  - Tool and platform selection
- **Output**: Kickoff meeting minutes (template-driven)

**Day 2-3: Requirements Gathering Workshop**
- **Method**: Joint Application Development (JAD) session
- **Participants**: Core team + 2 neurologist advisors
- **Focus Areas**:
  - Clinical workflow "day in the life" scenarios
  - Pain points with current diagnostic pathway
  - Desired features and capabilities
  - Constraints and non-negotiable requirements
- **Output**: Raw requirements list (50-100 items)

**Day 4-5: Requirements Categorization**
- **Framework**: MoSCoW Method
  - **Must Have**: Essential for FDA/MDR approval and clinical utility
  - **Should Have**: High value but not critical for v1.0
  - **Could Have**: Nice-to-have features for future versions
  - **Won't Have**: Explicitly out of scope
- **Output**: Prioritized requirements matrix

**Day 6-7: Roche Technical Discovery**
- **Activities**:
  - Deep dive on Navify Algorithm Suite technical specs
  - Review Navify Integrator architecture documentation
  - Understand FHIR resource requirements
  - Clarify security and audit requirements
- **Method**: Technical documentation review + Q&A session with Roche liaison
- **Output**: Technical constraints document (15-20 pages)

**Day 8-10: Regulatory Strategy Session**
- **Lead**: Regulatory Consultant
- **Topics**:
  - FDA De Novo pathway feasibility analysis
  - MDR Class IIa requirements review
  - Predicate device analysis (Prenosis Sepsis ImmunoScore)
  - Software lifecycle standards (IEC 62304) overview
  - Risk management framework (ISO 14971) introduction
- **Output**: Regulatory Strategy Document v0.1

#### Documentation Templates

**Project Charter Template**
```markdown
# Project Charter: NeuroFusion-AD

## 1. Project Purpose
[High-level business justification]

## 2. Project Objectives
- Objective 1: [SMART criteria]
- Objective 2: [SMART criteria]
...

## 3. Success Criteria
- Clinical: [Metrics]
- Technical: [Metrics]
- Regulatory: [Milestones]

## 4. High-Level Scope
In Scope:
- [Item 1]
Out of Scope:
- [Item 1]

## 5. Key Stakeholders
[Name, Role, Organization, Influence Level]

## 6. High-Level Timeline
[Phase breakdown with major milestones]

## 7. Budget & Resources
[Estimated costs and FTE allocation]

## 8. Assumptions & Constraints
Assumptions:
- [Assumption 1]
Constraints:
- [Constraint 1]

## 9. Approval
[Signature blocks for key stakeholders]
```

---

### Week 3-4: Detailed Requirements Specification

#### Deliverables
1. **Software Requirements Specification (SRS)** - IEC 62304 compliant (Complete by Week 4)
2. **Use Case Diagrams** (Complete by Day 17)
3. **User Stories** with acceptance criteria (Complete by Day 20)
4. **Data Requirements Document** (Complete by Day 22)

#### Activities

**Day 11-13: Functional Requirements Definition**

**Functional Requirement Categories**:

1. **Data Ingestion Requirements (FRI-001 to FRI-020)**
   - FRI-001: System SHALL accept FHIR Observation resources for plasma pTau-217
   - FRI-002: System SHALL accept FHIR Observation resources for Abeta42/40 ratio
   - FRI-003: System SHALL accept FHIR Observation resources for NfL
   - FRI-004: System SHALL validate biomarker values against physiologically plausible ranges
     - pTau-217: 0.1 - 100 pg/mL
     - Abeta42/40: 0.01 - 0.30 ratio
     - NfL: 5 - 200 pg/mL
   - FRI-005: System SHALL accept FHIR QuestionnaireResponse for acoustic biomarkers
   - FRI-006: System SHALL parse acoustic jitter values (0.0001 - 0.05 range)
   - FRI-007: System SHALL parse acoustic shimmer values (0.001 - 0.3 range)
   - FRI-008: System SHALL parse pause duration (0 - 10 seconds)
   - FRI-009: System SHALL compute semantic density from text transcription
   - FRI-010: System SHALL accept FHIR QuestionnaireResponse for motor biomarkers
   - FRI-011: System SHALL parse gait speed (0.1 - 2.0 m/s)
   - FRI-012: System SHALL parse turn variability (0 - 180 degrees)
   - FRI-013: System SHALL accept FHIR Patient resource for demographics
   - FRI-014: System SHALL extract age from birthDate (40 - 100 years valid)
   - FRI-015: System SHALL extract sex/gender
   - FRI-016: System SHALL accept years of education (0 - 30 years)
   - FRI-017: System SHALL accept APOE ε4 genotype (0, 1, or 2 alleles)
   - FRI-018: System SHALL accept baseline MMSE score (0 - 30)
   - FRI-019: System SHALL handle missing optional features gracefully
   - FRI-020: System SHALL log all validation failures to audit trail

2. **Processing Requirements (FRP-001 to FRP-030)**
   - FRP-001: System SHALL normalize all features to [0,1] range using pre-defined scalers
   - FRP-002: System SHALL construct patient similarity graph with k=20 neighbors
   - FRP-003: System SHALL compute edge weights using cosine similarity
   - FRP-004: System SHALL filter graph edges below similarity threshold (0.3)
   - FRP-005: System SHALL apply 3-layer GraphSAGE convolution
   - FRP-006: System SHALL apply 2-layer Graph Attention Network (GAT)
   - FRP-007: System SHALL use 8-head multi-head attention mechanism
   - FRP-008: System SHALL compute cross-modal attention weights
   - FRP-009: System SHALL apply layer normalization after each GNN layer
   - FRP-010: System SHALL implement residual connections in GNN stack
   - FRP-011: System SHALL apply dropout (p=0.3) during training only
   - FRP-012: System SHALL generate temporal progression trajectory using LSTM
   - FRP-013: System SHALL predict MMSE at 6, 12, and 24 months
   - FRP-014: System SHALL predict CDR-SB trajectory
   - FRP-015: System SHALL compute amyloid positivity probability
   - FRP-016: System SHALL classify risk as High (p>0.7), Medium (0.4<p<0.7), Low (p<0.4)
   - FRP-017: System SHALL compute 95% confidence intervals for predictions
   - FRP-018: System SHALL execute inference in <2 seconds (p95)
   - FRP-019: System SHALL execute inference in <5 seconds (p99)
   - FRP-020: System SHALL support batch processing of up to 100 patients
   - FRP-021: System SHALL compute SHAP values for top 5 features
   - FRP-022: System SHALL extract attention weights per modality
   - FRP-023: System SHALL identify 3 most similar patients from training cohort
   - FRP-024: System SHALL generate natural language explanation
   - FRP-025: System SHALL flag predictions with low confidence (<60%)
   - FRP-026: System SHALL detect out-of-distribution inputs using Mahalanobis distance
   - FRP-027: System SHALL reject inputs outside 3-sigma from training distribution
   - FRP-028: System SHALL log all inference requests with pseudonymized identifiers
   - FRP-029: System SHALL version-stamp all outputs with model version
   - FRP-030: System SHALL maintain stateless operation (no session dependencies)

3. **Output Requirements (FRO-001 to FRO-015)**
   - FRO-001: System SHALL generate FHIR RiskAssessment resource
   - FRO-002: System SHALL include probabilityDecimal (0.00 to 1.00, 2 decimal places)
   - FRO-003: System SHALL include qualitativeRisk (high/medium/low)
   - FRO-004: System SHALL include prediction timeframe (24 months)
   - FRO-005: System SHALL include note with natural language explanation
   - FRO-006: System SHALL include trajectory extension with MMSE predictions
   - FRO-007: System SHALL include explainability extension with attention weights
   - FRO-008: System SHALL include confidence interval extension
   - FRO-009: System SHALL include model version in performer reference
   - FRO-010: System SHALL include timestamp in ISO 8601 format
   - FRO-011: System SHALL include recommended next action (if risk=high)
   - FRO-012: System SHALL generate PDF report for clinician review (optional)
   - FRO-013: System SHALL support JSON output format (alternative to FHIR)
   - FRO-014: System SHALL return HTTP 200 for successful inference
   - FRO-015: System SHALL return appropriate error codes (400, 500, 503)

4. **Integration Requirements (FRINT-001 to FRINT-010)**
   - FRINT-001: System SHALL expose RESTful API endpoint at /api/v1/predict
   - FRINT-002: System SHALL accept POST requests with FHIR Bundle
   - FRINT-003: System SHALL authenticate requests using JWT tokens
   - FRINT-004: System SHALL validate JWT signature against Navify public key
   - FRINT-005: System SHALL authorize requests based on RBAC roles
   - FRINT-006: System SHALL support HL7 v2.x input via adapter service
   - FRINT-007: System SHALL publish results to Navify Hub via webhook
   - FRINT-008: System SHALL retry failed webhook deliveries (3 attempts)
   - FRINT-009: System SHALL support synchronous (response) and asynchronous (job ID) modes
   - FRINT-010: System SHALL provide health check endpoint at /health

**Day 14-16: Non-Functional Requirements Definition**

**Non-Functional Requirement Categories**:

1. **Performance Requirements (NFR-P001 to NFR-P010)**
   - NFR-P001: System SHALL process single inference in <2s (95th percentile)
   - NFR-P002: System SHALL process single inference in <5s (99th percentile)
   - NFR-P003: System SHALL support 50 concurrent API requests
   - NFR-P004: System SHALL process 10,000 inferences per day
   - NFR-P005: System SHALL scale horizontally to 10 pods in Kubernetes
   - NFR-P006: System SHALL have <500MB memory footprint per pod
   - NFR-P007: System SHALL have <2 CPU cores per pod
   - NFR-P008: System SHALL have <500MB model file size
   - NFR-P009: System SHALL optimize for CPU inference (no GPU required)
   - NFR-P010: System SHALL cache frequently accessed reference data

2. **Reliability Requirements (NFR-R001 to NFR-R008)**
   - NFR-R001: System SHALL have 99.9% uptime (SLA)
   - NFR-R002: System SHALL have <1 hour MTTR (Mean Time To Repair)
   - NFR-R003: System SHALL have <0.1% error rate
   - NFR-R004: System SHALL implement circuit breaker for external dependencies
   - NFR-R005: System SHALL gracefully degrade if optional services unavailable
   - NFR-R006: System SHALL retry transient failures with exponential backoff
   - NFR-R007: System SHALL have automated health monitoring
   - NFR-R008: System SHALL have automated failover in multi-region deployment

3. **Security Requirements (NFR-S001 to NFR-S020)**
   - NFR-S001: System SHALL encrypt all data at rest using AES-256
   - NFR-S002: System SHALL encrypt all data in transit using TLS 1.3
   - NFR-S003: System SHALL never log PHI (Protected Health Information)
   - NFR-S004: System SHALL implement pseudonymization for all patient identifiers
   - NFR-S005: System SHALL use secure random number generation for session IDs
   - NFR-S006: System SHALL implement rate limiting (100 req/min per client)
   - NFR-S007: System SHALL implement IP whitelisting for production environments
   - NFR-S008: System SHALL enforce strong password policy for admin accounts
   - NFR-S009: System SHALL implement multi-factor authentication for admin access
   - NFR-S010: System SHALL rotate JWT signing keys every 90 days
   - NFR-S011: System SHALL conduct annual penetration testing
   - NFR-S012: System SHALL undergo OWASP Top 10 vulnerability scanning
   - NFR-S013: System SHALL patch critical vulnerabilities within 7 days
   - NFR-S014: System SHALL maintain audit logs for 7 years (HIPAA requirement)
   - NFR-S015: System SHALL implement immutable audit trail (write-once)
   - NFR-S016: System SHALL alert on suspicious access patterns
   - NFR-S017: System SHALL implement network segmentation (DMZ architecture)
   - NFR-S018: System SHALL use secrets management (AWS Secrets Manager / Vault)
   - NFR-S019: System SHALL implement least privilege access control
   - NFR-S020: System SHALL comply with NIST Cybersecurity Framework

4. **Compliance Requirements (NFR-C001 to NFR-C015)**
   - NFR-C001: System SHALL comply with HIPAA Privacy Rule
   - NFR-C002: System SHALL comply with HIPAA Security Rule
   - NFR-C003: System SHALL comply with GDPR (EU)
   - NFR-C004: System SHALL comply with FDA 21 CFR Part 11 (electronic records)
   - NFR-C005: System SHALL comply with IEC 62304 (software lifecycle)
   - NFR-C006: System SHALL comply with ISO 14971 (risk management)
   - NFR-C007: System SHALL comply with ISO 13485 (quality management)
   - NFR-C008: System SHALL comply with ISO/IEC 27001 (information security)
   - NFR-C009: System SHALL implement GDPR right to erasure
   - NFR-C010: System SHALL implement GDPR data portability
   - NFR-C011: System SHALL conduct DPIA (Data Protection Impact Assessment)
   - NFR-C012: System SHALL maintain Design History File (DHF)
   - NFR-C013: System SHALL maintain Device Master Record (DMR)
   - NFR-C014: System SHALL implement change control per 21 CFR 820.70
   - NFR-C015: System SHALL support post-market surveillance reporting

5. **Usability Requirements (NFR-U001 to NFR-U010)**
   - NFR-U001: System SHALL achieve SUS score >70 in usability testing
   - NFR-U002: System SHALL provide explanations understandable to non-ML clinicians
   - NFR-U003: System SHALL display results in <3 clicks from EMR
   - NFR-U004: System SHALL provide visual trajectory graphs
   - NFR-U005: System SHALL highlight key decision factors
   - NFR-U006: System SHALL use color-coded risk indicators
   - NFR-U007: System SHALL provide printable PDF reports
   - NFR-U008: System SHALL support accessibility standards (WCAG 2.1 AA)
   - NFR-U009: System SHALL provide in-app help documentation
   - NFR-U010: System SHALL support multiple languages (English, German, French - Phase 2)

**Day 17-19: Use Case Modeling**

**Primary Use Cases**:

**Use Case 1: Primary Care Screening**
```
Use Case ID: UC-001
Title: Primary Care Physician screens elderly patient for AD risk
Actor: Primary Care Physician
Precondition: Patient presents with memory complaints
Trigger: PCP decides to evaluate cognitive decline
Main Flow:
  1. PCP orders digital biomarker assessment via tablet
  2. Patient completes Cookie Theft description task (2 min)
  3. Patient completes brief gait assessment (1 min)
  4. System analyzes digital biomarkers
  5. System generates preliminary risk score
  6. IF risk = High or Medium:
       System recommends ordering Elecsys pTau-217 blood test
  7. PCP reviews recommendation
  8. PCP orders blood test
  9. Lab processes sample on cobas analyzer
  10. System receives biomarker results via Navify
  11. System fuses digital + biomarker data
  12. System generates comprehensive risk assessment
  13. System displays results in EMR
  14. PCP reviews explanation panel
  15. PCP discusses results with patient
  16. IF risk = High:
        PCP refers to neurology
Postcondition: Patient has documented AD risk assessment
Alternative Flow 1: Patient unable to complete digital tasks
  4a. System flags incomplete data
  4b. System generates biomarker-only prediction (if available)
  4c. System notes reduced confidence
Alternative Flow 2: Biomarker in "grey zone"
  11a. System assigns higher weight to digital biomarkers
  11b. System flags borderline case in explanation
Exception Flow 1: System unavailable
  E1. System returns error message
  E2. PCP proceeds with standard clinical assessment
Business Rules:
  - BR-001: Digital screening must take <5 minutes
  - BR-002: Blood test only ordered if initial screen positive
  - BR-003: High risk always triggers neurology referral recommendation
```

**Use Case 2: Neurology Diagnostic Confirmation**
```
Use Case ID: UC-002
Title: Neurologist uses NeuroFusion-AD for diagnostic confidence
Actor: Neurologist
Precondition: Patient referred from PCP with positive screen
Trigger: Neurologist receives referral
Main Flow:
  1. Neurologist reviews PCP's screening results in Navify
  2. Neurologist orders comprehensive panel:
     - Plasma pTau-217, Abeta42/40, NfL
     - APOE genotyping
     - Baseline MMSE
  3. Neurologist administers digital biomarker battery (10 min)
  4. All data feeds into NeuroFusion-AD
  5. System generates high-confidence diagnostic prediction
  6. System displays 24-month progression trajectory
  7. System highlights most influential features
  8. Neurologist reviews attention weights
  9. Neurologist compares to 3 similar patients in system
  10. Neurologist makes clinical diagnosis
  11. IF diagnosis = MCI due to AD:
        System recommends DMT eligibility assessment
  12. Neurologist counsels patient using trajectory graph
Postcondition: Patient has confirmed diagnosis with prognosis
Business Rules:
  - BR-004: Full panel increases prediction confidence by >20%
  - BR-005: Trajectory shown with 95% confidence bands
  - BR-006: DMT recommendation only if amyloid positive
```

**Use Case 3: Longitudinal Monitoring**
```
Use Case ID: UC-003
Title: Monitor patient on Disease-Modifying Therapy
Actor: Neurologist, Patient
Precondition: Patient diagnosed and started on Lecanemab
Trigger: 6-month follow-up visit
Main Flow:
  1. Patient completes digital assessment at home (smartphone app)
  2. System compares current digital biomarkers to baseline
  3. Patient visits clinic for blood draw
  4. System receives new biomarker values
  5. System re-runs prediction with updated data
  6. System compares actual trajectory to predicted trajectory
  7. IF actual > predicted (worse than expected):
       System flags accelerated decline
       System recommends treatment review
  8. Neurologist reviews comparison report
  9. Neurologist adjusts treatment if needed
Postcondition: Patient trajectory monitored; treatment optimized
Business Rules:
  - BR-007: Monitoring frequency = every 6 months
  - BR-008: Alert if decline >3 MMSE points vs. predicted
```

**Day 20-22: Data Requirements Specification**

**Data Requirement Document Structure**:

1. **Input Data Sources**
   - Roche cobas Analyzer (HL7/FHIR)
   - Laboratory Information System (HL7 v2.x)
   - EMR (FHIR R4)
   - Digital Biomarker App (HTTPS/JSON)

2. **Data Element Dictionary**

| Element ID | Name | Type | Units | Range | Required | Source | FHIR Mapping |
|-----------|------|------|-------|-------|----------|--------|--------------|
| DE-001 | Patient Pseudo ID | String | N/A | N/A | Yes | All | Patient.id |
| DE-002 | Birth Date | Date | N/A | 1926-2016 | Yes | EMR | Patient.birthDate |
| DE-003 | Sex | Enum | N/A | male/female/other | Yes | EMR | Patient.gender |
| DE-004 | Plasma pTau-217 | Float | pg/mL | 0.1-100 | Yes | cobas | Observation (LOINC TBD) |
| DE-005 | Abeta42/40 Ratio | Float | Ratio | 0.01-0.30 | No | cobas | Observation (LOINC TBD) |
| DE-006 | NfL | Float | pg/mL | 5-200 | No | cobas | Observation (LOINC 96687-9) |
| DE-007 | Education Years | Integer | Years | 0-30 | Yes | EMR | Patient.extension |
| DE-008 | APOE ε4 Alleles | Integer | Count | 0/1/2 | No | EMR | Observation (LOINC 48002-9) |
| DE-009 | MMSE Score | Integer | Points | 0-30 | Yes | EMR | Observation (LOINC 72106-8) |
| DE-010 | Jitter | Float | % | 0.0001-0.05 | Yes | App | QuestionnaireResponse |
| DE-011 | Shimmer | Float | % | 0.001-0.3 | Yes | App | QuestionnaireResponse |
| DE-012 | Pause Duration | Float | Seconds | 0-10 | Yes | App | QuestionnaireResponse |
| DE-013 | Semantic Density | Float | Ratio | 0-1 | Yes | App | QuestionnaireResponse |
| DE-014 | Gait Speed | Float | m/s | 0.1-2.0 | Yes | App | QuestionnaireResponse |
| DE-015 | Turn Variability | Float | Degrees | 0-180 | Yes | App | QuestionnaireResponse |

3. **Data Quality Rules**

| Rule ID | Description | Enforcement |
|---------|-------------|-------------|
| DQ-001 | All numeric values must be within specified ranges | Pre-processing validation |
| DQ-002 | Missing required fields → reject request | API gateway |
| DQ-003 | Timestamps must be within 24 hours of current time | Pre-processing validation |
| DQ-004 | Duplicate submissions (same patient + timestamp) → return cached result | Deduplication layer |
| DQ-005 | Biomarker results from non-Roche analyzers → flag for review | Post-processing |
| DQ-006 | MMSE score inconsistent with expected range for age → flag | Pre-processing warning |

4. **Data Volume Projections**

| Metric | Value | Basis |
|--------|-------|-------|
| Training data size | 5,000 samples | ADNI + Bio-Hermes |
| Features per sample | 15 core features | Data element dictionary |
| Average feature vector size | 120 bytes | Float32 x 15 + metadata |
| Model parameter count | ~2M parameters | GNN architecture estimate |
| Inference input size | <5 KB | Single patient JSON |
| Inference output size | <50 KB | FHIR RiskAssessment + extensions |

---

### Week 5-6: System Architecture Design

#### Deliverables
1. **Software Architecture Document (SAD)** - IEC 62304 compliant (Complete by Week 6)
2. **Component Diagrams** (Complete by Day 30)
3. **Sequence Diagrams** for key flows (Complete by Day 32)
4. **Deployment Architecture** (Complete by Day 35)
5. **Technology Stack Justification** (Complete by Day 35)

#### Activities

**Day 23-25: High-Level Architecture Design**

**Architecture Patterns Selected**:
- **Microservices Architecture**: Decoupled services for scalability
- **Hexagonal Architecture (Ports & Adapters)**: Clean separation of business logic from infrastructure
- **Event-Driven Architecture**: Async processing via message queues
- **CQRS (Command Query Responsibility Segregation)**: Separate write (inference) and read (results) paths

**Component Breakdown**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Navify Ecosystem                         │
└─────────────────────────────────────────────────────────────────┘
                               ▲
                               │ HTTPS/TLS 1.3
                               │ JWT Authentication
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway (Kong/AWS API Gateway)           │
│  - Rate Limiting (100 req/min per client)                      │
│  - JWT Validation                                               │
│  - Request/Response Logging                                     │
│  - CORS Handling                                                │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              NeuroFusion-AD Service (FastAPI)                   │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │         Inference Controller (REST Endpoint)              │ │
│  │  POST /api/v1/predict                                     │ │
│  │  GET /api/v1/status/{job_id}                              │ │
│  │  GET /health                                               │ │
│  └───────────────────────────────────────────────────────────┘ │
│                               │                                 │
│                               ▼                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │     Input Validation & Parsing Layer (Pydantic)           │ │
│  │  - FHIR Bundle parsing                                    │ │
│  │  - Data quality checks                                    │ │
│  │  - Schema validation                                      │ │
│  └───────────────────────────────────────────────────────────┘ │
│                               │                                 │
│                               ▼                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │            Feature Engineering Service                     │ │
│  │  - Biomarker normalization                                │ │
│  │  - Digital feature extraction                             │ │
│  │  - Clinical feature encoding                              │ │
│  │  - Missing data imputation                                │ │
│  └───────────────────────────────────────────────────────────┘ │
│                               │                                 │
│                               ▼                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │         Patient Similarity Network (PSN) Builder          │ │
│  │  - Graph construction (k-NN, k=20)                        │ │
│  │  - Edge weight computation (cosine similarity)            │ │
│  │  - Graph pruning (threshold=0.3)                          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                               │                                 │
│                               ▼                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │      GNN Inference Engine (PyTorch Geometric)             │ │
│  │                                                            │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  Feature Encoders (3 parallel streams)              │ │ │
│  │  │  - Biomarker: Dense(15->64->128)                    │ │ │
│  │  │  - Digital: CNN(6->32) + LSTM(32->64)               │ │ │
│  │  │  - Clinical: Embedding(5->32)                       │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │                          │                                 │ │
│  │                          ▼                                 │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  GNN Layers (3 layers)                               │ │ │
│  │  │  - Layer 1: GraphSAGE(128->256, aggregator=mean)    │ │ │
│  │  │  - Layer 2: GAT(256->256, heads=8)                  │ │ │
│  │  │  - Layer 3: GraphSAGE(256->128)                     │ │ │
│  │  │  Each layer: ReLU + LayerNorm + Residual            │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │                          │                                 │ │
│  │                          ▼                                 │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  Cross-Modal Attention (Transformer)                 │ │ │
│  │  │  - MultiheadAttention(embed_dim=128, heads=8)       │ │ │
│  │  │  - Query: Biomarker embeddings                      │ │ │
│  │  │  - Key/Value: Digital + Clinical embeddings         │ │ │
│  │  │  - Output: Attention weights (for XAI)              │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │                          │                                 │ │
│  │                          ▼                                 │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  Temporal Progression Module                         │ │ │
│  │  │  - LSTM(128->64, 2 layers)                           │ │ │
│  │  │  - Dense(64->3) for 6/12/24 month MMSE              │ │ │
│  │  │  - Dense(64->1) for amyloid probability             │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │                                                            │ │
│  └───────────────────────────────────────────────────────────┘ │
│                               │                                 │
│                               ▼                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │         Explainability Service (SHAP + Attention)         │ │
│  │  - Compute SHAP values for top 5 features                │ │
│  │  - Extract attention weights per modality                │ │
│  │  - Generate natural language explanation                 │ │
│  │  - Find 3 nearest neighbors in training set              │ │
│  └───────────────────────────────────────────────────────────┘ │
│                               │                                 │
│                               ▼                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │     Output Formatter (FHIR RiskAssessment Builder)        │ │
│  │  - Construct FHIR resource                                │ │
│  │  - Add extensions (trajectory, explainability)            │ │
│  │  - Validate output schema                                │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Supporting Services Layer                      │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Model      │  │    Audit     │  │   Metrics    │         │
│  │  Registry    │  │   Logger     │  │  Collector   │         │
│  │  (MLflow)    │  │ (PostgreSQL) │  │ (Prometheus) │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Cache      │  │   Queue      │  │   Secrets    │         │
│  │   (Redis)    │  │  (Celery)    │  │   (Vault)    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Day 26-28: Data Flow & Sequence Diagrams**

**Sequence Diagram: Synchronous Inference**

```
User         API Gateway    Inference Controller    Feature Eng.    GNN Engine    Explainer    Output Formatter    Audit Logger
│                │                  │                     │             │            │                │                │
│ POST /predict  │                  │                     │             │            │                │                │
│───────────────>│                  │                     │             │            │                │                │
│                │ Validate JWT     │                     │             │            │                │                │
│                │ Rate limit OK    │                     │             │            │                │                │
│                │                  │                     │             │            │                │                │
│                │ Forward request  │                     │             │            │                │                │
│                │─────────────────>│                     │             │            │                │                │
│                │                  │ Parse FHIR Bundle   │             │            │                │                │
│                │                  │ Validate schema     │             │            │                │                │
│                │                  │                     │             │            │                │                │
│                │                  │ Extract features    │             │            │                │                │
│                │                  │────────────────────>│             │            │                │                │
│                │                  │                     │ Normalize   │            │                │                │
│                │                  │                     │ Impute missing │         │                │                │
│                │                  │                     │             │            │                │                │
│                │                  │ Feature vector      │             │            │                │                │
│                │                  │<────────────────────│             │            │                │                │
│                │                  │                     │             │            │                │                │
│                │                  │ Construct PSN       │             │            │                │                │
│                │                  │ Run inference       │             │            │                │                │
│                │                  │────────────────────────────────>│            │                │                │
│                │                  │                     │             │ Forward pass │              │                │
│                │                  │                     │             │ Compute attn │              │                │
│                │                  │                     │             │            │                │                │
│                │                  │ Predictions + embeddings         │            │                │                │
│                │                  │<────────────────────────────────│            │                │                │
│                │                  │                     │             │            │                │                │
│                │                  │ Explain prediction  │             │            │                │                │
│                │                  │────────────────────────────────────────────>│                │                │
│                │                  │                     │             │            │ Compute SHAP  │                │
│                │                  │                     │             │            │ Extract attn  │                │
│                │                  │                     │             │            │ Generate NL   │                │
│                │                  │                     │             │            │                │                │
│                │                  │ Explanation object  │             │            │                │                │
│                │                  │<────────────────────────────────────────────│                │                │
│                │                  │                     │             │            │                │                │
│                │                  │ Format output       │             │            │                │                │
│                │                  │────────────────────────────────────────────────────────────>│                │
│                │                  │                     │             │            │                │ Build FHIR    │
│                │                  │                     │             │            │                │ Add extensions │
│                │                  │                     │             │            │                │                │
│                │                  │ FHIR RiskAssessment │             │            │                │                │
│                │                  │<────────────────────────────────────────────────────────────│                │
│                │                  │                     │             │            │                │                │
│                │                  │ Log audit trail     │             │            │                │                │
│                │                  │──────────────────────────────────────────────────────────────────────────────>│
│                │                  │                     │             │            │                │                │
│                │ HTTP 200 + FHIR  │                     │             │            │                │                │
│                │<─────────────────│                     │             │            │                │                │
│                │                  │                     │             │            │                │                │
│<───────────────│                  │                     │             │            │                │                │
│  Response      │                  │                     │             │            │                │                │
```

**Day 29-32: Technology Stack Detailed Justification**

**Decision Matrix for Core Technologies**:

| Technology | Alternatives Considered | Selected | Justification |
|-----------|------------------------|----------|---------------|
| ML Framework | TensorFlow, JAX | **PyTorch 2.1** | - Superior GNN library support (PyTorch Geometric)<br>- Better debugging experience<br>- Industry standard for research→production<br>- Native ONNX export for optimization |
| GNN Library | DGL, Spektral | **PyTorch Geometric 2.4** | - Most mature GNN library<br>- Excellent documentation<br>- Pre-built layers (GraphSAGE, GAT)<br>- Active community |
| API Framework | Flask, Django REST | **FastAPI 0.104** | - Async support (better concurrency)<br>- Automatic OpenAPI docs<br>- Pydantic validation built-in<br>- Type hints enforce code quality |
| Containerization | Podman, LXC | **Docker 24.0** | - Industry standard<br>- Excellent CI/CD support<br>- Roche/Navify uses Docker<br>- Rich ecosystem |
| Orchestration | Docker Swarm, Nomad | **Kubernetes 1.28** | - Navify runs on K8s<br>- Horizontal scaling<br>- Self-healing<br>- Helm chart packaging |
| Database (Audit) | MySQL, MongoDB | **PostgreSQL 16** | - ACID compliance<br>- JSON support for FHIR<br>- Excellent performance<br>- HIPAA-compliant hosting available |
| Cache/Queue | Memcached, RabbitMQ | **Redis 7.2** | - In-memory speed<br>- Pub/sub for events<br>- Celery backend<br>- Persistence options |
| FHIR Library | hapi-fhir, firely | **fhirclient 4.1 (Python)** | - Pure Python<br>- FHIR R4 support<br>- Well-maintained<br>- Good documentation |

**Day 33-35: Deployment Architecture Design**

**Multi-Environment Strategy**:

1. **Development Environment** (Local Laptops + AWS Dev Account)
   - Docker Compose for local development
   - Mock data generators
   - No PHI
   - Relaxed security for debugging

2. **Staging Environment** (AWS Staging Account)
   - Kubernetes cluster (3 nodes)
   - Anonymized ADNI data subset
   - Production-like security
   - Integration testing

3. **Production Environment** (AWS Production Account / Roche Navify Cloud)
   - Kubernetes cluster (5-10 nodes, auto-scaling)
   - Multi-AZ deployment
   - Full security hardening
   - Real patient data (HIPAA-compliant)

**Infrastructure as Code (IaC)**:
- **Terraform** for AWS resource provisioning
- **Helm** for Kubernetes application deployment
- **Ansible** for configuration management

**CI/CD Pipeline**:
```
GitHub Push
    │
    ▼
GitHub Actions
    │
    ├──> Linting (flake8, mypy)
    ├──> Unit Tests (pytest)
    ├──> Security Scan (Snyk, Trivy)
    ├──> Build Docker Image
    │       │
    │       ▼
    │   Push to AWS ECR
    │       │
    │       ▼
    ├──> Deploy to Staging (Helm)
    │       │
    │       ▼
    ├──> Integration Tests (pytest)
    ├──> Performance Tests (Locust)
    │       │
    │       ▼
    └──> Manual Approval Gate
            │
            ▼
        Deploy to Production (Helm)
            │
            ▼
        Post-Deploy Health Checks
```

---

### Week 7-8: Risk Management & Regulatory Documentation

#### Deliverables
1. **Risk Management File (RMF)** - ISO 14971 compliant (Complete by Week 8)
2. **Hazard Analysis** (Complete by Day 45)
3. **Failure Modes and Effects Analysis (FMEA)** (Complete by Day 48)
4. **Software Development Plan (SDP)** - IEC 62304 compliant (Complete by Day 50)
5. **Design History File (DHF) - Phase 1 Section** (Complete by Day 52)

#### Activities

**Day 36-40: Hazard Identification & Risk Analysis**

**Hazard Categories (per ISO 14971)**:
1. **Incorrect or Missing Information Hazards (H1.x)**
2. **Processing Failure Hazards (H2.x)**
3. **Security/Privacy Hazards (H3.x)**
4. **Infrastructure Failure Hazards (H4.x)**
5. **Human Factors Hazards (H5.x)**

**Sample Risk Analysis Table**:

| Hazard ID | Hazard Description | Potential Cause | Harm | Severity | Probability | Risk Level | Risk Control Measure | Residual Risk |
|-----------|-------------------|-----------------|------|----------|-------------|-----------|---------------------|---------------|
| H1.1 | False Negative (missed AD risk) | Training data bias toward APOE4+ | Patient not referred, disease progresses untreated | Serious | Low | Medium | - Stratified validation by APOE status<br>- Calibration on APOE4- subgroup<br>- Sensitivity threshold tuning | Low |
| H1.2 | False Positive (overdiagnosis) | Low specificity threshold | Unnecessary patient anxiety, wasteful testing | Moderate | Medium | Medium | - Specificity target ≥70%<br>- Confidence thresholds<br>- Explainability to support clinical judgment | Low |
| H2.1 | Inference timeout | High server load, memory leak | Result not available, clinical workflow disrupted | Minor | Low | Low | - Load balancing<br>- Circuit breaker<br>- Graceful degradation | Very Low |
| H3.1 | PHI data breach | Insufficient encryption | Patient privacy violation, HIPAA fine | Critical | Very Low | Medium | - AES-256 encryption<br>- Pseudonymization<br>- Access controls<br>- Annual pen test | Low |
| H3.2 | Unauthorized access | Weak authentication | Malicious actor manipulates results | Critical | Very Low | Medium | - JWT with rotation<br>- MFA for admin<br>- IP whitelisting<br>- Rate limiting | Low |
| H5.1 | Clinician misinterprets result | Poor UI design, unclear explanation | Inappropriate treatment decision | Serious | Medium | High | - UX testing with clinicians (n=10)<br>- SUS score >70<br>- Clear risk categories<br>- In-app help | Medium |
| H5.2 | Over-reliance on AI | Automation bias | Clinician ignores contrary clinical judgment | Serious | Low | Medium | - Prominent "Aid, not replacement" disclaimer<br>- Display confidence intervals<br>- Encourage clinical override | Low |

**Risk Acceptability Criteria**:
- **Unacceptable**: Severity=Critical AND Probability≥Medium
- **ALARP (As Low As Reasonably Practicable)**: All Medium/High risks
- **Acceptable**: Low/Very Low risks after mitigation

**Day 41-45: Failure Modes and Effects Analysis (FMEA)**

**FMEA Table Structure**:

| Component | Failure Mode | Effect of Failure | Severity (1-10) | Occurrence (1-10) | Detection (1-10) | RPN (S×O×D) | Recommended Action | Responsibility | Action Taken | New RPN |
|-----------|--------------|-------------------|-----------------|-------------------|------------------|-------------|-------------------|----------------|--------------|---------|
| Feature Encoder | Incorrect normalization parameters | Wrong feature scaling → prediction drift | 8 | 3 | 4 | 96 | - Unit test normalization<br>- Log min/max values<br>- Validate against known samples | ML Engineer | ✓ Tests added | 24 |
| GNN Layer | Gradient explosion during training | Model divergence, no convergence | 7 | 4 | 2 | 56 | - Gradient clipping (max_norm=1.0)<br>- Layer normalization<br>- Learning rate scheduler | ML Engineer | ✓ Implemented | 14 |
| Attention Module | Attention weights sum ≠ 1.0 | Incorrect explainability, invalid SHAP | 6 | 2 | 6 | 72 | - Softmax normalization assertion<br>- Unit test attention outputs | ML Engineer | ✓ Assertion added | 12 |
| FHIR Parser | Malformed JSON input | API crash, no result | 5 | 5 | 3 | 75 | - Pydantic strict validation<br>- Try-except with logging<br>- Return 400 error | Software Engineer | ✓ Validation added | 15 |
| Database | Connection pool exhaustion | Audit log failure (compliance risk) | 9 | 2 | 4 | 72 | - Connection pooling (max 50)<br>- Health check on startup<br>- Fallback to file logging | DevOps Engineer | ✓ Pooling configured | 18 |

**FMEA Review Frequency**: Quarterly during development, annually post-deployment

**Day 46-48: Software Development Plan (SDP)**

**SDP Structure (per IEC 62304)**:

```markdown
# Software Development Plan

## 1. Purpose and Scope
[Define software item and intended medical purpose]

## 2. Roles and Responsibilities
| Role | Name | Responsibilities | % Time |
|------|------|------------------|--------|
| Software Architect | [Name] | Architecture, design reviews | 100% |
| ML Engineer | [Name] | Model development, training | 100% |
| QA Engineer | [Name] | Testing, validation | 50% |
...

## 3. Development Lifecycle Model
- **Model Selected**: Iterative Development (Agile with FDA overlay)
- **Justification**: Allows refinement based on clinical feedback while maintaining traceability
- **Phases**: 5 phases as defined in project plan
- **Iteration Length**: 2-week sprints within each phase

## 4. Development Standards and Methods
- **Coding Standard**: PEP 8 (Python), Google Style Guide
- **Version Control**: Git with feature branch workflow
- **Code Review**: Mandatory peer review before merge (2 approvers)
- **Testing Standard**: IEEE 829 Test Documentation
- **Risk Management**: ISO 14971

## 5. Software Configuration Management
- **Tool**: GitHub Enterprise
- **Branching Strategy**: GitFlow (main, develop, feature/*, release/*, hotfix/*)
- **Versioning**: Semantic Versioning (MAJOR.MINOR.PATCH)
- **Change Control**: All changes require JIRA ticket + design review for > 100 LOC

## 6. Supporting Items
- **Infrastructure**: AWS (dev, staging, prod accounts)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Issue Tracking**: JIRA

## 7. Software Maintenance Plan
[To be detailed in Phase 5]

## 8. Document Control
- **Document ID**: SDP-001
- **Version**: 1.0
- **Approval**: [Signature blocks]
```

**Day 49-52: Phase 1 Design History File (DHF) Compilation**

**DHF Contents (Phase 1)**:
1. Project Charter (signed)
2. Software Requirements Specification (SRS) v1.0
3. Software Architecture Document (SAD) v1.0
4. Risk Management File (RMF) - Preliminary
5. FMEA Report
6. Software Development Plan (SDP) v1.0
7. Meeting Minutes (all requirement workshops, architecture reviews)
8. Decision Logs (technology stack selections, architecture patterns)
9. Traceability Matrix (Requirements → Design Elements) - Initial
10. Review Records (SRS review, SAD review)

**DHF Organization**:
```
DHF/
├── 00_Project_Management/
│   ├── Project_Charter_v1.0_signed.pdf
│   ├── Stakeholder_Register_v1.0.xlsx
│   └── Communication_Plan_v1.0.docx
├── 01_Requirements/
│   ├── SRS_v1.0.pdf
│   ├── Use_Case_Diagrams.pdf
│   ├── User_Stories_v1.0.xlsx
│   └── SRS_Review_Record_2026-03-15.pdf
├── 02_Design/
│   ├── SAD_v1.0.pdf
│   ├── Component_Diagrams.pdf
│   ├── Sequence_Diagrams.pdf
│   ├── Deployment_Architecture.pdf
│   └── SAD_Review_Record_2026-03-30.pdf
├── 03_Risk_Management/
│   ├── RMF_Preliminary_v1.0.pdf
│   ├── Hazard_Analysis_v1.0.xlsx
│   ├── FMEA_v1.0.xlsx
│   └── Risk_Review_Meeting_Minutes_2026-04-05.pdf
├── 04_Planning/
│   ├── SDP_v1.0.pdf
│   ├── Phase_1_Plan.md
│   ├── Phase_2_Plan.md
│   └── ...
├── 05_Traceability/
│   └── Traceability_Matrix_v0.1.xlsx
└── 06_Meeting_Records/
    ├── Kickoff_Meeting_Minutes_2026-02-15.pdf
    ├── Requirements_Workshop_1_Notes.pdf
    └── ...
```

**Traceability Matrix Example**:

| Req ID | Requirement | Design Element | Test Case ID | Status |
|--------|-------------|----------------|--------------|--------|
| FRI-004 | Validate pTau-217 range (0.1-100 pg/mL) | InputValidator.validate_biomarkers() | TC-001 | Designed |
| FRP-005 | Apply 3-layer GraphSAGE convolution | GNN.GraphSAGE_layer1/2/3 | TC-045 | Designed |
| FRO-001 | Generate FHIR RiskAssessment | OutputFormatter.build_fhir() | TC-089 | Designed |
| NFR-P001 | Inference <2s (p95) | - | TC-120 | Designed |

---

## Phase 1 Exit Criteria

**Mandatory Deliverables Checklist**:
- [ ] Software Requirements Specification (SRS) v1.0 - Approved
- [ ] Software Architecture Document (SAD) v1.0 - Approved
- [ ] Risk Management File (RMF) - Preliminary - Approved
- [ ] FMEA Report v1.0 - Completed
- [ ] Software Development Plan (SDP) v1.0 - Approved
- [ ] Design History File (DHF) - Phase 1 Section - Compiled
- [ ] Regulatory Strategy Document v1.0 - Approved
- [ ] Data Requirements Document v1.0 - Approved
- [ ] Traceability Matrix v0.1 - Initial baseline

**Review Gates**:
- [ ] SRS Peer Review (2 reviewers) - Pass
- [ ] SAD Technical Review (ML Architect + External Expert) - Pass
- [ ] Risk Management Review (Regulatory + Clinical) - Pass
- [ ] Phase 1 Gate Review (All stakeholders) - Approved

**Quality Metrics**:
- [ ] Requirements completeness: 100% of user stories have acceptance criteria
- [ ] Requirements testability: 100% of requirements can be verified by a test case
- [ ] Risk coverage: All hazards have ≥1 mitigation measure
- [ ] Traceability: All requirements link to design elements

**Approval Signatures**:
- [ ] ML Architect (Technical Approval)
- [ ] Clinical Specialist (Clinical Approval)
- [ ] Regulatory Consultant (Regulatory Approval)
- [ ] Project Sponsor (Business Approval)

---

## Phase 1 Key Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| Requirements ambiguity → rework in later phases | - Daily stand-ups with team<br>- Weekly stakeholder reviews<br>- Prototype mockups for UI requirements |
| Regulatory strategy rejected by FDA/Notified Body | - Early FDA pre-submission meeting (scheduled in Phase 2)<br>- Regulatory consultant with De Novo experience |
| Architecture too complex to implement in timeline | - Complexity analysis using cyclomatic complexity<br>- Simplification workshop if needed (Day 35) |
| Stakeholder misalignment on priorities | - Monthly steering committee meetings<br>- MoSCoW re-validation at end of Phase 1 |

---

## Phase 1 Tools & Templates

**Collaboration Tools**:
- **Requirements Management**: JIRA + Confluence
- **Architecture Diagramming**: Lucidchart / draw.io
- **Document Collaboration**: Google Workspace (for drafts) → PDF (for controlled versions)
- **Version Control**: GitHub Enterprise
- **Communication**: Slack (daily) + Zoom (weekly reviews)

**Template Library**:
1. SRS Template (IEC 62304 compliant)
2. SAD Template (IEC 62304 compliant)
3. Risk Analysis Template (ISO 14971)
4. FMEA Template
5. Meeting Minutes Template
6. Design Review Checklist
7. Traceability Matrix Template

---

## Next Phase Preview

**Phase 2 Preparations** (to begin Week 9):
- ADNI data access application (submit in Week 1, expect approval by Week 9)
- AWS account provisioning and IAM setup
- MLOps tool selection and proof-of-concept
- Hiring/contracting Data Engineer #2 if needed

---

## Document Control

| Version | Date | Author | Changes | Approvers |
|---------|------|--------|---------|-----------|
| 1.0 | 2026-02-15 | Development Team | Initial Phase 1 plan | Pending |

---

**End of Phase 1 Document**
