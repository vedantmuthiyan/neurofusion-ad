# NeuroFusion-AD: MoSCoW Prioritization Summary & Quick Reference

**Document Type**: Executive Summary + Visual Prioritization  
**Audience**: Project Manager, Scrum Master, Tech Leads  
**Date**: February 15, 2026  
**Version**: 1.0

---

## QUICK STATISTICS

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Raw Requirements** | 103 | From comprehensive requirements document |
| **MUST HAVE** | 40 (39%) | FDA/MDR approval critical; cannot launch without |
| **SHOULD HAVE** | 35 (34%) | Expected by clinicians; v1.1 timeline |
| **COULD HAVE** | 15 (15%) | Nice-to-have; v2.0+ timeline |
| **WON'T HAVE** | 13 (12%) | Out of scope v1.0; future products |
| **Phase 1 Requirements** | ~35 | Regulatory + foundational architecture |
| **Phase 2 Requirements** | ~45 | Clinical validation + model optimization |
| **Phase 3 Requirements** | ~23 | Integration + deployment + regulatory submission |

---

## REQUIREMENT BREAKDOWN BY CATEGORY

### Functional Requirements (FR): 75 Requirements

#### Data Ingestion & Preprocessing (FR-DIP): 20 Requirements
| Priority | MUST | SHOULD | COULD | WON'T | Total |
|----------|------|--------|-------|-------|-------|
| Count | 8 | 3 | 0 | 0 | 20 |
| % | 40% | 15% | 0% | 0% | 100% |

**Key MUST HAVE**: FHIR input, audio/IMU acceptance, normalization, pseudonymization  
**Key SHOULD HAVE**: Semantic density, async processing, QuestionnaireResponse mapping

#### Model Inference & Prediction (FR-MIP): 30 Requirements
| Priority | MUST | SHOULD | COULD | WON'T | Total |
|----------|------|--------|-------|-------|-------|
| Count | 7 | 8 | 4 | 0 | 30 |
| % | 23% | 27% | 13% | 0% | 100% |

**Key MUST HAVE**: Core GNN architecture, classification/regression/survival heads, confidence intervals, SHAP, attention weights  
**Key SHOULD HAVE**: Unit tests, model rollback, k-nearest neighbors, model ensemble

#### Output & Reporting (FR-OUT): 20 Requirements
| Priority | MUST | SHOULD | COULD | WON'T | Total |
|----------|------|--------|-------|-------|-------|
| Count | 4 | 4 | 0 | 0 | 20 |
| % | 20% | 20% | 0% | 0% | 100% |

**Key MUST HAVE**: FHIR RiskAssessment, clinical disclaimer, clinician-friendly language  
**Key SHOULD HAVE**: JSON, PDF, HTML outputs; discordant flags

#### Clinical Decision Support & Alerts (FR-CDS): 15 Requirements
| Priority | MUST | SHOULD | COULD | WON'T | Total |
|----------|------|--------|-------|-------|-------|
| Count | 2 | 3 | 1 | 0 | 15 |
| % | 13% | 20% | 7% | 0% | 100% |

**Key MUST HAVE**: HIGH RISK, DECLINING, ADVERSE alerts with evidence  
**Key SHOULD HAVE**: Discordant alerts, threshold customization, EHR inbox integration

#### API & Integration (FR-API): 25 Requirements
| Priority | MUST | SHOULD | COULD | WON'T | Total |
|----------|------|--------|-------|-------|-------|
| Count | 6 | 6 | 2 | 0 | 25 |
| % | 24% | 24% | 8% | 0% | 100% |

**Key MUST HAVE**: /fhir/RiskAssessment/$process endpoint, FHIR/OAuth/RBAC, status codes  
**Key SHOULD HAVE**: Async processing, OpenAPI, HL7 v2.x legacy support, CORS

---

### Non-Functional Requirements (NFR): 78 Requirements

#### Performance (NFR-PERF): 10 Requirements
| Priority | MUST | SHOULD | COULD | WON'T | Total |
|----------|------|--------|-------|-------|-------|
| Count | 1 | 2 | 1 | 0 | 10 |
| % | 10% | 20% | 10% | 0% | 100% |

**Key MUST HAVE**: p95 latency <2.0s  
**Key SHOULD HAVE**: Throughput 100 pred/hr, batch processing, startup time

#### Security & Compliance (NFR-SEC + NFR-COMP): 35 Requirements
| Priority | MUST | SHOULD | COULD | WON'T | Total |
|----------|------|--------|-------|-------|-------|
| Count | 8 | 5 | 2 | 0 | 35 |
| % | 23% | 14% | 6% | 0% | 100% |

**Key MUST HAVE**: HTTPS/TLS 1.3, OAuth 2.0, patient pseudonymization, audit logging, IEC 62304, ISO 14971, HIPAA  
**Key SHOULD HAVE**: GDPR, SOC 2 Type II, EHR-specific integrations

#### Reliability & Monitoring (NFR-REL + NFR-MAINT): 18 Requirements
| Priority | MUST | SHOULD | COULD | WON'T | Total |
|----------|------|--------|-------|-------|-------|
| Count | 2 | 8 | 2 | 0 | 18 |
| % | 11% | 44% | 11% | 0% | 100% |

**Key MUST HAVE**: 99.5% uptime, error logging  
**Key SHOULD HAVE**: Circuit breaker, retries, backup/restore, CI/CD, monitoring stack

#### Scalability & Interoperability (NFR-SCALE + NFR-INTER): 15 Requirements
| Priority | MUST | SHOULD | COULD | WON'T | Total |
|----------|------|--------|-------|-------|-------|
| Count | 0 | 8 | 2 | 0 | 15 |
| % | 0% | 53% | 13% | 0% | 100% |

**Key SHOULD HAVE**: Kubernetes HPA, FHIR/HL7 mapping, Epic/Cerner integration

#### Explainability (NFR-EXPLAIN): 10 Requirements
| Priority | MUST | SHOULD | COULD | WON'T | Total |
|----------|------|--------|-------|-------|-------|
| Count | 2 | 4 | 2 | 0 | 10 |
| % | 20% | 40% | 20% | 0% | 100% |

**Key MUST HAVE**: Attention weights, SHAP values, confidence intervals  
**Key SHOULD HAVE**: Similar patient cases, uncertainty flags, decomposition

---

## PHASE-BASED REQUIREMENT ALLOCATION

### Phase 1 (Months 1-4): Foundation & Architecture
**Estimated Effort**: 35-40 requirements  
**Focus**: Regulatory compliance, data pipelines, model architecture

| Category | MUST | SHOULD | COULD | Effort |
|----------|------|--------|-------|--------|
| Regulatory (SRS, SAD, RMF) | 5 | 0 | 0 | 40 hrs |
| Data Ingestion (FR-DIP) | 8 | 2 | 0 | 120 hrs |
| Model Architecture (FR-MIP-001 to -011) | 6 | 2 | 0 | 200 hrs |
| API Skeleton (FR-API) | 2 | 1 | 0 | 80 hrs |
| Unit Tests (NFR-MAINT) | 0 | 2 | 0 | 60 hrs |
| **Total Phase 1** | **21** | **7** | **0** | **500 hrs** |

**Phase 1 Exit Criteria**:
- ✅ SRS, SAD, RMF v1.0 approved by regulatory officer
- ✅ ADNI preprocessed & baseline model training initiated
- ✅ GNN architecture implemented & unit tested
- ✅ Prototype achieves ≥70% classification accuracy
- ✅ FHIR API skeleton deployed (latency <3s)

---

### Phase 2 (Months 5-10): Model Training & Validation
**Estimated Effort**: 45-50 requirements  
**Focus**: Clinical validation, hyperparameter optimization, explainability

| Category | MUST | SHOULD | COULD | Effort |
|----------|------|--------|-------|--------|
| Model Training & Tuning (FR-MIP) | 7 | 8 | 4 | 600 hrs |
| Explainability (FR-MIP + NFR-EXPLAIN) | 2 | 4 | 2 | 150 hrs |
| Clinical Validation | 0 | 2 | 0 | 200 hrs |
| Output & Reporting (FR-OUT) | 4 | 3 | 0 | 100 hrs |
| Alerts & CDS (FR-CDS) | 2 | 2 | 1 | 80 hrs |
| Integration Testing (NFR-MAINT) | 0 | 3 | 1 | 120 hrs |
| **Total Phase 2** | **15** | **22** | **8** | **1250 hrs** |

**Phase 2 Exit Criteria**:
- ✅ Classification AUC ≥0.85 on test set + ADNI validation
- ✅ Regression RMSE ≤3.0 MMSE points
- ✅ Survival C-index ≥0.75
- ✅ Subgroup performance gap <0.05 (racial/gender equity)
- ✅ Clinical Validation Report completed & approved
- ✅ SHAP + attention explainability validated by blind clinician review
- ✅ Hyperparameter optimization converged (Optuna 50 trials)

---

### Phase 3 (Months 11-16): Integration, Deployment & Regulatory Submission
**Estimated Effort**: 23-30 requirements  
**Focus**: Navify integration, security hardening, regulatory dossier, deployment

| Category | MUST | SHOULD | COULD | Effort |
|----------|------|--------|-------|--------|
| API Production Hardening (FR-API) | 4 | 5 | 2 | 200 hrs |
| Security & Compliance (NFR-SEC, COMP) | 8 | 5 | 2 | 300 hrs |
| Containerization & Deployment (NFR-MAINT, SCALE) | 0 | 3 | 1 | 150 hrs |
| Monitoring & Observability | 1 | 3 | 0 | 120 hrs |
| Load Testing & Performance (NFR-PERF) | 1 | 2 | 1 | 100 hrs |
| Regulatory Documentation (DHF compilation) | 0 | 0 | 0 | 400 hrs |
| FDA/MDR Submissions | 0 | 0 | 0 | 200 hrs |
| **Total Phase 3** | **14** | **18** | **6** | **1470 hrs** |

**Phase 3 Exit Criteria**:
- ✅ Docker container built & deployable (runs on Kubernetes)
- ✅ Load testing: 1000 concurrent requests, p95 <2.0s
- ✅ Navify API conformance tests passed
- ✅ Security audit passed (penetration testing, HIPAA compliance)
- ✅ Design History File (DHF) compiled (~300 pages)
- ✅ FDA De Novo application filed
- ✅ MDR Technical File submitted to Notified Body (TÜV SÜD)
- ✅ User manual & training materials finalized

---

## RISK-BASED PRIORITIZATION

### High-Risk Requirements (If Not Met = Project Fails)

| Requirement | Risk | Mitigation | Owner |
|-------------|------|-----------|-------|
| **MUST HAVE: FHIR API (FR-API-001)** | Navify integration impossible without standardized API contract | Start API design week 1; get Roche API spec early | ML Architect + Data Engineer |
| **MUST HAVE: Classification AUC ≥0.85 (Phase 2)** | FDA/clinicians require diagnostic-grade accuracy; <0.85 = rejection | Begin training month 5; reserve 6 months for model iteration | ML Architect + Research Engineer |
| **MUST HAVE: Security compliance (NFR-SEC)** | Data breach = regulatory violation, lawsuit, loss of patient trust | Hire security consultant month 1; penetration testing month 15 | Regulatory Officer + DevOps |
| **MUST HAVE: IEC 62304 documentation (NFR-COMP)** | FDA/MDR cannot approve without comprehensive design history file | Start documenting month 1; allocate 400 hours | Regulatory Officer |
| **MUST HAVE: ADNI/Bio-Hermes access (FR-DIP)** | No training data = no model; project blocked | Submit access requests week 1; plan for 2-4 week approval | Data Engineer |
| **SHOULD HAVE: Navify monitoring (Phase 3)** | Post-launch, cannot detect if model degrading in production | Build monitoring infrastructure month 13 | DevOps |

---

## EFFORT ESTIMATION SUMMARY

### Total Development Hours (All Phases)

| Phase | Low Est. | High Est. | Avg. | Notes |
|-------|----------|----------|------|-------|
| Phase 1 (16 weeks) | 450 | 550 | **500 hrs** | Heavy regulatory documentation |
| Phase 2 (24 weeks) | 1100 | 1400 | **1250 hrs** | Model training + validation (parallelizable) |
| Phase 3 (24 weeks) | 1300 | 1600 | **1470 hrs** | Regulatory submission + security audit |
| **Total (64 weeks)** | **2850** | **3550** | **3220 hrs** | ~50 hrs/week per 6-FTE team |

### FTE Allocation by Role

| Role | Phase 1 | Phase 2 | Phase 3 | Total |
|------|---------|---------|---------|-------|
| ML Architect | 1.0 | 1.0 | 0.8 | 2.8 FTE |
| Data Engineer | 1.0 | 0.8 | 0.5 | 2.3 FTE |
| ML Research Engineer | 0.8 | 1.0 | 0.5 | 2.3 FTE |
| Clinical Specialist | 0.5 | 0.5 | 0.3 | 1.3 FTE |
| Regulatory Officer | 0.4 | 0.2 | 0.8 | 1.4 FTE |
| DevOps/MLOps Engineer | 0.3 | 0.5 | 0.8 | 1.6 FTE |
| **Total** | **4.0** | **4.5** | **3.7** | **6.0 FTE avg** |

---

## DEPENDENCY GRAPH: Critical Path

```
Phase 1 (Weeks 1-16):
├─ Week 1-2: Kickoff + Regulatory Framework (SRS, SAD, RMF)
├─ Week 1-4: Dataset Access (ADNI, Bio-Hermes) ← CRITICAL PATH
├─ Week 4-8: Data Preprocessing Pipeline
├─ Week 4-12: Model Architecture Implementation
├─ Week 12-16: Unit Testing + Baseline Training
└─ Gate Review: Approve Phase 1 deliverables

Phase 2 (Weeks 17-40):
├─ Week 17-28: Full-Scale Model Training (150 epochs on ADNI) ← CRITICAL PATH
├─ Week 28-34: Hyperparameter Optimization (Optuna 50 trials)
├─ Week 34-38: Bio-Hermes Fine-Tuning (if dataset available)
├─ Week 28-40: Clinical Validation (parallel: SHAP, attention analysis)
└─ Gate Review: Validate AUC ≥0.85, approve Phase 2

Phase 3 (Weeks 41-64):
├─ Week 41-48: Navify API Integration + Containerization ← CRITICAL PATH
├─ Week 49-56: Security Audit + Penetration Testing ← CRITICAL PATH
├─ Week 52-60: DHF Compilation (200+ pages)
├─ Week 57-64: FDA/MDR Submission Preparation
└─ Gate Review: FDA/MDR submissions filed, deployable system ready

Critical Dependencies:
- Phase 2 BLOCKED until Phase 1 data pipelines complete (Week 8)
- Phase 3 API integration BLOCKED until Phase 2 model validation (Week 40)
- FDA submission BLOCKED until DHF complete (Week 60)
```

---

## STAKEHOLDER REQUIREMENTS vs. MoSCoW PRIORITY

### Roche Stakeholder Priorities

| Roche Goal | Requirement Mapping | Priority | Impact |
|------------|-------------------|----------|--------|
| **Navify Algorithm Suite compatibility** | FR-API-001, FR-API-002, FR-OUT-001 | MUST | Without integration, no deployment path |
| **Reagent pull-through (Elecsys pTau-217)** | FR-DIP-001, FR-OUT-006 | MUST | Business model depends on recommending biomarker testing |
| **FDA/MDR approval** | NFR-COMP-008 to 015, regulatory docs | MUST | Legal requirement |
| **Explainability for clinicians** | FR-MIP-024, FR-MIP-025, NFR-EXPLAIN | MUST | Clinician adoption depends on understanding why |
| **Security & compliance** | NFR-SEC, HIPAA, GDPR | MUST | Patient data protection requirement |
| **Performance for clinical workflow** | NFR-PERF-001 (p95 <2.0s) | MUST | Cannot block clinician for 10+ seconds |
| **Digital biomarker validation** | Phase 2 clinical validation | MUST | Must prove acoustic/motor features add value |
| **Post-market surveillance capability** | NFR-SEC-008-010, monitoring | SHOULD | Required for ongoing FDA/MDR compliance |
| **Model versioning & rollback** | FR-MIP-019, FR-MIP-020 | SHOULD | Safety feature for detecting drift |
| **Admin customization** | FR-CDS-009, Feature 15 | COULD | Hospital variability; v1.1 feature |
| **Epic/Cerner specific integration** | NFR-INTER-009-010 | SHOULD | Deployment at major hospital systems |
| **Multi-language support** | WON'T HAVE | v2.0 | Nice-to-have, post-launch |

---

## V1.0 vs. V1.1 FEATURE ROADMAP

### What Ships in V1.0 (End of Phase 3, Month 16)

**Core Capabilities**:
- ✅ FHIR-compliant API accepting fluid biomarkers, digital biomarkers, clinical data
- ✅ Multimodal GNN producing 3 outputs (amyloid classification, MMSE regression, survival prediction)
- ✅ Attention-based explainability (modality weights) + SHAP feature importance
- ✅ High/medium/low risk stratification with confidence intervals
- ✅ Clinician alerts (high risk, declining, adverse)
- ✅ HIPAA/GDPR compliance, audit logging, encryption
- ✅ Docker containerization + Kubernetes deployment
- ✅ Single-instance throughput (~50 pred/hr), latency p95 <2.0s
- ✅ FDA De Novo + MDR submitted (pending approval)
- ✅ User manual + training materials

**Supported Modalities**:
- ✅ Fluid: pTau-217, Aβ42/40, NfL (from Roche cobas analyzer)
- ✅ Acoustic: Jitter, shimmer, MFCC, pitch (from voice recording)
- ✅ Motor: Gait speed, stride variability, double support time (from smartphone IMU)
- ✅ Clinical: Age, sex, education, APOE, MMSE

**Deployment Context**:
- ✅ Academic medical centers (2-3 pilot sites)
- ✅ Specialist neurology clinics
- ✅ Roche diagnostic labs
- ✅ On-premise (Navify Integrator edge) + AWS cloud
- ✅ Epic integration (FHIR API basic support)

---

### What Arrives in V1.1 (Month 18-22, Post-Pilot Feedback)

**New Features** (SHOULD HAVE from v1.0):
- ✅ Async processing (for batch predictions, high-volume scenarios)
- ✅ Semantic density extraction (advanced NLP-based speech analysis)
- ✅ PDF output (patient-friendly reports for handouts)
- ✅ HTML dashboard (clinician web UI for result review)
- ✅ Discordant alert logic (flag when biomodalities disagree)
- ✅ Threshold customization (hospital-specific risk categories)
- ✅ Epic SmartText integration (native EHR workflow)
- ✅ Auto-scaling from 2-10 replicas (Kubernetes HPA)
- ✅ Advanced monitoring (drift detection, explainability monitoring)
- ✅ GDPR right-to-erasure automation

**Performance Improvements**:
- ⚡ Throughput: 50 pred/hr → 200 pred/hr (4x increase via optimization)
- ⚡ Latency: p95 2.0s → p95 <1.0s (quantization, caching)

**Supported Integrations**:
- ✅ Cerner (via FHIR adapter)
- ✅ Meditech (via FHIR adapter)
- ✅ Laboratory information systems (additional LOINC codes)

**Post-Pilot Validations**:
- ✅ Real-world performance validation (1000+ patients)
- ✅ Operational feedback integration (hospital IT workflows)
- ✅ Safety & adverse event monitoring (0 serious events expected)

---

### What Ships in V2.0+ (Post-FDA Approval, Year 2-3)

**Advanced Features** (COULD HAVE from v1.0):
- 📱 Patient mobile app (iOS/Android remote monitoring)
- 🔬 MRI/PET imaging integration (multimodal with neuroimaging)
- 🧬 Genetic risk score (APOE → polygenic risk score)
- 🏥 Multi-center federated learning (train without centralizing PHI)
- 📊 Treatment response monitoring (compare baseline → 6-month follow-up)
- 🎯 Pharmaceutical recommendation engine (DMT selection based on biomarker profile)
- 🌍 Multilingual support (Spanish, Mandarin, Japanese)
- ⌚ Wearable integration (Fitbit, Apple Watch for continuous monitoring)
- 🤖 Causal inference layer (identify biomarker→cognition relationships)

**Expanded Deployments**:
- Primary care settings (with refined screening threshold)
- Telemedicine platforms (remote assessment support)
- Nursing homes & assisted living
- International markets (EU, Asia-Pacific)

---

## DECISION TREE: Should a Requirement Be MUST, SHOULD, COULD, or WON'T?

Use this framework when evaluating new requirements or changes:

```
┌─ Does it block FDA/MDR approval?
│  ├─ YES → MUST HAVE (unless waivers possible)
│  └─ NO → Continue...
│
├─ Does it enable primary use case (diagnosis + prognosis)?
│  ├─ YES → MUST HAVE
│  └─ NO → Continue...
│
├─ Does it impact security/compliance?
│  ├─ YES → MUST HAVE (or critical SHOULD)
│  └─ NO → Continue...
│
├─ Would clinicians expect it (clinical best practice)?
│  ├─ YES → SHOULD HAVE
│  └─ NO → Continue...
│
├─ Is it essential for Navify integration?
│  ├─ YES → MUST HAVE
│  └─ NO → Continue...
│
├─ Does it significantly improve UX but not critical?
│  ├─ YES → SHOULD HAVE
│  └─ NO → Continue...
│
├─ Is it novel, high-effort, or research-oriented?
│  ├─ YES → COULD HAVE
│  └─ NO → Continue...
│
├─ Is it out of scope for this project phase?
│  ├─ YES → WON'T HAVE (defer to v2.0+)
│  └─ NO → Continue...
│
└─ Default → COULD HAVE (unless objections)
```

---

## TRACEABILITY: Requirements → Design → Code → Tests → Validation

### Example: Requirement FR-MIP-015 (Confidence Intervals)

```
┌─ FR-MIP-015 (Requirement)
│  "System shall compute confidence intervals (95% CI) for all 3 outputs via Monte Carlo dropout"
│
├─ Design Document (Section 4.2.3)
│  "Implement MC Dropout layer in model:
│   - Enable dropout at inference (stochastic forward passes)
│   - Draw 100 forward passes per input
│   - Compute percentiles: [2.5%, 97.5%] for confidence bounds"
│
├─ Code Module
│  File: src/models/neurofusion.py
│  Function: compute_uncertainty(logits, n_samples=100)
│  - Monte Carlo dropout (100 stochastic passes)
│  - Percentile computation (numpy.percentile)
│  - Return: mean, lower_ci, upper_ci
│
├─ Unit Test
│  File: tests/test_uncertainty.py
│  Test Case: test_mc_dropout_produces_ci()
│  - Input: synthetic batch (n=32)
│  - Expected: CI width 0.10-0.20 (depends on output variance)
│  - Status: PASS
│
├─ Integration Test
│  File: tests/test_api_uncertainty.py
│  Test Case: test_fhir_output_includes_ci()
│  - Input: FHIR Bundle with patient data
│  - Expected: RiskAssessment resource includes "range" field (CI bounds)
│  - Status: PASS
│
├─ System Test
│  File: tests/test_system_uncertainty.py
│  Test Case: test_high_uncertainty_flag()
│  - Scenario: Patient with borderline biomarkers (ambiguous case)
│  - Expected: Wide CI (confidence <60%) → flag for clinician
│  - Status: PASS
│
└─ Clinical Validation
   "Blind clinician review: CI widths correctly convey model uncertainty"
   Result: 95% agreement (clinicians find CIs interpretable)
```

---

## COMMUNICATION PLAN: Requirements Stakeholder Updates

### Weekly (Scrum Team)
- Sprint planning review which requirements are in-scope
- Daily standup: blockers related to requirements
- Sprint review: demo which requirements implemented

### Bi-Weekly (Project Steering Committee)
- Roadmap update: progress on MUST → SHOULD → COULD
- Risk review: any requirements at risk of missing timeline
- Scope change requests: evaluate against MoSCoW framework

### Monthly (Roche Executive Sponsor)
- Phase progress: % of requirements implemented by category
- Clinical validation status: tracking toward AUC ≥0.85
- Regulatory status: FDA/MDR timeline on track

### Phase Gate Reviews (Months 4, 10, 16)
- Phase 1 Exit: 100% of MUST requirements implemented + unit tested
- Phase 2 Exit: Clinical validation results (AUC, RMSE, C-index)
- Phase 3 Exit: FDA/MDR submissions filed, Navify integration verified

---

## APPENDIX: Requirement Traceability Matrix (Sample)

| Req ID | Title | Phase | Priority | Owner | Status | Test Case | Comments |
|--------|-------|-------|----------|-------|--------|-----------|----------|
| FR-DIP-001 | Accept FHIR Observation (pTau-217, Aβ42/40, NfL) | 1 | MUST | Data Eng | Done | TC-DIP-001 | Roche assay codes mapped |
| FR-DIP-020 | Patient pseudonymization | 1 | MUST | Data Eng | In Progress | TC-DIP-020 | SHA-256 hash function selected |
| FR-MIP-015 | Confidence intervals (95% CI) | 2 | MUST | ML Res Eng | Not Started | TC-MIP-015 | MC Dropout implemented in Phase 2 |
| FR-API-001 | POST /fhir/RiskAssessment/$process | 1 | MUST | DevOps | In Progress | TC-API-001 | FastAPI skeleton done; FHIR validation WIP |
| NFR-PERF-001 | p95 latency <2.0s | 3 | MUST | ML Arch | Not Started | TC-PERF-001 | Optimization in Phase 3 |
| FR-CDS-009 | Alert threshold customization | 3 | SHOULD | DevOps | Backlog | TC-CDS-009 | v1.1 feature; deferred |
| Feature 13 | Clinician web dashboard | 3 | COULD | DevOps | Backlog | - | v2.0 timeline |

---

**END OF MOSCOW PRIORITIZATION SUMMARY**

*This document should be reviewed at each phase gate and updated based on scope changes.*
