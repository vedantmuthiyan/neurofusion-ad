# NeuroFusion-AD: Requirements Documentation Suite - Index & Navigation

**Date**: February 15, 2026  
**Version**: 1.0  
**Status**: Ready for Phase 1 Development  
**Audience**: Project Managers, Tech Leads, Stakeholders

---

## OVERVIEW

This document suite contains comprehensive requirements analysis for NeuroFusion-AD, derived from three source documents:
1. **Roche_AI_Algorithm_Acquisition_Strategy.pdf** - Strategic context and market positioning
2. **PROJECT_OVERVIEW_MASTER.md** - Technical architecture and development roadmap  
3. **PHASE_123_EXECUTION_GUIDE.md** - Week-by-week execution plan with deliverables

The analysis synthesizes these materials into four actionable requirements documents for the 16-month development cycle.

---

## THE FOUR DOCUMENTS IN THIS SUITE

### 📋 DOCUMENT 1: Comprehensive Requirements Document
**File**: `NEUROFUSION_AD_COMPREHENSIVE_REQUIREMENTS.md`  
**Length**: ~70 pages (printed)  
**Purpose**: Authoritative requirements specification for all design, implementation, and testing

**Key Sections**:
1. **Project Vision & Business Case** (Section 1)
   - Strategic positioning vs. Siemens/Philips/Evidencio
   - Market opportunity: Liquid biomarkers + digital screening
   - Revenue model: Reagent pull-through (100-500K tests/year = $20-100M)

2. **Clinical Workflow Scenarios** (Section 2)
   - **Scenario 1: Primary Care Triage** - 2-minute digital screening identifies high-risk MCI patients
   - **Scenario 2: Neurologist Staging** - Multimodal fusion for prognostication + treatment planning
   - **Scenario 3: Treatment Monitoring** - Remote monitoring for response assessment

3. **Current Pain Points** (Section 3)
   - 80% of MCI patients never diagnosed (identification gap)
   - 6-12 month neurology wait times (access bottleneck)
   - 30% of MCI patients revert to normal (prognostication uncertainty)

4. **Desired Features & Capabilities** (Section 4)
   - 15 major features (FHIR integration, GNN model, real-time alerts, explainability, etc.)
   - Grouped by clinical value: data ingestion, inference, output, API, alerts, monitoring

5. **Constraints & Non-Negotiable Requirements** (Section 5)
   - FDA 510(k) De Novo compliance (IEC 62304, ISO 14971)
   - EU MDR Class IIa (Technical File, Notified Body review)
   - Navify Algorithm Suite compatibility (FHIR R4 contract)
   - Clinical validation threshold (AUC ≥0.85, RMSE ≤3.0, C-index ≥0.75)
   - 16-month timeline, $1.44M budget, 6-person team

6. **Raw Requirements List** (Section 6)
   - **103 total requirements** spanning:
     - Functional: 75 requirements (data ingestion, model, output, API, alerts)
     - Non-Functional: 78 requirements (performance, security, compliance, reliability)
   - Organized by category with Requirement IDs (FR-xxx, NFR-xxx)
   - Each requirement includes acceptance criteria and justification

7. **MoSCoW Priority Analysis** (Section 7)
   - **40 MUST HAVE** (FDA/MDR essential): Core FHIR/API, security, compliance, GNN
   - **35 SHOULD HAVE** (Expected by clinicians): Async processing, Epic integration, monitoring
   - **15 COULD HAVE** (Nice-to-have): Dashboards, remote monitoring, imaging
   - **13 WON'T HAVE** (Out of scope v1.0): Streaming, federated learning, phone apps
   - Clear rationale for each deferral decision

**When to Use**:
- Requirements traceability (link design to requirements)
- Test plan development (what to test)
- Change impact analysis (how does change affect scope)
- Regulatory compliance documentation (FDA/MDR submission)

**Key Metric**: 103 requirements total; ~40 MUST HAVE for v1.0, ~35 SHOULD for v1.1

---

### 📊 DOCUMENT 2: MoSCoW Prioritization Summary
**File**: `MOSCOW_PRIORITIZATION_SUMMARY.md`  
**Length**: ~25 pages (printed)  
**Purpose**: Executive summary with visual prioritization matrices and quick reference

**Key Sections**:
1. **Quick Statistics**
   - 40 MUST, 35 SHOULD, 15 COULD, 13 WON'T (103 total)
   - Phase allocation: Phase 1 = 35 reqs, Phase 2 = 45 reqs, Phase 3 = 23 reqs

2. **Requirement Breakdown by Category**
   - Data Ingestion & Preprocessing: 20 requirements (8 MUST, 3 SHOULD)
   - Model Inference & Prediction: 30 requirements (7 MUST, 8 SHOULD)
   - Output & Reporting: 20 requirements (4 MUST, 4 SHOULD)
   - Clinical Decision Support: 15 requirements (2 MUST, 3 SHOULD)
   - API & Integration: 25 requirements (6 MUST, 6 SHOULD)
   - Performance: 10 requirements (1 MUST, 2 SHOULD)
   - Security & Compliance: 35 requirements (8 MUST, 5 SHOULD)
   - Reliability & Monitoring: 18 requirements (2 MUST, 8 SHOULD)
   - Scalability & Interoperability: 15 requirements (0 MUST, 8 SHOULD)
   - Explainability: 10 requirements (2 MUST, 4 SHOULD)

3. **Phase-Based Allocation**
   - Phase 1 (Months 1-4): 35-40 requirements, 500 hours, regulatory foundation
   - Phase 2 (Months 5-10): 45-50 requirements, 1250 hours, model training & validation
   - Phase 3 (Months 11-16): 23-30 requirements, 1470 hours, integration & deployment

4. **Risk-Based Prioritization**
   - High-risk MUST HAVE: FHIR API, AUC ≥0.85, security compliance, IEC 62304, dataset access
   - Mitigation strategies for each critical requirement

5. **Effort Estimation**
   - Total: 3220 hours (vs. 3200 budgeted at 50 hrs/week × 16 weeks × 6 FTE)
   - Allocation: Phase 1 (500h), Phase 2 (1250h), Phase 3 (1470h)
   - FTE breakdown by role (ML Architect, Data Eng, Research Eng, Clinical, Regulatory, DevOps)

6. **Dependency Graph & Critical Path**
   - Data access (ADNI, Bio-Hermes) is critical path item Week 1-8
   - Model training (150 epochs) is critical path item Week 17-28
   - Navify API integration & security audit critical for Phase 3 completion
   - FDA/MDR submission blocked by DHF completion (Week 60)

7. **Stakeholder vs. MoSCoW Mapping**
   - Roche priorities → MUST HAVE (Navify, pTau-217 pullthrough, FDA/MDR, explainability)
   - Clinician priorities → SHOULD HAVE (monitoring, Epic integration, thresholds)
   - Hospital IT → SHOULD HAVE (async, Epic, scaling, observability)

8. **V1.0 vs. V1.1 Feature Roadmap**
   - V1.0 (Month 16): FHIR API, GNN model, alerts, security, FDA submission
   - V1.1 (Month 18-22): Async, PDF, dashboard, Epic SmartText, monitoring, scaling
   - V2.0 (Year 2-3): Mobile app, imaging, federated learning, causal inference

9. **Decision Framework**: Template for evaluating new requirements against MoSCoW

10. **Traceability Matrix Sample**: How requirements link to design, code, tests, validation

**When to Use**:
- Bi-weekly steering committee meetings (status updates)
- Phase gate reviews (what constitutes completion)
- Stakeholder communication (why something is deferred)
- Resource planning (effort estimation by phase)

**Key Metric**: 40% of requirements are MUST HAVE (non-negotiable for FDA/Roche)

---

### 🎯 DOCUMENT 3: Requirements Management Templates
**File**: `REQUIREMENTS_MANAGEMENT_TEMPLATES.md`  
**Length**: ~30 pages (printed)  
**Purpose**: Operational tools for ongoing requirements management throughout development

**Key Sections**:
1. **Template 1: Requirement Specification Card**
   - Use for each individual functional requirement
   - Includes: Description, acceptance criteria, dependencies, test cases, status, owner
   - Example: FR-DIP-001 (FHIR Observation acceptance)

2. **Template 2: Requirement Change Request (RCR)**
   - Use when requirements change mid-project
   - Captures: Change description, justification, impact assessment, approval decision
   - Example: Adding HL7 v2.x support (impacts Phase 3, 40 hours effort)

3. **Template 3: Requirements Traceability Matrix (RTM)**
   - Links each requirement to: SRS section, SAD module, code module, unit test, integration test, system test, clinical validation
   - Use for Phase gate reviews and regulatory audit
   - Example: 10-requirement sample showing 90% completion status

4. **Template 4: Stakeholder Requirement Priorities Map**
   - Quantifies which stakeholders care most about each requirement
   - Scores: A=Roche, B=Clinicians, C=FDA, D=Hospital IT, E=DevOps
   - Example: FHIR API (importance 4.2/5 across all stakeholders)

5. **Template 5: Requirements Impact Assessment**
   - Use when making scope/schedule/budget trade-offs
   - Compares options: extend timeline, cut SHOULD features, reduce MUST features
   - Example: 2-week schedule delay → choose to cut 7 SHOULD features instead

6. **Visual Workflow: Requirements Lifecycle**
   - Phase 1: Elicitation → SRS writing → approval (Weeks 1-3)
   - Phase 2: Design → Implementation → Testing (Weeks 4-52)
   - Phase 3: Validation → Traceability Matrix → Regulatory submission (Weeks 40-64)
   - Post-Launch: Monitoring & v1.1 planning (Months 17-24)

7. **Requirements Dashboard (Sample Metrics)**
   - Completion status by priority (MUST/SHOULD/COULD)
   - Progress by phase (Phase 1: 17%, Phase 2: 0%, Phase 3: 0%)
   - Critical path items and risks
   - Weekly update template for steering meetings

**When to Use**:
- Daily: Team updates on requirement completion
- Weekly: Status metrics dashboard
- Per-requirement: Specification cards (development guidance)
- Per-change: RCR form (scope management)
- Per-gate: RTM (verification that requirements met)

**Key Benefit**: Operationalizes requirements management as ongoing daily practice

---

### 🗓️ DOCUMENT 4: Execution Timeline (From Source Document 3)
**Reference**: `PHASE_123_EXECUTION_GUIDE.md`  
**Length**: ~80 pages (detailed week-by-week guide)  
**Purpose**: Concrete, day-by-day execution plan linking requirements to deliverables

**Key Sections** (Summary):

**Phase 1 (Months 1-4): Foundation & Architecture**
- Week 1: Kickoff, GPU setup, project structure
- Weeks 2-3: SRS v1.0 authoring (50-100 requirements)
- Weeks 2-3: SAD v1.0 development (architecture specification)
- Week 4: Risk analysis (FMEA, mitigation strategies)
- Weeks 5-8: ADNI preprocessing, digital biomarker synthesis
- Weeks 8-12: Model architecture implementation (encoders, attention, GNN)
- Week 16: Phase 1 exit gate (SRS/SAD/RMF approved, prototype 70%+ accuracy)

**Phase 2 (Months 5-10): Model Training & Validation**
- Weeks 17-28: Full-scale training (150 epochs on ADNI, p3.8xlarge GPU)
- Weeks 28-34: Hyperparameter optimization (Optuna 50 trials)
- Weeks 34-38: Bio-Hermes fine-tuning (if dataset available)
- Weeks 28-40: Clinical validation (AUC 0.86, RMSE 2.8, C-index 0.76)
- Weeks 28-40: Explainability validation (SHAP, attention weights)
- Weeks 40-50: External validation cohort testing
- Week 40: Phase 2 exit gate (AUC ≥0.85 validated, Clinical Validation Report complete)

**Phase 3 (Months 11-16): Integration, Deployment & Regulatory**
- Weeks 41-48: Navify FHIR API integration and testing
- Weeks 49-56: Docker containerization + Kubernetes deployment
- Weeks 52-64: Security audit + penetration testing
- Weeks 52-60: DHF compilation (200+ pages of regulatory documentation)
- Weeks 60-64: FDA De Novo + MDR technical file submissions
- Weeks 57-64: Load testing (1000 concurrent requests, p95 <2.0s latency)
- Week 64: Phase 3 exit gate (FDA/MDR submissions filed, deployable system ready)

**Post-Launch (Months 17-24): Pilot & Commercial**
- Months 17-18: Pilot deployment to 5 beta sites
- Months 19-21: Full commercial launch on Navify Algorithm Suite
- Months 22-24: Post-market surveillance + v1.1 feature development

**Key Documents Generated** (Links to Requirements):
- SRS v1.0 (Weeks 2-3): Formalizes all 50-100 raw requirements → 50 functional requirements
- SAD v1.0 (Weeks 3-4): Maps requirements to architecture components
- RMF v1.0 (Week 4): Risk management for regulatory compliance
- DHF (Weeks 52-60): Compiles evidence that all requirements met (traceability matrix)

**When to Use**:
- Weekly team standups (are we on track for this week's milestones?)
- Dependency management (what blocks this requirement from starting?)
- Effort tracking (budget remaining for this requirement?)
- Burn-down charts (% of week's requirements completed)

**Key Benefit**: Converts abstract 16-month project into concrete weekly milestones

---

## HOW TO USE THESE FOUR DOCUMENTS

### For Project Managers / Scrum Masters

**Week 1 (Kickoff)**:
1. Read: Project Vision & Business Case (Doc 1, Section 1)
2. Read: MoSCoW Summary (Doc 2, statistics & decision framework)
3. Distribute: Requirement specification cards (Doc 3, Template 1) to team
4. Action: Create week-by-week milestones from execution guide (Phase timing)

**Weekly (Status)**:
1. Use: Requirements Management Dashboard (Doc 3, Section 7) to track % complete
2. Update: Which requirements completed, which are blocked, which are at-risk
3. Reference: Execution timeline (Doc 4) for this week's planned deliverables
4. Escalate: Any risks to critical path items (ADNI access, API spec, dataset quality)

**Phase Gates (Months 4, 10, 16)**:
1. Verify: Requirements Traceability Matrix (Doc 3, Template 3) shows 100% of MUST HAVE verified
2. Confirm: All acceptance criteria met (Doc 1, Section 6 acceptance criteria)
3. Approve: Phase exit (no phase can progress without prior phase MUST HAVE completion)
4. Plan: Next phase requirements (Doc 2, phase allocation)

---

### For Technical Leads / Architects

**Architecture Design (Weeks 1-4)**:
1. Read: Desired Features & Capabilities (Doc 1, Section 4) → understand clinical context
2. Read: Constraints & Non-Negotiables (Doc 1, Section 5) → understand hard constraints
3. Read: Technical Architecture (Doc 1, Tables describing model, data flows, API)
4. Create: SAD v1.0 that maps each requirement to architecture component
5. Use: Requirement specification cards (Doc 3, Template 1) as design input

**Implementation Planning**:
1. Review: Raw requirements list by category (Doc 1, Section 6) → understand scope
2. Prioritize: Focus on MUST HAVE (Doc 2, Table of 40 MUST requirements)
3. Defer: SHOULD HAVE to v1.1 if schedule pressure (Doc 2, v1.0 vs. v1.1 roadmap)
4. Estimate: Effort by requirement (Doc 2, Phase-based allocation tables)

**Code Review / Acceptance**:
1. Check: Each PR references requirement ID (FR-001, NFR-003, etc.)
2. Verify: Code meets acceptance criteria (Doc 1, Section 6 for each requirement)
3. Ensure: Unit tests exist (Doc 3, Template 1 test case examples)
4. Link: Code to traceability matrix (Doc 3, Template 3)

---

### For Regulatory / Compliance Officers

**SRS/SAD Authoring (Weeks 2-4)**:
1. Organize: 50-100 raw requirements → 75 functional + 78 non-functional (Doc 1, Section 6)
2. Structure: By category (data ingestion, model, output, API, security, compliance)
3. Reference: Constraints section (Doc 1, Section 5) for compliance requirements
4. Document: IEC 62304 compliance (state which software lifecycle phase each addresses)

**Risk Management (Week 4)**:
1. Identify: Top hazards from clinician perspective (Doc 1, Section 3 pain points)
2. Analyze: FMEA for each hazard (example in Phase 1 execution guide)
3. Mitigate: Design controls for high-risk requirements (e.g., FR-MIP-015 for AUC validation)
4. Document: Risk Management File (RMF) v1.0

**FDA/MDR Submission (Weeks 52-64)**:
1. Compile: Design History File (DHF) with evidence of requirements compliance
2. Include: SRS v1.0, SAD v1.0, RMF v1.0 (all design specifications)
3. Provide: Requirements Traceability Matrix (Doc 3, Template 3) linking requirement → test → validation
4. Demonstrate: Clinical Validation Report showing requirements met (AUC 0.85, RMSE 3.0, C-index 0.75)

---

### For Clinical Stakeholders / KOLs

**Understanding Clinical Value (Week 1)**:
1. Read: Clinical Workflow Scenarios (Doc 1, Section 2) → see how algorithm used in practice
2. Review: Desired Features (Doc 1, Section 4) → understand capabilities from clinician perspective
3. Understand: Pain Points (Doc 1, Section 3) → why algorithm needed
4. See: SHOULD HAVE features (Doc 2) → what's expected by end-users

**Influencing Design (Weeks 2-3)**:
1. Attend: JAD workshop (Doc 1, Section 6 mentions week 2) → voice clinical requirements
2. Review: Acceptance criteria for clinical requirements (Doc 1, Section 6 for each FR)
3. Validate: That clinical validation tests are rigorous (blind clinician review, >80% agreement)
4. Defer: Nice-to-have features to v1.1 vs. MUST HAVE core requirements

**Clinical Validation (Weeks 28-40, Phase 2)**:
1. Review: 50 case studies generated by algorithm
2. Evaluate: Explainability (SHAP, attention weights) comprehensibility
3. Assess: Clinical utility (would you use this in practice?)
4. Sign-off: Clinical Validation Report (gates Phase 2 completion)

---

### For Stakeholder Executives (Roche Sponsor)

**Month 1 (Foundation Phase)**:
- Milestone: SRS v1.0 approved (converts raw requirements to formal specifications)
- Status: MUST HAVE requirements defined (40 critical requirements documented)
- Confidence: ~70% of Roche vision captured in specifications
- Risk: None; regulatory framework being established

**Month 4 (Phase 1 Exit)**:
- Milestone: Architecture approved, prototype 70%+ accuracy
- Status: All MUST HAVE requirements traced to design
- Budget: On track ($320K spent of $320K budget)
- Timeline: All Phase 1 deliverables delivered on schedule

**Month 10 (Phase 2 Exit)**:
- Milestone: AUC ≥0.85 validated, Clinical Validation Report complete
- Status: All MUST HAVE + priority SHOULD HAVE requirements validated
- Budget: $540K spent (within 15% contingency)
- Timeline: Ready for Phase 3 integration

**Month 16 (Phase 3 Exit / FDA Submission)**:
- Milestone: FDA De Novo + MDR applications filed
- Status: 100% of MUST HAVE requirements verified (RTM at 100%)
- Budget: $1.44M total spent (on budget)
- Timeline: Deployable system ready; pending FDA/MDR approvals (expected 6-12 months)
- Next: Roche due diligence + acquisition discussions begin

**Business Impact**: Requirements traceability demonstrates product meets:
- ✅ FDA regulatory requirements (IEC 62304, ISO 14971)
- ✅ Roche technical requirements (Navify integration, FHIR API, reagent pull-through)
- ✅ Clinician expectations (explainability, accuracy, workflow integration)
- ✅ Market differentiation vs. competitors (GNN multimodal, not simple calculators)

---

## QUICK REFERENCE: WHERE TO FIND WHAT

| Question | Answer Location |
|----------|-----------------|
| "What's the project about?" | Doc 1, Section 1 (Vision & Business Case) |
| "How will the algorithm be used?" | Doc 1, Section 2 (Clinical Workflow Scenarios) |
| "Why do we need this?" | Doc 1, Section 3 (Pain Points) |
| "What are the features?" | Doc 1, Section 4 (Desired Features) |
| "What are constraints?" | Doc 1, Section 5 (Constraints) |
| "What are the requirements?" | Doc 1, Section 6 (Raw Requirements List) |
| "What's MUST vs. SHOULD vs. COULD?" | Doc 1, Section 7 + Doc 2 (MoSCoW Analysis) |
| "How many requirements total?" | Doc 2 (Quick Statistics: 103 total) |
| "What's the deadline?" | Doc 2 (Phase-Based Allocation: Month 16 FDA submission) |
| "How much budget?" | Doc 2 (Risk-Based Prioritization: $1.44M total) |
| "What's the team?" | Doc 1, Section 5 (6 FTE team structure) |
| "How do I track requirements?" | Doc 3 (Templates: spec cards, RTM, dashboard) |
| "What happens each week?" | Doc 4 (Phase 1-3 Execution Guide, week-by-week) |
| "When is Phase X due?" | Doc 2 (Phase timeline) + Doc 4 (detailed schedule) |
| "How do I know if we met requirements?" | Doc 3, Template 3 (Requirements Traceability Matrix) |
| "What if something changes?" | Doc 3, Template 2 (Change Request Process) |
| "Is requirement X MUST or SHOULD?" | Doc 2 (MoSCoW breakdown tables) |
| "How do I prioritize competing requirements?" | Doc 3, Template 4 (Stakeholder Priorities) |
| "What are the clinical workflows?" | Doc 1, Section 2 (3 detailed scenarios) |

---

## SUMMARY: THE COMPLETE PICTURE

| Dimension | Scope | Reference |
|-----------|-------|-----------|
| **What** | 103 total requirements (40 MUST, 35 SHOULD, 15 COULD, 13 WON'T) | Doc 1, Section 6 + Doc 2 |
| **Why** | Diagnose Alzheimer's earlier via multimodal (biomarker+digital) screening | Doc 1, Section 1-3 |
| **Who** | 6-person team: ML Architect, Data Eng, Research Eng, Clinical, Regulatory, DevOps | Doc 1, Section 5 |
| **How** | 3 phases (foundation, training, integration), 16 months, $1.44M budget | Doc 2 + Doc 4 |
| **When** | Phase 1 (Mo 1-4), Phase 2 (Mo 5-10), Phase 3 (Mo 11-16), FDA/MDR submission Mo 16 | Doc 4 |
| **Where** | Deployed to Navify Algorithm Suite (cloud + edge deployment options) | Doc 1, Section 1 |
| **Success** | FDA/MDR approval + Roche acquisition within 18 months | Doc 1, Section 1 |

---

## HOW TO KEEP THESE DOCUMENTS UPDATED

**Weekly**:
- Update requirements completion status (Doc 3, Template 7 dashboard)
- Log any requirement changes (Doc 3, Template 2 RCR form)
- Track blockers/risks against critical path items (Doc 2 risk table)

**Per-Phase**:
- Phase gate review: Verify all prior phase MUST HAVE requirements met
- Update MoSCoW allocation as scope changes (Doc 2)
- Communicate any deferred requirements to stakeholders

**Per-Requirement-Implemented**:
- Create traceability entry (Doc 3, Template 3 RTM)
- Update status (planned → in-progress → testing → verified)
- Link to code (GitHub PR), tests (pytest), and validation (clinical review)

**Monthly** (Steering Committee):
- Review requirements completion dashboard (Doc 3)
- Report on MUST HAVE priority vs. schedule
- Communicate any trade-offs (defer SHOULD to v1.1, etc.)

---

## APPROVAL & SIGN-OFF

This requirements documentation suite represents the authoritative specification for NeuroFusion-AD development. All design, implementation, testing, and validation activities must be traceable back to requirements defined in this suite.

**Stakeholders to Approve**:
- ☐ Project Sponsor (Roche): Confirms business requirements captured
- ☐ Technical Lead (ML Architect): Confirms feasible within 16 months
- ☐ Regulatory Officer: Confirms FDA/MDR compliance pathway clear
- ☐ Clinical Advisor: Confirms clinical workflows realistic and valuable
- ☐ DevOps Lead: Confirms deployment/operations requirements feasible

**Approval Sign-Off** (to be completed Week 2, before SRS final approval):

Sponsor: _________________ Date: _______
Tech Lead: _________________ Date: _______
Regulatory: _________________ Date: _______
Clinical: _________________ Date: _______
DevOps: _________________ Date: _______

---

**END OF REQUIREMENTS DOCUMENTATION SUITE INDEX**

*For detailed questions on any requirement, refer to the specific document and section noted above.*
*For day-to-day requirements management, use Doc 3 Templates in conjunction with Doc 4 Execution Guide.*
*For regulatory submission, use Doc 1 complete specifications + Doc 3 RTM as primary artifacts.*
