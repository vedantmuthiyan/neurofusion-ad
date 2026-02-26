# NeuroFusion-AD: Requirements Management Templates & Visual Workflows

**Document Type**: Templates for ongoing requirements management  
**Audience**: Project managers, scrum masters, requirement traceability leads  
**Purpose**: Operational requirements tracking throughout 16-month development

---

## TEMPLATE 1: Requirement Specification Card

### Use This Template for Each Functional Requirement

```
╔══════════════════════════════════════════════════════════════════╗
║                    REQUIREMENT SPECIFICATION CARD               ║
╚══════════════════════════════════════════════════════════════════╝

REQ ID:                FR-DIP-001
TITLE:                 Accept FHIR Observation (pTau-217, Aβ42/40, NfL)
CATEGORY:              Data Ingestion & Preprocessing
PHASE:                 Phase 1 (Months 1-4)
MOSCOW PRIORITY:       ⭐ MUST HAVE
OWNER:                 Data Engineer
STAKEHOLDER:           Roche, Clinicians
STATUS:                🟢 In Progress [Est. 2/28/2026]

──────────────────────────────────────────────────────────────────

DESCRIPTION:
System shall accept FHIR R4 Observation resources containing Roche 
Elecsys biomarker results. Specifically:
- pTau-217 (Phosphorylated Tau-217) in pg/mL
- Aβ42/40 ratio (Amyloid-Beta 42/40)
- NfL (Neurofilament Light) in pg/mL

FORMAT:
System shall parse FHIR Observation with:
- code.coding: LOINC code (e.g., LOINC 96387-7 for pTau-217)
- value.quantity: numeric value + unit
- effectiveDateTime: test date
- specimen: reference to sample details

──────────────────────────────────────────────────────────────────

ACCEPTANCE CRITERIA:
✓ Parse valid FHIR Observation with all 3 biomarker types
✓ Extract numeric values with <0.1% precision loss
✓ Validate LOINC codes against Roche assay specification
✓ Store biomarker values in PostgreSQL with timestamp
✓ Reject malformed FHIR (not valid R4 schema)
✓ Log all parse attempts (success/failure) for audit

EXAMPLE INPUT:
{
  "resourceType": "Observation",
  "status": "final",
  "code": {
    "coding": [{
      "system": "http://loinc.org",
      "code": "96387-7",
      "display": "Phosphorylated tau (181) [Mass/volume] in Plasma"
    }]
  },
  "valueQuantity": {
    "value": 45.2,
    "unit": "pg/mL",
    "system": "http://unitsofmeasure.org",
    "code": "pg/mL"
  },
  "effectiveDateTime": "2026-02-15T10:30:00Z"
}

EXAMPLE OUTPUT (on success):
{
  "biomarker_id": "obs_12345",
  "patient_id": "hash_abc123",
  "biomarker_type": "pTau-217",
  "value": 45.2,
  "unit": "pg/mL",
  "test_date": "2026-02-15",
  "status": "accepted"
}

──────────────────────────────────────────────────────────────────

DEPENDENCIES:
- Requires: FHIR schema validator (Pydantic library)
- Requires: PostgreSQL database (configured & accessible)
- Requires: Roche LOINC code mapping document

BLOCKING ISSUES:
- None currently

RELATED REQUIREMENTS:
- FR-DIP-007: FHIR schema validation (prerequisite)
- FR-DIP-008: Z-score normalization (downstream)
- FR-API-001: API endpoint to submit Observation

──────────────────────────────────────────────────────────────────

EFFORT ESTIMATE:
Story Points:         13 (medium-high complexity)
Dev Days:             3-4 days
Testing Days:         1-2 days
Total Timeline:       5-6 days
Owner:                Data Engineer
Reviewer:             ML Architect

CODE LOCATION:
- Module: src/data/fhir_validator.py
- Test: tests/test_fhir_biomarker.py
- Docs: docs/api/biomarker_ingestion.md

──────────────────────────────────────────────────────────────────

TESTING PLAN:
┌─ Unit Tests (test_fhir_biomarker.py)
│  ├─ test_valid_ptau217_observation()
│  ├─ test_valid_amyloid_ratio_observation()
│  ├─ test_valid_nfl_observation()
│  ├─ test_malformed_fhir_rejection()
│  ├─ test_missing_loinc_code_rejection()
│  └─ test_value_precision_preservation()
│
├─ Integration Tests (test_api_biomarker_ingestion.py)
│  ├─ test_fhir_observation_via_api_endpoint()
│  ├─ test_database_insert_on_success()
│  └─ test_audit_log_creation()
│
└─ System Tests (test_system_ingestion.py)
   ├─ test_real_roche_cobas_output_parsing()
   └─ test_latency_<500ms()

──────────────────────────────────────────────────────────────────

REGULATORY NOTES:
- Requirement: IEC 62304 Software Development Lifecycle
- Validation: Must be documented in Design History File (DHF)
- Traceability: Link to SRS Section 5.2.1 (Data Ingestion Requirements)
- Compliance: HIPAA (no PII in values), FHIR R4 standard

──────────────────────────────────────────────────────────────────

REVISION HISTORY:
Version 1.0 | 2026-02-15 | Initial creation | Data Engineer
Version 1.1 | TBD        | Updated after review | TBD

└──────────────────────────────────────────────────────────────────┘
```

---

## TEMPLATE 2: Requirement Change Request

### Use This Template When Requirements Change

```
╔══════════════════════════════════════════════════════════════════╗
║                  REQUIREMENT CHANGE REQUEST (RCR)               ║
╚══════════════════════════════════════════════════════════════════╝

RCR ID:                RCR-2026-0015
DATE SUBMITTED:        2026-02-15
SUBMITTED BY:          Roche Technical Liaison (Dr. Jane Smith)
PRIORITY:              🔴 HIGH
STATUS:                ⏳ PENDING REVIEW

──────────────────────────────────────────────────────────────────

AFFECTED REQUIREMENT:
FR-API-001: POST /fhir/RiskAssessment/$process endpoint

CHANGE DESCRIPTION:
Roche has requested that the API endpoint also accept HL7 v2.x 
ADT (Admit-Discharge-Transfer) messages in addition to FHIR 
bundles. This is to support legacy hospital systems that do not 
yet have FHIR capabilities (estimated 30% of target hospitals).

JUSTIFICATION:
"Many community hospitals still use HL7 v2.x for EHR messaging. 
Without v2.x support, NeuroFusion-AD cannot deploy to >50% of 
our target market (regional/rural hospitals). This is a market 
expansion requirement."

ESTIMATED IMPACT:
Development Effort:    +40 hours (HL7 message transformer)
Testing Effort:        +20 hours (HL7 test cases)
Timeline Impact:       +1 week (can be parallelized in Phase 3)
Cost Impact:           +$1.5K (external HL7 consultant)
Scope Impact:          MEDIUM (adds new API input format)

──────────────────────────────────────────────────────────────────

MOSCOW PRIORITY RECOMMENDATION:
Current: MUST HAVE (only FHIR in v1.0)
Proposed: SHOULD HAVE (HL7 v2.x in v1.1)

RATIONALE:
- v1.0 FHIR-only launch is simpler, easier to validate
- HL7 v2.x support can be added in v1.1 (Month 18-22) without 
  major rework (just new message transformer)
- De-prioritizing v2.x allows earlier Phase 3 completion & 
  faster time-to-market
- Roche can handle HL7→FHIR translation at their hub layer 
  (acceptable workaround for 4 months)

ROCHE ACCEPTANCE:
"Acceptable. Deploy v1.0 FHIR-only at academic sites. Add 
HL7 v2.x in v1.1 for broader rollout. This aligns with our 
phased deployment strategy."

──────────────────────────────────────────────────────────────────

DECISION:
☑ APPROVED (with MoSCoW downgrade to SHOULD)
☐ APPROVED (implement in Phase 3, extend timeline)
☐ DEFERRED (defer to v2.0, revisit after v1.0 launch)
☐ REJECTED (not aligned with project goals)

APPROVED BY:       Project Manager (John Doe)
DATE APPROVED:     2026-02-16
IMPLEMENTATION:    Phase 3, Weeks 52-55 (HL7 transformer)

──────────────────────────────────────────────────────────────────

ACTION ITEMS:
☑ Update SRS v1.1 (Section 5.2.1) with HL7 v2.x mention
☑ Create new requirement: FR-API-002b "Accept HL7 v2.x ADT"
☑ Schedule HL7 design review with Data Engineer
☑ Create integration test cases for HL7 messages
☑ Communicate change to team (standby meeting)
Assigned To: Regulatory Officer
Due Date:    2026-02-17

└──────────────────────────────────────────────────────────────────┘
```

---

## TEMPLATE 3: Requirements Traceability Matrix (RTM)

### Use This for Phase Gate Reviews and Regulatory Audit

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                         REQUIREMENTS TRACEABILITY MATRIX (RTM)                                       ║
║                            Phase 2 Exit Review (Month 10)                                           ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

REQ_ID  │ TITLE                           │ SRS  │ SAD  │ CODE  │ UT   │ IT   │ ST   │ CV   │ STATUS
        │                                 │ REF  │ REF  │ MODULE│ PASS │ PASS │ PASS │ PASS │
────────┼─────────────────────────────────┼──────┼──────┼───────┼──────┼──────┼──────┼──────┼──────
FR-DIP-001 │ Accept FHIR Observation     │5.2.1 │3.1.2 │DIP-01 │ ✅   │ ✅   │ ✅   │ ✅   │Done
FR-DIP-002 │ Accept FHIR Patient         │5.2.1 │3.1.2 │DIP-02 │ ✅   │ ✅   │ ✅   │ ✅   │Done
FR-DIP-020 │ Pseudonymization            │5.2.7 │3.1.5 │DIP-20 │ ✅   │ ✅   │ ✅   │ ✅   │Done
FR-MIP-001 │ 4 Modality Encoders         │5.3.1 │4.1.1 │GNN-01 │ ✅   │ ✅   │ ✅   │ ✅   │Done
FR-MIP-015 │ Confidence Intervals        │5.3.4 │4.1.4 │GNN-15 │ ✅   │ ✅   │ ✅   │ ✅   │Done
FR-OUT-001 │ FHIR RiskAssessment        │5.4.1 │3.2.1 │OUT-01 │ ✅   │ ✅   │ ✅   │ ✅   │Done
FR-API-001 │ /fhir/RiskAssessment/$     │5.5.1 │3.3.1 │API-01 │ ⏳    │ 🔄   │ 🔄   │ ⏳    │WIP
NFR-PERF-001 │ p95 latency <2.0s        │6.1.1 │4.2.1 │API-01 │ ⏳    │ 🔄   │ ⏳    │ ⏳    │WIP
NFR-SEC-001 │ HTTPS/TLS 1.3             │6.2.1 │3.4.1 │API-02 │ ✅   │ ✅   │ ✅   │ ✅   │Done
NFR-COMP-008 │ IEC 62304 compliance      │7.1.1 │All   │DHF    │ ✅   │ ✅   │ ✅   │ ✅   │Done

Legend:
✅  = Verified/Passed
🔄  = In Progress
⏳  = Not Started
❌  = Failed (remediation in progress)

SRS = Software Requirements Spec document section
SAD = Software Architecture Document section
UT = Unit Test
IT = Integration Test
ST = System Test
CV = Clinical Validation

SUMMARY:
- Total Requirements Reviewed: 10 (sample)
- Complete (All tests passed): 9/10 (90%)
- In Progress: 1/10 (10%)
- Failed: 0/10 (0%)

PHASE 2 EXIT GATE STATUS: 🟢 APPROVED (9/10 MUST HAVE requirements verified)

Remaining work: FR-API-001 latency optimization (ongoing, expect completion by Week 38)
```

---

## TEMPLATE 4: Stakeholder Requirement Priorities Map

### Use This During Requirements Gathering Phase

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║              STAKEHOLDER REQUIREMENT PRIORITIES MATRIX                            ║
║                          Phase 1 (Month 1, Week 2)                               ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

Stakeholder Groups:
A = Roche (Product Sponsor + Customer)
B = Clinicians (End Users - Neurologists & PCPs)
C = FDA/Regulatory (Compliance & Approval)
D = Hospital IT (System Integration)
E = DevOps/Security (Ops Team)

                                   PRIORITY IMPORTANCE ────→
                           A    B    C    D    E   Avg  Moscow
────────────────────────────────────────────────────────────────────
FHIR API Integration      5    3    4    5    4   4.2  MUST
Clinical Accuracy (AUC≥0.85)  5    5    5    2    3   4.0  MUST
Security & Encryption     4    2    5    3    5   3.8  MUST
Explainability (SHAP)     4    5    3    2    2   3.2  MUST
Model Deployment (Docker) 3    1    3    4    5   3.2  MUST
Auto-Scaling             2    1    2    4    5   2.8  SHOULD
Web Dashboard            3    4    1    3    2   2.6  SHOULD
Async API Processing     2    1    1    3    3   2.0  SHOULD
Epic SmartText Integration 2   2    1    4    2   2.2  SHOULD
HL7 v2.x Backward Compat. 1    1    1    3    1   1.4  SHOULD
Admin Panel/Config UI    1    1    1    3    3   1.8  COULD
Patient Mobile App       1    4    1    1    1   1.6  COULD
Federated Learning       0    0    1    0    0   0.2  WON'T

Scoring: 5=Critical | 4=Important | 3=Desired | 2=Nice | 1=Minor | 0=Not Important

KEY INSIGHTS:
1. FHIR API & Clinical Accuracy drive all stakeholder value (4.2, 4.0)
2. Roche (Sponsor) & Clinicians highly aligned on core features
3. Hospital IT emphasizes integration & infrastructure (API, Docker, scaling)
4. Security team has outsized importance for v1.0 (FDA requirement)
5. Patient app & federated learning are post-launch priorities
6. Recommendation: Use this matrix to sequence Phase 1 requirements

PHASE PRIORITIZATION:
Phase 1 (Foundation):    FHIR, Security, Model Core, Deployment (Avg >3.5)
Phase 2 (Validation):    Accuracy, Explainability, Monitoring (Avg >3.0)
Phase 3 (Integration):   Hospital Integration, Scaling, Epic (Avg >2.5)
v1.1+ (Polish):          Dashboard, Admin, Apps (Avg <2.5)
```

---

## TEMPLATE 5: Requirements Impact Assessment

### Use When Making Trade-Offs (Scope, Schedule, Budget)

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                    REQUIREMENTS IMPACT ASSESSMENT                             ║
║              Question: Can we defer SHOULD HAVE features to v1.1?             ║
╚════════════════════════════════════════════════════════════════════════════════╝

SCENARIO:
Development is tracking 2 weeks behind schedule. Options:
A) Extend timeline by 2 weeks (delays FDA submission)
B) Cut SHOULD HAVE features (7 requirements: async, PDF, dashboard, Epic, etc.)
C) Reduce scope of MUST HAVE (e.g., single modality only)

ASSESSMENT:

Option A: EXTEND TIMELINE
Impact:
  ✗ Roche acquisition timeline slip (LOI by Month 15, now Month 17)
  ✓ All features included, v1.0 is comprehensive
  ✓ Less technical debt
Cost:   +$200K (2 additional weeks: 6 FTE × $50K/week)
Risk:   MEDIUM (acquisition deal may cool)
Recommendation: AVOID if possible

Option B: CUT SHOULD HAVE FEATURES
Impact:
  ✓ Maintain Phase 3 deadline (FDA submission Month 16)
  ✓ MUST HAVE requirements complete & validated
  ✗ Hospital IT frustrated (no Epic integration, manual config)
  ✗ Async processing not available (single-threaded API)
  ~ Dashboard deferred (use Navify UI as temporary)
Cost:   -$250K (avoided implementation effort)
Risk:   LOW (SHOULD features can be added Month 18 in v1.1)
Recommendation: ACCEPTABLE - aligns with MoSCoW philosophy

Option C: REDUCE MUST HAVE SCOPE
Impact:
  ✗ FDA may reject incomplete submission (modal reduction unvalidated)
  ✗ Clinical utility reduced (no decision support)
  ✗ Roche unwilling to deploy (doesn't meet contractual specs)
Cost:   -$300K (major scope reduction)
Risk:   CRITICAL (project failure risk)
Recommendation: NOT VIABLE

DECISION FRAMEWORK:
┌─ Maintain all MUST HAVE (non-negotiable for FDA/Roche)
├─ Cut SHOULD HAVE features as needed to meet deadline
├─ Defer COULD HAVE to v2.0 without discussion
└─ Communicate changes to stakeholders immediately

SELECTED OPTION: B (Cut SHOULD HAVE features)

CHANGE MANAGEMENT:
- Notify Roche: "FHIR API v1.0 is single-threaded; async in v1.1 (Month 20)"
- Notify Hospital IT: "Epic SmartText deferred to v1.1; use FHIR API for now"
- Update documentation: Mark affected requirements as "v1.1 candidate"
- Manage expectations: Position as "lean MVP" strategy

REVISED TIMELINE:
  Phase 1: Weeks 1-16 (on track)
  Phase 2: Weeks 17-40 (on track)
  Phase 3: Weeks 41-64 (recovers 2 weeks slack)
  FDA/MDR submission: Week 63 (2 weeks early buffer)
  v1.1 planning: Begins Month 17 (8 features deferred)

STAKEHOLDER COMMUNICATION:
Subject: NeuroFusion-AD Schedule Recovery Plan

Dear Stakeholders,

To maintain our Phase 3 deadline and FDA submission timeline, 
we are deferring the following SHOULD HAVE features to v1.1 
(Month 18-22):

Deferred Features (v1.1):
- Asynchronous API processing
- PDF report generation
- Web-based clinician dashboard
- Epic SmartText integration
- Alert threshold customization
- GDPR automation
- Advanced monitoring

These features will be available 2 months post-launch and do 
not impact the core diagnostic capability.

V1.0 Launch Remains: Month 16 (FDA/MDR submissions complete)

Questions? Contact Project Manager by 2026-02-18.

Best regards,
Project Management

└─────────────────────────────────────────────────────────────┘
```

---

## VISUAL WORKFLOW: Requirements Lifecycle

```
┏─────────────────────────────────────────────────────────────────────────────────┓
║                     REQUIREMENTS LIFECYCLE WORKFLOW                             ║
┗─────────────────────────────────────────────────────────────────────────────────┘

PHASE 1: ELICITATION & SPECIFICATION (Weeks 1-3)
                    ↓
  ┌─────────────────────────────────────┐
  │  JAD Workshop (Week 2)              │
  │  Stakeholders gather requirements   │
  │  → 50-100 raw requirements captured │
  └────────────────┬────────────────────┘
                   ↓
  ┌─────────────────────────────────────┐
  │  Write SRS v1.0 (Weeks 2-3)         │
  │  Organize raw reqs → functional     │
  │  → 50 formal requirements defined   │
  └────────────────┬────────────────────┘
                   ↓
  ┌─────────────────────────────────────┐
  │  SRS Review & Approval (Week 3)     │
  │  Stakeholder sign-off               │
  │  → SRS v1.0 APPROVED               │
  └────────────────┬────────────────────┘
                   ↓

PHASE 2: DESIGN & IMPLEMENTATION (Weeks 4-52)
                   ↓
  ┌─────────────────────────────────────┐
  │  Design Review (Week 4)             │
  │  Map requirements to architecture   │
  │  → SAD created (design specifics)   │
  └────────────────┬────────────────────┘
                   ↓
  ┌─────────────────────────────────────┐
  │  Development (Weeks 5-52)           │
  │  Implement requirements             │
  │  → Code modules created & reviewed  │
  │  → Each PR linked to requirement ID │
  └────────────────┬────────────────────┘
                   ↓
  ┌─────────────────────────────────────┐
  │  Unit Testing (Parallel)            │
  │  Test each requirement in isolation │
  │  → 80%+ code coverage achieved      │
  └────────────────┬────────────────────┘
                   ↓
  ┌─────────────────────────────────────┐
  │  Integration Testing (Parallel)     │
  │  Test requirement interactions      │
  │  → Integration test cases executed  │
  └────────────────┬────────────────────┘
                   ↓

PHASE 3: VALIDATION & VERIFICATION (Weeks 40-64)
                   ↓
  ┌─────────────────────────────────────┐
  │  System Testing (Week 50)           │
  │  Test complete system against reqs  │
  │  → System test results documented   │
  └────────────────┬────────────────────┘
                   ↓
  ┌─────────────────────────────────────┐
  │  Traceability Matrix (Week 55)      │
  │  Map: Requirement → Design → Code   │
  │        → Test → Validation          │
  │  → RTM completed (auditable)        │
  └────────────────┬────────────────────┘
                   ↓
  ┌─────────────────────────────────────┐
  │  Clinical Validation (Weeks 50-60)  │
  │  Blind clinician review of outputs  │
  │  → 3 neurologists validate 50 cases │
  └────────────────┬────────────────────┘
                   ↓
  ┌─────────────────────────────────────┐
  │  Design History File (Week 60)      │
  │  Compile all evidence of compliance │
  │  → 300+ page DHF assembled          │
  └────────────────┬────────────────────┘
                   ↓
  ┌─────────────────────────────────────┐
  │  FDA/MDR Submission (Week 64)       │
  │  Submit with complete traceability  │
  │  → Submissions filed                │
  └────────────────┬────────────────────┘
                   ↓

POST-LAUNCH: MONITORING & UPDATES (Months 17-24)
                   ↓
  ┌─────────────────────────────────────┐
  │  Post-Market Surveillance           │
  │  Monitor real-world requirement     │
  │  compliance (drift, errors)         │
  │  → Annual compliance report         │
  └────────────────┬────────────────────┘
                   ↓
  ┌─────────────────────────────────────┐
  │  v1.1 Planning (Month 18)           │
  │  Define SHOULD HAVE features        │
  │  → SRS v1.1 planning begins         │
  └─────────────────────────────────────┘

KEY GATES (Decision Points):
📌 Phase 1 Gate (Week 4): SRS approved? → Continue to Phase 2
📌 Phase 2 Gate (Week 40): AUC≥0.85? → Continue to Phase 3
📌 Phase 3 Gate (Week 64): FDA/MDR ready? → Launch v1.0
```

---

## REQUIREMENTS DASHBOARD (Sample Metrics)

### Use This During Bi-Weekly Steering Meetings

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                    REQUIREMENTS STATUS DASHBOARD                            ║
║                    As of: 2026-02-15 (End of Week 1)                        ║
╚═════════════════════════════════════════════════════════════════════════════╝

📊 REQUIREMENTS COMPLETION STATUS
┌─────────────────────────────────────────────┐
│ MUST HAVE: 40 requirements                  │
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │ 10% (4 of 40)
│                                              │
│ SHOULD HAVE: 35 requirements                │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  5% (2 of 35)
│                                              │
│ COULD HAVE: 15 requirements                 │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  0% (0 of 15)
│                                              │
│ TOTAL: 90 requirements (Phase 1-3)          │
│ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  7% (6 of 90)
└─────────────────────────────────────────────┘

✅ COMPLETED REQUIREMENTS (This Week):
  • FR-DIP-001: FHIR Observation acceptance (Data Eng)
  • FR-DIP-002: FHIR Patient resource (Data Eng)
  • NFR-SEC-001: HTTPS/TLS 1.3 (DevOps)
  • NFR-MAINT-001: Git version control (DevOps)
  • Project infrastructure setup (all)

🔄 IN PROGRESS REQUIREMENTS (This Week):
  • SRS v1.0 authoring (Regulatory Officer) - 60% complete
  • SAD v1.0 architecture design (ML Architect) - 40% complete
  • GPU infrastructure setup (DevOps) - 80% complete

⏳ NOT STARTED REQUIREMENTS:
  • 83 requirements (all others in Phase 2-3)

📈 PHASE PROGRESS
  Phase 1 (Target: 35 reqs, Week 16): 6/35 (17%) ✅ On Track
  Phase 2 (Target: 45 reqs, Week 40): 0/45 (0%) ⏳ Planned
  Phase 3 (Target: 23 reqs, Week 64): 0/23 (0%) ⏳ Planned

🎯 CRITICAL PATH ITEMS (Blocking Other Work):
  ⏳ ADNI dataset access (due Week 3) - In progress with Data Use Agr.
  ⏳ Roche API spec finalization (due Week 2) - Waiting for Roche input
  ⏳ ML Architect hiring (due Week 1) - Offer extended, awaiting acceptance

⚠️  RISKS & ISSUES:
  🔴 HIGH: Roche API spec delay (3-day schedule impact if delayed >1 week)
  🟡 MEDIUM: ADNI approval timing (typical 2 weeks, could be 4)
  🟡 MEDIUM: ML Architect hiring (backup candidate identified)

💡 RECOMMENDATIONS:
  1. Follow up with Roche by EOW for API spec (affects Phase 1 design)
  2. Escalate ADNI access request (data-critical path item)
  3. Finalize ML Architect hiring (key blocker for model implementation)

Prepared By: Project Manager
Contact:     project_manager@company.com
Next Update: 2026-02-22 (EOW 2)
```

---

**END OF REQUIREMENTS MANAGEMENT TEMPLATES**

*Print these templates and customize for your project. Update weekly/bi-weekly during development.*
