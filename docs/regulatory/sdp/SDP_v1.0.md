# Software Development Plan (SDP)

---

## Document Control

| Field | Value |
|---|---|
| **Document ID** | SDP-001 |
| **Version** | 1.0 |
| **Status** | APPROVED |
| **Product Name** | NeuroFusion-AD |
| **Product Version** | 1.0.0 |
| **Classification** | IEC 62304 Class B Medical Device Software |
| **Date** | 2025-01-15 |
| **Author** | Regulatory Affairs Office |
| **Reviewed By** | ML Architect, QA Lead, DevOps Lead |
| **Approved By** | Chief Medical Officer, VP Engineering |

---

## Revision History

| Version | Date | Author | Description | Change Control Ref |
|---|---|---|---|---|
| 0.1 | 2024-10-01 | RAO | Initial draft | CCR-001 |
| 0.2 | 2024-11-15 | ML Architect | Architecture sections added | CCR-008 |
| 0.3 | 2024-12-10 | QA Lead | Testing standards integrated | CCR-015 |
| 1.0 | 2025-01-15 | RAO | Approved baseline release | CCR-022 |

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Roles and Responsibilities](#2-roles-and-responsibilities)
3. [Development Lifecycle Model](#3-development-lifecycle-model)
4. [Development Standards](#4-development-standards)
5. [Software Configuration Management](#5-software-configuration-management)
6. [Infrastructure and Environments](#6-infrastructure-and-environments)
7. [Phase Timeline Summary](#7-phase-timeline-summary)
8. [Risk Management Reference](#8-risk-management-reference)
9. [Appendices](#9-appendices)

---

## 1. Purpose and Scope

### 1.1 Purpose

This Software Development Plan (SDP) establishes the governing framework for the development, verification, validation, maintenance, and lifecycle management of **NeuroFusion-AD** — a multimodal Graph Neural Network Clinical Decision Support (CDS) system for Alzheimer's Disease Progression Prediction. This document fulfills the planning requirements specified in:

- **IEC 62304:2006+AMD1:2015** — Medical Device Software: Software Life Cycle Processes (Clauses 4–9)
- **FDA 21 CFR Part 820** — Quality System Regulation
- **FDA Guidance: Software as a Medical Device (SaMD): Clinical Evaluation** (2017)
- **FDA De Novo Request Requirements** (predicate: Prenosis Sepsis ImmunoScore, DEN200080)
- **EU MDR 2017/745** — Annex I (GSPR), Annex IX/XI
- **ISO 14971:2019** — Application of Risk Management to Medical Devices
- **IEC 62366-1:2015** — Usability Engineering for Medical Devices

This SDP is a **living document** subject to formal change control. All amendments require documented approval per Section 5.4 (Change Control Process) before implementation.

### 1.2 Scope

#### 1.2.1 Software System Description

NeuroFusion-AD is a **Software as a Medical Device (SaMD)** that integrates four input modality encoders through a cross-modal attention mechanism and GraphSAGE Graph Neural Network to generate probabilistic predictions of Alzheimer's Disease progression risk in patients diagnosed with Mild Cognitive Impairment (MCI).

**Intended Use Population:** Adult patients aged 50–90 years with confirmed MCI diagnosis.

**Intended Users:** Licensed neurologists, geriatric psychiatrists, and trained clinical staff operating within qualified healthcare institutions.

**Clinical Output:**
- **Classification output:** Binary progression risk (AUC ≥ 0.85)
- **Regression output:** Cognitive decline rate estimate (RMSE ≤ 3.0 MMSE points/year)
- **Survival output:** Time-to-progression probability (C-index ≥ 0.75)

#### 1.2.2 System Architecture Boundaries

The scope of this SDP encompasses all software components defined within the NeuroFusion-AD system boundary:

| Component | Description | IEC 62304 Class |
|---|---|---|
| Fluid Biomarker Encoder | Processes pTau-217, Abeta42/40, NfL inputs | Class B |
| Acoustic Feature Encoder | Processes speech/acoustic biomarker features | Class B |
| Motor Signal Encoder | Processes motor assessment signals | Class B |
| Clinical/Demographic Encoder | Processes structured clinical data | Class B |
| Cross-Modal Attention Module | 768-dim, 8-head attention fusion | Class B |
| GraphSAGE GNN | 3-layer graph neural network | Class B |
| Multi-Task Output Head | Classification, regression, survival outputs | Class B |
| Input Validation Layer | Hard-constraint enforcement engine | Class B |
| FastAPI Inference Service | REST API serving predictions | Class B |
| Audit Trail Service | Immutable audit log generation | Class B |
| Configuration Management Service | System parameter management | Class B |

> **Note on Classification Justification:** NeuroFusion-AD is classified as **IEC 62304 Class B** because an injury to a patient is possible but not likely to be serious. The device outputs are advisory/CDS in nature and require clinician interpretation before any clinical action. If a subsequent risk analysis (RMF-001) identifies a pathway to serious injury, reclassification to Class C will be initiated per IEC 62304 Clause 4.3.

#### 1.2.3 Out of Scope

The following are explicitly excluded from this SDP:

- Hardware medical devices interfacing with NeuroFusion-AD
- Electronic Health Record (EHR) systems at customer sites
- Network infrastructure managed by customer IT departments
- Third-party laboratory information systems providing biomarker inputs
- Clinical trial management systems

#### 1.2.4 Regulatory Pathway Summary

| Jurisdiction | Pathway | Predicate/Reference | Review Body |
|---|---|---|---|
| USA | FDA De Novo (21 CFR 513(f)(2)) | Prenosis Sepsis ImmunoScore (DEN200080) | FDA CDRH |
| European Union | MDR 2017/745, Class IIa, Rule 11 | MDR Annex IX | Notified Body (TÜV SÜD) |

---

## 2. Roles and Responsibilities

### 2.1 Organizational RACI Framework

All personnel assigned to NeuroFusion-AD development operate under the Quality Management System (QMS-001) and have documented training records in the Training Management System (TMS). All roles require signed Conflict of Interest declarations and completion of IEC 62304 awareness training prior to assignment.

### 2.2 Role Definitions and Responsibility Allocation

| Role | Personnel | Key Responsibilities | IEC 62304 Clause Ownership | Estimated Allocation |
|---|---|---|---|---|
| **ML Architect** | Pending — Senior ML Engineer | • Design and maintain model architecture (encoders, attention, GNN) <br>• Define training/validation pipelines and ML-specific quality metrics <br>• Lead algorithm verification activities <br>• Approve ML-related architecture changes <br>• Author SOUP evaluation for PyTorch, PyTorch Geometric <br>• Ensure reproducibility of training via seeded experiments <br>• Review model drift monitoring design | Clauses 5.1, 5.2, 5.3, 5.5, 7.1 | **70%** |
| **Data Engineer** | Pending — Senior Data Engineer | • Design and implement data ingestion and preprocessing pipelines <br>• Maintain input validation layer with hard-constraint enforcement <br>• Manage training/validation/test dataset splits and versioning with DVC <br>• Implement data provenance tracking <br>• Execute and document dataset integrity checks <br>• Support SOUP evaluation for data processing libraries <br>• Maintain dataset documentation for regulatory submissions | Clauses 5.1, 5.4, 7.1, 9.1 | **60%** |
| **API Engineer** | Pending — Backend Software Engineer | • Design, implement, and test FastAPI inference service <br>• Implement authentication, authorization, and session management <br>• Develop audit trail service per 21 CFR Part 11 requirements <br>• Define and maintain OpenAPI specification <br>• Implement input/output validation middleware <br>• Ensure inference latency SLA (p95 < 2.0 seconds) <br>• Author integration test suites for API endpoints | Clauses 5.4, 5.5, 5.6, 5.7, 8.1 | **80%** |
| **DevOps Engineer** | Pending — Senior DevOps/SRE Engineer | • Design and maintain CI/CD pipelines (GitHub Actions) <br>• Manage environment configurations (Dev, Staging, Production) <br>• Implement infrastructure-as-code (Terraform) for AWS environments <br>• Configure and maintain security controls (AES-256, TLS 1.3, IAM) <br>• Manage container orchestration (Docker, Kubernetes) <br>• Monitor availability SLA (99.5% uptime) <br>• Execute and document deployment verification procedures <br>• Maintain backup and disaster recovery procedures | Clauses 5.8, 6.1, 8.1, 9.1 | **75%** |
| **Regulatory Affairs Officer** | Pending — Regulatory Affairs Specialist | • Author and maintain all IEC 62304 lifecycle documents <br>• Maintain regulatory submission packages (FDA De Novo, EU MDR) <br>• Manage QMS documentation and document control <br>• Coordinate with Notified Body (TÜV SÜD) and FDA <br>• Conduct regulatory impact assessments for change requests <br>• Maintain SOUP register and evaluation records <br>• Interface with ISO 14971 Risk Management activities <br>• Ensure IEC 62366-1 usability engineering compliance | Clauses 4.1–4.4, 5.1, 8.1–8.2, 9.1 | **100%** |
| **QA Engineer** | Pending — Medical Device QA Engineer | • Develop and maintain verification and validation test plans (IEEE 829) <br>• Execute system-level, integration, and regression test suites <br>• Manage defect lifecycle in issue tracking system (Jira) <br>• Conduct and document code review quality audits <br>• Maintain test traceability matrix (TTM) from requirements to tests <br>• Execute and document anomaly reporting per IEC 62304 Clause 9 <br>• Perform independent QA audits of development artifacts <br>• Support FDA/Notified Body inspection preparation | Clauses 5.6, 5.7, 5.8, 8.1, 9.1–9.8 | **90%** |

### 2.3 Accountability and Escalation

```
Chief Medical Officer (Executive Sponsor)
        │
        ├── VP Engineering (Technical Authority)
        │       ├── ML Architect
        │       ├── Data Engineer
        │       ├── API Engineer
        │       └── DevOps Engineer
        │
        ├── Regulatory Affairs Officer (Regulatory Authority)
        │       └── Interfaces with FDA, Notified Body, ISO bodies
        │
        └── QA Lead (Quality Authority)
                └── QA Engineer
```

**Conflict Resolution:** Technical disputes are escalated to VP Engineering. Regulatory disputes are escalated to Regulatory Affairs Officer with CMO final authority. Quality gate failures are non-negotiable and require QA Lead approval to proceed.

---

## 3. Development Lifecycle Model

### 3.1 Selected Model: Iterative Agile with FDA/IEC 62304 Regulatory Overlay

NeuroFusion-AD employs an **Iterative Agile development lifecycle** structured around two-week sprints, organized into three formal phases (defined in Section 7), with mandatory regulatory gate reviews at each phase boundary.

### 3.2 Justification

| Criterion | Justification |
|---|---|
| **Complexity of ML system** | The multimodal GNN architecture requires iterative experimentation with architecture decisions informed by empirical performance data. A waterfall model would introduce unacceptable schedule risk given uncertainty in model convergence. |
| **FDA SaMD Guidance Compatibility** | FDA's 2019 Proposed Regulatory Framework for AI/ML-Based SaMD explicitly acknowledges iterative development models and the Total Product Lifecycle (TPLC) approach, aligning with Agile iteration. |
| **IEC 62304 Compatibility** | IEC 62304 is lifecycle-model agnostic and requires documented plans, traceability, and change control — all of which are implemented within the Agile framework via this SDP and associated procedures. |
| **Risk Mitigation** | Early and frequent integration and testing, combined with continuous CI/CD pipelines, enables earlier detection of defects compared to sequential development, reducing risk per ISO 14971 principles. |
| **Regulatory Gate Enforcement** | FDA De Novo and EU MDR requirements are satisfied through formal phase gate reviews (PGR) that enforce document completeness before progression, independent of sprint cadence. |

### 3.3 Lifecycle Process Description

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NeuroFusion-AD Development Lifecycle                 │
│                                                                         │
│  ┌──────────────┐    PGR-1    ┌──────────────┐    PGR-2    ┌─────────┐ │
│  │   Phase 1    │ ──────────► │   Phase 2    │ ──────────► │Phase 3  │ │
│  │  Foundation  │             │ Development  │             │ V&V +   │ │
│  │  & Planning  │             │ & Integration│             │ Release │ │
│  └──────────────┘             └──────────────┘             └─────────┘ │
│                                                                         │
│  Sprint 1  Sprint 2      Sprint 3–8          Sprint 9  Sprint 10–11    │
│                                                                         │
│  Within each Sprint:                                                    │
│  Plan → Dev → Unit Test → Code Review → Integration → CI → Demo       │
│                                                                         │
│  Regulatory Overlays (continuous):                                      │
│  ├─ Risk Management (RMF-001) — updated each sprint                    │
│  ├─ Requirements Traceability Matrix — updated each sprint              │
│  ├─ SOUP Register — updated on dependency change                        │
│  └─ Anomaly Reports — filed within 24h of detection                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Phase Gate Review (PGR) Criteria

Each Phase Gate Review (PGR) is a formal, documented checkpoint. Development teams **may not proceed** to the next phase without written PGR approval signed by the QA Lead and Regulatory Affairs Officer.

**PGR-1 Entry Criteria (Phase 1 → Phase 2):**
- [ ] All Phase 1 documents baselined and under version control
- [ ] Software Requirements Specification (SRS-001) approved
- [ ] Software Architecture Document (SAD-001) approved
- [ ] Risk Management Plan (RMF-001) approved with initial hazard analysis
- [ ] Development environment validated and operational
- [ ] SOUP register (SOUP-001) populated with all Phase 1 dependencies
- [ ] All team members confirmed with training records

**PGR-2 Entry Criteria (Phase 2 → Phase 3):**
- [ ] All software units have passing unit tests (coverage ≥ 80%)
- [ ] Integration test suite passing on staging environment
- [ ] Model performance meets defined thresholds on held-out validation set
- [ ] All Critical and High severity defects resolved or formally accepted
- [ ] Inference latency p95 < 2.0 seconds validated on staging
- [ ] Security scan (SAST/DAST) completed with no Critical/High findings unresolved
- [ ] Risk Management file updated with residual risk evaluation

### 3.5 Sprint Structure and Regulatory Integration

Each two-week sprint incorporates the following regulatory activities:

| Sprint Activity | Regulatory Requirement | Frequency |
|---|---|---|
| Sprint planning with requirements traceability update | IEC 62304 §5.1 | Per sprint |
| Anomaly report triage and classification | IEC 62304 §9.1 | Per sprint |
| Risk register review | ISO 14971 §10 | Per sprint |
| SOUP dependency audit (automated via Dependabot) | IEC 62304 §8.1 | Per sprint |
| Security vulnerability scan | FDA Cybersecurity Guidance | Per sprint |
| Sprint review with QA sign-off | IEC 62304 §5.6 | Per sprint |

---

## 4. Development Standards

### 4.1 Programming Language

| Parameter | Specification | Rationale |
|---|---|---|
| **Primary Language** | Python 3.10.x (3.10.12 pinned) | Long-term support, type hint maturity, ML ecosystem compatibility |
| **Version Pinning** | Exact version specified in `.python-version` (pyenv) | Reproducibility, regulatory traceability |
| **Version Upgrade Process** | Formal change request (CCR) required for any Python version change | IEC 62304 §8.2 |
| **Prohibited Language Constructs** | `eval()`, `exec()`, dynamic imports without explicit allowlist | Security and auditability |

All Python version upgrades require:
1. New CCR submitted and approved
2. Full regression test suite re-execution
3. SOUP register update
4. Regulatory impact assessment by RAO

### 4.2 Coding Style and Enforcement

#### 4.2.1 Code Style Standard: PEP 8

All Python source code **shall** conform to PEP 8 with the following project-specific extensions:

```toml
# pyproject.toml — Authoritative configuration
[tool.ruff]
target-version = "py310"
line-length = 88
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # Pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "ANN",  # flake8-annotations (type hints required)
    "S",    # flake8-bandit (security)
    "B",    # flake8-bugbear
    "C90",  # mccabe complexity
]
ignore = ["E501"]  # line length managed by formatter

[tool.ruff.mccabe]
max-complexity = 10  # Cyclomatic complexity limit

[tool.ruff.per-file-ignores]
"tests/*" = ["ANN", "S101"]  # Assertions and unannotated tests permitted

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

#### 4.2.2 Type Annotations

All public functions, methods, and module-level variables **shall** include complete type annotations conforming to PEP 484/526. Type checking is enforced via `mypy` in strict mode:

```toml
[tool.mypy]
python_version = "3.10"
strict = true
disallow_untyped_defs = true
disallow_any_explicit = true
warn_return_any = true
warn_unused_ignores = true
```

#### 4.2.3 Documentation Standards

All public APIs, classes, and functions **shall** include docstrings conforming to Google Python Style Guide docstring format, including:
- Summary line
- Args section with types and descriptions
- Returns section with type and description
- Raises section for all explicitly raised exceptions
- Examples section for public-facing API functions

#### 4.2.4 Input Validation Coding Requirements

All functions processing patient data **shall** implement explicit input validation with documented hard constraints. Validation failures **shall** raise typed exceptions that are logged to the audit trail:

```python
# Required pattern for all patient data processing functions
from neurofusion.validation import InputValidator, ValidationError
from neurofusion.audit import AuditLogger

BIOMARKER_CONSTRAINTS = {
    "ptau_217": (0.1, 100.0),       # pg/mL
    "abeta42_40_ratio": (0.01, 0.30),
    "nfl": (5.0, 200.0),            # pg/mL
    "mmse": (0, 30),                # integer
}
```

### 4.3 Version Control

#### 4.3.1 Version Control System

| Parameter | Specification |
|---|---|
| **System** | Git 2.43+ |
| **Hosting** | GitHub Enterprise (SOC 2 Type II certified) |
| **Repository URL** | `github.com/[org]/neurofusion-ad` (private) |
| **Access Control** | GitHub Teams with role-based permissions; MFA required for all contributors |
| **Repository Backup** | Automated daily backup to AWS S3 (separate account) |

#### 4.3.2 GitFlow Branching Strategy

The NeuroFusion-AD repository implements the **GitFlow** branching model as the authoritative version control strategy:

```
main (protected)
├── ONLY accepts merges from: release/*, hotfix/*
├── Requires: 2 approving reviews + QA Lead sign-off
├── Tag format: v{MAJOR}.{MINOR}.{PATCH}
└── Represents production-deployed, regulatory-approved code

develop (protected)
├── Integration branch for completed features
├── Requires: 2 approving reviews
├── CI must pass before merge
└── Represents current development state

feature/{ticket-id}-{description}
├── Branched FROM: develop
├── Merged TO: develop
├── Naming: feature/NFA-{Jira-ID}-short-description
├── Lifetime: Single sprint maximum (requires extension CCR if longer)
└── Example: feature/NFA-042-ptau217-validation-layer

release/{version}
├── Branched FROM: develop
├── Merged TO: main AND develop
├── Naming: release/v{MAJOR}.{MINOR}.{PATCH}
├── Activities: Bug fixes, documentation, final QA only (no new features)
└── Requires: RAO and QA Lead approval to open

hotfix/{version}
├── Branched FROM: main
├── Merged TO: main AND develop
├── Naming: hotfix/v{MAJOR}.{MINOR}.{PATCH+1}
├── Requires: Expedited CCR (CCR-HOTFIX template) before branch creation
└── Post-merge: Incident report (INC-XXX) required within 48h

experiment/{description}  [NON-REGULATORY]
├── Branched FROM: develop
├── NEVER merged to main or develop without full review
├── Used for: ML architecture exploration, data analysis
└── Must be tagged with disclaimer: "EXPERIMENTAL - NOT FOR REGULATORY USE"
```

#### 4.3.3 Commit Standards

All commits **shall** conform to **Conventional Commits 1.0.0** specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
Refs: NFA-{Jira-ID}
Signed-off-by: {Author Name} <{email}>
```

**Permitted types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`

**Regulatory commits** (affecting patient safety, security, or regulatory artifacts) **shall** additionally include:
```
REGULATORY-IMPACT: [NONE|LOW|MEDIUM|HIGH]
RISK-CONTROL: [RMF-001 reference if applicable]
```

Commit signing with GPG keys is **mandatory** for all commits to `main`, `develop`, and `release/*` branches.

### 4.4 Code Review Process

#### 4.4.1 Mandatory Peer Review Policy

**No code shall be merged to `develop`, `release/*`, or `main` branches without completing the code review process.** This is a non-negotiable quality gate.

#### 4.4.2 Review Requirements

| Branch Target | Minimum Approvers | Required Reviewers | Additional Requirements |
|---|---|---|---|
| `develop` | 2 | 1 domain expert + 1 peer | CI pipeline must be green |
| `release/*` | 2 | QA Engineer + domain expert | Security scan must pass |
| `main` | 2 | QA Lead + RAO | PGR documentation attached |
| `hotfix/*` → `main` | 2 | QA Lead + VP Engineering | Expedited CCR must be approved |

#### 4.4.3 Code Review Checklist

Reviewers **shall** complete and document the following checklist for each review. The checklist is implemented as a GitHub Pull Request template:

**Correctness:**
- [ ] Logic implements the specified requirements (RTM reference provided)
- [ ] Edge cases and boundary conditions handled
- [ ] Input validation for all patient data inputs present
- [ ] Error handling is complete and appropriate exception types used

**Safety and Security:**
- [ ] No hardcoded credentials, API keys, or PHI in code or comments
- [ ] Input validation hard constraints enforced for all biomarker inputs
- [ ] No use of prohibited language constructs (`eval`, `exec`)
- [ ] PHI/PII handling conforms to data handling procedure (DHP-001)

**ML-Specific (for model code):**
- [ ] Random seeds documented and controlled
- [ ] No data leakage between train/validation/test splits
- [ ] Model version referenced in audit trail
- [ ] Performance thresholds checked in test assertions

**Code Quality:**
- [ ] PEP 8 / ruff linting passes (automated, but reviewer confirms)
- [ ] Type annotations complete
- [ ] Docstrings complete and accurate
- [ ] Cyclomatic complexity ≤ 10
- [ ] No code duplication (DRY principle)
- [ ] Logging at appropriate levels (DEBUG/INFO/WARNING/ERROR)

**Testing:**
- [ ] Unit tests added/updated for new/modified functionality
- [ ] Test coverage ≥ 80% for modified modules
- [ ] Tests are deterministic (no time-dependent or order-dependent tests)
- [ ] IEEE 829 test case IDs referenced in test docstrings

**Documentation:**
- [ ] CHANGELOG.md updated
- [ ] API documentation updated (if applicable)
- [ ] Relevant SDP/SRS/SAD documents updated or CCR raised

#### 4.4.4 Review Timeline SLA

- Reviewer assignment: within 4 business hours of PR opening
- Initial review completion: within 2 business days
- Re-review after changes: within 1 business day
- Escalation path: PRs open > 5 business days escalate to VP Engineering

### 4.5 Testing Standards

#### 4.5.1 Testing Framework

| Tool | Version | Purpose |
|---|---|---|
| `pytest` | 7.4.x | Primary test runner |
| `pytest-cov` | 4.x | Coverage measurement |
| `pytest-asyncio` | 0.21.x | Async API endpoint testing |
| `pytest-benchmark` | 4.x | Performance regression testing |
| `hypothesis` | 6.x | Property-based testing for validation layer |
| `locust` | 2.x | Load testing (latency SLA validation) |

#### 4.5.2 Test Documentation Standard: IEEE 829

All formal test cases **shall** be documented per **IEEE 829-2008** (Standard for Software and System Test Documentation) with the following required fields:

| IEEE 829 Field | NeuroFusion-AD Implementation |
|---|---|
| Test Case Identifier | `TC-{module}-{NNN}` (e.g., `TC-VAL-001`) |
| Test Item | Software unit/integration/system reference |
| Input Specifications | Specific input values including edge cases |
| Output Specifications | Expected output, acceptance criteria |
| Environmental Needs | Docker image tag, environment configuration |
| Special Procedural Requirements | Pre-conditions, test data setup |
| Intercase Dependencies | References to dependent test cases |
| Pass/Fail Criteria | Explicit, measurable criteria |
| Traceability | SRS requirement ID, RTM reference |

#### 4.5.3 Test Coverage Requirements

| Test Level | Coverage Target | Mandatory |
|---|---|---|
| Unit Tests | ≥ 80% line coverage per module | Yes — CI gate |
| Integration Tests | All defined API endpoints | Yes — CI gate |
| System Tests | All SRS functional requirements | Yes — PGR gate |
| Performance Tests | p95 latency < 2.0s at 50 concurrent users | Yes — PGR gate |
| Security Tests | OWASP Top 10, SAST/DAST | Yes — PGR gate |
| Adversarial Input Tests | All hard-constraint boundaries | Yes — CI gate |

#### 4.5.4 Critical Test Cases for Input Validation

The following test cases are **mandatory** and their absence constitutes a blocking defect:

```python
# Mandatory test case pattern — TC-VAL-001 through TC-VAL-008
@pytest.mark.parametrize("biomarker,value,expected_valid", [
    ("ptau_217", 0.1,   True),    # TC-VAL-001: Lower bound (inclusive)
    ("ptau_217", 0.09,  False),   # TC-VAL-002: Below lower bound
    ("ptau_217", 100.0, True),    # TC-VAL-003: Upper bound (inclusive)
    ("ptau_217", 100.1, False),   # TC-VAL-004: Above upper bound
    ("mmse",     0,     True),    # TC-VAL-005: MMSE lower bound
    ("mmse",     30,    True),    # TC-VAL-006: MMSE upper bound
    ("mmse",     -1,    False),   # TC-VAL-007: MMSE below range
    ("mmse",     31,    False),   # TC-VAL-008: MMSE above range
])
def test_biomarker_hard_constraints(biomarker, value, expected_valid):
    """
    TC-VAL-001 through TC-VAL-008: Verify hard-constraint enforcement.
    
    IEEE 829 Ref: TP-VAL-001
    SRS Ref: SRS-FR-012, SRS-FR-013
    Safety Class: IEC 62304 Class B
    """
    ...
```

---

## 5. Software Configuration Management

### 5.1 Overview

The Software Configuration Management (SCM) system for NeuroFusion-AD ensures the integrity, traceability, and controlled evolution of all software configuration items (SCIs) throughout the product lifecycle. This section fulfills IEC 62304 Clause 8 requirements.

### 5.2 Software Configuration Items (SCIs)

The following items are designated as Software Configuration Items and **shall** be subject to formal configuration management:

| SCI ID | Item | Storage Location | Version Scheme |
|---|---|---|---|
| SCI-001 | Source code (all modules) | GitHub repository | Git SHA + Semantic Version |
| SCI-002 | ML model weights and checkpoints | AWS S3 (versioned bucket) | Semantic Version + run hash |
| SCI-003 | Training dataset manifests | DVC (Data Version Control) | SHA256 hash |
| SCI-004 | Docker container images | AWS ECR | Semantic Version + Git SHA |
| SCI-005 | Infrastructure-as-code (Terraform) | GitHub repository `/infra` | Semantic Version |
| SCI-006 | CI/CD pipeline definitions | GitHub repository `/.github` | Semantic Version |
| SCI-007 | Configuration files (non-secrets) | GitHub repository `/config` | Semantic Version |
| SCI-008 | Regulatory documents | Controlled Document System (CDS) | Document version (e.g., v1.0) |
| SCI-009 | Third-party