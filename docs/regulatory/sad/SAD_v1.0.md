# SOFTWARE ARCHITECTURE DOCUMENT
## NeuroFusion-AD: Multimodal GNN for Alzheimer's Disease Progression Prediction

---

```
Document ID:        SAD-001
Version:            1.0
Status:             RELEASED FOR REVIEW
Classification:     CONTROLLED DOCUMENT
Date:               2025-01-01
Author:             Regulatory Affairs / Software Architecture Team
Reviewer:           QA Lead, Clinical Systems Architect
Approver:           Chief Medical Officer, Head of Software Engineering

Regulatory Context:
  - IEC 62304:2006+AMD1:2015 (Software Safety Class: Class B)
  - FDA De Novo Pathway (21 CFR Part 882)
  - EU MDR 2017/745 Class IIa
  - ISO 14971:2019 (Risk Management)
  - IEC 62366-1:2015 (Usability Engineering)
  - HIPAA 45 CFR Part 164 / GDPR Article 25
  - HL7 FHIR R4

Change History:
  v0.1  2024-10-01  Initial draft
  v0.2  2024-11-15  Added security architecture
  v0.3  2024-12-01  Added GPU deployment spec
  v1.0  2025-01-01  Released for regulatory review
```

---

## TABLE OF CONTENTS

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Regulatory Compliance References](#2-regulatory-compliance-references)
3. [Architectural Overview](#3-architectural-overview)
4. [Component Architecture](#4-component-architecture)
5. [Data Flow Description](#5-data-flow-description)
6. [Security Architecture](#6-security-architecture)
7. [Deployment Architecture](#7-deployment-architecture)
8. [Interface Definitions](#8-interface-definitions)
9. [Risk and Hazard Mitigations at Architecture Level](#9-risk-and-hazard-mitigations-at-architecture-level)
10. [Traceability Matrix](#10-traceability-matrix)
11. [Glossary](#11-glossary)

---

## 1. PURPOSE AND SCOPE

### 1.1 Document Purpose

This Software Architecture Document (SAD) defines the architecture of the **NeuroFusion-AD** Software as a Medical Device (SaMD). It satisfies the requirements of **IEC 62304:2006+AMD1:2015 Section 5.3 (Software Architecture)**, establishing the structural foundation upon which all subsequent design, implementation, verification, and validation activities are based.

### 1.2 Intended Use (IEC 62304 §5.1.1 Context)

NeuroFusion-AD is a **Clinical Decision Support (CDS) system** intended to:

> Aid licensed clinicians in assessing the risk of Alzheimer's Disease (AD) progression in patients diagnosed with Mild Cognitive Impairment (MCI), aged 50–90 years, by integrating fluid biomarkers, acoustic features, motor assessments, and clinical/demographic data through a multimodal Graph Neural Network.

**NOT intended** to replace clinical judgment. All predictions are advisory.

### 1.3 Software Safety Classification

| Classification Axis | Assignment | Rationale |
|---|---|---|
| IEC 62304 Class | **Class B** | Software failure can contribute to serious injury; clinician oversight required before any clinical action |
| FDA Device Class | **Class II** (De Novo) | Novel SaMD with controls; predicate: Prenosis Sepsis ImmunoScore |
| EU MDR Class | **Class IIa** (Rule 11) | Diagnostic SaMD influencing clinical decisions |
| ISO 14971 Severity | **Serious** | Incorrect progression prediction could delay or inappropriately accelerate treatment |

### 1.4 Scope of This Document

This document covers:

- The system-level architectural pattern and its justification
- All major software components (SOUP and developed software)
- Component-to-component interfaces and data contracts
- Security and privacy architecture
- Deployment topology for the target runtime environment
- Traceability from architectural decisions to system requirements

---

## 2. REGULATORY COMPLIANCE REFERENCES

| Reference | Title | Relevance to This Document |
|---|---|---|
| IEC 62304:2006+AMD1:2015 §5.3 | Software Architecture | Primary compliance target |
| IEC 62304 §5.3.1 | Transform software requirements into architecture | §3, §4 |
| IEC 62304 §5.3.2 | Develop architecture for interfaces | §8 |
| IEC 62304 §5.3.3 | Specify functional and performance requirements allocated to SOUP | §4.x SOUP tables |
| IEC 62304 §5.3.4 | Identify segregation necessary for risk control | §6, §9 |
| IEC 62304 §5.3.5 | Verify the software architecture | §10 traceability |
| ISO 14971:2019 | Risk Management | §9 architecture-level mitigations |
| FDA 21st Century Cures Act | SaMD CDS Guidance | Intended use framing |
| FDA AI/ML Action Plan 2021 | Predetermined Change Control | Change management hooks |
| EU MDR 2017/745 Annex I | General Safety/Performance Requirements | §6 security |
| GDPR Article 25 | Privacy by Design | §6.4 |
| HL7 FHIR R4 | Data interoperability standard | §8 |
| NVIDIA CUDA 12.x | GPU compute platform (SOUP) | §7 |

---

## 3. ARCHITECTURAL OVERVIEW

### 3.1 Selected Architectural Pattern: Microservices

NeuroFusion-AD implements a **microservices architecture** deployed on containerized infrastructure. Each functional capability is encapsulated as an independently deployable service communicating via well-defined APIs.

#### 3.1.1 Architectural Decision Record (ADR-001): Microservices over Monolith

```
Decision:    Adopt microservices architecture
Status:      Accepted
Context:     NeuroFusion-AD must satisfy IEC 62304 §5.3.4 (segregation for 
             risk control), support independent verification of the ML inference 
             component, allow rolling upgrades without service interruption, 
             and meet 99.5% availability SLA.

Options Considered:
  Option A: Monolithic FastAPI application
    Pros:  Lower operational complexity
    Cons:  Cannot independently update Model Inference Engine (AIR change 
           management), single failure domain, harder to verify in isolation

  Option B: Microservices (Selected)
    Pros:  Independent IEC 62304 lifecycle per component, failure isolation, 
           horizontal scaling of GPU inference, SOUP boundary clarity, 
           supports PDCP (Predetermined Change Control Plan)
    Cons:  Network latency overhead (mitigated: all services on same K8s cluster,
           p95 < 2.0s requirement met per §7.4 latency budget analysis)

  Option C: Serverless (Lambda/Cloud Functions)
    Pros:  Zero-ops scaling
    Cons:  GPU cold-start latency incompatible with p95 < 2.0s; 
           audit trail atomicity difficult

Justification for Microservices:
  1. IEC 62304 §5.3.4: The Model Inference Engine (Class B) is segregated from 
     the API Gateway and Audit Logger, limiting the impact of inference failures.
  2. Regulatory change control: Model updates (retraining) do not require 
     revalidation of FHIR Validator or Audit Logger.
  3. GPU resource isolation: Inference Engine has dedicated GPU node affinity; 
     other services share CPU nodes, preventing resource contention.
  4. Security: Authentication boundary enforced at API Gateway; 
     internal services operate in private Kubernetes network namespace.
```

#### 3.1.2 Architectural Decision Record (ADR-002): Kubernetes Container Orchestration

```
Decision:    Deploy on Kubernetes (K8s) with Docker containers
Status:      Accepted
Context:     99.5% availability (≤43.8 hours downtime/year) requires 
             automated failover, rolling deployments, and health-based 
             traffic management.

Justification:
  1. Health probes (liveness/readiness) enable automatic pod restart on failure.
  2. ReplicaSets ensure minimum 2 replicas of each critical service.
  3. GPU operator plugin enables deterministic GPU allocation to inference pods.
  4. Kubernetes RBAC maps to HIPAA workforce access controls.
  5. Helm charts provide reproducible, version-controlled deployments 
     (supports IEC 62304 configuration management).
  6. Horizontal Pod Autoscaler (HPA) manages burst load without manual intervention.
```

### 3.2 High-Level System Context Diagram

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         EXTERNAL ACTORS                                      ║
║                                                                              ║
║   ┌─────────────┐    ┌─────────────────┐    ┌───────────────────────────┐   ║
║   │  Clinician  │    │   EHR System    │    │  Regulatory Auditor       │   ║
║   │  (Browser/  │    │  (FHIR Client)  │    │  (Read-only audit access) │   ║
║   │   EHR App)  │    │                 │    │                           │   ║
║   └──────┬──────┘    └────────┬────────┘    └────────────┬──────────────┘   ║
╚══════════│═══════════════════│═════════════════════════│════════════════════╝
           │                  │                          │
           │ HTTPS/TLS 1.3    │ FHIR R4 REST             │ HTTPS/TLS 1.3
           │ OAuth 2.0        │ OAuth 2.0                │ OAuth 2.0 (Auditor)
           │                  │                          │
           ▼                  ▼                          ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NEUROFUSION-AD SYSTEM BOUNDARY                            ║
║                    (Kubernetes Cluster - Private VPC)                        ║
║                                                                              ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │                     API GATEWAY (Nginx)                              │   ║
║  │              TLS Termination · Rate Limiting · mTLS internal        │   ║
║  └──────────────────────────┬───────────────────────────────────────────┘   ║
║                             │                                                ║
║        ┌────────────────────┼─────────────────────┐                         ║
║        ▼                    ▼                     ▼                          ║
║  ┌───────────┐    ┌──────────────────┐    ┌─────────────────┐              ║
║  │   FHIR    │    │  Data            │    │   Audit         │              ║
║  │ Validator │───▶│  Preprocessor   │    │   Logger        │              ║
║  └───────────┘    └────────┬─────────┘    └────────┬────────┘              ║
║                            │                       │                        ║
║                     ┌──────▼──────┐          ┌────▼──────────┐             ║
║                     │   Cache     │          │  PostgreSQL   │             ║
║                     │  (Redis)    │          │  (Audit DB)   │             ║
║                     └──────┬──────┘          └───────────────┘             ║
║                            │                                                ║
║                   ┌────────▼────────┐                                       ║
║                   │  Model          │                                        ║
║                   │  Inference      │ ◀── GPU Node (NVIDIA T4+)            ║
║                   │  Engine         │                                        ║
║                   └────────┬────────┘                                       ║
║                            │                                                ║
║              ┌─────────────┼──────────────┐                                 ║
║              ▼             ▼              ▼                                  ║
║  ┌──────────────────┐ ┌─────────────┐ ┌──────────────────┐                 ║
║  │ Explainability   │ │   Output    │ │    Metrics        │                 ║
║  │ Engine (SHAP)    │ │  Formatter  │ │  (Prometheus +    │                 ║
║  └─────────┬────────┘ └──────┬──────┘ │   Grafana)        │                 ║
║            │                │        └──────────────────┘                  ║
║            └────────────────┘                                               ║
║                     │ FHIR RiskAssessment                                   ║
║                     ▼                                                        ║
║              ┌──────────────┐                                               ║
║              │ API Gateway  │ (response path)                               ║
║              │  (Egress)    │                                               ║
║              └──────────────┘                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 3.3 Software Item Inventory (IEC 62304 §5.3.3 SOUP Identification)

| Software Item | Type | Version | Safety Class | SOUP? | Supplier |
|---|---|---|---|---|---|
| API Gateway (Nginx config) | Developed | 1.0 | Class B | No | NeuroFusion team |
| FHIR Validator service | Developed | 1.0 | Class B | No | NeuroFusion team |
| Data Preprocessor service | Developed | 1.0 | Class B | No | NeuroFusion team |
| Model Inference Engine | Developed | 1.0 | Class B | No | NeuroFusion team |
| Explainability Engine | Developed | 1.0 | Class B | No | NeuroFusion team |
| Output Formatter service | Developed | 1.0 | Class B | No | NeuroFusion team |
| Audit Logger service | Developed | 1.0 | Class B | No | NeuroFusion team |
| **PyTorch 2.1.2** | SOUP | 2.1.2 | Class B | Yes | Meta AI |
| **PyTorch Geometric 2.5.0** | SOUP | 2.5.0 | Class B | Yes | Pyg Team |
| **FastAPI 0.110.x** | SOUP | 0.110 | Class B | Yes | Sebastián Ramírez |
| **Pydantic v2.6.x** | SOUP | 2.6 | Class B | Yes | Pydantic team |
| **SHAP 0.44.x** | SOUP | 0.44 | Class B | Yes | Lundberg et al. |
| **Nginx 1.25.x** | SOUP | 1.25 | Class B | Yes | F5/Nginx Inc. |
| **PostgreSQL 14.x** | SOUP | 14 | Class B | Yes | PostgreSQL Global Group |
| **Redis 7.2.x** | SOUP | 7.2 | Class B | Yes | Redis Ltd. |
| **Prometheus 2.48.x** | SOUP | 2.48 | Class A | Yes | CNCF |
| **Grafana 10.x** | SOUP | 10 | Class A | Yes | Grafana Labs |
| **Kubernetes 1.29.x** | SOUP | 1.29 | Class B | Yes | CNCF |
| **Docker Engine 25.x** | SOUP | 25 | Class B | Yes | Docker Inc. |
| **CUDA 12.3** | SOUP | 12.3 | Class B | Yes | NVIDIA |
| **Python 3.11.x** | SOUP | 3.11 | Class B | Yes | Python Software Foundation |

---

## 4. COMPONENT ARCHITECTURE

### 4.1 Component Overview Matrix

| Component | IEC 62304 Unit | Safety Class | Primary SRS References | Deployment Unit |
|---|---|---|---|---|
| API Gateway | SWU-001 | Class B | SRS-SEC-001..005 | K8s Ingress + Pod |
| FHIR Validator | SWU-002 | Class B | SRS-FHIR-001..010 | K8s Pod |
| Data Preprocessor | SWU-003 | Class B | SRS-PRE-001..015 | K8s Pod |
| Model Inference Engine | SWU-004 | Class B | SRS-INF-001..020 | K8s Pod (GPU) |
| Explainability Engine | SWU-005 | Class B | SRS-EXP-001..008 | K8s Pod |
| Output Formatter | SWU-006 | Class B | SRS-OUT-001..010 | K8s Pod |
| Audit Logger | SWU-007 | Class B | SRS-AUD-001..010 | K8s Pod |
| Cache (Redis) | SWU-008 | Class B | SRS-PERF-001..005 | K8s StatefulSet |
| Metrics (Prom/Grafana) | SWU-009 | Class A | SRS-MON-001..005 | K8s Pod |

---

### 4.2 SWU-001: API Gateway (Nginx)

#### 4.2.1 Responsibility

The API Gateway is the **single ingress point** for all external traffic. It enforces transport security, authenticates callers, rate-limits requests, routes to downstream services, and returns responses to clients. It provides the security perimeter between untrusted external networks and the trusted internal Kubernetes service mesh.

#### 4.2.2 Functional Responsibilities

```
┌─────────────────────────────────────────────────────────────────┐
│                    API GATEWAY (SWU-001)                        │
│                                                                 │
│  1. TLS 1.3 Termination                                         │
│     - Certificate management via cert-manager (Let's Encrypt    │
│       or enterprise CA)                                         │
│     - mTLS for internal service-to-service communication        │
│                                                                 │
│  2. OAuth 2.0 / SMART on FHIR Token Validation                  │
│     - Validate Bearer tokens with JWKS endpoint                 │
│     - Reject expired or malformed tokens (HTTP 401)             │
│     - Extract roles from token claims                           │
│                                                                 │
│  3. Rate Limiting                                               │
│     - Per-client: 60 requests/minute (burst: 10)                │
│     - Per-IP: 200 requests/minute                               │
│     - Return HTTP 429 with Retry-After header                   │
│                                                                 │
│  4. Request Routing                                             │
│     - POST /fhir/r4/Observation → FHIR Validator               │
│     - GET  /fhir/r4/RiskAssessment/{id} → Output Formatter     │
│     - GET  /audit/events → Audit Logger                         │
│     - GET  /health → Health endpoint (unauthenticated)          │
│                                                                 │
│  5. Response Sanitization                                       │
│     - Strip internal headers (X-Pod-Name, X-Trace-Internal)    │
│     - Inject security headers (HSTS, CSP, X-Frame-Options)     │
│                                                                 │
│  6. Audit Event Emission                                        │
│     - Log all ingress requests to Audit Logger                  │
│     - Include: timestamp, client_id, method, path, status_code  │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.2.3 Interfaces

| Interface | Direction | Protocol | Format |
|---|---|---|---|
| External HTTPS | Inbound | TLS 1.3 / HTTP/2 | FHIR R4 JSON |
| → FHIR Validator | Outbound | HTTP/1.1 (cluster internal) | FHIR R4 JSON |
| → Audit Logger | Outbound | HTTP/1.1 (async) | JSON event |
| ← Client response | Outbound | TLS 1.3 / HTTP/2 | FHIR R4 JSON |
| JWKS endpoint | Outbound | HTTPS | JSON Web Key Set |

#### 4.2.4 Dependencies

| Dependency | Type | Version | Rationale |
|---|---|---|---|
| Nginx | SOUP | 1.25.x | Reverse proxy and TLS terminator |
| cert-manager | SOUP | 1.14.x | Automated TLS certificate lifecycle |
| OAuth 2.0 Authorization Server | External | RFC 6749 | Token issuance (out of scope) |

#### 4.2.5 Configuration Parameters (Safety-Significant)

```nginx
# nginx.conf (safety-significant excerpts)
ssl_protocols TLSv1.3;                        # TLS version enforcement
ssl_prefer_server_ciphers on;
ssl_ciphers TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256;

limit_req_zone $binary_remote_addr zone=per_ip:10m rate=200r/m;
limit_req_zone $http_authorization zone=per_client:10m rate=60r/m;

client_max_body_size 10M;                     # Prevent payload DoS
proxy_read_timeout 5s;                        # Enforce latency budget
proxy_connect_timeout 2s;

add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Frame-Options "DENY" always;
add_header X-Content-Type-Options "nosniff" always;
```

---

### 4.3 SWU-002: FHIR Validator (FastAPI + Pydantic v2)

#### 4.3.1 Responsibility

The FHIR Validator receives incoming FHIR R4 Observation bundles from the API Gateway, validates their structural conformance to the FHIR R4 specification and NeuroFusion-AD profiles, and enforces the hard-constraint input validation ranges defined in the System Requirements Specification.

#### 4.3.2 Functional Responsibilities

```
┌─────────────────────────────────────────────────────────────────┐
│                  FHIR VALIDATOR (SWU-002)                       │
│                                                                 │
│  1. FHIR R4 Schema Validation                                   │
│     - Validate Observation.resourceType = "Observation"         │
│     - Validate LOINC codes for each biomarker observation       │
│     - Verify required fields: subject, effectiveDateTime,       │
│       valueQuantity, code                                       │
│                                                                 │
│  2. NeuroFusion-AD Profile Conformance                          │
│     - pTau-217:   0.1 ≤ value ≤ 100 pg/mL (HARD CONSTRAINT)   │
│     - Abeta42/40: 0.01 ≤ value ≤ 0.30   (HARD CONSTRAINT)     │
│     - NfL:        5.0 ≤ value ≤ 200 pg/mL (HARD CONSTRAINT)   │
│     - MMSE:       0 ≤ value ≤ 30          (HARD CONSTRAINT)    │
│     - Age:        50 ≤ value ≤ 90  years  (INTENDED USE)       │
│                                                                 │
│  3. Missing Data Assessment                                     │
│     - Identify which of 4 modalities are present               │
│     - Flag missing modalities in ValidationReport              │
│     - Block inference if <2 modalities present (safety rule)   │
│                                                                 │
│  4. Patient De-identification                                   │
│     - Replace Patient.id with HMAC-SHA256(patient_id, key)     │
│     - Strip: Patient.name, Patient.address, Patient.telecom    │
│     - Retain: age, sex, relevant clinical codes                 │
│                                                                 │
│  5. Validation Report Generation                                │
│     - Return OperationOutcome on failure                        │
│     - Return validated + pseudonymized payload on success       │
└─────────────────────────────────────────────────────────────────┘
```

#### 4.3.3 Input Validation Schema (Pydantic v2)

```python
# fhir_validator/models/input_schema.py
# IEC 62304 §5.3.3 - SOUP: Pydantic v2.6.x

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum

class BiomarkerCode(str, Enum):
    PTAU_217  = "416855002"   # SNOMED CT: pTau-217
    ABETA_4240 = "416124001"  # SNOMED CT: Abeta42/40 ratio
    NFL        = "416125000"  # SNOMED CT: NfL
    MMSE       = "273724008"  # SNOMED CT: MMSE score

class QuantityValue(BaseModel):
    value: float
    unit: str
    system: str = "http://unitsofmeasure.org"
    code: str

class NfADObservation(BaseModel):
    """NeuroFusion-AD validated FHIR Observation"""
    resourceType: Literal["Observation"] = "Observation"
    status: Literal["final", "preliminary"]
    code_system: str
    code_value: BiomarkerCode
    value_quantity: QuantityValue
    effective_date_time: str  # ISO 8601

    @field_validator("value_quantity")
    @classmethod
    def validate_range(cls, v: QuantityValue, info) -> QuantityValue:
        """
        IEC 62304 §5.3 Safety Constraint: Hard input validation ranges.
        Out-of-range values MUST be rejected; never imputed.
        Ref: SRS-PRE-003 through SRS-PRE-006
        """
        code = info.data.get("code_value")
        ranges = {
            BiomarkerCode.PTAU_217:   (0.1, 100.0),
            BiomarkerCode.ABETA_4240: (0.01, 0.30),
            BiomarkerCode.NFL:        (5.0, 200.0),
            BiomarkerCode.MMSE:       (0.0, 30.0),
        }
        if code in ranges:
            lo, hi = ranges[code]
            if not (lo <= v.value <= hi):
                raise ValueError(
                    f"Value {v.value} for {code} outside valid range "
                    f"[{lo}, {hi}]. Inference blocked. "
                    f"SAFETY-CONSTRAINT: SRS-PRE-{code.name}"
                )
        return v

class NfADInferenceRequest(BaseModel):
    """Complete multimodal inference request"""
    patient_pseudonym_id: str  # Pre-hashed before this point
    request_id: str            # UUID v4, client-provided
    observations: list[NfADObservation]
    acoustic_features: Optional[dict] = None   # 128-dim feature vector
    motor_features:    Optional[dict] = None   # 64-dim feature vector
    
    @field_validator("observations")
    @classmethod
    def validate_minimum_modalities(cls, v) -> list:
        """Safety rule: minimum 2 modalities required"""
        if len(v) < 1:
            raise ValueError(
                "At least one biomarker observation required. "
                "Safety rule: SR-MOD-001"
            )
        return v
```

#### 4.3.4 Interfaces

| Interface | Direction | Protocol | Format |
|---|---|---|---|
| ← API Gateway | Inbound | HTTP/1.1 | FHIR R4 Bundle JSON |
| → Data Preprocessor | Outbound | HTTP/1.1 | NfAD Internal JSON |
| → Audit Logger | Outbound | HTTP/1.1 async | Validation Event JSON |
| ← Validation failure | Outbound | HTTP/1.1 | FHIR OperationOutcome |

---

### 4.4 SWU-003: Data Preprocessor

#### 4.4.1 Responsibility

The Data Preprocessor transforms validated FHIR observations into normalized tensor inputs suitable for the Model Inference Engine. It applies modality-specific normalization pipelines, handles clinically appropriate missing data imputation, and interfaces with the Redis cache to avoid redundant computation.

#### 4.4.2 Preprocessing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│               DATA PREPROCESSOR (SWU-003)                       │
│                                                                  │
│  Input: Validated NfAD Internal JSON                            │
│                                                                  │
│  Pipeline Stage 1: Modality Router                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Route observations to modality-specific processors:     │   │
│  │  - Fluid Biomarkers: [pTau-217, Abeta42/40, NfL]        │   │
│  │  - Clinical/Demographic: [MMSE, age, sex, education]     │   │
│  │  - Acoustic: [128-dim pre-extracted features]            │   │
│  │  - Motor: [64-dim pre-extracted features]                │   │
│  └──────────────────────────────────────────────────────────┘   │
│  Pipeline Stage 2: Missing Value Handling                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Per-modality strategy (from training distribution):     │   │
│  │  - pTau-217:   median imputation (training cohort)       │   │
│  │  - Abeta42/40: median imputation (training cohort)       │   │
│  │  - NfL:        median imputation (training cohort)       │   │
│  │  - Acoustic:   zero-vector (with missingness flag=1)     │   │
│  │  - Motor:      zero-vector (with missingness flag=1)     │   │
│  │  NOTE: MMSE missing → block inference (safety rule)      │   │
│  └──────────────────────────────────────────────────────────┘   │
│  Pipeline Stage 3: Normalization                                │
│  ┌