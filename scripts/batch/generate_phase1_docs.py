#!/usr/bin/env python3
"""
NeuroFusion-AD: Batch API Document Generator
Phase 1 — Regulatory Documentation

Uses Anthropic Batch API (50% discount) for all large document generation.
Run this for: SRS, SAD, RMF, FMEA, SDP, Regulatory Strategy, Data Requirements.

Usage:
    python scripts/batch/generate_phase1_docs.py --submit
    python scripts/batch/generate_phase1_docs.py --check --batch-id <id>
    python scripts/batch/generate_phase1_docs.py --retrieve --batch-id <id>
"""

import anthropic
import json
import argparse
from pathlib import Path
from datetime import datetime

client = anthropic.Anthropic()  # Reads ANTHROPIC_API_KEY from env

# ── Document context loaded once, cached across all requests ──────────────────

NEUROFUSION_CONTEXT = """
You are the Regulatory Affairs Officer for NeuroFusion-AD, a multimodal Graph Neural 
Network for Alzheimer's Disease Progression Prediction, targeting FDA De Novo clearance 
and EU MDR Class IIa certification.

PROJECT SPECIFICATIONS:
- Intended Use: CDS to aid assessment of AD progression risk in MCI patients age 50-90
- Regulatory Class: SaMD, IEC 62304 Class B, ISO 14971
- FDA Pathway: De Novo (predicate: Prenosis Sepsis ImmunoScore)
- EU Pathway: MDR Class IIa, Notified Body review

MODEL ARCHITECTURE:
- 4 Modality Encoders (fluid biomarkers, acoustic, motor, clinical/demographic)
- Cross-Modal Attention (768-dim, 8 heads)
- GraphSAGE GNN (3 layers)
- Multi-task outputs: classification (AUC≥0.85), regression (RMSE≤3.0), survival (C-index≥0.75)

INPUT VALIDATION RANGES (hard constraints):
- pTau-217: 0.1–100 pg/mL
- Abeta42/40: 0.01–0.30
- NfL: 5–200 pg/mL
- MMSE: 0–30

PERFORMANCE REQUIREMENTS:
- Inference latency: p95 < 2.0 seconds
- Availability: 99.5% uptime
- Security: AES-256 at rest, TLS 1.3 in transit, full audit trail

TECHNOLOGY STACK:
- PyTorch 2.1.2, PyTorch Geometric 2.5.0, FastAPI, PostgreSQL 14, Docker, Kubernetes
"""

# ── Batch request definitions ──────────────────────────────────────────────────

PHASE1_DOCUMENTS = [
    {
        "custom_id": "srs-section-1-4",
        "prompt": """Generate Sections 1-4 of a Software Requirements Specification (SRS) 
        for NeuroFusion-AD, fully compliant with IEC 62304 Section 5.2.
        
        Include:
        1. Introduction (purpose, scope, definitions, abbreviations, overview)
        2. Overall Description (product perspective, functions, users, constraints, assumptions)
        3. Functional Requirements — Data Ingestion (FRI-001 to FRI-020):
           - FHIR Observation parsing for pTau-217, Aβ42/40, NfL, acoustic, motor features
           - Validation ranges for each input
           - Each requirement formatted as: ID | Description | Priority | Verification Method
        4. Functional Requirements — Preprocessing (FRP-001 to FRP-015):
           - Z-score normalization, median imputation, APOE/sex encoding
        
        Format: Professional regulatory document in Markdown. Every requirement must be 
        uniquely numbered, testable, and traceable. Document ID: SRS-001 v1.0."""
    },
    {
        "custom_id": "srs-section-5-8",
        "prompt": """Continue the SRS for NeuroFusion-AD (SRS-001 v1.0).
        
        Generate Sections 5-8:
        5. Functional Requirements — Model Inference (FRM-001 to FRM-020):
           - GNN forward pass requirements
           - Attention mechanism requirements
           - Multi-task output requirements (classification, regression, survival)
        6. Functional Requirements — Output (FRO-001 to FRO-015):
           - FHIR RiskAssessment generation
           - SHAP explainability requirements
           - Audit logging requirements
        7. Non-Functional Requirements:
           - Performance (NFR-P001: p95 latency < 2.0s)
           - Security (NFR-S001-003: AES-256, TLS 1.3, audit trail)
           - Explainability (NFR-EXPLAIN-001-005: SHAP + attention weights)
           - Availability (NFR-A001: 99.5% uptime)
        8. External Interface Requirements:
           - FHIR R4 API interface
           - EHR integration (HL7/FHIR)
           - Navify Algorithm Suite interface
        
        Format: Continuation of IEC 62304 compliant regulatory Markdown document."""
    },
    {
        "custom_id": "rmf-hazard-analysis",
        "prompt": """Generate the Risk Management File (RMF) for NeuroFusion-AD compliant 
        with ISO 14971:2019. Document ID: RMF-001 v1.0.
        
        Include:
        1. Risk Management Plan (scope, responsibilities, criteria for risk acceptability)
        2. Hazard Identification and Analysis — minimum 8 hazards including:
           - False negative (missed diagnosis)
           - False positive (unnecessary testing)
           - Model bias (demographic subgroups)
           - Data quality failure (corrupted inputs)
           - Cybersecurity breach (PHI exposure)
           - Over-reliance (automation bias)
           - System downtime (unavailability)
           - Software error (wrong calculation)
        
        For each hazard provide:
        | Hazard ID | Hazard Description | Hazardous Situation | Harm | Severity | Probability | Risk Level | Mitigation | Residual Risk |
        
        Use this scale: Severity (Catastrophic/Critical/Serious/Moderate/Negligible)
        Probability (High/Medium/Low/Very Low)
        Risk = Severity × Probability matrix
        
        3. Risk Acceptability Criteria:
           - Unacceptable: Critical+Medium or above
           - ALARP: All Medium/High risks
           - Acceptable: Low/Very Low after mitigation
        
        Format: ISO 14971 compliant Markdown document."""
    },
    {
        "custom_id": "fmea-component-analysis",
        "prompt": """Generate a Failure Mode and Effects Analysis (FMEA) for NeuroFusion-AD 
        as part of RMF-001. 
        
        Create a comprehensive FMEA table covering these 8 components:
        1. Feature Encoder (FluidBiomarkerEncoder, AcousticEncoder, MotorEncoder)
        2. Input Validator (range validation, FHIR parsing)
        3. GNN Layer (GraphSAGE convolution)
        4. Attention Module (CrossModalAttention)
        5. Output Formatter (FHIR RiskAssessment builder)
        6. Audit Logger (PostgreSQL)
        7. API Gateway (FastAPI + Nginx)
        8. Database Connection Pool
        
        For each component, identify at least 2 failure modes.
        
        Format each row:
        | Component | Failure Mode | Effect | Severity(1-10) | Occurrence(1-10) | Detection(1-10) | RPN | Action | New RPN |
        
        RPN = Severity × Occurrence × Detection
        Flag all RPN > 100 as high priority.
        Include recommended corrective actions for each failure mode.
        
        Format: Professional FMEA table in Markdown."""
    },
    {
        "custom_id": "sad-architecture",
        "prompt": """Generate a Software Architecture Document (SAD) for NeuroFusion-AD 
        compliant with IEC 62304 Section 5.3. Document ID: SAD-001 v1.0.
        
        Include:
        1. Architectural Overview
           - Selected pattern: Microservices
           - Deployment: Docker containers on Kubernetes
           - Justification for architectural decisions
        
        2. Component Architecture (describe each component's responsibility, interfaces, and dependencies):
           - API Gateway (Nginx): TLS termination, rate limiting
           - FHIR Validator (FastAPI + Pydantic v2)
           - Data Preprocessor: normalize, impute, validate
           - Model Inference Engine: PyTorch on GPU
           - Explainability Engine: SHAP + attention weights
           - Output Formatter: FHIR RiskAssessment builder
           - Audit Logger: PostgreSQL 14, append-only
           - Cache: Redis for preprocessing results
           - Metrics: Prometheus + Grafana
        
        3. Data Flow Description (step-by-step for a single inference request)
        
        4. Security Architecture:
           - Authentication: OAuth 2.0 (SMART on FHIR)
           - Authorization: Role-based (Physician, Admin, Auditor)
           - Encryption: AES-256 at rest, TLS 1.3 in transit
           - Audit trail: every prediction logged with hashed patient ID
        
        5. Deployment Architecture:
           - Kubernetes cluster configuration
           - GPU node requirements (NVIDIA T4 minimum)
           - High availability setup
        
        6. Interface Definitions:
           - API endpoint contracts (FHIR R4)
           - Internal service interfaces
        
        Format: IEC 62304 compliant Markdown document with ASCII diagrams where helpful."""
    },
    {
        "custom_id": "sdp-development-plan",
        "prompt": """Generate a Software Development Plan (SDP) for NeuroFusion-AD 
        compliant with IEC 62304. Document ID: SDP-001 v1.0.
        
        Include:
        1. Purpose and Scope
        2. Roles and Responsibilities table (ML Architect, Data Engineer, API Engineer, 
           DevOps Engineer, Regulatory Officer, QA Engineer — with responsibilities and % time)
        3. Development Lifecycle Model (Iterative/Agile with FDA overlay, justification)
        4. Development Standards:
           - Language: Python 3.10
           - Style: PEP 8 (ruff enforced)
           - Version control: Git, GitFlow branching
           - Code review: mandatory peer review, 2 approvers
           - Testing: pytest, IEEE 829
        5. Software Configuration Management:
           - Branching strategy (main, develop, feature/*, release/*, hotfix/*)
           - Semantic versioning (MAJOR.MINOR.PATCH)
           - Change control process
        6. Infrastructure:
           - Development: Local Docker
           - Staging: AWS (dev account)
           - Production: AWS (prod account, separate IAM)
           - CI/CD: GitHub Actions
        7. Phase Timeline Summary (Phase 1-3 with milestones)
        8. Risk Management reference (points to RMF-001)
        
        Format: IEC 62304 compliant Markdown document."""
    },
    {
        "custom_id": "regulatory-strategy",
        "prompt": """Generate a Regulatory Strategy Document for NeuroFusion-AD.
        Document ID: REG-001 v1.0.
        
        Include:
        1. Executive Summary of regulatory approach
        
        2. FDA De Novo Pathway Analysis:
           - Predicate device: Prenosis Sepsis ImmunoScore (De Novo DEN200057)
           - Justification for De Novo vs 510(k)
           - Substantial equivalence argument
           - Special controls that will apply
           - Submission timeline estimate (Phase 3, months 15-16)
        
        3. EU MDR Class IIa Analysis:
           - Classification rule: Rule 11 (software driving clinical decisions)
           - Notified Body requirements
           - Clinical evaluation pathway
           - IVDR vs MDR consideration (conclusion: MDR applies)
        
        4. IEC 62304 Compliance Plan:
           - Software Safety Class B rationale (injury possible, not life-threatening)
           - Required activities per class
           - Documentation requirements
        
        5. ISO 14971 Risk Management Plan summary
        
        6. Key Regulatory Milestones:
           - Phase 1: Pre-submission meeting preparation
           - Phase 2: FDA Pre-Sub Q-submission
           - Phase 3: De Novo submission, MDR Technical File
        
        7. Regulatory Risks and Mitigations
        
        Format: Professional regulatory strategy Markdown document."""
    },
    {
        "custom_id": "data-requirements",
        "prompt": """Generate a Data Requirements Document (DRD) for NeuroFusion-AD.
        Document ID: DRD-001 v1.0.
        
        Include:
        1. Data Sources:
           - ADNI (Alzheimer's Disease Neuroimaging Initiative): access process, expected ~1,200 MCI patients
           - Bio-Hermes-002: access process, expected ~500 patients with plasma biomarkers
           - DementiaBank (Pitt Corpus): access process, ~300 audio recordings
        
        2. Data Schema Specification:
           Table format for each data source: Field Name | Type | Valid Range | Units | Required/Optional | FHIR Mapping
           Cover: pTau-217, Aβ42/40, NfL, MMSE, APOE, Age, Sex, Acoustic features (15), Motor features (20)
        
        3. Data Quality Requirements:
           - Completeness: ≥80% of required fields per patient
           - Validity: All values within physiologically plausible ranges
           - Consistency: Longitudinal records must have ≥2 time points
        
        4. Data Preprocessing Specification:
           - Normalization: StandardScaler (fit on training set only)
           - Imputation: Median (continuous), Mode (categorical)
           - Train/Val/Test split: 70/15/15, stratified by progression label
        
        5. Phase 1 Synthetic Data Plan:
           - Rationale: ADNI access takes 1-2 weeks; synthetic data enables development
           - Generation method: literature-based distributions + correlated noise
           - Validation: statistical comparison to published ADNI summary statistics
           - Limitation disclosure: must be clearly labeled in all documents
        
        6. Data Privacy and Security:
           - De-identification: HIPAA Safe Harbor method
           - Storage: encrypted at rest (AES-256)
           - Access control: role-based, audit logged
        
        Format: Professional data requirements Markdown document."""
    },
]

# ── Submission function ────────────────────────────────────────────────────────

def submit_batch():
    """Submit all Phase 1 documents as a single batch job."""
    
    requests = []
    for doc in PHASE1_DOCUMENTS:
        requests.append({
            "custom_id": doc["custom_id"],
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 8000,
                "system": [
                    {
                        "type": "text",
                        "text": NEUROFUSION_CONTEXT,
                        "cache_control": {"type": "ephemeral"}  # Cache the shared context
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": doc["prompt"]
                    }
                ]
            }
        })
    
    print(f"Submitting batch with {len(requests)} document requests...")
    print("Estimated cost: ~$2-4 (50% batch discount applied automatically)")
    print("Estimated completion: 1-24 hours\n")
    
    batch = client.messages.batches.create(requests=requests)
    
    print(f"✅ Batch submitted successfully!")
    print(f"   Batch ID: {batch.id}")
    print(f"   Status: {batch.processing_status}")
    print(f"\nSave this ID. Check status with:")
    print(f"   python scripts/batch/generate_phase1_docs.py --check --batch-id {batch.id}")
    
    # Save batch ID to file for reference
    Path("scripts/batch/.last_batch_id").write_text(batch.id)
    
    return batch.id


def check_batch(batch_id: str):
    """Check status of a batch job."""
    batch = client.messages.batches.retrieve(batch_id)
    
    print(f"Batch ID: {batch_id}")
    print(f"Status: {batch.processing_status}")
    
    if hasattr(batch, 'request_counts'):
        counts = batch.request_counts
        print(f"Processing: {counts.processing}")
        print(f"Succeeded: {counts.succeeded}")
        print(f"Errored: {counts.errored}")
        print(f"Canceled: {counts.canceled}")
    
    if batch.processing_status == "ended":
        print("\n✅ Batch complete! Retrieve results with:")
        print(f"   python scripts/batch/generate_phase1_docs.py --retrieve --batch-id {batch_id}")
    else:
        print("\n⏳ Still processing. Check again in a few minutes.")


def retrieve_and_save(batch_id: str):
    """Retrieve batch results and save to docs/regulatory/."""
    
    # Define output paths
    output_map = {
        "srs-section-1-4":       "docs/regulatory/srs/SRS_v1.0_sections1-4.md",
        "srs-section-5-8":       "docs/regulatory/srs/SRS_v1.0_sections5-8.md",
        "rmf-hazard-analysis":   "docs/regulatory/rmf/RMF_v1.0_hazard_analysis.md",
        "fmea-component-analysis": "docs/regulatory/rmf/RMF_v1.0_fmea.md",
        "sad-architecture":      "docs/regulatory/sad/SAD_v1.0.md",
        "sdp-development-plan":  "docs/regulatory/sdp/SDP_v1.0.md",
        "regulatory-strategy":   "docs/regulatory/regulatory_strategy_v1.0.md",
        "data-requirements":     "docs/regulatory/data_requirements_v1.0.md",
    }
    
    # Create output directories
    for path in output_map.values():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    errors = 0
    
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        
        if result.result.type == "succeeded":
            content = result.result.message.content[0].text
            output_path = output_map.get(custom_id)
            
            if output_path:
                # Add metadata header
                header = f"""---
document_id: {custom_id}
generated: {datetime.now().isoformat()}
batch_id: {batch_id}
status: DRAFT — requires human review before approval
---

"""
                Path(output_path).write_text(header + content)
                print(f"✅ Saved: {output_path}")
                saved += 1
        else:
            print(f"❌ Error on {custom_id}: {result.result.error}")
            errors += 1
    
    print(f"\n{'='*50}")
    print(f"Retrieved: {saved} documents saved, {errors} errors")
    print(f"\nNext step: Review documents in docs/regulatory/")
    print(f"Then run Claude Code to refine and complete remaining Phase 1 tasks.")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroFusion-AD Phase 1 Batch Document Generator")
    parser.add_argument("--submit", action="store_true", help="Submit batch job")
    parser.add_argument("--check", action="store_true", help="Check batch status")
    parser.add_argument("--retrieve", action="store_true", help="Retrieve and save results")
    parser.add_argument("--batch-id", type=str, help="Batch ID to check/retrieve")
    
    args = parser.parse_args()
    
    if args.submit:
        submit_batch()
    elif args.check:
        batch_id = args.batch_id or Path("scripts/batch/.last_batch_id").read_text().strip()
        check_batch(batch_id)
    elif args.retrieve:
        batch_id = args.batch_id or Path("scripts/batch/.last_batch_id").read_text().strip()
        retrieve_and_save(batch_id)
    else:
        parser.print_help()
