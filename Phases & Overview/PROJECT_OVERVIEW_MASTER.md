# NeuroFusion-AD: Complete Development Plan Overview

**Project**: Multimodal Graph Neural Network for Alzheimer's Disease Progression Prediction  
**Target**: Roche Information Solutions Acquisition  
**Duration**: 16 Months (Phases 1-3)  
**Document Version**: 2.0  
**Last Updated**: February 15, 2026

---

## Executive Summary

NeuroFusion-AD is a Clinical Decision Support (CDS) system designed for acquisition by Roche Information Solutions. The system predicts Alzheimer's Disease progression from Mild Cognitive Impairment (MCI) by fusing:

1. **Fluid Biomarkers** (Roche Elecsys): Plasma pTau-217, Aβ42/40 ratio, NfL
2. **Digital Biomarkers**: Acoustic features (speech jitter/shimmer) and motor features (gait analysis)
3. **Clinical Data**: Demographics, APOE ε4 genotype, MMSE scores

**Core Innovation**: Cross-Modal Attention mechanism + Patient Similarity Graph Neural Network (GNN)

**Regulatory Targets**:
- FDA 510(k) De Novo (Class II SaMD)
- EU MDR Class IIa

**Strategic Fit for Roche**:
- Drives utilization of Elecsys pTau-217 assays (reagent pull-through)
- Fills strategic gap in Roche's neurology portfolio
- Plug-and-play integration with Navify Algorithm Suite
- Trained on Roche-partnered Bio-Hermes dataset

---

## Project Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1 (Months 1-4)                      │
│  • Regulatory framework (IEC 62304, ISO 14971)              │
│  • Data pipelines (ADNI, Bio-Hermes)                        │
│  • Model architecture implementation                         │
│  • Unit testing of all components                            │
│  Deliverable: Working prototype with 70% accuracy           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 2 (Months 5-10)                     │
│  • Full-scale training (150 epochs on ADNI)                 │
│  • Hyperparameter optimization (Optuna, 50 trials)          │
│  • Bio-Hermes fine-tuning (50 epochs)                       │
│  • Clinical validation (AUC >0.85, RMSE <3.0)               │
│  • Explainability (SHAP, attention weights)                 │
│  • Subgroup analysis (age, sex, APOE fairness)              │
│  Deliverable: Production-ready model checkpoints            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 3 (Months 11-16)                    │
│  • Navify Algorithm Suite integration                        │
│  • FHIR R4 API implementation                                │
│  • Docker containerization + Kubernetes deployment           │
│  • Security hardening (HIPAA, GDPR compliance)              │
│  • System testing (load, penetration, failover)             │
│  • FDA/MDR submission documentation                          │
│  Deliverable: Deployable system + regulatory dossier        │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Specifications

### Model Architecture

**NeuroFusion-AD Model Components**:

1. **Modality-Specific Encoders**:
   - **Fluid Encoder**: 3-layer MLP (input_dim=3, output_dim=768)
     - Inputs: pTau-217, Aβ42/40, NfL
     - Architecture: Linear → ReLU → LayerNorm → Dropout(0.2) → Linear
   - **Acoustic Encoder**: 4-layer MLP (input_dim=15, output_dim=768)
     - Inputs: Jitter, shimmer, pause duration, pitch, F0, F1, F2, MFCC features
   - **Motor Encoder**: 4-layer MLP (input_dim=20, output_dim=768)
     - Inputs: Gait speed, stride length/variability, double support time, accelerometer features
   - **Clinical Encoder**: Embedding + MLP (output_dim=768)
     - Inputs: Age, sex, education, APOE ε4 status, MMSE

2. **Cross-Modal Attention Fusion**:
   - 8-head multi-head attention mechanism
   - Query: Fluid embeddings (the "anchor")
   - Keys/Values: Acoustic, Motor, Clinical embeddings
   - Output: Fused embedding (768-dim) + attention weights for explainability

3. **Patient Similarity Graph Neural Network**:
   - Graph construction: Cosine similarity between patient embeddings (threshold=0.7)
   - GNN architecture: 3-layer GraphSAGE with mean aggregation
   - Node features: Fused embeddings from step 2
   - Output: Refined patient embedding (768-dim)

4. **Multi-Task Output Heads**:
   - **Classification Head** (Amyloid Positivity): 3-layer MLP → Sigmoid
     - Output: Probability ∈ [0, 1]
   - **Regression Head** (MMSE Trajectory): 3-layer MLP
     - Output: MMSE slope (points/year)
   - **Survival Head** (Time to Progression): 2-layer MLP
     - Output: [Cox risk score, predicted survival time]

**Total Parameters**: ~60M (estimate)  
**Inference Time**: <500ms per batch (batch_size=32) on CPU, <100ms on GPU

---

### Datasets

**Primary Training Dataset: ADNI (Alzheimer's Disease Neuroimaging Initiative)**:
- **Source**: https://adni.loni.usc.edu/
- **Cohort**: ~1,100 MCI patients with longitudinal follow-up
- **Data Collected**:
  - Clinical: MMSE, CDR, ADAS-Cog scores (every 6 months for up to 10 years)
  - Biomarkers: CSF Aβ42, tau, pTau-181 (proxy for pTau-217)
  - Genetics: APOE ε4 genotype
  - Imaging: MRI (T1-weighted), PET (amyloid, FDG) [optional for Phase 1]
- **Access**: Data Use Agreement required (1-2 weeks approval)

**Fine-Tuning Dataset: Bio-Hermes**:
- **Source**: Global Alzheimer's Platform Foundation (https://globalalzplatform.org/)
- **Cohort**: ~300-500 patients (MCI + early AD)
- **Data Collected**:
  - Biomarkers: Roche Elecsys plasma pTau-217 (key alignment with Roche portfolio)
  - Digital: Linus Health digital cognitive tests, speech analysis, wearable sensors
  - Clinical: Standard cognitive assessments
- **Strategic Importance**: 
  - Roche is a partner in Bio-Hermes-002 study
  - Training on Roche's own assay data = scientific validation + business alignment
- **Access**: Bio-Hermes-001 released August 2025 (AD Workbench), Bio-Hermes-002 pending

**Auxiliary Dataset: DementiaBank (for digital biomarker synthesis)**:
- **Source**: TalkBank Project (https://dementia.talkbank.org/)
- **Purpose**: Extract acoustic features from speech recordings
- **Cohort**: ~200 AD patients, ~100 controls
- **Format**: Audio (.wav) + transcripts (.cha)

---

### Development Technology Stack

| Component | Technology | Version | Justification |
|-----------|------------|---------|---------------|
| **Deep Learning Framework** | PyTorch | 2.1.2 | Industry standard, dynamic graphs for GNNs |
| **Graph Neural Networks** | PyTorch Geometric | 2.5.0 | Best-in-class GNN library, GraphSAGE implementation |
| **Audio Processing** | Librosa, SpeechBrain | 0.10.1, 0.5.16 | Acoustic feature extraction (jitter, shimmer, MFCC) |
| **NLP** | Spacy, Transformers | 3.7.2, 4.36.2 | Semantic density, language features |
| **Medical Standards** | fhir.resources (Pydantic) | 7.1.0 | FHIR R4 compliance, input validation |
| **Survival Analysis** | Lifelines, scikit-survival | 0.28.0, 0.22.2 | Cox Proportional Hazards, Kaplan-Meier |
| **Explainability** | SHAP, Captum | 0.44.1, 0.7.0 | Model interpretability for FDA/MDR |
| **API Framework** | FastAPI | 0.108.0 | Async support, auto-generated OpenAPI docs |
| **Containerization** | Docker | 24.0+ | Reproducibility, deployment portability |
| **Experiment Tracking** | Weights & Biases | 0.16.2 | Hyperparameter logging, model versioning |
| **Testing** | pytest, pytest-cov | 7.4.4, 4.1.0 | Unit/integration testing, code coverage |

**Infrastructure**:
- **Development**: AWS p3.8xlarge (4× V100 16GB) or equivalent
- **Training**: AWS p4d.24xlarge (8× A100 40GB) or on-premise cluster
- **Production**: CPU-based inference (cost-effective) with GPU acceleration option

---

## Regulatory Compliance Framework

### FDA 510(k) De Novo Pathway

**Classification Rationale**:
- **Product Code**: QFI (Clinical Decision Support Software)
- **Device Class**: Class II (moderate risk)
- **Predicate Device**: Prenosis Sepsis ImmunoScore (K213371) - AI-based risk stratification using biomarkers + clinical data

**Regulatory Controls**:
1. **General Controls**: Labeling, adverse event reporting, establishment registration
2. **Special Controls**:
   - Software validation per IEC 62304
   - Clinical performance data (ADNI + Bio-Hermes validation studies)
   - Cybersecurity documentation (FDA Cybersecurity Guidance 2023)
   - Labeling requirements (intended use, limitations, user training)

**Submission Timeline**:
- **Month 13**: FDA Q-Sub (Pre-Submission meeting request)
- **Month 14**: Prepare De Novo classification request
- **Month 15**: Submit De Novo application
- **Month 16-21**: FDA review (6-month average)

---

### EU MDR Class IIa

**Classification Justification** (MDR 2017/745, Annex VIII, Rule 11):
- Software providing information for diagnostic/therapeutic decisions → Class IIa (not life-critical → not Class IIb/III)

**Requirements**:
1. **Technical Documentation** (MDR Annex II):
   - Design History File (DHF)
   - Risk Management File (ISO 14971)
   - Clinical Evaluation Report (MEDDEV 2.7/1 rev 4)
   - Software validation documentation (IEC 62304)
2. **Quality Management System**: ISO 13485:2016
3. **Notified Body Assessment**: Required for Class IIa
   - Potential Notified Bodies: TÜV SÜD, BSI Group, DEKRA
   - Timeline: 6-12 months for initial conformity assessment

---

### IEC 62304 Software Lifecycle

**Safety Classification**: Class B (non-serious injury possible)
- Justification: Incorrect prediction could delay diagnosis but not immediately life-threatening

**Class B Requirements**:
- Detailed design documentation (Software Architecture Document)
- Software unit testing with traceability
- Software integration testing
- Risk management integration (ISO 14971)
- Change control process

**Phase-by-Phase IEC 62304 Compliance**:
- **Phase 1**: Software Development Planning (5.1), Requirements Analysis (5.2), Architectural Design (5.3)
- **Phase 2**: Detailed Design (5.4), Unit Implementation & Verification (5.5)
- **Phase 3**: Integration Testing (5.6), System Testing (5.7), Release (5.8)

---

### ISO 14971 Risk Management

**Risk Management Activities**:

1. **Hazard Identification** (Phase 1, Week 3):
   - Brainstorming: Clinical team + ML team
   - Categories: Information hazards, operational hazards, cybersecurity hazards

2. **Risk Analysis** (Phase 1, Week 3):
   - Severity scale: Critical / Serious / Moderate / Minor
   - Probability scale: Frequent / Probable / Occasional / Remote / Improbable
   - Risk matrix: Unacceptable / ALARP / Acceptable

3. **Risk Control** (Phase 1-2):
   - Design mitigations (e.g., confidence intervals, out-of-distribution detection)
   - Implementation mitigations (e.g., input validation, unit tests)
   - Labeling mitigations (e.g., "CDS, not standalone diagnostic" warning)

4. **Residual Risk Evaluation** (Phase 3):
   - Post-mitigation risk assessment
   - Risk-benefit analysis (clinical benefit > residual risk)

5. **Post-Market Surveillance** (Post-launch):
   - Adverse event monitoring
   - Performance trending (drift detection)
   - Annual safety reviews

**Key Hazards Identified**:
- H1.1: False Negative (high-risk patient predicted low-risk) → Severity: Serious, Probability: Medium
- H1.2: False Positive (low-risk patient predicted high-risk) → Severity: Moderate, Probability: Medium
- H4.1: Model bias (underrepresented demographics) → Severity: Serious, Probability: Medium

---

## Performance Targets & Validation Criteria

### Clinical Performance Metrics

**Primary Endpoint: Amyloid Positivity Classification**:
- **Target AUC**: ≥0.85 (95% CI: 0.82-0.88)
- **Sensitivity**: ≥0.80 (to minimize false negatives)
- **Specificity**: ≥0.75
- **Benchmark**: Published AD blood biomarker studies (pTau-217 alone: AUC 0.82-0.88)

**Secondary Endpoint: MMSE Trajectory Prediction**:
- **Target RMSE**: ≤3.0 points/year (95% CI: 2.5-3.5)
- **R²**: ≥0.60

**Tertiary Endpoint: Survival Analysis**:
- **Concordance Index (C-index)**: ≥0.75
- **Calibration**: Observed vs. predicted progression rates within 5% at 12, 24 months

### Subgroup Performance (Fairness)

**Requirement**: Performance gap <0.05 AUC across subgroups
- Age groups: 50-65, 65-75, 75-90
- Sex: Male vs. Female
- APOE ε4 status: 0, 1, 2 alleles
- Race/ethnicity (if available in data): White, Black, Hispanic, Asian

**Fairness Metrics**:
- Demographic parity: P(Ŷ=1|Male) ≈ P(Ŷ=1|Female) (within 0.05)
- Equal opportunity: Sensitivity gap across subgroups <0.05

### Technical Performance Metrics

**Inference Latency**:
- **p50**: <1.0s
- **p95**: <2.0s (hard requirement for FDA submission)
- **p99**: <3.0s

**Throughput**:
- Single CPU instance: ≥100 requests/hour
- Single GPU instance: ≥500 requests/hour

**Resource Utilization**:
- CPU usage: <80% at peak load
- GPU memory: <8GB (to support mid-tier GPUs like NVIDIA T4)
- RAM: <16GB per instance

---

## Integration with Roche Navify Ecosystem

### Navify Algorithm Suite Architecture

**Integration Points**:
1. **Navify Hub** (cloud-based orchestration layer)
   - Receives HL7 v2.x messages from hospital LIS
   - Converts to FHIR R4 resources
   - Routes to NeuroFusion-AD microservice
   - Receives FHIR RiskAssessment response
   - Converts back to HL7 for EMR delivery

2. **Navify Integrator** (edge device, optional)
   - On-premise deployment for hospitals with data residency requirements
   - Local data pseudonymization before cloud transmission
   - Fallback processing if cloud connectivity lost

**Data Flow**:
```
[Hospital LIS] → HL7 ORU → [Navify Integrator] → FHIR Observation → [Navify Hub]
                                                                          ↓
                                                          [NeuroFusion-AD Microservice]
                                                                          ↓
[EMR Display] ← HL7 ORU ← [Navify Hub] ← FHIR RiskAssessment
```

### FHIR R4 API Specification

**Endpoint**: `POST /fhir/RiskAssessment/$process`

**Input**: FHIR Parameters resource containing:
- Patient (demographics)
- Observations (pTau-217, Aβ42/40, NfL)
- QuestionnaireResponses (acoustic, motor digital biomarkers)

**Output**: FHIR RiskAssessment resource with:
- Prediction fields (probabilityDecimal, qualitativeRisk)
- Basis references (links to input Observations)
- Extensions (attention weights for explainability)
- Note (human-readable summary)

**Authentication**: OAuth 2.0 Bearer Token (scope: `patient/*.read`, `observation/*.read`)

**Rate Limiting**: 100 requests/hour per API key

---

## Team Structure & Roles

### Core Team (6 FTE)

1. **ML Architect / Tech Lead** (100% FTE)
   - Owns model architecture, training strategy, performance optimization
   - Reviews all code contributions
   - PhD in ML/AI + 5+ years experience
   - Expert in GNNs, PyTorch, healthcare AI

2. **Clinical Domain Specialist** (50% FTE)
   - Validates clinical requirements, interprets results
   - Liaises with neurologist advisors
   - Authors clinical sections of regulatory submissions
   - MD or PhD in Neuroscience/Neurology + clinical trials experience

3. **Regulatory Compliance Officer** (40% FTE)
   - Leads FDA/MDR submission strategy
   - Ensures IEC 62304, ISO 14971 compliance
   - Prepares Design History File (DHF)
   - RAC certification + 5+ years medical device regulatory experience

4. **Senior Data Engineer** (100% FTE)
   - Builds data pipelines (ADNI, Bio-Hermes)
   - Implements FHIR/HL7 interoperability
   - Manages data security and pseudonymization
   - 5+ years medical data engineering + FHIR R4 expertise

5. **ML Research Engineer** (100% FTE)
   - Implements encoders, attention, GNN modules
   - Runs training loops, hyperparameter tuning
   - Conducts EDA and feature engineering
   - MS/PhD + 3+ years deep learning implementation

6. **DevOps / MLOps Engineer** (70% FTE)
   - Sets up GPU infrastructure (cloud or on-prem)
   - Configures CI/CD pipelines
   - Implements Docker containerization
   - Manages monitoring and alerting
   - 3+ years DevOps + ML workflow orchestration

### External Advisors

- **Neurologist Advisory Panel** (3-5 clinicians): Monthly reviews, use case validation
- **Roche Technical Liaison**: Navify integration specs, Elecsys assay details
- **Biostatistician Consultant**: Statistical analysis plan, sample size calculations

---

## Budget Estimate

**Phase 1 (4 months)**: $320K
- Personnel (6 FTE × $20K/month avg): $240K
- Cloud infrastructure (AWS p3.8xlarge): $32K
- Dataset access fees (ADNI, Bio-Hermes): $5K
- External advisors (neurologists, biostatistician): $15K
- Software licenses (W&B, cloud storage): $8K
- Contingency (15%): $20K

**Phase 2 (6 months)**: $540K
- Personnel: $360K
- Cloud infrastructure (training on A100s): $120K
- External validation dataset: $10K
- Contingency: $50K

**Phase 3 (6 months)**: $580K
- Personnel: $360K
- Cloud infrastructure (deployment, load testing): $60K
- Regulatory submission fees (FDA: $20K, Notified Body: $50K): $70K
- Security audit & penetration testing: $30K
- Legal review (DUA, IP): $20K
- Contingency: $40K

**Total Project Budget**: $1.44M

**ROI Justification for Roche**:
- Acquisition price (estimated): $10-25M
- Elecsys pTau-217 reagent revenue uplift: $50-100M/year (if deployed to 10% of US neurology practices)
- Strategic value: Fills neurology portfolio gap, competitive differentiation vs. Siemens/Philips

---

## Key Success Metrics

### Technical Milestones

**Phase 1 Exit Criteria**:
- ✅ All datasets accessed and preprocessed (ADNI n≥1000, Bio-Hermes when available)
- ✅ Model architecture implemented and unit tested
- ✅ Baseline training achieves >70% accuracy
- ✅ SRS, SAD, RMF documents approved

**Phase 2 Exit Criteria**:
- ✅ Classification AUC ≥0.85
- ✅ Regression RMSE ≤3.0
- ✅ Survival C-index ≥0.75
- ✅ Subgroup performance gap <0.05
- ✅ Explainability artifacts (SHAP, attention) generated

**Phase 3 Exit Criteria**:
- ✅ Navify FHIR API integration tested and validated
- ✅ Docker container deployed and stress-tested (1000 concurrent requests)
- ✅ Security audit passed (penetration testing, HIPAA compliance)
- ✅ FDA De Novo submission filed, MDR technical file submitted to Notified Body

### Business Milestones

- **Month 12**: Roche partnership discussions initiated (technical due diligence)
- **Month 15**: LOI (Letter of Intent) signed
- **Month 18**: Acquisition closed OR licensing agreement finalized

---

## Risk Register (Project Risks, Not Clinical Risks)

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Bio-Hermes dataset access delayed | High | Medium | Use DementiaBank synthetic features, document limitation |
| ADNI data quality issues (missing biomarkers) | Medium | Medium | Imputation strategy, sensitivity analysis on completeness |
| FDA rejects De Novo pathway (requires PMA) | Critical | Low | Early FDA Q-Sub to confirm pathway; predicate analysis |
| Model fails to achieve AUC >0.85 | High | Low | Ensemble methods, larger training dataset, architecture search |
| Roche changes Navify API specifications | Medium | Low | Maintain abstraction layer, version API contracts |
| Key personnel turnover (ML Architect) | High | Low | Knowledge transfer docs, pair programming, backup lead |
| Regulatory submission timeline extends | Medium | Medium | Buffer 3 months in project plan; parallel FDA/MDR prep |

---

## Phase-Specific Document Links

- **[Phase 1: Requirements, Architecture & Data Pipeline (Months 1-4)](PHASE_1_COMPREHENSIVE.md)**
- **[Phase 2: Model Training & Validation (Months 5-10)](PHASE_2_COMPREHENSIVE.md)**
- **[Phase 3: Integration, Testing & Regulatory Submission (Months 11-16)](PHASE_3_COMPREHENSIVE.md)**

---

## Appendices

### Appendix A: Technology Decision Matrix

| Decision | Options Considered | Selected | Rationale |
|----------|-------------------|----------|-----------|
| DL Framework | TensorFlow, PyTorch, JAX | PyTorch | Best GNN support (PyG), industry standard, dynamic graphs |
| GNN Library | DGL, PyG, Spektral | PyTorch Geometric | Largest community, GraphSAGE implementation, active development |
| API Framework | Flask, Django, FastAPI | FastAPI | Async support, auto OpenAPI docs, Pydantic validation |
| Deployment | Serverless (Lambda), Kubernetes, Docker Compose | Kubernetes (+ Docker Compose dev) | Scalability, self-healing, industry standard |
| Experiment Tracking | TensorBoard, MLflow, W&B | Weights & Biases | Best UI, hyperparam logging, model versioning |

### Appendix B: Glossary

- **ADNI**: Alzheimer's Disease Neuroimaging Initiative
- **Bio-Hermes**: Biomarker-Digital Hybrid Study (GAP Foundation)
- **CDS**: Clinical Decision Support
- **De Novo**: FDA pathway for novel devices without valid predicates
- **DHF**: Design History File (IEC 62304 required documentation)
- **FHIR**: Fast Healthcare Interoperability Resources (HL7 standard)
- **GNN**: Graph Neural Network
- **MCI**: Mild Cognitive Impairment
- **MDR**: Medical Device Regulation (EU)
- **pTau-217**: Phosphorylated tau protein at threonine 217
- **SaMD**: Software as a Medical Device
- **SHAP**: SHapley Additive exPlanations (interpretability method)

### Appendix C: References

1. Roche Strategic Analysis (Uploaded PDF)
2. IEC 62304:2006 - Medical Device Software Lifecycle Processes
3. ISO 14971:2019 - Medical Devices Risk Management
4. FDA Guidance: Clinical Decision Support Software (2022)
5. EU MDR 2017/745
6. FHIR R4 Specification: http://hl7.org/fhir/R4/
7. Roche Navify Algorithm Suite Developer Guide (request access)
8. König et al. (2015). "Automatic speech analysis for the assessment of patients with predementia and Alzheimer's disease." *Alzheimer's & Dementia*
9. Palmqvist et al. (2020). "Performance of Fully Automated Plasma Assays as Screening Tests for Alzheimer Disease–Related β-Amyloid Status." *JAMA Neurology*

---

**END OF OVERVIEW DOCUMENT**

*For detailed week-by-week breakdowns, see individual phase documents.*
