# NeuroFusion-AD: Comprehensive Requirements Document

**Project**: Multimodal Graph Neural Network for Alzheimer's Disease Progression Prediction  
**Target**: Roche Information Solutions Acquisition  
**Document Version**: 1.0  
**Date**: February 15, 2026  
**Status**: Ready for Phase 1 Development

---

## TABLE OF CONTENTS

1. [Project Vision & Business Case](#1-project-vision--business-case)
2. [Clinical Workflow Scenarios](#2-clinical-workflow-scenarios)
3. [Current Diagnostic Pathway Pain Points](#3-current-diagnostic-pathway-pain-points)
4. [Desired Features & Capabilities](#4-desired-features--capabilities)
5. [Constraints & Non-Negotiable Requirements](#5-constraints--non-negotiable-requirements)
6. [Raw Requirements List (50-100 items)](#6-raw-requirements-list)
7. [MoSCoW Priority Analysis](#7-moscow-priority-analysis)

---

## 1. PROJECT VISION & BUSINESS CASE

### 1.1 Strategic Vision

NeuroFusion-AD addresses a critical inflection point in Alzheimer's disease care. The recent FDA approval of disease-modifying therapies (lecanemab, donanemab) has created an unprecedented demand for accurate, accessible disease detection—yet clinicians lack an efficient tool to triage the millions of patients with memory complaints.

**Core Value Proposition**:

- **For Primary Care Physicians (PCPs)**: A low-cost, non-invasive digital screener to identify high-risk patients requiring confirmatory testing
- **For Neurologists**: Integration with advanced fluid biomarkers to enable early diagnosis and personalized prognostication
- **For Roche**: A strategic asset that drives adoption of Elecsys pTau-217 assays, fills the neurology portfolio gap, and positions Navify as the operating system for cognitive care

### 1.2 Market Context

**The Unmet Need**:

- ~6 million Americans with Alzheimer's or MCI; only 10% diagnosed
- Primary care physicians test only 1-2% of symptomatic patients (overwhelmed, lacking tools)
- Liquid biomarkers (Roche pTau-217) are poised to replace PET scans as the gold standard for amyloid screening
- Digital tools can extend reach beyond memory clinics to primary care

**Competitive Landscape**:

- **Siemens**: Focused on imaging AI (AI-Rad Companion). Weak in wet lab integration
- **Philips**: Workflow orchestration (IntelliSpace). No Alzheimer's specific solution
- **Evidencio**: Simple risk calculators (no automation, low clinical adoption)
- **Altoida**: AR-based motor tasks. Pilots with Roche but lacks integrated platform

**Roche's Strategic Opportunity**:

- Owns the biomarker assays (Elecsys pTau-217, Aβ42/40)
- Has Navify platform for deployment
- Lacks integrated "Digital Companion" for Neurology (unlike oncology with Tumor Board)
- Bio-Hermes partnership validates Roche's commitment to digital biomarkers

### 1.3 Business Model

**Revenue Driver**: Reagent Pull-Through

- Algorithm recommends Elecsys pTau-217 testing for high-risk patients flagged by digital screening
- 1 test per year per at-risk patient identified = 100-500K tests/year at $200/test = $20-100M annual reagent revenue
- Acquisition price: $10-25M (12-18 month payback)

**Deployment Path**: Navify Algorithm Suite

- Integration = "plug-and-play" addition to existing hospital workflows
- No additional hardware required
- Can run on premise (Navify Integrator edge) or cloud (AWS-backed)

### 1.4 Clinical Impact

**Patient Population**: Adults 50-90 years old with Mild Cognitive Impairment (MCI)

- **Screening Role**: Identifies 30-40% of MCI patients at high risk of progression to dementia within 24 months
- **Monitoring Role**: Tracks treatment response in patients on disease-modifying therapies
- **Expected Accuracy**: AUC 0.85-0.90 (diagnostic discrimination), C-index 0.75+ (survival prediction)

**Clinical Workflow Transformation**:

- Reduces unnecessary testing by 40% (lower false positive rate vs. symptom-based screening)
- Accelerates diagnosis by 6-12 months (enables early intervention)
- Improves treatment adherence (clearer prognostic information)

---

## 2. CLINICAL WORKFLOW SCENARIOS

### Scenario 1: Primary Care Triage (The "Screener" Role)

**Actor**: Primary Care Physician at community practice
**Context**: 62-year-old patient presents with wife's concern about mild memory loss over past 6 months; word-finding difficulty; still independent with ADLs

**Current State (Without NeuroFusion-AD)**:

1. PCP performs informal cognitive screening (asks 3 questions, feels uncertain)
2. Considers referral to neurology → 6-month wait, 150-mile drive for patient
3. Does nothing, documents "cognitive concern" in chart
4. Patient potentially misses critical diagnostic window

**Future State (With NeuroFusion-AD)**:

1. **Office Visit** (~2 minutes):
   - Patient takes 2-minute voice recording via clinic tablet (Cookie Theft picture description task)
   - Patient walks 10 steps while accelerometer records gait
   - EHR auto-populates: age, education, APOE result (if available)

2. **NeuroFusion-AD Processing** (real-time):
   - Extracts acoustic features from voice: jitter (pitch variation), shimmer (amplitude variation), semantic density
   - Extracts motor features: gait speed, stride variability, double support time
   - Fuses with MMSE score from EHR
   - Produces risk score: **"High Risk of Amyloid Pathology (78% probability)"**

3. **Clinical Decision Support Output**:
   - Risk stratification: **HIGH (patient is in top 20% risk for progression)**
   - Predicted MMSE decline: **1.2 points/year** (without treatment)
   - Recommended action: **Order Elecsys pTau-217 + Aβ42/40 plasma assay**
   - Confidence interval and explainability: "High risk driven 60% by elevated plasma pTau-217 (if test ordered) and 40% by reduced speech fluency"

4. **Outcome**:
   - Blood test results back in 24 hours
   - If positive: Referral to neurology for amyloid PET confirmation + disease-modifying therapy eligibility assessment
   - If negative: Reassurance, plan to re-screen in 1 year
   - **Time-to-diagnosis reduced from 6 months to 2 weeks**

### Scenario 2: Neurologist Staging & Prognostication (The "Optimizer" Role)

**Actor**: Neurologist in academic medical center specializing in dementia
**Context**: 68-year-old patient with MCI diagnosed 2 years ago; now presenting for assessment of progression risk and treatment planning

**Current State (Without NeuroFusion-AD)**:

1. Clinician reviews: MMSE (25→23 over 2 years, slow decline), amyloid PET positive, CSF tau elevated
2. Must decide: Amyloid-targeting DMT vs. observation
3. Clinical judgment + intuition (no data-driven prognostic score)
4. Patient uncertainty about treatment benefit

**Future State (With NeuroFusion-AD)**:

1. **Multidisciplinary Visit**:
   - Patient completes full battery:
     - Acoustic testing: 5-minute spontaneous speech recording
     - Motor testing: 2-minute gait & balance assessment via smartphone
     - Labs: Elecsys pTau-217, Aβ42/40, NfL (quantitative neurofilament light) from blood draw
     - Cognitive: MMSE, CDR

2. **NeuroFusion-AD Advanced Analysis**:
   - Integrates all modalities into patient similarity graph
   - Compares to "neighborhood" of 200 similar patients in training dataset
   - Produces personalized trajectory:
     - **Predicted MMSE at 12 months without therapy: 20 (mod dementia)**
     - **Predicted MMSE at 12 months WITH DMT: 23 (stable MCI)**
     - **Benefit: 3-point preservation (~30 months delayed progression)**

3. **Clinical Decision Support Output**:
   - Progression risk trajectory (Kaplan-Meier-style survival curves)
   - Biomarker-driven sub-phenotype: "Tau-driven + speech-decline pattern"
   - Confidence intervals: Shows uncertainty in prediction (±1.5 MMSE points)
   - Explainability: Attention weights show which biomarkers drove the prediction
     - 50% pTau-217 (blood biomarker)
     - 25% speech fluency (digital biomarker)
     - 15% NfL (neurodegeneration marker)
     - 10% motor slowing (digital biomarker)

4. **Shared Decision-Making**:
   - Patient & family see evidence-based progression trajectory
   - Clinician recommends DMT based on amyloid status + high-risk trajectory
   - Patient consents, starts lecanemab infusions
   - NeuroFusion-AD scheduled for 6-month re-assessment to monitor response

### Scenario 3: Treatment Monitoring & Response Assessment (The "Tracker" Role)

**Actor**: Nurse case manager at infusion center
**Context**: 71-year-old patient on month 6 of lecanemab therapy for amyloid-positive MCI; family reports "seems sharper but hard to say"

**Current State (Without NeuroFusion-AD)**:

1. Limited objective data on treatment response
2. Clinician relies on patient/family subjective reports
3. Formal testing (MMSE) done once per year only
4. No early warning for treatment failure or amyloid-related imaging abnormalities (ARIA)

**Future State (With NeuroFusion-AD)**:

1. **Remote Monitoring** (Every 3 months via home assessment):
   - Patient records 2-minute voice sample via smartphone app
   - Patient walks 20 steps, smartphone records gait
   - Labs ordered: Elecsys pTau-217, NfL (digital fluid biomarkers)

2. **NeuroFusion-AD Monitoring Analysis**:
   - Compares current trajectory to baseline
   - Stability index: **"Stable (trajectory unchanged from 3-month prior)"**
   - Flag if declining: "ALERT: Speech fluency declining faster than expected. Review for ARIA-E or treatment tolerance."
   - Quantitative trend: Speech jitter increased 15% (suggests developing cognitive loss)

3. **Clinical Decision Support Output**:
   - Six-month progression trajectory update:
     - **MMSE observed: 24 (stable)** vs. **predicted untreated: 21 (would have declined)**
     - **Biomarker response: pTau-217 declining 10% per 6 months (good response)**
     - **Digital biomarker stability: Speech fluency stable (no change)**
   - Risk assessment: "Low risk of ARIA-related complications based on NfL trajectory"
   - Recommendation: "Continue current DMT. Re-assess in 6 months."

4. **Outcome**:
   - Early detection of non-responders → allows therapy switch
   - Objective data reduces subjective uncertainty
   - **Prevents unnecessary discontinuation** in responders with slow decline trajectory

---

## 3. CURRENT DIAGNOSTIC PATHWAY PAIN POINTS

### 3.1 Diagnostic Bottlenecks

**Pain Point 1: Identification Gap (The "Silent MCI" Problem)**

- **Problem**: ~80% of MCI patients never formally diagnosed; symptoms attributed to "normal aging"
- **Root Cause**: PCPs lack quick, non-invasive screening tool; cognitive testing requires specialist appointment
- **Impact**: Critical diagnostic window missed; patients only diagnosed after progression to dementia
- **NeuroFusion-AD Solution**: 2-minute office-based digital screening identifies at-risk patients before formal neurology referral

**Pain Point 2: Access Inequality (The "Wait-List" Problem)**

- **Problem**: Neurology wait times 6-12 months; rural areas have <1 neurologist per 50K population
- **Root Cause**: Specialist-dependent diagnosis; centralized expertise bottleneck
- **Impact**: Patients with fastest decline wait longest for diagnosis
- **NeuroFusion-AD Solution**: Enables diagnosis in primary care + telehealth settings; scalable to rural areas

**Pain Point 3: Biomarker Uncertainty (The "Grey Zone" Problem)**

- **Problem**: Single biomarkers often discordant (e.g., amyloid PET positive, tau negative)
- **Root Cause**: Amyloid pathology complex; imaging has limited spatial resolution
- **Impact**: Clinician uncertainty in diagnosis; multiple testing rounds needed
- **NeuroFusion-AD Solution**: Multimodal fusion (3 biomarkers + 2 digital modalities) reduces ambiguity; attention mechanism clarifies which features drove decision

**Pain Point 4: Prognostication Uncertainty (The "Which Patients Progress?" Problem)**

- **Problem**: 30% of MCI patients revert to normal cognition; 40% stable for 10+ years; 30% progress quickly. No way to identify which patient is which.
- **Root Cause**: No integrated prognostic algorithm; clinicians use intuition
- **Impact**: Over-treatment in stable patients; under-treatment in rapidly progressing
- **NeuroFusion-AD Solution**: Graph Neural Network learns patient "neighborhoods" to predict individual trajectories; 75+ C-index survival discrimination

### 3.2 Clinical Evidence Gaps

**Pain Point 5: Limited Digital Validation (The "Black Box" Problem)**

- **Problem**: Few digital biomarkers validated in large prospective cohorts; Roche partnership (Bio-Hermes) is novel
- **Root Cause**: Speech/gait analysis requires specialized phonetics expertise
- **Impact**: Digital tools viewed as experimental; not integrated into diagnostic criteria
- **NeuroFusion-AD Solution**: Trained on Bio-Hermes data (Roche-partnered); published validation studies; FDA-cleared algorithm provides clinical legitimacy

**Pain Point 6: Cognitive Testing Inconsistency (The "Subjective" Problem)**

- **Problem**: MMSE, Montreal Cognitive Assessment (MoCA) highly variable with education, language, cultural factors; ~25% misclassification
- **Root Cause**: Pen-and-paper tests lack standardization; examiner-dependent
- **Impact**: Patients misclassified as normal/MCI/dementia based on test only
- **NeuroFusion-AD Solution**: Acoustic & motor biomarkers provide objective, education-independent measurement of cognitive reserve

**Pain Point 7: Fluid Biomarker Accessibility (The "Cost" Problem)**

- **Problem**: Elecsys pTau-217 assay $150-250/test; requires specialized lab; not covered by all insurers
- **Root Cause**: Complex assay, limited high-throughput capacity
- **Impact**: Unnecessary testing ordered (cost), or diagnostic testing delayed (access)
- **NeuroFusion-AD Solution**: Digital screener (cost ~$5) triages which patients NEED biomarker testing, reducing unnecessary testing by 40%

### 3.3 Treatment Planning Gaps

**Pain Point 8: Baseline Severity Heterogeneity (The "Eligibility" Problem)**

- **Problem**: Disease-modifying therapies (DMT) have different efficacy by stage: DMT in MCI effective (LCIBB trial); less so in early AD
- **Root Cause**: Clinical staging (CSF amyloid, imaging) does not perfectly align with cognitive reserve
- **Impact**: Clinicians uncertain about DMT candidacy for borderline cases
- **NeuroFusion-AD Solution**: Multimodal assessment clarifies stage & capacity for benefit; informs shared decision-making

**Pain Point 9: No Objective Response Monitoring (The "Are We Helping?" Problem)**

- **Problem**: Treatment response assessed subjectively; formal testing once per year only
- **Root Cause**: No scalable way to measure subtle cognitive/biomarker changes between office visits
- **Impact**: Non-responders identified too late; unnecessary therapy continuation
- **NeuroFusion-AD Solution**: Home-based remote monitoring (voice, gait via smartphone) enables monthly assessment; early detection of non-response

**Pain Point 10: Adverse Event Under-Detection (The "ARIA Risk" Problem)**

- **Problem**: Amyloid-related imaging abnormalities (ARIA) can develop silently; detected only on routine MRI or when patient symptomatic
- **Root Cause**: MRI is expensive ($2K), not done frequently; subjective symptom reporting unreliable
- **Impact**: ARIA misses lead to ICH, cognitive worsening; missed opportunity to pause therapy
- **NeuroFusion-AD Solution**: NfL biomarker trajectory + speech fluency decline can flag early neurotoxicity; prompts earlier imaging assessment

---

## 4. DESIRED FEATURES & CAPABILITIES

### 4.1 Core Functional Features

**Feature 1: Multimodal Data Ingestion**

- **Requirement**: Accept structured data from 4 modalities simultaneously
  - **Fluid**: Plasma pTau-217, Aβ42/40 ratio, NfL (from Roche cobas analyzer via LIS interface)
  - **Acoustic**: Voice recording (10-60 seconds, .wav format) with auto-extraction of jitter, shimmer, MFCC
  - **Motor**: Accelerometer data (10-30 seconds, smartphone motion sensors) with auto-extraction of gait parameters
  - **Clinical**: Age, sex, education, APOE ε4 genotype, MMSE score (from EHR via FHIR)
- **Capability**: Real-time or batch processing; asynchronous job handling for long-running analyses

**Feature 2: Automated Feature Engineering**

- **Requirement**: Extract domain-specific features from raw inputs without manual intervention
  - **From Fluid**: Z-score normalization, log transformation, ratio calculations
  - **From Acoustic**: Librosa + SpeechBrain feature extraction (pitch, formants, spectral centroid, MFCC, speaking rate, pause patterns, semantic density via NLP)
  - **From Motor**: IMU-based gait segmentation, stance/swing phase detection, stride length estimation, turn variability, double support time
  - **From Clinical**: APOE risk encoding (0/1/2 ε4 alleles), sex/age group stratification
- **Capability**: Handles missing values (median imputation), outliers (robust scaling), and data quality issues

**Feature 3: Cross-Modal Attention Fusion**

- **Requirement**: Intelligently weight importance of different modalities for each patient
  - **Mechanism**: 8-head multi-head attention where fluid biomarkers are "query" and acoustic/motor/clinical are "keys/values"
  - **Output**: Attention weights (e.g., "60% pTau-217 driven, 25% speech fluency, 10% motor") for explainability
  - **Behavior**: Automatically up-weight digital modalities if biomarker is borderline; down-weight if biomarker is clearly abnormal
- **Capability**: Produces interpretable, clinician-friendly attention allocation

**Feature 4: Patient Similarity Graph Construction**

- **Requirement**: Build patient "neighborhoods" for prognostic inference
  - **Method**: Cosine similarity between patient embeddings from attention fusion; threshold = 0.7
  - **Graph**: 200-500 neighbors per patient (customizable density)
  - **Features**: Graph edge weights encode patient-to-patient similarity; node features are multimodal embeddings
- **Capability**: Dynamically constructs graph per inference request (no static, stale neighbors)

**Feature 5: Multi-Task Output Generation**

- **Requirement**: Produce 3 complementary clinical predictions simultaneously
  - **Task 1: Classification** → Probability of amyloid positivity (0-100%) with confidence interval
  - **Task 2: Regression** → Predicted MMSE decline rate (points/year) with ±σ uncertainty
  - **Task 3: Survival** → Risk score for progression to dementia within 12/24 months (0-100%); Kaplan-Meier curve
- **Capability**: Multi-task learning ensures predictions are internally consistent & complementary

**Feature 6: Explainability & Transparency**

- **Requirement**: Produce clinician-interpretable explanations for every prediction
  - **Mechanism 1**: Attention weights (which modalities mattered most?)
  - **Mechanism 2**: SHAP values (which specific features within each modality?)
  - **Mechanism 3**: Similar patient cases (show 3-5 most similar training patients & their outcomes)
  - **Mechanism 4**: Confidence intervals (quantify model uncertainty)
- **Capability**: Outputs explanations in clinician-friendly format (no ML jargon); allows clinician override if needed

**Feature 7: FHIR R4 Integration**

- **Requirement**: Seamless integration with hospital EHR/LIS via HL7 FHIR standards
  - **Input**: Accept FHIR Observation (biomarkers), Patient (demographics), QuestionnaireResponse (digital tests) resources
  - **Output**: Produce FHIR RiskAssessment resource with structured prediction data
  - **Protocol**: HTTPS/JSON REST API; can also accept HL7 v2.x via legacy adapter
- **Capability**: EMR-agnostic (works with Epic, Cerner, Meditech); hospital IT does one integration, not per-algorithm

**Feature 8: Real-Time Clinical Alerts**

- **Requirement**: Flag abnormal or unexpected findings automatically
  - **Alert Types**:
    - HIGH RISK: Probability of progression >70%, recommend urgent DMT evaluation
    - DISCORDANT: Fluid vs. digital biomarkers contradictory, recommend further testing
    - DECLINING: Patient trajectory worsening faster than predicted, review for non-response or ARIA
    - ADVERSE: NfL rising steeply, recommend neuroimaging to rule out ARIA-E
  - **Severity**: Categorized by urgency (red/yellow/green) for triage
- **Capability**: Alerts embedded in EHR inbox or Navify dashboard; customizable thresholds

### 4.2 Non-Functional Features (Performance, Security, Usability)

**Feature 9: High-Performance Inference**

- **Requirement**: Sub-2-second response time for clinical use
  - **Target**: p95 latency <2.0s, p99 <3.0s
  - **Throughput**: >100 predictions/hour on single CPU instance
  - **Scalability**: Auto-scales to 10 replicas under load (Kubernetes HPA)
- **Capability**: Optimization via batch inference, GPU acceleration (optional), model quantization

**Feature 10: HIPAA & GDPR Compliance**

- **Requirement**: Data security & privacy by design
  - **Encryption at rest**: AES-256 for PostgreSQL audit logs, model checkpoints
  - **Encryption in transit**: TLS 1.3 for all HTTPS endpoints
  - **Pseudonymization**: Patient identifiers stripped before model inference (session ID only)
  - **Audit trails**: Immutable logs of every prediction request (7-year retention)
  - **Right to erasure**: Automated anonymization procedure for GDPR compliance
- **Capability**: SOC 2 Type II certified; passes penetration testing

**Feature 11: Monitoring & Observability**

- **Requirement**: Continuous monitoring of system health & model performance
  - **Metrics**: Request latency, throughput, error rates, model prediction drift
  - **Dashboards**: Real-time Prometheus/Grafana dashboards for ops team
  - **Alerts**: PagerDuty integration for latency threshold breaches, model degradation
  - **Logging**: ELK stack (Elasticsearch/Logstash/Kibana) for centralized log aggregation
- **Capability**: Proactive detection of issues before clinicians encounter them

**Feature 12: Version Control & Reproducibility**

- **Requirement**: Full audit trail of model versions, training data, hyperparameters
  - **Model Registry**: Weights & Biases (W&B) tracks every model checkpoint with metrics
  - **Data Lineage**: DVC (Data Version Control) tracks which dataset version was used for training
  - **Reproducibility**: Docker container captures entire runtime; can re-train identically
  - **Rollback**: Can revert to prior model version if newer version has drift
- **Capability**: Regulatory compliance (FDA/MDR require full version history)

### 4.3 User Experience Features

**Feature 13: Clinician Dashboard**

- **Requirement**: Web interface for reviewing predictions & patient history
  - **Displays**: Patient summary, multimodal biomarkers, prediction scores, attention visualization
  - **Interaction**: Click-through to similar patient cases, adjust thresholds, override prediction if needed
  - **Mobile**: Responsive design for tablet/phone in clinic
- **Capability**: Role-based access (MD vs. RN vs. admin); customizable filters

**Feature 14: Patient Portal**

- **Requirement**: Optionally, patient-facing interface to track own disease trajectory
  - **Displays**: Simplified risk score, predicted progression path, treatment recommendations
  - **Interaction**: Enroll in remote monitoring, submit voice/gait samples from home
  - **Educational**: Links to Alzheimer's Association resources, clinical trial finder
- **Capability**: Optional; enhances engagement but not required for v1.0

**Feature 15: Admin & Configuration Panel**

- **Requirement**: System administrators configure thresholds, toggle features, manage users
  - **Controls**: Risk score thresholds (e.g., define HIGH as >70%), alert routing, data retention policies
  - **Audit**: User access logs, configuration change history
- **Capability**: Role-based access control (RBAC); granular permissions

---

## 5. CONSTRAINTS & NON-NEGOTIABLE REQUIREMENTS

### 5.1 Regulatory Constraints

**Constraint 1: FDA 510(k) De Novo Pathway Compliance**

- **Requirement**: Product must be engineered from inception to pass FDA review
  - **IEC 62304** Software Development Lifecycle compliance (documentation-heavy)
  - **ISO 14971** Risk Management compliance (hazard analysis, FMEA, mitigation)
  - **Software Validation** (V&V): Unit, integration, system, and clinical validation
  - **Cybersecurity**: FDA guidance on software cybersecurity (authentication, encryption, audit trails)
- **Non-Negotiable**: Cannot cut corners on documentation or testing rigor; adds 3-4 months to schedule

**Constraint 2: EU MDR Class IIa Compliance**

- **Requirement**: EU deployment requires Technical File + Notified Body review
  - **Design History File (DHF)**: 200+ pages of design specs, test reports, clinical evidence
  - **Clinical Evaluation Report**: Per MEDDEV 2.7/1 template; demonstrates clinical safety/performance
  - **Post-Market Surveillance Plan**: How we monitor after launch
- **Non-Negotiable**: MDR timeline longer than FDA (6-12 months); requires external Notified Body (TÜV SÜD)

**Constraint 3: Clinical Data Quality Standards**

- **Requirement**: Training data must meet clinical trial standards
  - **ADNI**: De-identified, validated, >10 years of longitudinal follow-up; >1,100 MCI patients
  - **Bio-Hermes**: Prospective, Roche-partnered, gold-standard fluid biomarkers (pTau-217)
  - **Data Use Agreements**: Regulatory-compliant; define permitted uses, restrictions
- **Non-Negotiable**: Cannot use unvalidated or proprietary datasets; regulatory approval depends on open-source data sources

### 5.2 Clinical Constraints

**Constraint 4: Clinical Validation Threshold**

- **Requirement**: Algorithm must achieve specified clinical accuracy before any human use
  - **Classification AUC ≥0.85** (sensitivity ≥80%, specificity ≥70% at operating point)
  - **Regression RMSE ≤3.0 MMSE points** (clinically meaningful threshold)
  - **Survival C-index ≥0.75** (discriminative ability for 24-month progression)
  - **Subgroup Performance Gap <0.05** (no racial/gender disparities)
- **Non-Negotiable**: Must validate on independent test set + external cohort; any degradation halts deployment

**Constraint 5: "Aid, Not Replacement" Positioning**

- **Requirement**: Algorithm is Clinical Decision Support (CDS), not diagnostic
  - **Labeling**: All outputs labeled "Clinical Decision Support" with disclaimer
  - **User Manual**: Emphasizes algorithm is aid, not substitute for clinical judgment
  - **Contraindication**: Cannot use for diagnosis in patients <50 or >90 years old (outside training data range)
- **Non-Negotiable**: Liability/regulatory requirement; cannot position as autonomous diagnostic tool

**Constraint 6: Informed Consent & Transparency**

- **Requirement**: Patients must understand algorithm's role, limitations, explainability
  - **Consent**: If used in research, requires IRB approval + patient consent
  - **Explainability**: Every prediction includes confidence interval + explanation of which factors mattered
  - **Fairness**: Validated in African American, Hispanic, Asian subgroups to ensure equitable performance
- **Non-Negotiable**: Ethical requirement; FDA expects transparency

### 5.3 Technical Constraints

**Constraint 7: Navify Algorithm Suite Compatibility**

- **Requirement**: Must integrate seamlessly with Roche's existing Navify ecosystem
  - **FHIR R4 Compliance**: All inputs/outputs must be valid FHIR resources
  - **HL7 v2.x Backward Compatibility**: Must work with legacy hospital systems via adapter
  - **API Contract**: Roche defines exact endpoint specification (/fhir/RiskAssessment/$process); cannot deviate
  - **Latency Budget**: <2 seconds end-to-end (including EHR data fetch + inference + result formatting)
- **Non-Negotiable**: Without Navify compatibility, product has no deployment path at Roche

**Constraint 8: Open Source Data Only**

- **Requirement**: Training/validation cannot use proprietary datasets (regulatory & IP concerns)
  - **ADNI**: Public (Data Use Agreement required but open access)
  - **Bio-Hermes**: Roche partnered; expects open release for publication
  - **DementiaBank**: Public corpus for speech analysis
  - **Cannot use**: Patient data from any commercial EMR vendor, proprietary biomarker datasets
- **Non-Negotiable**: Proprietary data would create dependency on data provider; limits Roche's IP ownership

**Constraint 9: Model Explainability Requirement**

- **Requirement**: Model must be interpretable (not a black box)
  - **No**: Deep recurrent networks or massive transformers that are hard to interpret
  - **Yes**: Graph Neural Networks with attention (inherently interpretable), SHAP-compatible
  - **Output**: Every prediction includes quantitative explanation (attention weights, SHAP values, similar patient cases)
  - **Validation**: Explainability validated in blind review by neurologists (can they understand the explanation?)
- **Non-Negotiable**: FDA/clinicians require transparency; black-box models face regulatory rejection

**Constraint 10: No Real-Time Streaming Data**

- **Requirement**: Algorithm processes discrete snapshots, not continuous data streams
  - **Model Input**: Single time-point (one voice recording, one gait test, one blood draw)
  - **Not Required**: Real-time waveform processing, online learning, continuous model updates
  - **Rationale**: Simplifies regulatory approval; batch inference easier to validate
- **Non-Negotiable**: Streaming data would require different validation framework (more complex)

### 5.4 Data Constraints

**Constraint 11: Missing Data Handling**

- **Requirement**: Model must handle incomplete inputs gracefully
  - **Scenario**: Patient didn't do voice test (no acoustic features) → model proceeds with fluid + motor + clinical
  - **Method**: Median imputation for continuous features; learned embeddings mask missing modalities
  - **Validation**: Test on intentionally incomplete data; document performance degradation by missing modality
- **Non-Negotiable**: Real-world data always has gaps; algorithm must be robust

**Constraint 12: Patient Privacy (Pseudonymization)**

- **Requirement**: Algorithm never sees personally identifying information (PII)
  - **Input**: Data reaches algorithm as {patient_id: "SHA256_HASH_1234", age: 68, pTau: 45.2, ...}
  - **Processing**: Uses only clinical features + hash ID; no names, SSNs, medical record numbers
  - **Output**: RiskAssessment result paired only with hash ID; EHR translation layer maps back to patient
  - **Audit Logs**: Log transaction but not patient identity
- **Non-Negotiable**: HIPAA/GDPR requirement; personal data breach = regulatory violation

### 5.5 Project Constraints

**Constraint 13: 16-Month Development Timeline**

- **Requirement**: Product must be market-ready in 16 months (Phase 1-3)
  - **Phase 1 (4 mo)**: Architecture + data pipelines
  - **Phase 2 (6 mo)**: Model training + validation
  - **Phase 3 (6 mo)**: Integration + regulatory submission
  - **Trade-offs**: Cannot extend timeline; must run phases in series with overlap
- **Non-Negotiable**: Roche acquisition discussion timeline depends on having regulatory submission by month 16

**Constraint 14: $1.44M Total Budget**

- **Requirement**: Cannot exceed budget (constraint drives technology choices, outsourcing decisions)
  - **High-cost items**: Cloud compute (A100 GPUs for training), regulatory consultants, security audit
  - **Budget tradeoffs**: Use open-source frameworks (PyTorch, FastAPI) vs. proprietary ML platforms
- **Non-Negotiable**: Budget is fixed; scope must be managed accordingly

**Constraint 15: Core Team of 6 FTE**

- **Requirement**: Must accomplish all work with small team (no bloat)
  - **Roles**: ML Architect, Clinical Specialist, Regulatory Officer, Data Engineer, Research Engineer, DevOps/MLOps
  - **External Help**: Neurologist advisors (part-time), regulatory consultants (contract)
  - **Knowledge Transfer**: Critical; if key person leaves, project must continue
- **Non-Negotiable**: Cannot significantly expand team; must optimize for efficiency

---

## 6. RAW REQUIREMENTS LIST

### Functional Requirements

#### Data Ingestion & Preprocessing (FR-DIP-001 to FR-DIP-020)

**FR-DIP-001**: System shall accept FHIR Observation resources (LOINC codes for pTau-217, Aβ42/40, NfL) via FHIR API  
**FR-DIP-002**: System shall accept Patient demographic resource (age, sex, education, APOE ε4 genotype)  
**FR-DIP-003**: System shall accept QuestionnaireResponse resources (MMSE, CDR scores)  
**FR-DIP-004**: System shall accept HL7 v2.x messages via legacy adapter for backward compatibility  
**FR-DIP-005**: System shall accept audio files (.wav, 8-16 kHz sampling rate, mono, 10-60 seconds)  
**FR-DIP-006**: System shall accept accelerometer data (smartphone IMU: 3-axis acceleration, 50-100 Hz sampling)  
**FR-DIP-007**: System shall validate FHIR resources against R4 schema before processing  
**FR-DIP-008**: System shall perform Z-score normalization of continuous features using pre-fitted StandardScaler  
**FR-DIP-009**: System shall impute missing continuous values using median of training set  
**FR-DIP-010**: System shall impute missing categorical values using mode of training set  
**FR-DIP-011**: System shall extract acoustic features from audio (jitter, shimmer, MFCC, pitch, formants, speaking rate)  
**FR-DIP-012**: System shall extract motor features from IMU (gait speed, stride length, variability, double support time)  
**FR-DIP-013**: System shall detect and flag outliers (values >3σ from training distribution)  
**FR-DIP-014**: System shall handle missing modalities gracefully (e.g., no audio → proceed with other features)  
**FR-DIP-015**: System shall log all preprocessing steps for audit trail & reproducibility  
**FR-DIP-016**: System shall support both synchronous (immediate response) and asynchronous (job queue) modes  
**FR-DIP-017**: System shall batch preprocess multiple patients (batch_size configurable)  
**FR-DIP-018**: System shall cache preprocessed features for repeated inference requests  
**FR-DIP-019**: System shall reject inputs outside valid ranges (e.g., age <50 or >90, MMSE <0 or >30)  
**FR-DIP-020**: System shall pseudonymize patient data (hash patient ID, strip PII) before model inference

#### Model Inference & Prediction (FR-MIP-001 to FR-MIP-030)

**FR-MIP-001**: System shall implement 4 modality-specific encoders (fluid, acoustic, motor, clinical)  
**FR-MIP-002**: Fluid encoder shall be 3-layer MLP (input=3, output=768-dim embedding)  
**FR-MIP-003**: Acoustic encoder shall be 4-layer MLP (input=15, output=768-dim embedding)  
**FR-MIP-004**: Motor encoder shall be 4-layer MLP (input=20, output=768-dim embedding)  
**FR-MIP-005**: Clinical encoder shall embed categorical features (APOE, sex) then concatenate with numeric features  
**FR-MIP-006**: System shall apply 8-head multi-head attention to fuse embeddings from 4 modalities  
**FR-MIP-007**: Attention query shall be fluid embeddings; keys/values shall be acoustic/motor/clinical  
**FR-MIP-008**: System shall output attention weights (sum-normalized to 1.0) for each modality pair  
**FR-MIP-009**: System shall construct patient similarity graph via cosine similarity (threshold=0.7)  
**FR-MIP-010**: System shall implement 3-layer GraphSAGE GNN with mean aggregation  
**FR-MIP-011**: System shall output refined patient embeddings (768-dim) from GNN  
**FR-MIP-012**: System shall implement classification head (amyloid positivity) → Sigmoid output ∈ [0, 1]  
**FR-MIP-013**: System shall implement regression head (MMSE slope) → continuous output (points/year)  
**FR-MIP-014**: System shall implement survival head (time to progression) → Cox risk score + predicted time  
**FR-MIP-015**: System shall compute confidence intervals (95% CI) for all 3 outputs via Monte Carlo dropout  
**FR-MIP-016**: System shall validate model forward pass with unit tests (n=100 synthetic inputs)  
**FR-MIP-017**: System shall support batch inference (batch_size up to 256)  
**FR-MIP-018**: System shall log prediction request + timestamp + result for audit trail  
**FR-MIP-019**: System shall implement model versioning (track checkpoint hash, training date, metrics)  
**FR-MIP-020**: System shall support model rollback (revert to prior version if newer shows drift)  
**FR-MIP-021**: System shall implement gradient clipping (max norm=1.0) to prevent training instability  
**FR-MIP-022**: System shall apply layer normalization after each modality encoder  
**FR-MIP-023**: System shall apply dropout (p=0.2) in all hidden layers  
**FR-MIP-024**: System shall implement SHAP value computation for feature importance  
**FR-MIP-025**: System shall compute attention weight heatmaps for visualization  
**FR-MIP-026**: System shall identify k-nearest neighbor patients (k=5) in training set for comparison  
**FR-MIP-027**: System shall retrieve training outcomes for similar patients (e.g., progression rate)  
**FR-MIP-028**: System shall compute prediction uncertainty (epistemic + aleatoric)  
**FR-MIP-029**: System shall flag high-uncertainty predictions (confidence <60%)  
**FR-MIP-030**: System shall support ensemble predictions (multiple model checkpoints for robustness)

#### Output & Reporting (FR-OUT-001 to FR-OUT-020)

**FR-OUT-001**: System shall generate FHIR RiskAssessment resource as output  
**FR-OUT-002**: RiskAssessment shall include amyloid positivity probability (0-100%)  
**FR-OUT-003**: RiskAssessment shall include MMSE decline trajectory (points/year, ±σ)  
**FR-OUT-004**: RiskAssessment shall include progression risk score (0-100%)  
**FR-OUT-005**: RiskAssessment shall include 95% confidence intervals for all metrics  
**FR-OUT-006**: RiskAssessment shall include risk stratification category (HIGH/MEDIUM/LOW)  
**FR-OUT-007**: System shall generate attention weight summary (e.g., "60% pTau, 25% speech, 10% gait, 5% demographics")  
**FR-OUT-008**: System shall generate SHAP value explanation (which features drove the prediction)  
**FR-OUT-009**: System shall generate similar patient case summaries (3-5 most similar training patients + outcomes)  
**FR-OUT-010**: System shall generate Kaplan-Meier survival curves (predicted progression trajectory over 24 months)  
**FR-OUT-011**: System shall generate recommended action (e.g., "Order pTau-217 test" or "Continue monitoring")  
**FR-OUT-012**: System shall generate clinical disclaimer ("Aid, not substitute for clinical judgment")  
**FR-OUT-013**: System shall format all outputs in clinician-friendly language (no ML jargon)  
**FR-OUT-014**: System shall support JSON output format (for EHR/API consumption)  
**FR-OUT-015**: System shall support PDF output format (for patient handouts)  
**FR-OUT-016**: System shall support HTML output format (for web dashboard)  
**FR-OUT-017**: System shall embed output timestamp & model version in all reports  
**FR-OUT-018**: System shall link outputs to immutable audit log entry  
**FR-OUT-019**: System shall compute and display result confidence intervals  
**FR-OUT-020**: System shall flag discordant results (e.g., high fluid biomarker but low digital risk) for clinician review

#### Clinical Decision Support & Alerts (FR-CDS-001 to FR-CDS-015)

**FR-CDS-001**: System shall generate HIGH RISK alert if amyloid positivity probability >70%  
**FR-CDS-002**: System shall generate HIGH RISK alert if progression risk score >70%  
**FR-CDS-003**: System shall generate DECLINING alert if MMSE slope worse than predicted by >1.0 points/year  
**FR-CDS-004**: System shall generate DISCORDANT alert if biomarker modalities disagree (e.g., pTau high, speech normal)  
**FR-CDS-005**: System shall generate ADVERSE alert if NfL rising steeply (>0.5 ng/mL per 6 months)  
**FR-CDS-006**: System shall assign severity color (RED=urgent, YELLOW=monitor, GREEN=stable)  
**FR-CDS-007**: System shall include recommended action for each alert type  
**FR-CDS-008**: System shall route alerts to appropriate clinician role (MD for diagnostic decisions, RN for monitoring)  
**FR-CDS-009**: System shall support alert threshold customization by institution  
**FR-CDS-010**: System shall log clinician response to alerts (ignored, acted on, overridden)  
**FR-CDS-011**: System shall suppress duplicate alerts within 24 hours (avoid alert fatigue)  
**FR-CDS-012**: System shall integrate with EHR inbox (Epic SmartText / Cerner alerts)  
**FR-CDS-013**: System shall support PagerDuty escalation for critical alerts (optional)  
**FR-CDS-014**: System shall provide evidence summary for each alert (which factors triggered it)  
**FR-CDS-015**: System shall allow clinician to override alert classification if disputed

#### Integration & API (FR-API-001 to FR-API-025)

**FR-API-001**: System shall expose RESTful API endpoint: POST /fhir/RiskAssessment/$process  
**FR-API-002**: System shall accept FHIR R4 Bundle as request payload  
**FR-API-003**: System shall validate request against FHIR schema before processing  
**FR-API-004**: System shall return FHIR RiskAssessment (or OperationOutcome for errors) as response  
**FR-API-005**: System shall use HTTPS/TLS 1.3 for all API endpoints  
**FR-API-006**: System shall implement OAuth 2.0 authentication (client credentials or JWT)  
**FR-API-007**: System shall implement role-based access control (Clinician, Administrator, Auditor)  
**FR-API-008**: System shall rate-limit API to 100 requests/minute per authenticated user  
**FR-API-009**: System shall implement request tracing (correlate request → processing → audit log)  
**FR-API-010**: System shall support asynchronous processing via job queue (return Job ID, client polls for result)  
**FR-API-011**: System shall support synchronous processing (wait up to 10 seconds for response)  
**FR-API-012**: System shall return HTTP 200 (OK) with result on success  
**FR-API-013**: System shall return HTTP 400 (Bad Request) for invalid FHIR input  
**FR-API-014**: System shall return HTTP 401 (Unauthorized) for authentication failures  
**FR-API-015**: System shall return HTTP 422 (Unprocessable Entity) for valid FHIR but missing required fields  
**FR-API-016**: System shall return HTTP 503 (Service Unavailable) if model is not loaded  
**FR-API-017**: System shall auto-generate OpenAPI 3.0 specification for API  
**FR-API-018**: System shall support FHIR Prefer header (respond-async, return=representation, return=minimal)  
**FR-API-019**: System shall implement Navify API contract compliance (per Roche spec)  
**FR-API-020**: System shall support legacy HL7 v2.x via message transformation gateway  
**FR-API-021**: System shall implement request/response logging (verbose for debugging, minimal for production)  
**FR-API-022**: System shall support CORS for cross-origin browser requests (if web dashboard)  
**FR-API-023**: System shall implement circuit breaker pattern (fail gracefully if dependencies down)  
**FR-API-024**: System shall retry failed requests with exponential backoff (up to 3 attempts)  
**FR-API-025**: System shall publish API metrics (latency, throughput, errors) to Prometheus

### Non-Functional Requirements

#### Performance (NFR-PERF-001 to NFR-PERF-010)

**NFR-PERF-001**: Inference latency p95 <2.0 seconds (end-to-end, including data fetch & formatting)  
**NFR-PERF-002**: Inference latency p99 <3.0 seconds  
**NFR-PERF-003**: Throughput ≥100 predictions/hour on single CPU instance  
**NFR-PERF-004**: Batch processing latency <5 seconds for batch_size=32  
**NFR-PERF-005**: API response time p95 <500ms (excludes network latency)  
**NFR-PERF-006**: Model inference time p95 <100ms on GPU, <500ms on CPU  
**NFR-PERF-007**: Memory usage <2GB per instance during inference  
**NFR-PERF-008**: CPU utilization <80% at peak load (100 req/min)  
**NFR-PERF-009**: Startup time <10 seconds (load model checkpoint + initialize)  
**NFR-PERF-010**: Auto-scaling from 2 to 10 replicas should occur within <30 seconds of load increase

#### Reliability & Availability (NFR-REL-001 to NFR-REL-015)

**NFR-REL-001**: System uptime target: 99.5% per month (acceptable downtime: 3.6 hours)  
**NFR-REL-002**: System shall implement graceful degradation (return cached result if model temporarily unavailable)  
**NFR-REL-003**: System shall implement circuit breaker pattern (stop calling failed service, retry after cooldown)  
**NFR-REL-004**: System shall implement request retry logic (exponential backoff, max 3 attempts)  
**NFR-REL-005**: System shall log all errors with context (request payload, error message, stack trace)  
**NFR-REL-006**: System shall send alerts to ops team for error rate >1%  
**NFR-REL-007**: System shall implement health check endpoint (GET /health → {status: "healthy"})  
**NFR-REL-008**: System shall implement deep health check (verify database, model checkpoint, cache connectivity)  
**NFR-REL-009**: System shall implement automated failover (traffic rerouted to healthy instance within <30s)  
**NFR-REL-010**: System shall backup model checkpoints daily to S3  
**NFR-REL-011**: System shall backup database (PostgreSQL) daily with point-in-time recovery  
**NFR-REL-012**: System shall test backup restore procedure monthly  
**NFR-REL-013**: System shall implement connection pooling (database, cache) to prevent resource exhaustion  
**NFR-REL-014**: System shall implement timeout logic (max 10s per inference request, fail gracefully)  
**NFR-REL-015**: System shall implement persistent job queue (survive service restarts)

#### Security (NFR-SEC-001 to NFR-SEC-020)

**NFR-SEC-001**: All HTTPS endpoints shall use TLS 1.3 (no TLS 1.2 or lower)  
**NFR-SEC-002**: API shall implement OAuth 2.0 (client credentials or JWT) authentication  
**NFR-SEC-003**: API shall implement role-based access control (RBAC) with at least 3 roles (Clinician, Admin, Auditor)  
**NFR-SEC-004**: API shall rate-limit clients (100 req/min per user)  
**NFR-SEC-005**: System shall hash patient IDs using SHA-256 + salt before any processing  
**NFR-SEC-006**: System shall encrypt sensitive data at rest using AES-256 (PostgreSQL, S3)  
**NFR-SEC-007**: System shall encrypt data in transit using TLS 1.3  
**NFR-SEC-008**: System shall implement audit logging (every API call logged with timestamp, user, action)  
**NFR-SEC-009**: System shall make audit logs immutable (append-only, encrypted)  
**NFR-SEC-010**: System shall retain audit logs for ≥7 years (regulatory requirement)  
**NFR-SEC-011**: System shall implement input validation (FHIR schema validation + SQL injection prevention)  
**NFR-SEC-012**: System shall implement output encoding (prevent XSS in web UI)  
**NFR-SEC-013**: System shall not log sensitive data (PII, passwords, authentication tokens)  
**NFR-SEC-014**: System shall implement secrets management (API keys, database passwords in secure vault, not code)  
**NFR-SEC-015**: System shall run as non-root user in Docker container  
**NFR-SEC-016**: System shall scan Docker images for vulnerabilities (Trivy, Snyk)  
**NFR-SEC-017**: System shall implement CORS policy (allow only registered domain)  
**NFR-SEC-018**: System shall implement CSRF protection (token-based for state-changing operations)  
**NFR-SEC-019**: System shall pass penetration testing (external security vendor)  
**NFR-SEC-020**: System shall obtain SOC 2 Type II certification (post-launch)

#### Compliance & Regulatory (NFR-COMP-001 to NFR-COMP-015)

**NFR-COMP-001**: System shall comply with HIPAA Security Rule (encryption, audit trails, access controls)  
**NFR-COMP-001**: System shall comply with HIPAA Privacy Rule (no PHI in logs, data use agreements)  
**NFR-COMP-003**: System shall comply with HIPAA Breach Notification Rule (detect, notify, document incidents)  
**NFR-COMP-004**: System shall comply with GDPR Article 6 (lawful basis for processing patient data)  
**NFR-COMP-005**: System shall comply with GDPR Article 9 (special category data: health data)  
**NFR-COMP-006**: System shall support GDPR right to erasure (anonymize/delete patient data on request)  
**NFR-COMP-007**: System shall maintain Data Processing Agreement (DPA) with Roche  
**NFR-COMP-008**: System shall follow IEC 62304 (medical device software lifecycle)  
**NFR-COMP-009**: System shall follow ISO 14971 (risk management for medical devices)  
**NFR-COMP-010**: System shall provide Design History File (DHF) with >200 pages of documentation  
**NFR-COMP-011**: System shall document all design decisions, test results, risk mitigation  
**NFR-COMP-012**: System shall pass FDA cybersecurity guidance review (vulnerability disclosure, update process)  
**NFR-COMP-013**: System shall provide Clinical Validation Report (clinical performance on ADNI/Bio-Hermes)  
**NFR-COMP-014**: System shall support Post-Market Surveillance (PMS) data collection & analysis  
**NFR-COMP-015**: System shall maintain labeling & user manual (updated for any software changes)

#### Maintainability & Operations (NFR-MAINT-001 to NFR-MAINT-010)

**NFR-MAINT-001**: Source code shall use version control (Git) with meaningful commit messages  
**NFR-MAINT-002**: Source code shall pass linting (Pylint, Flake8) with score >8.0/10  
**NFR-MAINT-003**: Source code shall have ≥80% code coverage (pytest)  
**NFR-MAINT-004**: All functions shall have docstrings (NumPy style)  
**NFR-MAINT-005**: System shall use Docker for reproducible deployment  
**NFR-MAINT-006**: System shall use Kubernetes for orchestration (ingress, scaling, self-healing)  
**NFR-MAINT-007**: System shall use CI/CD pipeline (GitHub Actions / GitLab CI) to automate testing, building, deployment  
**NFR-MAINT-008**: System shall have comprehensive monitoring (Prometheus metrics, Grafana dashboards)  
**NFR-MAINT-009**: System shall have centralized logging (ELK stack or equivalent)  
**NFR-MAINT-010**: System shall document deployment procedures (runbooks for common tasks)

#### Interoperability (NFR-INTER-001 to NFR-INTER-010)

**NFR-INTER-001**: System shall accept FHIR R4 compliant messages  
**NFR-INTER-002**: System shall output FHIR R4 compliant RiskAssessment resource  
**NFR-INTER-003**: System shall support HL7 v2.x for backward compatibility with legacy systems  
**NFR-INTER-004**: System shall implement FHIR Observation mapping (LOINC codes for biomarkers)  
**NFR-INTER-005**: System shall implement FHIR Patient mapping (demographics)  
**NFR-INTER-006**: System shall implement FHIR QuestionnaireResponse mapping (cognitive scores)  
**NFR-INTER-007**: System shall implement FHIR RiskAssessment mapping (predictions)  
**NFR-INTER-008**: System shall support HL7 CCD (Consolidated ClinicalDocument Architecture) export  
**NFR-INTER-009**: System shall integrate with Epic via FHIR API (Roche's preferred EHR)  
**NFR-INTER-010**: System shall integrate with Cerner, Meditech via FHIR adapter layer

#### Explainability & Interpretability (NFR-EXPLAIN-001 to NFR-EXPLAIN-010)

**NFR-EXPLAIN-001**: Every prediction shall include attention weight breakdown (sum to 100%)  
**NFR-EXPLAIN-002**: Every prediction shall include SHAP values for top-5 influential features  
**NFR-EXPLAIN-003**: Every prediction shall include 3-5 similar patient case summaries  
**NFR-EXPLAIN-004**: Explainability shall be validated by clinician blind review (≥80% comprehensibility)  
**NFR-EXPLAIN-005**: System shall provide confidence intervals for all estimates  
**NFR-EXPLAIN-006**: System shall flag high-uncertainty predictions (confidence <60%)  
**NFR-EXPLAIN-007**: System shall provide uncertainty decomposition (epistemic vs. aleatoric)  
**NFR-EXPLAIN-008**: System shall support attention heatmap visualization  
**NFR-EXPLAIN-009**: System shall explain discordant predictions (when modalities disagree)  
**NFR-EXPLAIN-010**: System shall provide feature importance ranking (which variables matter most across cohort)

#### Scalability (NFR-SCALE-001 to NFR-SCALE-005)

**NFR-SCALE-001**: System shall scale horizontally (add replicas) under load via Kubernetes HPA  
**NFR-SCALE-002**: System shall scale from 2 to 10 replicas based on CPU utilization (threshold: 70%)  
**NFR-SCALE-003**: System shall support 1000+ concurrent requests with p95 latency <2.0s  
**NFR-SCALE-004**: Database shall support 10,000 predictions/day without performance degradation  
**NFR-SCALE-005**: System shall implement caching (Redis) to reduce redundant inference

---

## 7. MOSCOW PRIORITY ANALYSIS

The MoSCoW method prioritizes requirements into four categories for v1.0 release:

- **MUST HAVE**: Essential for FDA/MDR approval and minimal clinical utility. Without these, product cannot launch.
- **SHOULD HAVE**: High value, expected by clinicians, but not strictly required for v1.0. Can be added in v1.1.
- **COULD HAVE**: Nice-to-have features for future versions; low clinical impact or high implementation cost.
- **WON'T HAVE**: Explicitly out of scope for v1.0; deferred to post-launch phases.

---

### MUST HAVE Requirements (for FDA/MDR Approval & Minimum Viability)

#### Data Ingestion & Preprocessing

| Req ID     | Requirement                                          | Justification                                              |
| ---------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| FR-DIP-001 | FHIR Observation acceptance (pTau-217, Aβ42/40, NfL) | Core biomarker input; non-negotiable for Roche integration |
| FR-DIP-002 | FHIR Patient resource acceptance (age, sex, APOE)    | Required for risk stratification                           |
| FR-DIP-005 | Audio file acceptance (.wav, 10-60s)                 | Digital biomarker input; key differentiator vs. Evidencio  |
| FR-DIP-006 | Accelerometer data acceptance (smartphone IMU)       | Motor biomarker input; alternative to research gait mats   |
| FR-DIP-007 | FHIR schema validation before processing             | Regulatory requirement (prevent garbage-in, garbage-out)   |
| FR-DIP-008 | Z-score normalization                                | Required for model stability & reproducibility             |
| FR-DIP-009 | Missing value imputation (median)                    | Real-world data always incomplete; algorithm must handle   |
| FR-DIP-020 | Patient pseudonymization (hash ID, strip PII)        | HIPAA/GDPR non-negotiable                                  |

**MUST HAVE Count**: 8 requirements

#### Model Inference & Prediction

| Req ID                   | Requirement                                                    | Justification                                              |
| ------------------------ | -------------------------------------------------------------- | ---------------------------------------------------------- |
| FR-MIP-001 to FR-MIP-010 | Core GNN architecture (4 encoders, attention, GNN, embeddings) | Core model specification; cannot deviate from requirements |
| FR-MIP-012               | Classification head (amyloid positivity) → Sigmoid             | Primary clinical output (diagnosis support)                |
| FR-MIP-013               | Regression head (MMSE slope)                                   | Secondary clinical output (prognosis support)              |
| FR-MIP-014               | Survival head (time to progression)                            | Tertiary clinical output (survival prediction)             |
| FR-MIP-015               | Confidence intervals (95% CI)                                  | FDA requirement (quantify uncertainty)                     |
| FR-MIP-024               | SHAP value computation                                         | Explainability requirement for clinician trust             |
| FR-MIP-025               | Attention weight heatmaps                                      | Explainability + regulatory requirement                    |

**MUST HAVE Count**: 7 requirements (bundled as single architecture block)

#### Output & Reporting

| Req ID                   | Requirement                                     | Justification                                        |
| ------------------------ | ----------------------------------------------- | ---------------------------------------------------- |
| FR-OUT-001 to FR-OUT-010 | FHIR RiskAssessment output with prediction data | Regulatory standard; required for Navify integration |
| FR-OUT-012               | Clinical disclaimer ("Aid, not substitute")     | FDA/liability requirement                            |
| FR-OUT-013               | Clinician-friendly language (no ML jargon)      | Usability requirement for adoption                   |
| FR-OUT-018               | Link outputs to audit log                       | Traceability requirement for regulatory approval     |

**MUST HAVE Count**: 4 requirements (bundled as single output specification)

#### API & Integration

| Req ID                   | Requirement                                                         | Justification                                      |
| ------------------------ | ------------------------------------------------------------------- | -------------------------------------------------- |
| FR-API-001               | POST /fhir/RiskAssessment/$process endpoint                         | Required for Navify integration (API contract)     |
| FR-API-002               | Accept FHIR R4 Bundle payload                                       | Regulatory standard                                |
| FR-API-005               | HTTPS/TLS 1.3                                                       | HIPAA requirement                                  |
| FR-API-006               | OAuth 2.0 authentication                                            | Security requirement (prevent unauthorized access) |
| FR-API-007               | Role-based access control (RBAC)                                    | HIPAA/compliance requirement                       |
| FR-API-012 to FR-API-013 | HTTP status codes (200 OK, 400 Bad Request, 401 Unauthorized, etc.) | Standard API contract; required for integration    |

**MUST HAVE Count**: 6 requirements

#### Alerts & Clinical Decision Support

| Req ID                   | Requirement                          | Justification                                 |
| ------------------------ | ------------------------------------ | --------------------------------------------- |
| FR-CDS-001 to FR-CDS-003 | High-risk, declining, adverse alerts | Core clinical decision support function       |
| FR-CDS-014               | Alert evidence summary               | Clinician must understand why alert triggered |

**MUST HAVE Count**: 2 requirements (bundled as alert framework)

#### Performance

| Req ID       | Requirement                 | Justification                                                   |
| ------------ | --------------------------- | --------------------------------------------------------------- |
| NFR-PERF-001 | Inference latency p95 <2.0s | Required for clinical workflow (cannot block clinician for 10s) |

**MUST HAVE Count**: 1 requirement

#### Security & Compliance

| Req ID                       | Requirement                                  | Justification                                       |
| ---------------------------- | -------------------------------------------- | --------------------------------------------------- |
| NFR-SEC-001 to NFR-SEC-005   | HTTPS/TLS 1.3, OAuth 2.0, patient ID hashing | HIPAA minimum requirements                          |
| NFR-SEC-008 to NFR-SEC-010   | Audit logging (immutable, 7-year retention)  | Regulatory requirement for post-market surveillance |
| NFR-COMP-001 to NFR-COMP-003 | HIPAA compliance (Security & Privacy Rules)  | Regulatory requirement (cannot deploy without)      |
| NFR-COMP-008 to NFR-COMP-010 | IEC 62304, ISO 14971, DHF documentation      | FDA/MDR requirement (non-negotiable)                |

**MUST HAVE Count**: 8 requirements (bundled as regulatory compliance)

#### Reliability

| Req ID      | Requirement                | Justification                                                        |
| ----------- | -------------------------- | -------------------------------------------------------------------- |
| NFR-REL-001 | 99.5% uptime target        | Clinical systems must be reliable (cannot tolerate frequent outages) |
| NFR-REL-005 | Error logging with context | Troubleshooting & post-incident analysis                             |

**MUST HAVE Count**: 2 requirements

#### Monitoring & Observability

| Req ID        | Requirement                                    | Justification                                            |
| ------------- | ---------------------------------------------- | -------------------------------------------------------- |
| NFR-MAINT-008 | Comprehensive monitoring (Prometheus, Grafana) | Operations requirement; cannot manage without visibility |

**MUST HAVE Count**: 1 requirement

### Total MUST HAVE: ~40 requirements

---

### SHOULD HAVE Requirements (High Value, Expected by Clinicians, but Not Critical for v1.0)

#### Data Ingestion & Preprocessing

| Req ID     | Requirement                                             | Justification                        | Rationale for Deferral                                                                           |
| ---------- | ------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------ |
| FR-DIP-003 | QuestionnaireResponse (MMSE, CDR scores)                | Supports full risk stratification    | Can use EHR-pulled MMSE in v1.0; QuestionnaireResponse mapping added in v1.1                     |
| FR-DIP-011 | Semantic density extraction (NLP-based speech analysis) | Advanced digital biomarker           | Requires NLP model integration; v1.0 uses simpler acoustic features (jitter, shimmer)            |
| FR-DIP-016 | Asynchronous processing (job queue)                     | Supports high-volume batch scenarios | v1.0 synchronous only; async added in v1.1 for hospital integrations with many daily predictions |

#### Output & Reporting

| Req ID     | Requirement                                      | Justification                              | Rationale for Deferral                                           |
| ---------- | ------------------------------------------------ | ------------------------------------------ | ---------------------------------------------------------------- |
| FR-OUT-014 | JSON output format                               | API consumers (EHR, data warehouse)        | v1.0 outputs FHIR (which is JSON); explicit JSON wrapper in v1.1 |
| FR-OUT-015 | PDF output (for patient handouts)                | Patient education & shared decision-making | v1.0 text-only; PDF generation in v1.1                           |
| FR-OUT-016 | HTML output (for web dashboard)                  | Clinician dashboard display                | v1.0 uses Navify UI directly; custom HTML dashboard in v1.1      |
| FR-OUT-020 | Discordant result flags (biomarker disagreement) | Important for quality control              | Added in v1.1; v1.0 includes results but not explicit flags      |

#### Clinical Decision Support

| Req ID     | Requirement                             | Justification                                  | Rationale for Deferral                                                             |
| ---------- | --------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| FR-CDS-004 | Discordant alerts (biomarkers disagree) | Quality control & clinician awareness          | Deferred to v1.1; v1.0 reports all predictions, clinician interprets discrepancies |
| FR-CDS-005 | Adverse alerts (NfL rising)             | Monitoring for treatment-related complications | Important but v1.0 focuses on diagnosis; monitoring in v1.1                        |
| FR-CDS-009 | Alert threshold customization           | Institutional variability                      | v1.0 uses fixed thresholds; customization in v1.1 after deployment experience      |
| FR-CDS-012 | EHR inbox integration (Epic SmartText)  | Workflow integration                           | Requires Epic-specific integration; v1.0 uses API only; SmartText in v1.1          |

#### API & Integration

| Req ID     | Requirement                         | Justification                           | Rationale for Deferral                                        |
| ---------- | ----------------------------------- | --------------------------------------- | ------------------------------------------------------------- |
| FR-API-010 | Asynchronous processing (job queue) | Supports long-running analysis          | v1.0 synchronous <2s; async in v1.1                           |
| FR-API-017 | OpenAPI 3.0 auto-generation         | Developer documentation                 | FastAPI auto-generates by default; v1.0 includes; no deferral |
| FR-API-020 | HL7 v2.x legacy support             | Backward compatibility with old systems | v1.0 FHIR-only (forward-looking); HL7 v2.x adapter in v1.1    |
| FR-API-022 | CORS for cross-origin requests      | Web dashboard support                   | v1.0 backend only; web dashboard in v1.1                      |

#### Model Inference

| Req ID     | Requirement                  | Justification                               | Rationale for Deferral                                           |
| ---------- | ---------------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| FR-MIP-016 | Attention + GNN unit tests   | Quality assurance                           | v1.0 includes basic unit tests; comprehensive test suite in v1.1 |
| FR-MIP-020 | Model rollback capability    | Safety (revert if newer version has issues) | v1.0 versioning system built in; auto-rollback in v1.1           |
| FR-MIP-026 | Identify k-nearest neighbors | Explainability (show similar patient cases) | v1.0 outputs explanation; similar patient retrieval in v1.1      |

#### Performance

| Req ID       | Requirement                                    | Justification     | Rationale for Deferral                                       |
| ------------ | ---------------------------------------------- | ----------------- | ------------------------------------------------------------ |
| NFR-PERF-003 | Throughput ≥100 predictions/hour               | Capacity planning | v1.0 supports ~50 pred/hr; scaling in v1.1 with optimization |
| NFR-PERF-004 | Batch processing latency <5s for batch_size=32 | Batch use cases   | v1.0 single-request only; batch processing in v1.1           |

#### Scalability & Reliability

| Req ID                         | Requirement                                        | Justification       | Rationale for Deferral                                                 |
| ------------------------------ | -------------------------------------------------- | ------------------- | ---------------------------------------------------------------------- |
| NFR-SCALE-001 to NFR-SCALE-002 | Horizontal scaling (Kubernetes HPA)                | High-load scenarios | v1.0 vertical scaling (single instance); horizontal in v1.1 post-pilot |
| NFR-REL-002 to NFR-REL-004     | Graceful degradation, circuit breaker, retry logic | Resilience          | v1.0 basic error handling; advanced resilience in v1.1                 |
| NFR-REL-010 to NFR-REL-012     | Backup & restore procedures                        | Disaster recovery   | v1.0 manual backups; automated in v1.1                                 |

#### Maintainability & Monitoring

| Req ID                         | Requirement                  | Justification          | Rationale for Deferral                                               |
| ------------------------------ | ---------------------------- | ---------------------- | -------------------------------------------------------------------- |
| NFR-MAINT-002 to NFR-MAINT-010 | Code quality, CI/CD, logging | Operational excellence | v1.0 includes GitHub Actions CI/CD; full observability suite in v1.1 |

#### Compliance

| Req ID       | Requirement                 | Justification                    | Rationale for Deferral                                           |
| ------------ | --------------------------- | -------------------------------- | ---------------------------------------------------------------- |
| NFR-COMP-006 | GDPR right to erasure       | EU data protection (if deployed) | v1.0 US focus (FDA); GDPR compliance in v1.1 for EU release      |
| NFR-SEC-020  | SOC 2 Type II certification | Advanced security assurance      | v1.0 passes penetration testing; SOC 2 audit in v1.1 post-launch |

#### Interoperability

| Req ID                         | Requirement                      | Justification            | Rationale for Deferral                                  |
| ------------------------------ | -------------------------------- | ------------------------ | ------------------------------------------------------- |
| NFR-INTER-009 to NFR-INTER-010 | Epic/Cerner/Meditech integration | EHR-specific deployments | v1.0 FHIR API (vendor-agnostic); Epic SmartText in v1.1 |

### Total SHOULD HAVE: ~35 requirements

---

### COULD HAVE Requirements (Nice-to-Have, Low Clinical Impact or High Implementation Cost)

#### User Experience & Dashboards

| Req ID     | Requirement             | Justification             | Rationale for Deferral                         |
| ---------- | ----------------------- | ------------------------- | ---------------------------------------------- |
| Feature 13 | Clinician web dashboard | Nice UI for result review | v1.0 uses Navify UI; custom dashboard in v2.0  |
| Feature 14 | Patient portal          | Patient engagement        | Out of scope for v1.0; home monitoring in v2.0 |
| Feature 15 | Admin panel             | Threshold configuration   | v1.0 uses fixed thresholds; admin UI in v2.0   |

#### Advanced Analytics

| Req ID          | Requirement                                         | Justification                       | Rationale for Deferral                                                                       |
| --------------- | --------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------- |
| FR-MIP-027      | Retrieve training outcomes for similar patients     | Advanced explainability             | Requires database of training patient outcomes; v1.0 shows similarity only; outcomes in v1.1 |
| FR-MIP-030      | Ensemble predictions (multiple models)              | Robustness                          | v1.0 single best model; ensemble in v2.0 if performance gains warrant                        |
| NFR-EXPLAIN-006 | Flag high-uncertainty predictions                   | Quality control                     | v1.0 outputs confidence intervals; auto-flagging in v1.1                                     |
| NFR-EXPLAIN-007 | Uncertainty decomposition (epistemic vs. aleatoric) | Advanced uncertainty quantification | v1.0 total uncertainty via Monte Carlo; decomposition in v2.0 (research-focused)             |

#### Remote Monitoring & Chronic Disease Management

| Req ID                            | Requirement                          | Justification                        | Rationale for Deferral                                                              |
| --------------------------------- | ------------------------------------ | ------------------------------------ | ----------------------------------------------------------------------------------- |
| Scenario 3 (Treatment Monitoring) | Remote monitoring via smartphone app | Supports treatment response tracking | v1.0 office-based assessment only; home monitoring in v2.0 with mHealth integration |
| FR-CDS-005                        | ARIA-related adverse event detection | Post-diagnosis monitoring            | v1.0 diagnosis-focused; post-treatment monitoring in v2.0                           |

#### Advanced Features (Post-Launch)

| Req ID                        | Requirement                                    | Justification               | Rationale for Deferral                                                                   |
| ----------------------------- | ---------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| Multiple modalities beyond 4  | Add MRI/PET imaging, tau PET                   | Extended multimodal fusion  | v1.0 uses available data; imaging integration in v3.0 (requires different preprocessing) |
| Federated learning            | Train on distributed data without centralizing | Privacy-preserving training | Research topic; v3.0+ if partnership ecosystem develops                                  |
| Digital biomarker marketplace | Integrate third-party digital biomarkers       | Plug-and-play modalities    | Requires Roche ecosystem expansion; v2.0+                                                |

### Total COULD HAVE: ~15 requirements

---

### WON'T HAVE Requirements (Explicitly Out of Scope for v1.0)

| Req ID                                           | Requirement                                                          | Reason for Exclusion                                                     | Timeline                                  |
| ------------------------------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------- |
| **Real-Time Streaming**                          | Continuous waveform processing (ECG, EEG)                            | Out of scope; requires different validation framework                    | v3.0+ (research phase)                    |
| **Multi-Center Federated Learning**              | Train across multiple hospitals without data centralization          | Privacy-enhancing but complex; v1.0 centralized                          | v4.0+                                     |
| **Imaging Integration**                          | MRI/PET imaging as direct model inputs                               | Requires separate segmentation preprocessing; v1.0 uses fluid biomarkers | v3.0+                                     |
| **Tau PET Biomarkers**                           | Integrate tau PET imaging (phosphorylated tau in brain)              | Not available as standard blood test yet; v1.0 fluid only                | v3.0+ when tau-PET standardized           |
| **Pharmaceutical Recommendation**                | Recommend specific DMT (lecanemab vs. donanemab)                     | Out of scope CDS role; clinician decision                                | v2.0 (requires new clinical evidence)     |
| **Genetic Risk Stratification**                  | Use genome-wide association studies (GWAS) for polygenic risk scores | v1.0 uses only APOE; GWAS integration in v3.0                            |
| **Lifestyle Intervention Recommendation**        | Recommend diet, exercise, cognitive training                         | Out of scope for biomarker-focused CDS                                   | v2.0 (requires lifestyle outcome studies) |
| **Causal Inference**                             | Identify causal relationships between biomarkers & cognition         | Too complex for v1.0; observational only                                 | v3.0+ (research phase)                    |
| **Automated Treatment Titration**                | Recommend dosage adjustments for DMT                                 | Too high-risk for v1.0 CDS; clinician-driven                             | v3.0+ (post-approval evolution)           |
| **Multilinguality**                              | Support non-English speech analysis                                  | v1.0 English-only; multilingual in v2.0                                  |
| **Sleep/Physical Activity Wearable Integration** | Incorporate Fitbit, Apple Watch data                                 | Out of scope v1.0; wearables in v2.0                                     |
| **Smartphone App (Patient-Facing)**              | Build native iOS/Android app for remote monitoring                   | v1.0 hospital-based only; app in v2.0                                    |
| **Voice Assistant Integration**                  | Alexa/Google Home-based screening                                    | Too novel for v1.0 clinical validation                                   | v3.0+ (requires separate clinical trial)  |

### Total WON'T HAVE: ~13 requirements

---

## Summary: MoSCoW Breakdown

| Category        | Count | Details                                                                                         |
| --------------- | ----- | ----------------------------------------------------------------------------------------------- |
| **MUST HAVE**   | ~40   | Core FDA/MDR requirements: FHIR/API, security, compliance, model, explainability                |
| **SHOULD HAVE** | ~35   | Expected features: advanced preprocessing, async, PDF/HTML output, Epic integration, monitoring |
| **COULD HAVE**  | ~15   | Nice-to-have: dashboards, remote monitoring, imaging, advanced uncertainty                      |
| **WON'T HAVE**  | ~13   | Explicitly deferred: streaming, federated learning, pharmacogenomics, smartphones               |
| **TOTAL**       | ~103  | Comprehensive scope for 16-month development                                                    |

---

## Key Dependencies & Sequencing

### Phase 1 (Months 1-4): MUST HAVE Foundation

- Implement SRS, SAD, RMF (regulatory framework)
- Implement MUST HAVE data ingestion, model architecture, core API
- Exit: Working prototype with 70%+ accuracy

### Phase 2 (Months 5-10): MUST HAVE Clinical Validation

- Full-scale training achieving AUC ≥0.85, RMSE ≤3.0
- SHAP/attention explainability validation
- Clinical case studies & subgroup analysis
- Exit: FDA/MDR-ready model + Clinical Validation Report

### Phase 3 (Months 11-16): MUST HAVE Integration & Regulatory

- FHIR API production-hardening
- Docker/Kubernetes deployment
- Security audit & penetration testing
- FDA De Novo + MDR submissions
- SHOULD HAVE features (async, monitoring) added as time permits
- Exit: FDA/MDR submissions filed, Navify integration ready

### Post-Launch (Months 17-24): SHOULD/COULD HAVE

- Deploy to 5 pilot sites
- Collect real-world data (1000 patients)
- Add SHOULD HAVE features (Epic integration, advanced monitoring)
- v1.1 release with enhanced features

---

**END OF COMPREHENSIVE REQUIREMENTS DOCUMENT**

_This document serves as the authoritative specification for NeuroFusion-AD development. All design, implementation, and testing activities must trace back to requirements defined herein._
