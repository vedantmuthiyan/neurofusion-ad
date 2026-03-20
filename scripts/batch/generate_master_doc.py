#!/usr/bin/env python3
"""
NeuroFusion-AD — Master Learning Document Generator
One comprehensive document that explains everything: what we built, how, why,
and what every technical decision means.

Written for: intelligent non-expert who needs to discuss this project
with a PhD Data Scientist / ML expert.

Usage:
  python scripts/batch/generate_master_doc.py --submit
  python scripts/batch/generate_master_doc.py --check
  python scripts/batch/generate_master_doc.py --retrieve
"""
import anthropic, argparse
from pathlib import Path
from datetime import datetime

client = anthropic.Anthropic()
BATCH_ID_FILE = Path("scripts/batch/.master_doc_batch_id")

# This is split into 4 parts so each fits within max_tokens
# Each part is a self-contained section of the master document

SHARED_CONTEXT = """
You are a senior ML engineer writing a comprehensive technical narrative for NeuroFusion-AD.
The reader is intelligent and motivated to understand, but is NOT an ML expert.
They need to discuss this confidently with a PhD ML scientist.

WRITE STYLE:
- Explain every acronym on first use: e.g., "FHIR (Fast Healthcare Interoperability Resources)"
- Use analogies for complex concepts
- Be direct about what worked, what failed, and why
- No hedging or vague language
- Assume the reader can handle math if explained clearly
- Length: each section should be thorough — 800–1500 words per major section

PROJECT FACTS (use these exact numbers everywhere):
- ADNI test AUC: 0.8897 (95% CI: 0.790–0.990)
- Bio-Hermes-001 test AUC: 0.9071 (95% CI: 0.860–0.950)
- ADNI MMSE RMSE: 1.804 pts/year
- Survival C-index: 0.651
- ECE (calibration error): 0.083
- Model parameters: 2,244,611
- embed_dim: 256 (was 768, reduced in Phase 2B)
- fluid_input_dim: 2 (pTau181 + NfL only — ABETA42_CSF removed due to data leakage)
- Temperature scaling: T = 0.756
- ADNI: N=494 MCI patients (train=345, val=74, test=75)
- Bio-Hermes-001: N=945 (train=662, val=141, test=142)
- API p95 latency: 125ms on RTX 3090
- 212 tests passing
"""

DOCUMENTS = [
    {
        "id": "master-doc-part1",
        "path": "docs/NEUROFUSION_AD_MASTER_LEARNING_DOCUMENT.md",
        "append": False,
        "prompt": f"""{SHARED_CONTEXT}

Write PART 1 of the NeuroFusion-AD Master Learning Document.
This is the opening of a single comprehensive document — write it with a professional header.

---

# NeuroFusion-AD: Complete Technical Narrative
# Everything We Built, How, and Why
**Date**: March 2026 | **Author**: NeuroFusion-AD Development Team

---

Write the following sections in full:

## 1. What This Project Is (and Why It Matters)

Cover:
- What Alzheimer's disease is and why early detection matters clinically
- What MCI (Mild Cognitive Impairment) means — the patient population
- Why existing diagnostic tools fail (6-month neurology waits, expensive PET scans, 90% undiagnosed)
- What NeuroFusion-AD does in plain English: it combines a blood test + speech recording + gait measurement + demographics to predict whether a patient's brain is accumulating amyloid protein (the hallmark of Alzheimer's)
- The three things it predicts simultaneously:
  (1) Amyloid positivity — is amyloid building up? (yes/no probability)
  (2) MMSE trajectory — how fast will cognition decline? (points per year)
  (3) Survival risk — when will this patient progress to dementia? (C-index, Kaplan-Meier)
- Why this matters for Roche: their Elecsys pTau-217 assay drives reagent revenue;
  NeuroFusion-AD is the triage tool that decides WHO gets that test

## 2. The Four Modalities — What Data We Use

For each modality, explain: what it measures, where it comes from, and why it matters clinically.

### 2a. Fluid Biomarkers
- pTau181 (phosphorylated Tau protein 181): a CSF (cerebrospinal spinal fluid) marker of neurofibrillary tangles.
  In ADNI we use CSF pTau181. In Bio-Hermes we use plasma (blood) pTau217.
  Explain the difference and why it matters (plasma is much more accessible — blood draw vs. lumbar puncture).
- NfL (Neurofilament Light Chain): a blood/CSF marker of general neuronal damage.
  Not specific to AD, but rises with any axonal injury.
- What we REMOVED and why: ABETA42_CSF (the amyloid beta CSF measurement).
  This was the most important bug in the project — explain data leakage fully.

### 2b. Acoustic (Speech) Features  
- The Cookie Theft picture description task: patient describes a picture for 60 seconds
- 15 extracted features: MFCCs (Mel-frequency cepstral coefficients — like fingerprints of how the voice sounds),
  pitch statistics, speaking rate, pause patterns
- Why speech changes in AD: reduced semantic density, increased pauses, simplified grammar
- CRITICAL LIMITATION: For ADNI, these features are SYNTHESIZED (generated from statistical distributions)
  because ADNI doesn't have speech recordings. Bio-Hermes-001 has some real data.
  Explain why we still trained on synthetic: proof-of-concept, system validation, real data in deployment

### 2c. Motor Features
- 20 features from smartphone accelerometer during walking
- Gait speed, stride variability, double support time (when both feet are on the ground simultaneously)
- Why gait changes in AD: motor cortex involvement, executive function controls walking
- Same limitation: ADNI motor features are synthetic

### 2d. Clinical Demographics
- Age (continuous, normalized)
- Sex (binary: 0=male, 1=female)
- APOE ε4 genotype (0, 1, or 2 alleles) — the strongest known genetic risk factor for late-onset AD
- MMSE score (Mini-Mental State Examination) — 30-point cognitive test (higher = better)

## 3. The Architecture — How We Process This Data

### 3a. Encoders — Turning Raw Features Into Embeddings

Explain "embedding" using an analogy: just as GPS converts an address into coordinates
that allow mathematical comparison between locations, an encoder converts raw clinical features
into a vector of numbers (256 numbers per modality) that captures the essential meaning.

Each encoder is a Multi-Layer Perceptron (MLP) — layers of matrix multiplication + non-linear activation:
- Fluid: 2 numbers → 256 numbers (3 linear layers with ReLU activations)
- Acoustic: 15 numbers → 256 numbers (4 linear layers)
- Motor: 20 numbers → 256 numbers (4 linear layers)
- Clinical: Age + MMSE (continuous) + APOE embedding + Sex embedding → 256 numbers

Why do we need encoders at all? Because raw numbers from different modalities are on completely
different scales and have different meanings. The encoder brings them into a shared 256-dimensional space.

### 3b. Cross-Modal Attention — Deciding Which Modalities Matter

Explain attention mechanism using an analogy: imagine a hospital committee meeting where 4 specialists
present their findings. The committee chair (attention mechanism) decides how much weight to give each
specialist's opinion for THIS particular patient.

The math: Multi-Head Attention with 4 heads.
- Each "head" learns a different aspect of cross-modal interaction
- Query = concatenation of all 4 embeddings
- Keys and Values = all 4 embeddings
- Output: attention weights per modality (4 numbers summing to 1.0)
- Example: "For this patient, fluid biomarkers (pTau181 very elevated) get 32% weight,
  acoustic 26%, motor 24%, clinical 18%"
- These attention weights ARE the explainability output — clinicians see them

### 3c. Graph Neural Network — Learning from Similar Patients

This is the most novel part. Explain:
- Why GNN? Each patient is not isolated — similar patients share outcomes
- Graph construction: for each batch, compute cosine similarity between patient embeddings
  Threshold at 0.55 → connect patients who are sufficiently similar
- GraphSAGE (2 layers): each patient's embedding is updated by aggregating embeddings
  from its neighbors → a patient "borrows information" from similar patients
- Why 2 layers not 3? 2-hop neighborhood is sufficient; more layers cause oversmoothing
  (all patients start to look the same)

### 3d. Multi-Task Output Heads

Three separate linear layers on top of the 256-dim GNN output:

1. Classification head: 256 → 1 number → Sigmoid → probability ∈ [0,1]
   Loss: BCEWithLogitsLoss (Binary Cross-Entropy with Logits)
   
2. Regression head: 256 → 1 number (MMSE slope, e.g., -1.8 means 1.8 points/year decline)
   Loss: MSELoss (Mean Squared Error)
   
3. Survival head: 256 → 2 numbers (risk score, predicted time)
   Loss: Cox Partial Likelihood (the standard survival analysis loss)

The multi-task loss combines all three:
   Total = 0.5×cls_loss + 0.3×reg_loss + 0.2×surv_loss

Explain why multi-task: the tasks share the same underlying biology. Training them together
forces the model to learn representations that capture all aspects of disease.
"""
    },
    {
        "id": "master-doc-part2",
        "path": "docs/NEUROFUSION_AD_MASTER_LEARNING_DOCUMENT.md",
        "append": True,
        "prompt": f"""{SHARED_CONTEXT}

Write PART 2 of the NeuroFusion-AD Master Learning Document.
Continue the document directly — no new header, no preamble.

---

## 4. The Data — What We Actually Have and Its Limitations

### 4a. ADNI (Alzheimer's Disease Neuroimaging Initiative)

What it is: A major publicly-funded longitudinal study running since 2003.
How we access it: Application + Data Use Agreement at adni.loni.usc.edu.
What we got: 494 MCI patients with longitudinal follow-up.
Key columns we use and what they mean:
- RID: patient identifier (hashed for privacy in our system)
- VISCODE: visit code ('bl' = baseline)
- DX_bl: baseline diagnosis (we filter to MCI only)
- PTAU181: CSF phosphorylated tau at amino acid 181 (pg/mL)
- ABETA: CSF amyloid beta 42 (pg/mL) — used ONLY for label creation, NOT as model input
- NFL/NfL: plasma neurofilament light chain
- MMSE: score at each visit
- APOE4: count of ε4 alleles

Missing value codes: ADNI uses -1 and -4 to mean "missing" (not actual values).
We replace both with NaN, then impute with training-set medians.

Label creation:
- AMYLOID_POSITIVE = 1 if CSF_ABETA42 < 192 pg/mL (standard clinical cutoff), else 0
- MMSE_SLOPE: linear regression of MMSE scores over time per patient → points/year
- TIME_TO_EVENT + EVENT_INDICATOR: months from baseline to first "Dementia" diagnosis

### 4b. Bio-Hermes-001

What it is: A Roche-partnered prospective study (~1,001 enrolled, ~945 usable)
Why it matters: Uses REAL plasma pTau217 (Roche Elecsys assay) — the target clinical test
24% underrepresented communities — diversity matters for fairness validation
Cross-sectional: ONE visit per patient. No longitudinal outcomes.
Therefore: Bio-Hermes used for CLASSIFICATION only. No MMSE slope, no survival.

Plasma pTau217 vs. CSF pTau181:
- pTau217: more specific to AD pathology than pTau181; accessible from blood (not spinal tap)
- pTau181: older biomarker, requires CSF (lumbar puncture), less specific
- Correlation: moderate (~0.7), but they're measuring related but distinct phosphorylation sites
- The FDA approved Lumipulse pTau217 in May 2025 for clinical use

### 4c. The Data Leakage Problem — Most Important Technical Decision in the Project

This is critical to understand for any expert discussion.

WHAT HAPPENED: AMYLOID_POSITIVE = 1 if ABETA42_CSF < 192 pg/mL.
ABETA42_CSF was also included as a feature in the fluid encoder.
The model learned to threshold one number to predict the label derived from that number.
Result: Validation AUC = 1.0 (impossible — a giveaway of leakage). Test AUC = 0.579.

WHY TEST AUC WAS 0.579 (not 1.0): The test set had slightly different CSF distributions,
and the model that "memorized" the threshold generalized poorly.

HOW WE FOUND IT: Correlation between ABETA42_CSF and AMYLOID_POSITIVE was >0.99 in training data.

FIX: Remove ABETA42_CSF from the fluid encoder inputs entirely.
Keep it ONLY for label computation (where it belongs).
New fluid encoder: [pTau181, NfL] — 2 features instead of 3.

RESULT AFTER FIX: ADNI val AUC 0.895 → test AUC 0.8897. Legitimate.

WHY THIS HAPPENS EASILY: The preprocessing pipeline created labels from raw data and then
normalized all raw data columns and put them into the feature matrix.
The fix required explicitly separating "label source columns" from "feature columns."

### 4d. Data Splits

ADNI: 70/15/15 stratified by AMYLOID_POSITIVE status
- Train: 345 patients — all model learning happens here
- Val: 74 patients — early stopping, HPO decisions
- Test: 75 patients — NEVER touched until final evaluation

Bio-Hermes: 70/15/15 stratified by AMYLOID_POSITIVE
- Train: 662 patients — used for fine-tuning
- Val: 141 patients — fine-tuning early stopping
- Test: 142 patients — held out for external validation

## 5. Training — How the Model Learns

### 5a. Optimizer and Scheduler

AdamW optimizer (Adam + weight decay):
- Weight decay: 1e-3 (100x more than the original spec because our N=345 is small)
- Learning rate: ~4e-4 (found by Optuna)
- Momentum: β₁=0.9, β₂=0.999

OneCycleLR scheduler: starts at lr/10, warms up to max_lr over first 30% of training,
then cosines down to lr/100.
Why OneCycleLR over CosineAnnealingLR? Better convergence for small datasets.
The warmup prevents unstable early gradients that are common with tiny batches.

### 5b. Gradient Accumulation — Simulating Bigger Batches

Problem: batch_size=32 with N=345 training patients = only 10-11 batches per epoch.
Very noisy gradient estimates.

Solution: gradient_accumulation_steps=4
We accumulate gradients over 4 mini-batches before updating weights.
Effective batch size = 32×4 = 128. More stable training.

### 5c. Mixed Precision Training (AMP)

PyTorch Automatic Mixed Precision (torch.cuda.amp):
- Forward pass: float16 (half precision) — 2x faster, half the memory
- Backward pass: float32 (full precision) — prevents numerical issues
- GradScaler: scales loss to prevent float16 underflow

Bug we hit: Cox loss used logcumsumexp which was not implemented in float16.
Fix: explicitly cast survival tensors to float32 before that computation.

### 5d. The Masked MultiTaskLoss — Handling Missing Labels

Critical design: Bio-Hermes has NO regression or survival labels.
ADNI has 36.2% missing classification labels (patients without CSF Abeta data).

Solution: masked loss computation.
```python
# Classification: only compute where label is not NaN
mask = ~torch.isnan(label_classification)
cls_loss = BCE(logits[mask], labels[mask])

# Regression: skip entirely for Bio-Hermes batches
if weights['reg'] > 0 and label_regression is not None:
    reg_loss = MSE(pred[mask], labels[mask])

# Survival: skip entirely for Bio-Hermes
if weights['surv'] > 0 and survival_time is not None:
    surv_loss = cox_loss(risk[mask], time[mask], events[mask])
```

For Bio-Hermes fine-tuning: loss_weights = {{cls: 1.0, reg: 0.0, surv: 0.0}}

### 5e. Training Sequence

1. ADNI baseline: 150 epochs (early stopped at ep22–43 in practice)
2. Optuna HPO: 15 trials × 40 epochs each → finds best hyperparameters
3. Retrain with best config: 150 epochs
4. Bio-Hermes fine-tuning: encoders frozen, only attention+GNN+classification_head trained, 50 epochs
   - Why freeze encoders? Assay batch effects (Fujirebio CSF vs. Roche plasma are different scales)
   - Frozen encoders = the model keeps its learned ADNI representations
   - Only the fusion + output layers adapt to Bio-Hermes distribution

### 5f. Hyperparameter Optimization — Optuna

We used Optuna with a SQLite persistent study (so it resumes after pod termination).
30 trials initially planned; reduced to 15 due to budget.
MedianPruner: kills unpromising trials early (if median val_auc at epoch 20 is below
the median of completed trials, prune it).

Best found config:
- learning_rate: 4.068e-4
- dropout: 0.308
- gradient_accumulation_steps: 2 (Note: different from our default of 4)
- cls_weight: 1.679 (unnormalized; after normalization, cls gets ~0.67 of total loss)
- graph_threshold: 0.553 (similarity threshold for GNN edge creation)
"""
    },
    {
        "id": "master-doc-part3",
        "path": "docs/NEUROFUSION_AD_MASTER_LEARNING_DOCUMENT.md",
        "append": True,
        "prompt": f"""{SHARED_CONTEXT}

Write PART 3 of the NeuroFusion-AD Master Learning Document.
Continue the document directly — no new header, no preamble.

---

## 6. Evaluation — How We Measure Performance

### 6a. Classification Metrics

AUC (Area Under the ROC Curve): the probability that, given one randomly chosen positive
patient and one randomly chosen negative patient, the model ranks the positive higher.
- AUC = 0.5: random guessing
- AUC = 1.0: perfect discrimination
- AUC = 0.89: our ADNI result — very strong for this clinical task

Sensitivity (Recall): of all patients who ARE amyloid-positive, what fraction does the model catch?
- 0.793 means we catch 79.3% of true positives
- High sensitivity = few missed cases (important for a screening tool)

Specificity: of all patients who are NOT amyloid-positive, what fraction does the model correctly identify?
- 0.933 means 93.3% of negatives correctly classified
- High specificity = few false alarms

PPV (Positive Predictive Value): if the model says "positive," how often is it right?
- 0.958 means 95.8% of our positive predictions are correct
- Very important clinically: we don't want to send patients for unnecessary expensive tests

NPV (Negative Predictive Value): if the model says "negative," how often is it right?
- 0.700 means 70% of negative predictions are correct
- This is our weakest metric — 30% of negatives might actually be positive

Optimal threshold: We use Youden's J index (sensitivity + specificity - 1, maximized).
Our optimal threshold: 0.6443. This means predictions above 0.644 are classified as "positive."
(Not 0.5 — the model is calibrated to be conservative about positives given class imbalance)

AUPRC (Area Under Precision-Recall Curve): more informative than AUC when classes are imbalanced.
Our positive rate: ~63.5% in ADNI labeled subset. So AUPRC is relevant.

### 6b. Regression Metrics

MMSE RMSE (Root Mean Square Error): 1.804 points per year.
What does 1.804 mean? On average, our MMSE slope prediction is off by 1.8 points per year.
Clinical context: a 3-point MMSE change is considered clinically significant.
So 1.8 point error is meaningful but not disqualifying for a CDS tool.

R²: negative in Phase 2 (before leakage fix), positive after. Measures variance explained.

### 6c. Survival Metrics

C-index (Concordance Index): probability that for two patients, the one who progresses first
has a higher predicted risk score.
- C-index = 0.5: random
- C-index = 1.0: perfect
- Our result: 0.651 — moderate discrimination
- Context: survival in small datasets (N=75 test) is very hard to learn

### 6d. Calibration

ECE (Expected Calibration Error): measures whether predicted probabilities match actual frequencies.
If the model says "70% probability," does the event happen ~70% of the time?
- ECE = 0: perfect calibration
- ECE = 0.083 after temperature scaling — well-calibrated

Temperature scaling (T=0.756): divide all logits by 0.756 before sigmoid.
T < 1.0 means the model was OVERCONFIDENT (extreme probabilities like 0.95 or 0.05).
Dividing by 0.756 < 1 sharpens... wait, actually:
T = 0.756 < 1 makes probabilities MORE extreme (sharpens the distribution).
This corrects a model that was being too conservative (probability bunching near 0.5).
Fitted on validation set using L-BFGS to minimize cross-entropy.

### 6e. Subgroup Analysis — The APOE4 Gap

We compute AUC separately for subgroups of age, sex, and APOE status.
Our APOE4 carrier vs. non-carrier gap: 0.131 (0.906 non-carrier, 0.775 carrier).
Our threshold was 0.12 — so this is a gate FAIL.

WHY THIS HAPPENS:
1. APOE4 carriers have more heterogeneous amyloid pathology — some have high amyloid
   without the typical pTau181 pattern. The model's fluid biomarker dominance hurts here.
2. This is a known field-wide issue: Vanderlip et al. 2025 (Alzheimer's & Dementia) shows
   the same pattern across multiple published models.
3. Our N=36 APOE4 carriers in the test set is very small — AUC estimates are noisy.

The age_lt65 subgroup (N=11) shows AUC=1.0, which is a statistical artifact of tiny N.
This is why we require N≥10 for any subgroup to be included in the gap calculation,
and why we're cautious about interpreting subgroup results with small denominators.

## 7. The FHIR API — How the System Integrates with Hospitals

### 7a. What FHIR Is

FHIR (Fast Healthcare Interoperability Resources — pronounced "fire") is the international
standard for healthcare data exchange. Version R4 is current.

FHIR defines data formats for:
- Patient: demographics (age, sex, birth date)
- Observation: lab results (coded by LOINC — Logical Observation Identifiers Names and Codes)
- RiskAssessment: model outputs (our response format)
- Parameters: wrapping input resources for custom operations

LOINC codes we use:
- 82154-1: Tau protein 181 in CSF (pTau181)
- 81600-4: Neurofilament light chain in blood
- 72107-6: MMSE score
- 30155-3: APOE genotype

Why FHIR matters for Roche: Navify Algorithm Suite (Roche's clinical AI platform) accepts
FHIR-native algorithms. Without FHIR compliance, there is NO deployment path.
Epic and Cerner (the two largest US EHR vendors) both expose FHIR R4 APIs.

### 7b. API Endpoint

`POST /fhir/RiskAssessment/$process`

The `$process` suffix is FHIR convention for custom operations.

Request: FHIR Parameters bundle containing Patient + Observations + QuestionnaireResponse
Response: FHIR RiskAssessment with prediction + confidence intervals + modality weights + recommendation

### 7c. What Happens Inside the API Call

1. FHIR Validator: parses the bundle, extracts clinical values, hashes the patient ID (SHA-256)
2. Inference Preprocessor: normalizes inputs using the ADNI-fitted StandardScaler,
   imputes missing values with training medians, builds tensor dict
3. Model forward pass (PyTorch, eval mode, no dropout)
4. Temperature scaling: logit / 0.756 → sigmoid → calibrated probability
5. Monte Carlo CI: re-run with dropout enabled 30 times → 2.5th and 97.5th percentiles
6. FHIR RiskAssessment builder: wraps outputs in FHIR-compliant JSON
7. Audit log: writes to PostgreSQL (patient hash, probability, model version, latency)

Latency: 125ms p95 on RTX 3090. Fast because the model is 2.2M parameters (small).

### 7d. The Demo — Is It the Real Model?

YES and NO — nuanced answer for your expert discussion:

The demo application has two modes:
- `demo/backend/demo_api.py` runs a lightweight FastAPI that returns PRE-COMPUTED results
  from a Python dictionary. It does NOT call the actual model at inference time.
- WHY: The main model requires a GPU (RunPod RTX 3090) and PyTorch.
  A demo app that requires a GPU is not showable in a boardroom or on a laptop.

The pre-computed results ARE from the real model:
- We ran the three demo patients through the actual trained model
- We captured the exact probabilities, modality importance weights, SHAP values
- These real numbers are stored in DEMO_RESULTS in demo_api.py
- So what you see in the demo IS the model's actual output for these patients

For production: the full API (src/api/main.py) loads the real model and does live inference.
It requires the Docker container with the model checkpoint mounted.

The distinction: demo = real results, pre-cached | production API = real results, live inference.

## 8. Performance in Context — How We Compare to the Literature

### 8a. The FDA-Approved Comparator

Lumipulse G (Fujirebio, FDA-approved May 2025): measures pTau217 in CSF.
Clinical validation AUC: 0.896 for amyloid PET positivity.
N: 499 subjects.

Our Bio-Hermes AUC: 0.907 (also using plasma pTau217 via Roche Elecsys).
We are slightly better, with N=142 (smaller — our CI is wider: 0.860–0.950).

IMPORTANT NUANCE: Lumipulse uses CSF pTau217 (lumbar puncture required).
NeuroFusion-AD uses PLASMA pTau217 (blood draw). Plasma is far more accessible.
The Bio-Hermes assay Roche uses (Elecsys) is the plasma version.

### 8b. Academic Benchmarks

From our literature review (Ali et al. 2025, Zhang et al. 2023, Mashhadi & Marinescu 2025):
- Pure neuroimaging GNNs (MRI+PET): AUC 0.94–0.97
  BUT these require PET scanners ($2,000/scan, limited availability)
- Digital biomarker-only models (speech+gait): average AUC ~0.821
  We beat this by adding fluid biomarkers
- Multimodal tabular (Tekkesinoglu & Pudas 2024, N=2,212 ADNI): strong but no digital biomarkers

### 8c. What NeuroFusion-AD Uniquely Provides

No single published system combines ALL of:
1. Plasma biomarkers (blood-accessible, no lumbar puncture)
2. Digital speech features (automated speech analysis)
3. Digital motor features (smartphone gait assessment)
4. GNN patient similarity graph (personalized prognostication)
5. Multi-task: amyloid + MMSE regression + survival
6. FHIR R4 native API (EHR integrable)
7. Validated on Roche-partnered prospective cohort (Bio-Hermes-001)

This is the "gap" in the literature that NeuroFusion-AD fills.
"""
    },
    {
        "id": "master-doc-part4",
        "path": "docs/NEUROFUSION_AD_MASTER_LEARNING_DOCUMENT.md",
        "append": True,
        "prompt": f"""{SHARED_CONTEXT}

Write PART 4 (FINAL) of the NeuroFusion-AD Master Learning Document.
Continue the document directly — no new header, no preamble.

---

## 9. Infrastructure — Where and How This Runs

### 9a. RunPod vs. AWS/Azure

We used RunPod (runpod.io) instead of AWS because of cost.
RTX 3090 on RunPod: ~$0.44/hour.
Equivalent on AWS (p3.xlarge, V100): ~$3.06/hour.
For 30+ hours of training: $13 vs. $92.

RunPod uses a "Network Volume" for persistence:
- The GPU pod itself is ephemeral (terminate anytime, billing stops)
- A network volume (/workspace/) persists even after pod termination
- All model checkpoints, processed data, and logs live on the network volume
- New pod: attach same volume, everything continues

SSH endpoint changes when pod restarts — this is expected behavior.
We've had: 213.192.2.120:40012 → 213.192.2.67:40046 after one restart.

### 9b. Docker and why it matters

Docker packages everything (code, dependencies, Python version, OS libraries) into a container.
Without Docker: "works on my machine" problem — different library versions cause failures.
With Docker: `docker run` → identical environment everywhere.

Our Dockerfile:
- Base: python:3.10-slim (minimal OS)
- Non-root user (security requirement)
- Multi-stage: dependencies installed first (cached layer), code copied second
- Health check endpoint: /health (Kubernetes uses this to know if the pod is ready)
- Model checkpoint mounted as volume (not baked into image — image would be huge)

docker-compose.yml runs 3 services together:
- neurofusion-api: the FastAPI app
- postgres: PostgreSQL database for audit logs
- redis: caching layer for repeated predictions

### 9c. The PostgreSQL Audit Trail

Every prediction is logged immutably:
- patient_hash (16-char SHA-256 prefix — not the real patient ID)
- model_version (which checkpoint was used)
- amyloid_prob, risk_category
- modality weights (fluid/acoustic/motor/clinical)
- latency_ms
- fhir_request_hash

The table has REVOKE UPDATE, DELETE — nobody can modify or delete audit records.
This is a regulatory requirement (IEC 62304, FDA AI/ML guidance): complete audit trail.

### 9d. API Authentication (in production)

OAuth 2.0 Client Credentials flow:
- Hospital IT registers NeuroFusion-AD as a client application
- Client ID + secret → request access token
- Access token (JWT) → included in Authorization: Bearer header on every API call
- Token expires (typically 1 hour) → hospital app gets a new one

This flow is hospital-side: from the clinician's perspective, it's invisible.

## 10. Regulatory — Why This Is Built the Way It Is

### 10a. SaMD (Software as a Medical Device)

NeuroFusion-AD is classified as SaMD — any software that has a medical purpose
without being part of a hardware medical device.
Being SaMD triggers regulatory requirements.

### 10b. IEC 62304 — Medical Device Software Lifecycle

This standard defines how you must build and document medical software:
- Software Development Plan (we have SDP_v1.0.md)
- Software Requirements Specification (SRS_v1.0.md — 25 requirements)
- Software Architecture Document (SAD_v1.0.md)
- Risk Management File (RMF_v1.0.md — per ISO 14971)
- All changes are documented (DHF — Design History File)
- Traceability matrix: every requirement traces to design → code → test

We classify as Class B (moderate risk) because a wrong prediction could:
- Cause unnecessary treatment (false positive)
- Miss a patient who needs treatment (false negative)
But the output is "Clinical Decision Support" — a physician reviews it before acting.
This keeps it Class B rather than Class C.

### 10c. FDA De Novo Pathway

De Novo is for novel devices with no predicate (no existing substantially equivalent device).
NeuroFusion-AD has no predicate — no prior FDA-cleared multimodal AD progression prediction algorithm.

Unlike 510(k) (which requires a predicate), De Novo creates a new classification.
If approved, it becomes the predicate for all future similar devices.

Expected timeline: 6–12 months after submission (FDA AI/ML De Novo average).
Required in submission: SRS, SAD, CVR, traceability matrix, cybersecurity documentation.

### 10d. EU MDR Class IIa

EU Medical Device Regulation requires:
- CE marking (Conformité Européenne)
- Notified Body review (we'd use TÜV SÜD)
- Technical File with Clinical Evaluation Report
- Post-Market Surveillance Plan

## 11. Known Limitations and Honest Assessment

### 11a. What Works Well
- Bio-Hermes-001 AUC 0.907: strong external validation using the actual target assay (plasma pTau217)
- Calibration ECE 0.083: well-calibrated probabilities (clinicians can trust the numbers)
- API latency 125ms: fast enough for real-time clinical use
- Multi-task training converged: all three heads train simultaneously without degradation
- The leakage detection and fix is a demonstration of rigorous ML practice

### 11b. Real Limitations (be honest with the PhD expert)

1. Small training N: N=345 ADNI training patients for a multimodal GNN.
   Expert ML scientists will immediately note this. Our response:
   - We apply heavy regularization (weight_decay=1e-3, dropout=0.4, gradient clipping 0.5)
   - The 2.24M parameter model is appropriately sized for this dataset
   - External validation (Bio-Hermes, N=142 held-out) confirms generalization

2. Synthetic acoustic/motor features for ADNI:
   - These 35 features are generated from statistical distributions, not real speech/gait
   - They add limited predictive signal; the model relies more on pTau181 + MMSE
   - Expert challenge: "How do you know digital features add value if they're synthetic?"
   - Our answer: They demonstrate the pipeline's capability. Real speech data from
     DementiaBank was not accessible. Bio-Hermes has some real data. Future clinical
     deployment generates real data from actual patients.

3. APOE4 subgroup gap 0.131:
   - Expert will ask about this. Answer: consistent with Vanderlip et al. 2025,
     which shows pTau217-based models perform worse in APOE4 carriers.
     APOE4 carriers have more heterogeneous amyloid pathology.
     Post-market monitoring plan addresses this.

4. Bio-Hermes-001 is cross-sectional:
   - No MMSE slope or survival labels for the 945 Bio-Hermes patients
   - These outputs are ADNI-only
   - Bio-Hermes fine-tuning only trains the classification head

5. pTau181 CSF (ADNI) ≠ pTau217 plasma (Bio-Hermes):
   - Different amino acid positions on the tau protein
   - Different measurement methods (CSF vs. blood)
   - The model learns from pTau181 in ADNI and generalizes to pTau217 in Bio-Hermes
   - This actually demonstrates robustness (different assay, still AUC 0.907)

## 12. Key Numbers to Know by Heart

| Fact | Number |
|------|--------|
| ADNI test AUC | 0.8897 |
| Bio-Hermes-001 test AUC | 0.9071 |
| ADNI sensitivity | 79.3% |
| ADNI specificity | 93.3% |
| MMSE RMSE | 1.804 pts/yr |
| C-index (survival) | 0.651 |
| ECE (calibration) | 0.083 |
| Model parameters | 2,244,611 |
| Training patients (ADNI) | 345 |
| External validation N | 142 |
| API latency (p95) | 125ms |
| Temperature scaling | 0.756 |
| Optimal threshold | 0.6443 |
| Lumipulse comparator AUC | 0.896 |
| Our Bio-Hermes AUC | 0.907 |

## 13. Glossary — Every Acronym Defined

| Acronym | Expansion | What it means |
|---------|-----------|--------------|
| AD | Alzheimer's Disease | The neurodegenerative disease we target |
| MCI | Mild Cognitive Impairment | Early-stage cognitive decline, before dementia |
| GNN | Graph Neural Network | Neural network that operates on graphs |
| GraphSAGE | Graph Sample and Aggregate | A specific GNN algorithm |
| AUC | Area Under the (ROC) Curve | Primary classification performance metric |
| ROC | Receiver Operating Characteristic | True positive rate vs. false positive rate curve |
| MMSE | Mini-Mental State Examination | 30-point cognitive test |
| pTau181 | Phosphorylated Tau at amino acid 181 | CSF biomarker of neurofibrillary tangles |
| pTau217 | Phosphorylated Tau at amino acid 217 | Plasma biomarker, more AD-specific |
| NfL | Neurofilament Light Chain | Blood/CSF marker of neuronal damage |
| APOE | Apolipoprotein E | Gene; ε4 variant is AD risk factor |
| CSF | Cerebrospinal Fluid | Fluid from spinal tap; has biomarkers |
| ADNI | Alzheimer's Disease Neuroimaging Initiative | Longitudinal study, our main dataset |
| FHIR | Fast Healthcare Interoperability Resources | Healthcare data exchange standard |
| LOINC | Logical Observation Identifiers Names and Codes | Standard lab test coding system |
| SaMD | Software as a Medical Device | Regulatory classification |
| IEC 62304 | International standard for medical device software lifecycle | How we document development |
| ISO 14971 | International standard for medical device risk management | How we manage risks |
| DHF | Design History File | Complete regulatory documentation package |
| FDA | Food and Drug Administration | US regulatory body |
| MDR | Medical Device Regulation | EU regulatory framework |
| EHR | Electronic Health Record | Hospital's patient record system |
| API | Application Programming Interface | How systems communicate |
| MLP | Multi-Layer Perceptron | Standard feedforward neural network |
| AMP | Automatic Mixed Precision | Training technique using float16 + float32 |
| HPO | Hyperparameter Optimization | Finding the best model settings |
| ECE | Expected Calibration Error | Measures probability accuracy |
| PPV | Positive Predictive Value | Precision of positive predictions |
| NPV | Negative Predictive Value | Precision of negative predictions |
| CDS | Clinical Decision Support | Tool that aids, not replaces, clinician judgment |
| DMT | Disease-Modifying Therapy | Lecanemab, donanemab — new AD treatments |
| ARIA | Amyloid-Related Imaging Abnormalities | Side effect of anti-amyloid therapies |
| PET | Positron Emission Tomography | Brain imaging scan (expensive, requires scanner) |
| MFCC | Mel-Frequency Cepstral Coefficient | Acoustic feature from speech signal |
| BCE | Binary Cross-Entropy | Loss function for binary classification |
| MSE | Mean Squared Error | Loss function for regression |
| CI | Confidence Interval | Range within which the true value likely falls |
| RMSE | Root Mean Square Error | Prediction error in original units |
| W&B | Weights & Biases | Experiment tracking platform |
| JWT | JSON Web Token | Authentication token format |
| TLS | Transport Layer Security | HTTPS encryption protocol |
| HIPAA | Health Insurance Portability and Accountability Act | US health data privacy law |
| GDPR | General Data Protection Regulation | EU data privacy law |
| PHI | Protected Health Information | Patient data that must be protected |

---

*End of NeuroFusion-AD Master Learning Document*
*Document generated March 2026. For internal use and expert review.*
"""
    }
]


def submit_batch():
    requests_list = []
    for doc in DOCUMENTS:
        requests_list.append({
            "custom_id": doc["id"],
            "params": {
                "model": "claude-sonnet-4-6",
                "max_tokens": 8000,
                "system": [{"type": "text",
                            "text": "You are writing a technical learning document. Write in full, complete sections with no truncation. Every section must be finished completely.",
                            "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": doc["prompt"]}]
            }
        })

    print(f"Submitting {len(requests_list)} document parts")
    print("Estimated cost: ~$2–3 (batch discount)")
    batch = client.messages.batches.create(requests=requests_list)
    BATCH_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    BATCH_ID_FILE.write_text(batch.id)
    print(f"[OK] Batch ID: {batch.id}")
    return batch.id


def check_status():
    batch_id = BATCH_ID_FILE.read_text().strip()
    batch = client.messages.batches.retrieve(batch_id)
    c = batch.request_counts
    print(f"Status: {batch.processing_status}")
    print(f"  Succeeded: {c.succeeded} | Processing: {c.processing} | Errored: {c.errored}")
    if batch.processing_status == "ended":
        print("  → Run: --retrieve")


def retrieve_and_assemble():
    """Retrieve all parts and assemble into one complete document."""
    batch_id = BATCH_ID_FILE.read_text().strip()
    output_path = Path("docs/NEUROFUSION_AD_MASTER_LEARNING_DOCUMENT.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect results indexed by custom_id
    results = {}
    errors = []
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            results[result.custom_id] = result.result.message.content[0].text
        else:
            errors.append(f"{result.custom_id}: {result.result.error}")

    if errors:
        print(f"[WARN] Errors: {errors}")

    # Assemble in order
    order = ["master-doc-part1", "master-doc-part2", "master-doc-part3", "master-doc-part4"]
    parts_found = [k for k in order if k in results]

    if not parts_found:
        print("[ERROR] No results retrieved")
        return

    # Write assembled document
    assembled = ""
    for part_id in order:
        if part_id in results:
            assembled += results[part_id]
            assembled += "\n\n"  # section separator

    output_path.write_text(assembled, encoding="utf-8")
    print(f"[OK] Assembled document: {output_path}")
    print(f"   Total length: {len(assembled):,} characters")
    print(f"   Parts assembled: {len(parts_found)}/4")

    if len(parts_found) < 4:
        missing = [k for k in order if k not in results]
        print(f"   [WARN] Missing parts: {missing}")
        print("   Re-run --submit to retry missing parts")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Master Learning Document Generator")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--retrieve", action="store_true")
    args = p.parse_args()
    if args.submit:    submit_batch()
    elif args.check:   check_status()
    elif args.retrieve: retrieve_and_assemble()
    else: p.print_help()
