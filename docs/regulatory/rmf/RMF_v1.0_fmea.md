# NeuroFusion-AD Failure Mode and Effects Analysis (FMEA)

## Risk Management Document: RMF-001-FMEA-v1.0

---

### Document Control

| Field | Details |
|---|---|
| **Document ID** | RMF-001-FMEA-v1.0 |
| **Product** | NeuroFusion-AD Multimodal GNN for AD Progression Prediction |
| **Regulatory Pathway** | FDA De Novo / EU MDR Class IIa |
| **SaMD Classification** | IEC 62304 Class B / ISO 14971 |
| **Prepared By** | Regulatory Affairs Officer |
| **Review Standard** | IEC 62304, ISO 14971:2019, FDA AI/ML SaMD Guidance |
| **FMEA Scope** | System-level Software FMEA (SFMEA) |
| **Risk Acceptability Threshold** | RPN > 100 = HIGH PRIORITY (immediate action required) |

---

### Severity Scale Reference

| Rating | Level | Clinical Impact |
|---|---|---|
| 1–2 | Negligible | No clinical impact; cosmetic issue only |
| 3–4 | Minor | Minimal patient impact; easily corrected |
| 5–6 | Moderate | Delayed or suboptimal clinical decision |
| 7–8 | Serious | Significant patient harm potential; misdiagnosis risk |
| 9–10 | Critical | Severe patient harm; incorrect treatment; life-altering misdiagnosis |

### Occurrence Scale Reference

| Rating | Level | Frequency Estimate |
|---|---|---|
| 1–2 | Remote | < 1 in 10,000 inference calls |
| 3–4 | Low | 1 in 1,000 – 1 in 10,000 |
| 5–6 | Moderate | 1 in 100 – 1 in 1,000 |
| 7–8 | High | 1 in 10 – 1 in 100 |
| 9–10 | Very High | > 1 in 10 |

### Detection Scale Reference

| Rating | Level | Detection Capability |
|---|---|---|
| 1–2 | Almost Certain | Automated detection with validated alerts |
| 3–4 | High | Detected by monitoring; low escape probability |
| 5–6 | Moderate | Detected by periodic review; some escape risk |
| 7–8 | Low | Difficult to detect without specific testing |
| 9–10 | Very Low | Undetected until patient-level harm occurs |

---

## FMEA Table

> **Legend:** 🔴 HIGH PRIORITY (RPN > 100) | ✅ Acceptable Risk | S = Severity | O = Occurrence | D = Detection

---

### Component 1: Feature Encoders

*(FluidBiomarkerEncoder · AcousticEncoder · MotorEncoder)*

| Component | Failure Mode | Effect on System / Patient | S | O | D | RPN | Priority | Corrective Actions | Responsible | Target Date | New S | New O | New D | New RPN |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|:---:|:---:|:---:|:---:|
| **FluidBiomarkerEncoder** | **FM-101:** Silent numerical underflow in pTau-217 or Abeta42/40 encoding — floating-point values collapse to zero or subnormal without raising exception | Biomarker features zeroed out; model receives corrupted latent representation; AUC degrades silently; false-low progression risk output → MCI patient classified as stable and denied intervention | **9** | **4** | **7** | **🔴 252** | **HIGH** | 1. Implement post-encoding L2-norm assertion gate: flag any encoder output with ‖z‖₂ < ε (ε=1e-6). 2. Add PyTorch `torch.isfinite()` tensor audit after each forward pass. 3. Enable mixed-precision anomaly detection (`torch.autograd.set_detect_anomaly(True)`) in staging. 4. Unit test with boundary-value biomarker inputs including subnormal floats. 5. CI/CD gate: fail build if numerical health checks do not pass on synthetic edge-case dataset. | ML Engineering | Sprint+2 | 9 | 4 | **2** | **72** ✅ |
| **FluidBiomarkerEncoder** | **FM-102:** Model weight file corruption or partial load — encoder checkpoint loads with mismatched tensor shapes or NaN weights after storage fault or failed deployment | Encoder produces NaN embeddings propagated through entire graph; all downstream outputs become NaN or infinity; inference returns null result or crashes | **8** | **3** | **4** | **🔴 96** | **HIGH** | 1. Compute and verify SHA-256 checksum of all `.pt` weight files at container startup before serving. 2. Implement model integrity self-test: run 5 synthetic reference inputs at boot and assert outputs match expected reference values within ±0.01. 3. Store redundant weight copies in immutable S3 bucket with versioning. 4. Kubernetes readiness probe must pass model self-test before pod accepts traffic. | MLOps / DevOps | Sprint+1 | 8 | 3 | **2** | **48** ✅ |
| **AcousticEncoder** | **FM-103:** Acoustic feature extraction failure on atypical audio input — sample rate mismatch, mono/stereo confusion, or codec artifact produces spectrogram with extreme energy distribution outside training domain | Acoustic latent vector falls outside learned manifold; cross-modal attention receives out-of-distribution embedding; model extrapolates unpredictably; speech-based progression signal lost | **7** | **5** | **6** | **🔴 210** | **HIGH** | 1. Add pre-processing validation: assert sample rate == 16 kHz, duration in [3s, 120s], RMS energy in [−60dB, −10dB]. 2. Implement audio-specific OOD detector (Mahalanobis distance on MFCC features vs. training distribution; flag if distance > 3σ). 3. If OOD detected, gracefully degrade: route inference through remaining 3 modalities only and append `dataAbsent` flag to FHIR output. 4. Log all OOD audio events with full metadata for post-market surveillance. | ML Engineering | Sprint+3 | 7 | 5 | **3** | **105** → further review |
| **AcousticEncoder** | **FM-104:** Personally identifiable voice data retained in inference cache — audio tensors inadvertently persisted in Redis or GPU memory between patient sessions | HIPAA / GDPR breach; voice biometric data of one patient accessible in subsequent inference session; regulatory non-compliance and patient privacy violation | **9** | **3** | **5** | **🔴 135** | **HIGH** | 1. Enforce stateless inference: explicitly zero-fill and `del` all audio tensors after encoding step within same request context. 2. Disable CUDA memory caching for audio pipeline (`torch.cuda.empty_cache()` post-inference). 3. Implement session isolation via unique per-request UUID namespacing. 4. Quarterly DAST/memory-scrape penetration test to confirm no cross-session data leakage. 5. DPO sign-off required before production deployment. | Security / DPO | Sprint+2 | 9 | **1** | **2** | **18** ✅ |
| **MotorEncoder** | **FM-105:** Sensor dropout or partial gait data — wearable sensor transmits incomplete time-series (< 30% of expected timesteps) due to connectivity loss or device fault | Motor feature tensor padded with zeros or mean-imputed without flagging; model interprets imputed values as real signal; progression risk score biased toward normal | **8** | **6** | **5** | **🔴 240** | **HIGH** | 1. Calculate completeness ratio for each motor time-series; if < 70% present, reject modality and set motor confidence weight to zero. 2. Propagate `DataAbsentReason` code to FHIR output with reason code `not-performed`. 3. Clinician UI must display explicit warning: "Motor data insufficient — score based on N modalities." 4. Retrain imputation robustness: augment training set with synthetically masked motor sequences at 10–50% dropout rate. | ML Engineering / Clinical | Sprint+2 | 8 | 6 | **2** | **96** ✅ |
| **MotorEncoder** | **FM-106:** Temporal misalignment between motor sensor timestamps and clinical visit date — gait data from prior visit (> 90 days) silently paired with current biomarker values | Motor features reflect outdated patient state; multimodal fusion combines temporally inconsistent data; progression estimate anchored to stale motor status | **7** | **4** | **6** | **🔴 168** | **HIGH** | 1. Enforce temporal coherence check: reject motor data if acquisition timestamp differs from biomarker collection date by > 90 days. 2. Include data acquisition timestamp in FHIR input bundle and validate in InputValidator. 3. Add regression test simulating date-mismatched inputs; assert rejection with error code `TEMPORAL_COHERENCE_FAIL`. | Clinical Informatics | Sprint+3 | 7 | 4 | **2** | **56** ✅ |

---

### Component 2: Input Validator

*(Range Validation · FHIR Parsing)*

| Component | Failure Mode | Effect on System / Patient | S | O | D | RPN | Priority | Corrective Actions | Responsible | Target Date | New S | New O | New D | New RPN |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|:---:|:---:|:---:|:---:|
| **Input Validator — Range Check** | **FM-201:** Out-of-range biomarker value accepted due to floating-point comparison boundary error — e.g., pTau-217 = 100.0000001 passes `value <= 100` check due to IEEE 754 rounding | Input violates validated range specification; model receives extrapolated input outside training envelope; prediction reliability undefined; no clinician warning issued | **8** | **3** | **5** | **🔴 120** | **HIGH** | 1. Replace all range comparisons with `Decimal`-precision validators using explicit inclusive bounds with 1e-9 tolerance. 2. Implement `pydantic` v2 validators with `ge`/`le` constraints on all biomarker fields. 3. Add fuzz-testing suite with boundary ±epsilon values for all 4 biomarker inputs. 4. Boundary test vectors must be included in CI regression suite with 100% pass requirement. | Software Engineering | Sprint+1 | 8 | 3 | **2** | **48** ✅ |
| **Input Validator — Range Check** | **FM-202:** Missing required biomarker silently treated as zero — null or absent pTau-217 / MMSE field not caught by validator; defaults to 0.0 instead of raising validation error | Model receives physiologically impossible zero value (pTau-217 minimum valid = 0.1 pg/mL); classification output severely distorted; patient misclassified | **9** | **4** | **4** | **🔴 144** | **HIGH** | 1. All required biomarker fields declared as non-optional in Pydantic schema with `...` (no default). 2. Validator must return HTTP 422 with structured error body listing all missing fields before reaching encoder. 3. Integration test: POST with each required field omitted individually; assert 422 response and no model invocation. 4. Add FastAPI middleware to log all 422 validation errors to audit trail. | Software Engineering | Sprint+1 | 9 | 4 | **1** | **36** ✅ |
| **Input Validator — FHIR Parsing** | **FM-203:** Malformed FHIR R4 bundle accepted due to lenient parser — non-standard LOINC codes or incorrect `Observation.value[x]` types parsed without error; incorrect biomarker value extracted from wrong field | Biomarker values populated from wrong FHIR field (e.g., unit string parsed as numeric value); catastrophic input distortion with no warning; all downstream outputs invalid | **9** | **5** | **5** | **🔴 225** | **HIGH** | 1. Integrate HAPI FHIR validator library with strict profile enforcement against published NeuroFusion-AD FHIR IG. 2. Validate LOINC codes against allowlist: {pTau-217: 99110-7, Abeta42/40: 99998-5, NfL: 99997-7, MMSE: 72106-8}. 3. Require explicit `Observation.valueQuantity.unit` match; reject if unit not in expected set (pg/mL, ratio, score). 4. Return HTTP 400 with OperationOutcome listing specific FHIR path errors. | Clinical Informatics / Software | Sprint+2 | 9 | 5 | **1** | **45** ✅ |
| **Input Validator — FHIR Parsing** | **FM-204:** FHIR injection attack — adversarially crafted FHIR bundle with SQL injection payload in `Observation.note.text` or XXE in XML-encoded bundle body | Database corruption or unauthorized data access; audit log poisoning; potential exfiltration of prior patient records | **9** | **3** | **4** | **🔴 108** | **HIGH** | 1. All FHIR text fields sanitized with allowlist-based regex before persistence (strip all non-alphanumeric except medical notation characters). 2. XML parser configured with `FEATURE_EXTERNAL_GENERAL_ENTITIES = false` (XXE prevention per OWASP). 3. Use parameterized queries exclusively (SQLAlchemy ORM — no raw SQL string interpolation). 4. Annual penetration test including FHIR-specific injection payloads. 5. WAF rule deployed at Nginx layer for common injection signatures. | Security Engineering | Sprint+1 | 9 | **1** | **2** | **18** ✅ |

---

### Component 3: GNN Layer

*(GraphSAGE Convolution — 3 Layers)*

| Component | Failure Mode | Effect on System / Patient | S | O | D | RPN | Priority | Corrective Actions | Responsible | Target Date | New S | New O | New D | New RPN |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|:---:|:---:|:---:|:---:|
| **GNN Layer — GraphSAGE** | **FM-301:** Graph edge construction failure — patient-population similarity graph built with incorrect neighbor indices due to race condition in concurrent inference; node receives wrong neighbor embeddings | Patient's latent representation contaminated with features from a different patient's graph neighborhood; cross-patient data leakage; incorrect individualized prediction | **10** | **3** | **6** | **🔴 180** | **HIGH** | 1. Implement per-request isolated graph construction: each inference call creates a private `Data` object with no shared state. 2. Graph construction function declared thread-safe via request-scoped dependency injection (FastAPI `Depends`). 3. Load test with 100 concurrent requests; assert output determinism per request ID. 4. Add graph node count assertion: node count must equal 1 (single-patient inference) + K neighbors from frozen reference cohort. | ML Engineering | Sprint+2 | 10 | **1** | **2** | **20** ✅ |
| **GNN Layer — GraphSAGE** | **FM-302:** Over-smoothing in 3-layer GraphSAGE — repeated neighborhood aggregation causes patient node embedding to converge toward population mean; individual progression signal lost | Model outputs same near-mean risk score for all patients regardless of true individual risk; high-risk patients under-detected; AUC degrades below 0.85 requirement | **8** | **4** | **7** | **🔴 224** | **HIGH** | 1. Monitor per-layer embedding variance in staging: add variance tracking hook; alert if layer-3 variance < 10% of layer-1 variance. 2. Add skip connections (residual GraphSAGE) to preserve individual signal. 3. Evaluate DropEdge regularization during training to reduce over-smoothing. 4. Include Dirichlet energy metric in quarterly model performance report. 5. Performance threshold gate: AUC < 0.85 on validation set triggers model rollback. | ML Engineering / Clinical Validation | Sprint+3 | 8 | 4 | **3** | **96** ✅ |
| **GNN Layer — GraphSAGE** | **FM-303:** GPU out-of-memory error during sparse graph tensor allocation for large neighborhood samples — Kubernetes pod OOM-killed mid-inference | Inference request fails with 500 error; no prediction returned; clinician receives no CDS guidance; if unhandled, pod crash loop affects availability SLA (99.5%) | **6** | **4** | **4** | **🔴 96** | **HIGH** | 1. Cap GraphSAGE neighborhood sample size: `num_neighbors=[10, 5, 3]` per layer (enforced in `NeighborLoader` config). 2. Set explicit GPU memory reservation per pod in Kubernetes resource limits (`nvidia.com/gpu: 1`, memory limit 8Gi). 3. Implement circuit breaker: if OOM detected, route request to CPU fallback with degraded latency SLA and alert on-call. 4. Monitor GPU utilization via DCGM; auto-scale pods before memory ceiling reached. | MLOps / DevOps | Sprint+2 | 6 | 4 | **2** | **48** ✅ |
| **GNN Layer — GraphSAGE** | **FM-304:** Reference cohort graph poisoning — adversary or data entry error introduces extreme outlier node into reference population graph, shifting neighborhood aggregation for all subsequent patients | All patient predictions systematically biased toward outlier's clinical profile; population-wide prediction drift undetected until post-market surveillance | **8** | **2** | **7** | **🔴 112** | **HIGH** | 1. Freeze reference cohort graph at model release; version-control graph snapshot with cryptographic hash. 2. Implement cohort data quality gate: outlier detection (Isolation Forest) on all new nodes proposed for reference graph; require clinical data manager approval for additions. 3. Monitor prediction distribution monthly via KS-test; alert if distribution shifts > 2σ from baseline. | ML Engineering / Clinical | Sprint+3 | 8 | 2 | **2** | **32** ✅ |

---

### Component 4: Attention Module

*(Cross-Modal Attention — 768-dim, 8 heads)*

| Component | Failure Mode | Effect on System / Patient | S | O | D | RPN | Priority | Corrective Actions | Responsible | Target Date | New S | New O | New D | New RPN |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|:---:|:---:|:---:|:---:|
| **CrossModalAttention** | **FM-401:** Attention weight collapse — all 8 attention heads degenerate to attend exclusively to a single modality (e.g., fluid biomarkers only); remaining modalities ignored despite valid data | Loss of multimodal fusion benefit; model effectively becomes univariate; acoustic and motor progression signals discarded; overall predictive performance degraded; AUC/C-index may fall below spec | **8** | **4** | **7** | **🔴 224** | **HIGH** | 1. Add attention diversity monitor: compute per-head entropy H(α) after each inference; log and alert if any head entropy < 0.5 bits. 2. Log modal attention weight distribution in FHIR output extensions for clinical transparency. 3. Training regularization: add attention entropy loss term to penalize head collapse during fine-tuning. 4. Quarterly head diversity audit on production inference logs; trigger retraining if collapse rate > 5%. | ML Engineering | Sprint+3 | 8 | 4 | **2** | **64** ✅ |
| **CrossModalAttention** | **FM-402:** NaN propagation from softmax overflow — extreme pre-softmax logit values (e.g., from OOD inputs) cause `exp()` overflow → NaN attention weights → NaN context vector propagated to GNN | Entire inference output becomes NaN; clinical decision support returns null or crashes; no prediction delivered to clinician | **9** | **3** | **5** | **🔴 135** | **HIGH** | 1. Apply scaled dot-product with explicit temperature: `scores = QKᵀ / sqrt(768)` enforced in code review checklist. 2. Add `torch.clamp(scores, -50, 50)` pre-softmax guard. 3. Post-attention tensor health check: `assert torch.isfinite(context).all()` with `NaNError` exception routed to safe fallback. 4. Unit test suite: inject extreme values (1e38, -1e38, inf) into each modality; assert finite output or graceful error. | ML Engineering | Sprint+1 | 9 | 3 | **1** | **27** ✅ |
| **CrossModalAttention** | **FM-403:** Attention weight leakage exposing training patient embeddings — cross-attention key/value matrices retain residual information from training cohort accessible via gradient-based model inversion attack | Patient privacy breach; GDPR/HIPAA violation; regulatory non-compliance; potential reputational and legal harm | **9** | **2** | **8** | **🔴 144** | **HIGH** | 1. Apply differential privacy noise injection to attention key/value projections (DP-SGD during training, ε ≤ 8). 2. Deploy model inversion attack red-team test pre-launch; document results in Technical File. 3. Limit model API: do not return raw attention weights in production response (return only summary modality influence scores). 4. Annual third-party privacy audit required per EU MDR Article 10(2). | Security / ML Engineering | Sprint+4 | 9 | **1** | **4** | **36** ✅ |
| **CrossModalAttention** | **FM-404:** 768-dimensional projection matrix dimension mismatch at runtime — modality encoder output dimension changes after hotfix deployment without updating attention projection layer | Runtime `RuntimeError: mat1 and mat2 shapes cannot be multiplied` exception; inference service crashes; 500 returned to EHR integration; availability SLA breached | **6** | **3** | **3** | **54** ✅ | Acceptable | 1. Declare encoder output dimensions as typed constants in shared config module; attention layer reads from same constants. 2. Add model architecture smoke test to CI/CD: verify all layer input/output shapes on dummy tensor before deployment. 3. Kubernetes rolling deployment with readiness probe: pod must pass shape validation test before receiving traffic. | Software Engineering | Sprint+1 | 6 | 3 | **1** | **18** ✅ |

---

### Component 5: Output Formatter

*(FHIR RiskAssessment Builder)*

| Component | Failure Mode | Effect on System / Patient | S | O | D | RPN | Priority | Corrective Actions | Responsible | Target Date | New S | New O | New D | New RPN |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|:---:|:---:|:---:|:---:|
| **Output Formatter — FHIR** | **FM-501:** Probability score serialized with incorrect FHIR resource type — risk score written to `DiagnosticReport` instead of `RiskAssessment`; EHR system silently ignores or misroutes output | Clinician's EHR never surfaces the AD progression risk score; clinician makes treatment decision without CDS input; intended use not fulfilled | **8** | **3** | **6** | **🔴 144** | **HIGH** | 1. Define and version-lock FHIR output profile in NeuroFusion-AD Implementation Guide; validate every response against StructureDefinition using HAPI FHIR validator. 2. Integration test: parse response with reference EHR simulator (SMART on FHIR sandbox); assert `RiskAssessment.prediction.probabilityDecimal` populated. 3. Add contract test (Pact) with EHR vendor to detect format drift. | Clinical Informatics | Sprint+2 | 8 | 3 | **1** | **24** ✅ |
| **Output Formatter — FHIR** | **FM-502:** Confidence interval truncated to zero — regression RMSE and classification confidence bounds not included in FHIR output due to serialization bug; clinician sees point estimate only with no uncertainty indication | Clinician treats deterministic-appearing output as certainty; overconfident clinical decision made on inherently uncertain model output; labeling non-compliance (21 CFR 882) | **8** | **4** | **5** | **🔴 160** | **HIGH** | 1. Confidence interval serialization covered by mandatory automated integration test asserting `RiskAssessment.prediction.probabilityRange` present and non-null. 2. UI layer must display uncertainty bounds with equal visual prominence to point estimate (clinician review and sign-off required). 3. IFU explicitly states output must not be interpreted without uncertainty range. | Software Engineering / Clinical | Sprint+2 | 8 | 4 | **1** | **32** ✅ |
| **Output Formatter — FHIR** | **FM-503:** Model version not embedded in FHIR output — production inference uses stale model version after deployment without updating `RiskAssessment.method.coding`; version traceability lost | Post-market surveillance cannot attribute outputs to correct model version; adverse event investigation unable to determine if incident model is current or prior version; regulatory audit failure | **7** | **4** | **5** | **🔴 140** | **HIGH** | 1. Inject model version from environment variable at container build time; assert in readiness probe. 2. Every `RiskAssessment` response MUST include `method.coding[0].code` = `NeuroFusion-AD-v{SEMVER}` and `method.coding[0].system` = `https://neurofusion-ad.example.com/model-versions`. 3. Automated test asserts version string matches current container image tag on every CI build. | Software Engineering / MLOps | Sprint+1 | 7 | **1** | **2** | **14** ✅ |
| **Output Formatter — FHIR** | **FM-504:** Special-character encoding failure — patient name or note field containing non-ASCII characters (e.g., accented European names) causes UTF-8 encoding error; FHIR bundle serialization fails silently, returning empty or partial response | Patients with non-ASCII demographics receive no CDS output; potential equity disparity in CDS availability across patient populations; FHIR R4 non-compliance | **6** | **4** | **5** | **🔴 120** | **HIGH** | 1. Enforce UTF-8 encoding on all string fields in FHIR serializer; use Python `str.encode('utf-8', errors='replace')` with logging on replacement. 2. Integration test dataset must include patients with names containing umlauts, accents, and CJK characters. 3. FastAPI response headers enforce `Content-Type: application/fhir+json; charset=utf-8`. | Software Engineering | Sprint+2 | 6 | 4 | **2** | **48** ✅ |

---

### Component 6: Audit Logger

*(PostgreSQL Audit Trail)*

| Component | Failure Mode | Effect on System / Patient | S | O | D | RPN | Priority | Corrective Actions | Responsible | Target Date | New S | New O | New D | New RPN |
|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|---|:---:|:---:|:---:|:---:|
| **Audit Logger — PostgreSQL** | **FM-601:** Audit log write failure — PostgreSQL connection timeout or disk-full condition causes audit record to be silently dropped; inference continues without log entry | Inference occurs without traceable audit record; regulatory non-compliance (21 CFR 11, EU MDR Annex IX); adverse event investigation has incomplete evidence trail; potential enforcement action | **9** | **3** | **5** | **🔴 135** | **HIGH** | 1. Audit log write must be transactional with inference: use two-phase commit or write-ahead queue (Kafka/Redis Streams) to guarantee delivery before returning response. 2. If audit write fails, inference MUST be aborted and HTTP 503 returned — no output without audit record. 3. Monitor PostgreSQL disk utilization; automated alert at 80% capacity; auto-expansion policy in Kubernetes PVC. 4. Test failure scenario in staging: induce DB failure during inference; assert no response returned and error logged to SIEM. | DevOps / Compliance | Sprint+2 | 9 | 3 | **1** | **27** ✅ |
| **Audit Logger — PostgreSQL** | **FM-602:** Audit log tampering — audit records modified or deleted by privileged insider or compromised service account; immutability of audit trail compromised | Fraudulent audit trail; inability to reconstruct clinical decision history; regulatory and legal non-compliance; loss of evidence in malpractice investigation | **9** | **2** | **6** | **🔴 108** | **HIGH** | 1. Implement append-only audit table with PostgreSQL row-level security: `GRANT INSERT ON audit_log TO neurofusion_app; REVOKE UPDATE, DELETE ON audit_log FROM ALL`. 2. Hash-chain audit records: each row stores `SHA256(previous_row_hash \|\| current_row_data)`; integrity verifiable on demand. 3. Daily export of audit log to immutable S3 bucket with object lock (WORM). 4. Quarterly integrity verification job comparing DB records against S3 archive. 5. Privileged access management: DBA access to production audit tables requires dual approval. | Security / Compliance | Sprint+2 | 9 | **1** | **2** | **18** ✅ |
| **Audit Logger — PostgreSQL** | **FM-603:** PHI logged in plaintext in audit fields — patient name, DOB, or raw biomarker values stored in audit `details` JSON column without encryption; column encryption not applied | HIPAA breach if audit database compromised; patient data exposed to unauthorized DB administrators; GDPR Article 32 non-compliance | **9** | **3** | **4** | **🔴 108** | **HIGH** | 1. Apply PostgreSQL column-level encryption (pgcrypto) to all PHI fields in audit table. 2. Audit records store patient reference via pseudonymous UUID only; PHI linkage maintained in separate access-controlled identity table. 3. AES-256 encryption at rest enabled at tablespace level (confirmed in deployment checklist). 4. DBA access to identity linkage table requires separate RBAC role `phi_access` with audit of every access. | Security / DPO | Sprint+1 | 9 | **1** | **2** | **18** ✅ |
| **Audit Logger — PostgreSQL** | **FM-604:** Log injection via crafted user-agent or request metadata — attacker submits HTTP request with newline characters in header fields that corrupt structured audit log format; log parsing tools misinterpret injected log entries | Log analysis and SIEM tooling produces incorrect audit reports; security incident may be masked; regulatory audit produces misleading evidence | **7** | **3** | **4** | **84** ✅ | Acceptable | 1. Sanitize all string fields before audit persistence: strip `\n`, `\r`, `\t`, null bytes. 2. Use structured JSON logging exclusively (never interpolated strings); SIEM configured to parse JSON fields. 3. Header value length limits enforced at Nginx layer (max 8KB per header). | Security Engineering | Sprint+2 | 7 | 3 | **2** | **42** ✅ |

---

### Component 7: API Gateway

*(FastAPI + Nginx)*

| Component | Failure Mode | Effect on System / Patient | S | O | D | RPN | Priority | Corrective Actions | Responsible | Target Date | New S | New O | New D | New RPN |
|---|---|---|:---:|:---:|:---:|:---:|:---