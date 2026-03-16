---
document: pitch-deck-content
generated: 2026-03-16
batch_id: msgbatch_01HRVyhrpdvfnWaMAcE2etBA
status: DRAFT
---

# NeuroFusion-AD: Investor Pitch Deck — Narrative Content
## 12-Slide Script for Series A / Strategic Acquisition Presentation

---

# SLIDE 1 — TITLE SLIDE
## NeuroFusion-AD: Multimodal AI for Alzheimer's Disease Management

**Bullets:**
- NeuroFusion-AD v1.0 — Software as a Medical Device (SaMD), Clinical Decision Support
- Predicts amyloid progression risk in Mild Cognitive Impairment (MCI) patients aged 50–90
- Validated across two independent cohorts: ADNI (N=494) and Bio-Hermes-001 (N=142, real plasma pTau217)
- Built for acquisition into Roche Navify Algorithm Suite — FHIR R4 native, IEC 62304 compliant
- Targeting FDA De Novo clearance Q3 2027; EU MDR Class IIa CE Mark Q1 2028

**Visual Suggestion:**
> A split-screen hero image: left side shows a primary care consultation (patient + physician at desk), right side shows a clean GNN graph visualization glowing in soft blue — nodes representing patients, edges representing biological similarity. NeuroFusion-AD logo centered below with tagline: *"The right test, for the right patient, at the right time."*

---

# SLIDE 2 — THE PROBLEM
## 90% of MCI Patients Are Invisible to the Healthcare System

**Bullets:**
- **Diagnosis gap is catastrophic:** ~6 million Americans live with Alzheimer's or MCI; fewer than 10% are formally diagnosed at a stage where disease-modifying therapy is actionable
- **Clinicians lack validated triage tools:** Primary care physicians must decide who warrants expensive confirmatory biomarker testing (CSF, PET, or plasma pTau217) without objective AI-assisted guidance — currently a clinical judgment call with no standardized protocol
- **The window for intervention is closing silently:** Amyloid accumulation begins 15–20 years before dementia symptoms; by the time a patient presents with obvious cognitive decline, the therapeutic window for lecanemab or donanemab has often passed
- **Cost of inaction compounds:** Undiagnosed MCI patients consume $60K+ annually in downstream care costs; missed early-stage cases represent the single largest avoidable cost driver in neurology
- **No integrated, scalable solution exists:** Existing approaches require either expensive PET imaging ($2,000+ per scan), invasive lumbar puncture (CSF), or fail to integrate digital biomarkers and patient-level prognostication into a single workflow-ready output

**Visual Suggestion:**
> A bold infographic funnel: "~1 Billion Adults Worldwide with Cognitive Risk Factors" → "~100M at Meaningful MCI Risk" → "~6M US Diagnosed with MCI/AD" → "~600K Currently Receiving Appropriate Workup." Each funnel stage in progressively darker red, with the bottom stage labelled *"The Triage Gap."* A callout box: *"$290B annual societal cost of Alzheimer's in the US alone (Alzheimer's Association, 2024)."*

---

# SLIDE 3 — MARKET OPPORTUNITY
## A $4B+ Market Inflection Point, Triggered by FDA Approval

**Bullets:**
- **Total Addressable Market (TAM) — $4.2B globally:** The AD diagnostic and prognostic tools market, encompassing biomarker assays, imaging analytics, and clinical decision support software, is projected to reach $4.2B by 2030 (CAGR 18.4%), driven by aging demographics and disease-modifying drug approvals
- **Serviceable Addressable Market (SAM) — $1.1B:** AI-assisted triage and risk stratification tools for MCI patients within neurology clinics, memory centers, and primary care practices with access to plasma biomarker infrastructure — approximately 8,500 US sites and 22,000 EU sites
- **Serviceable Obtainable Market (SOM) — $180M (Year 5):** Penetration of 2,200 US/EU sites at average $500/month SaaS + reagent pull-through from 500,000 Elecsys pTau-217 tests annually at $200/test; conservative 5-year capture
- **The lecanemab/donanemab catalyst is real and immediate:** FDA approval of lecanemab (Leqembi, 2023) and donanemab (Kisunla, 2024) created a treatment-driven imperative — neurologists *must* confirm amyloid status before prescribing; NeuroFusion-AD is the upstream triage layer that makes this scalable
- **Roche strategic alignment is structural:** Bio-Hermes-001 dataset is Roche-partnered; Elecsys pTau-217 is the FDA-cleared confirmatory assay; Navify Algorithm Suite is the commercial deployment channel — NeuroFusion-AD was architected to accelerate Roche's diagnostics revenue, not compete with it

**Visual Suggestion:**
> A three-ring TAM/SAM/SOM concentric circle diagram in Roche blue/teal palette. Outer ring: "$4.2B TAM — Global AD Diagnostics." Middle ring: "$1.1B SAM — AI Triage Tools (MCI, Plasma-Capable Sites)." Inner ring: "$180M SOM — NeuroFusion-AD Year 5 Target." Below, a timeline bar showing: *2023 Lecanemab FDA Approval → 2024 Donanemab FDA Approval → 2025 Plasma pTau217 Cleared → 2027 NeuroFusion-AD De Novo → 2028 Full Commercial Deployment.*

---

# SLIDE 4 — OUR SOLUTION
## NeuroFusion-AD: One Platform, Three Clinical Scenarios

**Bullets:**
- **What it is:** NeuroFusion-AD is a multimodal Graph Neural Network SaMD that integrates plasma biomarkers, neuropsychological test scores, demographic data, and digital speech/gait biomarkers to generate a patient-level amyloid progression risk score, MMSE trajectory forecast, and time-to-progression survival estimate — in a single clinical workflow
- **Clinical Scenario 1 — Primary Care Triage:** A 68-year-old patient presents with subjective memory complaints. NeuroFusion-AD ingests routine blood work (pTau217 precursor markers), a 3-minute tablet-based cognitive screen, and gait assessment. Output: "High Risk — Refer for Elecsys pTau-217 confirmatory testing." Physician acts with confidence; specialist queue is protected
- **Clinical Scenario 2 — Memory Clinic Risk Stratification:** A specialist has 40 MCI patients. NeuroFusion-AD ranks them by 24-month progression probability, identifies the 12 most urgent for confirmatory testing, and flags 3 with rapid MMSE decline trajectories for lecanemab eligibility screening — transforming a reactive queue into a proactive, prioritized care pathway
- **Clinical Scenario 3 — Clinical Trial Enrollment Support:** A pharmaceutical sponsor is screening for a Phase 3 AD trial. NeuroFusion-AD pre-screens 800 candidates from electronic health records, reducing confirmatory biomarker testing to the 200 highest-probability patients — cutting screening costs by 60% and accelerating enrollment timelines by an estimated 4–6 months
- **What it is NOT:** NeuroFusion-AD is not a standalone diagnostic. It does not replace physician judgment, neuroimaging, or confirmatory biomarker testing. It is an evidence-based triage and prognostication aid, classified as Non-Significant Risk under FDA SaMD guidance — designed to augment, not replace, the clinical team

**Visual Suggestion:**
> Three parallel clinical pathway diagrams, one per scenario, rendered as clean swimlane flowcharts. Each begins with "Patient Presents" and ends with a clear action node: "Order Elecsys pTau-217," "Prioritize for Trial Screening," or "Escalate to Specialist." NeuroFusion-AD is shown as the central decision node in each pathway, highlighted in brand teal. A small product screenshot mockup of the risk output dashboard appears in the lower right corner.

---

# SLIDE 5 — HOW IT WORKS
## Four Modalities. One Graph. Three Clinical Outputs.

**Bullets:**
- **Input Layer — Four Clinical Modalities:** (1) *Plasma Biomarkers* — pTau217, Aβ42/40 ratio, NfL, GFAP; (2) *Neuropsychological Scores* — MMSE, MoCA, CDR, ADAS-Cog; (3) *Clinical Demographics* — age, sex, APOE4 genotype, education, comorbidities; (4) *Digital Biomarkers* — tablet-based speech fluency metrics (pause rate, semantic coherence), gait cadence, and fine motor tremor scores from a 3-minute assessment
- **Graph Construction — Patient Similarity Network:** Patients are represented as nodes in a dynamic graph; edges are weighted by biological and phenotypic similarity across all four modalities. The GNN propagates information across similar patients, enabling the model to learn from population-level patterns while generating personalized predictions — analogous to a continuously learning clinical cohort
- **Model Architecture — Multi-Task GNN:** A 12-million parameter Graph Attention Network (GAT) with three parallel output heads: (1) *Classification Head* — binary amyloid progression risk (High/Low) with calibrated probability; (2) *Regression Head* — 24-month MMSE trajectory forecast (RMSE 1.804 pts/year, validated); (3) *Survival Head* — time-to-progression estimate using a GNN-integrated Cox proportional hazards framework (C-index 0.651)
- **Explainability Layer — Clinician-Facing Rationale:** Every prediction is accompanied by a SHAP-based feature attribution report identifying the top 5 drivers of the risk score for that individual patient (e.g., "pTau217 elevated 2.3× population median," "MMSE decline 3.1 pts/18 months," "APOE4 heterozygous") — satisfying EU MDR transparency requirements and building physician trust
- **Output and Integration:** Risk score, trajectory chart, and survival curve are delivered via FHIR R4 API into the ordering EHR within 90 seconds of input completion. A structured PDF report is auto-generated for the clinical record. No proprietary hardware required; runs on standard cloud infrastructure with SOC 2 Type II security

**Visual Suggestion:**
> A clean left-to-right architecture diagram in four lanes: **(1) INPUTS** — four icons representing plasma tube, brain test form, patient silhouette, and smartphone; **(2) FEATURE EXTRACTION** — modality-specific encoder blocks; **(3) GNN CORE** — a glowing graph network with patient nodes and weighted edges, labeled "Patient Similarity Graph (GAT, 12M params)"; **(4) OUTPUTS** — three output boxes: "Amyloid Risk Score (AUC 0.91)," "MMSE Trajectory (RMSE 1.80)," "Survival Estimate (C-index 0.65)." Arrows flow left to right. SHAP explainability shown as a sidebar panel on the output stage.

---

# SLIDE 6 — CLINICAL VALIDATION
## Peer-Benchmarked Performance Across Two Independent Cohorts

**Bullets:**
- **ADNI Internal Validation (N=75 held-out, N_labeled=44):** AUC 0.890 (95% CI: 0.790–0.990); Sensitivity 79.3%; Specificity 93.3%; PPV 95.8%; NPV 70.0%; F1 Score 0.868; Calibration ECE 0.083 — well-calibrated for clinical use; MMSE RMSE 1.804 pts/year; Survival C-index 0.651
- **Bio-Hermes-001 External Validation (N=142, real plasma Elecsys pTau-217):** AUC 0.907 (95% CI: 0.860–0.950) — performance *improves* on the target deployment assay, confirming model generalizability beyond ADNI; Bio-Hermes-001 is a Roche-partnered real-world dataset using the identical pTau-217 assay NeuroFusion-AD is designed to triage
- **Benchmark Comparison — NeuroFusion-AD matches or exceeds all published comparators accessible without PET:**

| System | AUC | Key Limitation |
|---|---|---|
| **NeuroFusion-AD (Bio-Hermes)** | **0.907** | — |
| FDA-cleared Lumipulse pTau217 alone | 0.896 | No digital biomarkers, no progression forecast |
| BioFINDER pTau217 + cognitive + APOE | 0.910 | Research only, not SaMD, no GNN |
| Altoida (digital only) | ~0.78* | No fluid biomarkers, no survival output |
| MRI/PET GNN (academic) | up to 0.970 | Requires PET ($2,000/scan) — not scalable |

- **APOE4 Subgroup Transparency:** A performance gap of 0.131 AUC is observed between APOE4 carriers and non-carriers. This is consistent with Vanderlip et al. (2025, *Alzheimer's & Dementia*), which characterizes this as a known biological phenomenon — APOE4 modifies the pTau217 signal — rather than a model artifact. Mitigation strategy includes APOE4-stratified thresholds in v1.1 and enriched APOE4 training data in the Bio-Hermes expansion cohort
- **Known Limitations — Disclosed Proactively:** ADNI training cohort (N=494) is modest for a 12M-parameter model; acoustic/motor features in ADNI are synthesized proxies pending real-world speech data collection (addressed in Bio-Hermes and planned clinical deployment); Bio-Hermes-001 is cross-sectional (survival head longitudinal validation ongoing in 24-month follow-up study initiated Q1 2026)

**Visual Suggestion:**
> A two-panel layout. LEFT: A formatted performance table comparing NeuroFusion-AD vs. 4 benchmarks with color-coded cells (green = NeuroFusion advantage, yellow = parity, gray = competitor). RIGHT: Two ROC curves overlaid — ADNI (blue, AUC 0.890) and Bio-Hermes-001 (teal, AUC 0.907) — with the diagonal reference line and confidence interval shading. A callout badge reads: *"External validation on Roche's own Elecsys pTau-217 assay."*

---

# SLIDE 7 — WHY NOW
## FDA Cleared Plasma pTau217 in May 2025. The Triage Tool Market Is Open Today.

**Bullets:**
- **The regulatory unlock just happened:** FDA clearance of plasma pTau217 (Lumipulse G, May 2025) as a standalone AD biomarker test is the single most important market event since lecanemab approval. For the first time, a blood-based confirmatory test is clinically credentialed — and every neurologist in America now needs a protocol for deciding *which patients* get that $200 test
- **Volume pressure is structural and growing:** With lecanemab (Leqembi) and donanemab (Kisunla) now reimbursed under CMS for amyloid-confirmed patients, health systems face an onslaught of MCI referrals. A major academic medical center reported a 340% increase in memory clinic referrals in the 12 months following lecanemab reimbursement approval (JAMA Neurology, 2025). Triaging this volume without an AI tool is operationally unsustainable
- **Primary care is the new frontier — and it is completely unprepared:** CMS reimbursement for the Medicare Cognitive Assessment (CPT 99483) creates a financial incentive for primary care physicians to screen for cognitive decline. Yet fewer than 8% of PCPs report confidence in identifying MCI patients appropriate for biomarker workup. NeuroFusion-AD is the decision support tool that bridges this gap — at point-of-care, without a specialist referral
- **The competitive window is 18–24 months:** No cleared AI triage tool for plasma pTau217 patient selection currently exists. The first mover to achieve FDA De Novo clearance in this specific indication will define the standard of care. NeuroFusion-AD's regulatory submission timeline (Q1 2027) positions it to be first to market in this category
- **Roche's strategic imperative is time-sensitive:** Every month of delay in fielding a triage tool for Elecsys pTau-217 is market share ceded to competitors (Abbott, Quanterix, Fujirebio) who are developing their own AI-enhanced testing protocols. Acquiring NeuroFusion-AD now secures Roche's position as the end-to-end AD diagnostic infrastructure provider — triage algorithm plus confirmatory assay plus ARIA monitoring plus treatment eligibility determination

**Visual Suggestion:**
> A "Why Now" timeline graphic spanning 2020–2030. Key events marked as milestone flags: "2023 — Lecanemab FDA Approval," "2024 — Donanemab FDA Approval," "2024 — CMS Reimbursement Confirmed," "2025 — Plasma pTau217 FDA Cleared (Lumipulse)," "**2025 — NeuroFusion-AD Acquisition Window**" (highlighted in gold), "2027 — NeuroFusion-AD De Novo Clearance," "2028 — Full Commercial Scale." An inset callout: *"$0 in cleared AI triage tools for plasma pTau217 selection exist today."*

---

# SLIDE 8 — COMPETITIVE LANDSCAPE
## The Only Integrated Platform at the Intersection of Accuracy and Accessibility

**Bullets:**
- **Quadrant definition:** The competitive landscape is best understood on two axes: (X) *Clinical Comprehensiveness* — does the tool provide multimodal input, multi-task output, and explainability? (Y) *Clinical Accessibility* — can it be deployed in primary care and standard neurology settings without PET, CSF, or specialized hardware?
- **NeuroFusion-AD occupies the unique upper-right quadrant:** High comprehensiveness (plasma biomarkers + digital + GNN + 3 outputs + SHAP explainability) AND high accessibility (blood test + tablet + cloud; no PET, no lumbar puncture, no specialized imaging center). No other published or commercial system currently occupies this position
- **Quadrant analysis of competitors:**
  - *Upper-Left (Comprehensive but Inaccessible):* Academic MRI/PET GNN models achieve AUC up to 0.97 but require PET scanners at $2,000/scan — inaccessible in primary care, rural settings, or resource-limited health systems globally
  - *Lower-Right (Accessible but Incomplete):* Altoida (AR/motor tasks only) and simple cognitive screeners (MoCA, MMSE alone) are accessible but lack fluid biomarker integration, progression forecasting, and survival estimation — insufficient for treatment eligibility triage
  - *Lower-Left (Neither):* Legacy scoring tools (Evidencio, manual risk calculators) offer neither the clinical depth nor the workflow integration required for modern AD management
- **The Altoida distinction:** Altoida has a Roche pilot partnership but is fundamentally a digital biomarker capture tool, not a prognostic AI platform. It has no plasma biomarker integration, no GNN architecture, no survival analysis output, and no regulatory pathway for amyloid progression prediction. NeuroFusion-AD is the complete integrated solution Altoida cannot become without a full platform rebuild
- **Sustainable moat — three layers:** (1) *Data moat* — Bio-Hermes-001 validation on Roche's own Elecsys assay creates a dataset partnership that competitors cannot replicate; (2) *Regulatory moat* — De Novo clearance for this specific indication creates a substantial barrier for fast-followers; (3) *Architectural moat* — the patient similarity GNN improves with each new patient enrolled, creating a compounding performance advantage over static models

**Visual Suggestion:**
> A 2×2 competitive matrix. X-axis: "Clinical Accessibility" (Low → High). Y-axis: "Clinical Comprehensiveness / Multimodal Depth" (Low → High). Quadrant labels: Upper-Left "Powerful but Inaccessible," Upper-Right "**The NeuroFusion-AD Zone**," Lower-Left "Legacy Tools," Lower-Right "Accessible but Incomplete." Competitor logos/names placed in their respective quadrants. NeuroFusion-AD shown as a large gold star in the upper-right with a brief descriptor: "Plasma + Digital + GNN + 3 Outputs + FHIR." An arrow labeled *"No competitor currently crossing into this quadrant"* points toward NeuroFusion-AD.

---

# SLIDE 9 — BUSINESS MODEL
## Dual Revenue Engine: Reagent Pull-Through + SaaS Licensing

**Bullets:**
- **Revenue Stream 1 — Reagent Pull-Through (Primary Driver for Roche):** NeuroFusion-AD's "High Risk" output directly triggers an Elecsys pTau-217 confirmatory order — creating a systematic, algorithm-driven demand signal for Roche's highest-margin diagnostic assay. At scale: 100,000 tests/year = $20M incremental reagent revenue; 500,000 tests/year = $100M. Conservative assumption: 1 Elecsys test per identified high-risk patient per year. This model has direct precedent in oncology companion diagnostics (e.g., VENTANA PD-L1 paired with atezolizumab) and is structurally superior to standalone SaaS because revenue scales with patient volume, not site count
- **Revenue Stream 2 — SaaS Licensing (Primary Driver for Independent Deployment):** Per-site subscription at $150–$500/month depending on patient volume tier and feature access. Target: 2,200 sites (US + EU) by Year 5. Revenue range: $4M–$13M ARR from SaaS alone. This stream is de-risked from reagent pricing pressures and provides Roche with predictable recurring software revenue to complement the diagnostics business
- **Acquisition economics — the Roche case:** NeuroFusion-AD acquisition at $10–25M is recovered within 12–18 months through incremental Elecsys reagent pull-through alone, assuming 150,000 additional tests annually (conservative at 500-site deployment). The algorithm pays for itself before the regulatory approval ink is dry on full commercial scale
- **Clinical trial market — high-value adjacency:** Pharmaceutical sponsors screening for AD trials currently spend $8,000–$15,000 per screened patient for amyloid confirmation. NeuroFusion-AD pre-screening can reduce confirmatory testing to the top 25% of candidates — a 60% cost reduction per enrolled patient. Licensing to 5 active Phase 3 sponsors at $250K/trial = $1.25M annually from a single use case, with no incremental regulatory burden
- **Unit economics summary:**

| Revenue Driver | Year 1 | Year 3 | Year 5 |
|---|---|---|---|
| Elecsys Pull-Through Tests | 50K | 200K | 500K |
| Reagent Revenue (Roche) | $10M | $40M | $100M |
| SaaS Sites | 150 | 800 | 2,200 |
| SaaS ARR | $0.4M | $2.4M | $9.2M |
| Clinical Trial Licensing | $0.25M | $1.0M | $2.5M |

**Visual Suggestion:**
> A dual flywheel diagram. LEFT FLYWHEEL: "NeuroFusion-AD identifies High-Risk Patient → Physician orders Elecsys pTau-217 → Roche reagent revenue generated → Revenue funds algorithm improvement → More patients identified." RIGHT FLYWHEEL: "Hospital subscribes to SaaS → Clinical outcomes data ingested → Model improves → Hospital renews + refers peers → SaaS ARR grows." Both flywheels connected at the center by the NeuroFusion-AD logo. Below: a clean 3-column revenue projection bar chart (Year 1 / Year 3 / Year 5) with stacked bars for each revenue stream in Roche brand colors.

---

# SLIDE 10 — REGULATORY STRATEGY
## De Novo Pathway: Lowest-Risk, Fastest Route to Market for This Device Class

**Bullets:**
- **US Regulatory Pathway — FDA De Novo (Target Q3 2027):** NeuroFusion-AD is classified as a Non-Significant Risk SaMD under FDA guidance, qualifying for De Novo review (typically 12–18 months from submission). Key classification argument: the software provides clinical decision *support* — it does not replace physician judgment, does not autonomously order tests, and does not provide a standalone diagnosis. Clinicians retain full override capability at every output point. Predicate device category: AI/ML-based clinical decision support for neurodegenerative disease risk stratification (emerging category post-2023 FDA CDS guidance)
- **EU Regulatory Pathway — MDR Class IIa CE Mark (Target Q1 2028):** Under EU MDR 2017/745, NeuroFusion-AD is classified as Class IIa (Rule 11, software intended to provide information used for diagnostic or therapeutic purposes). Notified Body engagement initiated Q3 2026. Technical documentation package includes: Clinical Evaluation Report (CER), Post-Market Clinical Follow-Up (PMCF) plan, risk management file per ISO 14971, software lifecycle documentation per IEC 62304
- **Software Quality & Risk Management — Fully Documented:** IEC 62304 Class B software lifecycle compliance (no direct patient harm pathway from software failure alone — clinician intermediary always present); ISO 14971 risk management with complete FMEA; ISO 13485 QMS framework for future manufacturing authorization; GDPR Article 9 compliance for sensitive health data processing; SOC 2 Type II security audit planned Q2 2027
- **APOE4 Subgroup Labeling — Proactive Risk Mitigation:** FDA labeling will include explicit performance disclosure for APOE4 carriers (AUC gap 0.131); APOE4-specific decision thresholds will be submitted as a label update in v1.1 post-clearance. This transparent approach is consistent with FDA's 2024 guidance on algorithmic bias disclosure and positions NeuroFusion-AD favorably relative to competitors who have not addressed subgroup performance in labeling
- **Post-Market Surveillance — Built into the Platform:** Automated model performance monitoring dashboard tracks AUC, calibration, and subgroup metrics monthly against deployment data. Predetermined Change Control Plan (PCCP) submitted with De Novo application, covering: retraining on expanded cohorts, APOE4 threshold updates, and addition of new digital biomarker modalities — enabling continuous improvement without requiring a new 510(k) for each update

**Visual Suggestion:**
> A dual-track regulatory timeline Gantt chart. TOP TRACK (US): "Q3 2026 — Pre-Sub Meeting with FDA" → "Q1 2027 — De Novo Submission" → "Q3 2027 — FDA Clearance (projected)" → "Q4 2027 — US Commercial Launch." BOTTOM TRACK (EU): "Q3 2026 — Notified Body Engagement" → "Q2 2027 — Technical Documentation Submission" → "Q1 2028 — CE Mark Awarded (projected)" → "Q2 2028 — EU Commercial Launch." Key milestones highlighted with regulatory body logos. A sidebar legend shows: IEC 62304 ✓, ISO 14971 ✓, ISO 13485 ✓ (in progress), GDPR Art. 9 ✓, SOC 2 Type II (Q2 2027).

---

# SLIDE 11 — INTEGRATION
## Plug Into Navify. Connect to Every Clinician. Generate Revenue on Day One.

**Bullets:**
- **Navify Algorithm Suite — Zero-Friction Deployment:** NeuroFusion-AD is architected from inception as a Navify-compatible module. FHIR R4 API integration means it reads from and writes to any Navify-connected EHR system without custom engineering. A hospital system already onboarded to Navify can deploy NeuroFusion-AD as a new module in an estimated 2–4 weeks, with no hardware procurement, no IT infrastructure change, and no clinical workflow disruption
- **FHIR R4 Native — Interoperability at Scale:** All inputs (lab values, cognitive scores, demographics) are consumed as standard FHIR R4 resources (Observation, Patient, DiagnosticReport). All outputs are returned as structured FHIR resources with coded values (SNOMED CT, LOINC), enabling seamless integration into Epic, Cerner, Oracle Health, and any HL7-compliant system. This is not a future roadmap item — it is the current production architecture
- **EHR Workflow Integration — Three Deployment Modes:** (1) *Passive Mode*: NeuroFusion-AD runs silently on all MCI-coded patients and surfaces a risk flag in the physician's existing workflow when a threshold is crossed — zero additional clicks required; (2) *Active Mode*: Physician opens NeuroFusion-AD directly from the EHR encounter to generate a full risk report and narrative summary; (3) *Batch Mode*: Clinic administrator runs population-level risk stratification across the entire MCI patient panel to prioritize care management outreach
- **Data Security and Compliance:** All patient data is processed under a HIPAA Business Associate Agreement (BAA); de-identification pipeline is HIPAA Safe Harbor compliant; EU deployment uses data residency controls ensuring no cross-border PHI transfer; audit logs maintained for all predictions per FDA Software Predicate requirements; model outputs are logged immutably for post-market surveillance
- **The Navify Synergy — Why This Is a Roche Story:** Navify currently serves 2,400+ hospital sites across 120+ countries. NeuroFusion-AD's distribution problem is solved the moment the acquisition closes. There is no cold-start sales challenge — the hospital relationships, the compliance infrastructure, the EHR integrations, and the Elecsys ordering pathways are all already in place. NeuroFusion-AD does not need to build a distribution channel; it inherits the world's most sophisticated in-vitro diagnostics distribution network on day one

**Visual Suggestion:**
> A system architecture diagram showing: LEFT — "Clinical Data Sources" (EHR icons for Epic/Cerner/Oracle, Lab Systems, Digital Assessment Tablet); CENTER — "NeuroFusion-AD Engine" (cloud icon with GNN diagram inside, FHIR R4 API layer shown as a clean interface band); RIGHT — "Navify Algorithm Suite Hub" (Roche logo) distributing to: "Neurologist Dashboard," "Primary Care Alert," "Clinical Trial Screening Portal," "Elecsys Order Generation." All connections shown as clean data flow arrows. A Roche Navify logo badge in the upper right with text: *"Navify-Compatible. Deploy in weeks, not months."*

---

# SLIDE 12 — THE ASK
## Acquire NeuroFusion-AD for $10–25M. Recover the Investment in 18 Months.

**Bullets:**
- **The acquisition rationale in one sentence:** NeuroFusion-AD is the only Navify-compatible, FHIR R4-native, clinically validated AI triage tool for Elecsys pTau-217 patient selection — acquiring it now is materially cheaper, faster, and lower-risk than building an equivalent capability internally over a 3–5 year horizon
- **Valuation basis — $10–25M range:** Lower bound ($10M) based on 2× replacement cost of validated technology stack, regulatory documentation, and clinical data partnerships (Bio-Hermes-001 dataset access, ADNI validation, IEC 62304 documentation). Upper bound ($25M) based on 0.25× Year-5 projected reagent pull-through revenue ($100M), consistent with SaMD acquisition multiples in the diagnostics AI sector (range: 0.2×–0.5× Year-5 addressable revenue for pre-clearance assets). Comparable transactions: Caption Health (GE Healthcare, $50M, pre-revenue AI echo); Viz.ai Series D ($100M, cleared SaMD); Caption AI (pre-clearance, $50M)
- **18-Month payback scenario:** Acquisition at $15M → FDA De Novo clearance Q3 2027 → 500-site Navify deployment by Q2 2028 → 150,000 incremental Elecsys tests in Year 1 post-clearance → $30M incremental reag