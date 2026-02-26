# NeuroFusion-AD: Master Agent Constitution

## PROJECT IDENTITY

- **Name**: NeuroFusion-AD
- **Purpose**: Multimodal Graph Neural Network for Alzheimer's Disease Progression Prediction
- **Target**: Roche Information Solutions acquisition (Navify Algorithm Suite integration)
- **Regulatory Class**: SaMD (Software as a Medical Device) — FDA De Novo + EU MDR Class IIa
- **Current Phase**: PHASE 1 — Foundation, Requirements & Architecture

---

## CRITICAL RULES (read before doing anything)

1. **Every file you create must be committed to git** — run `git add . && git commit -m "[role]: description"` after each deliverable
2. **Never cross file ownership boundaries** (see File Ownership below) — if you're the ML agent, don't touch docs/regulatory/
3. **Write your completion summary to `docs/agent_handoffs/[your_role]_[YYYY-MM-DD].md`** before ending any session
4. **Check `docs/agent_handoffs/`** at the start of every session before doing any work — avoid duplication
5. **Never hardcode patient data, API keys, or PHI anywhere** — use environment variables
6. **Never log PHI** — hash patient IDs before any logging (`hashlib.sha256(patient_id.encode()).hexdigest()`)
7. **All code requires a docstring + at least one unit test** — no exceptions
8. **IEC 62304 compliance is non-negotiable** — every design decision must be documented in the DHF

---

## PHASE 1 EXIT CRITERIA (the definition of "done" for Phase 1)

Phase 1 is complete ONLY when ALL of the following are checked off in `docs/PHASE1_CHECKLIST.md`:

- [ ] Software Requirements Specification (SRS) v1.0 — IEC 62304 compliant, 40-60 pages
- [ ] Software Architecture Document (SAD) v1.0 — IEC 62304 compliant, 50-70 pages
- [ ] Risk Management File (RMF) v1.0 — ISO 14971 compliant (Hazard Analysis + FMEA)
- [ ] Software Development Plan (SDP) v1.0
- [ ] Regulatory Strategy Document v1.0
- [ ] Data Requirements Document v1.0
- [ ] Design History File (DHF) Phase 1 folder compiled
- [ ] Traceability Matrix v0.1 (all requirements linked to design elements)
- [ ] All 4 modality encoders implemented + unit tested
- [ ] Cross-modal attention mechanism implemented + unit tested
- [ ] Patient similarity GNN implemented + unit tested
- [ ] Full NeuroFusion-AD model end-to-end forward pass working
- [ ] DataLoader tested (no errors on synthetic data)
- [ ] Gate review checklist complete (docs/regulatory/gate_review_phase1.md)

**STOP after Phase 1 is complete. Write PHASE1_COMPLETE.md and wait for human review.**

---

## FILE OWNERSHIP (parallel agents must not cross these)

| Agent Role            | Owns                                                   | Never touches                     |
| --------------------- | ------------------------------------------------------ | --------------------------------- |
| `regulatory-agent`    | `docs/regulatory/`, `docs/dhf/`, `docs/clinical/`      | `src/`, `tests/`                  |
| `ml-architect-agent`  | `src/models/`, `src/training/`, `configs/`             | `docs/regulatory/`, `src/api/`    |
| `data-engineer-agent` | `src/data/`, `data/`, `notebooks/`                     | `docs/regulatory/`, `src/models/` |
| `api-agent`           | `src/api/`, `src/utils/`                               | `src/models/`, `docs/regulatory/` |
| `devops-agent`        | `Dockerfile`, `docker-compose.yml`, `k8s/`, `.github/` | `src/`, `docs/regulatory/`        |

**Shared files (coordinate before editing):** `requirements.txt`, `README.md`, `CLAUDE.md`

---

## SUB-AGENT ROUTING RULES

**Spawn parallel agents when ALL are true:**

- 3+ independent tasks with no shared state
- Clear file boundary separation (see ownership above)
- No task needs output from another to start

**Run sequentially when ANY is true:**

- Task B needs Task A's output
- Tasks touch shared files
- Unclear scope (plan first, then execute)

**Use background dispatch for:**

- Research tasks (no file modifications)
- Reading/summarizing existing documents
- Results not immediately blocking

---

## TECHNICAL SPECIFICATIONS (non-negotiable)

### Model Architecture

- Framework: PyTorch 2.1.2 + PyTorch Geometric 2.5.0
- Encoders: 4 modality encoders (fluid biomarker, acoustic, motor, clinical/demographic)
- All encoders output dimension: 768
- Fusion: CrossModalAttention (embed_dim=768, num_heads=8)
- GNN: GraphSAGE (3 layers, hidden_dim=768)
- Output heads: classification (BCEWithLogits), regression (MSE), survival (Cox)

### Validated Input Ranges (hard constraints — reject outside these)

- pTau-217: 0.1–100 pg/mL
- Abeta42/40 ratio: 0.01–0.30
- NfL: 5–200 pg/mL
- Acoustic jitter: 0.0001–0.05
- Acoustic shimmer: 0.001–0.3
- MMSE: 0–30

### Performance Targets

- Inference latency: p95 < 2.0 seconds
- Classification AUC: ≥ 0.85
- Regression RMSE: ≤ 3.0 points/year
- Survival C-index: ≥ 0.75

### API Contract

- Protocol: FHIR R4
- Endpoint: POST /fhir/RiskAssessment/$process
- Input: FHIR Parameters resource
- Output: FHIR RiskAssessment resource
- Auth: OAuth 2.0 (SMART on FHIR)

### Security (non-negotiable)

- Encryption at rest: AES-256
- Encryption in transit: TLS 1.3 minimum
- Audit trail: Every prediction logged to PostgreSQL with hashed patient ID
- No PHI in logs, ever

### Technology Stack

- Python: 3.10
- API: FastAPI + Pydantic v2
- DB: PostgreSQL 14
- Container: Docker + Kubernetes
- CI/CD: GitHub Actions
- Monitoring: Prometheus + Grafana
- Experiment tracking: Weights & Biases

---

## REGULATORY FRAMEWORK

- **FDA pathway**: De Novo (predicate: Prenosis Sepsis ImmunoScore)
- **EU pathway**: MDR Class IIa, Notified Body review
- **Software lifecycle**: IEC 62304 Class B
- **Risk management**: ISO 14971
- **Intended use**: "Clinical decision support tool to aid assessment of Alzheimer's disease progression risk in patients aged 50–90 with Mild Cognitive Impairment"
- **NOT a diagnostic device** — always label outputs as "Aid, not replacement for clinical judgment"
- **Mandatory disclaimer on every output**: "This tool is intended to support, not replace, clinical judgment."

### Document IDs (use these consistently)

- SRS-001, SAD-001, RMF-001, SDP-001, REG-001 (regulatory strategy), DRD-001 (data requirements)

---

## CODING STANDARDS

- Style: PEP 8 (enforced by ruff, not by inline instructions)
- Docstrings: Google style, every public function
- Type hints: required on all function signatures
- Error handling: try/except with specific exception types + logging
- Logging: structlog, JSON format, never log PHI
- Testing: pytest, minimum one test per public function
- Git commits: `[role]: concise description` (e.g., `[ml-architect]: implement FluidBiomarkerEncoder`)

---

## PROJECT STRUCTURE

```
neurofusion-ad/
├── CLAUDE.md                          ← YOU ARE HERE
├── docs/
│   ├── regulatory/
│   │   ├── srs/SRS_v1.0.md
│   │   ├── sad/SAD_v1.0.md
│   │   ├── rmf/RMF_v1.0.md
│   │   ├── sdp/SDP_v1.0.md
│   │   └── regulatory_strategy_v1.0.md
│   ├── dhf/                           ← Design History File
│   ├── clinical/
│   └── agent_handoffs/                ← Inter-session memory
├── src/
│   ├── models/
│   │   ├── encoders.py
│   │   ├── cross_modal_attention.py
│   │   ├── gnn.py
│   │   └── neurofusion_model.py
│   ├── data/
│   │   ├── adni_preprocessing.py
│   │   ├── digital_biomarker_synthesis.py
│   │   └── dataset.py
│   ├── training/
│   ├── api/
│   └── utils/
├── tests/
│   ├── unit/
│   └── integration/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── scripts/
│   └── batch/                         ← Batch API scripts
└── .claude/
    └── agents/                        ← Custom subagent definitions
```

---

## HANDOFF PROTOCOL

At the end of every session (or when context approaches limit), write to `docs/agent_handoffs/`:

```markdown
# [Agent Role] Handoff — [YYYY-MM-DD HH:MM]

## Completed This Session

- [file created/modified] — [what it does]

## Decisions Made (with rationale)

- [decision]: [why]

## Current State

- Working: [list]
- Blocked: [list]

## Next Session Must Start With

1. [specific first task]
2. [specific second task]

## Open Questions for Human Review

- [question if any]
```

---

## PHASE GATE — HARD STOP INSTRUCTION

When Phase 1 exit criteria are ALL met:

1. Run the full test suite: `pytest tests/ -v`
2. Verify all documents exist in `docs/regulatory/`
3. Write `PHASE1_COMPLETE.md` with a full summary
4. Commit everything: `git add . && git commit -m "Phase 1 complete — awaiting human gate review"`
5. **STOP. Do not begin Phase 2. Wait for human approval.**
