# Phase 3 Exit Checklist — Integration, Demo, Investor Package

**Status**: IN PROGRESS
**RunPod**: root@213.192.2.67:40046 (updated after restart)
**Budget**: $34 API credits | Model: claude-sonnet-4-6 only

---

## Step 1: Check Available Skills (do first)
- [ ] `ls /mnt/skills/public/` — check what's available
- [ ] `ls /mnt/skills/examples/` — check example skills
- [ ] Use healthcare/FHIR skill if available for api-integration-agent
- [ ] Use frontend-design skill for demo-agent

## Step 2: API Integration (api-integration-agent)

- [ ] `src/data/inference_preprocessor.py` — single-patient inference preprocessor
  - [ ] fluid_input_dim = 2 (NOT 3 — ABETA removed)
  - [ ] Handles missing modalities with median imputation
  - [ ] from_fhir() method parses FHIR bundle
- [ ] `src/api/fhir_validator.py` — FHIR R4 validator + extractor
- [ ] `src/api/fhir_output.py` — builds FHIR RiskAssessment
- [ ] `src/api/main.py` — FastAPI app with /health + /fhir/RiskAssessment/$process
  - [ ] Temperature scaling applied (T=0.756)
  - [ ] Monte Carlo CI (30 samples)
  - [ ] Audit logging
- [ ] `scripts/db/init.sql` — PostgreSQL audit table
- [ ] `Dockerfile` — multi-stage, non-root, health check
- [ ] `docker-compose.yml` — API + PostgreSQL + Redis
- [ ] `tests/integration/test_api.py` — API tests
- [ ] `pytest tests/integration/ -v` — 0 failures
- [ ] `docker-compose up --build` succeeds on RunPod
- [ ] Latency: p95 < 2000ms (measured with 10 requests)

## Step 3: Clinical Demo Application (demo-agent)

- [ ] `demo/backend/demo_api.py` — lightweight FastAPI with pre-computed results
- [ ] `demo/frontend/` — React app
  - [ ] Scenario 1: Primary Care Triage (Margaret Chen)
  - [ ] Scenario 2: Neurologist Staging (Robert Martinez)
  - [ ] Scenario 3: Treatment Monitoring (Dorothy Walsh)
  - [ ] RiskGauge component (semicircular)
  - [ ] ModalityImportanceChart (horizontal bars)
  - [ ] KaplanMeierCurve (Recharts)
  - [ ] AlertBanner (RED/YELLOW/GREEN)
- [ ] `demo/docker-compose.demo.yml`
- [ ] `demo/README.md` with launch instructions
- [ ] Demo runs: `docker-compose -f demo/docker-compose.demo.yml up`
- [ ] All 3 scenarios show correct predictions

## Step 4: Investor Documents (Batch API)

- [ ] `python scripts/batch/generate_phase3_docs.py --submit`
- [ ] Wait for batch completion
- [ ] `python scripts/batch/generate_phase3_docs.py --retrieve`
- [ ] `docs/investor/executive_summary.md` — no placeholder text
- [ ] `docs/investor/pitch_deck_content.md` — 12 slides narrative
- [ ] `docs/investor/competitive_analysis.md` — benchmarked vs. Lumipulse, Altoida
- [ ] `docs/investor/technical_due_diligence.md` — honest, complete
- [ ] `docs/clinical/CVR_v2.0.md` — updated with Phase 2B metrics
- [ ] `docs/dhf/DHF_final_index.md` — complete DHF compilation

## Step 5: Final Quality Gates

- [ ] `pytest tests/ -v` — 0 failures (all 142+ tests)
- [ ] `pytest tests/integration/ -v` — 0 API test failures
- [ ] All documents reviewed for placeholder text
- [ ] `git commit -m "Phase 3 complete"` + `git push`

## Step 6: Completion

- [ ] `PHASE3_COMPLETE.md` written with:
  - API endpoint URL
  - Demo access URL
  - All investor document paths
  - Final metrics summary
  - Known remaining items for Phase 4
- [ ] **STOP — Human gate review and investor presentation preparation**

---

## API Performance Targets (hard gates)
| Metric | Target | Status |
|--------|--------|--------|
| p95 latency | < 2000ms | ⬜ |
| FHIR R4 compliance | Pass | ⬜ |
| Docker build | Success | ⬜ |
| Integration tests | 0 failures | ⬜ |
