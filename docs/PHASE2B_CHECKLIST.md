# Phase 2B Exit Checklist — Model Remediation

**Status**: Step 1 COMPLETE — awaiting RunPod training (Step 2)
**Purpose**: Fix Phase 2 failures before Phase 3 begins
**Budget**: ~$12 remaining — all agents use claude-sonnet-4-6 only

---

## Local Code Fixes (remediation-agent)

- [x] Leakage confirmed: ABETA42_CSF Pearson r = -0.8644 with AMYLOID_POSITIVE documented
- [x] `src/data/adni_preprocessing.py` — ABETA42_CSF removed from fluid features (`_FLUID_DIM = 2`)
- [x] `src/data/adni_preprocessing.py` — ABETA42_CSF kept as `ABETA42_CSF_LABEL_SOURCE` (metadata only)
- [ ] ADNI processed CSVs regenerated without ABETA42 in fluid columns (RunPod Step 2)
- [x] `src/models/encoders.py` — FluidBiomarkerEncoder INPUT_DIM = 2, OUTPUT_DIM = 256
- [x] `src/models/encoders.py` — all encoders reduced: embed_dim=256, dropout=0.4 (~2.24M params)
- [x] `src/data/csv_dataset.py` — ADNI_FLUID_COLS / BH_FLUID_COLS = ["PTAU217", "NFL_PLASMA"]
- [x] `src/data/csv_dataset.py` — abeta_ptau_ratio → abeta4240_plasma_ratio (plasma-based, non-leaking)
- [x] `src/evaluation/shap_explainability.py` — FLUID_DIM=2, TOTAL_FEATURES=32, feature names updated
- [x] `configs/training/remediated_config.yaml` — embed_dim=256, dropout=0.4, wd=1e-3
- [x] `configs/training/remediated_hpo_config.yaml` — 15 trials, updated search space
- [x] `scripts/create_bh_test_split.py` — creates biohermes001_test.csv (stratified 70/15/15)
- [ ] `data/processed/biohermes/biohermes001_test.csv` exists (N≈142) (RunPod Step 2)
- [x] `scripts/evaluate.py` — computes PPV, NPV, F1, sensitivity, specificity at Youden threshold
- [x] `pytest tests/ -v` — 141/141 passing, 0 failures
- [x] Forward pass test: model params = 2,244,611 (< 5M ✓)
- [x] `git commit` + `git push`

## RunPod Training (via SSH MCP)

- [ ] `git pull` on RunPod
- [ ] `python scripts/create_bh_test_split.py` — run once
- [ ] `python scripts/train_baseline.py --config configs/training/remediated_config.yaml`
  - ADNI val AUC must reach ≥ 0.68 — if not, stop and report
- [ ] `python scripts/hpo_optuna.py --config configs/training/remediated_hpo_config.yaml`
  - 15 trials (budget constraint — ~5 hrs at 35 epochs each)
- [ ] `python scripts/train_best_model.py` — 150 epochs with best HPO params
  - ADNI val AUC ≥ 0.72 — GATE TO PROCEED
- [ ] `python scripts/finetune_biohermes.py`
  - Bio-Hermes val AUC ≥ 0.80 — GATE TO PROCEED
- [ ] `python scripts/evaluate.py` — evaluates on ADNI test AND Bio-Hermes test
  - Produces `docs/results/phase2b_results.json`
- [ ] `python scripts/run_shap.py` — updated figures
- [ ] Pull results locally: rsync docs/results/ and docs/figures/ from RunPod

## Documentation (Batch API — ~$2-3)

- [ ] `python scripts/batch/generate_phase2_docs.py --submit`
  (script must read phase2b_results.json — update RESULTS_PATH if needed)
- [ ] Batch retrieved — 5 documents updated with new metrics
- [ ] CVR_v1.0_part1.md and part2.md — no placeholder text, updated AUC values
- [ ] fairness_report.md — APOE4 gap documented with new values
- [ ] model_card.md — caveats include leakage fix explanation

## Final Gate

- [ ] `pytest tests/ -v` — 0 failures
- [ ] `PHASE2B_COMPLETE.md` written with all metrics
- [ ] `git commit -m "Phase 2B complete"` + `git push`
- [ ] STOP — human reviews before Phase 3

## Minimum Acceptable Results to Proceed to Phase 3

| Check | Minimum | Status |
|-------|---------|--------|
| ADNI test AUC | ≥ 0.65 | ⬜ pending training |
| Bio-Hermes test AUC | ≥ 0.75 | ⬜ pending training |
| Subgroup max gap | < 0.12 | ⬜ pending training |
| PPV/NPV/F1 reported | yes | ✅ implemented |
| 0 test failures | yes | ✅ 141/141 passing |

If ADNI test AUC < 0.65 after full remediation — DO NOT proceed to Phase 3.
Report to human with diagnosis.
