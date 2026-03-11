# NeuroFusion-AD Phase 2 Complete

**Date**: 2026-03-11
**Status**: COMPLETE — awaiting human gate review before Phase 3
**Environment**: RunPod RTX 3090 (root@213.192.2.120:40012)

---

## Summary

Phase 2 implemented and ran the full training, validation, and clinical documentation pipeline for NeuroFusion-AD. All code changes are committed and pushed. All five clinical documents are generated.

---

## W&B Run IDs

| Experiment | W&B Run ID | Notes |
|---|---|---|
| ADNI Baseline | `jehkd9ud` | Early stopped epoch 38, best at epoch 13 |
| Best Model (150ep) | `ybbh5fky` | Early stopped epoch 31, best at epoch 6 |
| Bio-Hermes Fine-tune | `eicxum0n` | Early stopped epoch 32, best at epoch 17 |

W&B Project: `neurofusion-ad` (entity: `vedant-muthiyan-globalcure`)

---

## HPO Best Parameters (Trial 1 of 30)

```yaml
learning_rate: 4.068e-4
weight_decay: 6.028e-5
dropout: 0.1991
gradient_accumulation_steps: 2
batch_size: 16
cls_weight: 1.940
reg_weight: 0.140
surv_weight: 0.189
graph_threshold: 0.6221
onecycle_pct_start: 0.397
```

HPO study: `neurofusion_hpo` (SQLite: `optuna_study.db`, 30/30 trials)

---

## Final Performance Metrics

### ADNI (Internal Validation) — N=100 held-out test set

| Metric | Value | 95% CI |
|---|---|---|
| Classification AUC | 0.579 | 0.452–0.671 |
| AUPRC | 0.479 | — |
| Sensitivity | 0.750 | — |
| Specificity | 0.467 | — |
| MMSE RMSE | 2.23 pts/yr | 1.93–2.49 |
| MMSE R² | -0.842 | — |
| Survival C-index | 0.509 | 0.444–0.591 |
| ECE (before calibration) | 0.200 | — |
| ECE (after temperature scaling) | 0.021 | — |
| Temperature parameter T | 3.30 | — |

**Note on ADNI val AUC = 1.0 during training**: The ADNI training/val splits had very high AUC due to the `ABETA42_CSF` feature being a near-perfect predictor of the `AMYLOID_POSITIVE` label (CSF Aβ42 < 192 pg/mL defines amyloid positivity). The held-out test set (N=100, stratified by clinical site) shows true generalization performance at AUC=0.579.

### Bio-Hermes-001 (External Validation) — N=189 val set

| Metric | Value | 95% CI (approx) |
|---|---|---|
| Classification AUC | 0.829 | 0.78–0.87 |

Note: Bio-Hermes AUC from fine-tuning validation checkpoint (epoch 17); Bio-Hermes test set not available at time of evaluation.

### Modality Importance (mean attention weights)

| Modality | Score |
|---|---|
| Motor | 0.261 |
| Acoustic | 0.248 |
| Fluid | 0.246 |
| Clinical | 0.245 |

---

## Key Bugs Fixed in Phase 2

1. **Encoder range validation blocked training** (`ValueError` at first forward pass): StandardScaler-normalized values rejected by physiological range checks. Fixed by moving range validation to InputValidator (API boundary only); encoders always use `skip_range_check=True`. (Commit: `7d3b8fb`)

2. **Cox loss AMP float16 incompatibility** (`logcumsumexp_backward not implemented for Half`): Fixed by casting tensors to float32 before `torch.logcumsumexp`. (Commit: `365b941`)

3. **Bio-Hermes `ABETA40_PLASMA` NaN in clinical tensor**: Column used in clinical tensor but excluded from imputation list. Fixed by adding to `BH_CLINICAL_BASE_COLS`. (Commit: `9138835`)

4. **SHAP OOM kill** (`construct_patient_similarity_graph` at N=10,000): KernelExplainer called model with batches of ~10,000 → O(N²) GNN adjacency matrix. Fixed by chunking `_model_wrapper` at max 64 samples. (Commit: `b7a2c6c`)

---

## Known Limitations Carried Forward to Phase 3

1. **ADNI test AUC = 0.579**: Below the 0.85 target. Root causes: (a) ABETA42_CSF overfitting on train/val — need to remove or deweight this feature; (b) ADNI N=345 is too small for a 12.7M parameter model — needs regularization or smaller architecture; (c) synthetic acoustic/motor features have no predictive signal.

2. **ADNI acoustic/motor = 100% synthetic**: These features were synthesized from clinical distributions (see DRD-001). They add noise rather than signal. For Phase 3, real acoustic/motor features from DementiaBank should be used.

3. **No CSF features in Bio-Hermes**: ABETA42_CSF, PTAU181_CSF, TAU_CSF are not available. The fluid encoder uses plasma biomarkers only, which may reduce sensitivity.

4. **APOE4 carrier subgroup AUC = 0.501** (vs non-carrier 0.690, gap 0.189): Fails the 0.07 fairness threshold. Likely confounded by the synthetic features. Phase 3 must address with real digital biomarkers and APOE4-stratified training.

5. **No PPV/NPV/F1 computed**: The evaluate.py script needs to be updated to threshold predictions and compute additional classification metrics. Currently only AUC is computed.

6. **Temperature calibration T=3.30**: Very high temperature suggests systematic overconfidence in predictions. This is expected given ABETA42_CSF domination. After removing/deweighting that feature in Phase 3, recalibration should be performed.

7. **Bio-Hermes test set not evaluated**: The `evaluate.py` script did not find a Bio-Hermes test CSV. Phase 3 should create a proper train/val/test split for Bio-Hermes.

---

## Phase 3 Recommendations (for human review)

1. **Remove or deweight ABETA42_CSF** from the fluid encoder input — or train two separate models: one for CSF-available patients, one for plasma-only.

2. **Reduce model capacity** or add stronger regularization (dropout > 0.3, L2 > 1e-3) for ADNI N=345.

3. **Replace synthetic acoustic/motor features** with real DementiaBank features.

4. **Address APOE4 fairness gap** via APOE4-stratified loss weighting.

5. **Run proper Bio-Hermes test split evaluation** — create and use a held-out Bio-Hermes test set.

---

## Files Created This Phase

### Code
- `src/training/losses.py` — MultiTaskLoss, Cox partial likelihood, augment_batch
- `src/training/trainer.py` — NeuroFusionTrainer (AMP, grad accum, OneCycleLR)
- `src/data/csv_dataset.py` — NeuroFusionCSVDataset (ADNI + Bio-Hermes)
- `src/evaluation/metrics.py` — ModelEvaluator with bootstrap CI
- `src/evaluation/calibration.py` — ECE + TemperatureScaling
- `src/evaluation/subgroup_analysis.py` — SubgroupAnalyzer
- `src/evaluation/attention_analysis.py` — AttentionAnalyzer
- `src/evaluation/shap_explainability.py` — NeuralFusionSHAPExplainer
- `configs/training/baseline_config.yaml`
- `configs/training/hpo_config.yaml`
- `configs/training/best_config.yaml` (generated by HPO)
- `configs/training/finetune_biohermes_config.yaml`
- `scripts/train_baseline.py`
- `scripts/hpo_optuna.py`
- `scripts/train_best_model.py`
- `scripts/finetune_biohermes.py`
- `scripts/evaluate.py`
- `scripts/run_shap.py`
- `scripts/batch/generate_phase2_docs.py`
- `tests/unit/test_training.py` (25 tests)
- `tests/unit/test_evaluation.py` (28 tests)

### Results
- `docs/results/phase2_results.json`
- `docs/figures/` (11 figures: roc, confusion, calibration, modality_importance, subgroup_auc, attention_heatmap, shap_summary, shap_waterfall_0/1/2)

### Clinical Documentation (DRAFT — requires human review)
- `docs/clinical/CVR_v1.0_part1.md` — Clinical Validation Report Sections 1–5
- `docs/clinical/CVR_v1.0_part2.md` — Clinical Validation Report Sections 6–11
- `docs/clinical/fairness_report.md` — Fairness and Bias Analysis Report
- `docs/clinical/model_card.md` — Model Card
- `docs/dhf/phase2/DHF_phase2.md` — Phase 2 Design History File

---

## Test Suite Final State

```
142 tests passing, 0 failures
  tests/unit/test_encoders.py     — 25 tests
  tests/unit/test_cross_modal.py  — 12 tests
  tests/unit/test_gnn.py          — 12 tests
  tests/unit/test_model.py        — 20 tests
  tests/unit/test_training.py     — 25 tests
  tests/unit/test_evaluation.py   — 28 tests
  tests/unit/test_dataset.py      — 20 tests
```

---

## STOP — Phase 3 Must Not Begin Without Human Approval

The following require human review before Phase 3:

1. Review `docs/clinical/CVR_v1.0_part1.md` and `CVR_v1.0_part2.md` — check for placeholder text, verify all metrics match `phase2_results.json`
2. Review `docs/clinical/fairness_report.md` — confirm APOE4 gap analysis is documented honestly
3. Review `docs/clinical/model_card.md` — confirm out-of-scope uses are listed
4. Review `docs/dhf/phase2/DHF_phase2.md` — confirm all training decisions are documented
5. Decide whether to proceed to Phase 3 given ADNI test AUC = 0.579 (below 0.85 target)
6. Approve Phase 3 plan (real acoustic/motor data, reduced overfitting, APOE4 fairness)
