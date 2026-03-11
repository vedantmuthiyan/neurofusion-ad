# Phase 2 Exit Checklist

**Status**: IN PROGRESS  
**Last Updated**: (agents update this)  
**Gate Review**: PENDING

---

## Training Infrastructure

- [ ] Single-GPU training config verified — `src/training/train_config.py`
- [ ] Multi-task loss functions implemented — `src/training/losses.py`
  - [ ] BCEWithLogitsLoss (classification)
  - [ ] MSELoss (regression)
  - [ ] Cox Partial Likelihood Loss (survival)
  - [ ] Combined MultiTaskLoss with configurable weights
- [ ] Training loop implemented — `src/training/trainer.py`
  - [ ] AdamW optimizer
  - [ ] CosineAnnealingLR scheduler with warmup
  - [ ] Gradient clipping (max_norm=1.0)
  - [ ] Early stopping (patience=15, monitor=val_auc)
  - [ ] W&B logging
  - [ ] Checkpoint saving every 10 epochs + best model
- [ ] Data loading verified with real ADNI data (or synthetic if data not yet available)

## Baseline Training

- [ ] Baseline training run complete — `models/checkpoints/adni_baseline/`
- [ ] W&B run logged and archived
- [ ] Results documented: Classification AUC > 0.80
- [ ] Results documented: Regression RMSE (target ≤ 3.0)
- [ ] Results documented: Survival C-index (target ≥ 0.72 at baseline)
- [ ] Learning curves generated — `notebooks/training/baseline_curves.ipynb`

## Hyperparameter Optimization

- [ ] Optuna HPO script implemented — `scripts/hpo_optuna.py`
- [ ] Search space defined (lr, batch_size, gnn_layers, attention_heads, dropout, loss_weights)
- [ ] 50 trials complete (or as many as feasible within GPU budget)
- [ ] Best hyperparameters documented — `configs/training/best_hparams.yaml`
- [ ] Best model retrained with optimal hyperparameters
- [ ] Final results: AUC ≥ 0.85 on ADNI test set

## Bio-Hermes-001 External Validation

- [ ] Bio-Hermes-001 preprocessor implemented — `src/data/biohermes_preprocessing.py`
  - [ ] Column mapping documented (Bio-Hermes fields → NeuroFusion schema)
  - [ ] All references to "Bio-Hermes-001" replaced with "Bio-Hermes-001"
- [ ] Fine-tuning run complete (transfer learning from ADNI best model)
- [ ] External validation results: AUC > 0.83 on Bio-Hermes-001 test split
- [ ] Domain shift analysis documented

## Explainability & Interpretability

- [ ] SHAP explainability pipeline — `src/evaluation/shap_explainability.py`
  - [ ] SHAP values computed for all test samples
  - [ ] Feature importance ranking generated
  - [ ] SHAP summary plot saved — `notebooks/validation/shap_summary.png`
  - [ ] SHAP beeswarm plot saved
- [ ] Attention weight analysis — `src/evaluation/attention_analysis.py`
  - [ ] Modality importance scores extracted for all test samples
  - [ ] Mean modality contributions: [fluid, acoustic, motor, clinical] bar chart
- [ ] Case studies (3 patients) — `notebooks/validation/case_studies.ipynb`
  - [ ] Case 1: High-risk patient (true positive)
  - [ ] Case 2: Low-risk patient (true negative)
  - [ ] Case 3: Uncertain prediction (near threshold)

## Fairness & Subgroup Analysis

- [ ] Subgroup analysis script — `src/evaluation/subgroup_analysis.py`
- [ ] AUC by age group (50-65, 65-75, 75-90)
- [ ] AUC by sex (male vs female)
- [ ] AUC by APOE status (e4 carrier vs non-carrier)
- [ ] AUC by education level
- [ ] AUC gap across all subgroups < 0.05 (pass/fail)
- [ ] Bias report — `docs/clinical/fairness_report.md`

## Model Calibration

- [ ] Calibration analysis — `src/evaluation/calibration.py`
- [ ] Reliability diagram (calibration curve) generated
- [ ] Expected Calibration Error (ECE) computed
- [ ] Temperature scaling applied if ECE > 0.05
- [ ] Uncertainty quantification (Monte Carlo Dropout, n=20 passes)

## Clinical Validation Report

- [ ] CVR drafted (40-60 pages) — `docs/clinical/CVR_v1.0.md`
  - [ ] Executive summary
  - [ ] Study population description (ADNI + Bio-Hermes-001)
  - [ ] Performance results table (AUC, RMSE, C-index, sensitivity, specificity)
  - [ ] Subgroup analysis results
  - [ ] Explainability analysis results
  - [ ] Calibration results
  - [ ] Case studies (3 patients)
  - [ ] Limitations section (synthetic digital biomarkers in Phase 1, real in Phase 2)
  - [ ] Conclusions
- [ ] CVR reviewed and approved for regulatory use

## Documentation & DHF

- [ ] Phase 2 DHF section compiled — `docs/dhf/phase2/`
  - [ ] Training configuration logs
  - [ ] W&B experiment references
  - [ ] Model version history
  - [ ] Decision log (architecture changes, HPO results)
- [ ] Traceability matrix updated — Phase 2 test results added
- [ ] All Bio-Hermes-001 references corrected to Bio-Hermes-001

## Quality Gates

- [ ] All Phase 2 tests passing: `pytest tests/ -v` — 0 failures
- [ ] Model card written — `docs/clinical/model_card.md`
- [ ] Final model checkpoint committed — `models/final/neurofusion_v1.0_phase2.pth`
- [ ] PHASE2_COMPLETE.md written with full performance summary

---

## Completion Instructions

When ALL items above are checked:
1. Run `pytest tests/ -v` — confirm 0 failures
2. Write `PHASE2_COMPLETE.md` with performance summary table
3. Final commit: `git add . && git commit -m "Phase 2 complete — awaiting human gate review"`
4. **STOP — do not begin Phase 3**
