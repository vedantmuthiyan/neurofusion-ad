# training-agent Handoff — 2026-03-11

## Completed This Session

### Files Created

| File | Description |
|------|-------------|
| `src/training/__init__.py` | Package init for training module |
| `src/training/losses.py` | MultiTaskLoss, cox_partial_likelihood_loss, augment_batch |
| `src/training/trainer.py` | NeuroFusionTrainer with AMP, grad accum, OneCycleLR, W&B, early stopping |
| `src/data/csv_dataset.py` | NeuroFusionCSVDataset: ADNI + Bio-Hermes CSV loader with imputation |
| `configs/training/baseline_config.yaml` | ADNI baseline training config (N=345, 100 epochs) |
| `configs/training/hpo_config.yaml` | Optuna HPO config (30 trials, MedianPruner, SQLite) |
| `configs/training/finetune_biohermes_config.yaml` | BH fine-tune config (frozen encoders, cls only) |
| `scripts/train_baseline.py` | ADNI baseline training script |
| `scripts/hpo_optuna.py` | Hyperparameter optimization with Optuna |
| `scripts/train_best_model.py` | Full 150-epoch retrain with best HPO config |
| `scripts/finetune_biohermes.py` | Bio-Hermes fine-tuning with frozen encoders |
| `tests/unit/test_training.py` | 25 unit tests (all passing) |

### Test Results

```
25 passed in 29.91s (tests/unit/test_training.py)
23 passed (tests/unit/test_data.py — no regressions)
```

---

## Decisions Made (with rationale)

### 1. `torch.logcumsumexp` for Cox loss (not manual loop)
PyTorch 2.1.2 provides `torch.logcumsumexp` which is numerically stable and
differentiable. This avoids the overflow risk of `torch.exp(log_h).cumsum()` for
large log-hazard values. The function sorts by time descending before cumsum so
that each patient's risk set sum includes all patients with equal or longer survival.

### 2. `torch.tensor(arr.tolist())` instead of `torch.from_numpy(arr)` in csv_dataset.py
The local environment has NumPy 2.4.1 and PyTorch 2.1.2, which have a C-level ABI
incompatibility. All tensor construction from Python lists or `.tolist()` calls
avoids `torch.from_numpy()` to prevent segfaults.

### 3. Minimal model in trainer tests (not NeuroFusionAD)
`NeuroFusionAD` requires `torch-geometric` which may not be installed in all CI
environments. A `_MinimalModel` class that mimics the exact forward() signature
is used in trainer tests to ensure the tests are self-contained and fast.

### 4. Cox loss returns zero (not NaN/error) for no-event batches
When a batch has no events, `cox_partial_likelihood_loss` returns a zero tensor
with `requires_grad=True`. This avoids training crashes during warm-up epochs
when survival labels may be all-NaN (Bio-Hermes) or when random batches have no
events. The trainer continues without gradient contribution from survival.

### 5. AMP only enabled on CUDA (not on CPU/MPS)
`torch.cuda.amp.GradScaler` and `autocast` are only created when `device == 'cuda'`.
CPU and MPS devices run in full float32 precision. This is safe and avoids
compatibility issues across environments.

### 6. `fit_imputation=True` on train split, pass stats to val/test
Imputation medians are always computed on the train CSV and passed to val/test
constructors as `imputation_stats` dict. This prevents information leakage from
val/test into training statistics.

### 7. Bio-Hermes acoustic padded to 12 with zeros
Bio-Hermes-001 has 10 acoustic features (Aural Analytics) vs ADNI's 12 (synthetic
jitter/shimmer-like). Positions 10 and 11 are zero-padded. The acoustic encoder
(trained on ADNI synthetic data) will learn to ignore these zero positions. When
real jitter/shimmer data becomes available, the padding zeros can be replaced.

---

## Architecture Implemented

### `MultiTaskLoss` (src/training/losses.py)
- `BCEWithLogitsLoss(reduction='none')` for cls — NaN-masked per sample
- `MSELoss(reduction='none')` for reg — NaN-masked per sample
- `cox_partial_likelihood_loss()` for surv — NaN-masked via valid_mask inside function
- `loss_weights = {'cls', 'reg', 'surv'}` — weight=0.0 skips computation entirely
- `augment_batch()` — adds Gaussian noise to feature keys only, preserves labels

### `NeuroFusionTrainer` (src/training/trainer.py)
- `AdamW` optimizer (lr=3e-4, wd=1e-4 default)
- `OneCycleLR` scheduler (pct_start=0.3, stepped per optimizer step)
- Gradient accumulation: default 4 steps, clips grad norm to 1.0
- AMP: `GradScaler` + `autocast` on CUDA only
- `evaluate()`: collects preds/targets, calls `compute_metrics()` for AUC/RMSE/C-index
- `fit()`: early stopping by val_auc (patience from config), W&B logging per epoch
- Checkpoints: best_model.pth (best AUC) + periodic checkpoint_epoch_NNNN.pth

### `NeuroFusionCSVDataset` (src/data/csv_dataset.py)
- `mode='adni'`: 41-column ADNI CSV, computes 2 derived clinical features
- `mode='biohermes'`: 30-column BH CSV, pads acoustic 10→12, maps motor to 8 cols
- `fit_imputation=True`: computes column medians from this CSV (train only)
- `fit_biohermes_scaler()`: classmethod, fits StandardScaler on BH train acoustic+motor,
  saves to `data/processed/biohermes/biohermes_digital_scaler.pkl`
- PHI safe: all patient IDs (RID / USUBJID) are SHA-256 hashed immediately

---

## Current State

### Working
- All 25 unit tests pass: `pytest tests/unit/test_training.py -v`
- All 23 pre-existing data tests still pass (no regressions)
- MultiTaskLoss handles all NaN edge cases correctly
- NeuroFusionCSVDataset handles ADNI and Bio-Hermes column mappings
- Trainer runs on CPU with grad accumulation

### Not Yet Run (awaiting real data)
- `scripts/train_baseline.py` — requires `data/processed/adni/adni_train.csv`
- `scripts/hpo_optuna.py` — requires ADNI data + Optuna installed
- `scripts/finetune_biohermes.py` — requires BH data + trained ADNI checkpoint

### Known Gaps
1. **Bio-Hermes digital scaler not yet fit**: `biohermes_digital_scaler.pkl` will be
   created when `scripts/train_baseline.py` is first run (it calls `fit_biohermes_scaler()`
   at startup). The scaler file is absent until then.
2. **W&B entity not set**: The `wandb.entity` field is empty in all configs. Set via
   `WANDB_ENTITY` environment variable or edit configs before running.
3. **ADNI sentinel value replacement**: `csv_dataset.py` replaces -1 and -4 with NaN
   (as documented by data-explorer-agent). However, some ADNI columns may use -2
   as a sentinel for "not assessed" — verify with data-engineer-agent.

---

## Next Session Must Start With

1. **Run full test suite** to confirm environment is healthy:
   ```
   pytest tests/ -v
   ```

2. **Verify ADNI CSVs exist** before running training:
   ```
   ls data/processed/adni/
   ls data/processed/biohermes/
   ```

3. **Run baseline training** (if ADNI data is present):
   ```
   python scripts/train_baseline.py --config configs/training/baseline_config.yaml
   ```
   Expected: val_auc >= 0.74 within 100 epochs. If AUC below threshold, investigate
   ADNI data quality (check data-engineer-agent handoff for known demographic NaN issues).

4. **If baseline AUC < 0.74**: Check that `adni_train.csv` has:
   - AMYLOID_POSITIVE non-null rate > 60% (data-engineer reports 63.8%)
   - SEX_CODE and EDUCATION_YEARS non-null (data-engineer notes potential all-NaN issue)

5. **Run HPO** (optional, after baseline confirms training pipeline works):
   ```
   python scripts/hpo_optuna.py --config configs/training/hpo_config.yaml
   ```

---

## Open Questions for Human Review

1. **ADNI -2 sentinel**: The data-explorer-agent noted -1 and -4 as ADNI sentinels.
   Should -2 also be treated as NaN? `csv_dataset.py` currently only replaces -1 and -4.

2. **Bio-Hermes acoustic scaler padding**: The BH digital scaler is fit on [10 acoustic
   cols + 8 motor cols] = 18 features, but the padded zero columns (positions 10, 11)
   are included in the 20-feature scaler. Should the scaler be fit on the 10 real
   acoustic features only (ignoring the padding zeros)?

3. **ADNI MMSE_SLOPE column name**: The task spec says ADNI has `MMSE_SLOPE` (7.8% NaN).
   Verify the exact column name in the processed CSV output from `adni_preprocessing.py`.
   If the column is named differently, update `ADNI_CLINICAL_BASE_COLS` reference in
   `csv_dataset.py`.

4. **W&B offline mode**: For air-gapped training environments (HIPAA), W&B can be run
   in offline mode via `WANDB_MODE=offline`. Document this in the training README.
