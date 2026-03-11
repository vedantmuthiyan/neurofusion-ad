# Phase 2A Complete — Data Exploration, Cleaning & RunPod Setup

**Date**: 2026-03-11
**Status**: COMPLETE — awaiting human gate review before Phase 2 training

---

## Summary

Phase 2A is complete. Both datasets have been preprocessed, validated, and uploaded to RunPod.
The NeuroFusion-AD model passes end-to-end forward pass on the RTX 3090.

---

## Dataset Statistics

### ADNI

| Split | N   | Amyloid+ (of all) | Amyloid+ (of labeled) | Amyloid+ label coverage |
|-------|-----|-------------------|-----------------------|-------------------------|
| Train | 345 | 40.3% (139/345)   | 62.6% (139/222)       | 64.4% (222/345)         |
| Val   |  74 | 43.2%  (32/74)    | 65.3%  (32/49)        | 66.2%  (49/74)          |
| Test  |  75 | 38.7%  (29/75)    | 65.9%  (29/44)        | 58.7%  (44/75)          |
| Total | 494 | 40.5% (200/494)   | 63.5% (200/315)       | 63.8% (315/494)         |

Note on amyloid+ rates: The "of all" rate counts NaN labels as negative (full denominator).
The "of labeled" rate is what pandas `.mean()` returns (NaN-ignored). Both are correct;
the relevant figure for classification training is the "of labeled" rate among patients
with valid CSF Abeta42 data.

### Bio-Hermes-001

| Split | N   | Amyloid+ Rate |
|-------|-----|---------------|
| Train | 756 | 35.4% (268/756) |
| Val   | 189 | 39.2%  (74/189) |
| Total | 945 | 36.2% (342/945) |

Note: 60 of 1,005 enrolled participants excluded (missing AMYCLAS amyloid label).

---

## RunPod Environment

| Item | Value |
|------|-------|
| Host | root@213.192.2.120:40012 |
| GPU | NVIDIA GeForce RTX 3090 (24GB VRAM) |
| CUDA | 12.1 |
| Python | 3.10.12 |
| PyTorch | 2.1.2+cu121 |
| PyTorch Geometric | 2.5.0 |
| NumPy | 1.24.1 |
| W&B | 0.25.1 (authenticated) |
| Repo path | /workspace/neurofusion-ad/ |
| Data path | /workspace/neurofusion-ad/data/processed/ |
| Config | /workspace/.env |

---

## Key Data Issues Found and Resolved

1. **Bio-Hermes-002 references** — 24 occurrences in 8 files replaced with "Bio-Hermes-001".
   Bio-Hermes-002 does not exist; the clinical trial closes ~2028.

2. **PTDEMOG demographics bug** — `VISCODE=='bl'` filter returned only 8 rows.
   Fixed to prefer `VISCODE=='sc'` (screening visit). After fix: age_null=0.0%, sex_null=0.0%.

3. **FTACPTFL quality flag** — entirely NaN in Bio-Hermes-001 data export. Preprocessing
   falls back to including all acoustic/motor records. Flagged for data provider follow-up.

4. **Scaler version mismatch** — scaler.pkl created with sklearn 1.2.2; RunPod has 1.7.2.
   Works correctly (StandardScaler API stable), generates a UserWarning only.
   Will be regenerated in Phase 2 if needed.

---

## Known Issues for Phase 2 Training

1. **Assay batch effects**: ADNI uses Fujirebio/Quanterix assays; Bio-Hermes-001 uses
   Lilly (pTau217) and Roche (NfL, GFAP, Abeta). Domain adaptation required before
   joint model training.

2. **Bio-Hermes-001 acoustic/motor unscaled**: Only 7 features normalized with ADNI scaler.
   Acoustic and motor features need per-modality standardization during Phase 2.

3. **ADNI amyloid label coverage 63.8%**: 179 of 494 MCI patients lack CSF amyloid data.
   Consider using plasma pTau217 ratio as secondary label source.

4. **ADNI acoustic/motor synthesized**: All 20 features (12 acoustic + 8 motor) for ADNI
   are synthesized from clinical distributions (see DRD-001). Not real measurements.

5. **Bio-Hermes-001 cross-sectional only**: No MMSE_SLOPE, TIME_TO_EVENT, or EVENT_INDICATOR.
   Bio-Hermes-001 used only for amyloid classification and cross-sectional feature learning.

---

## Files Produced

| File | Description |
|------|-------------|
| `data/processed/adni/adni_train.csv` | ADNI train split (N=345) |
| `data/processed/adni/adni_val.csv` | ADNI validation split (N=74) |
| `data/processed/adni/adni_test.csv` | ADNI test split (N=75) — HOLD OUT |
| `data/processed/adni/scaler.pkl` | StandardScaler fitted on train only |
| `data/processed/biohermes/biohermes001_train.csv` | Bio-Hermes-001 train (N=756) |
| `data/processed/biohermes/biohermes001_val.csv` | Bio-Hermes-001 val (N=189) |
| `docs/data/adni_file_inventory.md` | Full ADNI raw file inventory |
| `docs/data/biohermes_file_inventory.md` | Full Bio-Hermes-001 raw file inventory |
| `docs/data/data_quality_report.md` | Data quality report with all statistics |
| `configs/data/adni_column_map.yaml` | Real ADNI column name mappings |
| `configs/data/biohermes_column_map.yaml` | Real Bio-Hermes-001 column name mappings |
| `notebooks/eda/01_adni_eda.ipynb` | ADNI EDA notebook (executed, 6 figures) |
| `notebooks/eda/02_biohermes_eda.ipynb` | Bio-Hermes-001 EDA notebook (executed, 6 figures) |
| `src/data/adni_preprocessing.py` | ADNI preprocessing pipeline v2 (real columns) |
| `src/data/biohermes_preprocessing.py` | Bio-Hermes-001 preprocessing pipeline v1 |

---

## Test Status

```
pytest tests/unit/test_data.py -v    — 23 tests PASSED
pytest tests/unit/ -v                — 89 tests PASSED (all modules)
scripts/sanity_check_e2e.py          — PASSED on RunPod RTX 3090
```

---

## Gate: STOP HERE

**DO NOT BEGIN PHASE 2 TRAINING WITHOUT HUMAN APPROVAL.**

Human reviewer must confirm:
1. Data quality report reviewed and accepted
2. Known issues understood and mitigation plan approved
3. Training configuration (learning rates, batch sizes, loss weights) reviewed
4. W&B project and entity confirmed
5. ADNI test split held out (confirmed not used during development)
