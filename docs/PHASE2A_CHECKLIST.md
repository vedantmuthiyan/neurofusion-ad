# Phase 2A Exit Checklist — Data Exploration, Cleaning & RunPod Setup

**Status**: IN PROGRESS — Steps 1-6 complete, awaiting RunPod credentials for Step 7
**Last Updated**: 2026-03-11

---

## 1. File Inventory (data-explorer-agent completes this first)

- [x] `docs/data/adni_file_inventory.md` — all ADNI files listed, key files identified
- [x] `docs/data/biohermes_file_inventory.md` — all Bio-Hermes files listed, key files identified
- [x] `configs/data/adni_column_map.yaml` — actual column names mapped to NeuroFusion schema
- [x] `configs/data/biohermes_column_map.yaml` — actual column names mapped

## 2. ADNI Preprocessing (data-engineer-agent)

- [x] `src/data/adni_preprocessing.py` updated with real column names
  - [x] Handles -1 and -4 as NaN (both codes)
  - [x] Merges correct files (master + biomarkers + APOE + diagnosis)
  - [x] Filters to MCI baseline correctly
  - [x] Computes MMSE_SLOPE via linear regression per patient
  - [x] Computes AMYLOID_POSITIVE from CSF Abeta
  - [x] Computes TIME_TO_EVENT and EVENT_INDICATOR
  - [x] Normalizes with StandardScaler (fit on train only)
  - [x] Synthesizes acoustic/motor features
- [x] `data/processed/adni/adni_train.csv` produced (N=345, amyloid+ 40.3%)
- [x] `data/processed/adni/adni_val.csv` produced (N=74, amyloid+ 43.2%)
- [x] `data/processed/adni/adni_test.csv` produced (N=75, amyloid+ 38.7%) (HOLD OUT — never touch during training)
- [x] `data/processed/adni/scaler.pkl` saved

## 3. Bio-Hermes-001 Preprocessing (data-engineer-agent)

- [x] All "Bio-Hermes-002" references replaced with "Bio-Hermes-001" across all files (24 occurrences in 8 files)
- [x] `src/data/biohermes_preprocessing.py` updated with real column names
  - [x] Filtered to amyloid-confirmed participants only (945 of 1,005)
  - [x] Uses ADNI-fitted scaler (does NOT refit)
  - [x] Real acoustic/motor features from Aural Analytics and Linus Health files
- [x] `data/processed/biohermes/biohermes001_train.csv` produced (N=756, amyloid+ 35.4%)
- [x] `data/processed/biohermes/biohermes001_val.csv` produced (N=189, amyloid+ 39.2%)

## 4. EDA Notebooks (data-explorer-agent)

- [x] `notebooks/eda/01_adni_eda.ipynb` complete with executed outputs (11 cells, 6 figures)
- [x] `notebooks/eda/02_biohermes_eda.ipynb` complete with executed outputs (11 cells, 6 figures)

## 5. Data Quality Report

- [x] `docs/data/data_quality_report.md` written with:
  - [x] Final row counts for all processed files
  - [x] Class balance: amyloid positive % in train/val/test
  - [x] Missing data rates after imputation
  - [x] Any range violations found and fixed (PTDEMOG screening visit bug documented)
  - [x] Limitation: ADNI uses CSF pTau181 (not plasma pTau217); assay batch effects documented

## 6. Tests

- [x] `pytest tests/unit/test_data.py -v` — 0 failures with real processed data (23 tests passing)

## 7. RunPod Setup (runpod-setup-agent)

- [ ] RunPod pod created (RTX 3090, Network Volume attached)
- [ ] SSH MCP connected from local Claude Code session
- [ ] Python packages installed on RunPod
- [ ] Project repo cloned to `/workspace/neurofusion-ad/`
- [ ] Processed data uploaded to RunPod
- [ ] W&B authenticated on RunPod
- [ ] Environment variables configured in `/workspace/.env`
- [ ] Final verification passed (model loads, data loads, GPU confirmed)

**BLOCKED**: Awaiting user to provide RunPod credentials (SSH IP/port, password, W&B API key, GitHub repo URL)

## 8. Completion

- [ ] `PHASE2A_COMPLETE.md` written with final statistics
- [ ] Everything committed to git
- [ ] **STOPPED — awaiting human gate review before Phase 2 training**
