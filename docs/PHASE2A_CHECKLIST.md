# Phase 2A Exit Checklist — Data Exploration, Cleaning & RunPod Setup

**Status**: IN PROGRESS  
**Last Updated**: (agents update this)

---

## 1. File Inventory (data-explorer-agent completes this first)

- [ ] `docs/data/adni_file_inventory.md` — all ADNI files listed, key files identified
- [ ] `docs/data/biohermes_file_inventory.md` — all Bio-Hermes files listed, key files identified
- [ ] `configs/data/adni_column_map.yaml` — actual column names mapped to NeuroFusion schema
- [ ] `configs/data/biohermes_column_map.yaml` — actual column names mapped

## 2. ADNI Preprocessing (data-engineer-agent)

- [ ] `src/data/adni_preprocessing.py` updated with real column names
  - [ ] Handles -1 and -4 as NaN (both codes)
  - [ ] Merges correct files (master + biomarkers + APOE + diagnosis)
  - [ ] Filters to MCI baseline correctly
  - [ ] Computes MMSE_SLOPE via linear regression per patient
  - [ ] Computes AMYLOID_POSITIVE from CSF Abeta
  - [ ] Computes TIME_TO_EVENT and EVENT_INDICATOR
  - [ ] Normalizes with StandardScaler (fit on train only)
  - [ ] Synthesizes acoustic/motor features
- [ ] `data/processed/adni/adni_train.csv` produced
- [ ] `data/processed/adni/adni_val.csv` produced
- [ ] `data/processed/adni/adni_test.csv` produced (HOLD OUT — never touch during training)
- [ ] `data/processed/adni/scaler.pkl` saved

## 3. Bio-Hermes-001 Preprocessing (data-engineer-agent)

- [ ] All "Bio-Hermes-001" references replaced with "Bio-Hermes-001" across all files
- [ ] `src/data/biohermes_preprocessing.py` updated with real column names
  - [ ] Filtered to amyloid-confirmed participants only
  - [ ] Uses ADNI-fitted scaler (does NOT refit)
  - [ ] Synthesizes acoustic/motor features
- [ ] `data/processed/biohermes/biohermes001_train.csv` produced
- [ ] `data/processed/biohermes/biohermes001_val.csv` produced

## 4. EDA Notebooks (data-explorer-agent)

- [ ] `notebooks/eda/01_adni_eda.ipynb` complete with executed outputs
- [ ] `notebooks/eda/02_biohermes_eda.ipynb` complete with executed outputs

## 5. Data Quality Report

- [ ] `docs/data/data_quality_report.md` written with:
  - [ ] Final row counts for all processed files
  - [ ] Class balance: amyloid positive % in train/val/test
  - [ ] Missing data rates after imputation
  - [ ] Any range violations found and fixed
  - [ ] Limitation: ADNI uses CSF pTau181 (not plasma pTau217)

## 6. Tests

- [ ] `pytest tests/unit/test_data.py -v` — 0 failures with real processed data

## 7. RunPod Setup (runpod-setup-agent)

- [ ] RunPod pod created (RTX 3090, Network Volume attached)
- [ ] SSH MCP connected from local Claude Code session
- [ ] Python packages installed on RunPod
- [ ] Project repo cloned to `/workspace/neurofusion-ad/`
- [ ] Processed data uploaded to RunPod
- [ ] W&B authenticated on RunPod
- [ ] Environment variables configured in `/workspace/.env`
- [ ] Final verification passed (model loads, data loads, GPU confirmed)

## 8. Completion

- [ ] `PHASE2A_COMPLETE.md` written with final statistics
- [ ] Everything committed to git
- [ ] **STOPPED — awaiting human gate review before Phase 2 training**
