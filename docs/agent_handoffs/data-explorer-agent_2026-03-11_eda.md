# data-explorer-agent Handoff — 2026-03-11 18:45

## Completed This Session

- `notebooks/eda/01_adni_eda.ipynb` — ADNI EDA notebook, 11 cells, **fully executed**, 6 figures saved
- `notebooks/eda/02_biohermes_eda.ipynb` — Bio-Hermes-001 EDA notebook, 11 cells, **fully executed**, 6 figures saved
- `scripts/batch/create_eda_notebooks.py` — Builder/executor script (nbformat + nbconvert)
- 12 PNG figures saved to `notebooks/eda/` (see list below)

### Figures generated

**ADNI:**
- `fig_adni_class_balance.png` — Amyloid+ bar charts by split (Train/Val/Test)
- `fig_adni_demographics.png` — Age, sex, education by amyloid status
- `fig_adni_fluid_biomarkers.png` — pTau-217, Abeta42/40, NfL histograms
- `fig_adni_mmse_slope.png` — MMSE decline rate distributions
- `fig_adni_survival.png` — Time-to-event distributions by amyloid status
- `fig_adni_correlation.png` — Lower-triangle correlation heatmap (8 key features)

**Bio-Hermes-001:**
- `fig_biohermes_class_balance.png` — Amyloid+ bar + sex breakdown
- `fig_biohermes_ptau_abeta.png` — pTau-217 and Abeta42/40 histograms
- `fig_biohermes_acoustic.png` — All 10 acoustic features (2x5 grid)
- `fig_biohermes_motor.png` — All 15 motor/cognitive features (3x5 grid)
- `fig_biohermes_race.png` — Race/ethnicity fairness analysis
- `fig_biohermes_correlation.png` — Fluid + acoustic correlation heatmap

## Key Findings (from notebook outputs)

### ADNI
- **494 MCI patients** (345 train / 74 val / 75 test), amyloid+ rate: **63.5%** (higher than context stated — confirm with data engineer)
- **177 MCI→Dementia conversions (35.8%)** over follow-up
- Median time to conversion: **25.1 months**
- Mean MMSE slope: **-1.295 pts/yr** (amyloid+) vs **-0.046 pts/yr** (amyloid-) — large clinical effect
- APOE4 allele count strongly associated with amyloid positivity (mean 0.78 vs 0.22)
- Acoustic and motor features are **SYNTHESIZED** from literature — not real biomarker signals for ADNI

### Bio-Hermes-001
- **945 participants**, amyloid+ rate: **36.2%** — consistent with context description
- Age: 72.0 ± 6.7 years
- pTau-217 mean (amyloid+): **0.909** vs (amyloid-): **-0.001** (z-scaled) — strong biomarker signal
- **10 acoustic features** (Aural Analytics — REAL), **15 motor features** (Linus Health — REAL)
- Race distribution: 85% White, 11% Black/African American, ~1% Asian — flag for fairness analysis in RMF-001
- **No longitudinal labels** (MMSE_SLOPE = NaN, TIME_TO_EVENT = NaN) — cross-sectional study

## Decisions Made

- Used `matplotlib.use('Agg')` non-interactive backend in notebook cells to ensure kernel execution without display
- Used absolute paths for CSV reads (injected at notebook build time) to avoid working-directory issues with nbconvert
- Used `.values.tolist()` for all histogram data per NumPy 2.x ABI requirement
- Saved all figures to `notebooks/eda/` as PNGs alongside notebooks

## Current State

- Working: Both notebooks execute cleanly end-to-end, all 12 figures saved
- Blocked: None
- Note: ADNI amyloid+ rate is 63.5% (not 40.3% as previously stated) — possible discrepancy in preprocessing thresholds, should be reviewed

## Next Session Must Start With

1. Review ADNI amyloid+ rate discrepancy (63.5% observed vs 40.3% stated) — check threshold in `src/data/adni_preprocessing.py`
2. Confirm Bio-Hermes-001 race distribution is documented in RMF-001 fairness section
3. Consider adding feature importance plots (SHAP preview on logistic regression) as notebook 03

## Open Questions for Human Review

- ADNI amyloid+ rate discrepancy: context says 40.3% but observed value is 63.5%. Is the amyloid threshold calibrated differently than expected?
- Should the synthesized ADNI acoustic/motor features be flagged with a stronger warning in the DHF?
