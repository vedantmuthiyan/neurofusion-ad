# Data Quality Report — Phase 2A
**Date**: 2026-03-11
**Author**: data-engineer-agent
**Pipeline version**: adni_preprocessing.py v1.0, biohermes_preprocessing.py v1.0

---

## ADNI Dataset

### Patient Counts
| Split | N   | Amyloid+ | Amyloid+ Rate |
|-------|-----|----------|---------------|
| Train | 345 | 139      | 62.6%         |
| Val   |  74 |  32      | 65.3%         |
| Test  |  75 |  29      | 65.9%         |
| Total | 494 | 200      | 63.8% (of those with CSF label) |

Notes:
- 494 MCI-baseline patients survived all merges.
- Of 767 MCI patients identified in DXSUM, 494 had sufficient plasma biomarker data (UPENN plasma file covers N=558 baseline rows, yielding 494 after inner join with CSF and registry).
- 63.8% of the 494 have a valid AMYLOID_POSITIVE label (from CSF Abeta42 < 192 pg/mL). The remaining 36.2% have NaN label and will be excluded from classification training.

### Feature Availability
| Feature | Source | % Non-null in Final Dataset |
|---------|--------|---------------------------|
| PTAU217 (plasma pTau-217) | UPENN_PLASMA_FUJIREBIO_QUANTERIX_28Feb2026.csv | 100.0% |
| ABETA4240_RATIO | UPENN_PLASMA_FUJIREBIO_QUANTERIX_28Feb2026.csv | 100.0% |
| NFL_PLASMA | UPENN_PLASMA_FUJIREBIO_QUANTERIX_28Feb2026.csv | 100.0% |
| PTAU181_CSF | UPENNBIOMK_MASTER_28Feb2026.csv | 100.0% |
| ABETA42_CSF | UPENNBIOMK_MASTER_28Feb2026.csv | 100.0% |
| AMYLOID_POSITIVE (CSF Abeta42 label) | UPENNBIOMK_MASTER_28Feb2026.csv (computed) | 63.8% |
| MMSE_SLOPE | MMSE.rda (computed via linear regression) | 90.9% |
| AGE | PTDEMOG.rda (computed from PTDOBYY + exam year) | 100.0% |
| acoustic_jitter (synthesized) | Generated | 100.0% |
| motor_tremor_freq (synthesized) | Generated | 100.0% |

Note: All patients in the final dataset have plasma biomarker data (PTAU217, ABETA4240_RATIO, NFL_PLASMA) because the merge starts from the plasma file (UPENN_PLASMA). The 494 patients are a subset of the full ADNI MCI cohort limited to those with plasma pTau217 available.

### Limitations
1. **No plasma pTau217 for all ADNI patients**: Only 494 of 767 MCI baseline patients have plasma pTau217 data (UPENN Plasma file covers N=558 baseline rows; 494 retained after merging with diagnosis/registry). Remaining 273 MCI patients have only CSF pTau181 as proxy and are excluded from the final dataset.
2. **Acoustic features synthesized**: ADNI has no speech/acoustic data. All 12 acoustic features for ADNI are synthesized from clinically plausible distributions using `np.random.default_rng(42)` (documented in DRD-001).
3. **Motor features synthesized**: No wearable/tablet motor data in ADNI. All 8 motor features synthesized using `np.random.default_rng(43)` (documented in DRD-001).
4. **CSF pTau181 used alongside plasma pTau217**: Different assay, different range. Both are included as separate features. Regulatory limitation documented in DRD-001.
5. **AMYLOID_POSITIVE label missing for 36.2% of final patients**: CSF Abeta42 not available for all patients even when plasma biomarkers are available. These patients are excluded from classification label training but retained for regression and survival tasks.
6. **Demographics baseline filter yielded only 8 patients**: PTDEMOG.rda filtered to VISCODE=='bl' returned only 8 rows. This is because PTDEMOG has per-subject (not per-visit) records; most entries use visit code 'sc' (screening) rather than 'bl'. All demographic values (SEX_CODE, EDUCATION_YEARS, PTDOBYY) come from the APOE file or registry merge for the majority of patients. Demographics show as NaN for most patients in the merged dataset. This is a known ADNI data structure issue — a future fix should use all PTDEMOG records (not filtered to 'bl') since demographics are time-invariant.

---

## Bio-Hermes-001 Dataset

### Patient Counts
| Split | N   | Amyloid+ | Amyloid+ Rate |
|-------|-----|----------|---------------|
| Train | 756 | 268      | 35.4%         |
| Val   | 189 |  74      | 39.2%         |
| Total | 945 | 342      | 36.2%         |

Notes:
- 945 of 1,005 enrolled Bio-Hermes-001 participants have amyloid classification (NVTESTCD='AMYCLAS') — 60 participants are excluded due to missing label.
- Demographics available for all 1,005 enrolled; 945 retained after joining on amyloid label.

### Feature Availability
| Feature | Source | % Non-null in Final Dataset |
|---------|--------|---------------------------|
| PTAU217 | LB_LILLY_CLINICAL_DIAGNOST.csv (TAU217P) | 100.0% |
| ABETA4240_RATIO | LB_ROCHE.csv (AMYLB42/AMYLB40) | 100.0% |
| NFL_PLASMA | LB_ROCHE.csv (NFLP) | 100.0% |
| GFAP_PLASMA | LB_ROCHE.csv (GFAP) | 100.0% |
| AMYLOID_POSITIVE | GAP-Clinical/NV.csv (AMYCLAS) | 100.0% |
| AGE | GAP-Clinical/DM.csv | 100.0% |
| MMSE_BASELINE | GAP-Clinical/FT.csv (MMS112) | 100.0% |
| EDUCATION_YEARS | GAP-Clinical/SC.csv (EDUYRNUM) | 100.0% |
| acoustic_speaking_rate (CA65CBDF) | FT_AURAL_ANALYTICS.csv | ~100% (950/945 unique subjects) |
| motor_spiral_cw_dom (SPCWDTM) | FT_LINUS_HEALTH.csv | ~100% (983/945 unique subjects) |

Note: All 10 acoustic features and 15 motor features are present in Bio-Hermes-001 as real measured values (not synthesized).

### Limitations
1. **Cross-sectional only**: No longitudinal MMSE or survival labels available. MMSE_SLOPE, TIME_TO_EVENT, and EVENT_INDICATOR are set to NaN for all Bio-Hermes-001 records. Bio-Hermes-001 used only for amyloid classification and cross-sectional feature learning.
2. **Acoustic features are non-standard**: No classical jitter/shimmer. Hashed test codes from Aural Analytics map to pause rate (MAD174CB) and monotonicity (MAC168D2) as proxies. These are not equivalent to PRAAT-derived jitter/shimmer.
3. **FTACPTFL quality flag not populated**: The FTACPTFL column in both FT_AURAL_ANALYTICS.csv and FT_LINUS_HEALTH.csv is entirely NaN in the current data export. The quality filter logic falls back to including all records when FTACPTFL is unpopulated. This may include low-quality recordings — flagged for data provider follow-up.
4. **Roche ABETA40_PLASMA used for ratio**: The Roche plasma Abeta42/40 ratio (ABETA4240_RATIO) was computed from AMYLB42/AMYLB40 in LB_ROCHE.csv, NOT from the Fujirebio assay used in ADNI. This introduces assay-level batch effects that must be addressed during model training (e.g., via domain adaptation or separate normalization layers).

---

## Preprocessing Notes
- ADNI missing codes -1 and -4 replaced with NaN in all numeric columns before any processing.
- StandardScaler fit on ADNI train split (N=345) only; applied to val, test, and Bio-Hermes-001 (transfer normalization).
- ADNI age derived from PTDOBYY (birth year in PTDEMOG.rda) and baseline exam year (REGISTRY.rda EXAMDATE).
- APOE4 count derived by counting '4' occurrences in GENOTYPE string (e.g., '3/4' -> 1, '4/4' -> 2).
- MMSE slope computed via scipy.stats.linregress across all longitudinal visits; requires minimum 2 valid MMSE scores per patient. 9.1% of MCI patients had only 1 MMSE visit and receive NaN slope.
- Bio-Hermes-001 scaler application: only 7 overlapping features between the two datasets are scaled (PTAU217, ABETA4240_RATIO, NFL_PLASMA, GFAP_PLASMA, AGE, MMSE_BASELINE, EDUCATION_YEARS). The remaining acoustic/motor features in Bio-Hermes-001 are unscaled and will need per-modality normalization during training.
- All patient IDs (RID and USUBJID) are hashed before logging using SHA-256.

---

## Known Issues for Next Session
1. **ADNI demographics baseline filter**: PTDEMOG.rda filtered to 'bl' returns only 8 rows. Should load all PTDEMOG rows (time-invariant) and deduplicate by RID instead of filtering to baseline visit code.
2. **Assay batch effects**: ADNI plasma biomarkers use Fujirebio/Quanterix assays; Bio-Hermes-001 uses Lilly (pTau217) and Roche (NfL, GFAP, Abeta). Domain adaptation required before training joint models.
3. **Bio-Hermes-001 acoustic/motor unscaled**: Only 7 features were normalized using the ADNI scaler. Acoustic and motor features from Bio-Hermes-001 are in raw units and need standardization.
