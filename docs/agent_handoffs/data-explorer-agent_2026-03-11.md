# data-explorer-agent Handoff — 2026-03-11

## Completed This Session

- `docs/data/adni_file_inventory.md` — Full inventory of 12 ADNI root CSVs + 8 ADNIMERGE2 .rda files with verified column names, row counts, file sizes, and relevance notes
- `docs/data/biohermes_file_inventory.md` — Full inventory of Bio-Hermes-001 files across all 5 subdirectories; includes SDTM structure explanation, unique LBTESTCD/FTTESTCD/NVTESTCD codes discovered per file
- `configs/data/adni_column_map.yaml` — Verified ADNI column mapping with actual column names; all placeholders filled
- `configs/data/biohermes_column_map.yaml` — Verified Bio-Hermes-001 column mapping with SDTM filter logic for all key analytes

---

## Decisions Made (with rationale)

- **Primary pTau217 source for ADNI**: `UPENN_PLASMA_FUJIREBIO_QUANTERIX_28Feb2026.csv`, column `pT217_F`. This is the only ADNI file with plasma pTau-217. N=2,178 rows (subset of full cohort). The UPENNBIOMK_MASTER file has CSF pTau181 (`PTAU`), not pTau217.
- **Primary pTau217 source for Bio-Hermes-001**: `LB_LILLY_CLINICAL_DIAGNOST.csv`, filter `LBTESTCD == 'TAU217P'`, use `LBSTRESN`. University of Gothenburg file is secondary pTau217 source (same test code, different assay vendor).
- **Amyloid positivity label for ADNI**: `UPENNBIOMK_MASTER_28Feb2026.csv`, column `ABETA < 192 pg/mL` → amyloid positive (UPENN immunoassay standard cutoff).
- **Amyloid positivity label for Bio-Hermes-001**: `GAP-Clinical/NV.csv`, filter `NVTESTCD == 'AMYCLAS'`, value `NVSTRESC == 'POSITIVE'`.
- **ADNI DXSUM DIAGNOSIS has 3 values only**: 'MCI', 'CN', 'Dementia' — NOT 'EMCI' or 'LMCI'. The CLAUDE.md references EMCI/LMCI but these are not present in DXSUM. Filter on 'MCI' only.
- **ADNI demographics**: Use `PTDEMOG.rda` for age (`PTDOBYY`, derive age from exam year), sex (`PTGENDER`), education (`PTEDUCAT`). No direct `AGE` column — must compute from `PTDOBYY`.
- **Bio-Hermes-001 is cross-sectional**: Only amyloid positivity classification label applies. No longitudinal MMSE slope or survival labels.
- **Acoustic features**: Bio-Hermes-001 uses non-standard hashed FTTESTCD codes (e.g., CA65CBDF). There are NO classical jitter/shimmer values. The closest proxies are pause rate (MAD174CB) for jitter and monotonicity (MAC168D2) for shimmer.
- **ADNI has no acoustic data**: No jitter/shimmer/speech data exists in any ADNI CSV. Acoustic encoder will be trained exclusively on Bio-Hermes-001.
- **Motor features**: Linus Health spiral drawing (`SPCWDTM`, `SPCCWDTM`, etc.) maps to the NeuroFusion motor encoder.

---

## Current State

### Working (verified)
- ADNI CSF biomarker files readable and column-verified
- ADNI plasma biomarker file with pTau217 (pT217_F) readable
- ADNI APOE genotype file readable (GENOTYPE column confirmed)
- All 8 ADNIMERGE2 .rda files readable via pyreadr
- MMSE.rda: MMSCORE column confirmed (N=14,599 rows, 161 NaN)
- DXSUM.rda: DIAGNOSIS column confirmed (MCI/CN/Dementia)
- PTDEMOG.rda: PTGENDER, PTDOBYY, PTEDUCAT confirmed
- Bio-Hermes-001 DM.csv readable (1,005 subjects, AGE/SEX confirmed)
- Bio-Hermes-001 LB_LILLY pTau217 file readable (TAU217P confirmed)
- Bio-Hermes-001 NV.csv amyloid classification readable (AMYCLAS confirmed)
- Bio-Hermes-001 FT_AURAL_ANALYTICS readable (12 acoustic test codes)
- Bio-Hermes-001 FT_LINUS_HEALTH readable (21 motor/cognitive test codes)

### Blocked
- None — all files were successfully read

---

## Key Findings

### ADNI
- **Primary CSF biomarker file**: UPENNBIOMK_MASTER_28Feb2026.csv (5,876 rows)
  - Key cols: `RID`, `VISCODE`, `PTAU` (CSF pTau181), `ABETA` (CSF Abeta42), `TAU`
  - Missing: no pTau217; no Abeta40 (no ratio possible)
- **Primary plasma biomarker file**: UPENN_PLASMA_FUJIREBIO_QUANTERIX_28Feb2026.csv (2,178 rows)
  - Key cols: `RID`, `VISCODE`, `pT217_F` (pTau217), `AB42_F`, `AB40_F`, `AB42_AB40_F`, `NfL_Q`, `GFAP_Q`
  - **This is the preferred fluid encoder source** — has pTau217, NfL, GFAP, Abeta ratio
- **APOE file**: APOERES_28Feb2026.csv (3,253 rows), genotype as string (e.g., "3/4")
- **Demographics**: PTDEMOG.rda (6,219 rows), key: PTGENDER, PTDOBYY, PTEDUCAT
- **MMSE longitudinal**: MMSE.rda (14,599 rows), key: MMSCORE (161 NaN)
- **Diagnosis progression**: DXSUM.rda (15,881 rows), DIAGNOSIS = 'MCI'|'CN'|'Dementia'
- **ADNIMERGE2 .rda files** confirmed readable via pyreadr (RuntimeWarning about datetime cast is cosmetic, not an error)

### Bio-Hermes-001
- **pTau-217 source**: `LB_LILLY_CLINICAL_DIAGNOST.csv` — filter `LBTESTCD == 'TAU217P'`, value in `LBSTRESN`
- **Amyloid ground truth**: `GAP-Clinical/NV.csv` — filter `NVTESTCD == 'AMYCLAS'`, label in `NVSTRESC` ('POSITIVE'/'NEGATIVE')
- **Patient ID column**: `USUBJID` (format: "BIO-HERMES-00101-001"), 1,005 unique subjects
- **Acoustic features**: 12 features in FT_AURAL_ANALYTICS.csv (hashed FTTESTCD codes; use FTACPTFL='Y' quality filter)
- **Motor features**: 21 features in FT_LINUS_HEALTH.csv (spiral drawing, reaction time, SDMT, Trails B)
- **Education**: `GAP-Clinical/SC.csv` — filter `SCTESTCD == 'EDUYRNUM'`, value in `SCSTRESC`
- **Demographics**: `GAP-Clinical/DM.csv` — `AGE` (direct integer), `SEX` ('F'/'M'), `RACE`

### Bio-Hermes-002 References (INCORRECT — must fix)
Files containing incorrect "Bio-Hermes-002" references (should be "Bio-Hermes-001"):
1. `docs/regulatory/data_requirements_v1.0.md` — multiple occurrences throughout the document
2. `scripts/batch/generate_phase1_docs.py` — one occurrence

---

## Critical Notes for data-engineer-agent

1. **ADNI patient ID**: Use `RID` (integer) as the merge key; RID is stored as float in .rda files — cast to int before merging
2. **ADNI missing values**: Replace both `-1` AND `-4` with `NaN` before any processing
3. **ADNI VISCODE for baseline**: Use `'bl'` (lowercase); screening is `'sc'` (exclude from baseline join)
4. **ADNI age derivation**: No direct AGE column in PTDEMOG; compute `age = exam_year - PTDOBYY` using REGISTRY EXAMDATE
5. **ADNI DXSUM DIAGNOSIS values**: ONLY 'MCI', 'CN', 'Dementia' — NOT 'EMCI'/'LMCI' — update any code that filters for EMCI/LMCI
6. **ADNI pTau217 availability**: Only in UPENN_PLASMA_FUJIREBIO_QUANTERIX file (N=2,178 rows); not all ADNI subjects have plasma pTau217
7. **Bio-Hermes-001 patient ID column**: `USUBJID` (all files); format "BIO-HERMES-00101-NNN"
8. **Bio-Hermes-001 SDTM long format**: ALL lab files are long/tall format — must pivot/filter on LBTESTCD to extract specific analytes; use `LBSTRESN` for numeric values
9. **Bio-Hermes-001 acoustic quality filter**: Filter `FTACPTFL == 'Y'` before using FT_AURAL_ANALYTICS data
10. **CSF Abeta42 cutoff for amyloid positivity**: < 192 pg/mL from UPENNBIOMK_MASTER (UPENN immunoassay); Roche Elecsys uses different cutoff (~1000 pg/mL) — do NOT apply 192 pg/mL to Roche data
11. **APOE genotype parsing**: Split GENOTYPE on '/', count '4' occurrences → APOE4_count (0, 1, or 2)
12. **Bio-Hermes-001 is cross-sectional**: No longitudinal MMSE or survival labels. Use only for amyloid positivity and cross-sectional features.
13. **Roche Elecsys ABETA40**: Mostly NaN in early ADNI visits; Abeta42/40 ratio not computable for many rows — prefer UPENNMSMSABETA2 for ratio or UPENN_PLASMA for plasma ratio.

---

## Next Steps for data-engineer-agent

1. **Fix Bio-Hermes-002 references in**:
   - `docs/regulatory/data_requirements_v1.0.md` — replace all "Bio-Hermes-002" with "Bio-Hermes-001"
   - `scripts/batch/generate_phase1_docs.py` — replace "Bio-Hermes-002" with "Bio-Hermes-001"

2. **ADNI preprocessing** (`src/data/adni_preprocessing.py`):
   - Load UPENN_PLASMA_FUJIREBIO_QUANTERIX as primary fluid biomarker source (pTau217, NfL, GFAP, Abeta ratio)
   - Merge on RID + VISCODE baseline ('bl')
   - Load PTDEMOG.rda via pyreadr for demographics (RID as float → cast int)
   - Load DXSUM.rda for DIAGNOSIS, filter MCI only
   - Load MMSE.rda for MMSCORE longitudinal
   - Replace -1 and -4 with NaN across all numeric columns
   - Derive APOE4_count from APOERES GENOTYPE column
   - Compute age from PTDOBYY (PTDEMOG) + baseline exam year (REGISTRY EXAMDATE)

3. **Bio-Hermes-001 preprocessing** (`src/data/digital_biomarker_synthesis.py` or new `biohermes_preprocessing.py`):
   - Load DM.csv for demographics (USUBJID, AGE, SEX, RACE)
   - Load SC.csv filtered to SCTESTCD='EDUYRNUM' for education
   - Load LB_LILLY_CLINICAL_DIAGNOST.csv filtered to LBTESTCD='TAU217P' for pTau217
   - Load LB_QUANTERIX filtered to LBTESTCD='NFLP','GFAP','AB4042' for NfL, GFAP, Abeta ratio
   - Load NV.csv filtered to NVTESTCD='AMYCLAS' for amyloid label (NVSTRESC)
   - Load FT_AURAL_ANALYTICS with FTACPTFL='Y' quality filter; pivot to wide format
   - Load FT_LINUS_HEALTH with FTACPTFL='Y'; pivot to wide format (spiral, trails, SDMT)
   - Merge all on USUBJID

4. **Dataset integration** (`src/data/dataset.py`):
   - Create unified patient record merging ADNI and Bio-Hermes-001
   - Hash patient IDs before any logging: `hashlib.sha256(str(rid).encode()).hexdigest()`

---

## Open Questions for Human Review

1. **pTau217 units discrepancy**: ADNI `pT217_F` values observed (0.152–0.586 pg/mL) but NeuroFusion validated range is 0.1–100 pg/mL — values look low. Confirm whether Fujirebio pTau217 assay uses pg/mL or ng/L (1 pg/mL = 1 ng/L; no conversion needed). The range 0.1–100 pg/mL appears correct for Fujirebio.
2. **ADNI acoustic data absence**: No acoustic/speech data in ADNI files. Acoustic encoder will train only on Bio-Hermes-001 (N~1,005). Confirm if this is acceptable or if DementiaBank data should be sourced.
3. **ADNIMERGE2 ADSL.rda**: Not examined — this file likely contains a pre-merged summary dataset. Investigate if simpler than merging individual files.
4. **Bio-Hermes-001 motor data**: Spiral drawing times are in FT_LINUS_HEALTH. Confirm mapping: `SPCWDTM` (CW Dominant Time) → motor_encoder input vs. deriving jitter/shimmer-like measures from timing variability.
