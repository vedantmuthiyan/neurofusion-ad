# ADNI File Inventory
**Date**: 2026-03-11
**Agent**: data-explorer-agent

## Summary
- Total CSV files at root level: 12
- Total .rda datasets examined in ADNIMERGE2: 8
- Total .rda files available in ADNIMERGE2/data/: 100+
- Key files identified for preprocessing: UPENNBIOMK_MASTER, UPENNBIOMK_ROCHE_ELECSYS, APOERES, UPENN_PLASMA_FUJIREBIO_QUANTERIX, ADNIMERGE2/PTDEMOG, ADNIMERGE2/DXSUM, ADNIMERGE2/MMSE

---

## Root CSV Files

### UPENNBIOMK_MASTER_28Feb2026.csv
- **Path**: data/raw/adni/UPENNBIOMK_MASTER_28Feb2026.csv
- **Size**: 748,129 bytes
- **Rows**: 5,876
- **Columns (14)**: `['RID', 'VISCODE', 'BATCH', 'KIT', 'STDS', 'DRAWDTE', 'RUNDATE', 'ABETA', 'TAU', 'PTAU', 'ABETA_RAW', 'TAU_RAW', 'PTAU_RAW', 'update_stamp']`
- **Key columns**:
  - `RID` — integer participant ID (merge key)
  - `VISCODE` — visit code ('bl', 'm12', 'm24', 'm36', 'm48', 'm60', 'm72', 'm84')
  - `ABETA` — CSF Abeta42 (pg/mL); cutoff < 192 pg/mL = amyloid positive
  - `TAU` — CSF total tau (pg/mL)
  - `PTAU` — CSF pTau181 (pg/mL)
- **Relevance**: PRIMARY — CSF biomarkers for fluid encoder; ABETA is primary amyloid positivity label source

---

### UPENNBIOMK_ROCHE_ELECSYS_28Feb2026.csv
- **Path**: data/raw/adni/UPENNBIOMK_ROCHE_ELECSYS_28Feb2026.csv
- **Size**: 414,439 bytes
- **Rows**: 3,174
- **Columns (13)**: `['PHASE', 'PTID', 'RID', 'VISCODE2', 'EXAMDATE', 'BATCH', 'RUNDATE', 'ABETA40', 'ABETA42', 'TAU', 'PTAU', 'COMMENT', 'update_stamp']`
- **Key columns**:
  - `RID` — integer participant ID
  - `PTID` — string format (e.g., "011_S_0003")
  - `VISCODE2` — visit code ('bl', 'm12', 'm18', 'm24', etc.; extends to 'm186')
  - `ABETA40` — CSF Abeta40 from Roche Elecsys assay (NOTE: mostly NaN in early visits; different cutoff from master file)
  - `ABETA42` — CSF Abeta42 from Roche Elecsys assay (pg/mL)
  - `TAU` — CSF total tau
  - `PTAU` — CSF pTau181
- **Relevance**: SECONDARY — Roche Elecsys assay data; use for Abeta42/40 ratio when ABETA40 is available

---

### APOERES_28Feb2026.csv
- **Path**: data/raw/adni/APOERES_28Feb2026.csv
- **Size**: 329,977 bytes
- **Rows**: 3,253
- **Columns (16)**: `['PHASE', 'PTID', 'RID', 'VISCODE', 'GENOTYPE', 'APTESTDT', 'APVOLUME', 'APRECEIVE', 'APAMBTEMP', 'APRESAMP', 'APUSABLE', 'ID', 'SITEID', 'USERDATE', 'USERDATE2', 'update_stamp']`
- **Key columns**:
  - `RID` — integer participant ID
  - `PTID` — string format
  - `VISCODE` — visit code
  - `GENOTYPE` — APOE genotype string (values: '2/2', '2/3', '2/4', '3/3', '3/4', '4/4')
- **Relevance**: PRIMARY — APOE4 status; derive APOE4 count by splitting on '/' and counting '4' occurrences

---

### GENETIC_28Feb2026.csv
- **Path**: data/raw/adni/GENETIC_28Feb2026.csv
- **Size**: 2,929,073 bytes
- **Rows**: 10,222
- **Columns (57)**: `['PHASE', 'PTID', 'RID', 'VISCODE', 'VISCODE2', 'VISDATE', 'RECNO', 'APCOLLECT', 'DNAREASON', 'APTIME', 'APVOLUME', 'WBLDVOL1', 'WBLDVOL2', 'WBLDFROZT', 'RNACOLL', 'RNAREASON', 'RNADATE', 'RNATIME', 'RNAVOL', 'RNAVOL1', 'RNAVOL2', 'RNASHIP', 'RNAFROZT', 'BCOAT', 'BCREASON', 'BCVOL', 'BCVOL1', 'BCVOL2', 'BCEXTVOL1', 'BCEXTVOL2', 'BCSHIP', 'BCFROZT', 'RBCCOLL', 'RBCTUBE', 'RBCALIQ1', 'RBCALIQ2', 'RBCFROZT', 'PBMCCOLL', 'PBMCREASON', 'PBMCVOL', 'PBMCTIME', 'PBMCVOL1', 'PBMCVOL2', 'CLDATE', 'CLCOLL', 'CLREASON', 'CLTIME', 'CLVOLUME', 'GENFEDDATE', 'ID', 'SITEID', 'USERDATE', 'USERDATE2', 'DD_CRF_VERSION_LABEL', 'LANGUAGE_CODE', 'HAS_QC_ERROR', 'update_stamp']`
- **Key columns**: `RID`, `PTID`, `VISCODE` — collection tracking only; no biomarker measurements
- **Relevance**: LOW — blood/RNA/DNA collection logistics, not needed for NeuroFusion model features

---

### RMT_APOERES_28Feb2026.csv
- **Path**: data/raw/adni/RMT_APOERES_28Feb2026.csv
- **Size**: 56,364 bytes
- **Rows**: 1,083
- **Columns (4)**: `['PHASE', 'ADNIOnlineID', 'GENOTYPE', 'update_stamp']`
- **Key columns**: `ADNIOnlineID` — online participant ID (NOT the standard RID integer); `GENOTYPE` — same format as APOERES
- **Relevance**: SECONDARY/SUPPLEMENTAL — APOE genotype for participants in remote monitoring track; no RID column, must join via ADNIOnlineID mapping if needed

---

### UPENNMSMSABETA_28Feb2026.csv
- **Path**: data/raw/adni/UPENNMSMSABETA_28Feb2026.csv
- **Size**: 31,891 bytes
- **Rows**: 400
- **Columns (8)**: `['RID', 'VISCODE', 'DRAWDATE', 'RUNDATE', 'ABETA42', 'ABETA40', 'ABETA38', 'update_stamp']`
- **Key columns**:
  - `RID` — integer participant ID
  - `VISCODE` — visit code
  - `ABETA42` — CSF Abeta42 by mass spectrometry (pg/mL); values in thousands (e.g., 987, 1246)
  - `ABETA40` — CSF Abeta40 by mass spectrometry (pg/mL)
  - `ABETA38` — CSF Abeta38 by mass spectrometry
- **NOTE**: Units differ from UPENNBIOMK_MASTER; values are ~10x higher. Ratio ABETA42/ABETA40 comparable.
- **Relevance**: SUPPLEMENTAL — provides Abeta42/40 ratio from mass spec assay for earliest ADNI cohort (400 subjects, baseline only)

---

### UPENNMSMSABETA2_28Feb2026.csv
- **Path**: data/raw/adni/UPENNMSMSABETA2_28Feb2026.csv
- **Size**: 131,358 bytes
- **Rows**: 1,445
- **Columns (10)**: `['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'VID', 'RUNDATE', 'ABETA42', 'ABETA40', 'ABETA38', 'update_stamp']`
- **Key columns**: `RID`, `VISCODE`, `ABETA42`, `ABETA40`, `ABETA38`
- **Relevance**: SUPPLEMENTAL — extended mass spec Abeta cohort; includes longitudinal visits

---

### UPENNMSMSABETA2CRM_28Feb2026.csv
- **Path**: data/raw/adni/UPENNMSMSABETA2CRM_28Feb2026.csv
- **Size**: 140,473 bytes
- **Rows**: 1,445
- **Columns (11)**: `['RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'VID', 'RUNDATE', 'ABETA42', 'ABETA40', 'ABETA38', 'ABETA42CRM', 'update_stamp']`
- **Key columns**: Same as UPENNMSMSABETA2 plus `ABETA42CRM` — CRM-standardized Abeta42 value
- **Relevance**: SUPPLEMENTAL — CRM-corrected mass spec data; prefer ABETA42CRM for harmonization across assays

---

### UPENNPLASMA_28Feb2026.csv
- **Path**: data/raw/adni/UPENNPLASMA_28Feb2026.csv
- **Size**: 119,478 bytes
- **Rows**: 2,453
- **Columns (5)**: `['RID', 'VISCODE', 'AB40', 'AB42', 'update_stamp']`
- **Key columns**:
  - `RID` — integer participant ID
  - `VISCODE` — visit code
  - `AB40` — plasma Abeta40 (pg/mL)
  - `AB42` — plasma Abeta42 (pg/mL)
- **Relevance**: SUPPLEMENTAL — plasma Abeta42/40 ratio; no pTau, limited features

---

### UPENN_PLASMA_FUJIREBIO_QUANTERIX_28Feb2026.csv
- **Path**: data/raw/adni/UPENN_PLASMA_FUJIREBIO_QUANTERIX_28Feb2026.csv
- **Size**: 396,959 bytes
- **Rows**: 2,178
- **Columns (19)**: `['PHASE', 'PTID', 'RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'Primary', 'Additive', 'pT217_F', 'AB42_F', 'AB40_F', 'AB42_AB40_F', 'pT217_AB42_F', 'NfL_Q', 'GFAP_Q', 'NfL_F', 'GFAP_F', 'Comment', 'update_stamp']`
- **Key columns**:
  - `RID` — integer participant ID
  - `VISCODE` — visit code ('bl', 'm24', etc.)
  - `pT217_F` — plasma pTau217 by Fujirebio (pg/mL); matches NeuroFusion validated range 0.1–100
  - `AB42_F` — plasma Abeta42 by Fujirebio (pg/mL)
  - `AB40_F` — plasma Abeta40 by Fujirebio (pg/mL)
  - `AB42_AB40_F` — plasma Abeta42/Abeta40 ratio by Fujirebio
  - `NfL_Q` — plasma NfL by Quanterix Simoa (pg/mL)
  - `GFAP_Q` — plasma GFAP by Quanterix Simoa (pg/mL)
  - `NfL_F` — plasma NfL by Fujirebio
  - `GFAP_F` — plasma GFAP by Fujirebio
- **Relevance**: HIGH PRIORITY — contains plasma pTau217 (pT217_F), NfL, GFAP, Abeta42/40 ratio; directly maps to NeuroFusion fluid encoder features

---

### YASSINE_CSF_28Feb2026.csv
- **Path**: data/raw/adni/YASSINE_CSF_28Feb2026.csv
- **Size**: 34,179 bytes
- **Rows**: 188
- **Columns (25)**: `['RID', 'EXAMDATE', 'VISCODE2', 'Sample_ID', 'Number', 'Type', 'Box', 'Order_in_Box', 'MALDI', 'Date', 'Phenotype', 'E2_level', 'E3_level', 'E4_level', 'E3_E2', 'E4_E3', 'E2_glyc', 'E3_glyc', 'E4_glyc', 'Total_glyc', 'E2_2nd_glyc', 'E3_2nd_glyc', 'E4_2nd_glyc', 'Total_2nd_glyc', 'update_stamp']`
- **Key columns**: `RID`, `E2_level`, `E3_level`, `E4_level` — APOE isoform protein levels in CSF
- **Relevance**: LOW/SUPPLEMENTAL — APOE protein glycosylation in CSF; small N=188; not a primary feature

---

### YASSINE_PLASMA_28Feb2026.csv
- **Path**: data/raw/adni/YASSINE_PLASMA_28Feb2026.csv
- **Size**: 31,598 bytes
- **Rows**: 188
- **Columns (22)**: `['RID', 'EXAMDATE', 'VISCODE2', 'Sample_ID', 'Number', 'Type', 'Box', 'Order_in_Box', 'MALDI', 'Date', 'Phenotype', 'E2_level', 'E3_level', 'E4_level', 'E3_E2', 'E4_E2', 'E4_E3', 'E2_glyc', 'E3_glyc', 'E4_glyc', 'Total_glyc', 'update_stamp']`
- **Key columns**: `RID`, `E2_level`, `E3_level`, `E4_level` — APOE isoform protein levels in plasma
- **Relevance**: LOW/SUPPLEMENTAL — same as Yassine CSF; N=188 only

---

## ADNIMERGE2 R Package Datasets

### MMSE.rda
- **Path**: data/raw/adni/ADNIMERGE2/data/MMSE.rda
- **Size**: 225,547 bytes
- **Rows**: 14,599
- **Columns (59)**: `['ORIGPROT', 'COLPROT', 'PTID', 'RID', 'VISCODE', 'VISCODE2', 'VISDATE', 'DONE', 'NDREASON', 'SOURCE', 'MMDATE', 'MMYEAR', 'MMMONTH', 'MMDAY', 'MMSEASON', 'MMHOSPIT', 'MMFLOOR', 'MMCITY', 'MMAREA', 'MMSTATE', 'WORDLIST', 'WORD1', 'WORD2', 'WORD3', 'MMTRIALS', 'MMD', 'MML', 'MMR', 'MMO', 'MMW', 'MMLTR1', 'MMLTR2', 'MMLTR3', 'MMLTR4', 'MMLTR5', 'MMLTR6', 'MMLTR7', 'WORLDSCORE', 'WORD1DL', 'WORD2DL', 'WORD3DL', 'MMWATCH', 'MMPENCIL', 'MMREPEAT', 'MMHAND', 'MMFOLD', 'MMONFLR', 'MMREAD', 'MMWRITE', 'MMDRAW', 'MMSCORE', 'ID', 'SITEID', 'USERDATE', 'USERDATE2', 'DD_CRF_VERSION_LABEL', 'LANGUAGE_CODE', 'HAS_QC_ERROR', 'update_stamp']`
- **Key columns**: `RID`, `VISCODE`, `MMSCORE` (total MMSE score 0–30; 161 NaN values)
- **Relevance**: PRIMARY — MMSE longitudinal scores for regression target label (MMSE slope)

---

### DXSUM.rda
- **Path**: data/raw/adni/ADNIMERGE2/data/DXSUM.rda
- **Size**: 225,634 bytes
- **Rows**: 15,881
- **Columns (42)**: `['ORIGPROT', 'COLPROT', 'PTID', 'RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'DIAGNOSIS', 'DXNORM', 'DXNODEP', 'DXMCI', 'DXMDES', 'DXMPTR1', 'DXMPTR2', 'DXMPTR3', 'DXMPTR4', 'DXMPTR5', 'DXMPTR6', 'DXMDUE', 'DXMOTHET', 'DXDSEV', 'DXDDUE', 'DXAD', 'DXAPP', 'DXAPROB', 'DXAPOSS', 'DXPARK', 'DXPDES', 'DXPCOG', 'DXPATYP', 'DXDEP', 'DXOTHDEM', 'DXODES', 'DXCONFID', 'ID', 'SITEID', 'USERDATE', 'USERDATE2', 'DD_CRF_VERSION_LABEL', 'LANGUAGE_CODE', 'HAS_QC_ERROR', 'update_stamp']`
- **Key columns**:
  - `RID` — integer participant ID
  - `VISCODE` — visit code
  - `DIAGNOSIS` — consolidated diagnosis (values: 'MCI' n=6,565; 'CN' n=6,275; 'Dementia' n=2,996)
- **Relevance**: PRIMARY — filter to MCI patients; track progression from MCI to Dementia for survival label

---

### PTDEMOG.rda
- **Path**: data/raw/adni/ADNIMERGE2/data/PTDEMOG.rda
- **Size**: 130,539 bytes
- **Rows**: 6,219
- **Columns (85)**: `['ORIGPROT', 'COLPROT', 'PTID', 'RID', 'VISCODE', 'VISCODE2', 'VISDATE', 'PTSOURCE', 'PTGENDER', 'PTDOB', 'PTDOBYY', 'PTHAND', 'PTMARRY', 'PTEDUCAT', 'PTWORKHS', 'PTWORK', 'PTNOTRT', 'PTRTYR', 'PTHOME', 'PTTLANG', 'PTPLANG', 'PTADBEG', 'PTCOGBEG', 'PTADDX', 'PTETHCAT', 'PTRACCAT', 'PTIDENT', 'PTORIENT', 'PTORIENTOT', 'PTENGSPK', 'PTNLANG', 'PTENGSPKAGE', 'PTCLANG', 'PTLANGSP', 'PTLANGWR', 'PTSPTIM', 'PTSPOTTIM', 'PTLANGPR1', 'PTLANGSP1', 'PTLANGRD1', 'PTLANGWR1', 'PTLANGUN1', 'PTLANGPR2', 'PTLANGSP2', 'PTLANGRD2', 'PTLANGWR2', 'PTLANGUN2', 'PTLANGPR3', 'PTLANGSP3', 'PTLANGRD3', 'PTLANGWR3', 'PTLANGUN3', 'PTLANGPR4', 'PTLANGSP4', 'PTLANGRD4', 'PTLANGWR4', 'PTLANGUN4', 'PTLANGPR5', 'PTLANGSP5', 'PTLANGRD5', 'PTLANGWR5', 'PTLANGUN5', 'PTLANGPR6', 'PTLANGSP6', 'PTLANGRD6', 'PTLANGWR6', 'PTLANGUN6', 'PTLANGTTL', 'PTETHCATH', 'PTASIAN', 'PTOPI', 'PTBORN', 'PTBIRPL', 'PTIMMAGE', 'PTIMMWHY', 'PTBIRPR', 'PTBIRGR', 'ID', 'SITEID', 'USERDATE', 'USERDATE2', 'DD_CRF_VERSION_LABEL', 'LANGUAGE_CODE', 'HAS_QC_ERROR', 'update_stamp']`
- **Key columns**:
  - `RID` — integer participant ID (stored as float in rda, cast to int)
  - `PTGENDER` — sex ('Male', 'Female')
  - `PTDOBYY` — year of birth (derive age = exam_year - PTDOBYY)
  - `PTEDUCAT` — years of education (integer)
  - `PTETHCAT` — ethnicity ('Not Hispanic or Latino', 'Hispanic or Latino')
  - `PTRACCAT` — race ('White', 'Black', 'Asian', etc.)
- **Relevance**: PRIMARY — demographics for clinical encoder (age, sex, education, race/ethnicity)

---

### REGISTRY.rda
- **Path**: data/raw/adni/ADNIMERGE2/data/REGISTRY.rda
- **Size**: 291,511 bytes
- **Rows**: 28,858
- **Columns (27)**: `['ORIGPROT', 'COLPROT', 'PTID', 'RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'RGCONDCT', 'VISTYPE', 'RGSTATUS', 'RGREASON', 'RGSOURCE', 'RGRESCRN', 'RGPREVID', 'CHANGTR', 'CGTRACK', 'CGTRACK2', 'PTSTATUS', 'PTTYPE', 'ID', 'SITEID', 'USERDATE', 'USERDATE2', 'DD_CRF_VERSION_LABEL', 'LANGUAGE_CODE', 'HAS_QC_ERROR', 'update_stamp']`
- **Key columns**: `RID`, `PTID`, `VISCODE`, `EXAMDATE` — visit tracking and exam dates
- **Relevance**: SUPPLEMENTAL — use EXAMDATE for computing exact visit dates (years from baseline)

---

### APOERES.rda
- **Path**: data/raw/adni/ADNIMERGE2/data/APOERES.rda
- **Size**: 23,992 bytes
- **Rows**: 3,008
- **Columns (17)**: `['ORIGPROT', 'COLPROT', 'PTID', 'RID', 'VISCODE', 'GENOTYPE', 'APTESTDT', 'APVOLUME', 'APRECEIVE', 'APAMBTEMP', 'APRESAMP', 'APUSABLE', 'ID', 'SITEID', 'USERDATE', 'USERDATE2', 'update_stamp']`
- **Key columns**: `RID`, `GENOTYPE` — same data as root APOERES CSV but from R package
- **Relevance**: EQUIVALENT to APOERES_28Feb2026.csv; prefer the root CSV (has PTID, same column structure)

---

### BIOMARK.rda
- **Path**: data/raw/adni/ADNIMERGE2/data/BIOMARK.rda
- **Size**: 404,789 bytes
- **Rows**: 13,988
- **Columns (66)**: `['ORIGPROT', 'COLPROT', 'PTID', 'RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'RECNO', 'BIBLOOD', 'BIURINE', 'BLREASON', 'BIFAST', 'BITIME', 'BIREDTIME', 'BIREDAMT', 'BIREDCENT', 'BIREDTRNS', 'BIREDVOL', 'BIREDFROZ', 'BILAVTIME', 'BINEEDLE', 'BILAVAMT', 'BILAVCENT', 'BILAVTRNS', 'BILAVVOL', 'BILAVFROZ', 'BIURITIME', 'BIURIAMT', 'BIURITRNS', 'BIURIVOL', 'BIURIFROZ', 'BICSF', 'BINONE', 'REASON', 'BICSFFAST', 'BICSFTIME', 'BIMETHOD', 'NEEDLESIZE', 'INTERSPACE', 'PTPOSITION', 'COLTUBETYP', 'SHPTUBETYP', 'TUBEMIN', 'BICSFAMT', 'BICSFTRNS', 'BICSFVOL', 'BICSFFROZ', 'BILPPATCH', 'BILPFLURO', 'BILPSPFILM', 'BILPPADATE', 'BILPFLDATE', 'BILPSPDATE', 'BILPPADATEYR_DRVD', 'BILPFLDATEYR_DRVD', 'BILPSPDATEYR_DRVD', 'BILPOTPROC', 'BIFEDDATE', 'ID', 'SITEID', 'USERDATE', 'USERDATE2', 'DD_CRF_VERSION_LABEL', 'LANGUAGE_CODE', 'HAS_QC_ERROR', 'update_stamp']`
- **Key columns**: `RID`, `BICSF` — whether CSF was collected (flag); collection logistics tracking
- **Relevance**: LOW — biospecimen collection tracking only; no biomarker measurements

---

### CDR.rda
- **Path**: data/raw/adni/ADNIMERGE2/data/CDR.rda
- **Size**: 199,728 bytes
- **Rows**: 14,617
- **Columns (26)**: `['ORIGPROT', 'COLPROT', 'PTID', 'RID', 'VISCODE', 'VISCODE2', 'VISDATE', 'CDSOURCE', 'CDVERSION', 'SPID', 'CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE', 'CDGLOBAL', 'CDRSB', 'ID', 'SITEID', 'USERDATE', 'USERDATE2', 'DD_CRF_VERSION_LABEL', 'LANGUAGE_CODE', 'HAS_QC_ERROR', 'update_stamp']`
- **Key columns**: `RID`, `VISCODE`, `CDGLOBAL` (CDR global score 0/0.5/1/2/3), `CDRSB` (CDR Sum of Boxes)
- **Relevance**: SUPPLEMENTAL — CDR can be used as additional clinical severity feature; CDRSB is more sensitive than CDGLOBAL

---

### GDSCALE.rda
- **Path**: data/raw/adni/ADNIMERGE2/data/GDSCALE.rda
- **Size**: 195,767 bytes
- **Rows**: 13,694
- **Columns (33)**: `['ORIGPROT', 'COLPROT', 'PTID', 'RID', 'VISCODE', 'VISCODE2', 'VISDATE', 'SOURCE', 'GDUNABL', 'GDSATIS', 'GDDROP', 'GDEMPTY', 'GDBORED', 'GDSPIRIT', 'GDAFRAID', 'GDHAPPY', 'GDHELP', 'GDHOME', 'GDMEMORY', 'GDALIVE', 'GDWORTH', 'GDENERGY', 'GDHOPE', 'GDBETTER', 'GDTOTAL', 'ID', 'SITEID', 'USERDATE', 'USERDATE2', 'DD_CRF_VERSION_LABEL', 'LANGUAGE_CODE', 'HAS_QC_ERROR', 'update_stamp']`
- **Key columns**: `RID`, `VISCODE`, `GDTOTAL` — Geriatric Depression Scale total (0–15)
- **Relevance**: SUPPLEMENTAL — depression screening; may be feature in clinical encoder

---

## Files Identified for NeuroFusion Preprocessing

| Purpose | File | Key Columns |
|---------|------|-------------|
| CSF biomarkers primary (pTau181, Abeta42) | UPENNBIOMK_MASTER_28Feb2026.csv | RID, VISCODE, PTAU, ABETA, TAU |
| Plasma pTau217, NfL, GFAP (PRIMARY fluid encoder) | UPENN_PLASMA_FUJIREBIO_QUANTERIX_28Feb2026.csv | RID, VISCODE, pT217_F, AB42_F, AB40_F, AB42_AB40_F, NfL_Q, GFAP_Q |
| APOE genotype | APOERES_28Feb2026.csv | RID, VISCODE, GENOTYPE |
| Demographics (age, sex, education) | ADNIMERGE2/data/PTDEMOG.rda | RID, PTGENDER, PTDOBYY, PTEDUCAT, PTETHCAT, PTRACCAT |
| MMSE longitudinal scores | ADNIMERGE2/data/MMSE.rda | RID, VISCODE, MMSCORE |
| Diagnosis progression (MCI/CN/Dementia) | ADNIMERGE2/data/DXSUM.rda | RID, VISCODE, DIAGNOSIS |
| Visit dates (for time-to-event) | ADNIMERGE2/data/REGISTRY.rda | RID, VISCODE, EXAMDATE |
| CDR severity (supplemental feature) | ADNIMERGE2/data/CDR.rda | RID, VISCODE, CDGLOBAL, CDRSB |
| CSF Abeta42 Roche Elecsys (secondary) | UPENNBIOMK_ROCHE_ELECSYS_28Feb2026.csv | RID, VISCODE2, ABETA42, ABETA40, PTAU |
| CSF Abeta42/40 mass spec ratio (supplemental) | UPENNMSMSABETA2CRM_28Feb2026.csv | RID, VISCODE, ABETA42CRM, ABETA40 |

---

## Missing Data Notes
- **ADNI acoustic/speech features**: NO acoustic (jitter/shimmer) data found in any ADNI CSV. ADNI does not include voice/speech recordings. For acoustic encoder in NeuroFusion, use Bio-Hermes-001 FT_AURAL_ANALYTICS.csv exclusively.
- **ADNI motor data**: No motor assessment files found in the root CSV list. Motor features (spiral drawing etc.) are not available in the files provided.
- **ABETA40 in ROCHE_ELECSYS**: ABETA40 column exists but is mostly NaN in early visits; Abeta42/40 ratio cannot be computed for most rows.
- **pTau217 in ADNI**: Only available in UPENN_PLASMA_FUJIREBIO_QUANTERIX file (pT217_F column), not in UPENNBIOMK_MASTER. N=2,178 rows (subset of full ADNI cohort).
- **ADNIMERGE2 ADSL.rda**: Not examined but likely contains a merged summary table — investigate for Phase 2B if needed.
