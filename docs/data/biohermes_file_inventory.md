# Bio-Hermes-001 File Inventory
**Date**: 2026-03-11
**Prepared by**: NeuroFusion-AD Data Engineering Team

---

## Summary
- **Study**: Bio-Hermes-001 (NCT04854135), STUDYID = "BIO-HERMES"
- **Total subjects**: 1,005 (DM.csv USUBJID unique count)
- **Patient ID format**: `USUBJID` = "BIO-HERMES-00101-NNN" (e.g., "BIO-HERMES-00101-001")
- **Short patient ID**: `SUBJID` = "00101-NNN"
- **Demographics**: Age 59–85 (mean 72), 56% female, 86% White

### Three Main Sections
1. **BloodBasedBiomarkers** — SDTM-format LB (lab) files from 6 vendors: Roche Diagnostics, Lilly (pTau217), Quanterix, C2N, Eli Lilly Genome Subset, University of Gothenburg
2. **DigitalTests** — SDTM FT (functional tests) from Aural Analytics (acoustic), Linus Health (motor/cognitive), Cognivue; plus QS questionnaires
3. **GAP-Clinical** — SDTM clinical backbone: DM (demographics), DS (disposition), LB_EDC (lab central), NV (neurovisit/amyloid PET), RS (results/FAQ), SC (screening), FT (combined digital), NV (visits), etc.
4. **OtherAssessments** — Amyloid PET SUVR/Centiloid (Lilly), MRI (IXICO), Retinal (Retispec)
5. **Proteomics** — NULISA proteomics panel, EMtherapro proteomic data
6. **SDTMFiles** — Duplicate SDTM package (BIO HERMES-001 SDTM Package subdirectory)

---

## SDTM Data Structure Note
All Bio-Hermes-001 LB files use a **long/tall format** (SDTM standard):
- `LBTESTCD` — test code (e.g., "TAU217P", "AMYLB42")
- `LBTEST` — test name
- `LBORRES` — original result (string)
- `LBSTRESN` — numeric result (float)
- `LBSTRESU` — result units

To extract a specific analyte (e.g., pTau217), filter `LBTESTCD == 'TAU217P'` and use `LBSTRESN` as the numeric value.

---

## BloodBasedBiomarkers Files

### LB_ROCHE.csv
- **Path**: data/raw/biohermes/BIOHERMES001/BloodBasedBiomarkers/Roche Diagnostics/LB_ROCHE.csv
- **Size**: 1,290,493 bytes
- **Rows**: 6,937
- **Columns (29)**: `['STUDYID', 'DOMAIN', 'USUBJID', 'LBSEQ', 'LBREFID', 'LBSPID', 'LBTESTCD', 'LBTEST', 'LBCAT', 'LBSCAT', 'LBORRES', 'LBORRESU', 'LBORNRLO', 'LBORNRHI', 'LBSTRESC', 'LBSTRESN', 'LBSTRESU', 'LBSTNRLO', 'LBSTNRHI', 'LBNRIND', 'LBSTAT', 'LBREASND', 'LBNAM', 'LBSPEC', 'VISITNUM', 'VISIT', 'EPOCH', 'LBDTC', 'LBDY']`
- **Patient ID**: `USUBJID` (991 unique subjects)
- **Test codes present (LBTESTCD)**:
  - `AMYLB40` — Amyloid Beta 1-40
  - `AMYLB42` — Amyloid Beta 1-42
  - `APOE4` — Apolipoprotein E4
  - `GFAP` — Glial Fibrillary Acidic Protein
  - `NFLP` — Neurofilament Light Chain Protein
  - `TAU181P` — Phosphorylated Tau Protein 181
  - `TPROTP` — Phosphorylated Tau Protein (total)
- **NOTE**: Roche has pTau181 (`TAU181P`) and NfL (`NFLP`), NOT pTau217. Use LB_LILLY for pTau217.
- **Relevance**: HIGH — Roche plasma panel: Abeta42/40, NfL, GFAP, pTau181; pTau181 is proxy for pTau217

### SUPPLB_ROCHE.csv
- **Path**: data/raw/biohermes/BIOHERMES001/BloodBasedBiomarkers/Roche Diagnostics/SUPPLB_ROCHE.csv
- **Size**: 105,868 bytes
- **Rows**: 1,026
- **Columns (10)**: `['STUDYID', 'RDOMAIN', 'USUBJID', 'IDVAR', 'IDVARVAL', 'QNAM', 'QLABEL', 'QVAL', 'QORIG', 'QEVAL']`
- **Relevance**: SUPPLEMENTAL metadata for LB_ROCHE records

---

### LB_LILLY_CLINICAL_DIAGNOST.csv (PRIMARY pTau217 source)
- **Path**: data/raw/biohermes/BIOHERMES001/BloodBasedBiomarkers/Lilly/pTau217_Data/LB_LILLY_CLINICAL_DIAGNOST.csv
- **Size**: 234,537 bytes
- **Rows**: 990
- **Columns (29)**: `['STUDYID', 'DOMAIN', 'USUBJID', 'LBSEQ', 'LBREFID', 'LBSPID', 'LBTESTCD', 'LBTEST', 'LBCAT', 'LBSCAT', 'LBORRES', 'LBORRESU', 'LBORNRLO', 'LBORNRHI', 'LBSTRESC', 'LBSTRESN', 'LBSTRESU', 'LBSTNRLO', 'LBSTNRHI', 'LBNRIND', 'LBSTAT', 'LBREASND', 'LBNAM', 'LBSPEC', 'VISITNUM', 'VISIT', 'EPOCH', 'LBDTC', 'LBDY']`
- **Patient ID**: `USUBJID`
- **Test codes present (LBTESTCD)**:
  - `TAU217P` — Phosphorylated Tau Protein 217 (pTau-217 by Lilly immunoassay)
- **pTau217 extraction**: filter `LBTESTCD == 'TAU217P'`, use `LBSTRESN` for numeric value
- **Relevance**: PRIMARY pTau-217 source for Bio-Hermes-001 (N=990 rows, ~990 patient-visits)

### SUPPLB_LILLY_CLINICAL_DIAGNOST.csv
- **Path**: data/raw/biohermes/BIOHERMES001/BloodBasedBiomarkers/Lilly/pTau217_Data/SUPPLB_LILLY_CLINICAL_DIAGNOST.csv
- **Size**: 106,311 bytes
- **Rows**: 1,028
- **Columns (10)**: `['STUDYID', 'RDOMAIN', 'USUBJID', 'IDVAR', 'IDVARVAL', 'QNAM', 'QLABEL', 'QVAL', 'QORIG', 'QEVAL']`
- **Relevance**: SUPPLEMENTAL metadata

---

### LB_QUANTERIX_CORPORATION.csv
- **Path**: data/raw/biohermes/BIOHERMES001/BloodBasedBiomarkers/Quanterix/LB_QUANTERIX_CORPORATION.csv
- **Size**: 1,564,127 bytes
- **Rows**: 6,937
- **Columns (29)**: Same SDTM LB structure as LB_ROCHE.csv
- **Test codes (LBTESTCD)**:
  - `AB4042` — Amyloid Beta 1-40/Amyloid Beta 1-42 ratio
  - `AMYLB40` — Amyloid Beta 1-40
  - `AMYLB42` — Amyloid Beta 1-42
  - `GFAP` — Glial Fibrillary Acidic Protein (Simoa)
  - `NFLP` — Neurofilament Light Chain Protein (Simoa)
  - `TAU181P` — Phosphorylated Tau Protein 181 (Simoa)
  - `TPROT` — Tau Protein (total)
- **Relevance**: HIGH — Quanterix Simoa NfL and GFAP; AB4042 ratio pre-computed

---

### LB_C2N.csv
- **Path**: data/raw/biohermes/BIOHERMES001/BloodBasedBiomarkers/C2N/LB_C2N.csv
- **Size**: 1,111,973 bytes
- **Rows**: 5,946
- **Columns (29)**: Same SDTM LB structure
- **Test codes (LBTESTCD)**:
  - `AB4042` — Amyloid Beta 1-40/Amyloid Beta 1-42 ratio
  - `AMYLB40` — Amyloid Beta 1-40
  - `AMYLB42` — Amyloid Beta 1-42
  - `APOE` — Apolipoprotein E
  - `APS` — Amyloid Probability Score (C2N proprietary)
  - `INTP` — Interpretation
- **Relevance**: HIGH — C2N Precivity AD amyloid probability score; APS is a validated amyloid biomarker

---

### LB_ELI_LILLY.csv (Genome Subset)
- **Path**: data/raw/biohermes/BIOHERMES001/BloodBasedBiomarkers/Lilly/Genome Subset/LB_ELI_LILLY.csv
- **Size**: 773,247 bytes
- **Rows**: 5,712
- **Columns (29)**: Same SDTM LB structure
- **Relevance**: SUPPLEMENTAL — genomic subset participants; same assay structure

---

### LB_MERCK.csv
- **Path**: data/raw/biohermes/BIOHERMES001/BloodBasedBiomarkers/Merck/LB_MERCK.csv
- **Size**: 12,466,618 bytes
- **Rows**: 70,574
- **Columns (29)**: Same SDTM LB structure
- **Relevance**: SUPPLEMENTAL — large panel from Merck; examine LBTESTCD values before use

---

### LB_UNIVERSITY_OF_GOTHENBUR.csv
- **Path**: data/raw/biohermes/BIOHERMES001/BloodBasedBiomarkers/University of Gothenburg/LB_UNIVERSITY_OF_GOTHENBUR.csv
- **Size**: 202,702 bytes
- **Rows**: 991
- **Columns (29)**: Same SDTM LB structure
- **Test codes (LBTESTCD)**:
  - `TAU217P` — Phosphorylated Tau Protein 217 (SECONDARY pTau-217 source)
- **Relevance**: SECONDARY pTau-217 source (University of Gothenburg immunoassay variant)

---

## DigitalTests Files

### FT_AURAL_ANALYTICS.csv (PRIMARY acoustic features)
- **Path**: data/raw/biohermes/BIOHERMES001/DigitalTests/Aural Analytics/FT_AURAL_ANALYTICS.csv
- **Size**: 6,405,686 bytes
- **Rows**: 21,900
- **Columns (34)**: `['STUDYID', 'DOMAIN', 'USUBJID', 'FTSEQ', 'FTGRPID', 'FTREFID', 'FTTESTCD', 'FTTEST', 'FTTSTDTL', 'FTCAT', 'FTORRES', 'FTORRESU', 'FTORNRLO', 'FTORNRHI', 'FTSTRESC', 'FTSTRESN', 'FTSTRESU', 'FTSTNRLO', 'FTSTNRHI', 'FTNRIND', 'FTSTAT', 'FTREASND', 'FTNAM', 'FTEVALID', 'FTACPTFL', 'FTREPNUM', 'VISITNUM', 'VISIT', 'EPOCH', 'FTDTC', 'FTENDTC', 'FTDY', 'FTENDY', 'DANAFL']`
- **Patient ID**: `USUBJID`
- **Acoustic test codes (FTTESTCD)**:
  - `CA35F5C5` — MEAN IMAGE DESCR VOLITION 2
  - `CA37153B` — MEAN DELAYED STORY RECALL SCORE
  - `CA52304B` — MEAN OBJECT RECALL SCORE
  - `CA5550C4` — MEAN IMAGE DESCR SCORE
  - `CA55A559` — MEAN VISUAL NAMING INTRAWORD PAUSE TIME
  - `CA65CBDF` — MEAN SENTENCE ASR SPEAKING RATE
  - `CA67FDBA` — MEAN CATEGORY NAMING VERBAL FLUENCY
  - `CA6E2CA5` — MEAN IMAGE DESCR ASR SPEAKING RATE
  - `CAACFCCC` — MAX DELAYED WORD RECALL SCORE
  - `MA8D60D3` — MEAN VISUAL NAMING TOTAL DUR
  - `MAC168D2` — MEAN IMAGE DESCR MONOTONICITY
  - `MAD174CB` — MEAN SENTENCE PAUSE RATE
- **NOTE**: Acoustic features use hashed FTTESTCD codes (not standard CDISC). Use FTTEST for human-readable name. FTSTRESN is the numeric feature value. FTACPTFL = 'Y' indicates accepted/quality data.
- **NOTE**: No direct 'jitter' or 'shimmer' columns — these are derived acoustic features expressed via speaking rate, pause rate, monotonicity. Map CA55A559 (intraword pause time) → acoustic_jitter proxy; MAC168D2 (monotonicity) → acoustic_shimmer proxy.
- **Relevance**: PRIMARY — acoustic encoder features

---

### FT_LINUS_HEALTH.csv (PRIMARY motor features)
- **Path**: data/raw/biohermes/BIOHERMES001/DigitalTests/Linus/FT_LINUS_HEALTH.csv
- **Size**: 7,360,904 bytes
- **Rows**: 36,004
- **Columns (34)**: Same SDTM FT structure as FT_AURAL_ANALYTICS.csv
- **Motor/cognitive test codes (FTTESTCD)**:
  - `DCRCLS` — DCR Classification
  - `DCRDCTSC` — DCR DCTclock Score
  - `DCRDLRCL` — DCR Delayed Recall Score
  - `DCRIMRCL` — DCR Immediate Recall Score
  - `DCRSCR` — DCR Summary Score
  - `DNMEDRT` — Digit Naming Median Reaction Time
  - `DNMNRT` — Digit Naming Mean Reaction Time All Trials
  - `DNMNTHRU` — Digit Naming Mean Throughput
  - `DNPCTCOR` — Digit Naming Percent Correct
  - `DNPCTFST` — Digit Naming Percent Fast Responses
  - `DNPCTLAP` — Digit Naming Percent Lapsed
  - `DNSDRT` — Digit Naming STDEV Reaction Time
  - `PTHAVTIM` — Path Finding Time to Complete
  - `SDMTACC` — SDMT Accuracy
  - `SDMTATTP` — SDMT Attempted Boxes
  - `SPCCWDTM` — Spiral CCW Dominant Time
  - `SPCCWNTM` — Spiral CCW Non-Dominant Time
  - `SPCWDTM` — Spiral CW Dominant Time
  - `SPCWNTM` — Spiral CW Non-Dominant Time
  - `TRLSACC` — Trails B Accuracy
  - `TRLSTIME` — Trails B Time to Complete
- **Relevance**: PRIMARY — motor encoder features (spiral drawing, reaction time, digit naming)

---

### QS_LINUS_HEALTH.csv
- **Path**: data/raw/biohermes/BIOHERMES001/DigitalTests/Linus/QS_LINUS_HEALTH.csv
- **Size**: 9,638,509 bytes
- **Rows**: 71,574
- **Columns (23)**: `['STUDYID', 'DOMAIN', 'USUBJID', 'QSSEQ', 'QSREFID', 'QSTESTCD', 'QSTEST', 'QSCAT', 'QSORRES', 'QSSTRESC', 'QSSTRESN', 'QSSTAT', 'QSREASND', 'QSNAM', 'QSEVALID', 'VISITNUM', 'VISIT', 'EPOCH', 'QSDTC', 'QSENDTC', 'QSDY', 'QSENDY', 'DANAFL']`
- **Test codes**: 82 unique QSTESTCD codes including ANXQNR (anxiety questionnaire), LHQ001-LHQ027+ (Linus Health questionnaire items)
- **Relevance**: SUPPLEMENTAL — patient-reported outcomes and questionnaire data

---

### FT_COGNIVUE.csv
- **Path**: data/raw/biohermes/BIOHERMES001/DigitalTests/Cognivue/FT_COGNIVUE.csv
- **Size**: 4,710,221 bytes
- **Rows**: 24,024
- **Columns (34)**: Same SDTM FT structure
- **Test codes (FTTESTCD)**: CCCA101-CCCA124 — Adaptive Motor Control, Visual Salience, Letter/Word/Shape/Motion Discrimination/Memory subtests, reaction times
- **Relevance**: SUPPLEMENTAL — cognitive/motor assessment platform; consider for clinical encoder

---

## GAP-Clinical Files

### DM.csv (DEMOGRAPHICS — PRIMARY)
- **Path**: data/raw/biohermes/BIOHERMES001/GAP-Clinical/DM.csv
- **Size**: 204,815 bytes
- **Rows**: 1,005
- **Columns (25)**: `['STUDYID', 'DOMAIN', 'USUBJID', 'SUBJID', 'RFSTDTC', 'RFENDTC', 'RFXSTDTC', 'RFXENDTC', 'RFICDTC', 'RFPENDTC', 'DTHDTC', 'DTHFL', 'SITEID', 'AGE', 'AGEU', 'SEX', 'RACE', 'ETHNIC', 'ARMCD', 'ARM', 'ACTARMCD', 'ACTARM', 'ARMNRS', 'ACTARMUD', 'COUNTRY']`
- **Key columns**:
  - `USUBJID` — primary patient ID ("BIO-HERMES-00101-NNN")
  - `AGE` — age in years (integer; mean=72, range 59–85)
  - `SEX` — sex ('F' = 564, 'M' = 441)
  - `RACE` — race category
  - `ETHNIC` — ethnicity
  - `RFSTDTC` — study start date (ISO 8601)
- **Relevance**: PRIMARY — demographics for clinical encoder

---

### SC.csv (SCREENING — education data)
- **Path**: data/raw/biohermes/BIOHERMES001/GAP-Clinical/SC.csv
- **Size**: 749,758 bytes
- **Rows**: 9,158
- **Columns (10)**: `['STUDYID', 'DOMAIN', 'USUBJID', 'SCSEQ', 'SCTESTCD', 'SCTEST', 'SCORRES', 'SCSTRESC', 'SCSTAT', 'SCREASND']`
- **Key columns for education**: filter `SCTESTCD == 'EDUYRNUM'` → `SCORRES` = years of education (1,005 records, one per patient)
- **Other SCTESTCD values**: ALZFH (family history), ALZFHC/F/M/S/MGF/MGM/PGF/PGM (family history by relationship), HANDDOM, HOMETYPE, MARISTAT, EMPJOB
- **Relevance**: PRIMARY — years of education for clinical encoder (via SCTESTCD='EDUYRNUM', SCSTRESC)

---

### NV.csv (AMYLOID PET — PRIMARY ground truth)
- **Path**: data/raw/biohermes/BIOHERMES001/GAP-Clinical/NV.csv
- **Size**: 2,437,895 bytes
- **Rows**: 14,874
- **Columns (23)**: `['STUDYID', 'DOMAIN', 'USUBJID', 'NVSEQ', 'NVTESTCD', 'NVTEST', 'NVORRES', 'NVORRESU', 'NVSTRESC', 'NVSTRESN', 'NVSTRESU', 'NVSTAT', 'NVREASND', 'NVNAM', 'NVLOC', 'NVDIR', 'NVMETHOD', 'NVBLFL', 'VISITNUM', 'VISIT', 'EPOCH', 'NVDTC', 'NVDY']`
- **Amyloid classification codes**:
  - `AMYCLAS` — Subject amyloid classification (NVSTRESC: 'NEGATIVE' / 'POSITIVE')
  - `AMYCLASV` — VisQ Amyloid Classification (NVSTRESC: 'NEGATIVE' / 'POSITIVE')
  - `AMYLOID` — Amyloid status (NVSTRESC: 'NEGATIVE' / 'POSITIVE' / NaN)
  - `CENTLOID` — Centiloid value (numeric)
  - `SUVR` — Standard Uptake Value Ratio (numeric)
  - `R1SCORE`, `R2SCORE`, `R3SCORE`, `RSCORE` — regional scores
  - `SQC` — Safety QC Grade
- **Relevance**: PRIMARY — amyloid positivity label source; use AMYCLAS (NVSTRESC='POSITIVE'/'NEGATIVE')

---

### RS.csv (RESULTS — FAQ scores)
- **Path**: data/raw/biohermes/BIOHERMES001/GAP-Clinical/RS.csv
- **Size**: 2,114,418 bytes
- **Rows**: 11,045
- **Columns (19)**: `['STUDYID', 'DOMAIN', 'USUBJID', 'RSSEQ', 'RSTESTCD', 'RSTEST', 'RSORRES', 'RSSTRESC', 'RSSTRESN', 'RSSTAT', 'RSREASND', 'VISITNUM', 'VISIT', 'EPOCH', 'RSDTC', 'RSENDTC', 'RSDY', 'RSENDY']`
- **Test codes**: FAQ0101-FAQ0110 (individual FAQ items), FAQ0111 (FAQ total score), RSALL (clinical classification)
- **Relevance**: SUPPLEMENTAL — functional assessment questionnaire scores

---

### FT.csv (COMBINED FUNCTIONAL TESTS — GAP-Clinical)
- **Path**: data/raw/biohermes/BIOHERMES001/GAP-Clinical/FT.csv
- **Size**: 25,717,134 bytes
- **Rows**: 125,364
- **Columns (34)**: Same SDTM FT structure
- **Test codes**: Combines Aural Analytics (CA* and MA* codes), Linus Health (DCR*, DN*, SP*, SDMT*, TRLS*), Cognivue (CCCA*), AVL02 (AVLT memory), and MMSE (MMS1*) items
- **NOTE**: MMS112 = MMSE Total Score is embedded here; filter FTTESTCD='MMS112', use FTSTRESN for score
- **Relevance**: HIGH — consolidated all digital/functional tests; use for one-stop motor and cognitive feature extraction

---

### LB_EDC.csv
- **Path**: data/raw/biohermes/BIOHERMES001/GAP-Clinical/LB_EDC.csv
- **Size**: 824,229 bytes
- **Rows**: 4,993
- **Columns (29)**: Same SDTM LB structure
- **Test codes**: AB4042, AMYLB40, AMYLB42, CELLS, GLUC, HDL, INR, LDL, PLAT, PROT, TPROTP, TRIG
- **Relevance**: SUPPLEMENTAL — central lab EDC data including standard clinical labs and phospho-tau

---

### DS.csv (DISPOSITION)
- **Path**: data/raw/biohermes/BIOHERMES001/GAP-Clinical/DS.csv
- **Size**: 318,878 bytes
- **Rows**: 2,443
- **Columns (10)**: `['STUDYID', 'DOMAIN', 'USUBJID', 'DSSEQ', 'DSTERM', 'DSDECOD', 'DSCAT', 'EPOCH', 'DSSTDTC', 'DSSTDY']`
- **DSDECOD values**: COMPLETED, INFORMED CONSENT OBTAINED, REFERRED TO SUB-STUDY, SPONSOR REQUEST, WITHDRAWAL BY SUBJECT
- **Relevance**: SUPPLEMENTAL — study disposition/completion status

---

### NV.csv Other Fields
- Other NVTESTCD values also present: `SUVR` and `CENTLOID` from amyloid PET scans

---

## OtherAssessments Files

### NV_ELI_LILLY.csv (Amyloid PET SUVR/Centiloid)
- **Path**: data/raw/biohermes/BIOHERMES001/OtherAssessments/Lilly SUVR & Centiloid/NV_ELI_LILLY.csv
- **Size**: 1,531,322 bytes
- **Rows**: 8,289
- **Columns (23)**: Same SDTM NV structure
- **Test codes**: `CENTLOID` (Centiloid), `COUNTS` (Regional Mean Counts), `SUVR` (Standard Uptake Value Ratio)
- **Relevance**: HIGH — PET amyloid quantitative data; use CENTLOID > 24.4 as amyloid positive threshold (standard cutoff)

### NV_IXICO.csv (MRI)
- **Path**: data/raw/biohermes/BIOHERMES001/OtherAssessments/IXICO/NV_IXICO.csv
- **Size**: 550,148 bytes
- **Rows**: 3,780
- **Columns (23)**: Same SDTM NV structure
- **Relevance**: SUPPLEMENTAL — MRI brain volumetry data

### NV_RETISPEC.csv (Retinal)
- **Path**: data/raw/biohermes/BIOHERMES001/OtherAssessments/Retispec/NV_RETISPEC.csv
- **Size**: 223,257 bytes
- **Rows**: 1,800
- **Columns (23)**: Same SDTM NV structure
- **Relevance**: OUT OF SCOPE for NeuroFusion-AD encoders

---

## Files Identified for NeuroFusion Preprocessing

| Purpose | File | Key Filter | Key Columns |
|---------|------|-----------|-------------|
| pTau-217 plasma (PRIMARY) | LB_LILLY_CLINICAL_DIAGNOST.csv | LBTESTCD == 'TAU217P' | USUBJID, LBSTRESN, LBSTRESU, VISITNUM |
| pTau-217 plasma (SECONDARY) | LB_UNIVERSITY_OF_GOTHENBUR.csv | LBTESTCD == 'TAU217P' | USUBJID, LBSTRESN |
| pTau-181 plasma | LB_ROCHE.csv | LBTESTCD == 'TAU181P' | USUBJID, LBSTRESN |
| Abeta42/40 plasma | LB_QUANTERIX_CORPORATION.csv | LBTESTCD == 'AB4042' | USUBJID, LBSTRESN |
| NfL plasma (Simoa) | LB_QUANTERIX_CORPORATION.csv | LBTESTCD == 'NFLP' | USUBJID, LBSTRESN |
| GFAP plasma (Simoa) | LB_QUANTERIX_CORPORATION.csv | LBTESTCD == 'GFAP' | USUBJID, LBSTRESN |
| Amyloid positivity label | GAP-Clinical/NV.csv | NVTESTCD == 'AMYCLAS' | USUBJID, NVSTRESC ('POSITIVE'/'NEGATIVE') |
| Amyloid PET Centiloid | OtherAssessments/Lilly.../NV_ELI_LILLY.csv | NVTESTCD == 'CENTLOID' | USUBJID, NVSTRESN |
| Demographics | GAP-Clinical/DM.csv | — | USUBJID, AGE, SEX, RACE, ETHNIC |
| Education | GAP-Clinical/SC.csv | SCTESTCD == 'EDUYRNUM' | USUBJID, SCSTRESC |
| Acoustic features | DigitalTests/Aural Analytics/FT_AURAL_ANALYTICS.csv | FTACPTFL == 'Y' | USUBJID, FTTESTCD, FTSTRESN |
| Motor features | DigitalTests/Linus/FT_LINUS_HEALTH.csv | FTACPTFL == 'Y' | USUBJID, FTTESTCD, FTSTRESN |
| MMSE score | GAP-Clinical/FT.csv | FTTESTCD == 'MMS112' | USUBJID, FTSTRESN |

---

## Patient ID Column
- **ADNI**: uses `RID` (integer, e.g., 3, 4, 5) as primary merge key; `PTID` (string, e.g., "011_S_0003") as human-readable
- **Bio-Hermes-001**: uses `USUBJID` (string, e.g., "BIO-HERMES-00101-001") as primary ID across ALL files; `SUBJID` = "00101-001" short form

---

## Bio-Hermes-002 References Found

The following files **incorrectly reference "Bio-Hermes-002"** (the actual study used is Bio-Hermes-001):

1. `docs/regulatory/data_requirements_v1.0.md` — multiple references to "Bio-Hermes-002" throughout
2. `scripts/batch/generate_phase1_docs.py` — one reference to "Bio-Hermes-002"

The file `scripts/batch/generate_phase2_docs.py` contains a **correction note**: "This project uses Bio-Hermes-001 (NOT Bio-Hermes-002 — that study concludes ~2028)" — confirming Bio-Hermes-002 references are incorrect.

**Action required by data-engineer-agent**: Replace all "Bio-Hermes-002" with "Bio-Hermes-001" in:
- `docs/regulatory/data_requirements_v1.0.md`
- `scripts/batch/generate_phase1_docs.py`

---

## Acoustic Feature Mapping Note
The Bio-Hermes-001 Aural Analytics FTTESTCD codes use non-standard hashed identifiers. The mapping to NeuroFusion acoustic encoder inputs is:

| FTTESTCD | FTTEST | NeuroFusion Acoustic Feature |
|----------|--------|------------------------------|
| CA65CBDF | MEAN SENTENCE ASR SPEAKING RATE | speaking_rate |
| MAD174CB | MEAN SENTENCE PAUSE RATE | pause_rate (→ jitter proxy) |
| MAC168D2 | MEAN IMAGE DESCR MONOTONICITY | monotonicity (→ shimmer proxy) |
| CA55A559 | MEAN VISUAL NAMING INTRAWORD PAUSE TIME | intraword_pause_time |
| CA67FDBA | MEAN CATEGORY NAMING VERBAL FLUENCY | verbal_fluency |
| CA6E2CA5 | MEAN IMAGE DESCR ASR SPEAKING RATE | image_speaking_rate |

Note: Classical acoustic jitter and shimmer are not directly measured; the pause and monotonicity features serve as proxies within the validated ranges specified in CLAUDE.md.
