# NeuroFusion-AD — Master Agent Constitution (Updated: Phase 2A)

## PROJECT IDENTITY

- **Name**: NeuroFusion-AD
- **Purpose**: Multimodal GNN for Alzheimer's Disease Progression Prediction
- **Target**: Roche acquisition via Navify Algorithm Suite
- **Regulatory Class**: SaMD — FDA De Novo + EU MDR Class IIa

---

## PHASE HISTORY

- **Phase 1**: COMPLETE — 89 tests passing, all regulatory docs generated, $6.53 spent
- **Phase 2A**: IN PROGRESS — data exploration, cleaning, preprocessing
- **Phase 2**: PENDING — model training (blocked on Phase 2A completion)
- **Phase 3**: PENDING — integration, testing, regulatory submission

---

## CURRENT PHASE: 2A — Data Exploration, Cleaning & Preprocessing

### What Phase 2A Is

The raw ADNI and Bio-Hermes-001 downloads are large archives with many folders, files, CSVs, PDFs, and data dictionaries. Phase 2A's job is to:

1. Walk every file in `data/raw/adni/` and `data/raw/biohermes/` and understand what exists
2. Identify which files contain the data we need
3. Clean, merge, and produce model-ready CSVs in `data/processed/`
4. Then set up RunPod and upload ONLY the clean processed files for training

**All data cleaning happens on the local machine first. RunPod is only for GPU training.**

---

## PHASE 2A EXIT CRITERIA

All of these must be checked in `docs/PHASE2A_CHECKLIST.md` before stopping:

- [ ] `docs/data/adni_file_inventory.md` — every file in data/raw/adni/ listed and described
- [ ] `docs/data/biohermes_file_inventory.md` — every file in data/raw/biohermes/ listed
- [ ] `configs/data/adni_column_map.yaml` — actual column names found → NeuroFusion schema
- [ ] `configs/data/biohermes_column_map.yaml` — actual column names found → NeuroFusion schema
- [ ] `notebooks/eda/01_adni_eda.ipynb` — EDA with outputs saved
- [ ] `notebooks/eda/02_biohermes_eda.ipynb` — EDA with outputs saved
- [ ] `src/data/adni_preprocessing.py` — updated with real column names, produces clean output
- [ ] `src/data/biohermes_preprocessing.py` — updated with real column names
- [ ] `data/processed/adni/adni_train.csv` — 70% split
- [ ] `data/processed/adni/adni_val.csv` — 15% split
- [ ] `data/processed/adni/adni_test.csv` — 15% split (held out — never used during training)
- [ ] `data/processed/adni/scaler.pkl` — StandardScaler fitted on train set only
- [ ] `data/processed/biohermes/biohermes001_train.csv`
- [ ] `data/processed/biohermes/biohermes001_val.csv`
- [ ] `docs/data/data_quality_report.md` — final statistics and limitations
- [ ] `pytest tests/unit/test_data.py -v` — 0 failures with real data
- [ ] RunPod pod created, SSH configured, project cloned to `/workspace/neurofusion-ad/`
- [ ] Clean processed files uploaded to RunPod `/workspace/neurofusion-ad/data/processed/`
- [ ] `PHASE2A_COMPLETE.md` written

**STOP after Phase 2A. Do not begin model training. Wait for human gate review.**

---

## IMPORTANT CORRECTIONS (fixes from earlier phases)

1. **Bio-Hermes-002 does not exist** — the study ends ~2028. Use Bio-Hermes-001 everywhere.
   Find and replace ALL occurrences before doing anything else:

   ```bash
   grep -r "Bio-Hermes-002\|BioHermes002\|biohermes002\|bio_hermes_002" . \
     --include="*.py" --include="*.md" --include="*.yaml" -l
   ```

2. **ADNI missing values are coded as -1 AND -4** — both must be replaced with NaN
3. **ADNI has no single ADNIMERGE file in all downloads** — must find and identify the master file yourself by scanning what's actually present
4. **CSF pTau in ADNI is pTau181, NOT plasma pTau217** — use it as a proxy, document limitation

---

## ADNI: WHAT TO LOOK FOR

The ADNI archive contains many files. Scan for files matching these patterns:

**High priority — likely contain the data we need:**

- Any file with "MERGE" in the name → likely the master longitudinal file
- Any file with "BIOMK" or "BIOMARK" → CSF biomarkers (pTau, Abeta)
- Any file with "APOE" or "GENET" → APOE genotyping
- Any file with "DXSUM" or "DIAGNOSIS" → diagnosis progression data
- Any file with "MMSE" → cognitive scores
- Any file with "REGISTRY" → visit dates

**What columns to look for in each file:**

- Patient ID: usually `RID` or `PTID`
- Visit code: usually `VISCODE` or `VISCODE2` — baseline = `'bl'`
- Baseline diagnosis: `DX_bl` (values: CN, MCI, AD, EMCI, LMCI)
- Current diagnosis: `DX` or `DIAGNOSIS`
- Age: `AGE`
- Sex: `PTGENDER` (1=Male, 2=Female)
- Education: `PTEDUCAT`
- APOE: `APOE4` (count of e4 alleles: 0, 1, 2)
- MMSE score: `MMSE`
- CSF pTau: `PTAU` (in biomarker file)
- CSF Abeta42: `ABETA` (in biomarker file)
- CSF NfL: `NFL` or `NfL` (may not exist — impute if missing)

**Label creation logic:**

```python
# Classification label
AMYLOID_POSITIVE = 1 if ABETA < 192 else 0  # standard CSF cutoff

# Regression label — requires longitudinal records
# For each patient (RID), fit: MMSE ~ time_in_years
# MMSE_SLOPE = coefficient (negative = declining)

# Survival label — requires diagnosis progression records
# TIME_TO_EVENT = months from baseline to first Dementia diagnosis
# EVENT_INDICATOR = 1 if progressed to Dementia, 0 if censored (still MCI at last visit)
```

---

## BIO-HERMES-001: WHAT TO LOOK FOR

Bio-Hermes-001 is a large multi-file archive. Key things to find:

**Plasma biomarkers** (the key ones — this is what makes Bio-Hermes valuable):

- pTau-217 (Roche Elecsys assay — this IS the plasma pTau217 our model wants)
- Aβ42, Aβ40, Aβ42/Aβ40 ratio
- NfL if available

**Ground truth label:**

- Amyloid PET result OR CSF amyloid status
- Filter to ONLY participants with this confirmatory testing

**Demographics:**

- Age, Sex, Race/Ethnicity (important for fairness analysis), Education, APOE

**What to do if Bio-Hermes has sub-files:**

- Identify which file has biomarkers, which has demographics, which has diagnosis
- Merge on participant ID
- Note: Bio-Hermes participant IDs may use different column name than ADNI — find it

---

## RUNPOD WORKFLOW

**Architecture**: Your local Claude Code session controls RunPod via SSH MCP. You never run Claude Code on RunPod itself.

**Steps (in order):**

1. Local Claude Code cleans the data → produces `data/processed/` files
2. RunPod pod is created (RTX 3090, with Network Volume attached)
3. SSH MCP is configured in your local Claude Code session
4. Claude Code SSH-connects to RunPod, installs packages, clones repo
5. Claude Code uploads clean processed files to RunPod via SCP/rsync
6. Training scripts run on RunPod GPU, controlled remotely

**SSH MCP setup (one command in your terminal):**

```bash
claude mcp add --transport stdio ssh-mcp -- npx -y ssh-mcp \
  --host=<RUNPOD_IP> --port=<RUNPOD_SSH_PORT> --user=root --password=<RUNPOD_PASSWORD>
```

Get IP, port, and password from RunPod dashboard → Connect → SSH over exposed TCP.

**RunPod terminate/resume:** Terminate freely. All files in `/workspace/` survive on the Network Volume. Next pod start: attach same volume, run setup script, continue.

---

## PHASE 2A FILE OWNERSHIP

| Agent                 | Owns                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------ |
| `data-explorer-agent` | `docs/data/`, `configs/data/`, `notebooks/eda/`                                            |
| `data-engineer-agent` | `src/data/adni_preprocessing.py`, `src/data/biohermes_preprocessing.py`, `data/processed/` |
| `runpod-setup-agent`  | RunPod SSH connection, pod setup, file upload                                              |

---

## TECHNICAL SPECIFICATIONS (unchanged from Phase 1)

**Model input schema (what processed CSVs must produce):**

```python
{
    'fluid': tensor shape (3,)   # [PTAU_norm, ABETA4240_norm, NFL_norm]
    'acoustic': tensor shape (15,)
    'motor': tensor shape (20,)
    'clinical_cont': tensor shape (2,)  # [AGE_norm, MMSE_norm]
    'sex': long tensor           # 0 or 1
    'apoe': long tensor          # 0, 1, or 2
    'label_classification': float  # AMYLOID_POSITIVE
    'label_regression': float      # MMSE_SLOPE
    'survival_time': float         # TIME_TO_EVENT in months
    'event_indicator': float       # 0 or 1
}
```

**Validation ranges (hard — reject outside these):**

- PTAU (CSF pTau181 proxy): 5–500 pg/mL (CSF range, wider than plasma)
- ABETA: 100–2000 pg/mL (CSF range)
- MMSE: 0–30
- AGE: 50–90

---

## CODING STANDARDS (unchanged)

- PEP 8, Google docstrings, type hints on all functions
- Never log PHI — hash all patient IDs
- pytest for all tests
- Git commit after every major deliverable: `[role]: description`

---

## HANDOFF PROTOCOL

Write `docs/agent_handoffs/[role]_[YYYY-MM-DD].md` at end of every session.
Check handoffs before starting any session to avoid duplicate work.
