"""ADNI data preprocessing pipeline for NeuroFusion-AD.

Loads ADNI CSV and .rda files, merges on patient ID (RID) and baseline visit,
filters to MCI patients, computes labels (AMYLOID_POSITIVE, MMSE_SLOPE,
TIME_TO_EVENT, EVENT_INDICATOR), normalizes with StandardScaler, and
produces train/val/test CSVs in data/processed/adni/.

Document traceability:
    DRD-001 § 3.1 — ADNI data processing requirements
    SRS-001 § 4.2 — Input feature specifications

Missing value codes: ADNI uses -1 and -4 as missing (both replaced with NaN).
PHI: Patient IDs (RID) are SHA-256 hashed in all log messages.
"""

import hashlib
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadr
import structlog
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ADNI_MISSING_CODES = [-1, -4]
BASELINE_VISIT = "bl"
MCI_DIAGNOSES = ["MCI"]  # DXSUM DIAGNOSIS values for MCI
AMYLOID_POS_CUTOFF = 192.0  # CSF Abeta42 < 192 pg/mL = amyloid positive (UPENN assay)

RAW_DIR = Path("data/raw/adni")
PROCESSED_DIR = Path("data/processed/adni")
RDA_DIR = RAW_DIR / "ADNIMERGE2/data"

# ---------------------------------------------------------------------------
# File paths (verified by data-explorer-agent 2026-03-11)
# ---------------------------------------------------------------------------
PLASMA_FILE  = RAW_DIR / "UPENN_PLASMA_FUJIREBIO_QUANTERIX_28Feb2026.csv"
CSF_FILE     = RAW_DIR / "UPENNBIOMK_MASTER_28Feb2026.csv"
APOE_FILE    = RAW_DIR / "APOERES_28Feb2026.csv"
PTDEMOG_RDA  = RDA_DIR / "PTDEMOG.rda"
MMSE_RDA     = RDA_DIR / "MMSE.rda"
DXSUM_RDA    = RDA_DIR / "DXSUM.rda"
REGISTRY_RDA = RDA_DIR / "REGISTRY.rda"
CDR_RDA      = RDA_DIR / "CDR.rda"

# Feature dimension constants (preserved for backward compatibility with tests)
_FLUID_DIM = 6
_ACOUSTIC_DIM = 12
_MOTOR_DIM = 8
_CLINICAL_DIM = 10


def _hash_rid(rid: int) -> str:
    """Hash a patient RID for safe logging (PHI compliance)."""
    return hashlib.sha256(str(rid).encode()).hexdigest()[:12]


def _replace_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ADNI missing value codes (-1 and -4) with NaN in numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace(ADNI_MISSING_CODES, np.nan)
    return df


def _read_rda(path: Path, key: str) -> pd.DataFrame:
    """Read a single R data file (.rda) and return its dataframe.

    Args:
        path: Path to the .rda file.
        key: Name of the R object inside the file (usually same as filename without .rda).

    Returns:
        DataFrame from the .rda file.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pyreadr.read_r(str(path))
    df = result[key]
    if "RID" in df.columns:
        df["RID"] = pd.to_numeric(df["RID"], errors="coerce").astype("Int64")
    return df


def load_plasma_biomarkers() -> pd.DataFrame:
    """Load ADNI plasma biomarker file (pTau217, NfL, GFAP, Abeta ratio).

    Source: UPENN_PLASMA_FUJIREBIO_QUANTERIX_28Feb2026.csv
    Key columns: RID, VISCODE, pT217_F, AB42_AB40_F, NfL_Q, GFAP_Q

    Returns:
        DataFrame with baseline plasma biomarkers, one row per RID.
    """
    df = pd.read_csv(PLASMA_FILE)
    df = _replace_missing(df)
    # Filter to baseline
    df_bl = df[df["VISCODE"].str.lower() == BASELINE_VISIT].copy()
    # Keep most recent batch per patient if duplicates
    df_bl = df_bl.sort_values("RID").drop_duplicates(subset=["RID"], keep="last")
    df_bl = df_bl.rename(columns={
        "pT217_F":     "PTAU217",
        "AB42_AB40_F": "ABETA4240_RATIO",
        "NfL_Q":       "NFL_PLASMA",
        "GFAP_Q":      "GFAP_PLASMA",
        "AB42_F":      "ABETA42_PLASMA",
        "AB40_F":      "ABETA40_PLASMA",
    })
    log.info("Loaded plasma biomarkers", n_patients=len(df_bl))
    return df_bl[["RID", "PTAU217", "ABETA4240_RATIO", "NFL_PLASMA", "GFAP_PLASMA",
                   "ABETA42_PLASMA", "ABETA40_PLASMA"]]


def load_csf_biomarkers() -> pd.DataFrame:
    """Load ADNI CSF biomarker file (pTau181, Abeta42, total tau) for amyloid label.

    Source: UPENNBIOMK_MASTER_28Feb2026.csv
    Key columns: RID, VISCODE, PTAU, ABETA, TAU
    Used for: AMYLOID_POSITIVE label (ABETA < 192 pg/mL)

    Returns:
        DataFrame with baseline CSF biomarkers, one row per RID.
    """
    df = pd.read_csv(CSF_FILE)
    df = _replace_missing(df)
    df_bl = df[df["VISCODE"].str.lower() == BASELINE_VISIT].copy()
    df_bl = df_bl.sort_values("RID").drop_duplicates(subset=["RID"], keep="last")
    df_bl = df_bl.rename(columns={
        "PTAU":  "PTAU181_CSF",
        "ABETA": "ABETA42_CSF",
        "TAU":   "TAU_CSF",
    })
    # Compute amyloid positivity label
    df_bl["AMYLOID_POSITIVE"] = (df_bl["ABETA42_CSF"] < AMYLOID_POS_CUTOFF).astype(float)
    log.info("Loaded CSF biomarkers", n_patients=len(df_bl))
    return df_bl[["RID", "PTAU181_CSF", "ABETA42_CSF", "TAU_CSF", "AMYLOID_POSITIVE"]]


def load_apoe(df_apoe_raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load APOE genotype and derive APOE4 allele count.

    Source: APOERES_28Feb2026.csv
    GENOTYPE format: '3/3', '3/4', '4/4', etc.

    Returns:
        DataFrame with RID and APOE4_COUNT (0, 1, or 2).
    """
    if df_apoe_raw is None:
        df_apoe_raw = pd.read_csv(APOE_FILE)
    df = df_apoe_raw.copy()
    # Screening visit has APOE; use any visit since it's stable
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce").astype("Int64")
    df = df.drop_duplicates(subset=["RID"], keep="first")

    def _count_apoe4(genotype: str) -> int:
        if pd.isna(genotype):
            return 0
        return str(genotype).count("4")

    df["APOE4_COUNT"] = df["GENOTYPE"].apply(_count_apoe4)
    log.info("Loaded APOE genotypes", n_patients=len(df))
    return df[["RID", "APOE4_COUNT"]]


def load_demographics() -> pd.DataFrame:
    """Load ADNI patient demographics (sex, birth year, education).

    Source: ADNIMERGE2/data/PTDEMOG.rda
    Note: No direct AGE column — age is computed later using REGISTRY EXAMDATE.

    Returns:
        DataFrame with RID, SEX_CODE (0=Female, 1=Male), EDUCATION_YEARS, PTDOBYY.
    """
    df = _read_rda(PTDEMOG_RDA, "PTDEMOG")
    df = _replace_missing(df)
    # Baseline only — PTDEMOG is effectively cross-sectional but filter to bl
    df_bl = df[df["VISCODE"].str.lower() == BASELINE_VISIT].drop_duplicates("RID", keep="first")
    # Encode sex: Male=1, Female=0
    df_bl["SEX_CODE"] = df_bl["PTGENDER"].map({"Male": 1, "Female": 0}).fillna(-1).astype(int)
    df_bl = df_bl.rename(columns={"PTEDUCAT": "EDUCATION_YEARS"})
    log.info("Loaded demographics", n_patients=len(df_bl))
    return df_bl[["RID", "SEX_CODE", "EDUCATION_YEARS", "PTDOBYY"]]


def load_mmse_longitudinal() -> pd.DataFrame:
    """Load all longitudinal MMSE records for slope computation.

    Source: ADNIMERGE2/data/MMSE.rda
    MMSCORE: total MMSE score (0-30)

    Returns:
        DataFrame with RID, VISCODE, MMSCORE, VISDATE (all visits, not just baseline).
    """
    df = _read_rda(MMSE_RDA, "MMSE")
    df = _replace_missing(df)
    df = df.dropna(subset=["MMSCORE"])
    df["VISDATE"] = pd.to_datetime(df["VISDATE"], errors="coerce")
    log.info("Loaded MMSE longitudinal", n_records=len(df))
    return df[["RID", "VISCODE", "MMSCORE", "VISDATE"]]


def load_diagnosis_longitudinal() -> pd.DataFrame:
    """Load all longitudinal diagnosis records for MCI filter and survival label.

    Source: ADNIMERGE2/data/DXSUM.rda
    DIAGNOSIS values: 'MCI', 'CN', 'Dementia' (no EMCI/LMCI in this column)

    Returns:
        DataFrame with RID, VISCODE, DIAGNOSIS, EXAMDATE.
    """
    df = _read_rda(DXSUM_RDA, "DXSUM")
    df = _replace_missing(df)
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    log.info("Loaded diagnosis longitudinal", n_records=len(df))
    return df[["RID", "VISCODE", "DIAGNOSIS", "EXAMDATE"]]


def load_registry() -> pd.DataFrame:
    """Load visit registry for exam dates (needed for age and time-to-event).

    Source: ADNIMERGE2/data/REGISTRY.rda

    Returns:
        DataFrame with RID, VISCODE, EXAMDATE.
    """
    df = _read_rda(REGISTRY_RDA, "REGISTRY")
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    log.info("Loaded registry", n_records=len(df))
    return df[["RID", "VISCODE", "EXAMDATE"]]


def compute_mmse_slope(df_mmse: pd.DataFrame, df_registry: pd.DataFrame) -> pd.DataFrame:
    """Compute per-patient MMSE slope (points/year) via linear regression.

    Requires at least 2 visits with valid MMSE scores.
    Patients with only 1 visit get NaN slope.

    Args:
        df_mmse: MMSE longitudinal records (from load_mmse_longitudinal).
        df_registry: Visit registry with EXAMDATE (from load_registry).

    Returns:
        DataFrame with RID and MMSE_SLOPE (negative = cognitive decline).
    """
    # Merge MMSE with registry to get exam dates
    df = df_mmse.merge(
        df_registry[["RID", "VISCODE", "EXAMDATE"]],
        on=["RID", "VISCODE"], how="left"
    )
    # Get baseline exam date per patient
    bl_dates = (
        df_registry[df_registry["VISCODE"] == BASELINE_VISIT][["RID", "EXAMDATE"]]
        .rename(columns={"EXAMDATE": "BL_DATE"})
    )
    df = df.merge(bl_dates, on="RID", how="left")
    df = df.dropna(subset=["EXAMDATE", "BL_DATE"])
    df["years_from_bl"] = (df["EXAMDATE"] - df["BL_DATE"]).dt.days / 365.25

    slopes = []
    for rid, grp in df.groupby("RID"):
        grp = grp.dropna(subset=["MMSCORE", "years_from_bl"])
        if len(grp) < 2:
            slopes.append({"RID": rid, "MMSE_SLOPE": np.nan})
            continue
        slope, _, _, _, _ = stats.linregress(grp["years_from_bl"], grp["MMSCORE"])
        slopes.append({"RID": rid, "MMSE_SLOPE": slope})

    df_slopes = pd.DataFrame(slopes)
    log.info("Computed MMSE slopes", n_patients=len(df_slopes),
             n_valid=int(df_slopes["MMSE_SLOPE"].notna().sum()))
    return df_slopes


def compute_survival_labels(
    df_dx: pd.DataFrame,
    df_registry: pd.DataFrame,
    mci_rids: set,
) -> pd.DataFrame:
    """Compute time-to-event and event indicator for MCI patients.

    TIME_TO_EVENT: months from baseline to first Dementia diagnosis.
    EVENT_INDICATOR: 1 if patient progressed to Dementia, 0 if censored.

    Args:
        df_dx: Diagnosis longitudinal records.
        df_registry: Visit registry.
        mci_rids: Set of RIDs of baseline MCI patients.

    Returns:
        DataFrame with RID, TIME_TO_EVENT, EVENT_INDICATOR.
    """
    # Baseline dates
    bl_dates = (
        df_registry[df_registry["VISCODE"] == BASELINE_VISIT][["RID", "EXAMDATE"]]
        .rename(columns={"EXAMDATE": "BL_DATE"})
    )

    results = []
    for rid in mci_rids:
        rid_dx = df_dx[df_dx["RID"] == rid].sort_values("EXAMDATE")
        bl_row = bl_dates[bl_dates["RID"] == rid]
        if bl_row.empty or pd.isna(bl_row["BL_DATE"].iloc[0]):
            continue

        bl_date = bl_row["BL_DATE"].iloc[0]
        dementia_rows = rid_dx[rid_dx["DIAGNOSIS"] == "Dementia"]

        if not dementia_rows.empty:
            event_date = dementia_rows["EXAMDATE"].dropna().min()
            if pd.isna(event_date):
                event_date = rid_dx["EXAMDATE"].dropna().max()
                event = 0
            else:
                event = 1
        else:
            event_date = rid_dx["EXAMDATE"].dropna().max()
            event = 0

        if pd.isna(event_date):
            continue

        months = (event_date - bl_date).days / 30.44
        results.append({
            "RID": rid,
            "TIME_TO_EVENT": max(0.1, months),  # ensure positive
            "EVENT_INDICATOR": float(event),
        })

    df_surv = pd.DataFrame(results)
    log.info("Computed survival labels",
             n_patients=len(df_surv),
             n_events=int(df_surv["EVENT_INDICATOR"].sum()))
    return df_surv


def synthesize_acoustic_features(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthesize 12 acoustic features for ADNI patients (no real acoustic data in ADNI).

    NOTE: ADNI has no speech/acoustic data. These are synthesized from
    clinically plausible distributions. Documented limitation in DRD-001.

    Args:
        n: Number of patients.
        seed: Random seed.

    Returns:
        DataFrame with 12 acoustic feature columns.
    """
    rng = np.random.default_rng(seed)
    # Use clipped normal distributions; validated ranges from CLAUDE.md
    features = {
        "acoustic_jitter":     np.clip(rng.lognormal(-5.3, 0.6, n), 0.0001, 0.05),
        "acoustic_shimmer":    np.clip(rng.lognormal(-3.2, 0.5, n), 0.001, 0.3),
        "acoustic_hnr":        np.clip(rng.normal(15.0, 5.0, n), 0, 30),
        "acoustic_f0_mean":    np.clip(rng.normal(130.0, 30.0, n), 80, 250),
        "acoustic_f0_std":     np.clip(rng.normal(25.0, 10.0, n), 5, 80),
        "acoustic_mfcc1":      rng.normal(0, 1, n),
        "acoustic_mfcc2":      rng.normal(0, 1, n),
        "acoustic_mfcc3":      rng.normal(0, 1, n),
        "acoustic_mfcc4":      rng.normal(0, 1, n),
        "acoustic_mfcc5":      rng.normal(0, 1, n),
        "acoustic_mfcc6":      rng.normal(0, 1, n),
        "acoustic_mfcc7":      rng.normal(0, 1, n),
    }
    return pd.DataFrame(features)


def synthesize_motor_features(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthesize 8 motor features for ADNI patients (no real motor data in ADNI).

    NOTE: ADNI has no wearable/motor assessment data. These are synthesized.
    Documented limitation in DRD-001.

    Args:
        n: Number of patients.
        seed: Random seed.

    Returns:
        DataFrame with 8 motor feature columns.
    """
    rng = np.random.default_rng(seed + 1)
    features = {
        "motor_tremor_freq":        np.clip(rng.normal(4.0, 2.0, n), 0, 12),
        "motor_tremor_amp":         np.clip(rng.normal(0.3, 0.2, n), 0, 2),
        "motor_bradykinesia_score": np.clip(rng.normal(50.0, 20.0, n), 0, 100),
        "motor_spiral_rmse":        np.clip(rng.normal(2.0, 1.0, n), 0, 10),
        "motor_tapping_cv":         np.clip(rng.normal(0.15, 0.05, n), 0, 1),
        "motor_tapping_asymmetry":  np.clip(rng.normal(0.05, 0.03, n), 0, 0.5),
        "motor_grip_force_mean":    np.clip(rng.normal(25.0, 10.0, n), 5, 60),
        "motor_grip_force_cv":      np.clip(rng.normal(0.10, 0.05, n), 0, 0.5),
    }
    return pd.DataFrame(features)


def compute_baseline_mmse(df_mmse: pd.DataFrame) -> pd.DataFrame:
    """Extract baseline (visit='bl') MMSE score per patient.

    Args:
        df_mmse: Longitudinal MMSE records.

    Returns:
        DataFrame with RID and MMSE_BASELINE.
    """
    df_bl = df_mmse[df_mmse["VISCODE"] == BASELINE_VISIT].copy()
    df_bl = df_bl.drop_duplicates("RID", keep="first")
    df_bl = df_bl.rename(columns={"MMSCORE": "MMSE_BASELINE"})
    return df_bl[["RID", "MMSE_BASELINE"]]


def build_master_dataset() -> pd.DataFrame:
    """Build the master ADNI dataset for NeuroFusion-AD.

    Pipeline:
    1. Load plasma biomarkers (primary fluid encoder source)
    2. Load CSF biomarkers (amyloid label)
    3. Load APOE genotype
    4. Load demographics
    5. Load MMSE longitudinal -> compute slope + baseline
    6. Load diagnosis -> filter MCI baseline
    7. Load registry -> compute age + survival labels
    8. Merge all on RID
    9. Replace missing values
    10. Synthesize acoustic + motor features

    Returns:
        Master DataFrame with all features and labels.
    """
    log.info("Building ADNI master dataset")

    # Load all sources
    df_plasma = load_plasma_biomarkers()
    df_csf    = load_csf_biomarkers()
    df_apoe   = load_apoe()
    df_demog  = load_demographics()
    df_mmse   = load_mmse_longitudinal()
    df_dx     = load_diagnosis_longitudinal()
    df_reg    = load_registry()

    # Baseline diagnosis
    df_dx_bl = df_dx[df_dx["VISCODE"] == BASELINE_VISIT].drop_duplicates("RID", keep="first")

    # Filter to MCI at baseline
    mci_rids_set = set(df_dx_bl[df_dx_bl["DIAGNOSIS"].isin(MCI_DIAGNOSES)]["RID"].dropna())
    log.info("MCI baseline patients found", n=len(mci_rids_set))

    # MMSE slope and baseline
    df_slope   = compute_mmse_slope(df_mmse, df_reg)
    df_mmse_bl = compute_baseline_mmse(df_mmse)

    # Survival labels
    df_surv = compute_survival_labels(df_dx, df_reg, mci_rids_set)

    # Baseline exam date for age
    df_reg_bl = df_reg[df_reg["VISCODE"] == BASELINE_VISIT].drop_duplicates("RID", keep="first")

    # Merge all on RID, starting from plasma (primary fluid source)
    df = df_plasma.merge(df_csf, on="RID", how="outer")
    df = df.merge(df_apoe, on="RID", how="left")
    df = df.merge(df_demog, on="RID", how="left")
    df = df.merge(df_mmse_bl, on="RID", how="left")
    df = df.merge(df_slope, on="RID", how="left")
    df = df.merge(df_surv, on="RID", how="left")
    df = df.merge(df_reg_bl[["RID", "EXAMDATE"]], on="RID", how="left")

    # Compute age from PTDOBYY and baseline exam year
    df["AGE"] = df["EXAMDATE"].dt.year - df["PTDOBYY"].fillna(0)
    df.loc[df["AGE"] < 40, "AGE"] = np.nan  # guard against bad birth years
    df.loc[df["AGE"] > 100, "AGE"] = np.nan

    # Filter to MCI patients
    df = df[df["RID"].isin(mci_rids_set)].copy()

    # Add synthesized acoustic and motor features
    n = len(df)
    df_acoustic = synthesize_acoustic_features(n)
    df_motor    = synthesize_motor_features(n)
    df = df.reset_index(drop=True)
    df = pd.concat([df, df_acoustic, df_motor], axis=1)

    # Final missing value cleanup
    df = _replace_missing(df)

    n_amyloid = int(df["AMYLOID_POSITIVE"].sum()) if "AMYLOID_POSITIVE" in df.columns else "N/A"
    log.info("Master dataset built",
             n_patients=len(df),
             n_amyloid_pos=n_amyloid,
             n_with_ptau217=int(df["PTAU217"].notna().sum()))
    return df


def normalize_and_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Normalize features and split into train/val/test.

    StandardScaler is fit on train only, then applied to val/test.
    Scaler is saved to data/processed/adni/scaler.pkl.

    Args:
        df: Master dataset from build_master_dataset().
        train_frac: Fraction for training (default 0.70).
        val_frac: Fraction for validation (default 0.15).
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (df_train, df_val, df_test).
    """
    FEATURE_COLS_TO_SCALE = [
        "PTAU217", "ABETA4240_RATIO", "NFL_PLASMA", "GFAP_PLASMA",
        "PTAU181_CSF", "ABETA42_CSF",
        "AGE", "MMSE_BASELINE", "EDUCATION_YEARS",
        "acoustic_jitter", "acoustic_shimmer", "acoustic_hnr",
        "acoustic_f0_mean", "acoustic_f0_std",
        "acoustic_mfcc1", "acoustic_mfcc2", "acoustic_mfcc3",
        "acoustic_mfcc4", "acoustic_mfcc5", "acoustic_mfcc6", "acoustic_mfcc7",
        "motor_tremor_freq", "motor_tremor_amp", "motor_bradykinesia_score",
        "motor_spiral_rmse", "motor_tapping_cv", "motor_tapping_asymmetry",
        "motor_grip_force_mean", "motor_grip_force_cv",
    ]
    # Only scale columns that actually exist in df
    scale_cols = [c for c in FEATURE_COLS_TO_SCALE if c in df.columns]

    # Split: train / (val + test)
    df_train, df_valtest = train_test_split(
        df, test_size=(1 - train_frac), random_state=seed, shuffle=True
    )
    val_size_relative = val_frac / (1 - train_frac)
    df_val, df_test = train_test_split(
        df_valtest, test_size=(1 - val_size_relative), random_state=seed
    )

    # Fit scaler on train only
    scaler = StandardScaler()
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_test  = df_test.copy()
    df_train[scale_cols] = scaler.fit_transform(df_train[scale_cols].fillna(0))
    df_val[scale_cols]   = scaler.transform(df_val[scale_cols].fillna(0))
    df_test[scale_cols]  = scaler.transform(df_test[scale_cols].fillna(0))

    # Save scaler
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    log.info("Data split complete",
             n_train=len(df_train), n_val=len(df_val), n_test=len(df_test))
    return df_train, df_val, df_test


def run_adni_pipeline() -> None:
    """Main entry point: build, normalize, split, and save ADNI processed data."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = build_master_dataset()
    df_train, df_val, df_test = normalize_and_split(df)

    df_train.to_csv(PROCESSED_DIR / "adni_train.csv", index=False)
    df_val.to_csv(PROCESSED_DIR / "adni_val.csv", index=False)
    df_test.to_csv(PROCESSED_DIR / "adni_test.csv", index=False)

    log.info("ADNI pipeline complete",
             train=str(PROCESSED_DIR / "adni_train.csv"),
             val=str(PROCESSED_DIR / "adni_val.csv"),
             test=str(PROCESSED_DIR / "adni_test.csv"))


# ---------------------------------------------------------------------------
# Legacy class — preserved for backward compatibility with existing tests
# ---------------------------------------------------------------------------

try:
    import torch

    # Normalization clip bounds
    _CLIP_MIN = -5.0
    _CLIP_MAX = 5.0

    class ADNIPreprocessor:
        """Preprocesses ADNI patient records into NeuroFusion-AD model inputs.

        Handles missing value imputation, feature normalization (z-score), and
        data quality checks. Used at inference time by the API layer.

        Attributes:
            missing_strategy: Strategy for imputing missing values ("mean", "median", "zero").

        IEC 62304 Traceability:
            SRS-001 § 5.1 — Input preprocessing requirements
            SDP-001 § 6.2 — Data normalization specification
        """

        FLUID_MEAN    = torch.tensor([12.0, 0.10, 30.0, 150.0, 250.0, 800.0])
        FLUID_STD     = torch.tensor([15.0, 0.05, 20.0,  80.0, 100.0, 200.0])
        ACOUSTIC_MEAN = torch.tensor([0.005, 0.04, 15.0, 130.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ACOUSTIC_STD  = torch.tensor([0.005, 0.02,  5.0,  30.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        MOTOR_MEAN    = torch.tensor([4.0, 0.3, 50.0, 2.0, 0.15, 0.05, 25.0, 0.10])
        MOTOR_STD     = torch.tensor([2.0, 0.2, 20.0, 1.0, 0.05, 0.03, 10.0, 0.05])
        CLINICAL_MEAN = torch.tensor([72.0, 14.0, 0.5, 26.0, 1.5, 5.0, 27.0, 130.0, 0.3, 2.0])
        CLINICAL_STD  = torch.tensor([ 8.0,  3.0, 0.5,  3.0, 1.5, 3.0,  5.0,  15.0, 0.45, 1.5])

        _VALID_STRATEGIES = ("mean", "median", "zero")

        def __init__(self, missing_strategy: str = "mean") -> None:
            """Initialize ADNIPreprocessor.

            Args:
                missing_strategy: How to handle missing (NaN) values.

            Raises:
                ValueError: If missing_strategy is not valid.
            """
            if missing_strategy not in self._VALID_STRATEGIES:
                raise ValueError(
                    f"missing_strategy must be one of {self._VALID_STRATEGIES}, "
                    f"got '{missing_strategy}'"
                )
            self.missing_strategy = missing_strategy
            log.info("adni_preprocessor_initialized", missing_strategy=missing_strategy)

        def normalize(
            self,
            features: "torch.Tensor",
            mean: "torch.Tensor",
            std: "torch.Tensor",
        ) -> "torch.Tensor":
            """Apply z-score normalization and clip to [-5, 5].

            Args:
                features: Input tensor [feature_dim] or [batch, feature_dim].
                mean: Per-feature mean [feature_dim].
                std: Per-feature standard deviation [feature_dim].

            Returns:
                Normalized tensor clipped to [-5, 5].
            """
            safe_std = std.clone()
            safe_std[safe_std < 1e-8] = 1e-8
            normalized = (features - mean) / safe_std
            return torch.clamp(normalized, _CLIP_MIN, _CLIP_MAX)

        def impute_missing(
            self,
            features: "torch.Tensor",
            mean: "torch.Tensor",
        ) -> "torch.Tensor":
            """Replace NaN values with column-wise statistics.

            Args:
                features: Input tensor [feature_dim] or [batch, feature_dim].
                mean: Per-feature population mean [feature_dim].

            Returns:
                Tensor with no NaN values.
            """
            was_1d = features.dim() == 1
            if was_1d:
                features = features.unsqueeze(0)

            result = features.clone()
            nan_mask = torch.isnan(result)

            if not nan_mask.any():
                return result.squeeze(0) if was_1d else result

            if self.missing_strategy == "mean":
                fill_values = mean
            elif self.missing_strategy == "median":
                fill_values = torch.zeros(features.shape[1])
                for col_idx in range(features.shape[1]):
                    col = features[:, col_idx]
                    valid = col[~torch.isnan(col)]
                    fill_values[col_idx] = valid.median() if valid.numel() > 0 else mean[col_idx]
            else:  # "zero"
                fill_values = torch.zeros(features.shape[1])

            for col_idx in range(features.shape[1]):
                col_mask = nan_mask[:, col_idx]
                if col_mask.any():
                    result[col_mask, col_idx] = fill_values[col_idx]

            n_imputed = nan_mask.sum().item()
            log.debug("missing_values_imputed",
                      strategy=self.missing_strategy,
                      n_imputed=int(n_imputed))
            return result.squeeze(0) if was_1d else result

        def preprocess_record(self, record: dict[str, Any]) -> dict[str, "torch.Tensor"]:
            """Process a single ADNI patient record into normalized tensors.

            Args:
                record: Dict with keys 'fluid', 'acoustic', 'motor', 'clinical'.

            Returns:
                Dict with normalized float32 tensors per modality.

            Raises:
                KeyError: If any required key is missing.
                ValueError: If any feature array has incorrect length.
            """
            required_keys = ("fluid", "acoustic", "motor", "clinical")
            for key in required_keys:
                if key not in record:
                    raise KeyError(f"Record is missing required key: '{key}'")

            expected_dims = {
                "fluid": _FLUID_DIM, "acoustic": _ACOUSTIC_DIM,
                "motor": _MOTOR_DIM, "clinical": _CLINICAL_DIM,
            }
            means = {
                "fluid": self.FLUID_MEAN, "acoustic": self.ACOUSTIC_MEAN,
                "motor": self.MOTOR_MEAN, "clinical": self.CLINICAL_MEAN,
            }
            stds = {
                "fluid": self.FLUID_STD, "acoustic": self.ACOUSTIC_STD,
                "motor": self.MOTOR_STD, "clinical": self.CLINICAL_STD,
            }

            processed: dict[str, torch.Tensor] = {}
            for key in required_keys:
                raw = record[key]
                if isinstance(raw, torch.Tensor):
                    tensor = raw.float()
                elif hasattr(raw, "tolist"):
                    tensor = torch.tensor(raw.tolist(), dtype=torch.float32)
                else:
                    tensor = torch.tensor(list(raw), dtype=torch.float32)

                expected = expected_dims[key]
                if tensor.shape[-1] != expected:
                    raise ValueError(
                        f"Feature '{key}' has {tensor.shape[-1]} dimensions, "
                        f"expected {expected}."
                    )

                tensor = self.impute_missing(tensor, means[key])
                tensor = self.normalize(tensor, means[key], stds[key])
                processed[key] = tensor

            log.debug("record_preprocessed")
            return processed

        def preprocess_batch(
            self, records: list[dict[str, Any]]
        ) -> dict[str, "torch.Tensor"]:
            """Process a list of ADNI patient records into batched tensors.

            Args:
                records: List of record dicts.

            Returns:
                Dict with batched float32 tensors [batch, feature_dim].

            Raises:
                ValueError: If records is empty.
            """
            if not records:
                raise ValueError("records list must not be empty.")

            processed_records = [self.preprocess_record(r) for r in records]
            keys = ("fluid", "acoustic", "motor", "clinical")
            batched: dict[str, torch.Tensor] = {
                key: torch.stack([rec[key] for rec in processed_records], dim=0)
                for key in keys
            }
            log.info("batch_preprocessed", n_records=len(records))
            return batched

except ImportError:
    # torch not available in this environment; pipeline functions still work
    pass


if __name__ == "__main__":
    run_adni_pipeline()
